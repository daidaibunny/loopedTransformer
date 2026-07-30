import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from looped_vl.models.recurrent_qwen3vl_embedding import (
	_dynamic_scaled_dot_product_attention,
)
from looped_vl.training.checkpointing import (
	TrainingCursor,
	capture_rng_state,
	load_training_checkpoint,
	prune_training_checkpoints,
	rebase_training_cursor_batch_size,
	restore_rng_state,
	save_training_checkpoint,
)
from looped_vl.training.config import TrainingConfig
from looped_vl.training.optimizer import build_optimizer_and_scheduler
from looped_vl.training.reproducibility import seed_everything
from looped_vl.training.schedule import OneEpochTrainingPlan
from looped_vl.training.step import compose_training_loss
from looped_vl.training.train import (
	_accumulate_metric_tensors,
	_clear_parameter_gradients,
	_finalize_metric_tensors,
	_optimizer_step_limit,
	_resolve_git_commit,
	_should_save_checkpoint,
	_training_phase,
	_worker_loader_options,
	parse_args,
)


class _FakeGradientScaler:
	def __init__(self, scale: float) -> None:
		self.scale = scale

	def state_dict(self) -> dict[str, float]:
		return {"scale": self.scale}

	def load_state_dict(self, state_dict: dict[str, float]) -> None:
		self.scale = state_dict["scale"]


def test_single_training_config_matches_optimizer_and_warm_start_protocol() -> None:
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))

	assert config.optimizer == "AdamW"
	assert config.learning_rate == 1e-5
	assert config.weight_decay == 0.01
	assert config.betas == (0.9, 0.95)
	assert config.eps == 1e-8
	assert config.effective_batch_size == 512
	assert config.gradient_clip_norm == 1.0
	assert config.precision == "bf16"
	assert config.lr_scheduler == "cosine"
	assert config.warmup_ratio == 0.03
	assert config.warm_start_epoch_fraction == 0.35
	assert config.joint_activation_warmup_ratio == 0.03


def test_seed_42_reproduces_python_numpy_and_torch_streams() -> None:
	seed_everything(42)
	first = (random.random(), np.random.rand(), torch.rand(3))
	seed_everything(42)
	second = (random.random(), np.random.rand(), torch.rand(3))

	assert first[0] == second[0]
	assert first[1] == second[1]
	assert torch.equal(first[2], second[2])
	assert torch.backends.cudnn.benchmark is False
	assert torch.backends.cudnn.deterministic is True


def test_rng_capture_and_restore_continues_all_cpu_streams() -> None:
	seed_everything(42)
	state = capture_rng_state()
	expected = (random.random(), np.random.rand(), torch.rand(2))
	restore_rng_state(state)
	actual = (random.random(), np.random.rand(), torch.rand(2))

	assert expected[0] == actual[0]
	assert expected[1] == actual[1]
	assert torch.equal(expected[2], actual[2])


def test_checkpoint_restores_trainable_values_optimizer_scheduler_and_cursor(
	tmp_path: Path,
) -> None:
	model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
	model[1].requires_grad_(False)
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		config,
		warm_start_parameter_names=("0.weight", "0.bias"),
		joint_parameter_names=(),
		total_steps=10,
		warm_start_steps=3,
		joint_activation_steps=1,
	)
	cursor = TrainingCursor(
		stage=1,
		global_step=7,
		sampler_epoch=2,
		batch_in_epoch=11,
		gradient_accumulation_step=0,
	)
	checkpoint_path = tmp_path / "checkpoint.pt"
	original_weight = model[0].weight.detach().clone()
	save_training_checkpoint(
		path=checkpoint_path,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		cursor=cursor,
		rank_rng_states=[capture_rng_state()],
		metadata={"test": True},
	)
	with torch.no_grad():
		model[0].weight.add_(1)

	loaded_cursor, metadata = load_training_checkpoint(
		path=checkpoint_path,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		rank=0,
	)

	assert torch.equal(model[0].weight, original_weight)
	assert loaded_cursor == cursor
	assert metadata == {"test": True}


def test_checkpoint_restores_gradient_scaler_state(tmp_path: Path) -> None:
	model = nn.Linear(3, 2)
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		config,
		warm_start_parameter_names=("weight", "bias"),
		joint_parameter_names=(),
		total_steps=10,
		warm_start_steps=3,
		joint_activation_steps=1,
	)
	saved_scaler = _FakeGradientScaler(scale=4096.0)
	restored_scaler = _FakeGradientScaler(scale=1.0)
	checkpoint_path = tmp_path / "checkpoint-with-scaler.pt"

	save_training_checkpoint(
		path=checkpoint_path,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		cursor=TrainingCursor(
			stage=1,
			global_step=3,
			sampler_epoch=0,
			batch_in_epoch=4,
			gradient_accumulation_step=0,
		),
		rank_rng_states=[capture_rng_state()],
		metadata={},
		gradient_scaler=saved_scaler,
	)
	load_training_checkpoint(
		path=checkpoint_path,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		rank=0,
		gradient_scaler=restored_scaler,
	)

	assert restored_scaler.scale == 4096.0


def test_checkpoint_rejects_the_old_two_stage_protocol_before_loading(
	tmp_path: Path,
) -> None:
	model = nn.Linear(3, 2)
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		config,
		warm_start_parameter_names=("weight", "bias"),
		joint_parameter_names=(),
		total_steps=10,
		warm_start_steps=3,
		joint_activation_steps=1,
	)
	checkpoint_path = tmp_path / "old-protocol.pt"
	save_training_checkpoint(
		path=checkpoint_path,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		cursor=TrainingCursor(
			stage=1,
			global_step=3,
			sampler_epoch=0,
			batch_in_epoch=4,
			gradient_accumulation_step=0,
		),
		rank_rng_states=[capture_rng_state()],
		metadata={"training_protocol": "two_stage_v1"},
	)

	with pytest.raises(ValueError, match="single-stage"):
		load_training_checkpoint(
			path=checkpoint_path,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			rank=0,
			expected_training_protocol="single_stage_warm_start_v1",
		)


def test_checkpoint_retention_keeps_only_the_four_newest_files(tmp_path: Path) -> None:
	checkpoint_root = tmp_path / "checkpoints"
	checkpoint_root.mkdir()
	for step in range(1, 7):
		path = checkpoint_root / f"step{step:06d}.pt"
		path.write_bytes(str(step).encode())
		path.touch()

	removed = prune_training_checkpoints(
		checkpoint_root,
		max_checkpoints=4,
	)

	assert [path.name for path in removed] == [
		"step000001.pt",
		"step000002.pt",
	]
	assert sorted(path.name for path in checkpoint_root.glob("*.pt")) == [
		"step000003.pt",
		"step000004.pt",
		"step000005.pt",
		"step000006.pt",
	]


def test_checkpoint_cursor_rebases_exact_sample_position_for_larger_batch() -> None:
	cursor = TrainingCursor(
		stage=1,
		global_step=500,
		sampler_epoch=2,
		batch_in_epoch=28000,
		gradient_accumulation_step=0,
	)

	rebased = rebase_training_cursor_batch_size(
		cursor,
		source_per_device_batch_size=1,
		target_per_device_batch_size=4,
	)

	assert rebased == TrainingCursor(
		stage=1,
		global_step=500,
		sampler_epoch=2,
		batch_in_epoch=7000,
		gradient_accumulation_step=0,
	)


def test_checkpoint_cursor_rejects_inexact_or_partial_batch_rebase() -> None:
	partial = TrainingCursor(
		stage=1,
		global_step=3,
		sampler_epoch=0,
		batch_in_epoch=5,
		gradient_accumulation_step=1,
	)
	with pytest.raises(ValueError, match="accumulation boundary"):
		rebase_training_cursor_batch_size(
			partial,
			source_per_device_batch_size=1,
			target_per_device_batch_size=4,
		)

	inexact = TrainingCursor(
		stage=1,
		global_step=3,
		sampler_epoch=0,
		batch_in_epoch=5,
		gradient_accumulation_step=0,
	)
	with pytest.raises(ValueError, match="not divisible"):
		rebase_training_cursor_batch_size(
			inexact,
			source_per_device_batch_size=1,
			target_per_device_batch_size=4,
		)


def test_effective_batch_size_must_equal_512() -> None:
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	assert config.gradient_accumulation_steps(per_device_batch_size=1, world_size=2) == 256
	assert config.gradient_accumulation_steps(per_device_batch_size=4, world_size=2) == 64
	assert config.gradient_accumulation_steps(per_device_batch_size=8, world_size=2) == 32
	with pytest.raises(ValueError, match="divide"):
		config.gradient_accumulation_steps(per_device_batch_size=3, world_size=2)


def test_training_cli_defaults_to_per_device_batch_eight(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setattr(
		sys,
		"argv",
		[
			"train",
			"--dataset-root",
			str(tmp_path / "coco"),
			"--output-dir",
			str(tmp_path / "training"),
		],
	)

	args = parse_args()

	assert args.dataset_root == tmp_path / "coco"
	assert args.per_device_batch_size == 8
	assert args.attention_implementation == "auto"
	assert args.runtime_precision == "bf16"
	assert args.initial_gradient_scale == 65_536.0
	assert args.max_checkpoints == 4
	assert args.training_config == Path("configs/train.yaml")
	assert not hasattr(args, "start_stage")
	assert not hasattr(args, "semantic_decoder_root")


def test_training_cli_requires_an_explicit_aligned_dataset_root(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setattr(
		sys,
		"argv",
		["train", "--output-dir", str(tmp_path / "training")],
	)

	with pytest.raises(SystemExit):
		parse_args()


def test_training_workers_spawn_without_inheriting_cuda_context() -> None:
	assert _worker_loader_options(num_workers=0, prefetch_factor=4) == {}
	assert _worker_loader_options(num_workers=8, prefetch_factor=4) == {
		"multiprocessing_context": "spawn",
		"persistent_workers": True,
		"prefetch_factor": 4,
	}


def test_smoke_run_skips_large_optimizer_checkpoints() -> None:
	assert _should_save_checkpoint(
		global_step=3,
		optimizer_step_limit=3,
		checkpoint_every=100,
		smoke_optimizer_steps=3,
	) is False
	assert _should_save_checkpoint(
		global_step=100,
		optimizer_step_limit=2000,
		checkpoint_every=100,
		smoke_optimizer_steps=0,
	) is True


def test_dynamic_scaled_dot_product_attention_matches_explicit_attention() -> None:
	torch.manual_seed(7)
	query = torch.randn(2, 4, 3, 8)
	key = torch.randn(2, 4, 7, 8)
	value = torch.randn(2, 4, 7, 8)
	mask = torch.zeros(2, 1, 3, 7)
	mask[:, :, 0, 5:] = torch.finfo(mask.dtype).min
	scale = 8**-0.5
	weights = torch.softmax(
		(query @ key.transpose(2, 3) * scale + mask).float(),
		dim=-1,
	).to(query.dtype)
	expected = weights @ value

	actual = _dynamic_scaled_dot_product_attention(
		query=query,
		key=key,
		value=value,
		attention_mask=mask,
		scale=scale,
	)

	assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_additional_step_limit_is_relative_to_resumed_global_step() -> None:
	assert _optimizer_step_limit(
		configured_steps=2000,
		resumed_global_step=500,
		smoke_optimizer_steps=0,
		max_additional_optimizer_steps=3,
	) == 503
	assert _optimizer_step_limit(
		configured_steps=2000,
		resumed_global_step=1999,
		smoke_optimizer_steps=0,
		max_additional_optimizer_steps=3,
	) == 2000


def test_metric_accumulation_stays_on_device_until_step_boundary() -> None:
	accumulator: dict[str, torch.Tensor] = {}
	first = {"loss": torch.tensor(2.0, requires_grad=True)}
	second = {"loss": torch.tensor(4.0, requires_grad=True)}

	_accumulate_metric_tensors(accumulator, first, ("loss",))
	_accumulate_metric_tensors(accumulator, second, ("loss",))

	assert accumulator["loss"].item() == pytest.approx(6.0)
	assert accumulator["loss"].requires_grad is False
	assert _finalize_metric_tensors(accumulator, count=2) == {"loss": pytest.approx(3.0)}


def test_warm_start_and_joint_loss_weights_are_exact_without_decoder() -> None:
	components = {
		"final_infonce": torch.tensor(10.0),
		"slot_infonce": torch.tensor(2.0),
		"slot_diversity": torch.tensor(4.0),
	}

	warm_start = compose_training_loss(phase="warm_start", **components)
	joint = compose_training_loss(phase="joint", **components)

	assert warm_start.item() == pytest.approx(2.0 + 0.05 * 4.0)
	assert joint.item() == pytest.approx(10.0 + 0.2 * 2.0 + 0.05 * 4.0)


def test_joint_parameter_learning_rate_is_zero_then_smoothly_activates() -> None:
	model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		config,
		warm_start_parameter_names=("0.weight", "0.bias"),
		joint_parameter_names=("1.weight", "1.bias"),
		total_steps=20,
		warm_start_steps=5,
		joint_activation_steps=3,
	)

	assert optimizer.param_groups[1]["lr"] == 0.0
	for _ in range(5):
		optimizer.step()
		scheduler.step()
	assert optimizer.param_groups[1]["lr"] > 0.0
	assert optimizer.param_groups[1]["lr"] <= config.learning_rate


def test_warm_start_discards_joint_gradients_and_adamw_momentum() -> None:
	model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, _ = build_optimizer_and_scheduler(
		model,
		config,
		warm_start_parameter_names=("0.weight", "0.bias"),
		joint_parameter_names=("1.weight", "1.bias"),
		total_steps=20,
		warm_start_steps=5,
		joint_activation_steps=3,
	)
	model(torch.ones(1, 2)).sum().backward()

	_clear_parameter_gradients(model, ("1.weight", "1.bias"))
	optimizer.step()

	assert model[0].weight in optimizer.state
	assert model[1].weight not in optimizer.state
	assert model[1].bias not in optimizer.state


def test_training_phase_transitions_without_a_second_formal_stage() -> None:
	plan = OneEpochTrainingPlan(
		start_batch=0,
		end_batch=100,
		optimizer_steps=20,
		warm_start_optimizer_steps=5,
		joint_optimizer_steps=15,
		joint_activation_optimizer_steps=3,
	)

	assert _training_phase(0, plan) == "warm_start"
	assert _training_phase(4, plan) == "warm_start"
	assert _training_phase(5, plan) == "joint_activation"
	assert _training_phase(7, plan) == "joint_activation"
	assert _training_phase(8, plan) == "joint"


def test_explicit_commit_allows_a_non_git_launch_directory(tmp_path: Path) -> None:
	commit = "a" * 40

	assert _resolve_git_commit(tmp_path, commit) == commit

	with pytest.raises(RuntimeError, match="not a Git checkout"):
		_resolve_git_commit(tmp_path, None)


def test_explicit_commit_must_match_a_git_checkout(tmp_path: Path) -> None:
	subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
	(tmp_path / "tracked.txt").write_text("content\n", encoding="utf-8")
	subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
	subprocess.run(
		[
			"git",
			"-C",
			str(tmp_path),
			"-c",
			"user.name=Test",
			"-c",
			"user.email=test@example.com",
			"commit",
			"-q",
			"-m",
			"test",
		],
		check=True,
	)
	head = subprocess.run(
		["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
		check=True,
		capture_output=True,
		text=True,
	).stdout.strip()

	assert _resolve_git_commit(tmp_path, head) == head
	with pytest.raises(ValueError, match="does not match"):
		_resolve_git_commit(tmp_path, "a" * 40)
