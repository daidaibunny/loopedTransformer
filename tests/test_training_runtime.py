import json
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
	prepare_training_output_directory,
	prune_training_checkpoints,
	publish_latest_training_checkpoint,
	rebase_training_cursor_batch_size,
	restore_rng_state,
	save_training_checkpoint,
	truncate_metric_log,
	validate_checkpoint_metadata,
)
from looped_vl.training.config import TrainingConfig
from looped_vl.training.optimizer import build_optimizer_and_scheduler
from looped_vl.training.reproducibility import seed_everything
from looped_vl.training.schedule import (
	FORMAL_TRAINING_LOG_INTERVAL,
	OneEpochTrainingPlan,
	should_log_training_metrics,
)
from looped_vl.training.step import compose_training_loss
from looped_vl.training.train import (
	_accumulate_metric_tensors,
	_distributed_data_parallel_options,
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


def test_formal_training_logs_every_fifty_steps_and_at_boundaries() -> None:
	assert FORMAL_TRAINING_LOG_INTERVAL == 50
	assert not should_log_training_metrics(
		optimizer_steps_since_log=49,
		global_step=49,
		optimizer_step_limit=120,
	)
	assert should_log_training_metrics(
		optimizer_steps_since_log=50,
		global_step=50,
		optimizer_step_limit=120,
	)
	assert should_log_training_metrics(
		optimizer_steps_since_log=20,
		global_step=120,
		optimizer_step_limit=120,
	)
	assert should_log_training_metrics(
		optimizer_steps_since_log=17,
		global_step=67,
		optimizer_step_limit=120,
		force_boundary=True,
	)


def test_smoke_training_logs_every_optimizer_step() -> None:
	assert should_log_training_metrics(
		optimizer_steps_since_log=1,
		global_step=1,
		optimizer_step_limit=3,
		force_every_step=True,
	)
	assert should_log_training_metrics(
		optimizer_steps_since_log=1,
		global_step=2,
		optimizer_step_limit=3,
		force_every_step=True,
	)


def test_recurrent_ddp_options_support_gradient_accumulation() -> None:
	options = _distributed_data_parallel_options()

	assert options == {
		"broadcast_buffers": False,
		"find_unused_parameters": False,
		"gradient_as_bucket_view": True,
		"static_graph": False,
	}


def test_single_training_config_matches_full_objective_protocol() -> None:
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
	assert not hasattr(config, "auxiliary_emphasis_epoch_fraction")


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
		recurrent_core_parameter_names=("0.weight", "0.bias"),
		final_fusion_parameter_names=(),
		total_steps=10,
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
		recurrent_core_parameter_names=("weight", "bias"),
		final_fusion_parameter_names=(),
		total_steps=10,
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


@pytest.mark.parametrize(
	"old_protocol",
	[
		"two_stage_v1",
		"single_stage_warm_start_v1",
		"pure_recurrent_single_stage_v1",
		"pure_recurrent_full_objective_v2",
	],
)
def test_checkpoint_rejects_non_pure_recurrent_protocol_before_loading(
	tmp_path: Path,
	old_protocol: str,
) -> None:
	model = nn.Linear(3, 2)
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		config,
		recurrent_core_parameter_names=("weight", "bias"),
		final_fusion_parameter_names=(),
		total_steps=10,
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
		metadata={"training_protocol": old_protocol},
	)

	with pytest.raises(ValueError, match="damped recurrent"):
		load_training_checkpoint(
			path=checkpoint_path,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			rank=0,
			expected_training_protocol="pure_recurrent_single_stage_eos_weighted_aux_v4",
		)


def test_checkpoint_retention_keeps_only_the_latest_file(tmp_path: Path) -> None:
	checkpoint_root = tmp_path / "checkpoints"
	checkpoint_root.mkdir()
	for step in range(1, 7):
		path = checkpoint_root / f"step{step:06d}.pt"
		path.write_bytes(str(step).encode())
		path.touch()

	removed = prune_training_checkpoints(
		checkpoint_root,
		max_checkpoints=1,
	)

	assert [path.name for path in removed] == [
		"step000001.pt",
		"step000002.pt",
		"step000003.pt",
		"step000004.pt",
		"step000005.pt",
	]
	assert sorted(path.name for path in checkpoint_root.glob("*.pt")) == [
		"step000006.pt",
	]


def test_latest_checkpoint_pointer_is_published_before_old_file_is_removed(
	tmp_path: Path,
) -> None:
	checkpoint_root = tmp_path / "train" / "checkpoints"
	checkpoint_root.mkdir(parents=True)
	old_checkpoint = checkpoint_root / "step000100.pt"
	new_checkpoint = checkpoint_root / "step000200.pt"
	old_checkpoint.write_bytes(b"old")
	new_checkpoint.write_bytes(b"new")
	cursor = TrainingCursor(
		stage=0,
		global_step=200,
		sampler_epoch=0,
		batch_in_epoch=200,
		gradient_accumulation_step=0,
		processed_samples=51200,
	)

	removed = publish_latest_training_checkpoint(
		new_checkpoint,
		cursor,
		max_checkpoints=1,
	)

	assert removed == [old_checkpoint]
	assert not old_checkpoint.exists()
	assert new_checkpoint.exists()
	assert json.loads(
		(tmp_path / "train" / "latest_checkpoint.json").read_text(encoding="utf-8"),
	) == {
		"path": str(new_checkpoint),
		"cursor": {
			"batch_in_epoch": 200,
			"global_step": 200,
			"gradient_accumulation_step": 0,
			"processed_samples": 51200,
			"sampler_epoch": 0,
			"stage": 0,
		},
	}


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
			"--num-latent-slots",
			"64",
		],
	)

	args = parse_args()

	assert args.dataset_root == tmp_path / "coco"
	assert args.per_device_batch_size == 8
	assert args.attention_implementation == "auto"
	assert args.runtime_precision == "fp16"
	assert args.initial_gradient_scale == 32.0
	assert args.checkpoint_every == 100
	assert args.max_checkpoints == 1
	assert args.training_config == Path("configs/train.yaml")
	assert args.num_latent_slots == 64
	assert not hasattr(args, "start_stage")
	assert not hasattr(args, "semantic_decoder_root")


def test_training_output_directory_supports_exact_in_place_resume(tmp_path: Path) -> None:
	output_dir = tmp_path / "training"
	prepare_training_output_directory(output_dir, resume_checkpoint=None)
	checkpoint = output_dir / "checkpoints" / "step000100.pt"
	checkpoint.write_bytes(b"checkpoint")

	mode = prepare_training_output_directory(
		output_dir,
		resume_checkpoint=checkpoint,
	)

	assert mode == "resume"
	with pytest.raises(ValueError, match="belong"):
		prepare_training_output_directory(
			output_dir,
			resume_checkpoint=tmp_path / "outside.pt",
		)


def test_resume_metadata_and_metric_log_reject_or_remove_drift(tmp_path: Path) -> None:
	validate_checkpoint_metadata(
		{"dataset_root": "/data/coco", "world_size": 8},
		expected={"dataset_root": "/data/coco", "world_size": 8},
	)
	with pytest.raises(ValueError, match="world_size"):
		validate_checkpoint_metadata(
			{"dataset_root": "/data/coco", "world_size": 2},
			expected={"dataset_root": "/data/coco", "world_size": 8},
		)
	metrics = tmp_path / "train_metrics.jsonl"
	metrics.write_text(
		'{"global_step": 99, "loss": 2.0}\n'
		'{"global_step": 100, "loss": 1.0}\n'
		'{"global_step": 101, "loss": 0.5}\n',
		encoding="utf-8",
	)

	truncate_metric_log(metrics, maximum_global_step=100)

	assert metrics.read_text(encoding="utf-8").splitlines() == [
		'{"global_step": 99, "loss": 2.0}',
		'{"global_step": 100, "loss": 1.0}',
	]


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
		global_step=3,
		optimizer_step_limit=3,
		checkpoint_every=100,
		smoke_optimizer_steps=3,
		smoke_save_final_checkpoint=True,
	) is True
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


def test_single_stage_loss_weights_are_fixed_during_the_entire_epoch() -> None:
	components = {
		"final_infonce": torch.tensor(10.0),
		"loop_infonce": torch.tensor(2.0),
		"slot_diversity": torch.tensor(4.0),
	}

	total = compose_training_loss(**components)

	assert total.item() == pytest.approx(10.0 + 0.1 * 2.0 + 0.05 * 4.0)


def test_every_recurrent_parameter_group_uses_the_same_lr_from_step_one() -> None:
	model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(
		model,
		config,
		recurrent_core_parameter_names=("0.weight", "0.bias"),
		final_fusion_parameter_names=("1.weight", "1.bias"),
		total_steps=20,
	)

	assert optimizer.param_groups[0]["lr"] > 0.0
	assert optimizer.param_groups[1]["lr"] == optimizer.param_groups[0]["lr"]
	for _ in range(5):
		optimizer.step()
		scheduler.step()
		assert optimizer.param_groups[1]["lr"] == optimizer.param_groups[0]["lr"]


def test_first_optimizer_step_updates_both_recurrent_parameter_groups() -> None:
	model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
	config = TrainingConfig.from_yaml(Path("configs/train.yaml"))
	optimizer, _ = build_optimizer_and_scheduler(
		model,
		config,
		recurrent_core_parameter_names=("0.weight", "0.bias"),
		final_fusion_parameter_names=("1.weight", "1.bias"),
		total_steps=20,
	)
	model(torch.ones(1, 2)).sum().backward()

	optimizer.step()

	assert model[0].weight in optimizer.state
	assert model[1].weight in optimizer.state
	assert model[1].bias in optimizer.state


def test_training_phase_is_single_stage_for_the_full_epoch() -> None:
	plan = OneEpochTrainingPlan(
		start_batch=0,
		end_batch=100,
		optimizer_steps=20,
	)

	for global_step in range(20):
		assert _training_phase(global_step, plan) == "single_stage"


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
