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
	rebase_training_cursor_batch_size,
	restore_rng_state,
	save_training_checkpoint,
)
from looped_vl.training.config import TrainingStageConfig
from looped_vl.training.optimizer import build_optimizer_and_scheduler
from looped_vl.training.reproducibility import seed_everything
from looped_vl.training.step import compose_stage_loss
from looped_vl.training.train import (
	_accumulate_metric_tensors,
	_finalize_metric_tensors,
	_optimizer_step_limit,
	_resolve_git_commit,
	_should_save_checkpoint,
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


def test_stage_configs_match_all_fixed_optimizer_values() -> None:
	stage1 = TrainingStageConfig.from_yaml(Path("configs/stage1.yaml"))
	stage2 = TrainingStageConfig.from_yaml(Path("configs/stage2.yaml"))

	for stage, expected_number, expected_steps in ((stage1, 1, 2000), (stage2, 2, 3200)):
		assert stage.stage == expected_number
		assert stage.steps == expected_steps
		assert stage.optimizer == "AdamW"
		assert stage.learning_rate == 1e-5
		assert stage.weight_decay == 0.01
		assert stage.betas == (0.9, 0.95)
		assert stage.eps == 1e-8
		assert stage.effective_batch_size == 512
		assert stage.gradient_clip_norm == 1.0
		assert stage.precision == "bf16"
		assert stage.lr_scheduler == "cosine"
		assert stage.warmup_ratio == 0.03


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
	stage = TrainingStageConfig.from_yaml(Path("configs/stage1.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(model, stage)
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
	stage = TrainingStageConfig.from_yaml(Path("configs/stage1.yaml"))
	optimizer, scheduler = build_optimizer_and_scheduler(model, stage)
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
	stage = TrainingStageConfig.from_yaml(Path("configs/stage1.yaml"))
	assert stage.gradient_accumulation_steps(per_device_batch_size=1, world_size=2) == 256
	assert stage.gradient_accumulation_steps(per_device_batch_size=4, world_size=2) == 64
	assert stage.gradient_accumulation_steps(per_device_batch_size=8, world_size=2) == 32
	with pytest.raises(ValueError, match="divide"):
		stage.gradient_accumulation_steps(per_device_batch_size=3, world_size=2)


def test_training_cli_defaults_to_per_device_batch_eight(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setattr(
		sys,
		"argv",
		["train", "--output-dir", str(tmp_path / "training")],
	)

	args = parse_args()

	assert args.per_device_batch_size == 8
	assert args.attention_implementation == "auto"
	assert args.runtime_precision == "bf16"
	assert args.initial_gradient_scale == 65_536.0
	assert args.semantic_gradient_checkpointing is False


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


def test_stage_loss_weights_are_exact() -> None:
	components = {
		"final_infonce": torch.tensor(10.0),
		"slot_infonce": torch.tensor(2.0),
		"semantic_decoder_ce": torch.tensor(3.0),
		"slot_diversity": torch.tensor(4.0),
	}

	stage1 = compose_stage_loss(stage=1, **components)
	stage2 = compose_stage_loss(stage=2, **components)

	assert stage1.item() == pytest.approx(2.0 + 3.0 + 0.05 * 4.0)
	assert stage2.item() == pytest.approx(10.0 + 0.2 * 2.0 + 0.2 * 3.0 + 0.05 * 4.0)


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
