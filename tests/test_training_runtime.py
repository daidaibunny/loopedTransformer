import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from looped_vl.training.checkpointing import (
	TrainingCursor,
	capture_rng_state,
	load_training_checkpoint,
	restore_rng_state,
	save_training_checkpoint,
)
from looped_vl.training.config import TrainingStageConfig
from looped_vl.training.optimizer import build_optimizer_and_scheduler
from looped_vl.training.reproducibility import seed_everything
from looped_vl.training.step import compose_stage_loss


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


def test_effective_batch_size_must_equal_512() -> None:
	stage = TrainingStageConfig.from_yaml(Path("configs/stage1.yaml"))
	assert stage.gradient_accumulation_steps(per_device_batch_size=1, world_size=2) == 256
	with pytest.raises(ValueError, match="divide"):
		stage.gradient_accumulation_steps(per_device_batch_size=3, world_size=2)


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
