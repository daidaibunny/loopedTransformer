from __future__ import annotations

from pathlib import Path

from looped_vl.recurrent_v5_queue import (
	RecurrentV5Run,
	build_evaluation_command,
	build_training_command,
	default_runs,
)
from looped_vl.training.trainability import MAX_RECURRENT_TRAINABLE_PARAMETERS


def test_v5_queue_selects_three_high_potential_full_experiments() -> None:
	runs = default_runs()

	assert [(run.name, run.dataset, run.num_latent_slots, run.step_size) for run in runs] == [
		("coco_k16_alpha1", "coco", 16, 1.0),
		("gqa_k16_alpha1", "gqa_balanced", 16, 1.0),
		("gqa_k32_alpha1", "gqa_balanced", 32, 1.0),
	]
	assert MAX_RECURRENT_TRAINABLE_PARAMETERS == 5_000_000


def test_v5_training_command_uses_full_batch_layerscale_and_one_checkpoint(
	tmp_path: Path,
) -> None:
	run = RecurrentV5Run("gqa_k32_alpha1", "gqa_balanced", 32, 1.0)
	command = build_training_command(
		run,
		project_root=tmp_path / "project",
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
		code_commit="abc123",
		checkpoint_every=100,
		resume_checkpoint=None,
	)

	assert command[command.index("--num-latent-slots") + 1] == "32"
	assert command[command.index("--recurrent-step-size") + 1] == "1.0"
	assert "--use-recurrent-layer-scale" in command
	assert command[command.index("--per-device-batch-size") + 1] == "32"
	assert command[command.index("--expected-contrastive-global-batch-size") + 1] == "256"
	assert command[command.index("--max-checkpoints") + 1] == "1"
	assert command[command.index("--visual-length-buckets") + 1] == "3"
	assert "--gradient-checkpointing" in command
	assert "--resume-checkpoint" not in command


def test_v5_evaluation_command_matches_training_variant(tmp_path: Path) -> None:
	run = RecurrentV5Run("coco_k16_alpha1", "coco", 16, 1.0)
	checkpoint = tmp_path / "latest.pt"
	command = build_evaluation_command(
		run,
		project_root=tmp_path / "project",
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
		checkpoint=checkpoint,
	)

	assert command[command.index("--source") + 1] == "coco"
	assert command[command.index("--num-latent-slots") + 1] == "16"
	assert command[command.index("--recurrent-step-size") + 1] == "1.0"
	assert command[command.index("--checkpoint") + 1] == str(checkpoint)
	assert command[command.index("--batch-size") + 1] == "32"
