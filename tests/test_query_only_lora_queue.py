from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from looped_vl.baseline.query_only_queue import (
	QUERY_ONLY_LORA_DATASETS,
	QueryOnlyLoRARun,
	build_evaluation_command,
	build_training_command,
)


def _args(tmp_path: Path) -> SimpleNamespace:
	return SimpleNamespace(
		world_size=8,
		per_device_batch_size=32,
		evaluation_batch_size=32,
		num_workers=4,
		checkpoint_every=100,
		project_root=tmp_path / "project",
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		candidate_root=tmp_path / "banks",
	)


@pytest.mark.parametrize("dataset", QUERY_ONLY_LORA_DATASETS)
def test_query_only_lora_commands_cover_remaining_answer_datasets(
	dataset: str,
	tmp_path: Path,
) -> None:
	args = _args(tmp_path)
	run = QueryOnlyLoRARun(dataset)
	training_output = tmp_path / "outputs" / dataset / "train"
	evaluation_output = tmp_path / "outputs" / dataset / "test"
	train = build_training_command(run, args=args, output_dir=training_output)
	test = build_evaluation_command(
		run,
		args=args,
		training_output=training_output,
		evaluation_output=evaluation_output,
	)

	assert train[train.index("--dataset") + 1] == dataset
	assert train[train.index("--dataset-root") + 1] == str(args.dataset_root / dataset)
	assert train[train.index("--candidate-root") + 1] == str(args.candidate_root)
	assert train[train.index("--lora-decoder-layer-indices") + 1] == "24,25,26,27"
	assert train[train.index("--hard-negative-count") + 1] == "32"
	assert train[train.index("--epochs") + 1] == "1"
	assert train[train.index("--max-checkpoints") + 1] == "1"
	assert train[train.index("--per-device-batch-size") + 1] == "32"
	assert train[train.index("--expected-contrastive-global-batch-size") + 1] == "256"
	assert all("validation" not in argument for argument in train)

	assert test[test.index("--dataset") + 1] == dataset
	assert test[test.index("--dataset-root") + 1] == str(args.dataset_root / dataset)
	assert test[test.index("--candidate-root") + 1] == str(args.candidate_root)
	assert test[test.index("--adapter-root") + 1] == str(training_output / "adapter")
	assert test[test.index("--batch-size") + 1] == "32"


def test_query_only_lora_resume_keeps_the_same_training_protocol(tmp_path: Path) -> None:
	args = _args(tmp_path)
	checkpoint = tmp_path / "step000700.pt"
	command = build_training_command(
		QueryOnlyLoRARun("gqa_balanced"),
		args=args,
		output_dir=tmp_path / "train",
		resume_checkpoint=checkpoint,
	)

	assert command[command.index("--resume-checkpoint") + 1] == str(checkpoint)
	assert command[command.index("--checkpoint-every") + 1] == "100"
	assert command[command.index("--max-checkpoints") + 1] == "1"


def test_query_only_lora_queue_rejects_non_answer_dataset() -> None:
	with pytest.raises(ValueError, match="GQA Balanced or CLEVR"):
		QueryOnlyLoRARun("coco").validate()
