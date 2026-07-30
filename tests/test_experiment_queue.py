from pathlib import Path

from looped_vl.experiment_queue import (
	ExperimentSpec,
	build_evaluation_command,
	build_training_command,
	default_experiments,
)


def _value_after(command: list[str], option: str) -> str:
	return command[command.index(option) + 1]


def test_default_queue_contains_six_serial_single_dataset_experiments() -> None:
	experiments = default_experiments()

	assert [(item.family, item.dataset) for item in experiments] == [
		("baseline", "coco"),
		("baseline", "gqa_balanced"),
		("baseline", "clevr"),
		("recurrent", "coco"),
		("recurrent", "gqa_balanced"),
		("recurrent", "clevr"),
	]
	assert all(item.gradient_checkpointing for item in experiments)


def test_baseline_and_recurrent_commands_share_checkpoint_and_test_contract(
	tmp_path: Path,
) -> None:
	common = {
		"project_root": tmp_path / "project",
		"dataset_root": tmp_path / "datasets",
		"model_root": tmp_path / "model",
		"output_root": tmp_path / "outputs",
		"world_size": 8,
		"code_commit": "a" * 40,
		"checkpoint_every": 100,
		"max_checkpoints": 4,
		"resume_checkpoint": None,
	}
	for family in ("baseline", "recurrent"):
		spec = ExperimentSpec(
			family=family,
			dataset="coco",
			train_batch_size=32 if family == "baseline" else 8,
			train_workers=4,
			evaluation_batch_size=32 if family == "baseline" else 8,
			evaluation_workers=4,
		)
		training = build_training_command(spec, **common)
		evaluation = build_evaluation_command(
			spec,
			project_root=common["project_root"],
			dataset_root=common["dataset_root"],
			model_root=common["model_root"],
			output_root=common["output_root"],
			world_size=8,
		)

		assert _value_after(training, "--checkpoint-every") == "100"
		assert _value_after(training, "--max-checkpoints") == "4"
		assert "--validation" not in training
		assert "validation" not in evaluation
		if family == "baseline":
			assert _value_after(training, "--epochs") == "1"
			assert _value_after(training, "--expected-contrastive-global-batch-size") == "256"
		else:
			assert _value_after(training, "--expected-contrastive-global-batch-size") == "64"
			assert _value_after(training, "--initial-gradient-scale") == "32"
			assert "--gradient-checkpointing" in training
			assert _value_after(evaluation, "--split") == "test"
