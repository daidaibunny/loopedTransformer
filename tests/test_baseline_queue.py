from __future__ import annotations

from pathlib import Path

from looped_vl.baseline.queue import (
	BaselineRun,
	build_training_command,
)
from looped_vl.training.wait_and_launch import parse_gpu_snapshot


def test_gpu_snapshot_can_require_all_eight_v100_devices() -> None:
	output = "\n".join(f"{index}, 0, 0" for index in range(8))

	snapshot = parse_gpu_snapshot(
		output,
		compute_process_count=0,
		expected_indexes=tuple(range(8)),
	)

	assert snapshot.is_idle is True
	assert len(snapshot.gpus) == 8


def test_training_command_keeps_dataset_specific_parallel_parameters(tmp_path: Path) -> None:
	run = BaselineRun(
		dataset="clevr",
		per_device_batch_size=8,
		gradient_accumulation_steps=4,
		num_workers=6,
	)

	command = build_training_command(
		run,
		project_root=tmp_path,
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
	)

	assert command[command.index("--dataset") + 1] == "clevr"
	assert command[command.index("--per-device-batch-size") + 1] == "8"
	assert command[command.index("--gradient-accumulation-steps") + 1] == "4"
	assert command[command.index("--num-workers") + 1] == "6"
	assert command[command.index("--expected-world-size") + 1] == "8"
