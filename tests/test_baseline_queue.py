from __future__ import annotations

from pathlib import Path

from looped_vl.baseline.queue import (
	BaselineRun,
	build_training_command,
)
from looped_vl.baseline.smoke_search import _project_pythonpath
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
		per_device_batch_size=32,
		gradient_accumulation_steps=1,
		num_workers=6,
		gradient_checkpointing=False,
	)

	command = build_training_command(
		run,
		project_root=tmp_path,
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
	)

	assert command[1:3] == ["-m", "torch.distributed.run"]
	assert command[command.index("--dataset") + 1] == "clevr"
	assert command[command.index("--per-device-batch-size") + 1] == "32"
	assert command[command.index("--gradient-accumulation-steps") + 1] == "1"
	assert command[command.index("--expected-contrastive-global-batch-size") + 1] == "256"
	assert command[command.index("--num-workers") + 1] == "6"
	assert command[command.index("--expected-world-size") + 1] == "8"
	assert "--no-gradient-checkpointing" in command


def test_smoke_search_preserves_the_selected_runtime_pythonpath(tmp_path: Path) -> None:
	pythonpath = _project_pythonpath(
		tmp_path,
		"/usr/local/lib/python3.10/dist-packages",
	)

	assert pythonpath == (
		f"{tmp_path / 'src'}:/usr/local/lib/python3.10/dist-packages"
	)
