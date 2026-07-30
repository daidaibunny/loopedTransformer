from argparse import Namespace
from pathlib import Path

from looped_vl.training.wait_and_launch import (
	ContinuousIdleWindow,
	_training_command,
	parse_gpu_snapshot,
)


def test_idle_window_requires_three_continuous_minutes_and_resets_on_busy() -> None:
	window = ContinuousIdleWindow(required_seconds=180.0)

	assert window.update(is_idle=True, now=10.0) is False
	assert window.update(is_idle=True, now=189.9) is False
	assert window.update(is_idle=False, now=190.0) is False
	assert window.update(is_idle=True, now=200.0) is False
	assert window.update(is_idle=True, now=380.0) is True


def test_gpu_snapshot_requires_both_cards_zero_utilization_and_no_processes() -> None:
	idle = parse_gpu_snapshot("0, 2, 0\n1, 2, 0\n", compute_process_count=0)
	busy_memory = parse_gpu_snapshot("0, 200, 0\n1, 2, 0\n", compute_process_count=0)
	busy_compute = parse_gpu_snapshot("0, 2, 10\n1, 2, 0\n", compute_process_count=0)
	busy_process = parse_gpu_snapshot("0, 2, 0\n1, 2, 0\n", compute_process_count=1)

	assert idle.is_idle is True
	assert busy_memory.is_idle is False
	assert busy_compute.is_idle is False
	assert busy_process.is_idle is False


def test_full_training_command_uses_per_device_batch_eight(tmp_path: Path) -> None:
	args = Namespace(num_workers=8, checkpoint_every=100)

	command = _training_command(args, tmp_path / "training", smoke=False)

	assert command[command.index("--per-device-batch-size") + 1] == "8"


def test_smoke_command_keeps_per_device_batch_one(tmp_path: Path) -> None:
	args = Namespace(num_workers=8, checkpoint_every=100)

	command = _training_command(args, tmp_path / "smoke", smoke=True)

	assert command[command.index("--per-device-batch-size") + 1] == "1"
