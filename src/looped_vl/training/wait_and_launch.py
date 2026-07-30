"""Check both GPUs once, run a two-GPU smoke, then launch full training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuState:
	"""One physical GPU's memory and compute utilization snapshot."""

	index: int
	memory_used_mib: int
	utilization_percent: int


@dataclass(frozen=True)
class GpuSnapshot:
	"""Both assigned cards plus the global compute-process count."""

	gpus: tuple[GpuState, ...]
	compute_process_count: int
	is_idle: bool


class ContinuousIdleWindow:
	"""Track an idle interval and reset it on every busy observation."""

	def __init__(self, required_seconds: float) -> None:
		if required_seconds <= 0:
			raise ValueError("required_seconds must be positive")
		self.required_seconds = required_seconds
		self.idle_since: float | None = None

	def update(self, *, is_idle: bool, now: float) -> bool:
		"""Return true only after an uninterrupted required idle duration."""
		if not is_idle:
			self.idle_since = None
			return False
		if self.idle_since is None:
			self.idle_since = now
		return now - self.idle_since >= self.required_seconds

	def elapsed(self, now: float) -> float:
		"""Return current continuous-idle seconds."""
		return 0.0 if self.idle_since is None else max(0.0, now - self.idle_since)


def parse_gpu_snapshot(gpu_output: str, compute_process_count: int) -> GpuSnapshot:
	"""Parse two physical A800 states and apply the strict idle definition."""
	gpus: list[GpuState] = []
	for line in gpu_output.splitlines():
		if not line.strip():
			continue
		parts = [part.strip() for part in line.split(",")]
		if len(parts) != 3:
			raise ValueError(f"Unexpected nvidia-smi GPU row: {line}")
		gpus.append(
			GpuState(
				index=int(parts[0]),
				memory_used_mib=int(parts[1]),
				utilization_percent=int(parts[2]),
			),
		)
	expected_indexes = (0, 1)
	if tuple(gpu.index for gpu in gpus) != expected_indexes:
		raise RuntimeError(f"Expected physical GPUs {expected_indexes}, found {gpus}")
	is_idle = compute_process_count == 0 and all(
		gpu.memory_used_mib <= 128 and gpu.utilization_percent == 0 for gpu in gpus
	)
	return GpuSnapshot(
		gpus=tuple(gpus),
		compute_process_count=compute_process_count,
		is_idle=is_idle,
	)


def query_gpu_snapshot() -> GpuSnapshot:
	"""Read both assigned GPUs and every active compute process."""
	gpu_result = subprocess.run(
		[
			"nvidia-smi",
			"--query-gpu=index,memory.used,utilization.gpu",
			"--format=csv,noheader,nounits",
		],
		check=True,
		capture_output=True,
		text=True,
	)
	process_result = subprocess.run(
		[
			"nvidia-smi",
			"--query-compute-apps=pid",
			"--format=csv,noheader,nounits",
		],
		check=True,
		capture_output=True,
		text=True,
	)
	process_count = len([line for line in process_result.stdout.splitlines() if line.strip()])
	return parse_gpu_snapshot(gpu_result.stdout, process_count)


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_json_line(path: Path, value: Any) -> None:
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, sort_keys=True) + "\n")


def wait_for_idle_window(
	*,
	required_seconds: float,
	poll_seconds: float,
	log_path: Path,
) -> GpuSnapshot:
	"""Block until both GPUs satisfy the strict idle rule continuously."""
	window = ContinuousIdleWindow(required_seconds)
	while True:
		now = time.monotonic()
		try:
			snapshot = query_gpu_snapshot()
			ready = window.update(is_idle=snapshot.is_idle, now=now)
			record = {
				"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
				"gpus": [asdict(gpu) for gpu in snapshot.gpus],
				"compute_process_count": snapshot.compute_process_count,
				"is_idle": snapshot.is_idle,
				"continuous_idle_seconds": window.elapsed(now),
				"required_idle_seconds": required_seconds,
			}
		except Exception as error:
			window.update(is_idle=False, now=now)
			record = {
				"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
				"is_idle": False,
				"continuous_idle_seconds": 0.0,
				"required_idle_seconds": required_seconds,
				"error": repr(error),
			}
			ready = False
		_append_json_line(log_path, record)
		print(json.dumps(record, sort_keys=True), flush=True)
		if ready:
			return snapshot
		time.sleep(poll_seconds)


def _training_command(
	args: argparse.Namespace,
	output_dir: Path,
	*,
	smoke: bool,
) -> list[str]:
	torchrun = Path(sys.executable).with_name("torchrun")
	command = [
		str(torchrun),
		"--standalone",
		"--nproc_per_node=2",
		"-m",
		"looped_vl.training.train",
		"--expected-world-size",
		"2",
		"--output-dir",
		str(output_dir),
		"--per-device-batch-size",
		"1" if smoke else "8",
		"--num-workers",
		str(args.num_workers),
		"--checkpoint-every",
		str(args.checkpoint_every),
	]
	if smoke:
		command.extend(
			[
				"--start-stage",
				"1",
				"--end-stage",
				"2",
				"--smoke-optimizer-steps",
				"1",
				"--smoke-gradient-accumulation-steps",
				"1",
				"--num-workers",
				"0",
				"--checkpoint-every",
				"1",
			],
		)
	return command


def run_wait_smoke_and_train(args: argparse.Namespace) -> None:
	"""Perform immediate GPU checks, two-stage smoke, then the full run."""
	launcher_output = Path(args.launcher_output_dir)
	if launcher_output.exists():
		raise FileExistsError(f"Launcher output already exists: {launcher_output}")
	launcher_output.mkdir(parents=True)
	status_path = launcher_output / "status.json"
	_write_json(status_path, {"status": "checking_gpus_before_smoke"})
	passed_snapshot = query_gpu_snapshot()
	_write_json(
		launcher_output / "gpu_state_before_smoke.json",
		{
			"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"snapshot": {
				"gpus": [asdict(gpu) for gpu in passed_snapshot.gpus],
				"compute_process_count": passed_snapshot.compute_process_count,
				"is_idle": passed_snapshot.is_idle,
			},
		},
	)
	if not passed_snapshot.is_idle:
		raise RuntimeError("Both GPUs must be idle at submission time")
	environment = os.environ.copy()
	environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environment["CUDA_VISIBLE_DEVICES"] = "0,1"
	environment["PYTHONPATH"] = "src"
	_write_json(status_path, {"status": "running_two_gpu_smoke"})
	smoke_command = _training_command(args, Path(args.smoke_output_dir), smoke=True)
	smoke_result = subprocess.run(
		smoke_command,
		cwd=args.project_root,
		env=environment,
		check=False,
	)
	if smoke_result.returncode != 0:
		raise RuntimeError(f"Two-GPU smoke failed with exit code {smoke_result.returncode}")
	smoke_status = json.loads(
		(Path(args.smoke_output_dir) / "status.json").read_text(encoding="utf-8"),
	)
	if smoke_status.get("status") != "passed":
		raise RuntimeError(f"Two-GPU smoke did not pass: {smoke_status}")
	_write_json(status_path, {"status": "checking_gpus_before_full_training"})
	training_snapshot = query_gpu_snapshot()
	_write_json(
		launcher_output / "gpu_state_before_full_training.json",
		{
			"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"snapshot": {
				"gpus": [asdict(gpu) for gpu in training_snapshot.gpus],
				"compute_process_count": training_snapshot.compute_process_count,
				"is_idle": training_snapshot.is_idle,
			},
		},
	)
	if not training_snapshot.is_idle:
		raise RuntimeError("Both GPUs must be idle before full training")
	_write_json(status_path, {"status": "running_full_training"})
	training_command = _training_command(args, Path(args.train_output_dir), smoke=False)
	training_result = subprocess.run(
		training_command,
		cwd=args.project_root,
		env=environment,
		check=False,
	)
	if training_result.returncode != 0:
		raise RuntimeError(f"Full training failed with exit code {training_result.returncode}")
	_write_json(status_path, {"status": "passed"})


def parse_args() -> argparse.Namespace:
	"""Parse immediate GPU check, smoke, and full-run paths."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/loopedTransformer"),
	)
	parser.add_argument("--launcher-output-dir", type=Path, required=True)
	parser.add_argument("--smoke-output-dir", type=Path, required=True)
	parser.add_argument("--train-output-dir", type=Path, required=True)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--checkpoint-every", type=int, default=500)
	return parser.parse_args()


def main() -> int:
	"""Check GPUs, validate both ranks, and keep training attached to this process."""
	args = parse_args()
	try:
		run_wait_smoke_and_train(args)
		return 0
	except Exception as error:
		launcher_output = Path(args.launcher_output_dir)
		if launcher_output.exists():
			_write_json(
				launcher_output / "status.json",
				{"status": "failed", "error": repr(error)},
			)
		print(f"wait-and-launch failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
