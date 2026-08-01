"""Wait for candidate banks, then safely launch the eight-V100 recurrent queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from looped_vl.query_recurrent.queue import validate_all_candidate_banks


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_status(candidate_root: Path) -> str:
	status_path = candidate_root / "status.json"
	if not status_path.is_file():
		return "missing"
	return str(json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown"))


def _tmux_session_exists(name: str) -> bool:
	return subprocess.run(
		["tmux", "has-session", "-t", name],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		check=False,
	).returncode == 0


def wait_for_candidate_completion(
	*,
	candidate_root: Path,
	candidate_tmux: str,
	poll_seconds: float,
	maximum_wait_seconds: float,
) -> dict[str, str]:
	"""Wait for all READY files and for the candidate owner tmux to release GPUs."""
	start = time.monotonic()
	while True:
		status = _candidate_status(candidate_root)
		if status == "failed":
			raise RuntimeError(f"Candidate-bank encoding failed under {candidate_root}")
		if status == "passed":
			identities = validate_all_candidate_banks(candidate_root)
			if not _tmux_session_exists(candidate_tmux):
				return identities
		if maximum_wait_seconds and time.monotonic() - start >= maximum_wait_seconds:
			raise TimeoutError("Timed out waiting for all candidate banks")
		print(
			f"WAITING_FOR_CANDIDATE_BANKS status={status} elapsed={time.monotonic() - start:.0f}s",
			flush=True,
		)
		time.sleep(poll_seconds)


def validate_gpu_inventory(output: str, *, expected_count: int) -> tuple[str, ...]:
	"""Require exactly the assigned number of NVIDIA V100 devices."""
	names = tuple(line.strip() for line in output.splitlines() if line.strip())
	if len(names) != expected_count or any("V100" not in name for name in names):
		raise RuntimeError(
			f"Expected {expected_count} V100 GPUs, found {len(names)}: {names}",
		)
	return names


def _pause_guard(guard_script: Path) -> None:
	subprocess.run(["bash", str(guard_script), "pause"], check=True)
	status = subprocess.run(
		["bash", str(guard_script), "status"],
		check=True,
		capture_output=True,
		text=True,
	).stdout
	if "control=paused" not in status or "monitor=running" not in status:
		raise RuntimeError(f"GPU guard did not reach the safe paused state: {status!r}")


def _wait_for_gpu_release(*, poll_seconds: float) -> None:
	while True:
		processes = subprocess.run(
			[
				"nvidia-smi",
				"--query-compute-apps=pid",
				"--format=csv,noheader",
			],
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		if not processes:
			return
		print(f"WAITING_FOR_GPU_RELEASE pids={processes!r}", flush=True)
		time.sleep(poll_seconds)


def _assert_old_recurrent_absent() -> None:
	result = subprocess.run(
		["pgrep", "-f", "[l]ooped_vl.recurrent_v5_queue"],
		check=False,
		capture_output=True,
		text=True,
	)
	if result.returncode == 0 and result.stdout.strip():
		raise RuntimeError(f"Canceled recurrent v5 queue is still present: {result.stdout.strip()}")


def _queue_command(args: argparse.Namespace) -> list[str]:
	return [
		sys.executable,
		"-m",
		"looped_vl.query_recurrent.queue",
		"--project-root",
		str(args.project_root),
		"--dataset-root",
		str(args.dataset_root),
		"--model-root",
		str(args.model_root),
		"--candidate-root",
		str(args.candidate_root),
		"--output-root",
		str(args.output_root),
		"--world-size",
		str(args.world_size),
		"--per-device-batch-size",
		str(args.per_device_batch_size),
		"--evaluation-batch-size",
		str(args.evaluation_batch_size),
		"--num-workers",
		str(args.num_workers),
	]


def run_launch(args: argparse.Namespace) -> None:
	status_path = Path(args.output_root) / "launcher_status.json"
	actual_commit = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=args.project_root,
		check=True,
		capture_output=True,
		text=True,
	).stdout.strip()
	if actual_commit != args.expected_commit:
		raise RuntimeError(
			f"Recurrent worktree commit is {actual_commit}, expected {args.expected_commit}",
		)
	_write_json(
		status_path,
		{"status": "waiting_for_candidate_banks", "git_commit": actual_commit},
	)
	bank_identities = wait_for_candidate_completion(
		candidate_root=Path(args.candidate_root),
		candidate_tmux=args.candidate_tmux,
		poll_seconds=args.poll_seconds,
		maximum_wait_seconds=args.maximum_wait_seconds,
	)
	_assert_old_recurrent_absent()
	guard_paused = False
	try:
		_pause_guard(Path(args.guard_script))
		guard_paused = True
		_wait_for_gpu_release(poll_seconds=args.poll_seconds)
		gpu_output = subprocess.run(
			["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
			check=True,
			capture_output=True,
			text=True,
		).stdout
		gpu_names = validate_gpu_inventory(gpu_output, expected_count=args.world_size)
		command = _queue_command(args)
		_write_json(
			status_path,
			{
				"status": "running",
				"git_commit": actual_commit,
				"gpu_names": gpu_names,
				"candidate_bank_manifest_sha256": bank_identities,
				"command": command,
			},
		)
		environment = os.environ.copy()
		environment["PYTHONPATH"] = str(Path(args.project_root) / "src")
		result = subprocess.run(
			command,
			cwd=args.project_root,
			env=environment,
			check=False,
		)
		if result.returncode:
			raise RuntimeError(f"Query recurrent queue exited with {result.returncode}")
		_write_json(status_path, {"status": "passed", "git_commit": actual_commit})
	finally:
		if guard_paused:
			subprocess.run(["bash", str(args.guard_script), "resume"], check=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--expected-commit", required=True)
	parser.add_argument("--project-root", type=Path, required=True)
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--model-root", type=Path, required=True)
	parser.add_argument("--candidate-root", type=Path, required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--candidate-tmux", required=True)
	parser.add_argument(
		"--guard-script",
		type=Path,
		default=Path("/home/mnt/liyiwei/project/gpu_idle_guard.sh"),
	)
	parser.add_argument("--world-size", type=int, default=8)
	parser.add_argument("--per-device-batch-size", type=int, default=32)
	parser.add_argument("--evaluation-batch-size", type=int, default=32)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--poll-seconds", type=float, default=30.0)
	parser.add_argument("--maximum-wait-seconds", type=float, default=0.0)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_launch(args)
		return 0
	except Exception as error:
		print(f"query recurrent launch failed: {error!r}", file=sys.stderr, flush=True)
		status_path = Path(args.output_root) / "launcher_status.json"
		_write_json(status_path, {"status": "failed", "error": repr(error)})
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
