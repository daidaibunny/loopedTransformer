"""Hourly read-only monitor for the parallel-world and LoRA serial queue."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from looped_vl.query_recurrent.queue import QUERY_ONLY_LORA_RUNS


def _latest_evaluation_path(run_root: Path) -> Path:
	"""Follow the published retry pointer, or the newest in-progress retry."""
	pointer_path = run_root / "latest_test.json"
	if pointer_path.is_file():
		path = Path(json.loads(pointer_path.read_text(encoding="utf-8"))["path"])
		if not path.resolve().is_relative_to(run_root.resolve()):
			raise ValueError(f"Evaluation pointer escapes its run root: {path}")
		return path
	retry_paths = tuple(sorted(path for path in run_root.glob("test_retry_*") if path.is_dir()))
	if retry_paths:
		return retry_paths[-1]
	return run_root / "test"


def expected_stage_paths(
	*,
	output_root: Path,
	control_output_root: Path,
	existing_coco_control_run_root: Path | None = None,
) -> tuple[tuple[str, Path], ...]:
	"""Return the only valid production order for this launch."""
	recurrent_root = output_root / "coco_v11_p4_r4_final_mean"
	stages: list[tuple[str, Path]] = [
		(
			"smoke_coco_v11_p4_r4_final_mean",
			output_root / "smoke_coco_v11_p4_r4_final_mean",
		),
		("coco_v11_p4_r4_final_mean_train", recurrent_root / "train"),
		(
			"coco_v11_p4_r4_final_mean_test",
			_latest_evaluation_path(recurrent_root),
		),
	]
	for run in QUERY_ONLY_LORA_RUNS:
		run_root = (
			existing_coco_control_run_root
			if run.dataset == "coco" and existing_coco_control_run_root is not None
			else control_output_root / run.name
		)
		stages.extend(
			(
				(f"{run.name}_train", run_root / "train"),
				(f"{run.name}_test", _latest_evaluation_path(run_root)),
			),
		)
	return tuple(stages)


def _read_status(path: Path) -> str:
	status_path = path / "status.json"
	if not status_path.is_file():
		return "pending"
	return str(json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown"))


def read_stage_statuses(
	stages: tuple[tuple[str, Path], ...],
) -> tuple[dict[str, Any], ...]:
	"""Read stage status and enforce the one-rolling-checkpoint contract."""
	statuses = []
	for name, path in stages:
		checkpoint_paths = tuple(sorted((path / "checkpoints").glob("*.pt")))
		if len(checkpoint_paths) > 1:
			raise RuntimeError(f"{name} has more than one rolling checkpoints")
		statuses.append(
			{
				"name": name,
				"path": str(path),
				"status": _read_status(path),
				"rolling_checkpoints": tuple(str(item) for item in checkpoint_paths),
				"final_model": str(path / "query_recurrent_model.pt")
				if (path / "query_recurrent_model.pt").is_file()
				else None,
				"final_adapter": str(path / "adapter" / "adapter_model.safetensors")
				if (path / "adapter" / "adapter_model.safetensors").is_file()
				else None,
			},
		)
	return tuple(statuses)


def validate_stage_order(statuses: tuple[dict[str, Any], ...]) -> None:
	"""Reject any started stage whose predecessor did not pass."""
	for index, stage in enumerate(statuses):
		if stage["status"] == "pending":
			continue
		if any(previous["status"] != "passed" for previous in statuses[:index]):
			raise RuntimeError(f"{stage['name']} started before prior stage passed")


def _tmux_exists(name: str) -> bool:
	return subprocess.run(
		["tmux", "has-session", "-t", name],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		check=False,
	).returncode == 0


def _git_commit(project_root: Path) -> str:
	return subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	).stdout.strip()


def _gpu_names() -> tuple[str, ...]:
	output = subprocess.run(
		["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
		check=True,
		capture_output=True,
		text=True,
	).stdout
	return tuple(line.strip() for line in output.splitlines() if line.strip())


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
	"""Collect one read-only queue snapshot and fail on path or commit drift."""
	commit = _git_commit(args.project_root)
	if commit != args.expected_commit:
		raise RuntimeError(f"Project commit {commit} does not match {args.expected_commit}")
	stages = expected_stage_paths(
		output_root=args.output_root,
		control_output_root=args.control_output_root,
		existing_coco_control_run_root=args.existing_coco_control_run_root,
	)
	statuses = read_stage_statuses(stages)
	validate_stage_order(statuses)
	queue_status_path = args.output_root / "status.json"
	queue_status = (
		json.loads(queue_status_path.read_text(encoding="utf-8"))
		if queue_status_path.is_file()
		else {"status": "pending"}
	)
	return {
		"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
		"hostname": socket.gethostname(),
		"git_commit": commit,
		"queue_tmux": args.queue_tmux,
		"queue_tmux_alive": _tmux_exists(args.queue_tmux),
		"queue_status": queue_status,
		"gpu_names": _gpu_names(),
		"stages": statuses,
	}


def _append_json_line(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, sort_keys=True) + "\n")


def run_monitor(args: argparse.Namespace) -> None:
	"""Check immediately and then once per hour until the queue passes or fails."""
	while True:
		try:
			snapshot = collect_snapshot(args)
		except Exception as error:
			_append_json_line(
				args.log_path,
				{
					"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
					"monitor_status": "failed",
					"error": repr(error),
				},
			)
			raise
		_append_json_line(args.log_path, snapshot)
		queue_status = str(snapshot["queue_status"].get("status", "unknown"))
		if queue_status in {"passed", "failed", "interrupted"}:
			return
		if not snapshot["queue_tmux_alive"]:
			raise RuntimeError("Queue tmux exited before a terminal queue status")
		time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--project-root", type=Path, required=True)
	parser.add_argument("--expected-commit", required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--control-output-root", type=Path, required=True)
	parser.add_argument("--existing-coco-control-run-root", type=Path)
	parser.add_argument("--queue-tmux", required=True)
	parser.add_argument("--log-path", type=Path, required=True)
	parser.add_argument("--poll-seconds", type=float, default=3600.0)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_monitor(args)
		return 0
	except Exception as error:
		print(f"query recurrent monitor failed: {error!r}", flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
