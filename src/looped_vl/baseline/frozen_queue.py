"""Directly test frozen Qwen3-VL-Embedding-2B on all three held-out splits."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from looped_vl.baseline.data import BASELINE_DATASETS
from looped_vl.baseline.queue import build_frozen_evaluation_command
from looped_vl.training.wait_and_launch import query_gpu_snapshot


@dataclass(frozen=True)
class FrozenEvaluationRun:
	"""Per-dataset inference settings for the frozen base-model queue."""

	dataset: str
	batch_size: int
	num_workers: int

	def validate(self) -> None:
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported baseline dataset: {self.dataset}")
		if self.batch_size <= 0:
			raise ValueError("batch_size must be positive")
		if self.num_workers <= 0:
			raise ValueError("num_workers must be positive")


DEFAULT_FROZEN_RUNS = tuple(
	FrozenEvaluationRun(dataset, batch_size=32, num_workers=4)
	for dataset in BASELINE_DATASETS
)


def parse_v100_names(gpu_output: str, *, world_size: int) -> tuple[str, ...]:
	"""Validate the exact physical GPU indexes and V100 model names."""
	rows = []
	for line in gpu_output.splitlines():
		if not line.strip():
			continue
		index_text, separator, name = line.partition(",")
		if not separator:
			raise ValueError(f"Unexpected nvidia-smi GPU row: {line}")
		rows.append((int(index_text.strip()), name.strip()))
	expected_indexes = tuple(range(world_size))
	if tuple(index for index, _name in rows) != expected_indexes:
		raise RuntimeError(f"Expected physical GPUs {expected_indexes}, found {rows}")
	if any("V100" not in name for _index, name in rows):
		raise RuntimeError(f"Frozen 8XV100 queue requires V100 GPUs, found {rows}")
	return tuple(name for _index, name in rows)


def build_frozen_queue_commands(
	runs: tuple[FrozenEvaluationRun, ...],
	*,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
) -> list[list[str]]:
	"""Build one serial command per dataset without changing its held-out split."""
	if tuple(run.dataset for run in runs) != BASELINE_DATASETS:
		raise ValueError(
			"Frozen queue order must be exactly COCO, GQA Balanced, then CLEVR",
		)
	for run in runs:
		run.validate()
	return [
		build_frozen_evaluation_command(
			run.dataset,
			dataset_root=dataset_root,
			model_root=model_root,
			output_root=output_root,
			world_size=world_size,
			batch_size=run.batch_size,
			num_workers=run.num_workers,
		)
		for run in runs
	]


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _query_gpu_names(world_size: int) -> tuple[str, ...]:
	result = subprocess.run(
		[
			"nvidia-smi",
			"--query-gpu=index,name",
			"--format=csv,noheader",
		],
		check=True,
		capture_output=True,
		text=True,
	)
	return parse_v100_names(result.stdout, world_size=world_size)


def _git_commit(project_root: Path) -> str:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.strip()


def run_frozen_queue(
	args: argparse.Namespace,
	runs: tuple[FrozenEvaluationRun, ...] = DEFAULT_FROZEN_RUNS,
) -> None:
	"""Require one idle snapshot, then evaluate the three datasets serially."""
	output_root = Path(args.output_root)
	if output_root.exists():
		raise FileExistsError(f"Frozen evaluation output already exists: {output_root}")
	output_root.mkdir(parents=True)
	status_path = output_root / "status.json"
	_write_json(status_path, {"status": "checking_gpus"})
	gpu_names = _query_gpu_names(args.world_size)
	snapshot = query_gpu_snapshot(tuple(range(args.world_size)))
	if not snapshot.is_idle:
		raise RuntimeError("All eight V100 GPUs must be idle at submission time")
	commands = build_frozen_queue_commands(
		runs,
		dataset_root=Path(args.dataset_root),
		model_root=Path(args.model_root),
		output_root=output_root,
		world_size=args.world_size,
	)
	_write_json(
		output_root / "queue_manifest.json",
		{
			"scope": "frozen_qwen3_vl_embedding_2b_three_dataset_test",
			"hostname": socket.gethostname(),
			"gpu_names": gpu_names,
			"world_size": args.world_size,
			"code_commit": _git_commit(Path(args.project_root)),
			"project_root": str(args.project_root),
			"dataset_root": str(args.dataset_root),
			"model_root": str(args.model_root),
			"runs": [asdict(run) for run in runs],
			"commands": commands,
			"validation_used": False,
		},
	)
	environment = os.environ.copy()
	environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environment["CUDA_VISIBLE_DEVICES"] = ",".join(
		str(index) for index in range(args.world_size)
	)
	environment["PYTHONPATH"] = str(Path(args.project_root) / "src")
	for run, command in zip(runs, commands, strict=True):
		_write_json(
			status_path,
			{"status": "testing", "dataset": run.dataset, "command": command},
		)
		result = subprocess.run(
			command,
			cwd=args.project_root,
			env=environment,
			check=False,
		)
		if result.returncode:
			raise RuntimeError(
				f"{run.dataset} frozen test failed with exit code {result.returncode}",
			)
	_write_json(
		status_path,
		{"status": "passed", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/loopedTransformer"),
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/looped_vl_single_baselines_v1"),
	)
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path(
			"/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original",
		),
	)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--world-size", type=int, choices=(8,), default=8)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_frozen_queue(args)
		return 0
	except KeyboardInterrupt:
		output_root = Path(args.output_root)
		if output_root.exists():
			_write_json(output_root / "status.json", {"status": "interrupted"})
		return 130
	except Exception as error:
		output_root = Path(args.output_root)
		if output_root.exists():
			_write_json(
				output_root / "status.json",
				{"status": "failed", "error": repr(error)},
			)
		print(f"frozen evaluation queue failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
