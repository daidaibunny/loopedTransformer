"""Wait for eight idle V100 GPUs, then train and test three LoRA baselines serially."""

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

from looped_vl.baseline.data import BASELINE_DATASETS
from looped_vl.training.wait_and_launch import wait_for_idle_window


@dataclass(frozen=True)
class BaselineRun:
	"""Dataset-specific parallel settings selected by the smoke search."""

	dataset: str
	per_device_batch_size: int
	gradient_accumulation_steps: int
	num_workers: int
	gradient_checkpointing: bool = True

	def validate(self) -> None:
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported baseline dataset: {self.dataset}")
		for name, value in (
			("per_device_batch_size", self.per_device_batch_size),
			("gradient_accumulation_steps", self.gradient_accumulation_steps),
			("num_workers", self.num_workers),
		):
			if value <= 0:
				raise ValueError(f"{name} must be positive")


def build_training_command(
	run: BaselineRun,
	*,
	project_root: Path,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
	checkpoint_every: int = 100,
	max_checkpoints: int = 1,
) -> list[str]:
	"""Build one result-isolated eight-rank full training command."""
	run.validate()
	contrastive_global_batch_size = run.per_device_batch_size * world_size
	if contrastive_global_batch_size != 256:
		raise ValueError(
			"Full baseline training requires a true contrastive global batch of 256",
		)
	if run.gradient_accumulation_steps != 1:
		raise ValueError("Full baseline training uses one 256-pair optimizer batch")
	command = [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={world_size}",
		"-m",
		"looped_vl.baseline.train",
		"--dataset",
		run.dataset,
		"--dataset-root",
		str(dataset_root / run.dataset),
		"--model-root",
		str(model_root),
		"--output-dir",
		str(output_root / run.dataset / "train"),
		"--project-root",
		str(project_root),
		"--expected-world-size",
		str(world_size),
		"--per-device-batch-size",
		str(run.per_device_batch_size),
		"--gradient-accumulation-steps",
		str(run.gradient_accumulation_steps),
		"--expected-contrastive-global-batch-size",
		str(contrastive_global_batch_size),
		"--num-workers",
		str(run.num_workers),
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
		"--checkpoint-every",
		str(checkpoint_every),
		"--max-checkpoints",
		str(max_checkpoints),
	]
	command.append(
		"--gradient-checkpointing"
		if run.gradient_checkpointing
		else "--no-gradient-checkpointing",
	)
	return command


def build_evaluation_command(
	run: BaselineRun,
	*,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
) -> list[str]:
	"""Build the matching held-out test command for one saved adapter."""
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={world_size}",
		"-m",
		"looped_vl.baseline.evaluate",
		"--dataset",
		run.dataset,
		"--dataset-root",
		str(dataset_root / run.dataset),
		"--model-root",
		str(model_root),
		"--adapter-root",
		str(output_root / run.dataset / "train" / "adapter"),
		"--output-dir",
		str(output_root / run.dataset / "test"),
		"--expected-world-size",
		str(world_size),
		"--batch-size",
		str(run.per_device_batch_size),
		"--num-workers",
		str(run.num_workers),
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
	]


def build_frozen_evaluation_command(
	dataset: str,
	*,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
	batch_size: int,
	num_workers: int,
) -> list[str]:
	"""Build one direct multi-rank test command for the untouched Qwen checkpoint."""
	if dataset not in BASELINE_DATASETS:
		raise ValueError(f"Unsupported baseline dataset: {dataset}")
	for name, value in (
		("world_size", world_size),
		("batch_size", batch_size),
		("num_workers", num_workers),
	):
		if value <= 0:
			raise ValueError(f"{name} must be positive")
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={world_size}",
		"-m",
		"looped_vl.baseline.evaluate",
		"--dataset",
		dataset,
		"--dataset-root",
		str(dataset_root / dataset),
		"--model-root",
		str(model_root),
		"--output-dir",
		str(output_root / dataset),
		"--expected-world-size",
		str(world_size),
		"--batch-size",
		str(batch_size),
		"--num-workers",
		str(num_workers),
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
	]


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_queue(args: argparse.Namespace, runs: list[BaselineRun]) -> None:
	"""Own the selected GPUs serially and stop immediately if one stage fails."""
	if {run.dataset for run in runs} != set(BASELINE_DATASETS):
		raise ValueError("Queue must contain exactly one COCO, GQA Balanced, and CLEVR run")
	output_root = Path(args.output_root)
	if output_root.exists():
		raise FileExistsError(f"Queue output already exists: {output_root}")
	output_root.mkdir(parents=True)
	_write_json(
		output_root / "queue_manifest.json",
		{
			"runs": [asdict(run) for run in runs],
			"world_size": args.world_size,
			"required_idle_seconds": args.required_idle_seconds,
			"dataset_root": str(args.dataset_root),
			"model_root": str(args.model_root),
		},
	)
	status_path = output_root / "status.json"
	environment = os.environ.copy()
	environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environment["CUDA_VISIBLE_DEVICES"] = ",".join(
		str(index) for index in range(args.world_size)
	)
	environment["PYTHONPATH"] = str(Path(args.project_root) / "src")
	for run_index, run in enumerate(runs):
		_write_json(
			status_path,
			{
				"status": "waiting_for_idle",
				"dataset": run.dataset,
				"run_index": run_index,
			},
		)
		wait_for_idle_window(
			required_seconds=args.required_idle_seconds,
			poll_seconds=args.poll_seconds,
			log_path=output_root / f"{run.dataset}_idle_gate.jsonl",
			expected_indexes=tuple(range(args.world_size)),
		)
		training_command = build_training_command(
			run,
			project_root=Path(args.project_root),
			dataset_root=Path(args.dataset_root),
			model_root=Path(args.model_root),
			output_root=output_root,
			world_size=args.world_size,
			checkpoint_every=args.checkpoint_every,
			max_checkpoints=args.max_checkpoints,
		)
		_write_json(
			status_path,
			{"status": "training", "dataset": run.dataset, "command": training_command},
		)
		training_result = subprocess.run(
			training_command,
			cwd=args.project_root,
			env=environment,
			check=False,
		)
		if training_result.returncode:
			raise RuntimeError(
				f"{run.dataset} training failed with exit code {training_result.returncode}",
			)
		evaluation_command = build_evaluation_command(
			run,
			dataset_root=Path(args.dataset_root),
			model_root=Path(args.model_root),
			output_root=output_root,
			world_size=args.world_size,
		)
		_write_json(
			status_path,
			{"status": "testing", "dataset": run.dataset, "command": evaluation_command},
		)
		evaluation_result = subprocess.run(
			evaluation_command,
			cwd=args.project_root,
			env=environment,
			check=False,
		)
		if evaluation_result.returncode:
			raise RuntimeError(
				f"{run.dataset} test failed with exit code {evaluation_result.returncode}",
			)
	_write_json(
		status_path,
		{"status": "passed", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
	)


def _parse_run(value: str) -> BaselineRun:
	parts = value.split(",")
	if len(parts) not in (4, 5):
		raise argparse.ArgumentTypeError(
			"run must be DATASET,BATCH,ACCUMULATION,WORKERS[,CHECKPOINTING]",
		)
	gradient_checkpointing = True
	if len(parts) == 5:
		if parts[4] not in {"on", "off"}:
			raise argparse.ArgumentTypeError("CHECKPOINTING must be on or off")
		gradient_checkpointing = parts[4] == "on"
	run = BaselineRun(
		dataset=parts[0],
		per_device_batch_size=int(parts[1]),
		gradient_accumulation_steps=int(parts[2]),
		num_workers=int(parts[3]),
		gradient_checkpointing=gradient_checkpointing,
	)
	run.validate()
	return run


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
	parser.add_argument("--run", action="append", type=_parse_run, required=True)
	parser.add_argument("--world-size", type=int, default=8)
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--max-checkpoints", type=int, choices=(1,), default=1)
	parser.add_argument("--required-idle-seconds", type=float, default=120.0)
	parser.add_argument("--poll-seconds", type=float, default=5.0)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_queue(args, args.run)
		return 0
	except KeyboardInterrupt:
		output_root = Path(args.output_root)
		if output_root.exists():
			_write_json(output_root / "status.json", {"status": "interrupted"})
		print("baseline queue interrupted", file=sys.stderr, flush=True)
		return 130
	except Exception as error:
		output_root = Path(args.output_root)
		if output_root.exists():
			_write_json(
				output_root / "status.json",
				{"status": "failed", "error": repr(error)},
			)
		print(f"baseline queue failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
