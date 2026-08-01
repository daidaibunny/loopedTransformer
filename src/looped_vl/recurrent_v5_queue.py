"""Run the selected no-LoRA recurrent v5 full experiments serially on eight GPUs."""

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
from looped_vl.training.trainability import MAX_RECURRENT_TRAINABLE_PARAMETERS


@dataclass(frozen=True)
class RecurrentV5Run:
	"""One full-data recurrent v5 training and held-out test pair."""

	name: str
	dataset: str
	num_latent_slots: int
	step_size: float
	train_batch_size: int = 32
	evaluation_batch_size: int = 32
	num_workers: int = 4

	def validate(self, world_size: int) -> None:
		if not self.name or self.dataset not in BASELINE_DATASETS:
			raise ValueError("Run name and supported dataset are required")
		if self.num_latent_slots not in (16, 32):
			raise ValueError("Selected v5 queue supports K=16 or K=32")
		if self.step_size not in (0.5, 1.0):
			raise ValueError("step_size must be 0.5 or 1.0")
		if world_size != 8:
			raise ValueError("Formal v5 queue requires exactly eight ranks")
		if self.train_batch_size * world_size != 256:
			raise ValueError("Formal v5 queue requires a 256-pair contrastive batch")
		if self.evaluation_batch_size <= 0 or self.num_workers <= 0:
			raise ValueError("Evaluation batch size and workers must be positive")


def default_runs() -> list[RecurrentV5Run]:
	"""Prioritize the best old K values and the reasoning-focused GQA dataset."""
	return [
		RecurrentV5Run("coco_k16_alpha1", "coco", 16, 1.0),
		RecurrentV5Run("gqa_k16_alpha1", "gqa_balanced", 16, 1.0),
		RecurrentV5Run("gqa_k32_alpha1", "gqa_balanced", 32, 1.0),
	]


def _torchrun_prefix(world_size: int) -> list[str]:
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={world_size}",
	]


def _run_root(output_root: Path, run: RecurrentV5Run) -> Path:
	return output_root / run.name


def build_training_command(
	run: RecurrentV5Run,
	*,
	project_root: Path,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
	code_commit: str,
	checkpoint_every: int,
	resume_checkpoint: Path | None,
) -> list[str]:
	"""Build one full one-epoch training command with rolling checkpoint retention."""
	run.validate(world_size)
	if checkpoint_every <= 0:
		raise ValueError("checkpoint_every must be positive")
	command = _torchrun_prefix(world_size) + [
		"-m",
		"looped_vl.training.train",
		"--dataset-root",
		str(dataset_root / run.dataset),
		"--model-root",
		str(model_root),
		"--project-root",
		str(project_root),
		"--code-commit",
		code_commit,
		"--model-config",
		str(project_root / "configs" / "slot_count_smoke.yaml"),
		"--training-config",
		str(project_root / "configs" / "train.yaml"),
		"--master-slot-path",
		str(project_root / "artifacts" / "master_slot_init_seed42_kmax64.pt"),
		"--num-latent-slots",
		str(run.num_latent_slots),
		"--recurrent-step-size",
		str(run.step_size),
		"--use-recurrent-layer-scale",
		"--output-dir",
		str(_run_root(output_root, run) / "train"),
		"--expected-world-size",
		str(world_size),
		"--per-device-batch-size",
		str(run.train_batch_size),
		"--expected-contrastive-global-batch-size",
		str(run.train_batch_size * world_size),
		"--num-workers",
		str(run.num_workers),
		"--checkpoint-every",
		str(checkpoint_every),
		"--max-checkpoints",
		"1",
		"--runtime-precision",
		"fp16",
		"--initial-gradient-scale",
		"32",
		"--attention-implementation",
		"auto",
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
		"--gradient-checkpointing",
	]
	if resume_checkpoint is not None:
		command.extend(["--resume-checkpoint", str(resume_checkpoint)])
	return command


def build_evaluation_command(
	run: RecurrentV5Run,
	*,
	project_root: Path,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
	checkpoint: Path,
) -> list[str]:
	"""Build the matching full test command with every recurrent pass reported."""
	run.validate(world_size)
	return _torchrun_prefix(world_size) + [
		"-m",
		"looped_vl.evaluate_recurrent",
		"--source",
		run.dataset,
		"--dataset-root",
		str(dataset_root / run.dataset),
		"--model-root",
		str(model_root),
		"--master-slot-path",
		str(project_root / "artifacts" / "master_slot_init_seed42_kmax64.pt"),
		"--model-config",
		str(project_root / "configs" / "slot_count_smoke.yaml"),
		"--num-latent-slots",
		str(run.num_latent_slots),
		"--recurrent-step-size",
		str(run.step_size),
		"--use-recurrent-layer-scale",
		"--checkpoint",
		str(checkpoint),
		"--output-dir",
		str(_run_root(output_root, run) / "test"),
		"--split",
		"test",
		"--expected-world-size",
		str(world_size),
		"--batch-size",
		str(run.evaluation_batch_size),
		"--num-workers",
		str(run.num_workers),
		"--runtime-precision",
		"fp16",
		"--attention-implementation",
		"auto",
	]


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
	value = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(value, dict):
		raise ValueError(f"Expected JSON mapping: {path}")
	return value


def _status_passed(output_dir: Path) -> bool:
	path = output_dir / "status.json"
	return path.is_file() and _read_json(path).get("status") == "passed"


def _latest_checkpoint(train_output: Path) -> Path:
	latest = train_output / "latest_checkpoint.json"
	if not latest.is_file():
		raise FileNotFoundError(f"No latest checkpoint record under {train_output}")
	checkpoint = Path(str(_read_json(latest).get("path", "")))
	if not checkpoint.resolve().is_relative_to(train_output.resolve()):
		raise ValueError("Latest checkpoint must belong to the training output")
	if not checkpoint.is_file():
		raise FileNotFoundError(f"Latest checkpoint is missing: {checkpoint}")
	return checkpoint


def _run_logged(
	command: list[str],
	*,
	project_root: Path,
	environment: dict[str, str],
	log_path: Path,
	append: bool,
) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("a" if append else "w", encoding="utf-8") as log_handle:
		result = subprocess.run(
			command,
			cwd=project_root,
			env=environment,
			stdout=log_handle,
			stderr=subprocess.STDOUT,
			check=False,
		)
	if result.returncode:
		raise RuntimeError(f"Command failed with exit code {result.returncode}")


def run_queue(args: argparse.Namespace) -> None:
	"""Run every selected v5 experiment serially and resume only exact checkpoints."""
	project_root = Path(args.project_root)
	output_root = Path(args.output_root)
	if output_root.exists() and not args.resume:
		raise FileExistsError(f"Queue output already exists: {output_root}")
	output_root.mkdir(parents=True, exist_ok=True)
	code_commit = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		text=True,
		capture_output=True,
		check=True,
	).stdout.strip()
	if args.code_commit is not None and args.code_commit != code_commit:
		raise ValueError(f"Requested commit {args.code_commit} != checked-out {code_commit}")
	runs = default_runs()
	_write_json(
		output_root / "queue_manifest.json",
		{
			"architecture": "recurrent_v5_no_lora",
			"code_commit": code_commit,
			"world_size": args.world_size,
			"checkpoint_every": args.checkpoint_every,
			"max_checkpoints": 1,
			"trainable_parameter_limit": MAX_RECURRENT_TRAINABLE_PARAMETERS,
			"runs": [
				{
					**asdict(run),
					"expected_trainable_parameter_count": (
						run.num_latent_slots * 2048 + 8 * 2048
					),
				}
				for run in runs
			],
		},
	)
	environment = os.environ.copy()
	environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(8))
	environment["PYTHONPATH"] = str(project_root / "src")
	for index, run in enumerate(runs):
		run_root = _run_root(output_root, run)
		train_output = run_root / "train"
		test_output = run_root / "test"
		resume_checkpoint = None
		if not _status_passed(train_output):
			if train_output.exists():
				resume_checkpoint = _latest_checkpoint(train_output)
			command = build_training_command(
				run,
				project_root=project_root,
				dataset_root=Path(args.dataset_root),
				model_root=Path(args.model_root),
				output_root=output_root,
				world_size=args.world_size,
				code_commit=code_commit,
				checkpoint_every=args.checkpoint_every,
				resume_checkpoint=resume_checkpoint,
			)
			_write_json(
				output_root / "status.json",
				{"status": "training", "index": index, "run": run.name, "command": command},
			)
			_run_logged(
				command,
				project_root=project_root,
				environment=environment,
				log_path=output_root / "logs" / f"{run.name}_train.log",
				append=resume_checkpoint is not None,
			)
		if _status_passed(test_output):
			continue
		if test_output.exists():
			raise FileExistsError(f"Incomplete test output requires review: {test_output}")
		checkpoint = _latest_checkpoint(train_output)
		command = build_evaluation_command(
			run,
			project_root=project_root,
			dataset_root=Path(args.dataset_root),
			model_root=Path(args.model_root),
			output_root=output_root,
			world_size=args.world_size,
			checkpoint=checkpoint,
		)
		_write_json(
			output_root / "status.json",
			{"status": "testing", "index": index, "run": run.name, "command": command},
		)
		_run_logged(
			command,
			project_root=project_root,
			environment=environment,
			log_path=output_root / "logs" / f"{run.name}_test.log",
			append=False,
		)
	_write_json(
		output_root / "status.json",
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
		default=Path("/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--code-commit")
	parser.add_argument("--world-size", type=int, choices=(8,), default=8)
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--resume", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_queue(args)
		return 0
	except KeyboardInterrupt:
		if Path(args.output_root).exists():
			_write_json(Path(args.output_root) / "status.json", {"status": "interrupted"})
		return 130
	except Exception as error:
		if Path(args.output_root).exists():
			_write_json(
				Path(args.output_root) / "status.json",
				{"status": "failed", "error": repr(error)},
			)
		print(f"recurrent v5 queue failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
