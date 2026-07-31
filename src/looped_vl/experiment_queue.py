"""Run three baseline and three recurrent train/test experiments serially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from looped_vl.baseline.data import BASELINE_DATASETS

ExperimentFamily = Literal["baseline", "recurrent"]


@dataclass(frozen=True)
class ExperimentSpec:
	"""One source-pure experiment with independently chosen train and test batches."""

	family: ExperimentFamily
	dataset: str
	train_batch_size: int
	train_workers: int
	evaluation_batch_size: int
	evaluation_workers: int
	gradient_checkpointing: bool = True

	def validate(self, world_size: int) -> None:
		"""Reject settings that would change either fixed training protocol."""
		if self.family not in {"baseline", "recurrent"}:
			raise ValueError(f"Unsupported experiment family: {self.family}")
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported dataset: {self.dataset}")
		if world_size <= 0:
			raise ValueError("world_size must be positive")
		for name, value in (
			("train_batch_size", self.train_batch_size),
			("train_workers", self.train_workers),
			("evaluation_batch_size", self.evaluation_batch_size),
			("evaluation_workers", self.evaluation_workers),
		):
			if value <= 0:
				raise ValueError(f"{name} must be positive")
		contrastive_batch = self.train_batch_size * world_size
		if self.family == "baseline" and contrastive_batch != 256:
			raise ValueError("Baseline training requires a true 256-pair negative pool")
		if self.family == "recurrent" and 512 % contrastive_batch:
			raise ValueError("Recurrent global microbatch must divide optimizer batch 512")


def default_experiments() -> list[ExperimentSpec]:
	"""Return the fixed six-run order without any validation stage."""
	return [
		ExperimentSpec("baseline", dataset, 32, 4, 32, 4)
		for dataset in BASELINE_DATASETS
	] + [
		ExperimentSpec("recurrent", "coco", 8, 4, 8, 4, False),
		ExperimentSpec("recurrent", "gqa_balanced", 8, 4, 8, 4, True),
		ExperimentSpec("recurrent", "clevr", 8, 4, 8, 4, False),
	]


def _torchrun_prefix(world_size: int) -> list[str]:
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={world_size}",
	]


def _experiment_root(output_root: Path, spec: ExperimentSpec) -> Path:
	return output_root / spec.family / spec.dataset


def build_training_command(
	spec: ExperimentSpec,
	*,
	project_root: Path,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
	code_commit: str,
	checkpoint_every: int,
	max_checkpoints: int,
	resume_checkpoint: Path | None,
) -> list[str]:
	"""Build one full one-epoch command retaining only the latest checkpoint."""
	spec.validate(world_size)
	if checkpoint_every <= 0:
		raise ValueError("checkpoint_every must be positive")
	if max_checkpoints != 1:
		raise ValueError("max_checkpoints must be exactly one")
	train_output = _experiment_root(output_root, spec) / "train"
	command = _torchrun_prefix(world_size)
	if spec.family == "baseline":
		command.extend(
			[
				"-m",
				"looped_vl.baseline.train",
				"--dataset",
				spec.dataset,
				"--dataset-root",
				str(dataset_root / spec.dataset),
				"--model-root",
				str(model_root),
				"--project-root",
				str(project_root),
				"--output-dir",
				str(train_output),
				"--expected-world-size",
				str(world_size),
				"--per-device-batch-size",
				str(spec.train_batch_size),
				"--gradient-accumulation-steps",
				"1",
				"--expected-contrastive-global-batch-size",
				str(spec.train_batch_size * world_size),
				"--num-workers",
				str(spec.train_workers),
				"--epochs",
				"1",
				"--checkpoint-every",
				str(checkpoint_every),
				"--max-checkpoints",
				str(max_checkpoints),
				"--initial-gradient-scale",
				"4096",
				"--attention-implementation",
				"sdpa",
				(
					"--gradient-checkpointing"
					if spec.gradient_checkpointing
					else "--no-gradient-checkpointing"
				),
			],
		)
	else:
		command.extend(
			[
				"-m",
				"looped_vl.training.train",
				"--dataset-root",
				str(dataset_root / spec.dataset),
				"--model-root",
				str(model_root),
				"--project-root",
				str(project_root),
				"--code-commit",
				code_commit,
				"--model-config",
				str(project_root / "configs" / "base.yaml"),
				"--training-config",
				str(project_root / "configs" / "train.yaml"),
				"--master-slot-path",
				str(project_root / "artifacts" / "master_slot_init_seed42.pt"),
				"--output-dir",
				str(train_output),
				"--expected-world-size",
				str(world_size),
				"--per-device-batch-size",
				str(spec.train_batch_size),
				"--expected-contrastive-global-batch-size",
				str(spec.train_batch_size * world_size),
				"--num-workers",
				str(spec.train_workers),
				"--checkpoint-every",
				str(checkpoint_every),
				"--max-checkpoints",
				str(max_checkpoints),
				"--runtime-precision",
				"fp16",
				"--initial-gradient-scale",
				"32",
				"--attention-implementation",
				"auto",
				(
					"--gradient-checkpointing"
					if spec.gradient_checkpointing
					else "--no-gradient-checkpointing"
				),
			],
		)
	if resume_checkpoint is not None:
		command.extend(["--resume-checkpoint", str(resume_checkpoint)])
	return command


def build_evaluation_command(
	spec: ExperimentSpec,
	*,
	project_root: Path,
	dataset_root: Path,
	model_root: Path,
	output_root: Path,
	world_size: int,
	checkpoint: Path | None = None,
) -> list[str]:
	"""Build the matching full held-out test command, never a validation command."""
	spec.validate(world_size)
	experiment_root = _experiment_root(output_root, spec)
	command = _torchrun_prefix(world_size)
	if spec.family == "baseline":
		command.extend(
			[
				"-m",
				"looped_vl.baseline.evaluate",
				"--dataset",
				spec.dataset,
				"--dataset-root",
				str(dataset_root / spec.dataset),
				"--model-root",
				str(model_root),
				"--adapter-root",
				str(experiment_root / "train" / "adapter"),
				"--output-dir",
				str(experiment_root / "test"),
				"--expected-world-size",
				str(world_size),
				"--batch-size",
				str(spec.evaluation_batch_size),
				"--num-workers",
				str(spec.evaluation_workers),
				"--attention-implementation",
				"sdpa",
			],
		)
	else:
		resolved_checkpoint = (
			checkpoint
			if checkpoint is not None
			else experiment_root / "train" / "checkpoints" / "latest.pt"
		)
		command.extend(
			[
				"-m",
				"looped_vl.evaluate_recurrent",
				"--source",
				spec.dataset,
				"--dataset-root",
				str(dataset_root / spec.dataset),
				"--model-root",
				str(model_root),
				"--master-slot-path",
				str(project_root / "artifacts" / "master_slot_init_seed42.pt"),
				"--model-config",
				str(project_root / "configs" / "base.yaml"),
				"--checkpoint",
				str(resolved_checkpoint),
				"--output-dir",
				str(experiment_root / "test"),
				"--split",
				"test",
				"--expected-world-size",
				str(world_size),
				"--batch-size",
				str(spec.evaluation_batch_size),
				"--num-workers",
				str(spec.evaluation_workers),
				"--runtime-precision",
				"fp16",
				"--attention-implementation",
				"auto",
			],
		)
	return command


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
	value = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(value, dict):
		raise ValueError(f"Expected a JSON mapping: {path}")
	return value


def _status_passed(output_dir: Path) -> bool:
	status_path = output_dir / "status.json"
	return status_path.is_file() and _read_json(status_path).get("status") == "passed"


def _latest_checkpoint(train_output: Path) -> Path:
	latest_path = train_output / "latest_checkpoint.json"
	if not latest_path.is_file():
		raise FileNotFoundError(f"No resumable checkpoint under {train_output}")
	checkpoint = Path(str(_read_json(latest_path).get("path", "")))
	if not checkpoint.resolve().is_relative_to(train_output.resolve()):
		raise ValueError("Latest checkpoint does not belong to its training output")
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
		raise RuntimeError(
			f"Command failed with exit code {result.returncode}: {' '.join(command)}",
		)


def _resolve_git_commit(project_root: Path, requested: str | None) -> str:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	)
	actual = result.stdout.strip()
	if requested is not None and requested != actual:
		raise ValueError(f"Requested commit {requested} does not match checkout {actual}")
	return actual


def run_queue(args: argparse.Namespace) -> None:
	"""Run all six experiments serially and support explicit in-place queue resume."""
	project_root = Path(args.project_root)
	dataset_root = Path(args.dataset_root)
	model_root = Path(args.model_root)
	output_root = Path(args.output_root)
	experiments = default_experiments()
	for experiment in experiments:
		experiment.validate(args.world_size)
	code_commit = _resolve_git_commit(project_root, args.code_commit)
	manifest = {
		"training_epochs": 1,
		"validation_enabled": False,
		"test_after_training": True,
		"serial_execution": True,
		"code_commit": code_commit,
		"world_size": args.world_size,
		"checkpoint_every": args.checkpoint_every,
		"max_checkpoints": args.max_checkpoints,
		"dataset_root": str(dataset_root),
		"model_root": str(model_root),
		"experiments": [asdict(experiment) for experiment in experiments],
	}
	manifest_path = output_root / "queue_manifest.json"
	if args.resume:
		if not output_root.is_dir() or not manifest_path.is_file():
			raise FileNotFoundError(f"Cannot resume missing queue output: {output_root}")
		if _read_json(manifest_path) != manifest:
			raise ValueError("Queue resume configuration differs from the original manifest")
	else:
		if output_root.exists():
			raise FileExistsError(f"Queue output already exists: {output_root}")
		output_root.mkdir(parents=True)
		_write_json(manifest_path, manifest)
	environment = os.environ.copy()
	environment.update(
		{
			"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
			"CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(args.world_size)),
			"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
			"TOKENIZERS_PARALLELISM": "false",
		},
	)
	project_pythonpath = str(project_root / "src")
	existing_pythonpath = environment.get("PYTHONPATH")
	environment["PYTHONPATH"] = (
		f"{project_pythonpath}:{existing_pythonpath}"
		if existing_pythonpath
		else project_pythonpath
	)
	for experiment_index, experiment in enumerate(experiments):
		experiment_root = _experiment_root(output_root, experiment)
		train_output = experiment_root / "train"
		test_output = experiment_root / "test"
		name = f"{experiment.family}_{experiment.dataset}"
		resume_checkpoint = None
		if not _status_passed(train_output):
			if train_output.exists():
				resume_checkpoint = _latest_checkpoint(train_output)
			training_command = build_training_command(
				experiment,
				project_root=project_root,
				dataset_root=dataset_root,
				model_root=model_root,
				output_root=output_root,
				world_size=args.world_size,
				code_commit=code_commit,
				checkpoint_every=args.checkpoint_every,
				max_checkpoints=args.max_checkpoints,
				resume_checkpoint=resume_checkpoint,
			)
			_write_json(
				output_root / "status.json",
				{
					"status": "training",
					"experiment": name,
					"experiment_index": experiment_index,
					"command": training_command,
					"resumed_from": (
						str(resume_checkpoint) if resume_checkpoint is not None else None
					),
				},
			)
			_run_logged(
				training_command,
				project_root=project_root,
				environment=environment,
				log_path=output_root / "logs" / f"{name}_train.log",
				append=resume_checkpoint is not None,
			)
			if not _status_passed(train_output):
				raise RuntimeError(f"Training did not report passed: {name}")
		if _status_passed(test_output):
			continue
		if test_output.exists():
			raise FileExistsError(
				f"Failed or incomplete test output must be reviewed before resume: {test_output}",
			)
		checkpoint = (
			_latest_checkpoint(train_output)
			if experiment.family == "recurrent"
			else None
		)
		evaluation_command = build_evaluation_command(
			experiment,
			project_root=project_root,
			dataset_root=dataset_root,
			model_root=model_root,
			output_root=output_root,
			world_size=args.world_size,
			checkpoint=checkpoint,
		)
		_write_json(
			output_root / "status.json",
			{
				"status": "testing",
				"experiment": name,
				"experiment_index": experiment_index,
				"command": evaluation_command,
			},
		)
		_run_logged(
			evaluation_command,
			project_root=project_root,
			environment=environment,
			log_path=output_root / "logs" / f"{name}_test.log",
			append=False,
		)
		if not _status_passed(test_output):
			raise RuntimeError(f"Test did not report passed: {name}")
	_write_json(
		output_root / "status.json",
		{
			"status": "passed",
			"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
		},
	)


def parse_args() -> argparse.Namespace:
	"""Parse one immediate six-experiment serial queue."""
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
	parser.add_argument("--world-size", type=int, default=8)
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--max-checkpoints", type=int, choices=(1,), default=1)
	parser.add_argument("--resume", action="store_true")
	return parser.parse_args()


def main() -> int:
	"""Run the serial queue and preserve a machine-readable failure state."""
	args = parse_args()
	try:
		run_queue(args)
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
		print(f"six-experiment queue failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
