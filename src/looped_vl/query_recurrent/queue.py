"""Run the focused COCO v2 recurrent repair controls serially."""

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
from looped_vl.candidate_bank import CANDIDATE_BANK_SPECS, sha256_file


@dataclass(frozen=True)
class QueryRecurrentRun:
	"""One full-data train/test experiment that isolates a single design decision."""

	name: str
	dataset: str
	num_slots: int
	max_recurrent_steps: int
	exit_mode: str
	history_layers: tuple[int, ...] = (7, 14, 21, 28)

	def validate(self) -> None:
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported dataset: {self.dataset}")
		if self.num_slots not in (1, 4, 8):
			raise ValueError("Formal slot count must be 1, 4, or 8")
		if self.max_recurrent_steps not in (1, 4):
			raise ValueError("Formal recurrent steps must be 1 or 4")
		if self.exit_mode not in ("fixed", "dynamic"):
			raise ValueError("Exit mode must be fixed or dynamic")
		if self.exit_mode == "dynamic" and self.max_recurrent_steps == 1:
			raise ValueError("Dynamic exit requires four recurrent steps")


FORMAL_QUERY_RECURRENT_RUNS = (
	QueryRecurrentRun("coco_v2_k8_r1_fixed", "coco", 8, 1, "fixed"),
	QueryRecurrentRun("coco_v2_k8_r4_fixed", "coco", 8, 4, "fixed"),
)


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _queue_manifests_match(existing: dict[str, Any], current: dict[str, Any]) -> bool:
	"""Compare queue manifests after the same tuple-to-list JSON conversion as disk."""
	json_compatible_current = json.loads(json.dumps(current, sort_keys=True))
	return existing == json_compatible_current


def validate_all_candidate_banks(candidate_root: Path) -> dict[str, str]:
	"""Require all eight immutable banks before any recurrent GPU process starts."""
	identities = {}
	for spec in CANDIDATE_BANK_SPECS:
		bank_root = candidate_root / spec.relative_path
		manifest_path = bank_root / "bank_manifest.json"
		ready_path = bank_root / "READY"
		if not manifest_path.is_file() or not ready_path.is_file():
			raise FileNotFoundError(f"Candidate bank is not ready: {spec.key}")
		manifest_hash = sha256_file(manifest_path)
		if ready_path.read_text(encoding="utf-8").strip() != manifest_hash:
			raise ValueError(f"Candidate bank READY checksum mismatch: {spec.key}")
		identities[spec.key] = manifest_hash
	return identities


def build_training_command(
	run: QueryRecurrentRun,
	*,
	args: argparse.Namespace,
	output_dir: Path,
	resume_checkpoint: Path | None = None,
	smoke: bool = False,
) -> list[str]:
	"""Build an eight-rank no-LoRA command for one immutable run identity."""
	run.validate()
	command = [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={args.world_size}",
		"-m",
		"looped_vl.query_recurrent.train",
		"--dataset",
		run.dataset,
		"--dataset-root",
		str(Path(args.dataset_root) / run.dataset),
		"--candidate-root",
		str(args.candidate_root),
		"--model-root",
		str(args.model_root),
		"--project-root",
		str(args.project_root),
		"--output-dir",
		str(output_dir),
		"--expected-world-size",
		str(args.world_size),
		"--per-device-batch-size",
		str(args.per_device_batch_size),
		"--num-workers",
		str(args.num_workers),
		"--epochs",
		"1",
		"--checkpoint-every",
		str(args.checkpoint_every),
		"--max-checkpoints",
		"1",
		"--num-slots",
		str(run.num_slots),
		"--max-recurrent-steps",
		str(run.max_recurrent_steps),
		"--exit-mode",
		run.exit_mode,
		"--history-layers",
		",".join(str(layer) for layer in run.history_layers),
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
		"--hard-negative-count",
		"32",
		"--direct-pass-loss-weight",
		"1.0",
		"--progressive-loss-weight",
		"0.1",
		"--progressive-margin",
		"0.02",
	]
	if resume_checkpoint is not None:
		command.extend(["--resume-checkpoint", str(resume_checkpoint)])
		resume_source_git_commit = getattr(args, "resume_source_git_commit", None)
		resume_gradient_scale = getattr(args, "resume_gradient_scale", None)
		if resume_source_git_commit is not None:
			command.extend(
				["--resume-source-git-commit", str(resume_source_git_commit)],
			)
		if resume_gradient_scale is not None:
			command.extend(["--resume-gradient-scale", str(resume_gradient_scale)])
	if smoke:
		command.extend(
			[
				"--max-train-rows",
				str(args.smoke_rows),
				"--max-optimizer-steps",
				str(args.smoke_steps),
				"--skip-checkpoint-save",
				"--skip-final-save",
			],
		)
	return command


def build_evaluation_command(
	run: QueryRecurrentRun,
	*,
	args: argparse.Namespace,
	training_output: Path,
	evaluation_output: Path,
) -> list[str]:
	"""Build the full held-out test that reports every recurrent pass."""
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={args.world_size}",
		"-m",
		"looped_vl.query_recurrent.evaluate",
		"--dataset",
		run.dataset,
		"--dataset-root",
		str(Path(args.dataset_root) / run.dataset),
		"--candidate-root",
		str(args.candidate_root),
		"--model-root",
		str(args.model_root),
		"--recurrent-checkpoint",
		str(training_output / "query_recurrent_model.pt"),
		"--output-dir",
		str(evaluation_output),
		"--expected-world-size",
		str(args.world_size),
		"--batch-size",
		str(args.evaluation_batch_size),
		"--num-workers",
		str(args.num_workers),
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
	]


def _passed(path: Path) -> bool:
	if not path.is_file():
		return False
	return json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"


def _resume_checkpoint(training_output: Path) -> Path | None:
	pointer = training_output / "latest_checkpoint.json"
	if not pointer.is_file():
		return None
	checkpoint = Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])
	if not checkpoint.is_file():
		raise FileNotFoundError(f"Latest checkpoint pointer is stale: {checkpoint}")
	return checkpoint


def _next_evaluation_output(run_root: Path) -> Path:
	primary = run_root / "test"
	if not primary.exists():
		return primary
	if _passed(primary / "status.json"):
		return primary
	for attempt in range(1, 100):
		candidate = run_root / f"test_retry_{attempt:02d}"
		if not candidate.exists():
			return candidate
	raise RuntimeError(f"Too many failed evaluation attempts under {run_root}")


def _child_process_environment(args: argparse.Namespace) -> dict[str, str]:
	"""Build the stable environment inherited by every distributed child process."""
	environment = os.environ.copy()
	environment.update(
		{
			"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
			"CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(args.world_size)),
			"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
			"TOKENIZERS_PARALLELISM": "false",
			"PYTHONPATH": str(Path(args.project_root) / "src"),
		},
	)
	return environment


def _run(command: list[str], *, args: argparse.Namespace) -> None:
	environment = _child_process_environment(args)
	result = subprocess.run(
		command,
		cwd=args.project_root,
		env=environment,
		check=False,
	)
	if result.returncode:
		raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def run_queue(args: argparse.Namespace) -> None:
	if args.world_size != 8 or args.per_device_batch_size != 32:
		raise ValueError("Formal queue is locked to 8 GPUs and batch 32 per GPU")
	for run in FORMAL_QUERY_RECURRENT_RUNS:
		run.validate()
	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	bank_identities = validate_all_candidate_banks(Path(args.candidate_root))
	manifest_path = output_root / "queue_manifest.json"
	queue_manifest = {
		"runs": [asdict(run) for run in FORMAL_QUERY_RECURRENT_RUNS],
		"world_size": args.world_size,
		"per_device_batch_size": args.per_device_batch_size,
		"contrastive_global_batch_size": args.world_size * args.per_device_batch_size,
		"epochs": 1,
		"validation_used": False,
		"candidate_bank_manifest_sha256": bank_identities,
		"project_root": str(args.project_root),
		"dataset_root": str(args.dataset_root),
		"model_root": str(args.model_root),
		"candidate_root": str(args.candidate_root),
	}
	if manifest_path.exists():
		existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		if not _queue_manifests_match(existing_manifest, queue_manifest):
			raise ValueError("Existing queue manifest does not match this formal queue")
	else:
		_write_json(manifest_path, queue_manifest)
	status_path = output_root / "status.json"

	smoke_output = output_root / "smoke_coco_k8_r4_dynamic"
	if not _passed(smoke_output / "status.json"):
		if smoke_output.exists():
			raise FileExistsError(f"Failed smoke output requires diagnosis: {smoke_output}")
		_write_json(status_path, {"status": "smoke", "run": "coco_k8_r4_dynamic"})
		_run(
			build_training_command(
				FORMAL_QUERY_RECURRENT_RUNS[2],
				args=args,
				output_dir=smoke_output,
				smoke=True,
			),
			args=args,
		)
	for index, run in enumerate(FORMAL_QUERY_RECURRENT_RUNS):
		run_root = output_root / run.name
		training_output = run_root / "train"
		if not _passed(training_output / "status.json"):
			resume_checkpoint = (
				_resume_checkpoint(training_output) if training_output.exists() else None
			)
			if training_output.exists() and resume_checkpoint is None:
				raise FileExistsError(
					f"Incomplete training has no resumable checkpoint: {training_output}",
				)
			command = build_training_command(
				run,
				args=args,
				output_dir=training_output,
				resume_checkpoint=resume_checkpoint,
			)
			_write_json(
				status_path,
				{"status": "training", "run_index": index, "run": run.name, "command": command},
			)
			_run(command, args=args)
		if not (training_output / "query_recurrent_model.pt").is_file():
			raise FileNotFoundError(f"Missing final recurrent model: {training_output}")
		evaluation_output = _next_evaluation_output(run_root)
		if not _passed(evaluation_output / "status.json"):
			command = build_evaluation_command(
				run,
				args=args,
				training_output=training_output,
				evaluation_output=evaluation_output,
			)
			_write_json(
				status_path,
				{"status": "testing", "run_index": index, "run": run.name, "command": command},
			)
			_run(command, args=args)
		_write_json(
			run_root / "latest_test.json",
			{"path": str(evaluation_output), "report": str(evaluation_output / "report.json")},
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
		default=Path("/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument(
		"--candidate-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/looped_vl_candidate_banks_v1_13442b1"),
	)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--world-size", type=int, default=8)
	parser.add_argument("--per-device-batch-size", type=int, default=32)
	parser.add_argument("--evaluation-batch-size", type=int, default=32)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--smoke-rows", type=int, default=512)
	parser.add_argument("--smoke-steps", type=int, default=2)
	parser.add_argument("--resume-source-git-commit")
	parser.add_argument("--resume-gradient-scale", type=float)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_queue(args)
		return 0
	except KeyboardInterrupt:
		status = "interrupted"
		code = 130
	except Exception as error:
		print(f"query recurrent queue failed: {error!r}", file=sys.stderr, flush=True)
		status = "failed"
		code = 1
	output_root = Path(args.output_root)
	if output_root.exists():
		_write_json(output_root / "status.json", {"status": status})
	return code


if __name__ == "__main__":
	raise SystemExit(main())
