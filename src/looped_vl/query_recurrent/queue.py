"""Run the COCO parallel-world recurrent experiment before three LoRA controls."""

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
from looped_vl.baseline.model import BASELINE_LORA_LAST_FOUR_DECODER_LAYERS
from looped_vl.candidate_bank import CANDIDATE_BANK_SPECS, sha256_file


@dataclass(frozen=True)
class QueryRecurrentRun:
	"""One full-data fixed-recurrence experiment."""

	name: str
	dataset: str
	num_worlds: int
	max_recurrent_steps: int
	perturbation_scale: float = 0.02

	def validate(self) -> None:
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported dataset: {self.dataset}")
		if self.num_worlds not in (1, 2, 4):
			raise ValueError("Formal world count must be 1, 2, or 4")
		if self.max_recurrent_steps not in (1, 2, 3, 4):
			raise ValueError("Formal recurrent steps must be 1, 2, 3, or 4")
		if not 0 < self.perturbation_scale < 1:
			raise ValueError("Perturbation scale must be in (0, 1)")


@dataclass(frozen=True)
class QueryOnlyLoraRun:
	"""One last-four-layer query-only LoRA control against fixed candidates."""

	name: str
	dataset: str

	def validate(self) -> None:
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported dataset: {self.dataset}")


FORMAL_QUERY_RECURRENT_RUNS = (
	QueryRecurrentRun("coco_v11_p4_r4_final_mean", "coco", 4, 4),
)
QUERY_ONLY_LORA_RUNS = (
	QueryOnlyLoraRun("coco_query_only_last4_lora_frozen_candidates", "coco"),
	QueryOnlyLoraRun(
		"gqa_balanced_query_only_last4_lora_frozen_candidates",
		"gqa_balanced",
	),
	QueryOnlyLoraRun("clevr_query_only_last4_lora_frozen_candidates", "clevr"),
)


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _queue_manifests_match(existing: dict[str, Any], current: dict[str, Any]) -> bool:
	"""Compare queue manifests after the same tuple-to-list JSON conversion as disk."""
	json_compatible_current = json.loads(json.dumps(current, sort_keys=True))
	return existing == json_compatible_current


def validate_all_candidate_banks(candidate_root: Path) -> dict[str, str]:
	"""Require all eight immutable banks before any GPU process starts."""
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
	"""Build one eight-rank no-LoRA parallel-world training command."""
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
		"--num-worlds",
		str(run.num_worlds),
		"--max-recurrent-steps",
		str(run.max_recurrent_steps),
		"--perturbation-scale",
		str(run.perturbation_scale),
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
		"--hard-negative-count",
		"32",
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
	"""Build the full held-out test that reports Pass 0 through Pass R."""
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


def build_query_only_lora_training_command(
	run: QueryOnlyLoraRun,
	*,
	args: argparse.Namespace,
	output_dir: Path,
	resume_checkpoint: Path | None = None,
) -> list[str]:
	"""Build a last-four-layer LoRA command using the matching fixed gallery."""
	run.validate()
	command = [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={args.world_size}",
		"-m",
		"looped_vl.baseline.train",
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
		"--expected-contrastive-global-batch-size",
		str(args.world_size * args.per_device_batch_size),
		"--num-workers",
		str(args.num_workers),
		"--epochs",
		"1",
		"--checkpoint-every",
		str(args.checkpoint_every),
		"--max-checkpoints",
		"1",
		"--lora-decoder-layer-indices",
		",".join(str(index) for index in BASELINE_LORA_LAST_FOUR_DECODER_LAYERS),
		"--hard-negative-count",
		"32",
		"--visual-length-buckets",
		"3",
		"--min-visual-bucket-size",
		"8",
	]
	if resume_checkpoint is not None:
		command.extend(["--resume-checkpoint", str(resume_checkpoint)])
		resume_source_git_commit = getattr(args, "resume_source_git_commit", None)
		if resume_source_git_commit is not None:
			command.extend(
				["--resume-source-git-commit", str(resume_source_git_commit)],
			)
	return command


def build_query_only_lora_evaluation_command(
	run: QueryOnlyLoraRun,
	*,
	args: argparse.Namespace,
	training_output: Path,
	evaluation_output: Path,
) -> list[str]:
	"""Evaluate LoRA queries against the same immutable test candidates."""
	run.validate()
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={args.world_size}",
		"-m",
		"looped_vl.baseline.evaluate",
		"--dataset",
		run.dataset,
		"--dataset-root",
		str(Path(args.dataset_root) / run.dataset),
		"--candidate-root",
		str(args.candidate_root),
		"--model-root",
		str(args.model_root),
		"--adapter-root",
		str(training_output / "adapter"),
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
	if not primary.exists() or _passed(primary / "status.json"):
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
	result = subprocess.run(
		command,
		cwd=args.project_root,
		env=_child_process_environment(args),
		check=False,
	)
	if result.returncode:
		raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def _require_resumable_or_fresh(training_output: Path) -> Path | None:
	if not training_output.exists():
		return None
	checkpoint = _resume_checkpoint(training_output)
	if checkpoint is None:
		raise FileExistsError(
			f"Incomplete training has no resumable checkpoint: {training_output}",
		)
	return checkpoint


def _run_recurrent(args: argparse.Namespace, status_path: Path) -> None:
	run = FORMAL_QUERY_RECURRENT_RUNS[0]
	run_root = Path(args.output_root) / run.name
	training_output = run_root / "train"
	if not _passed(training_output / "status.json"):
		resume_checkpoint = _require_resumable_or_fresh(training_output)
		command = build_training_command(
			run,
			args=args,
			output_dir=training_output,
			resume_checkpoint=resume_checkpoint,
		)
		_write_json(status_path, {"status": "training", "run": run.name, "command": command})
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
		_write_json(status_path, {"status": "testing", "run": run.name, "command": command})
		_run(command, args=args)
	_write_json(
		run_root / "latest_test.json",
		{"path": str(evaluation_output), "report": str(evaluation_output / "report.json")},
	)


def _run_lora_control(
	run: QueryOnlyLoraRun,
	*,
	args: argparse.Namespace,
	status_path: Path,
) -> None:
	run_root = _lora_control_run_root(run, args=args)
	training_output = run_root / "train"
	if not _passed(training_output / "status.json"):
		resume_checkpoint = _require_resumable_or_fresh(training_output)
		command = build_query_only_lora_training_command(
			run,
			args=args,
			output_dir=training_output,
			resume_checkpoint=resume_checkpoint,
		)
		_write_json(status_path, {"status": "training", "run": run.name, "command": command})
		_run(command, args=args)
	if not (training_output / "adapter" / "adapter_model.safetensors").is_file():
		raise FileNotFoundError(f"Missing final query-only LoRA adapter: {training_output}")
	evaluation_output = _next_evaluation_output(run_root)
	if not _passed(evaluation_output / "status.json"):
		command = build_query_only_lora_evaluation_command(
			run,
			args=args,
			training_output=training_output,
			evaluation_output=evaluation_output,
		)
		_write_json(status_path, {"status": "testing", "run": run.name, "command": command})
		_run(command, args=args)
	_write_json(
		run_root / "latest_test.json",
		{"path": str(evaluation_output), "report": str(evaluation_output / "report.json")},
	)


def _lora_control_run_root(
	run: QueryOnlyLoraRun,
	*,
	args: argparse.Namespace,
) -> Path:
	"""Resolve an explicitly recorded legacy COCO control without guessing paths."""
	existing_coco_root = getattr(args, "existing_coco_control_run_root", None)
	if run.dataset == "coco" and existing_coco_root is not None:
		return Path(existing_coco_root)
	return Path(args.control_output_root) / run.name


def run_queue(args: argparse.Namespace) -> None:
	"""Smoke, run full COCO recurrence, then resume/run all three LoRA controls."""
	if args.world_size != 8 or args.per_device_batch_size != 32:
		raise ValueError("Formal queue is locked to 8 GPUs and batch 32 per GPU")
	for run in (*FORMAL_QUERY_RECURRENT_RUNS, *QUERY_ONLY_LORA_RUNS):
		run.validate()
	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	Path(args.control_output_root).mkdir(parents=True, exist_ok=True)
	bank_identities = validate_all_candidate_banks(Path(args.candidate_root))
	manifest_path = output_root / "queue_manifest.json"
	sequence = (
		"smoke_coco_v11_p4_r4_final_mean",
		FORMAL_QUERY_RECURRENT_RUNS[0].name,
		*(run.name for run in QUERY_ONLY_LORA_RUNS),
	)
	queue_manifest = {
		"sequence": sequence,
		"recurrent_runs": [asdict(run) for run in FORMAL_QUERY_RECURRENT_RUNS],
		"query_only_lora_controls": [asdict(run) for run in QUERY_ONLY_LORA_RUNS],
		"lora_decoder_layer_indices": BASELINE_LORA_LAST_FOUR_DECODER_LAYERS,
		"world_size": args.world_size,
		"per_device_batch_size": args.per_device_batch_size,
		"contrastive_global_batch_size": args.world_size * args.per_device_batch_size,
		"epochs": 1,
		"validation_used": False,
		"candidate_qwen_forward_calls": 0,
		"candidate_bank_manifest_sha256": bank_identities,
		"project_root": str(args.project_root),
		"dataset_root": str(args.dataset_root),
		"model_root": str(args.model_root),
		"candidate_root": str(args.candidate_root),
		"control_output_root": str(args.control_output_root),
		"existing_coco_control_run_root": (
			str(args.existing_coco_control_run_root)
			if args.existing_coco_control_run_root is not None
			else None
		),
	}
	if manifest_path.exists():
		existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		if not _queue_manifests_match(existing_manifest, queue_manifest):
			raise ValueError("Existing queue manifest does not match this formal queue")
	else:
		_write_json(manifest_path, queue_manifest)
	status_path = output_root / "status.json"
	smoke_output = output_root / "smoke_coco_v11_p4_r4_final_mean"
	if not _passed(smoke_output / "status.json"):
		if smoke_output.exists():
			raise FileExistsError(f"Failed smoke output requires diagnosis: {smoke_output}")
		command = build_training_command(
			FORMAL_QUERY_RECURRENT_RUNS[0],
			args=args,
			output_dir=smoke_output,
			smoke=True,
		)
		_write_json(status_path, {"status": "smoke", "run": sequence[0], "command": command})
		_run(command, args=args)
	_run_recurrent(args, status_path)
	for run in QUERY_ONLY_LORA_RUNS:
		_run_lora_control(run, args=args, status_path=status_path)
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
	parser.add_argument("--control-output-root", type=Path, required=True)
	parser.add_argument("--existing-coco-control-run-root", type=Path)
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
