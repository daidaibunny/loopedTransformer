"""Run GQA and CLEVR last-four-layer LoRA against immutable candidate banks."""

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

# The V100 image combines a legacy generated ONNX module with a newer protobuf runtime.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from looped_vl.baseline.model import BASELINE_LORA_LAST_FOUR_DECODER_LAYERS
from looped_vl.candidate_bank import CandidateBankSpec, sha256_file

QUERY_ONLY_LORA_DATASETS = ("gqa_balanced", "clevr")


@dataclass(frozen=True)
class QueryOnlyLoRARun:
	"""One answer-retrieval dataset under the parameter-matched LoRA control."""

	dataset: str

	def validate(self) -> None:
		if self.dataset not in QUERY_ONLY_LORA_DATASETS:
			raise ValueError("Query-only LoRA queue requires GQA Balanced or CLEVR")


def build_training_command(
	run: QueryOnlyLoRARun,
	*,
	args: argparse.Namespace,
	output_dir: Path,
	resume_checkpoint: Path | None = None,
) -> list[str]:
	"""Build a one-epoch query-only LoRA command with one rolling checkpoint."""
	run.validate()
	if args.world_size * args.per_device_batch_size != 256:
		raise ValueError("Query-only LoRA requires a true contrastive batch of 256")
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
		"256",
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
	return command


def build_evaluation_command(
	run: QueryOnlyLoRARun,
	*,
	args: argparse.Namespace,
	training_output: Path,
	evaluation_output: Path,
) -> list[str]:
	"""Build the matching full-test command without a candidate Qwen forward."""
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


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.tmp")
	temporary.write_text(
		json.dumps(value, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	temporary.replace(path)


def _passed(path: Path) -> bool:
	return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get(
		"status",
	) == "passed"


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


def _candidate_identities(candidate_root: Path) -> dict[str, str]:
	identities = {}
	for dataset in QUERY_ONLY_LORA_DATASETS:
		spec = CandidateBankSpec(dataset, "shared", "answer")
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


def _git_commit(project_root: Path) -> str:
	return subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	).stdout.strip()


def _run(command: list[str], *, args: argparse.Namespace) -> None:
	environment = os.environ.copy()
	environment.update(
		{
			"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
			"CUDA_VISIBLE_DEVICES": ",".join(
				str(index) for index in range(args.world_size)
			),
			"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
			"TOKENIZERS_PARALLELISM": "false",
			"PYTHONPATH": str(Path(args.project_root) / "src"),
		},
	)
	result = subprocess.run(
		command,
		cwd=args.project_root,
		env=environment,
		check=False,
	)
	if result.returncode:
		raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def run_queue(args: argparse.Namespace, runs: tuple[QueryOnlyLoRARun, ...]) -> None:
	"""Train and test GQA then CLEVR, resuming only exact rolling checkpoints."""
	if tuple(run.dataset for run in runs) != QUERY_ONLY_LORA_DATASETS:
		raise ValueError("Queue order must be exactly GQA Balanced then CLEVR")
	if args.world_size != 8 or args.per_device_batch_size != 32:
		raise ValueError("Eight-V100 query-only LoRA is locked to 8 GPUs and batch 32")
	for run in runs:
		run.validate()
	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	manifest = {
		"scope": "query_only_last_four_lora_fixed_candidate_answer_retrieval",
		"runs": [asdict(run) for run in runs],
		"git_commit": _git_commit(Path(args.project_root)),
		"world_size": args.world_size,
		"per_device_batch_size": args.per_device_batch_size,
		"contrastive_global_batch_size": 256,
		"evaluation_batch_size": args.evaluation_batch_size,
		"epochs": 1,
		"validation_used": False,
		"candidate_qwen_forward_calls": 0,
		"candidate_bank_manifest_sha256": _candidate_identities(
			Path(args.candidate_root),
		),
		"checkpoint_every": args.checkpoint_every,
		"max_checkpoints": 1,
		"decoder_layer_indices": list(BASELINE_LORA_LAST_FOUR_DECODER_LAYERS),
		"hard_negative_count": 32,
		"dataset_root": str(args.dataset_root),
		"model_root": str(args.model_root),
		"candidate_root": str(args.candidate_root),
	}
	manifest_path = output_root / "queue_manifest.json"
	if manifest_path.exists():
		if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
			raise ValueError("Existing queue manifest does not match this exact queue")
	else:
		_write_json(manifest_path, manifest)
	status_path = output_root / "status.json"
	for run_index, run in enumerate(runs):
		run_root = output_root / run.dataset
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
				{
					"status": "training",
					"run_index": run_index,
					"dataset": run.dataset,
					"resume_checkpoint": (
						str(resume_checkpoint) if resume_checkpoint is not None else None
					),
					"command": command,
				},
			)
			_run(command, args=args)
		adapter_path = training_output / "adapter" / "adapter_model.safetensors"
		if not adapter_path.is_file():
			raise FileNotFoundError(f"Missing final query-only LoRA adapter: {adapter_path}")
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
				{
					"status": "testing",
					"run_index": run_index,
					"dataset": run.dataset,
					"command": command,
				},
			)
			_run(command, args=args)
		_write_json(
			run_root / "latest_test.json",
			{
				"path": str(evaluation_output),
				"report": str(evaluation_output / "report.json"),
			},
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
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	runs = tuple(QueryOnlyLoRARun(dataset) for dataset in QUERY_ONLY_LORA_DATASETS)
	try:
		run_queue(args, runs)
		return 0
	except KeyboardInterrupt:
		status = "interrupted"
		error: Exception | None = None
	except Exception as caught_error:
		status = "failed"
		error = caught_error
	output_root = Path(args.output_root)
	if output_root.exists():
		payload: dict[str, Any] = {"status": status}
		if error is not None:
			payload["error"] = repr(error)
		_write_json(output_root / "status.json", payload)
	if error is not None:
		print(f"query-only LoRA queue failed: {error!r}", file=sys.stderr, flush=True)
	return 130 if status == "interrupted" else 1


if __name__ == "__main__":
	raise SystemExit(main())
