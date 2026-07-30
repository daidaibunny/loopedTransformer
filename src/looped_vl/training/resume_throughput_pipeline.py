"""Tune physical batches, run frozen test, and launch or resume full training."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from looped_vl.training.wait_and_launch import wait_for_idle_window


@dataclass(frozen=True)
class TrainingBenchmark:
	"""One successful resumed-training throughput measurement."""

	per_device_batch_size: int
	samples_per_second: float
	peak_memory_bytes: int
	output_dir: Path


@dataclass(frozen=True)
class FrozenEvaluationBenchmark:
	"""One successful two-GPU frozen evaluation throughput measurement."""

	batch_size: int
	encoding_seconds: float
	peak_memory_bytes: int
	output_dir: Path


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_training_command(
	*,
	torchrun: Path,
	output_dir: Path,
	resume_checkpoint: Path | None,
	per_device_batch_size: int,
	resume_per_device_batch_size: int | None,
	code_commit: str,
	max_additional_optimizer_steps: int,
	num_workers: int,
	prefetch_factor: int,
	end_stage: int,
) -> list[str]:
	"""Build one exact two-GPU Stage-1 benchmark or full-resume command."""
	command = [
		str(torchrun),
		"--standalone",
		"--nproc_per_node=2",
		"-m",
		"looped_vl.training.train",
		"--expected-world-size",
		"2",
		"--start-stage",
		"1",
		"--end-stage",
		str(end_stage),
		"--output-dir",
		str(output_dir),
		"--per-device-batch-size",
		str(per_device_batch_size),
		"--num-workers",
		str(num_workers),
		"--prefetch-factor",
		str(prefetch_factor),
		"--checkpoint-every",
		"100",
		"--code-commit",
		code_commit,
	]
	if resume_checkpoint is not None:
		command.extend(["--resume-checkpoint", str(resume_checkpoint)])
		if resume_per_device_batch_size is not None:
			command.extend(
				[
					"--resume-per-device-batch-size",
					str(resume_per_device_batch_size),
				],
			)
	if max_additional_optimizer_steps:
		command.extend(
			[
				"--max-additional-optimizer-steps",
				str(max_additional_optimizer_steps),
			],
		)
	return command


def build_frozen_evaluation_command(
	*,
	torchrun: Path,
	output_dir: Path,
	batch_size: int,
	num_workers: int,
	prefetch_factor: int,
	score_batch_size: int,
	max_test_rows: int,
) -> list[str]:
	"""Build a two-GPU frozen evaluation command for a smoke or complete test."""
	command = [
		str(torchrun),
		"--standalone",
		"--nproc_per_node=2",
		"-m",
		"looped_vl.evaluate_frozen",
		"--expected-world-size",
		"2",
		"--output-dir",
		str(output_dir),
		"--batch-size",
		str(batch_size),
		"--num-workers",
		str(num_workers),
		"--prefetch-factor",
		str(prefetch_factor),
		"--score-batch-size",
		str(score_batch_size),
	]
	if max_test_rows:
		command.extend(["--max-test-rows", str(max_test_rows)])
	return command


def select_best_training_benchmark(
	benchmarks: list[TrainingBenchmark],
	*,
	memory_limit_bytes: int,
) -> TrainingBenchmark:
	"""Select maximum throughput among runs that preserve explicit memory headroom."""
	eligible = [
		benchmark
		for benchmark in benchmarks
		if benchmark.peak_memory_bytes <= memory_limit_bytes
	]
	if not eligible:
		raise RuntimeError("No training batch candidate passed the memory-headroom gate")
	return max(eligible, key=lambda benchmark: benchmark.samples_per_second)


def _select_best_frozen_benchmark(
	benchmarks: list[FrozenEvaluationBenchmark],
	*,
	memory_limit_bytes: int,
) -> FrozenEvaluationBenchmark:
	eligible = [
		benchmark
		for benchmark in benchmarks
		if benchmark.peak_memory_bytes <= memory_limit_bytes
	]
	if not eligible:
		raise RuntimeError("No frozen evaluation batch passed the memory-headroom gate")
	return min(eligible, key=lambda benchmark: benchmark.encoding_seconds)


def _run_logged(
	command: list[str],
	*,
	cwd: Path,
	environment: dict[str, str],
	log_path: Path,
) -> int:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("w", encoding="utf-8") as log_handle:
		result = subprocess.run(
			command,
			cwd=cwd,
			env=environment,
			stdout=log_handle,
			stderr=subprocess.STDOUT,
			check=False,
		)
	return result.returncode


def _wait_for_checkpoint(
	*,
	checkpoint_path: Path,
	latest_checkpoint_path: Path,
	status_path: Path,
	poll_seconds: float,
) -> None:
	while True:
		latest = {}
		if latest_checkpoint_path.is_file():
			latest = json.loads(latest_checkpoint_path.read_text(encoding="utf-8"))
		if checkpoint_path.is_file() and latest.get("path") == str(checkpoint_path):
			return
		_write_json(
			status_path,
			{
				"status": "waiting_for_source_checkpoint",
				"checkpoint": str(checkpoint_path),
				"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			},
		)
		time.sleep(poll_seconds)


def _stop_source_training(tmux_session: str) -> None:
	session_check = subprocess.run(
		["tmux", "has-session", "-t", tmux_session],
		check=False,
		capture_output=True,
	)
	if session_check.returncode != 0:
		raise RuntimeError(f"Source training tmux is missing: {tmux_session}")
	subprocess.run(
		["tmux", "send-keys", "-t", tmux_session, "C-c"],
		check=True,
	)


def validate_resume_configuration(
	*,
	resume_checkpoint: Path | None,
	latest_checkpoint_json: Path | None,
	source_tmux_session: str | None,
) -> bool:
	"""Validate checkpoint-source arguments and return whether this is a resume."""
	values = (resume_checkpoint, latest_checkpoint_json, source_tmux_session)
	if any(value is not None for value in values) and not all(
		value is not None for value in values
	):
		raise ValueError(
			"resume_checkpoint, latest_checkpoint_json, and source_tmux_session "
			"must all be provided for checkpoint-resume mode",
		)
	return resume_checkpoint is not None


def _environment(cuda_visible_devices: str) -> dict[str, str]:
	environment = os.environ.copy()
	environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
	environment["PYTHONPATH"] = "src"
	return environment


def _run_architecture_acceptance(
	*,
	python: Path,
	project_root: Path,
	pipeline_root: Path,
) -> None:
	for mode in ("base_equivalence", "full_forward"):
		output_path = pipeline_root / "acceptance" / f"{mode}.json"
		command = [
			str(python),
			"-m",
			"looped_vl.training.model_acceptance",
			"--mode",
			mode,
			"--output-json",
			str(output_path),
		]
		if mode == "full_forward":
			command.append("--enable-lora")
		return_code = _run_logged(
			command,
			cwd=project_root,
			environment=_environment("0"),
			log_path=pipeline_root / "logs" / f"acceptance_{mode}.log",
		)
		if return_code != 0:
			raise RuntimeError(f"Architecture acceptance failed: {mode}")
		result = json.loads(output_path.read_text(encoding="utf-8"))
		if result.get("status") != "passed":
			raise RuntimeError(f"Architecture acceptance was not passed: {result}")


def _read_training_benchmark(
	output_dir: Path,
	per_device_batch_size: int,
) -> TrainingBenchmark:
	rows = [
		json.loads(line)
		for line in (output_dir / "train_metrics.jsonl").read_text(
			encoding="utf-8",
		).splitlines()
		if line.strip()
	]
	if not rows:
		raise RuntimeError(f"Training benchmark emitted no metrics: {output_dir}")
	stable_rows = rows[-2:] if len(rows) > 1 else rows
	return TrainingBenchmark(
		per_device_batch_size=per_device_batch_size,
		samples_per_second=statistics.fmean(
			float(row["samples_per_second"]) for row in stable_rows
		),
		peak_memory_bytes=max(int(row["gpu_peak_memory_allocated_bytes"]) for row in rows),
		output_dir=output_dir,
	)


def _read_frozen_benchmark(
	output_dir: Path,
	batch_size: int,
) -> FrozenEvaluationBenchmark:
	report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
	if report.get("status") != "passed":
		raise RuntimeError(f"Frozen benchmark was not passed: {report}")
	return FrozenEvaluationBenchmark(
		batch_size=batch_size,
		encoding_seconds=float(report["runtime"]["encoding_wall_seconds"]),
		peak_memory_bytes=max(
			int(rank["peak_gpu_memory_bytes"])
			for rank in report["distributed"]["ranks"]
		),
		output_dir=output_dir,
	)


def run_pipeline(args: argparse.Namespace) -> None:
	"""Execute the fresh-or-resumed benchmark, test, and training sequence."""
	project_root = Path(args.project_root)
	pipeline_root = Path(args.pipeline_root)
	if pipeline_root.exists():
		raise FileExistsError(f"Pipeline output already exists: {pipeline_root}")
	pipeline_root.mkdir(parents=True)
	status_path = pipeline_root / "status.json"
	is_resume = validate_resume_configuration(
		resume_checkpoint=args.resume_checkpoint,
		latest_checkpoint_json=args.latest_checkpoint_json,
		source_tmux_session=args.source_tmux_session,
	)
	checkpoint_path = args.resume_checkpoint
	if is_resume:
		assert checkpoint_path is not None
		assert args.latest_checkpoint_json is not None
		assert args.source_tmux_session is not None
		_wait_for_checkpoint(
			checkpoint_path=checkpoint_path,
			latest_checkpoint_path=args.latest_checkpoint_json,
			status_path=status_path,
			poll_seconds=args.poll_seconds,
		)
		_write_json(status_path, {"status": "stopping_source_training_at_checkpoint"})
		_stop_source_training(args.source_tmux_session)
	_write_json(status_path, {"status": "running_architecture_acceptance"})
	_run_architecture_acceptance(
		python=Path(args.python),
		project_root=project_root,
		pipeline_root=pipeline_root,
	)

	torchrun = Path(args.python).with_name("torchrun")
	training_benchmarks: list[TrainingBenchmark] = []
	training_failures: dict[int, int] = {}
	for batch_size in args.training_batch_sizes:
		output_dir = pipeline_root / f"training_batch{batch_size}"
		_write_json(
			status_path,
			{"status": "running_training_benchmark", "batch_size": batch_size},
		)
		command = build_training_command(
			torchrun=torchrun,
			output_dir=output_dir,
			resume_checkpoint=checkpoint_path,
			per_device_batch_size=batch_size,
			resume_per_device_batch_size=args.resume_per_device_batch_size,
			code_commit=args.code_commit,
			max_additional_optimizer_steps=args.training_benchmark_steps,
			num_workers=args.num_workers,
			prefetch_factor=args.prefetch_factor,
			end_stage=1,
		)
		return_code = _run_logged(
			command,
			cwd=project_root,
			environment=_environment("0,1"),
			log_path=pipeline_root / "logs" / f"training_batch{batch_size}.log",
		)
		if return_code == 0:
			training_benchmarks.append(_read_training_benchmark(output_dir, batch_size))
		else:
			training_failures[batch_size] = return_code
		wait_for_idle_window(
			required_seconds=10.0,
			poll_seconds=2.0,
			log_path=pipeline_root / f"gpu_idle_after_training_batch{batch_size}.jsonl",
		)
	best_training = select_best_training_benchmark(
		training_benchmarks,
		memory_limit_bytes=args.memory_limit_gib * 1024**3,
	)
	_write_json(
		pipeline_root / "training_benchmarks.json",
		{
			"passed": [
				{
					**asdict(benchmark),
					"output_dir": str(benchmark.output_dir),
				}
				for benchmark in training_benchmarks
			],
			"failed_return_codes": training_failures,
			"selected_batch_size": best_training.per_device_batch_size,
		},
	)

	frozen_benchmarks: list[FrozenEvaluationBenchmark] = []
	frozen_failures: dict[int, int] = {}
	for batch_size in args.frozen_batch_sizes:
		output_dir = pipeline_root / f"frozen_smoke_batch{batch_size}"
		_write_json(
			status_path,
			{"status": "running_frozen_benchmark", "batch_size": batch_size},
		)
		command = build_frozen_evaluation_command(
			torchrun=torchrun,
			output_dir=output_dir,
			batch_size=batch_size,
			num_workers=args.num_workers,
			prefetch_factor=args.prefetch_factor,
			score_batch_size=args.score_batch_size,
			max_test_rows=args.frozen_benchmark_rows,
		)
		return_code = _run_logged(
			command,
			cwd=project_root,
			environment=_environment("0,1"),
			log_path=pipeline_root / "logs" / f"frozen_smoke_batch{batch_size}.log",
		)
		if return_code == 0:
			frozen_benchmarks.append(_read_frozen_benchmark(output_dir, batch_size))
		else:
			frozen_failures[batch_size] = return_code
		wait_for_idle_window(
			required_seconds=10.0,
			poll_seconds=2.0,
			log_path=pipeline_root / f"gpu_idle_after_frozen_batch{batch_size}.jsonl",
		)
	best_frozen = _select_best_frozen_benchmark(
		frozen_benchmarks,
		memory_limit_bytes=args.memory_limit_gib * 1024**3,
	)
	_write_json(
		pipeline_root / "frozen_benchmarks.json",
		{
			"passed": [
				{
					**asdict(benchmark),
					"output_dir": str(benchmark.output_dir),
				}
				for benchmark in frozen_benchmarks
			],
			"failed_return_codes": frozen_failures,
			"selected_batch_size": best_frozen.batch_size,
		},
	)

	full_frozen_output = Path(args.full_frozen_output)
	_write_json(
		status_path,
		{"status": "running_complete_frozen_test", "batch_size": best_frozen.batch_size},
	)
	full_frozen_return_code = _run_logged(
		build_frozen_evaluation_command(
			torchrun=torchrun,
			output_dir=full_frozen_output,
			batch_size=best_frozen.batch_size,
			num_workers=args.num_workers,
			prefetch_factor=args.prefetch_factor,
			score_batch_size=args.score_batch_size,
			max_test_rows=0,
		),
		cwd=project_root,
		environment=_environment("0,1"),
		log_path=pipeline_root / "logs" / "frozen_complete_test.log",
	)
	if full_frozen_return_code != 0:
		raise RuntimeError(
			f"Complete frozen test failed with exit code {full_frozen_return_code}",
		)
	full_frozen_status = json.loads(
		(full_frozen_output / "status.json").read_text(encoding="utf-8"),
	)
	if full_frozen_status.get("status") != "passed":
		raise RuntimeError(f"Complete frozen test was not passed: {full_frozen_status}")

	training_output = Path(args.training_output)
	_write_json(
		status_path,
		{
			"status": "running_training",
			"training_mode": "resume" if is_resume else "fresh",
			"per_device_batch_size": best_training.per_device_batch_size,
			"frozen_report": str(full_frozen_output / "report.json"),
		},
	)
	training_return_code = _run_logged(
		build_training_command(
			torchrun=torchrun,
			output_dir=training_output,
			resume_checkpoint=checkpoint_path,
			per_device_batch_size=best_training.per_device_batch_size,
			resume_per_device_batch_size=args.resume_per_device_batch_size,
			code_commit=args.code_commit,
			max_additional_optimizer_steps=0,
			num_workers=args.num_workers,
			prefetch_factor=args.prefetch_factor,
			end_stage=2,
		),
		cwd=project_root,
		environment=_environment("0,1"),
		log_path=pipeline_root / "logs" / "training.log",
	)
	if training_return_code != 0:
		raise RuntimeError(f"Training failed with exit code {training_return_code}")
	_write_json(status_path, {"status": "passed"})


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
	batch_sizes = tuple(int(item) for item in value.split(","))
	if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
		raise argparse.ArgumentTypeError("Batch sizes must be positive comma-separated integers")
	return batch_sizes


def parse_args() -> argparse.Namespace:
	"""Parse the fresh-or-resumed performance pipeline."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/loopedTransformer"),
	)
	parser.add_argument(
		"--python",
		type=Path,
		default=Path("/mnt/afs/likangle/reserach/LOCUS-MLLM/envs/LOCUS/bin/python"),
	)
	parser.add_argument("--pipeline-root", type=Path, required=True)
	parser.add_argument("--source-tmux-session")
	parser.add_argument("--resume-checkpoint", type=Path)
	parser.add_argument("--latest-checkpoint-json", type=Path)
	parser.add_argument("--resume-per-device-batch-size", type=int)
	parser.add_argument("--training-batch-sizes", type=_parse_batch_sizes, default=(8,))
	parser.add_argument("--training-benchmark-steps", type=int, default=3)
	parser.add_argument("--frozen-batch-sizes", type=_parse_batch_sizes, default=(64, 128, 256))
	parser.add_argument("--frozen-benchmark-rows", type=int, default=800)
	parser.add_argument("--score-batch-size", type=int, default=1024)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--prefetch-factor", type=int, default=4)
	parser.add_argument("--memory-limit-gib", type=int, default=72)
	parser.add_argument("--poll-seconds", type=float, default=30.0)
	parser.add_argument("--code-commit", required=True)
	parser.add_argument("--full-frozen-output", type=Path, required=True)
	parser.add_argument(
		"--training-output",
		"--resumed-training-output",
		dest="training_output",
		type=Path,
		required=True,
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_pipeline(args)
		return 0
	except Exception as error:
		pipeline_root = Path(args.pipeline_root)
		if pipeline_root.exists():
			_write_json(
				pipeline_root / "status.json",
				{"status": "failed", "error": repr(error)},
			)
		print(f"resume throughput pipeline failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
