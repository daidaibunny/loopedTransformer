"""Measure safe per-device batch throughput for each single-dataset baseline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from looped_vl.baseline.data import BASELINE_DATASETS


def _project_pythonpath(project_root: Path, existing: str | None) -> str:
	"""Prepend experiment code without hiding the selected runtime libraries."""
	entries = [str(project_root / "src")]
	if existing:
		entries.append(existing)
	return os.pathsep.join(entries)


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_command(
	args: argparse.Namespace,
	dataset: str,
	batch_size: int,
	output_dir: Path,
) -> list[str]:
	return [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		f"--nproc_per_node={args.world_size}",
		"-m",
		"looped_vl.baseline.train",
		"--dataset",
		dataset,
		"--dataset-root",
		str(Path(args.dataset_root) / dataset),
		"--model-root",
		str(args.model_root),
		"--project-root",
		str(args.project_root),
		"--output-dir",
		str(output_dir),
		"--expected-world-size",
		str(args.world_size),
		"--per-device-batch-size",
		str(batch_size),
		"--gradient-accumulation-steps",
		"1",
		"--num-workers",
		str(args.num_workers),
		"--max-optimizer-steps",
		str(args.optimizer_steps),
		"--skip-adapter-save",
	]


def _read_passed_metrics(output_dir: Path) -> dict[str, Any]:
	status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
	if status.get("status") != "passed":
		raise RuntimeError(f"Smoke status did not pass: {status}")
	records = [
		json.loads(line)
		for line in (output_dir / "train_metrics.jsonl").read_text(
			encoding="utf-8",
		).splitlines()
		if line.strip()
	]
	if not records:
		raise RuntimeError("Smoke produced no optimizer-step metrics")
	stable_records = records[1:] or records
	return {
		"median_samples_per_second": statistics.median(
			float(record["samples_per_second"]) for record in stable_records
		),
		"peak_gpu_memory_bytes": max(
			int(record["gpu_peak_memory_allocated_bytes"]) for record in records
		),
		"optimizer_steps": len(records),
	}


def run_search(args: argparse.Namespace) -> dict[str, Any]:
	output_root = Path(args.output_root)
	if output_root.exists():
		raise FileExistsError(f"Smoke search output already exists: {output_root}")
	output_root.mkdir(parents=True)
	environment = os.environ.copy()
	environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environment["CUDA_VISIBLE_DEVICES"] = ",".join(
		str(index) for index in range(args.world_size)
	)
	environment["PYTHONPATH"] = _project_pythonpath(
		Path(args.project_root),
		environment.get("PYTHONPATH"),
	)
	all_results: dict[str, Any] = {}
	selected: dict[str, Any] = {}
	for dataset in args.datasets:
		dataset_results: list[dict[str, Any]] = []
		for batch_size in args.batch_sizes:
			run_output = output_root / dataset / f"batch{batch_size}"
			run_output.parent.mkdir(parents=True, exist_ok=True)
			command = _training_command(args, dataset, batch_size, run_output)
			log_path = output_root / dataset / f"batch{batch_size}.log"
			with log_path.open("w", encoding="utf-8") as log_handle:
				result = subprocess.run(
					command,
					cwd=args.project_root,
					env=environment,
					stdout=log_handle,
					stderr=subprocess.STDOUT,
					check=False,
				)
			record: dict[str, Any] = {
				"batch_size": batch_size,
				"num_workers": args.num_workers,
				"return_code": result.returncode,
				"command": command,
				"log_path": str(log_path),
				"output_dir": str(run_output),
			}
			if result.returncode == 0:
				record.update(_read_passed_metrics(run_output))
				record["status"] = "passed"
			else:
				log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
				record["status"] = "failed"
				record["out_of_memory"] = "out of memory" in log_tail.lower()
				record["log_tail"] = log_tail
			dataset_results.append(record)
			_write_json(output_root / dataset / "search_progress.json", dataset_results)
			if record.get("out_of_memory"):
				break
		passed = [record for record in dataset_results if record["status"] == "passed"]
		if not passed:
			raise RuntimeError(f"No safe smoke configuration passed for {dataset}")
		memory_limit = args.memory_headroom_fraction * args.gpu_memory_bytes
		safe = [
			record
			for record in passed
			if int(record["peak_gpu_memory_bytes"]) <= memory_limit
		] or passed
		best = max(
			safe,
			key=lambda record: (
				float(record["median_samples_per_second"]),
				-int(record["peak_gpu_memory_bytes"]),
			),
		)
		accumulation = args.effective_global_batch_size // (
			args.world_size * int(best["batch_size"])
		)
		if accumulation <= 0 or (
			args.world_size * int(best["batch_size"]) * accumulation
			!= args.effective_global_batch_size
		):
			raise RuntimeError("Selected batch size cannot preserve the effective batch size")
		selected[dataset] = {
			"dataset": dataset,
			"per_device_batch_size": int(best["batch_size"]),
			"gradient_accumulation_steps": accumulation,
			"num_workers": args.num_workers,
			"effective_global_batch_size": args.effective_global_batch_size,
			"measured_median_samples_per_second": best["median_samples_per_second"],
			"peak_gpu_memory_bytes": best["peak_gpu_memory_bytes"],
		}
		all_results[dataset] = dataset_results
	result = {
		"status": "passed",
		"world_size": args.world_size,
		"optimizer_steps_per_trial": args.optimizer_steps,
		"results": all_results,
		"selected": selected,
	}
	_write_json(output_root / "selected_parameters.json", result)
	return result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--datasets",
		nargs="+",
		choices=BASELINE_DATASETS,
		default=list(BASELINE_DATASETS),
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
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/loopedTransformer"),
	)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--world-size", type=int, default=8)
	parser.add_argument("--optimizer-steps", type=int, default=4)
	parser.add_argument("--effective-global-batch-size", type=int, default=256)
	parser.add_argument("--gpu-memory-bytes", type=int, default=32 * 1024**3)
	parser.add_argument("--memory-headroom-fraction", type=float, default=0.90)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		result = run_search(args)
		print(json.dumps(result, indent=2, sort_keys=True))
		return 0
	except Exception as error:
		print(f"baseline smoke search failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
