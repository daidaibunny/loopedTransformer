"""Forward-only throughput benchmark for the frozen Qwen3-VL-Embedding baseline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from looped_vl.data import DEFAULT_DATASET_ROOT, LoopedVLMixtureDataset, mixture_collate
from looped_vl.smoke import (
	assert_model_frozen,
	checkpoint_sha256,
	freeze_model,
	load_local_embedding_module,
)

LOGGER = logging.getLogger("throughput")
EXPECTED_BATCH_RATIO = {"coco": 10, "gqa_balanced": 7, "clevr": 3}


def summarize_timings(
	batch_size: int,
	batch_total_seconds: list[float],
	batch_load_seconds: list[float],
	batch_process_seconds: list[float],
	train_samples: int,
	full_samples: int,
) -> dict[str, Any]:
	"""Summarize observed rates and project one-pass dataset runtimes."""
	batch_count = len(batch_total_seconds)
	if batch_count == 0:
		raise ValueError("At least one measured batch is required")
	if len(batch_load_seconds) != batch_count or len(batch_process_seconds) != batch_count:
		raise ValueError("Timing arrays must have equal lengths")
	if any(value <= 0 for value in batch_total_seconds + batch_process_seconds):
		raise ValueError("Total and process timings must be positive")
	if any(value < 0 for value in batch_load_seconds):
		raise ValueError("Load timings cannot be negative")

	measured_samples = batch_size * batch_count
	total_seconds = sum(batch_total_seconds)
	load_seconds = sum(batch_load_seconds)
	process_seconds = sum(batch_process_seconds)
	end_to_end_rate = measured_samples / total_seconds
	process_rate = measured_samples / process_seconds
	load_rate = measured_samples / load_seconds if load_seconds > 0 else math.inf
	mean_batch_seconds = statistics.fmean(batch_total_seconds)
	batch_stddev = statistics.pstdev(batch_total_seconds)
	coefficient_of_variation = batch_stddev / mean_batch_seconds
	confidence_margin = 1.96 * batch_stddev / math.sqrt(batch_count)
	lower_rate = batch_size / (mean_batch_seconds + confidence_margin)
	upper_denominator = max(mean_batch_seconds - confidence_margin, 1e-12)
	upper_rate = batch_size / upper_denominator

	projected_train_seconds = train_samples / end_to_end_rate
	projected_full_seconds = full_samples / end_to_end_rate
	return {
		"measured_batches": batch_count,
		"measured_samples": measured_samples,
		"measured_end_to_end_seconds": total_seconds,
		"measured_data_load_seconds": load_seconds,
		"measured_process_seconds": process_seconds,
		"mean_batch_seconds": mean_batch_seconds,
		"batch_time_stddev_seconds": batch_stddev,
		"batch_time_coefficient_of_variation": coefficient_of_variation,
		"end_to_end_samples_per_second": end_to_end_rate,
		"process_samples_per_second": process_rate,
		"data_load_samples_per_second": load_rate,
		"throughput_95_percent_interval": [lower_rate, upper_rate],
		"projected_train_seconds": projected_train_seconds,
		"projected_train_hours": projected_train_seconds / 3600,
		"projected_train_95_percent_interval_seconds": [
			train_samples / upper_rate,
			train_samples / lower_rate,
		],
		"projected_full_seconds": projected_full_seconds,
		"projected_full_hours": projected_full_seconds / 3600,
		"projected_full_95_percent_interval_seconds": [
			full_samples / upper_rate,
			full_samples / lower_rate,
		],
	}


def validate_embeddings(embeddings: torch.Tensor, batch_size: int) -> tuple[float, float]:
	"""Validate one embedding batch and return its minimum and maximum norms."""
	if embeddings.ndim != 2 or embeddings.shape[0] != batch_size:
		raise RuntimeError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")
	if not torch.isfinite(embeddings).all():
		raise RuntimeError("Embeddings contain non-finite values")
	norms = torch.linalg.vector_norm(embeddings.float(), dim=1)
	if not torch.allclose(norms, torch.ones_like(norms), atol=5e-3, rtol=5e-3):
		raise RuntimeError(f"Embeddings are not unit normalized: {norms.tolist()}")
	if embeddings.requires_grad or embeddings.grad_fn is not None:
		raise RuntimeError("Inference embeddings unexpectedly track gradients")
	return float(norms.min().item()), float(norms.max().item())


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
	"""Run warmup and measured forward-only batches on one visible CUDA device."""
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU fallback is disabled")
	if torch.cuda.device_count() != 1:
		raise RuntimeError(
			"Benchmark expects exactly one visible GPU; set CUDA_VISIBLE_DEVICES explicitly",
		)
	if args.batch_size != 20:
		raise ValueError("batch_size must be 20 to preserve the exact 10:7:3 mixture per batch")
	if args.warmup_batches < 1 or args.measure_batches < 2:
		raise ValueError("Use at least one warmup batch and two measured batches")

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	model_root = Path(args.model_root)
	checkpoint_path = model_root / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(checkpoint_path)
	dataset = LoopedVLMixtureDataset(
		args.dataset_root,
		"train",
		args.gqa_materialized_root,
	)
	required_samples = args.batch_size * (args.warmup_batches + args.measure_batches)
	if required_samples > len(dataset):
		raise ValueError(f"Benchmark requires {required_samples} rows, dataset has {len(dataset)}")
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.batch_size,
		"shuffle": False,
		"drop_last": True,
		"num_workers": args.num_workers,
		"collate_fn": mixture_collate,
		"pin_memory": False,
	}
	if args.num_workers > 0:
		loader_kwargs.update(
			{
				"persistent_workers": True,
				"prefetch_factor": args.prefetch_factor,
			}
		)
	loader = DataLoader(**loader_kwargs)
	iterator = iter(loader)

	module = load_local_embedding_module(model_root)
	model_load_start = time.perf_counter()
	embedder = module.Qwen3VLEmbedder(
		model_name_or_path=str(model_root),
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
		torch_dtype=torch.bfloat16,
		attn_implementation="sdpa",
	)
	freeze_model(embedder.model)
	assert_model_frozen(embedder.model)
	model_load_seconds = time.perf_counter() - model_load_start
	torch.cuda.reset_peak_memory_stats()

	def process_next_batch() -> tuple[float, float, float, tuple[float, float]]:
		batch_start = time.perf_counter()
		batch = next(iterator)
		load_seconds = time.perf_counter() - batch_start
		source_counts = Counter(batch["sources"])
		if dict(source_counts) != EXPECTED_BATCH_RATIO:
			raise RuntimeError(f"Unexpected batch source ratio: {source_counts}")
		torch.cuda.synchronize()
		process_start = time.perf_counter()
		with torch.inference_mode():
			embeddings = embedder.process(batch["model_inputs"])
		torch.cuda.synchronize()
		process_seconds = time.perf_counter() - process_start
		total_seconds = time.perf_counter() - batch_start
		norm_range = validate_embeddings(embeddings, args.batch_size)
		return total_seconds, load_seconds, process_seconds, norm_range

	for _ in range(args.warmup_batches):
		process_next_batch()

	batch_total_seconds: list[float] = []
	batch_load_seconds: list[float] = []
	batch_process_seconds: list[float] = []
	norm_minimum = math.inf
	norm_maximum = -math.inf
	for _ in range(args.measure_batches):
		total_seconds, load_seconds, process_seconds, norm_range = process_next_batch()
		batch_total_seconds.append(total_seconds)
		batch_load_seconds.append(load_seconds)
		batch_process_seconds.append(process_seconds)
		norm_minimum = min(norm_minimum, norm_range[0])
		norm_maximum = max(norm_maximum, norm_range[1])

	assert_model_frozen(embedder.model)
	checkpoint_hash_after = checkpoint_sha256(checkpoint_path)
	if checkpoint_hash_after != checkpoint_hash_before:
		raise RuntimeError("Model checkpoint hash changed during throughput benchmark")
	parameter_count = sum(parameter.numel() for parameter in embedder.model.parameters())
	trainable_parameter_count = sum(
		parameter.numel()
		for parameter in embedder.model.parameters()
		if parameter.requires_grad
	)
	properties = torch.cuda.get_device_properties(0)
	timing_summary = summarize_timings(
		batch_size=args.batch_size,
		batch_total_seconds=batch_total_seconds,
		batch_load_seconds=batch_load_seconds,
		batch_process_seconds=batch_process_seconds,
		train_samples=len(dataset),
		full_samples=len(dataset) + args.validation_samples + args.test_samples,
	)
	return {
		"status": "passed",
		"scope": "frozen_forward_only_no_backward_no_optimizer",
		"dataset_train_samples": len(dataset),
		"dataset_validation_samples": args.validation_samples,
		"dataset_test_samples": args.test_samples,
		"batch_size": args.batch_size,
		"batch_source_counts": EXPECTED_BATCH_RATIO,
		"warmup_batches": args.warmup_batches,
		"measure_batches": args.measure_batches,
		"num_workers": args.num_workers,
		"prefetch_factor": args.prefetch_factor,
		"seed": args.seed,
		"precision": "bfloat16",
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
		"embedding_dimension": 2048,
		"embedding_norm_range": [norm_minimum, norm_maximum],
		"parameter_count": parameter_count,
		"trainable_parameter_count": trainable_parameter_count,
		"checkpoint_sha256_before": checkpoint_hash_before,
		"checkpoint_sha256_after": checkpoint_hash_after,
		"cuda_device_name": properties.name,
		"cuda_visible_device_count": torch.cuda.device_count(),
		"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
		"model_load_seconds": model_load_seconds,
		"batch_total_seconds": batch_total_seconds,
		"batch_load_seconds": batch_load_seconds,
		"batch_process_seconds": batch_process_seconds,
		"timing": timing_summary,
	}


def parse_args() -> argparse.Namespace:
	"""Parse throughput benchmark arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset-root",
		default=DEFAULT_DATASET_ROOT,
	)
	parser.add_argument(
		"--gqa-materialized-root",
		default="/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced",
	)
	parser.add_argument(
		"--model-root",
		default="/mnt/afs/liyiwei/models/Qwen3-VL-Embedding-2B/base_original",
	)
	parser.add_argument("--batch-size", type=int, default=20)
	parser.add_argument("--warmup-batches", type=int, default=5)
	parser.add_argument("--measure-batches", type=int, default=25)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--seed", type=int, default=20260729)
	parser.add_argument("--validation-samples", type=int, default=25_000)
	parser.add_argument("--test-samples", type=int, default=25_000)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=64 * 64)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument("--output-json")
	return parser.parse_args()


def main() -> int:
	"""Run the benchmark and emit its JSON result."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = parse_args()
	try:
		result = run_benchmark(args)
		serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
		print(serialized, end="")
		if args.output_json:
			Path(args.output_json).write_text(serialized, encoding="utf-8")
		return 0
	except Exception:
		LOGGER.exception("Frozen Qwen throughput benchmark failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
