"""Evaluate every recurrent pass against immutable candidate galleries."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from looped_vl.baseline.bucketing import (
	DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	DEFAULT_VISUAL_LENGTH_BUCKETS,
	group_baseline_model_inputs,
)
from looped_vl.baseline.data import BASELINE_DATASETS
from looped_vl.baseline.evaluate import (
	EvaluationDataset,
	_build_groups,
	_close_images,
	_collate,
	_load_rows,
	compute_ranking_metrics,
)
from looped_vl.baseline.model import BaselineInputProcessor, load_frozen_evaluation_model
from looped_vl.candidate_bank import CandidateBankSpec
from looped_vl.metrics import METRIC_SCALE, REQUIRED_RANKING_METRICS
from looped_vl.query_recurrent.backbone import (
	FrozenQueryBackbone,
)
from looped_vl.query_recurrent.candidate_store import ImmutableCandidateStore
from looped_vl.query_recurrent.config import QueryRecurrentConfig
from looped_vl.query_recurrent.model import (
	GroupedQueryRecurrentHead,
	QueryRecurrentHead,
	recurrent_fp32_context,
)
from looped_vl.smoke import checkpoint_sha256

LOGGER = logging.getLogger("query_recurrent_evaluate")
RETRIEVAL_CUTOFFS = (1, 5, 10, 20)


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU fallback is disabled")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="gloo")
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(f"Expected {expected_world_size} ranks, found {world_size}")
	return rank, world_size, torch.device("cuda", local_rank)


def _load_recurrent_head(
	checkpoint_path: Path,
	*,
	device: torch.device,
) -> tuple[QueryRecurrentHead, dict[str, Any], str]:
	payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	identity = payload["config"]
	config_names = {field.name for field in fields(QueryRecurrentConfig)}
	config = QueryRecurrentConfig(
		**{name: identity[name] for name in config_names if name in identity},
	)
	config.validate()
	head = QueryRecurrentHead(config)
	head.load_state_dict(payload["state_dict"], strict=True)
	head.to(device).eval()
	return head, payload["run_manifest"], checkpoint_sha256(checkpoint_path)


def _query_group_names(dataset: str) -> tuple[str, ...]:
	return ("text_query", "image_query") if dataset == "coco" else ("query",)


def _variant_names(max_recurrent_steps: int) -> tuple[str, ...]:
	"""Name the frozen baseline and every fixed recurrent pass."""
	return (
		"pass_0_frozen_backbone",
		*(f"pass_{step}" for step in range(1, max_recurrent_steps + 1)),
	)


def _append_finite_embedding_chunks(
	embedding_chunks: dict[str, list[torch.Tensor]],
	variants: dict[str, torch.Tensor],
	*,
	group_name: str,
) -> None:
	"""Append every valid batch tensor; reject non-finite values before caching."""
	if set(embedding_chunks) != set(variants):
		raise ValueError("Embedding chunk and variant names must match")
	for variant, embeddings in variants.items():
		if not torch.isfinite(embeddings).all():
			raise RuntimeError(f"Non-finite {variant} embeddings in {group_name}")
		embedding_chunks[variant].append(embeddings.detach().cpu().half())


def _encode_query_group(
	*,
	name: str,
	items: list[Any],
	backbone: FrozenQueryBackbone,
	head: QueryRecurrentHead,
	processor: BaselineInputProcessor,
	args: argparse.Namespace,
	rank: int,
	world_size: int,
	device: torch.device,
	output_dir: Path,
) -> None:
	indices = list(range(rank, len(items), world_size))
	dataset = EvaluationDataset(items, indices)
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.batch_size,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": _collate,
		"pin_memory": True,
	}
	if args.num_workers:
		loader_kwargs.update(
			{
				"multiprocessing_context": "spawn",
				"persistent_workers": True,
				"prefetch_factor": args.prefetch_factor,
			},
		)
	loader = DataLoader(**loader_kwargs)
	grouped_head = GroupedQueryRecurrentHead(head)
	index_chunks: list[torch.Tensor] = []
	embedding_chunks: dict[str, list[torch.Tensor]] = {
		variant: [] for variant in _variant_names(head.config.max_recurrent_steps)
	}
	start = time.perf_counter()
	processed_count = 0
	for batch_number, batch in enumerate(loader, start=1):
		try:
			groups = group_baseline_model_inputs(
				batch["model_inputs"],
				min_pixels=args.min_pixels,
				max_pixels=args.max_pixels,
				max_visual_buckets=args.visual_length_buckets,
				min_visual_bucket_size=args.min_visual_bucket_size,
			)
			processed = tuple(
				processor.prepare(list(group.model_inputs), device=device) for group in groups
			)
		finally:
			_close_images(batch["model_inputs"])
		frozen_groups = []
		with torch.autocast(device_type="cuda", dtype=torch.float16):
			for group, inputs in zip(groups, processed, strict=True):
				frozen = backbone(inputs)
				frozen_groups.append(
					(
						group.original_indices,
						frozen.base_embeddings,
					),
				)
		with torch.inference_mode(), recurrent_fp32_context(device.type):
			output = grouped_head(
				feature_groups=tuple(frozen_groups),
				total_rows=len(batch["global_indices"]),
			)
			base_embeddings = torch.cat(
				[feature_group[1] for feature_group in frozen_groups],
				dim=0,
			)
			restore_order = torch.argsort(
				torch.tensor(
					[
						index
						for feature_group in frozen_groups
						for index in feature_group[0]
					],
					device=device,
				),
			)
			base_embeddings = base_embeddings.index_select(0, restore_order)
		variants = {
			"pass_0_frozen_backbone": base_embeddings,
			**{
				f"pass_{step}": embedding
				for step, embedding in enumerate(output.step_embeddings, start=1)
			},
		}
		_append_finite_embedding_chunks(
			embedding_chunks,
			variants,
			group_name=name,
		)
		index_chunks.append(torch.tensor(batch["global_indices"], dtype=torch.long))
		processed_count += len(batch["global_indices"])
		if batch_number == 1 or batch_number % args.log_every_batches == 0:
			LOGGER.info(
				"group=%s rank=%d processed=%d/%d items/s=%.2f",
				name,
				rank,
				processed_count,
				len(indices),
				processed_count / (time.perf_counter() - start),
			)
	torch.save(
		{
			"indices": (
				torch.cat(index_chunks)
				if index_chunks
				else torch.empty(0, dtype=torch.long)
			),
			"embeddings": {
				variant: (
					torch.cat(chunks)
					if chunks
					else torch.empty((0, 2048), dtype=torch.float16)
				)
				for variant, chunks in embedding_chunks.items()
			},
		},
		output_dir / "embedding_cache" / f"{name}.rank{rank}.pt",
	)


def _combine_query_cache(
	name: str,
	*,
	item_count: int,
	world_size: int,
	output_dir: Path,
	variant_names: tuple[str, ...],
) -> dict[str, Any]:
	seen = torch.zeros(item_count, dtype=torch.bool)
	embeddings = {
		variant: torch.empty((item_count, 2048), dtype=torch.float16)
		for variant in variant_names
	}
	for rank in range(world_size):
		shard = torch.load(
			output_dir / "embedding_cache" / f"{name}.rank{rank}.pt",
			map_location="cpu",
			weights_only=True,
		)
		indices = shard["indices"]
		if seen[indices].any():
			raise RuntimeError(f"Duplicate distributed indexes for {name}")
		for variant in variant_names:
			embeddings[variant][indices] = shard["embeddings"][variant]
		seen[indices] = True
	if not seen.all():
		raise RuntimeError(f"Missing distributed query indexes for {name}")
	return {"embeddings": embeddings}


def _metric_deltas(
	metrics: dict[str, Any],
	baseline: dict[str, Any],
) -> dict[str, Any]:
	if "aggregate" in metrics:
		return {
			direction: {
				metric: metrics[direction][metric] - baseline[direction][metric]
				for metric in REQUIRED_RANKING_METRICS
			}
			for direction in ("aggregate", "text_to_image", "image_to_text")
		}
	return {
		metric: metrics[metric] - baseline[metric]
		for metric in REQUIRED_RANKING_METRICS
	}


def _score_variants(
	*,
	dataset: str,
	query_caches: dict[str, dict[str, Any]],
	relevance: dict[str, Any],
	candidate_root: Path,
	model_hash: str,
	device: torch.device,
	score_batch_size: int,
	variant_names: tuple[str, ...],
) -> dict[str, Any]:
	if dataset == "coco":
		image_store = ImmutableCandidateStore(
			candidate_root=candidate_root,
			spec=CandidateBankSpec("coco", "test", "image"),
			model_checkpoint_sha256=model_hash,
			validate_checksums=True,
		)
		text_store = ImmutableCandidateStore(
			candidate_root=candidate_root,
			spec=CandidateBankSpec("coco", "test", "text"),
			model_checkpoint_sha256=model_hash,
			validate_checksums=True,
		)
		image_candidates = image_store.embeddings.float()
		text_candidates = text_store.embeddings.float()
		results = {}
		for variant in variant_names:
			text_to_image, coverage_t2i = compute_ranking_metrics(
				query_caches["text_query"]["embeddings"][variant].float(),
				image_candidates,
				relevance["text_to_image"],
				device=device,
				score_batch_size=score_batch_size,
			)
			image_to_text, coverage_i2t = compute_ranking_metrics(
				query_caches["image_query"]["embeddings"][variant].float(),
				text_candidates,
				relevance["image_to_text"],
				device=device,
				score_batch_size=score_batch_size,
			)
			results[variant] = {
				"aggregate": {
					metric: (text_to_image[metric] + image_to_text[metric]) / 2.0
					for metric in REQUIRED_RANKING_METRICS
				},
				"text_to_image": text_to_image,
				"image_to_text": image_to_text,
				"coverage_percent": min(coverage_t2i, coverage_i2t),
			}
		return results
	answer_store = ImmutableCandidateStore(
		candidate_root=candidate_root,
		spec=CandidateBankSpec(dataset, "shared", "answer"),
		model_checkpoint_sha256=model_hash,
		validate_checksums=True,
	)
	answer_candidates = answer_store.embeddings.float()
	results = {}
	for variant in variant_names:
		metrics, coverage = compute_ranking_metrics(
			query_caches["query"]["embeddings"][variant].float(),
			answer_candidates,
			relevance["answer"],
			device=device,
			score_batch_size=score_batch_size,
		)
		results[variant] = {
			**metrics,
			"answer_accuracy": metrics["p_at_1"],
			"coverage_percent": coverage,
			"answer_gallery_size": relevance["gallery_size"],
		}
	return results


def run_evaluation(args: argparse.Namespace) -> dict[str, Any] | None:
	rank, world_size, device = _initialize_distributed(args.expected_world_size)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	output_dir = Path(args.output_dir)
	if rank == 0:
		if output_dir.exists():
			raise FileExistsError(f"Evaluation output already exists: {output_dir}")
		(output_dir / "embedding_cache").mkdir(parents=True)
		_write_json(output_dir / "status.json", {"status": "initializing"})
	dist.barrier()
	head, train_manifest, recurrent_hash = _load_recurrent_head(
		Path(args.recurrent_checkpoint),
		device=device,
	)
	if train_manifest["dataset"] != args.dataset:
		raise ValueError("Recurrent checkpoint dataset does not match evaluation dataset")
	model_root = Path(args.model_root)
	base_hash_before = checkpoint_sha256(model_root / "model.safetensors") if rank == 0 else None
	hash_values = [base_hash_before]
	dist.broadcast_object_list(hash_values, src=0)
	base_hash_before = str(hash_values[0])
	if train_manifest["base_checkpoint_sha256_before"] != base_hash_before:
		raise ValueError("Recurrent checkpoint and Qwen base checksum do not match")
	rows = _load_rows(Path(args.dataset_root), args.max_test_rows)
	groups, relevance = _build_groups(args.dataset, Path(args.dataset_root), rows)
	processor = BaselineInputProcessor.from_pretrained(
		model_root,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	base_model = load_frozen_evaluation_model(
		model_root,
		dtype=torch.float16,
		attention_implementation=args.attention_implementation,
	).to(device)
	backbone = FrozenQueryBackbone(base_model)
	torch.cuda.reset_peak_memory_stats(device)
	start = time.perf_counter()
	for name in _query_group_names(args.dataset):
		_encode_query_group(
			name=name,
			items=groups[name],
			backbone=backbone,
			head=head,
			processor=processor,
			args=args,
			rank=rank,
			world_size=world_size,
			device=device,
			output_dir=output_dir,
		)
	dist.barrier()
	report = None
	if rank == 0:
		variant_names = _variant_names(head.config.max_recurrent_steps)
		query_caches = {
			name: _combine_query_cache(
				name,
				item_count=len(groups[name]),
				world_size=world_size,
				output_dir=output_dir,
				variant_names=variant_names,
			)
			for name in _query_group_names(args.dataset)
		}
		metrics = _score_variants(
			dataset=args.dataset,
			query_caches=query_caches,
			relevance=relevance,
			candidate_root=Path(args.candidate_root),
			model_hash=base_hash_before,
			device=device,
			score_batch_size=args.score_batch_size,
			variant_names=variant_names,
		)
		baseline_metrics = metrics["pass_0_frozen_backbone"]
		improvements = {
			variant: _metric_deltas(values, baseline_metrics)
			for variant, values in metrics.items()
			if variant != "pass_0_frozen_backbone"
		}
		base_hash_after = checkpoint_sha256(model_root / "model.safetensors")
		if base_hash_after != base_hash_before:
			raise RuntimeError("Immutable Qwen checkpoint changed during recurrent evaluation")
		primary_variant = f"pass_{head.config.max_recurrent_steps}"
		report = {
			"status": "passed",
			"scope": "single_dataset_query_only_recurrent_test",
			"dataset": args.dataset,
			"metric_scale": METRIC_SCALE,
			"primary_variant": primary_variant,
			"metrics_by_recurrent_pass": metrics,
			"improvement_over_frozen_qwen_by_recurrent_pass": improvements,
			"protocol": {
				"split": "test",
				"test_rows": len(rows),
				"candidate_source": "immutable_preencoded_candidate_bank",
				"candidate_qwen_forward_calls": 0,
				"query_qwen_forward_calls_per_item": 1,
				"recurrent_passes_executed_per_item": head.config.max_recurrent_steps,
				"exit_policy": "fixed_pass_count",
				"validation_used": False,
				"score": "dot_product_of_unit_normalized_embeddings",
				"retrieval_cutoffs": RETRIEVAL_CUTOFFS,
				"ndcg_cutoff": 10,
			},
			"model": {
				"architecture": head.config.identity(),
				"trainable_parameter_count": head.trainable_parameter_count,
				"lora_enabled": False,
				"recurrent_checkpoint": str(args.recurrent_checkpoint),
				"recurrent_checkpoint_sha256": recurrent_hash,
				"base_checkpoint_sha256_before": base_hash_before,
				"base_checkpoint_sha256_after": base_hash_after,
				"runtime_precision": "frozen_qwen_fp16_recurrent_fp32",
			},
			"distributed": {
				"hostname": socket.gethostname(),
				"world_size": world_size,
				"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
			},
			"runtime_seconds": time.perf_counter() - start,
			"rank_zero_peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
		}
		_write_json(output_dir / "report.json", report)
		_write_json(output_dir / "status.json", {"status": "passed"})
	dist.barrier()
	dist.destroy_process_group()
	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset", choices=BASELINE_DATASETS, required=True)
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--candidate-root", type=Path, required=True)
	parser.add_argument("--model-root", type=Path, required=True)
	parser.add_argument("--recurrent-checkpoint", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=8)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--score-batch-size", type=int, default=256)
	parser.add_argument("--log-every-batches", type=int, default=20)
	parser.add_argument("--max-test-rows", type=int, default=0)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument("--visual-length-buckets", type=int, default=DEFAULT_VISUAL_LENGTH_BUCKETS)
	parser.add_argument(
		"--min-visual-bucket-size",
		type=int,
		default=DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	)
	parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_evaluation(args)
		return 0
	except Exception:
		logging.basicConfig(level=logging.INFO)
		LOGGER.exception("Query recurrent evaluation failed")
		if int(os.environ.get("RANK", "0")) == 0 and Path(args.output_dir).exists():
			_write_json(Path(args.output_dir) / "status.json", {"status": "failed"})
		if dist.is_available() and dist.is_initialized():
			dist.destroy_process_group()
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
