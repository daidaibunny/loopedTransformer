"""Two-GPU full-test evaluation for the frozen Qwen3-VL-Embedding-2B baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from looped_vl.data import DEFAULT_DATASET_ROOT, LoopedVLMixtureDataset
from looped_vl.metrics import (
	METRIC_SCALE,
	REQUIRED_RANKING_METRICS,
	aggregate_coco_directions,
	aggregate_mixture_metrics,
	validate_evaluation_report,
)
from looped_vl.smoke import (
	assert_model_frozen,
	checkpoint_sha256,
	freeze_model,
	load_local_embedding_module,
)
from looped_vl.throughput import validate_embeddings

LOGGER = logging.getLogger("evaluate_frozen")
RETRIEVAL_CUTOFFS = (1, 5, 10, 20)
COCO_TEXT_TO_IMAGE_INSTRUCTION = "Retrieve the image that best matches the caption."
COCO_IMAGE_TO_TEXT_INSTRUCTION = "Retrieve the caption that best describes the image."
VQA_INSTRUCTION = "Retrieve the correct answer to the visual question."


@dataclass(frozen=True)
class EncodingItem:
	"""One text, image, or image-text input to encode."""

	item_id: str
	text: str | None = None
	image_path: Path | None = None
	instruction: str | None = None


class DistributedEncodingDataset(Dataset[dict[str, Any]]):
	"""Decode only the items assigned to one distributed rank."""

	def __init__(self, items: list[EncodingItem], global_indices: list[int]) -> None:
		self.items = items
		self.global_indices = global_indices

	def __len__(self) -> int:
		return len(self.global_indices)

	def __getitem__(self, local_index: int) -> dict[str, Any]:
		global_index = self.global_indices[local_index]
		item = self.items[global_index]
		model_input: dict[str, Any] = {}
		if item.text is not None:
			model_input["text"] = item.text
		if item.image_path is not None:
			with Image.open(item.image_path) as source_image:
				image = source_image.convert("RGB")
				image.load()
			model_input["image"] = image
		if item.instruction is not None:
			model_input["instruction"] = item.instruction
		return {
			"global_index": global_index,
			"model_input": model_input,
		}


def encoding_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
	"""Keep multimodal processor inputs as a list and preserve global indexes."""
	return {
		"global_indices": [sample["global_index"] for sample in samples],
		"model_inputs": [sample["model_input"] for sample in samples],
	}


def normalize_answer(answer: str) -> str:
	"""Normalize whitespace and case for a closed answer gallery."""
	return " ".join(answer.strip().lower().split())


def build_answer_gallery(answers: list[str]) -> tuple[list[str], list[tuple[int, ...]]]:
	"""Build a sorted unique answer gallery and one positive index per query."""
	normalized_answers = [normalize_answer(answer) for answer in answers]
	if any(not answer for answer in normalized_answers):
		raise ValueError("Answer gallery contains an empty normalized answer")
	gallery = sorted(set(normalized_answers))
	answer_to_index = {answer: index for index, answer in enumerate(gallery)}
	positive_indices = [(answer_to_index[answer],) for answer in normalized_answers]
	return gallery, positive_indices


def build_coco_relevance(rows: list[dict[str, Any]]) -> dict[str, Any]:
	"""Build COCO galleries and positives for both retrieval directions."""
	image_ids: list[str] = []
	image_to_index: dict[str, int] = {}
	image_to_caption_indices: dict[str, list[int]] = defaultdict(list)
	text_to_image_positive_indices: list[tuple[int, ...]] = []
	for caption_index, row in enumerate(rows):
		image_id = str(row["image_id"])
		if image_id not in image_to_index:
			image_to_index[image_id] = len(image_ids)
			image_ids.append(image_id)
		text_to_image_positive_indices.append((image_to_index[image_id],))
		image_to_caption_indices[image_id].append(caption_index)
	return {
		"image_ids": image_ids,
		"text_to_image_positive_indices": text_to_image_positive_indices,
		"image_to_text_positive_indices": [
			tuple(image_to_caption_indices[image_id]) for image_id in image_ids
		],
	}


def compute_ranking_metrics(
	query_embeddings: torch.Tensor,
	target_embeddings: torch.Tensor,
	positive_indices: list[tuple[int, ...]],
	device: torch.device,
	score_batch_size: int,
) -> dict[str, float]:
	"""Compute exact mAP, precision, recall, MRR, and nDCG@10 from full rankings."""
	if query_embeddings.ndim != 2 or target_embeddings.ndim != 2:
		raise ValueError("Query and target embeddings must both be rank-two tensors")
	if query_embeddings.shape[1] != target_embeddings.shape[1]:
		raise ValueError("Query and target embedding dimensions do not match")
	if query_embeddings.shape[0] != len(positive_indices):
		raise ValueError("Every query must have one positive-index tuple")
	if target_embeddings.shape[0] < max(RETRIEVAL_CUTOFFS):
		raise ValueError("Ranking gallery must contain at least 20 targets")
	if score_batch_size <= 0:
		raise ValueError("score_batch_size must be positive")

	target_count = target_embeddings.shape[0]
	for query_positives in positive_indices:
		if not query_positives:
			raise ValueError("Every query must have at least one positive target")
		if len(set(query_positives)) != len(query_positives):
			raise ValueError("Positive target indexes must be unique per query")
		if min(query_positives) < 0 or max(query_positives) >= target_count:
			raise ValueError("Positive target index is outside the gallery")

	target_device = target_embeddings.to(device=device, dtype=torch.float32)
	metric_sums = {metric: 0.0 for metric in REQUIRED_RANKING_METRICS}
	rank_positions = torch.arange(1, target_count + 1, device=device, dtype=torch.float64)
	discounts = 1.0 / torch.log2(
		torch.arange(2, 12, device=device, dtype=torch.float64),
	)
	query_count = query_embeddings.shape[0]
	for start in range(0, query_count, score_batch_size):
		end = min(start + score_batch_size, query_count)
		query_device = query_embeddings[start:end].to(device=device, dtype=torch.float32)
		scores = query_device @ target_device.T
		order = torch.argsort(scores, dim=1, descending=True, stable=True)
		relevance = torch.zeros(
			(end - start, target_count),
			dtype=torch.bool,
			device=device,
		)
		for local_index, query_positives in enumerate(positive_indices[start:end]):
			relevance[local_index, list(query_positives)] = True
		sorted_relevance = torch.gather(relevance, 1, order).to(torch.float64)
		positive_counts = sorted_relevance.sum(dim=1)
		cumulative_relevance = sorted_relevance.cumsum(dim=1)

		for cutoff in RETRIEVAL_CUTOFFS:
			retrieved = sorted_relevance[:, :cutoff].sum(dim=1)
			metric_sums[f"p_at_{cutoff}"] += float((retrieved / cutoff).sum().item())
			metric_sums[f"r_at_{cutoff}"] += float(
				(retrieved / positive_counts).sum().item(),
			)

		precision_by_rank = cumulative_relevance / rank_positions
		average_precision = (
			(precision_by_rank * sorted_relevance).sum(dim=1) / positive_counts
		)
		metric_sums["map"] += float(average_precision.sum().item())
		first_positive_rank = sorted_relevance.argmax(dim=1).to(torch.float64) + 1.0
		metric_sums["mrr"] += float((1.0 / first_positive_rank).sum().item())

		dcg = (sorted_relevance[:, :10] * discounts).sum(dim=1)
		ideal_lengths = torch.minimum(
			positive_counts.to(torch.long),
			torch.tensor(10, device=device),
		)
		ideal_dcg = torch.stack(
			[discounts[: int(length.item())].sum() for length in ideal_lengths],
		)
		metric_sums["ndcg_at_10"] += float((dcg / ideal_dcg).sum().item())

	return {
		metric: 100.0 * metric_sums[metric] / query_count
		for metric in REQUIRED_RANKING_METRICS
	}


def _load_test_rows(dataset_root: Path, max_test_rows: int) -> list[dict[str, Any]]:
	"""Load the ordered test manifest and optionally take a smoke prefix."""
	paths = sorted((dataset_root / "test").glob("*.parquet"))
	if not paths:
		raise FileNotFoundError(f"No test Parquet files under {dataset_root}")
	tables = [pq.read_table(path) for path in paths]
	table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
	if max_test_rows:
		if max_test_rows % 20:
			raise ValueError("max_test_rows must be divisible by 20 to preserve 10:7:3")
		if max_test_rows > table.num_rows:
			raise ValueError("max_test_rows exceeds the test split")
		table = table.slice(0, max_test_rows)
	return table.to_pylist()


def _build_encoding_groups(
	rows: list[dict[str, Any]],
	dataset: LoopedVLMixtureDataset,
) -> tuple[dict[str, list[EncodingItem]], dict[str, Any]]:
	"""Build every query/target encoding group and its exact relevance mapping."""
	rows_by_source = {
		source: [row for row in rows if row["source"] == source]
		for source in ("coco", "gqa_balanced", "clevr")
	}
	if any(not source_rows for source_rows in rows_by_source.values()):
		raise ValueError("Test split must contain COCO, GQA Balanced, and CLEVR")

	coco_rows = rows_by_source["coco"]
	coco_relevance = build_coco_relevance(coco_rows)
	coco_first_row_by_image: dict[str, dict[str, Any]] = {}
	for row in coco_rows:
		coco_first_row_by_image.setdefault(str(row["image_id"]), row)
	coco_image_rows = [
		coco_first_row_by_image[image_id] for image_id in coco_relevance["image_ids"]
	]

	groups: dict[str, list[EncodingItem]] = {
		"coco_text_query": [
			EncodingItem(
				item_id=str(row["sample_id"]),
				text=str(row["text"]),
				instruction=COCO_TEXT_TO_IMAGE_INSTRUCTION,
			)
			for row in coco_rows
		],
		"coco_image_target": [
			EncodingItem(
				item_id=str(row["image_id"]),
				image_path=dataset.resolve_image_path(row),
			)
			for row in coco_image_rows
		],
		"coco_image_query": [
			EncodingItem(
				item_id=str(row["image_id"]),
				image_path=dataset.resolve_image_path(row),
				instruction=COCO_IMAGE_TO_TEXT_INSTRUCTION,
			)
			for row in coco_image_rows
		],
		"coco_text_target": [
			EncodingItem(item_id=str(row["sample_id"]), text=str(row["text"]))
			for row in coco_rows
		],
	}

	relevance: dict[str, Any] = {
		"coco": coco_relevance,
		"source_rows": {source: len(source_rows) for source, source_rows in rows_by_source.items()},
	}
	for source in ("gqa_balanced", "clevr"):
		source_rows = rows_by_source[source]
		answers, answer_positive_indices = build_answer_gallery(
			[str(row["answer"]) for row in source_rows],
		)
		groups[f"{source}_query"] = [
			EncodingItem(
				item_id=str(row["sample_id"]),
				text=str(row["text"]),
				image_path=dataset.resolve_image_path(row),
				instruction=VQA_INSTRUCTION,
			)
			for row in source_rows
		]
		groups[f"{source}_answer_target"] = [
			EncodingItem(item_id=f"{source}:answer:{index}", text=answer)
			for index, answer in enumerate(answers)
		]
		relevance[source] = {
			"answers": answers,
			"positive_indices": answer_positive_indices,
		}
	return groups, relevance


def _write_json(path: Path, value: Any) -> None:
	"""Write one UTF-8 JSON status or result file."""
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _close_batch_images(model_inputs: list[dict[str, Any]]) -> None:
	"""Release decoded PIL images after the processor has consumed them."""
	for model_input in model_inputs:
		image = model_input.get("image")
		if isinstance(image, Image.Image):
			image.close()


def _encode_group(
	name: str,
	items: list[EncodingItem],
	embedder: Any,
	args: argparse.Namespace,
	rank: int,
	world_size: int,
	output_dir: Path,
) -> None:
	"""Encode one deterministic distributed shard and save indexes with embeddings."""
	global_indices = list(range(rank, len(items), world_size))
	dataset = DistributedEncodingDataset(items, global_indices)
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.batch_size,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": encoding_collate,
		"pin_memory": False,
	}
	if args.num_workers > 0:
		loader_kwargs["prefetch_factor"] = args.prefetch_factor
	loader = DataLoader(**loader_kwargs)
	index_chunks: list[torch.Tensor] = []
	embedding_chunks: list[torch.Tensor] = []
	group_start = time.perf_counter()
	processed = 0
	for batch_number, batch in enumerate(loader, start=1):
		try:
			with torch.inference_mode():
				embeddings = embedder.process(batch["model_inputs"])
			torch.cuda.synchronize()
			validate_embeddings(embeddings, len(batch["global_indices"]))
			index_chunks.append(torch.tensor(batch["global_indices"], dtype=torch.long))
			embedding_chunks.append(embeddings.float().cpu())
			processed += len(batch["global_indices"])
		finally:
			_close_batch_images(batch["model_inputs"])

		if batch_number == 1 or batch_number % args.log_every_batches == 0:
			elapsed = time.perf_counter() - group_start
			LOGGER.info(
				"rank=%d group=%s processed=%d/%d rate=%.2f items/s",
				rank,
				name,
				processed,
				len(global_indices),
				processed / elapsed,
			)
			_write_json(
				output_dir / f"progress_rank{rank}.json",
				{
					"status": "running",
					"rank": rank,
					"group": name,
					"processed": processed,
					"rank_group_items": len(global_indices),
					"global_group_items": len(items),
					"elapsed_seconds": elapsed,
					"items_per_second": processed / elapsed,
					"gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
				},
			)

	indices = torch.cat(index_chunks) if index_chunks else torch.empty(0, dtype=torch.long)
	embeddings = (
		torch.cat(embedding_chunks)
		if embedding_chunks
		else torch.empty((0, 2048), dtype=torch.float32)
	)
	torch.save(
		{"indices": indices, "embeddings": embeddings},
		output_dir / "embedding_cache" / f"{name}.rank{rank}.pt",
	)
	LOGGER.info("rank=%d completed group=%s items=%d", rank, name, processed)


def _load_combined_embeddings(
	name: str,
	item_count: int,
	world_size: int,
	output_dir: Path,
) -> torch.Tensor:
	"""Restore a distributed embedding group in its original global order."""
	combined: torch.Tensor | None = None
	seen = torch.zeros(item_count, dtype=torch.bool)
	for rank in range(world_size):
		shard = torch.load(
			output_dir / "embedding_cache" / f"{name}.rank{rank}.pt",
			map_location="cpu",
			weights_only=True,
		)
		indices = shard["indices"]
		embeddings = shard["embeddings"]
		if combined is None:
			combined = torch.empty((item_count, embeddings.shape[1]), dtype=torch.float32)
		if seen[indices].any():
			raise RuntimeError(f"Duplicate distributed embedding indexes for {name}")
		combined[indices] = embeddings
		seen[indices] = True
	if combined is None or not seen.all():
		raise RuntimeError(f"Missing distributed embedding indexes for {name}")
	return combined


def _compute_report_metrics(
	groups: dict[str, list[EncodingItem]],
	relevance: dict[str, Any],
	args: argparse.Namespace,
	world_size: int,
	output_dir: Path,
	device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Load distributed embedding caches and compute all frozen metric-contract values."""
	coco_text_to_image = compute_ranking_metrics(
		_load_combined_embeddings(
			"coco_text_query",
			len(groups["coco_text_query"]),
			world_size,
			output_dir,
		),
		_load_combined_embeddings(
			"coco_image_target",
			len(groups["coco_image_target"]),
			world_size,
			output_dir,
		),
		relevance["coco"]["text_to_image_positive_indices"],
		device,
		args.score_batch_size,
	)
	coco_image_to_text = compute_ranking_metrics(
		_load_combined_embeddings(
			"coco_image_query",
			len(groups["coco_image_query"]),
			world_size,
			output_dir,
		),
		_load_combined_embeddings(
			"coco_text_target",
			len(groups["coco_text_target"]),
			world_size,
			output_dir,
		),
		relevance["coco"]["image_to_text_positive_indices"],
		device,
		args.score_batch_size,
	)
	coco_aggregate = aggregate_coco_directions(coco_text_to_image, coco_image_to_text)

	dataset_metrics: dict[str, dict[str, float]] = {}
	for source in ("gqa_balanced", "clevr"):
		dataset_metrics[source] = compute_ranking_metrics(
			_load_combined_embeddings(
				f"{source}_query",
				len(groups[f"{source}_query"]),
				world_size,
				output_dir,
			),
			_load_combined_embeddings(
				f"{source}_answer_target",
				len(groups[f"{source}_answer_target"]),
				world_size,
				output_dir,
			),
			relevance[source]["positive_indices"],
			device,
			args.score_batch_size,
		)

	mix = aggregate_mixture_metrics(
		coco_aggregate,
		dataset_metrics["gqa_balanced"],
		dataset_metrics["clevr"],
	)
	report_metrics = {
		"metric_scale": METRIC_SCALE,
		"mix": mix,
		"datasets": {
			"coco": {
				"aggregate": coco_aggregate,
				"text_to_image": coco_text_to_image,
				"image_to_text": coco_image_to_text,
			},
			"gqa_balanced": dataset_metrics["gqa_balanced"],
			"clevr": dataset_metrics["clevr"],
		},
	}
	validate_evaluation_report(report_metrics)
	gallery_statistics = {
		"coco_caption_queries": len(groups["coco_text_query"]),
		"coco_unique_image_targets": len(groups["coco_image_target"]),
		"coco_unique_image_queries": len(groups["coco_image_query"]),
		"coco_caption_targets": len(groups["coco_text_target"]),
		"gqa_queries": len(groups["gqa_balanced_query"]),
		"gqa_unique_answer_targets": len(groups["gqa_balanced_answer_target"]),
		"clevr_queries": len(groups["clevr_query"]),
		"clevr_unique_answer_targets": len(groups["clevr_answer_target"]),
	}
	return report_metrics, gallery_statistics


def _initialize_distributed(
	expected_world_size: int,
) -> tuple[int, int, int, torch.device]:
	"""Bind the local CUDA device before initializing the NCCL process group."""
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	device = torch.device("cuda", local_rank)
	dist.init_process_group(backend="nccl", device_id=device)
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(
			f"Expected {expected_world_size} distributed ranks, found {world_size}",
		)
	return rank, world_size, local_rank, device


def run_distributed_evaluation(args: argparse.Namespace) -> dict[str, Any] | None:
	"""Run the frozen evaluation under torchrun and return the report on rank zero."""
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU fallback is disabled")
	rank, world_size, local_rank, device = _initialize_distributed(
		args.expected_world_size,
	)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	torch.manual_seed(args.seed + rank)
	torch.cuda.manual_seed_all(args.seed + rank)

	output_dir = Path(args.output_dir)
	if rank == 0:
		if output_dir.exists():
			raise FileExistsError(f"Output directory already exists: {output_dir}")
		(output_dir / "embedding_cache").mkdir(parents=True)
		_write_json(
			output_dir / "status.json",
			{"status": "initializing", "world_size": world_size},
		)
	dist.barrier()

	evaluation_start = time.perf_counter()
	dataset_root = Path(args.dataset_root)
	rows = _load_test_rows(dataset_root, args.max_test_rows)
	dataset = LoopedVLMixtureDataset(dataset_root, "test", args.gqa_materialized_root)
	groups, relevance = _build_encoding_groups(rows, dataset)
	group_sizes = {name: len(items) for name, items in groups.items()}
	if rank == 0:
		_write_json(
			output_dir / "protocol.json",
			{
				"dataset_root": str(dataset_root),
				"test_rows": len(rows),
				"source_rows": relevance["source_rows"],
				"encoding_group_sizes": group_sizes,
				"metric_scale": METRIC_SCALE,
				"required_metrics": REQUIRED_RANKING_METRICS,
			},
		)

	model_root = Path(args.model_root)
	checkpoint_path = model_root / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(checkpoint_path) if rank == 0 else None
	module = load_local_embedding_module(model_root)
	model_load_start = time.perf_counter()
	embedder = module.Qwen3VLEmbedder(
		model_name_or_path=str(model_root),
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
		torch_dtype=torch.bfloat16,
		attn_implementation=args.attention_implementation,
	)
	freeze_model(embedder.model)
	assert_model_frozen(embedder.model)
	model_load_seconds = time.perf_counter() - model_load_start
	parameter_count = sum(parameter.numel() for parameter in embedder.model.parameters())
	trainable_parameter_count = sum(
		parameter.numel()
		for parameter in embedder.model.parameters()
		if parameter.requires_grad
	)
	torch.cuda.reset_peak_memory_stats()
	LOGGER.info(
		"model loaded device=%s parameters=%d trainable=%d seconds=%.2f",
		device,
		parameter_count,
		trainable_parameter_count,
		model_load_seconds,
	)
	_write_json(
		output_dir / f"progress_rank{rank}.json",
		{
			"status": "model_loaded",
			"rank": rank,
			"device": str(device),
			"device_name": torch.cuda.get_device_name(local_rank),
			"model_load_seconds": model_load_seconds,
			"trainable_parameter_count": trainable_parameter_count,
		},
	)
	dist.barrier()
	if rank == 0:
		_write_json(
			output_dir / "status.json",
			{"status": "encoding", "world_size": world_size, "group_sizes": group_sizes},
		)

	encoding_start = time.perf_counter()
	for name, items in groups.items():
		_encode_group(name, items, embedder, args, rank, world_size, output_dir)
	encoding_seconds = time.perf_counter() - encoding_start
	assert_model_frozen(embedder.model)
	peak_memory = torch.cuda.max_memory_allocated()
	local_runtime = {
		"rank": rank,
		"logical_device": local_rank,
		"device_name": torch.cuda.get_device_name(local_rank),
		"model_load_seconds": model_load_seconds,
		"encoding_seconds": encoding_seconds,
		"peak_gpu_memory_bytes": peak_memory,
		"encoded_items": sum(len(range(rank, size, world_size)) for size in group_sizes.values()),
	}
	runtimes: list[dict[str, Any] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(runtimes, local_runtime)
	dist.barrier()

	report: dict[str, Any] | None = None
	if rank == 0:
		_write_json(output_dir / "status.json", {"status": "scoring"})
		metric_values, gallery_statistics = _compute_report_metrics(
			groups,
			relevance,
			args,
			world_size,
			output_dir,
			device,
		)
		checkpoint_hash_after = checkpoint_sha256(checkpoint_path)
		if checkpoint_hash_after != checkpoint_hash_before:
			raise RuntimeError("Model checkpoint hash changed during evaluation")
		total_seconds = time.perf_counter() - evaluation_start
		report = {
			"status": "passed",
			"scope": "frozen_full_test_no_backward_no_optimizer",
			**metric_values,
			"protocol": {
				"dataset_root": str(dataset_root),
				"split": "test",
				"test_rows": len(rows),
				"source_rows": relevance["source_rows"],
				"gallery_statistics": gallery_statistics,
				"positive_definition": {
					"coco_text_to_image": "matching image_id",
					"coco_image_to_text": "all test captions sharing image_id",
					"gqa_balanced": "normalized canonical answer text",
					"clevr": "normalized canonical answer text",
				},
				"instructions": {
					"coco_text_to_image": COCO_TEXT_TO_IMAGE_INSTRUCTION,
					"coco_image_to_text": COCO_IMAGE_TO_TEXT_INSTRUCTION,
					"visual_question_answering": VQA_INSTRUCTION,
				},
				"retrieval_cutoffs": RETRIEVAL_CUTOFFS,
				"ndcg_cutoff": 10,
				"score": "dot_product_of_unit_normalized_embeddings",
			},
			"model": {
				"model_root": str(model_root),
				"parameter_count": parameter_count,
				"trainable_parameter_count": trainable_parameter_count,
				"precision": "bfloat16",
				"max_length": args.max_length,
				"min_pixels": args.min_pixels,
				"max_pixels": args.max_pixels,
				"attention_implementation": args.attention_implementation,
				"checkpoint_sha256_before": checkpoint_hash_before,
				"checkpoint_sha256_after": checkpoint_hash_after,
			},
			"distributed": {
				"hostname": socket.gethostname(),
				"backend": "nccl",
				"world_size": world_size,
				"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
				"ranks": runtimes,
			},
			"runtime": {
				"total_seconds": total_seconds,
				"encoding_wall_seconds": max(
					float(runtime["encoding_seconds"])
					for runtime in runtimes
					if runtime is not None
				),
			},
		}
		validate_evaluation_report(report)
		_write_json(output_dir / "report.json", report)
		_write_json(
			output_dir / "status.json",
			{"status": "passed", "report": str(output_dir / "report.json")},
		)
		LOGGER.info("evaluation completed report=%s", output_dir / "report.json")

	dist.barrier()
	_write_json(
		output_dir / f"progress_rank{rank}.json",
		{"status": "passed", "rank": rank},
	)
	dist.destroy_process_group()
	return report


def parse_args() -> argparse.Namespace:
	"""Parse the frozen distributed evaluation configuration."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
	parser.add_argument(
		"--gqa-materialized-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced"),
	)
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=2)
	parser.add_argument("--batch-size", type=int, default=20)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--score-batch-size", type=int, default=256)
	parser.add_argument("--log-every-batches", type=int, default=10)
	parser.add_argument("--max-test-rows", type=int, default=0)
	parser.add_argument("--seed", type=int, default=20260729)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument(
		"--attention-implementation",
		choices=("flash_attention_2", "sdpa", "eager"),
		default="flash_attention_2",
	)
	return parser.parse_args()


def main() -> int:
	"""Run the two-GPU frozen evaluator."""
	args = parse_args()
	try:
		run_distributed_evaluation(args)
		return 0
	except Exception:
		logging.basicConfig(level=logging.INFO)
		LOGGER.exception("Frozen distributed test evaluation failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
