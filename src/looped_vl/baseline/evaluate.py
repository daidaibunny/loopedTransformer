"""Distributed held-out retrieval evaluation for frozen Qwen or one LoRA adapter."""

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

from looped_vl.baseline.bucketing import (
	DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	DEFAULT_VISUAL_LENGTH_BUCKETS,
	group_baseline_model_inputs,
)
from looped_vl.baseline.data import (
	BASELINE_DATASETS,
	COCO_IMAGE_TO_TEXT_INSTRUCTION,
	COCO_TEXT_TO_IMAGE_INSTRUCTION,
	VQA_INSTRUCTION,
)
from looped_vl.baseline.model import (
	BaselineInputProcessor,
	describe_lora_decoder_scope,
	encode_grouped_baseline_batches,
	load_frozen_evaluation_model,
	load_lora_evaluation_model,
)
from looped_vl.candidate_bank import CandidateBankSpec, sha256_file
from looped_vl.metrics import METRIC_SCALE, REQUIRED_RANKING_METRICS
from looped_vl.query_recurrent.candidate_store import ImmutableCandidateStore
from looped_vl.smoke import checkpoint_sha256

LOGGER = logging.getLogger("baseline_evaluate")
RETRIEVAL_CUTOFFS = (1, 5, 10, 20)


@dataclass(frozen=True)
class EvaluationItem:
	"""One independently encoded query or target."""

	item_id: str
	text: str | None = None
	image_path: Path | None = None
	instruction: str | None = None


class EvaluationDataset(Dataset[dict[str, Any]]):
	"""Decode only the evaluation items assigned to this rank."""

	def __init__(self, items: list[EvaluationItem], indices: list[int]) -> None:
		self.items = items
		self.indices = indices

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, index: int) -> dict[str, Any]:
		global_index = self.indices[index]
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
		return {"global_index": global_index, "model_input": model_input}


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
	return {
		"global_indices": [sample["global_index"] for sample in samples],
		"model_inputs": [sample["model_input"] for sample in samples],
	}


def _close_images(model_inputs: list[dict[str, Any]]) -> None:
	for model_input in model_inputs:
		image = model_input.get("image")
		if isinstance(image, Image.Image):
			image.close()


def _load_rows(dataset_root: Path, max_test_rows: int) -> list[dict[str, Any]]:
	paths = sorted((dataset_root / "test").glob("*.parquet"))
	if not paths:
		raise FileNotFoundError(f"No baseline test manifests under {dataset_root}")
	tables = [pq.read_table(path) for path in paths]
	table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
	if max_test_rows:
		table = table.slice(0, min(max_test_rows, table.num_rows))
	return table.to_pylist()


def _build_groups(
	dataset: str,
	dataset_root: Path,
	rows: list[dict[str, Any]],
) -> tuple[dict[str, list[EvaluationItem]], dict[str, Any]]:
	if dataset == "coco":
		image_ids: list[str] = []
		first_row_by_image: dict[str, dict[str, Any]] = {}
		image_to_index: dict[str, int] = {}
		image_to_caption_indices: dict[str, list[int]] = defaultdict(list)
		text_to_image: list[tuple[int, ...]] = []
		for caption_index, row in enumerate(rows):
			image_id = str(row["image_id"])
			if image_id not in image_to_index:
				image_to_index[image_id] = len(image_ids)
				image_ids.append(image_id)
				first_row_by_image[image_id] = row
			text_to_image.append((image_to_index[image_id],))
			image_to_caption_indices[image_id].append(caption_index)
		image_rows = [first_row_by_image[image_id] for image_id in image_ids]
		groups = {
			"text_query": [
				EvaluationItem(
					item_id=str(row["sample_id"]),
					text=str(row["query_text"]),
					instruction=COCO_TEXT_TO_IMAGE_INSTRUCTION,
				)
				for row in rows
			],
			"image_target": [
				EvaluationItem(
					item_id=image_id,
					image_path=Path(row["image_path"]),
				)
				for image_id, row in zip(image_ids, image_rows, strict=True)
			],
			"image_query": [
				EvaluationItem(
					item_id=image_id,
					image_path=Path(row["image_path"]),
					instruction=COCO_IMAGE_TO_TEXT_INSTRUCTION,
				)
				for image_id, row in zip(image_ids, image_rows, strict=True)
			],
			"text_target": [
				EvaluationItem(item_id=str(row["sample_id"]), text=str(row["query_text"]))
				for row in rows
			],
		}
		relevance = {
			"text_to_image": text_to_image,
			"image_to_text": [
				tuple(image_to_caption_indices[image_id]) for image_id in image_ids
			],
		}
		return groups, relevance

	gallery_path = dataset_root / "answer_gallery.json"
	gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
	positive_to_index = {
		str(item["positive_id"]): index for index, item in enumerate(gallery)
	}
	groups = {
		"query": [
			EvaluationItem(
				item_id=str(row["sample_id"]),
				text=str(row["query_text"]),
				image_path=Path(row["image_path"]),
				instruction=VQA_INSTRUCTION,
			)
			for row in rows
		],
		"answer_target": [
			EvaluationItem(
				item_id=str(item["positive_id"]),
				text=str(item["text"]),
			)
			for item in gallery
		],
	}
	relevance = {
		"answer": [
			((positive_to_index[str(row["positive_id"])],)
			if str(row["positive_id"]) in positive_to_index
			else ())
			for row in rows
		],
		"gallery_size": len(gallery),
	}
	return groups, relevance


def _evaluation_group_names(dataset: str, *, query_only: bool) -> tuple[str, ...]:
	"""Select query groups only when candidates come from immutable banks."""
	if dataset == "coco":
		return (
			("text_query", "image_query")
			if query_only
			else ("text_query", "image_target", "image_query", "text_target")
		)
	return ("query",) if query_only else ("query", "answer_target")


def _validate_candidate_store_order(
	store: ImmutableCandidateStore,
	items: list[EvaluationItem],
) -> None:
	"""Require immutable candidates to match the exact held-out gallery order."""
	expected_item_ids = tuple(item.item_id for item in items)
	if store.item_ids != expected_item_ids:
		raise ValueError(f"Candidate ordering mismatch for {store.spec.key}")


def _load_query_only_candidate_embeddings(
	*,
	dataset: str,
	groups: dict[str, list[EvaluationItem]],
	candidate_root: Path,
	model_checkpoint_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
	"""Load exact test candidates without executing the Qwen candidate tower."""
	if dataset == "coco":
		specs_by_group = {
			"image_target": CandidateBankSpec("coco", "test", "image"),
			"text_target": CandidateBankSpec("coco", "test", "text"),
		}
	else:
		specs_by_group = {
			"answer_target": CandidateBankSpec(dataset, "shared", "answer"),
		}
	embeddings = {}
	manifest_hashes = {}
	for group_name, spec in specs_by_group.items():
		store = ImmutableCandidateStore(
			candidate_root=candidate_root,
			spec=spec,
			model_checkpoint_sha256=model_checkpoint_sha256,
			validate_checksums=True,
		)
		_validate_candidate_store_order(store, groups[group_name])
		embeddings[group_name] = store.embeddings.float()
		manifest_hashes[spec.key] = sha256_file(store.root / "bank_manifest.json")
	return embeddings, manifest_hashes


def _encode_group(
	*,
	name: str,
	items: list[EvaluationItem],
	model: torch.nn.Module,
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
	index_chunks: list[torch.Tensor] = []
	embedding_chunks: list[torch.Tensor] = []
	start = time.perf_counter()
	processed_count = 0
	for batch_number, batch in enumerate(loader, start=1):
		try:
			input_groups = group_baseline_model_inputs(
				batch["model_inputs"],
				min_pixels=args.min_pixels,
				max_pixels=args.max_pixels,
				max_visual_buckets=args.visual_length_buckets,
				min_visual_bucket_size=args.min_visual_bucket_size,
			)
			processed_batches = tuple(
				processor.prepare(list(group.model_inputs), device=device)
				for group in input_groups
			)
		finally:
			_close_images(batch["model_inputs"])
		with torch.inference_mode(), torch.autocast(
			device_type="cuda",
			dtype=torch.float16,
		):
			embeddings = encode_grouped_baseline_batches(
				model=model,
				processed_batches=processed_batches,
				original_indices=tuple(
					group.original_indices for group in input_groups
				),
				total_rows=len(batch["model_inputs"]),
			)
		if not torch.isfinite(embeddings).all():
			raise RuntimeError(f"Non-finite embeddings in {name}")
		index_chunks.append(torch.tensor(batch["global_indices"], dtype=torch.long))
		embedding_chunks.append(embeddings.cpu())
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
			"embeddings": (
				torch.cat(embedding_chunks)
				if embedding_chunks
				else torch.empty((0, 2048), dtype=torch.float32)
			),
		},
		output_dir / "embedding_cache" / f"{name}.rank{rank}.pt",
	)


def _combine_embeddings(
	name: str,
	item_count: int,
	world_size: int,
	output_dir: Path,
) -> torch.Tensor:
	combined = torch.empty((item_count, 2048), dtype=torch.float32)
	seen = torch.zeros(item_count, dtype=torch.bool)
	for rank in range(world_size):
		shard = torch.load(
			output_dir / "embedding_cache" / f"{name}.rank{rank}.pt",
			map_location="cpu",
			weights_only=True,
		)
		indices = shard["indices"]
		if seen[indices].any():
			raise RuntimeError(f"Duplicate distributed indexes for {name}")
		combined[indices] = shard["embeddings"].float()
		seen[indices] = True
	if not seen.all():
		raise RuntimeError(f"Missing distributed indexes for {name}")
	return combined


def compute_ranking_metrics(
	query_embeddings: torch.Tensor,
	target_embeddings: torch.Tensor,
	positive_indices: list[tuple[int, ...]],
	*,
	device: torch.device,
	score_batch_size: int,
) -> tuple[dict[str, float], float]:
	"""Compute exact rankings while counting out-of-training-vocabulary answers as zero."""
	if query_embeddings.shape[0] != len(positive_indices):
		raise ValueError("Every query requires one positive-index tuple, possibly empty")
	if target_embeddings.shape[0] < max(RETRIEVAL_CUTOFFS):
		raise ValueError("Target gallery must contain at least 20 items")
	target_device = target_embeddings.to(device=device, dtype=torch.float32)
	sums = {metric: 0.0 for metric in REQUIRED_RANKING_METRICS}
	covered = 0
	query_count = query_embeddings.shape[0]
	ranks = torch.arange(
		1,
		target_embeddings.shape[0] + 1,
		device=device,
		dtype=torch.float64,
	)
	discounts = 1.0 / torch.log2(torch.arange(2, 12, device=device, dtype=torch.float64))
	for start in range(0, query_count, score_batch_size):
		end = min(start + score_batch_size, query_count)
		scores = query_embeddings[start:end].to(device=device) @ target_device.T
		order = torch.argsort(scores, dim=1, descending=True, stable=True)
		relevance = torch.zeros_like(scores, dtype=torch.bool)
		covered_mask = torch.zeros(end - start, dtype=torch.bool, device=device)
		for local_index, positives in enumerate(positive_indices[start:end]):
			if positives:
				relevance[local_index, list(positives)] = True
				covered_mask[local_index] = True
		covered += int(covered_mask.sum().item())
		sorted_relevance = torch.gather(relevance, 1, order).to(torch.float64)
		positive_counts = sorted_relevance.sum(dim=1)
		safe_positive_counts = positive_counts.clamp_min(1.0)
		cumulative = sorted_relevance.cumsum(dim=1)
		for cutoff in RETRIEVAL_CUTOFFS:
			retrieved = sorted_relevance[:, :cutoff].sum(dim=1)
			sums[f"p_at_{cutoff}"] += float((retrieved / cutoff).sum().item())
			sums[f"r_at_{cutoff}"] += float(
				(retrieved / safe_positive_counts).sum().item(),
			)
		average_precision = (
			(cumulative / ranks * sorted_relevance).sum(dim=1) / safe_positive_counts
		)
		sums["map"] += float(average_precision.sum().item())
		first_rank = sorted_relevance.argmax(dim=1).to(torch.float64) + 1.0
		reciprocal_rank = torch.where(
			covered_mask,
			1.0 / first_rank,
			torch.zeros_like(first_rank),
		)
		sums["mrr"] += float(reciprocal_rank.sum().item())
		dcg = (sorted_relevance[:, :10] * discounts).sum(dim=1)
		ideal_lengths = torch.minimum(
			positive_counts.to(torch.long),
			torch.tensor(10, device=device),
		)
		ideal_dcg = torch.stack(
			[
				discounts[: int(length.item())].sum() if length else discounts.new_tensor(1.0)
				for length in ideal_lengths
			],
		)
		sums["ndcg_at_10"] += float((dcg / ideal_dcg).sum().item())
	metrics = {
		metric: 100.0 * sums[metric] / query_count
		for metric in REQUIRED_RANKING_METRICS
	}
	return metrics, 100.0 * covered / query_count


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _initialize_evaluation_distributed(
	expected_world_size: int,
) -> tuple[int, int, torch.device]:
	"""Use CPU collectives so waiting ranks never occupy GPUs during rank-zero scoring."""
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="gloo")
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(f"Expected {expected_world_size} ranks, found {world_size}")
	return rank, world_size, torch.device("cuda", local_rank)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any] | None:
	if args.visual_length_buckets <= 0:
		raise ValueError("visual_length_buckets must be positive")
	if args.min_visual_bucket_size <= 0:
		raise ValueError("min_visual_bucket_size must be positive")
	query_only = args.candidate_root is not None
	if query_only and args.adapter_root is None:
		raise ValueError("Query-only LoRA evaluation requires --adapter-root")
	rank, world_size, device = _initialize_evaluation_distributed(
		args.expected_world_size,
	)
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
	rows = _load_rows(Path(args.dataset_root), args.max_test_rows)
	groups, relevance = _build_groups(args.dataset, Path(args.dataset_root), rows)
	model_root = Path(args.model_root)
	adapter_root = Path(args.adapter_root) if args.adapter_root is not None else None
	model_variant = "lora" if adapter_root is not None else "frozen_base"
	lora_decoder_scope: dict[str, Any] | None = None
	base_hash_before = checkpoint_sha256(model_root / "model.safetensors") if rank == 0 else None
	adapter_hash = (
		checkpoint_sha256(adapter_root / "adapter_model.safetensors")
		if rank == 0 and adapter_root is not None
		else None
	)
	processor = BaselineInputProcessor.from_pretrained(
		model_root,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	if adapter_root is None:
		model = load_frozen_evaluation_model(
			model_root,
			dtype=torch.float16,
			attention_implementation=args.attention_implementation,
		).to(device)
	else:
		model = load_lora_evaluation_model(
			model_root,
			adapter_root,
			dtype=torch.float16,
			attention_implementation=args.attention_implementation,
		).to(device)
		lora_decoder_scope = describe_lora_decoder_scope(
			model.peft_config["default"],
		)
		model_variant = f"lora_{lora_decoder_scope['scope']}"
		if query_only:
			if lora_decoder_scope["scope"] != "last_4_decoder_layers":
				raise ValueError("Query-only LoRA evaluation requires a last-four-layer adapter")
			model_variant = "query_only_lora_last_4_decoder_layers"
	trainable_parameter_count = sum(
		parameter.numel() for parameter in model.parameters() if parameter.requires_grad
	)
	if trainable_parameter_count:
		raise RuntimeError(
			f"Evaluation model unexpectedly has {trainable_parameter_count} trainable parameters",
		)
	torch.cuda.reset_peak_memory_stats(device)
	start = time.perf_counter()
	for name in _evaluation_group_names(args.dataset, query_only=query_only):
		_encode_group(
			name=name,
			items=groups[name],
			model=model,
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
		candidate_data = (
			_load_query_only_candidate_embeddings(
				dataset=args.dataset,
				groups=groups,
				candidate_root=Path(args.candidate_root),
				model_checkpoint_sha256=str(base_hash_before),
			)
			if query_only
			else ({}, {})
		)
		frozen_candidate_embeddings, candidate_bank_manifest_sha256 = candidate_data
		if args.dataset == "coco":
			text_to_image, coverage_t2i = compute_ranking_metrics(
				_combine_embeddings(
					"text_query",
					len(groups["text_query"]),
					world_size,
					output_dir,
				),
				(
					frozen_candidate_embeddings["image_target"]
					if query_only
					else _combine_embeddings(
						"image_target",
						len(groups["image_target"]),
						world_size,
						output_dir,
					)
				),
				relevance["text_to_image"],
				device=device,
				score_batch_size=args.score_batch_size,
			)
			image_to_text, coverage_i2t = compute_ranking_metrics(
				_combine_embeddings(
					"image_query",
					len(groups["image_query"]),
					world_size,
					output_dir,
				),
				(
					frozen_candidate_embeddings["text_target"]
					if query_only
					else _combine_embeddings(
						"text_target",
						len(groups["text_target"]),
						world_size,
						output_dir,
					)
				),
				relevance["image_to_text"],
				device=device,
				score_batch_size=args.score_batch_size,
			)
			aggregate = {
				metric: (text_to_image[metric] + image_to_text[metric]) / 2.0
				for metric in REQUIRED_RANKING_METRICS
			}
			dataset_metrics: dict[str, Any] = {
				"aggregate": aggregate,
				"text_to_image": text_to_image,
				"image_to_text": image_to_text,
				"coverage_percent": min(coverage_t2i, coverage_i2t),
			}
		else:
			metrics, coverage = compute_ranking_metrics(
				_combine_embeddings(
					"query",
					len(groups["query"]),
					world_size,
					output_dir,
				),
				(
					frozen_candidate_embeddings["answer_target"]
					if query_only
					else _combine_embeddings(
						"answer_target",
						len(groups["answer_target"]),
						world_size,
						output_dir,
					)
				),
				relevance["answer"],
				device=device,
				score_batch_size=args.score_batch_size,
			)
			dataset_metrics = {
				**metrics,
				"answer_accuracy": metrics["p_at_1"],
				"coverage_percent": coverage,
				"answer_gallery_size": relevance["gallery_size"],
			}
		base_hash_after = checkpoint_sha256(model_root / "model.safetensors")
		if base_hash_after != base_hash_before:
			raise RuntimeError("Immutable Qwen checkpoint changed during evaluation")
		report = {
			"status": "passed",
			"scope": f"single_dataset_{model_variant}_test",
			"dataset": args.dataset,
			"metric_scale": METRIC_SCALE,
			"metrics": dataset_metrics,
			"protocol": {
				"split": "test",
				"test_rows": len(rows),
				"candidate_gallery": (
					"test_split_images_or_captions"
					if args.dataset == "coco"
					else "normalized_answers_observed_in_training_only"
				),
				"score": "dot_product_of_unit_normalized_embeddings",
				"retrieval_cutoffs": RETRIEVAL_CUTOFFS,
				"ndcg_cutoff": 10,
				"candidate_source": (
					"immutable_preencoded_candidate_bank"
					if query_only
					else "online_active_qwen"
				),
				"candidate_qwen_forward_calls": 0 if query_only else "online",
				"candidate_bank_manifest_sha256": candidate_bank_manifest_sha256,
				"visual_length_bucketing": {
					"enabled": args.visual_length_buckets > 1,
					"maximum_buckets": args.visual_length_buckets,
					"minimum_bucket_size": args.min_visual_bucket_size,
					"length_measure": "post_smart_resize_visual_tokens",
					"candidate_gallery_unchanged": True,
				},
			},
			"model": {
				"variant": model_variant,
				"model_root": str(model_root),
				"adapter_root": str(adapter_root) if adapter_root is not None else None,
				"candidate_root": str(args.candidate_root) if query_only else None,
				"base_checkpoint_sha256_before": base_hash_before,
				"base_checkpoint_sha256_after": base_hash_after,
				"adapter_sha256": adapter_hash,
				"lora_decoder_scope": lora_decoder_scope,
				"trainable_parameter_count": trainable_parameter_count,
				"runtime_precision": "fp16",
				"attention_implementation": args.attention_implementation,
			},
			"distributed": {
				"hostname": socket.gethostname(),
				"world_size": world_size,
				"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
			},
			"runtime_seconds": time.perf_counter() - start,
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
	parser.add_argument("--model-root", type=Path, required=True)
	parser.add_argument(
		"--candidate-root",
		type=Path,
		help="Use immutable candidates for the separate query-only LoRA control.",
	)
	parser.add_argument(
		"--adapter-root",
		type=Path,
		help="Optional LoRA adapter. Omit it to evaluate the fully frozen base checkpoint.",
	)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=8)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--score-batch-size", type=int, default=256)
	parser.add_argument("--log-every-batches", type=int, default=20)
	parser.add_argument("--max-test-rows", type=int, default=0)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument(
		"--visual-length-buckets",
		type=int,
		default=DEFAULT_VISUAL_LENGTH_BUCKETS,
	)
	parser.add_argument(
		"--min-visual-bucket-size",
		type=int,
		default=DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	)
	parser.add_argument(
		"--attention-implementation",
		choices=("sdpa", "eager"),
		default="sdpa",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_evaluation(args)
		return 0
	except Exception:
		logging.basicConfig(level=logging.INFO)
		LOGGER.exception("Baseline LoRA evaluation failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
