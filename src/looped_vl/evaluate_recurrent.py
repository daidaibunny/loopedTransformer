"""Evaluate one trained recurrent model on one full held-out test split."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from looped_vl.evaluate_frozen import (
	COCO_IMAGE_TO_TEXT_INSTRUCTION,
	COCO_TEXT_TO_IMAGE_INSTRUCTION,
	METRIC_SCALE,
	RETRIEVAL_CUTOFFS,
	VQA_INSTRUCTION,
	DistributedEncodingDataset,
	EncodingItem,
	_close_batch_images,
	aggregate_coco_directions,
	build_answer_gallery,
	build_coco_relevance,
	compute_ranking_metrics,
	encoding_collate,
)
from looped_vl.models.config import (
	RecurrentModelConfig,
	pure_recurrent_result_identity,
)
from looped_vl.models.loading import load_recurrent_components
from looped_vl.recurrent_data import RecurrentAlignedDataset, load_aligned_records
from looped_vl.runtime import (
	ATTENTION_IMPLEMENTATIONS,
	RUNTIME_PRECISIONS,
	resolve_attention_implementation,
	resolve_torch_dtype,
)
from looped_vl.smoke import checkpoint_sha256
from looped_vl.throughput import validate_embeddings

INFERENCE_PARAMETER_PREFIXES = (
	"latent_slots",
	"eos_delta",
	"late_fusion.",
)


def _initialize_evaluation_distributed(
	expected_world_size: int,
) -> tuple[int, int, int, torch.device]:
	"""Use CPU collectives so rank-zero scoring never creates a GPU barrier timeout."""
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	device = torch.device("cuda", local_rank)
	dist.init_process_group(backend="gloo")
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(f"Expected {expected_world_size} ranks, found {world_size}")
	return rank, world_size, local_rank, device


def _summarize_evaluation_runtime(
	*,
	runtimes: list[dict[str, Any]],
	total_encoded_items: int,
	total_seconds: float,
) -> dict[str, float | int]:
	"""Return wall throughput and the maximum exact allocated-memory peak."""
	if not runtimes or total_encoded_items <= 0 or total_seconds <= 0:
		raise ValueError("Runtime summary inputs must be non-empty and positive")
	encoding_wall_seconds = max(float(item["encoding_seconds"]) for item in runtimes)
	if encoding_wall_seconds <= 0:
		raise ValueError("Encoding wall time must be positive")
	return {
		"total_seconds": total_seconds,
		"encoding_wall_seconds": encoding_wall_seconds,
		"encoded_items": total_encoded_items,
		"encoding_items_per_second": total_encoded_items / encoding_wall_seconds,
		"peak_gpu_memory_bytes": max(
			int(item["peak_gpu_memory_bytes"]) for item in runtimes
		),
	}


def _primary_final_pass_metrics(
	*,
	source: str,
	loop_metrics: dict[str, Any],
	final_pass: int,
) -> dict[str, float]:
	"""Select the one primary metric row used by the cross-model result table."""
	final = loop_metrics[str(final_pass)]
	if source == "coco":
		return dict(final["aggregate"]["metrics"])
	return dict(final["metrics"])


def build_recurrent_improvement_summary(
	*,
	source: str,
	loop_metrics: dict[str, Any],
) -> dict[str, Any]:
	"""Map every pass to its completed recurrent-update count and primary mAP gain."""
	rows = []
	for pass_key in sorted(loop_metrics, key=int):
		pass_number = int(pass_key)
		pass_metrics = loop_metrics[pass_key]
		primary = pass_metrics["aggregate"] if source == "coco" else pass_metrics
		rows.append(
			{
				"pass_number": pass_number,
				"recurrent_updates": pass_number - 1,
				"map": primary["metrics"]["map"],
				"delta_from_previous_percentage_points": primary[
					"delta_from_previous_percentage_points"
				]["map"],
				"delta_from_pass_1_percentage_points": primary[
					"delta_from_r1_percentage_points"
				]["map"],
			},
		)
	return {
		"primary_metric": "map",
		"reference_pass": 1,
		"rows": rows,
	}


def _is_inference_parameter(name: str) -> bool:
	return name in INFERENCE_PARAMETER_PREFIXES[:2] or name.startswith(
		INFERENCE_PARAMETER_PREFIXES[2:],
	)


def load_recurrent_inference_checkpoint(
	model: nn.Module,
	path: str | Path,
	*,
	expected_base_hash: str,
	expected_model_config: dict[str, Any],
) -> dict[str, Any]:
	"""Strictly restore inference parameters while rejecting incompatible checkpoints."""
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if payload.get("format_version") != 1:
		raise ValueError("Unsupported recurrent checkpoint format version")
	metadata = payload.get("metadata")
	if not isinstance(metadata, dict):
		raise ValueError("Recurrent checkpoint metadata is missing")
	if metadata.get("model_checkpoint_sha256") != expected_base_hash:
		raise ValueError("Recurrent checkpoint base checkpoint hash does not match")
	if metadata.get("model_config") != expected_model_config:
		raise ValueError("Recurrent checkpoint model configuration does not match")
	state = payload.get("trainable_parameter_state")
	if not isinstance(state, dict):
		raise ValueError("Recurrent checkpoint parameter state is missing")
	if any("lora_" in str(name).lower() for name in state):
		raise ValueError("Damped recurrent checkpoints must not contain LoRA parameters")
	if any("recurrent_connector" in str(name) for name in state):
		raise ValueError("Damped recurrent checkpoints must not contain a recurrent connector")
	expected_identity = pure_recurrent_result_identity()
	if any(metadata.get(key) != value for key, value in expected_identity.items()):
		raise ValueError("Checkpoint does not declare the required damped recurrent identity")
	model_parameters = dict(model.named_parameters())
	expected_names = {
		name for name in model_parameters if _is_inference_parameter(name)
	}
	checkpoint_inference_state = {
		name.removeprefix("encoder."): value
		for name, value in state.items()
		if _is_inference_parameter(name.removeprefix("encoder."))
	}
	missing = expected_names - checkpoint_inference_state.keys()
	extra = checkpoint_inference_state.keys() - expected_names
	if missing:
		raise ValueError(f"Missing inference parameters: {sorted(missing)}")
	if extra:
		raise ValueError(f"Unexpected inference parameters: {sorted(extra)}")
	for name in sorted(expected_names):
		parameter = model_parameters[name]
		value = checkpoint_inference_state[name]
		if tuple(value.shape) != tuple(parameter.shape):
			raise ValueError(
				f"Inference parameter shape mismatch for {name}: "
				f"{tuple(value.shape)} != {tuple(parameter.shape)}",
			)
		parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
	return metadata


def build_loop_metric_series(
	metrics_by_pass: dict[int, dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
	"""Attach previous-pass and pass-1 percentage-point changes to every metric set."""
	if not metrics_by_pass or min(metrics_by_pass) != 1:
		raise ValueError("Loop metrics must begin at pass 1")
	pass_numbers = sorted(metrics_by_pass)
	if pass_numbers != list(range(1, max(pass_numbers) + 1)):
		raise ValueError("Loop metric passes must be contiguous")
	metric_names = set(metrics_by_pass[1])
	if any(set(metrics) != metric_names for metrics in metrics_by_pass.values()):
		raise ValueError("Every loop pass must report the same metrics")
	result: dict[str, dict[str, dict[str, float]]] = {}
	for pass_number in pass_numbers:
		metrics = metrics_by_pass[pass_number]
		previous = metrics_by_pass[max(1, pass_number - 1)]
		first = metrics_by_pass[1]
		result[str(pass_number)] = {
			"metrics": dict(metrics),
			"delta_from_previous_percentage_points": {
				name: metrics[name] - previous[name] for name in sorted(metric_names)
			},
			"delta_from_r1_percentage_points": {
				name: metrics[name] - first[name] for name in sorted(metric_names)
			},
		}
	return result


def _load_rows(
	dataset_root: Path,
	split: str,
	max_rows: int,
) -> list[dict[str, Any]]:
	return load_aligned_records(dataset_root, split, max_rows)


def _build_groups(
	rows: list[dict[str, Any]],
	dataset: RecurrentAlignedDataset,
	source: str,
) -> tuple[dict[str, list[EncodingItem]], dict[str, Any]]:
	if not rows:
		raise ValueError("Evaluation split is empty")
	found_sources = {str(row["source"]) for row in rows}
	if found_sources != {source}:
		raise ValueError(
			f"Expected only source {source}, found {sorted(found_sources)}",
		)
	if source == "coco":
		relevance = build_coco_relevance(rows)
		image_by_id: dict[str, dict[str, Any]] = {}
		for row in rows:
			image_by_id.setdefault(str(row["image_id"]), row)
		image_rows = [image_by_id[image_id] for image_id in relevance["image_ids"]]
		return {
			"text_query": [
				EncodingItem(
					item_id=str(row["sample_id"]),
					text=str(row["text"]),
					instruction=COCO_TEXT_TO_IMAGE_INSTRUCTION,
				)
				for row in rows
			],
			"image_target": [
				EncodingItem(
					item_id=str(row["image_id"]),
					image_path=dataset.resolve_image_path(row),
				)
				for row in image_rows
			],
			"image_query": [
				EncodingItem(
					item_id=str(row["image_id"]),
					image_path=dataset.resolve_image_path(row),
					instruction=COCO_IMAGE_TO_TEXT_INSTRUCTION,
				)
				for row in image_rows
			],
			"text_target": [
				EncodingItem(item_id=str(row["sample_id"]), text=str(row["text"]))
				for row in rows
			],
		}, relevance
	if source not in {"gqa_balanced", "clevr"}:
		raise ValueError(f"Unsupported evaluation source: {source}")
	answers, positive_indices = build_answer_gallery([str(row["answer"]) for row in rows])
	return {
		"query": [
			EncodingItem(
				item_id=str(row["sample_id"]),
				text=str(row["text"]),
				image_path=dataset.resolve_image_path(row),
				instruction=VQA_INSTRUCTION,
			)
			for row in rows
		],
		"answer_target": [
			EncodingItem(item_id=f"{source}:answer:{index}", text=answer)
			for index, answer in enumerate(answers)
		],
	}, {"answers": answers, "positive_indices": positive_indices}


def _encode_group(
	*,
	name: str,
	items: list[EncodingItem],
	model: nn.Module,
	processor: Any,
	args: argparse.Namespace,
	rank: int,
	world_size: int,
	device: torch.device,
	output_dir: Path,
) -> dict[str, Any]:
	global_indices = list(range(rank, len(items), world_size))
	dataset = DistributedEncodingDataset(items, global_indices)
	loader_options: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.batch_size,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": encoding_collate,
		"pin_memory": True,
	}
	if args.num_workers:
		loader_options.update(
			{
				"multiprocessing_context": "spawn",
				"persistent_workers": True,
				"prefetch_factor": args.prefetch_factor,
			},
		)
	loader = DataLoader(**loader_options)
	index_chunks: list[torch.Tensor] = []
	embedding_chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
	start = time.perf_counter()
	processed_items = 0
	for batch_number, batch in enumerate(loader, start=1):
		try:
			processed_inputs = processor.prepare(
				batch["model_inputs"],
				device=torch.device("cuda"),
			)
			with torch.inference_mode():
				output = model(
					**processed_inputs,
					return_all_loop_embeddings=True,
				)
			if output.loop_embeddings is None:
				raise RuntimeError("Recurrent model did not return per-loop embeddings")
			expected_passes = model.config.num_total_loop_passes
			if len(output.loop_embeddings) != expected_passes:
				raise RuntimeError(
					"Recurrent model returned "
					f"{len(output.loop_embeddings)} loop embeddings; "
					f"expected {expected_passes}",
				)
			if not torch.equal(output.embeddings, output.loop_embeddings[-1]):
				raise RuntimeError(
					"Final retrieval embedding does not equal the last loop-pass embedding",
				)
			for pass_number, embeddings in enumerate(output.loop_embeddings, start=1):
				validate_embeddings(embeddings, len(batch["global_indices"]))
				embedding_chunks[pass_number].append(embeddings.float().cpu())
			index_chunks.append(torch.tensor(batch["global_indices"], dtype=torch.long))
			processed_items += len(batch["global_indices"])
		finally:
			_close_batch_images(batch["model_inputs"])
		if batch_number == 1 or batch_number % args.log_every_batches == 0:
			progress = {
				"status": "encoding",
				"rank": rank,
				"group": name,
				"processed": processed_items,
				"rank_items": len(global_indices),
				"elapsed_seconds": time.perf_counter() - start,
			}
			(output_dir / f"progress_rank{rank}.json").write_text(
				json.dumps(progress, indent=2, sort_keys=True) + "\n",
				encoding="utf-8",
			)
	torch.cuda.synchronize(device)
	seconds = time.perf_counter() - start
	indices = torch.cat(index_chunks) if index_chunks else torch.empty(0, dtype=torch.long)
	for pass_number, chunks in embedding_chunks.items():
		torch.save(
			{
				"indices": indices,
				"embeddings": torch.cat(chunks),
			},
			output_dir / "embedding_cache" / f"{name}.pass{pass_number}.rank{rank}.pt",
		)
	return {
		"rank": rank,
		"group": name,
		"items": len(global_indices),
		"seconds": seconds,
	}


def _load_embeddings(
	*,
	name: str,
	pass_number: int,
	item_count: int,
	world_size: int,
	output_dir: Path,
) -> torch.Tensor:
	result: torch.Tensor | None = None
	seen = torch.zeros(item_count, dtype=torch.bool)
	for rank in range(world_size):
		shard = torch.load(
			output_dir / "embedding_cache" / (
				f"{name}.pass{pass_number}.rank{rank}.pt"
			),
			map_location="cpu",
			weights_only=True,
		)
		indices = shard["indices"]
		embeddings = shard["embeddings"]
		if result is None:
			result = torch.empty((item_count, embeddings.shape[1]))
		if seen[indices].any():
			raise ValueError(f"Duplicate encoded indices for {name} pass {pass_number}")
		result[indices] = embeddings
		seen[indices] = True
	if result is None or not seen.all():
		raise ValueError(f"Missing encoded indices for {name} pass {pass_number}")
	return result


def _score_passes(
	*,
	source: str,
	groups: dict[str, list[EncodingItem]],
	relevance: dict[str, Any],
	num_passes: int,
	world_size: int,
	output_dir: Path,
	device: torch.device,
	score_batch_size: int,
) -> dict[str, Any]:
	if source == "coco":
		text_to_image: dict[int, dict[str, float]] = {}
		image_to_text: dict[int, dict[str, float]] = {}
		aggregate: dict[int, dict[str, float]] = {}
		for pass_number in range(1, num_passes + 1):
			text_to_image[pass_number] = compute_ranking_metrics(
				_load_embeddings(
					name="text_query",
					pass_number=pass_number,
					item_count=len(groups["text_query"]),
					world_size=world_size,
					output_dir=output_dir,
				),
				_load_embeddings(
					name="image_target",
					pass_number=pass_number,
					item_count=len(groups["image_target"]),
					world_size=world_size,
					output_dir=output_dir,
				),
				relevance["text_to_image_positive_indices"],
				device,
				score_batch_size,
			)
			image_to_text[pass_number] = compute_ranking_metrics(
				_load_embeddings(
					name="image_query",
					pass_number=pass_number,
					item_count=len(groups["image_query"]),
					world_size=world_size,
					output_dir=output_dir,
				),
				_load_embeddings(
					name="text_target",
					pass_number=pass_number,
					item_count=len(groups["text_target"]),
					world_size=world_size,
					output_dir=output_dir,
				),
				relevance["image_to_text_positive_indices"],
				device,
				score_batch_size,
			)
			aggregate[pass_number] = aggregate_coco_directions(
				text_to_image[pass_number],
				image_to_text[pass_number],
			)
		direction_series = {
			"aggregate": build_loop_metric_series(aggregate),
			"text_to_image": build_loop_metric_series(text_to_image),
			"image_to_text": build_loop_metric_series(image_to_text),
		}
		return {
			str(pass_number): {
				name: series[str(pass_number)]
				for name, series in direction_series.items()
			}
			for pass_number in range(1, num_passes + 1)
		}
	metrics_by_pass = {
		pass_number: compute_ranking_metrics(
			_load_embeddings(
				name="query",
				pass_number=pass_number,
				item_count=len(groups["query"]),
				world_size=world_size,
				output_dir=output_dir,
			),
			_load_embeddings(
				name="answer_target",
				pass_number=pass_number,
				item_count=len(groups["answer_target"]),
				world_size=world_size,
				output_dir=output_dir,
			),
			relevance["positive_indices"],
			device,
			score_batch_size,
		)
		for pass_number in range(1, num_passes + 1)
	}
	return build_loop_metric_series(metrics_by_pass)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any] | None:
	"""Run distributed source-pure recurrent evaluation."""
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU fallback is disabled")
	rank, world_size, local_rank, device = _initialize_evaluation_distributed(
		args.expected_world_size,
	)
	output_dir = Path(args.output_dir)
	if rank == 0:
		if output_dir.exists():
			raise FileExistsError(f"Output directory already exists: {output_dir}")
		(output_dir / "embedding_cache").mkdir(parents=True)
	dist.barrier()
	evaluation_start = time.perf_counter()
	resolved_attention = resolve_attention_implementation(args.attention_implementation)
	runtime_dtype = resolve_torch_dtype(args.runtime_precision)
	model_config = RecurrentModelConfig.from_yaml(args.model_config)
	dataset_root = Path(args.dataset_root)
	dataset = RecurrentAlignedDataset(dataset_root, args.split)
	rows = _load_rows(dataset_root, args.split, args.max_rows)
	groups, relevance = _build_groups(rows, dataset, args.source)
	base_checkpoint_path = Path(args.model_root) / "model.safetensors"
	base_hash_objects: list[str | None] = [
		checkpoint_sha256(base_checkpoint_path) if rank == 0 else None,
	]
	dist.broadcast_object_list(base_hash_objects, src=0)
	base_hash = base_hash_objects[0]
	if base_hash is None:
		raise RuntimeError("Failed to broadcast the base checkpoint hash")
	model_load_start = time.perf_counter()
	components = load_recurrent_components(
		model_root=args.model_root,
		master_slot_path=args.master_slot_path,
		config=model_config,
		device=device,
		dtype=runtime_dtype,
		attention_implementation=resolved_attention,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	metadata = load_recurrent_inference_checkpoint(
		components.model,
		args.checkpoint,
		expected_base_hash=base_hash,
		expected_model_config=model_config.__dict__,
	)
	components.model.eval()
	model_load_seconds = time.perf_counter() - model_load_start
	inference_parameter_count = sum(
		parameter.numel()
		for name, parameter in components.model.named_parameters()
		if _is_inference_parameter(name)
	)
	torch.cuda.reset_peak_memory_stats(device)
	encoding_start = time.perf_counter()
	runtime_rows = []
	for name, items in groups.items():
		runtime_rows.append(
			_encode_group(
				name=name,
				items=items,
				model=components.model,
				processor=components.processor,
				args=args,
				rank=rank,
				world_size=world_size,
				device=device,
				output_dir=output_dir,
			),
		)
	torch.cuda.synchronize(device)
	encoding_seconds = time.perf_counter() - encoding_start
	local_runtime = {
		"rank": rank,
		"logical_device": local_rank,
		"device_name": torch.cuda.get_device_name(local_rank),
		"model_load_seconds": model_load_seconds,
		"encoding_seconds": encoding_seconds,
		"encoded_items": sum(int(row["items"]) for row in runtime_rows),
		"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
		"groups": runtime_rows,
	}
	gathered_runtimes: list[dict[str, Any] | None] = [
		None for _ in range(world_size)
	]
	dist.all_gather_object(gathered_runtimes, local_runtime)
	dist.barrier()
	report = None
	if rank == 0:
		loop_metrics = _score_passes(
			source=args.source,
			groups=groups,
			relevance=relevance,
			num_passes=model_config.num_total_loop_passes,
			world_size=world_size,
			output_dir=output_dir,
			device=device,
			score_batch_size=args.score_batch_size,
		)
		primary_metrics = _primary_final_pass_metrics(
			source=args.source,
			loop_metrics=loop_metrics,
			final_pass=model_config.num_total_loop_passes,
		)
		runtimes = [item for item in gathered_runtimes if item is not None]
		total_encoded_items = sum(len(items) for items in groups.values())
		runtime = _summarize_evaluation_runtime(
			runtimes=runtimes,
			total_encoded_items=total_encoded_items,
			total_seconds=time.perf_counter() - evaluation_start,
		)
		report = {
			"status": "passed",
			"scope": "single_dataset_recurrent_test",
			"source": args.source,
			"metric_scale": METRIC_SCALE,
			"metrics": primary_metrics,
			"loop_passes": loop_metrics,
			"recurrent_improvement_summary": build_recurrent_improvement_summary(
				source=args.source,
				loop_metrics=loop_metrics,
			),
			"protocol": {
				"dataset_root": str(dataset_root),
				"split": args.split,
				"sample_rows": len(rows),
				"group_sizes": {name: len(items) for name, items in groups.items()},
				"retrieval_cutoffs": RETRIEVAL_CUTOFFS,
				"ndcg_cutoff": 10,
			},
			"model": {
				**pure_recurrent_result_identity(),
				"model_root": str(args.model_root),
				"checkpoint": str(args.checkpoint),
				"base_checkpoint_sha256": base_hash,
				"checkpoint_metadata": metadata,
				"inference_parameter_count": inference_parameter_count,
				"runtime_precision": args.runtime_precision,
				"requested_attention_implementation": args.attention_implementation,
				"resolved_attention_implementation": resolved_attention,
			},
			"distributed": {
				"hostname": socket.gethostname(),
				"backend": "gloo",
				"world_size": world_size,
				"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
				"ranks": runtimes,
			},
			"runtime": runtime,
		}
		(output_dir / "report.json").write_text(
			json.dumps(report, indent=2, sort_keys=True) + "\n",
			encoding="utf-8",
		)
		(output_dir / "status.json").write_text(
			json.dumps({"status": "passed"}, indent=2) + "\n",
			encoding="utf-8",
		)
	dist.barrier()
	dist.destroy_process_group()
	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source", choices=("coco", "gqa_balanced", "clevr"), required=True)
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--model-root", type=Path, required=True)
	parser.add_argument("--master-slot-path", type=Path, required=True)
	parser.add_argument("--model-config", type=Path, default=Path("configs/base.yaml"))
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--split", choices=("test",), default="test")
	parser.add_argument("--expected-world-size", type=int, required=True)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--score-batch-size", type=int, default=256)
	parser.add_argument("--log-every-batches", type=int, default=10)
	parser.add_argument("--max-rows", type=int, default=0)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument(
		"--attention-implementation",
		choices=ATTENTION_IMPLEMENTATIONS,
		default="auto",
	)
	parser.add_argument(
		"--runtime-precision",
		choices=RUNTIME_PRECISIONS,
		default="fp16",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_evaluation(args)
		return 0
	except Exception as error:
		output_dir = Path(args.output_dir)
		if output_dir.is_dir():
			(output_dir / "status.json").write_text(
				json.dumps({"status": "failed", "error": repr(error)}, indent=2) + "\n",
				encoding="utf-8",
			)
		raise


if __name__ == "__main__":
	raise SystemExit(main())
