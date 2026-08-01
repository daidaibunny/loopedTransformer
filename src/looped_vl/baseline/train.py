"""Eight-GPU LoRA contrastive training for one unmodified Qwen3-VL dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from looped_vl.baseline.bucketing import (
	DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	DEFAULT_VISUAL_LENGTH_BUCKETS,
	group_baseline_model_inputs,
)
from looped_vl.baseline.data import (
	BASELINE_DATASETS,
	BaselineManifestDataset,
	baseline_pair_collate,
	close_baseline_batch_images,
	count_coco_retrieval_directions,
)
from looped_vl.baseline.model import (
	BASELINE_LORA_ALPHA,
	BASELINE_LORA_RANK,
	BASELINE_LORA_TARGETS,
	BaselineInputProcessor,
	BaselineLoRATrainingModel,
	describe_lora_decoder_scope,
	load_lora_training_model,
)
from looped_vl.smoke import checkpoint_sha256
from looped_vl.training.checkpointing import (
	TrainingCursor,
	capture_rng_state,
	load_training_checkpoint,
	prepare_training_output_directory,
	publish_latest_training_checkpoint,
	save_training_checkpoint,
	truncate_metric_log,
	validate_checkpoint_metadata,
)
from looped_vl.training.schedule import (
	FORMAL_TRAINING_LOG_INTERVAL,
	BatchOffsetSampler,
	should_log_training_metrics,
)

LOGGER = logging.getLogger("baseline_train")


def _parse_decoder_layer_indices(value: str) -> tuple[int, ...]:
	"""Parse a sorted comma-separated decoder-layer selection."""
	try:
		indices = tuple(int(part) for part in value.split(","))
	except ValueError as error:
		raise argparse.ArgumentTypeError(
			"decoder layer indices must be comma-separated integers",
		) from error
	if not indices or any(index < 0 for index in indices):
		raise argparse.ArgumentTypeError("decoder layer indices must be non-negative")
	if tuple(sorted(set(indices))) != indices:
		raise argparse.ArgumentTypeError(
			"decoder layer indices must be sorted and unique",
		)
	return indices


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_json_line(path: Path, value: Any) -> None:
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, sort_keys=True) + "\n")


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, int, torch.device]:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU fallback is disabled")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(f"Expected {expected_world_size} ranks, found {world_size}")
	return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _seed_everything(seed: int, rank: int) -> torch.Generator:
	torch.manual_seed(seed + rank)
	torch.cuda.manual_seed_all(seed + rank)
	generator = torch.Generator()
	generator.manual_seed(seed + rank)
	return generator


def _validate_parallel_batch_sizes(
	*,
	per_device_batch_size: int,
	world_size: int,
	gradient_accumulation_steps: int,
	expected_contrastive_global_batch_size: int,
) -> None:
	"""Require the negative pool itself, not an optimizer accumulation, to reach target."""
	if per_device_batch_size <= 0 or gradient_accumulation_steps <= 0:
		raise ValueError("Batch size and gradient accumulation must be positive")
	if expected_contrastive_global_batch_size <= 0:
		raise ValueError("Expected contrastive global batch size must be positive")
	contrastive_global_batch_size = per_device_batch_size * world_size
	if contrastive_global_batch_size != expected_contrastive_global_batch_size:
		raise ValueError(
			"Contrastive global batch is "
			f"{contrastive_global_batch_size}, expected "
			f"{expected_contrastive_global_batch_size}; gradient accumulation does "
			"not add in-batch negatives",
		)


def _accumulate_logging_metrics(
	accumulator: dict[str, torch.Tensor],
	output: dict[str, Any],
	*,
	sample_count: int,
) -> None:
	"""Accumulate sample-weighted detached metrics across every microbatch."""
	if sample_count <= 0:
		raise ValueError("Metric sample count must be positive")
	for key in ("loss", "query_norm", "candidate_norm"):
		value = output[key].detach().float()
		accumulator[key] = (
			accumulator.get(key, torch.zeros_like(value)) + value * sample_count
		)


def _finalize_logging_metrics(
	accumulator: dict[str, torch.Tensor],
	*,
	sample_count: int,
) -> dict[str, float]:
	"""Return the actual sample-weighted metrics for one optimizer update."""
	if sample_count <= 0:
		raise ValueError("Metric sample count must be positive")
	return {
		key: float((value / sample_count).item())
		for key, value in accumulator.items()
	}


def _reduce_logging_metrics(
	accumulator: dict[str, torch.Tensor],
	*,
	local_sample_count: int,
) -> tuple[dict[str, torch.Tensor], int]:
	"""Sum logging numerators and sample counts across all training ranks."""
	if local_sample_count <= 0:
		raise ValueError("Metric sample count must be positive")
	reduced = {key: value.clone() for key, value in accumulator.items()}
	count = torch.tensor(
		local_sample_count,
		device=next(iter(reduced.values())).device,
		dtype=torch.long,
	)
	if dist.is_available() and dist.is_initialized():
		for value in reduced.values():
			dist.all_reduce(value, op=dist.ReduceOp.SUM)
		dist.all_reduce(count, op=dist.ReduceOp.SUM)
	return reduced, int(count.item())


def _gather_direction_counts(local_counts: Counter[str], world_size: int) -> Counter[str]:
	"""Combine exact direction counts instead of extrapolating rank zero."""
	gathered: list[dict[str, int] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(gathered, dict(local_counts))
	total: Counter[str] = Counter()
	for counts in gathered:
		if counts is None:
			raise RuntimeError("Failed to gather direction counts from every rank")
		total.update(counts)
	return total


def _gather_rank_rng_states(world_size: int) -> list[dict[str, Any]]:
	states: list[dict[str, Any] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(states, capture_rng_state())
	if any(state is None for state in states):
		raise RuntimeError("Failed to gather every rank RNG state")
	return [state for state in states if state is not None]


def _save_baseline_checkpoint(
	*,
	output_dir: Path,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	scaler: torch.cuda.amp.GradScaler,
	cursor: TrainingCursor,
	metadata: dict[str, Any],
	rank: int,
	world_size: int,
	max_checkpoints: int,
) -> Path:
	rank_rng_states = _gather_rank_rng_states(world_size)
	path = output_dir / "checkpoints" / f"step{cursor.global_step:06d}.pt"
	if rank == 0 and not path.exists():
		save_training_checkpoint(
			path=path,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			cursor=cursor,
			rank_rng_states=rank_rng_states,
			metadata=metadata,
			gradient_scaler=scaler,
		)
		publish_latest_training_checkpoint(
			path,
			cursor,
			max_checkpoints=max_checkpoints,
		)
	dist.barrier()
	return path


def _validate_epoch_count(epochs: int) -> None:
	"""Keep every formal baseline run to one complete dataset pass."""
	if epochs != 1:
		raise ValueError("Baseline training must use exactly one epoch")


def _build_loader(
	args: argparse.Namespace,
	*,
	rank: int,
	world_size: int,
	generator: torch.Generator,
) -> tuple[DataLoader[dict[str, Any]], BatchOffsetSampler]:
	dataset = BaselineManifestDataset(
		args.dataset_root,
		"train",
		max_rows=args.max_train_rows,
	)
	distributed_sampler = DistributedSampler(
		dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True,
		seed=args.seed,
		drop_last=False,
	)
	sampler = BatchOffsetSampler(distributed_sampler, args.per_device_batch_size)
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.per_device_batch_size,
		"sampler": sampler,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": baseline_pair_collate,
		"drop_last": False,
		"pin_memory": True,
		"generator": generator,
	}
	if args.num_workers:
		loader_kwargs.update(
			{
				"multiprocessing_context": "spawn",
				"persistent_workers": True,
				"prefetch_factor": args.prefetch_factor,
			},
		)
	return DataLoader(**loader_kwargs), sampler


def _build_optimizer(
	model: torch.nn.Module,
	args: argparse.Namespace,
) -> torch.optim.AdamW:
	parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not parameters:
		raise RuntimeError("LoRA training has no trainable parameters")
	kwargs = {
		"lr": args.learning_rate,
		"weight_decay": args.weight_decay,
		"betas": (0.9, 0.95),
		"eps": 1e-8,
	}
	try:
		return torch.optim.AdamW(parameters, fused=True, **kwargs)
	except (RuntimeError, TypeError):
		return torch.optim.AdamW(parameters, **kwargs)


def _build_scheduler(
	optimizer: torch.optim.Optimizer,
	*,
	total_steps: int,
	warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
	warmup_steps = max(1, round(total_steps * warmup_ratio))

	def multiplier(step: int) -> float:
		if step < warmup_steps:
			return float(step + 1) / warmup_steps
		progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
		return max(0.0, 1.0 - progress)

	return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _resolve_git_commit(project_root: Path) -> str:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.strip()


def run_training(args: argparse.Namespace) -> dict[str, Any] | None:
	if args.dataset not in BASELINE_DATASETS:
		raise ValueError(args.dataset)
	_validate_epoch_count(args.epochs)
	if args.checkpoint_every <= 0:
		raise ValueError("checkpoint_every must be positive")
	if args.max_checkpoints != 1:
		raise ValueError("max_checkpoints must be exactly 1")
	if args.initial_gradient_scale <= 0:
		raise ValueError("initial_gradient_scale must be positive")
	if args.visual_length_buckets <= 0:
		raise ValueError("visual_length_buckets must be positive")
	if args.min_visual_bucket_size <= 0:
		raise ValueError("min_visual_bucket_size must be positive")
	rank, world_size, local_rank, device = _initialize_distributed(
		args.expected_world_size,
	)
	_validate_parallel_batch_sizes(
		per_device_batch_size=args.per_device_batch_size,
		world_size=world_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		expected_contrastive_global_batch_size=(
			args.expected_contrastive_global_batch_size
		),
	)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	generator = _seed_everything(args.seed, rank)
	output_dir = Path(args.output_dir)
	resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
	if rank == 0:
		output_mode = prepare_training_output_directory(
			output_dir,
			resume_checkpoint=resume_checkpoint,
		)
		if output_mode == "fresh":
			_write_json(output_dir / "status.json", {"status": "initializing"})
		else:
			_write_json(
				output_dir / "status.json",
				{"status": "resuming", "checkpoint": str(resume_checkpoint)},
			)
	dist.barrier()

	model_root = Path(args.model_root)
	checkpoint_path = model_root / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(checkpoint_path) if rank == 0 else None
	checkpoint_hash_values = [checkpoint_hash_before]
	dist.broadcast_object_list(checkpoint_hash_values, src=0)
	checkpoint_hash_before = checkpoint_hash_values[0]
	processor = BaselineInputProcessor.from_pretrained(
		model_root,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	peft_model = load_lora_training_model(
		model_root,
		dtype=torch.float16,
		attention_implementation=args.attention_implementation,
		gradient_checkpointing=args.gradient_checkpointing,
		decoder_layer_indices=args.lora_decoder_layer_indices,
	).to(device)
	training_model = BaselineLoRATrainingModel(
		peft_model,
		temperature=args.temperature,
	)
	training_model.train()
	lora_scope = describe_lora_decoder_scope(peft_model.peft_config["default"])
	trainable_names = [
		name for name, parameter in training_model.named_parameters() if parameter.requires_grad
	]
	if not trainable_names or any("lora_" not in name for name in trainable_names):
		raise RuntimeError("Only LoRA parameters may be trainable in the baseline")
	loader, sampler = _build_loader(
		args,
		rank=rank,
		world_size=world_size,
		generator=generator,
	)
	steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
	full_total_steps = steps_per_epoch * args.epochs
	total_steps = (
		min(full_total_steps, args.max_optimizer_steps)
		if args.max_optimizer_steps
		else full_total_steps
	)
	optimizer = _build_optimizer(training_model, args)
	scheduler = _build_scheduler(
		optimizer,
		total_steps=total_steps,
		warmup_ratio=args.warmup_ratio,
	)
	scaler = torch.cuda.amp.GradScaler(
		enabled=True,
		init_scale=args.initial_gradient_scale,
	)
	git_commit = _resolve_git_commit(Path(args.project_root))
	checkpoint_metadata = {
		"dataset": args.dataset,
		"dataset_root": str(Path(args.dataset_root)),
		"model_checkpoint_sha256": checkpoint_hash_before,
		"git_commit": git_commit,
		"world_size": world_size,
		"per_device_batch_size": args.per_device_batch_size,
		"gradient_accumulation_steps": args.gradient_accumulation_steps,
		"total_optimizer_steps": total_steps,
		"train_rows": len(loader.dataset),
		"seed": args.seed,
		"attention_implementation": args.attention_implementation,
		"gradient_checkpointing": args.gradient_checkpointing,
		"initial_gradient_scale": args.initial_gradient_scale,
		"temperature": args.temperature,
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
		"visual_length_buckets": args.visual_length_buckets,
		"min_visual_bucket_size": args.min_visual_bucket_size,
		"lora_decoder_scope": lora_scope,
	}
	cursor = TrainingCursor(
		stage=0,
		global_step=0,
		sampler_epoch=0,
		batch_in_epoch=0,
		gradient_accumulation_step=0,
		processed_samples=0,
	)
	if resume_checkpoint is not None:
		cursor, resume_metadata = load_training_checkpoint(
			path=resume_checkpoint,
			model=training_model,
			optimizer=optimizer,
			scheduler=scheduler,
			rank=rank,
			gradient_scaler=scaler,
		)
		validate_checkpoint_metadata(
			resume_metadata,
			expected=checkpoint_metadata,
		)
		if cursor.stage != 0 or cursor.gradient_accumulation_step != 0:
			raise ValueError("Baseline resume requires a complete optimizer-step checkpoint")
		if cursor.global_step >= total_steps:
			raise ValueError("Resume checkpoint already reached the configured training limit")
		if rank == 0:
			truncate_metric_log(
				output_dir / "train_metrics.jsonl",
				maximum_global_step=cursor.global_step,
			)
	ddp_model = DistributedDataParallel(
		training_model,
		device_ids=[local_rank],
		output_device=local_rank,
		broadcast_buffers=False,
		find_unused_parameters=False,
		gradient_as_bucket_view=True,
	)
	manifest = {
		"scope": "unmodified_qwen3_vl_embedding_2b_lora_retrieval",
		"dataset": args.dataset,
		"dataset_root": str(args.dataset_root),
		"train_rows": len(loader.dataset),
		"direction_counts": (
			count_coco_retrieval_directions(len(loader.dataset))
			if args.dataset == "coco"
			else {"visual_question_answering": len(loader.dataset)}
		),
		"hostname": socket.gethostname(),
		"git_commit": git_commit,
		"command": sys.argv,
		"world_size": world_size,
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"runtime_precision": "fp16",
		"initial_gradient_scale": args.initial_gradient_scale,
		"attention_implementation": args.attention_implementation,
		"gradient_checkpointing": args.gradient_checkpointing,
		"per_device_batch_size": args.per_device_batch_size,
		"gradient_accumulation_steps": args.gradient_accumulation_steps,
		"contrastive_global_batch_size": args.per_device_batch_size * world_size,
		"optimizer_global_batch_size": (
			args.per_device_batch_size
			* world_size
			* args.gradient_accumulation_steps
		),
		"expected_contrastive_global_batch_size": (
			args.expected_contrastive_global_batch_size
		),
		"num_workers": args.num_workers,
		"epochs": args.epochs,
		"total_optimizer_steps": total_steps,
		"learning_rate": args.learning_rate,
		"weight_decay": args.weight_decay,
		"warmup_ratio": args.warmup_ratio,
		"temperature": args.temperature,
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
		"visual_length_bucketing": {
			"enabled": args.visual_length_buckets > 1,
			"maximum_buckets": args.visual_length_buckets,
			"minimum_bucket_size": args.min_visual_bucket_size,
			"length_measure": "post_smart_resize_visual_tokens",
			"contrastive_batch_unchanged": True,
		},
		"seed": args.seed,
		"checkpoint_every": args.checkpoint_every,
		"formal_training_log_interval": FORMAL_TRAINING_LOG_INTERVAL,
		"max_checkpoints": args.max_checkpoints,
		"resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
		"lora": {
			"rank": BASELINE_LORA_RANK,
			"alpha": BASELINE_LORA_ALPHA,
			"dropout": 0.0,
			"target_modules": BASELINE_LORA_TARGETS,
			**lora_scope,
			"trainable_parameter_count": sum(
				parameter.numel()
				for parameter in training_model.parameters()
				if parameter.requires_grad
			),
			"trainable_parameter_names": trainable_names,
		},
		"model_checkpoint_sha256_before": checkpoint_hash_before,
	}
	if rank == 0:
		if resume_checkpoint is None:
			_write_json(output_dir / "run_manifest.json", manifest)
		else:
			_write_json(
				output_dir / f"resume_manifest_step{cursor.global_step:06d}.json",
				manifest,
			)
		_write_json(
			output_dir / "status.json",
			{
				"status": "training",
				"resumed_from_step": cursor.global_step,
			},
		)
	optimizer.zero_grad(set_to_none=True)
	global_step = cursor.global_step
	total_samples = cursor.processed_samples
	training_start = time.perf_counter()
	log_start = training_start
	log_samples = 0
	metric_samples = 0
	metric_accumulator: dict[str, torch.Tensor] = {}
	direction_counts: Counter[str] = Counter()
	optimizer_steps_since_log = 0
	torch.cuda.reset_peak_memory_stats(device)
	full_loader_batches = len(loader)
	for epoch in range(cursor.sampler_epoch, args.epochs):
		sampler.set_epoch(epoch)
		start_batch = cursor.batch_in_epoch if epoch == cursor.sampler_epoch else 0
		sampler.set_batch_range(start_batch, full_loader_batches)
		for relative_batch_index, batch in enumerate(loader):
			batch_index = start_batch + relative_batch_index
			group_start = (
				batch_index // args.gradient_accumulation_steps
			) * args.gradient_accumulation_steps
			group_size = min(
				args.gradient_accumulation_steps,
				full_loader_batches - group_start,
			)
			is_boundary = (
				(batch_index + 1) % args.gradient_accumulation_steps == 0
				or batch_index + 1 == full_loader_batches
			)
			try:
				combined_inputs = batch["query_inputs"] + batch["candidate_inputs"]
				input_groups = group_baseline_model_inputs(
					combined_inputs,
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
				close_baseline_batch_images(batch)
			synchronization_context = nullcontext() if is_boundary else ddp_model.no_sync()
			with synchronization_context:
				with torch.autocast(device_type="cuda", dtype=torch.float16):
					output = ddp_model(
						local_batch_size=len(batch["positive_ids"]),
						processed_batches=processed_batches,
						original_indices=tuple(
							group.original_indices for group in input_groups
						),
						positive_ids=batch["positive_ids"],
					)
					loss = output["loss"] / group_size
				scaler.scale(loss).backward()
			batch_global_samples = len(batch["positive_ids"]) * world_size
			total_samples += batch_global_samples
			log_samples += batch_global_samples
			metric_samples += len(batch["positive_ids"])
			_accumulate_logging_metrics(
				metric_accumulator,
				output,
				sample_count=len(batch["positive_ids"]),
			)
			direction_counts.update(batch["directions"])
			cursor = TrainingCursor(
				stage=0,
				global_step=global_step,
				sampler_epoch=epoch,
				batch_in_epoch=batch_index + 1,
				gradient_accumulation_step=(
					(cursor.gradient_accumulation_step + 1)
					% args.gradient_accumulation_steps
				),
				processed_samples=total_samples,
			)
			if not is_boundary:
				continue
			scaler.unscale_(optimizer)
			gradient_norm = torch.nn.utils.clip_grad_norm_(
				(
					parameter
					for parameter in training_model.parameters()
					if parameter.requires_grad
				),
				args.gradient_clip_norm,
			)
			scale_before_step = scaler.get_scale()
			scaler.step(optimizer)
			scaler.update()
			optimizer_step_skipped = scaler.get_scale() < scale_before_step
			optimizer.zero_grad(set_to_none=True)
			if optimizer_step_skipped:
				if rank == 0:
					LOGGER.warning(
						"Skipped non-finite FP16 optimizer step at data batch %d",
						batch_index,
					)
				raise FloatingPointError(
					"Non-finite FP16 gradients skipped an optimizer step; "
					"resume from the latest checkpoint instead of silently "
					"losing samples",
				)
			scheduler.step()
			global_step += 1
			optimizer_steps_since_log += 1
			cursor = TrainingCursor(
				stage=0,
				global_step=global_step,
				sampler_epoch=epoch,
				batch_in_epoch=batch_index + 1,
				gradient_accumulation_step=0,
				processed_samples=total_samples,
			)
			if should_log_training_metrics(
				optimizer_steps_since_log=optimizer_steps_since_log,
				global_step=global_step,
				optimizer_step_limit=total_steps,
				force_every_step=args.max_optimizer_steps > 0,
			):
				torch.cuda.synchronize(device)
				elapsed = time.perf_counter() - log_start
				reduced_metrics, global_metric_samples = _reduce_logging_metrics(
					metric_accumulator,
					local_sample_count=metric_samples,
				)
				averages = _finalize_logging_metrics(
					reduced_metrics,
					sample_count=global_metric_samples,
				)
				global_direction_counts = _gather_direction_counts(
					direction_counts,
					world_size,
				)
				record = {
					"epoch": epoch,
					"global_step": global_step,
					**averages,
					"gradient_norm": float(gradient_norm.detach().float().item()),
					"learning_rate": float(scheduler.get_last_lr()[0]),
					"samples_per_second": log_samples / elapsed,
					"total_samples": total_samples,
					"gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
					"gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
					"contrastive_global_batch_size": batch_global_samples,
					"optimizer_global_batch_size": (
						global_metric_samples / optimizer_steps_since_log
					),
					"logged_global_samples": global_metric_samples,
					"logged_optimizer_steps": optimizer_steps_since_log,
					"direction_counts": dict(global_direction_counts),
				}
				if rank == 0:
					_append_json_line(output_dir / "train_metrics.jsonl", record)
					LOGGER.info(
						"dataset=%s step=%d/%d loss=%.5f samples/s=%.2f",
						args.dataset,
						global_step,
						total_steps,
						record["loss"],
						record["samples_per_second"],
					)
				log_start = time.perf_counter()
				log_samples = 0
				metric_samples = 0
				metric_accumulator = {}
				direction_counts = Counter()
				optimizer_steps_since_log = 0
			if (
				not args.skip_checkpoint_save
				and (
					global_step % args.checkpoint_every == 0
					or global_step == total_steps
				)
			):
				_save_baseline_checkpoint(
					output_dir=output_dir,
					model=training_model,
					optimizer=optimizer,
					scheduler=scheduler,
					scaler=scaler,
					cursor=cursor,
					metadata=checkpoint_metadata,
					rank=rank,
					world_size=world_size,
					max_checkpoints=args.max_checkpoints,
				)
			if global_step >= total_steps:
				break
		if global_step >= total_steps:
			break
		cursor = TrainingCursor(
			stage=0,
			global_step=global_step,
			sampler_epoch=epoch + 1,
			batch_in_epoch=0,
			gradient_accumulation_step=0,
			processed_samples=total_samples,
		)
	if not args.max_optimizer_steps and (
		cursor.batch_in_epoch != full_loader_batches or global_step != full_total_steps
	):
		raise RuntimeError(
			"Baseline one-epoch run did not consume every batch with one optimizer update",
		)
	dist.barrier()
	if rank == 0 and not args.skip_adapter_save:
		adapter_root = output_dir / "adapter"
		training_model.model.save_pretrained(adapter_root, safe_serialization=True)
	dist.barrier()
	checkpoint_hash_after = checkpoint_sha256(checkpoint_path) if rank == 0 else None
	if rank == 0 and checkpoint_hash_after != checkpoint_hash_before:
		raise RuntimeError("Immutable Qwen checkpoint changed during LoRA training")
	result = None
	if rank == 0:
		result = {
			"status": "passed",
			"dataset": args.dataset,
			"optimizer_steps": global_step,
			"processed_samples": total_samples,
			"runtime_seconds": time.perf_counter() - training_start,
			"model_checkpoint_sha256_before": checkpoint_hash_before,
			"model_checkpoint_sha256_after": checkpoint_hash_after,
			"adapter_root": (
				None if args.skip_adapter_save else str(output_dir / "adapter")
			),
		}
		_write_json(output_dir / "training_result.json", result)
		_write_json(output_dir / "status.json", {"status": "passed"})
	dist.barrier()
	dist.destroy_process_group()
	return result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset", choices=BASELINE_DATASETS, required=True)
	parser.add_argument("--dataset-root", type=Path, required=True)
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
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=8)
	parser.add_argument("--per-device-batch-size", type=int, default=32)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
	parser.add_argument(
		"--expected-contrastive-global-batch-size",
		type=int,
		default=256,
	)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--max-optimizer-steps", type=int, default=0)
	parser.add_argument("--max-train-rows", type=int, default=0)
	parser.add_argument("--skip-adapter-save", action="store_true")
	parser.add_argument("--skip-checkpoint-save", action="store_true")
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--max-checkpoints", type=int, choices=(1,), default=1)
	parser.add_argument("--resume-checkpoint", type=Path)
	parser.add_argument(
		"--lora-decoder-layer-indices",
		type=_parse_decoder_layer_indices,
		help=(
			"Optional sorted decoder-layer indices. Omit for the original all-layer LoRA."
		),
	)
	parser.add_argument("--learning-rate", type=float, default=5e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.02)
	parser.add_argument("--temperature", type=float, default=0.02)
	parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
	parser.add_argument("--initial-gradient-scale", type=float, default=4096.0)
	parser.add_argument("--seed", type=int, default=42)
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
	parser.add_argument(
		"--gradient-checkpointing",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		run_training(args)
		return 0
	except KeyboardInterrupt:
		logging.basicConfig(level=logging.INFO)
		LOGGER.warning("Baseline LoRA training interrupted")
		if int(os.environ.get("RANK", "0")) == 0:
			output_dir = Path(args.output_dir)
			if output_dir.exists():
				_write_json(output_dir / "status.json", {"status": "interrupted"})
		if dist.is_available() and dist.is_initialized():
			dist.destroy_process_group()
		return 130
	except Exception:
		logging.basicConfig(level=logging.INFO)
		LOGGER.exception("Baseline LoRA training failed")
		if int(os.environ.get("RANK", "0")) == 0:
			output_dir = Path(args.output_dir)
			if output_dir.exists():
				_write_json(output_dir / "status.json", {"status": "failed"})
		if dist.is_available() and dist.is_initialized():
			dist.destroy_process_group()
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
