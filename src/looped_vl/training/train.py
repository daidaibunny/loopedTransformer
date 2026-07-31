"""Distributed single-stage training for recurrent Qwen3-VL embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from looped_vl.models.config import (
	ALLOWED_SLOT_COUNTS,
	PURE_RECURRENT_TRAINING_PROTOCOL,
	RecurrentModelConfig,
	pure_recurrent_result_identity,
)
from looped_vl.models.latent_slot_inserter import create_or_load_master_slot_initialization
from looped_vl.models.loading import load_recurrent_components
from looped_vl.recurrent_data import RecurrentAlignedDataset
from looped_vl.runtime import (
	ATTENTION_IMPLEMENTATIONS,
	RUNTIME_PRECISIONS,
	TrainingPrecision,
	resolve_attention_implementation,
	resolve_training_precision,
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
from looped_vl.training.config import TrainingConfig
from looped_vl.training.data import (
	close_training_batch_images,
	group_model_inputs_by_modality,
	paired_training_collate,
)
from looped_vl.training.model import RecurrentTrainingModel
from looped_vl.training.optimizer import build_optimizer_and_scheduler
from looped_vl.training.reproducibility import seed_everything
from looped_vl.training.schedule import (
	FORMAL_TRAINING_LOG_INTERVAL,
	BatchOffsetSampler,
	OneEpochTrainingPlan,
	resolve_one_epoch_training_plan,
	resolve_parallel_batch_sizes,
	should_log_training_metrics,
)
from looped_vl.training.trainability import (
	align_trainable_parameter_dtype,
	audit_gradient_scope,
	configure_trainable_parameters,
)
from looped_vl.visual_bucketing import DEFAULT_MIN_VISUAL_BUCKET_SIZE

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_json_line(path: Path, value: Any) -> None:
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, sort_keys=True) + "\n")


def _resolve_git_commit(project_root: Path, explicit_commit: str | None) -> str:
	"""Return a validated explicit commit or resolve HEAD from a real Git checkout."""
	if explicit_commit is not None:
		if len(explicit_commit) != 40 or any(
			character not in "0123456789abcdef" for character in explicit_commit.lower()
		):
			raise ValueError("code_commit must be a full 40-character hexadecimal commit")
		explicit_commit = explicit_commit.lower()
	try:
		result = subprocess.run(
			["git", "rev-parse", "HEAD"],
			cwd=project_root,
			check=True,
			capture_output=True,
			text=True,
		)
	except subprocess.CalledProcessError as error:
		if explicit_commit is not None:
			return explicit_commit
		raise RuntimeError(
			f"Training project root is not a Git checkout: {project_root}",
		) from error
	resolved_commit = result.stdout.strip()
	if explicit_commit is not None and explicit_commit != resolved_commit:
		raise ValueError(
			f"Explicit code commit {explicit_commit} does not match checkout {resolved_commit}",
		)
	return resolved_commit


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, int, torch.device]:
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU training fallback is disabled")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	device = torch.device("cuda", local_rank)
	dist.init_process_group(backend="nccl", device_id=device)
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(f"Expected {expected_world_size} ranks, found {world_size}")
	return rank, world_size, local_rank, device


def _worker_loader_options(num_workers: int, prefetch_factor: int) -> dict[str, Any]:
	"""Build worker options that never fork an initialized CUDA process."""
	if num_workers <= 0:
		return {}
	return {
		"multiprocessing_context": "spawn",
		"persistent_workers": True,
		"prefetch_factor": prefetch_factor,
	}


def _build_loader(
	args: argparse.Namespace,
	rank: int,
	world_size: int,
	generator: torch.Generator,
) -> tuple[DataLoader[dict[str, Any]], BatchOffsetSampler]:
	dataset = RecurrentAlignedDataset(args.dataset_root, "train")
	distributed_sampler = DistributedSampler(
		dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True,
		seed=42,
		drop_last=False,
	)
	sampler = BatchOffsetSampler(distributed_sampler, args.per_device_batch_size)
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.per_device_batch_size,
		"sampler": sampler,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": paired_training_collate,
		"drop_last": False,
		"pin_memory": True,
		"generator": generator,
	}
	loader_kwargs.update(_worker_loader_options(args.num_workers, args.prefetch_factor))
	return DataLoader(**loader_kwargs), sampler


def _set_training_modes(model: RecurrentTrainingModel) -> None:
	model.train()
	model.encoder.base_embedding_model.eval()
	model.encoder.auxiliary_embedding_head.train()
	model.encoder.late_fusion.train()


def _gather_rank_rng_states(world_size: int) -> list[dict[str, Any]]:
	states: list[dict[str, Any] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(states, capture_rng_state())
	if any(state is None for state in states):
		raise RuntimeError("Failed to gather every rank RNG state")
	return [state for state in states if state is not None]


def _save_checkpoint(
	*,
	output_dir: Path,
	model: RecurrentTrainingModel,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	cursor: TrainingCursor,
	rank: int,
	world_size: int,
	metadata: dict[str, Any],
	gradient_scaler: torch.cuda.amp.GradScaler,
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
			gradient_scaler=gradient_scaler,
		)
		publish_latest_training_checkpoint(
			path,
			cursor,
			max_checkpoints=max_checkpoints,
		)
	dist.barrier()
	return path


def _accumulate_metric_tensors(
	accumulator: dict[str, torch.Tensor],
	step_output: dict[str, Any],
	keys: tuple[str, ...],
	*,
	sample_count: int = 1,
) -> None:
	"""Accumulate sample-weighted device scalars without a microbatch sync."""
	if sample_count <= 0:
		raise ValueError("Metric sample count must be positive")
	for key in keys:
		value = step_output[key].detach().float()
		accumulator[key] = (
			accumulator.get(key, torch.zeros_like(value)) + value * sample_count
		)


def _finalize_metric_tensors(
	accumulator: dict[str, torch.Tensor],
	count: int,
) -> dict[str, float]:
	"""Synchronize accumulated logging scalars once at the optimizer boundary."""
	if count <= 0:
		raise ValueError("Metric accumulator count must be positive")
	return {
		key: float((value / count).item())
		for key, value in accumulator.items()
	}


def _reduce_metric_tensors(
	accumulator: dict[str, torch.Tensor],
	*,
	local_sample_count: int,
) -> tuple[dict[str, torch.Tensor], int]:
	"""Sum metric numerators and sample counts across every rank."""
	if local_sample_count <= 0:
		raise ValueError("Metric sample count must be positive")
	reduced = {key: value.clone() for key, value in accumulator.items()}
	count = torch.tensor(
		local_sample_count,
		device=next(iter(reduced.values())).device,
		dtype=torch.long,
	)
	for value in reduced.values():
		dist.all_reduce(value, op=dist.ReduceOp.SUM)
	dist.all_reduce(count, op=dist.ReduceOp.SUM)
	return reduced, int(count.item())


def _gather_training_counters(
	*,
	source_counts: Counter[str],
	direction_counts: Counter[str],
	world_size: int,
) -> tuple[Counter[str], Counter[str]]:
	"""Return exact all-rank source and retrieval-direction counts."""
	gathered: list[dict[str, dict[str, int]] | None] = [
		None for _ in range(world_size)
	]
	dist.all_gather_object(
		gathered,
		{
			"sources": dict(source_counts),
			"directions": dict(direction_counts),
		},
	)
	global_sources: Counter[str] = Counter()
	global_directions: Counter[str] = Counter()
	for counters in gathered:
		if counters is None:
			raise RuntimeError("Failed to gather training counters from every rank")
		global_sources.update(counters["sources"])
		global_directions.update(counters["directions"])
	return global_sources, global_directions


def _optimizer_step_limit(
	*,
	configured_steps: int,
	resumed_global_step: int,
	smoke_optimizer_steps: int,
	max_additional_optimizer_steps: int,
) -> int:
	"""Resolve an absolute stop step for fresh, smoke, or resumed benchmark runs."""
	limit = configured_steps
	if smoke_optimizer_steps:
		limit = min(limit, smoke_optimizer_steps)
	if max_additional_optimizer_steps:
		limit = min(limit, resumed_global_step + max_additional_optimizer_steps)
	return limit


def _should_save_checkpoint(
	*,
	global_step: int,
	optimizer_step_limit: int,
	checkpoint_every: int,
	smoke_optimizer_steps: int,
	smoke_save_final_checkpoint: bool = False,
) -> bool:
	"""Skip multi-gigabyte optimizer checkpoints for disposable smoke runs."""
	if smoke_optimizer_steps:
		return (
			smoke_save_final_checkpoint
			and global_step == optimizer_step_limit
		)
	return global_step % checkpoint_every == 0 or global_step == optimizer_step_limit


def _training_phase(
	global_step: int,
	training_plan: OneEpochTrainingPlan,
) -> str:
	"""Return the sole stage label while validating the optimizer cursor."""
	if not 0 <= global_step < training_plan.optimizer_steps:
		raise ValueError("global_step is outside the single-stage training plan")
	return "single_stage"


def _distributed_data_parallel_options() -> dict[str, bool]:
	"""Return options that support repeated no-sync backward passes.

	The recurrent run accumulates several independently constructed forward
	graphs before each optimizer step. PyTorch's static-graph reducer can assert
	internally on the second no-sync backward pass, so this run must use the
	regular reducer while retaining bucket views.
	"""
	return {
		"broadcast_buffers": False,
		"find_unused_parameters": False,
		"gradient_as_bucket_view": True,
		"static_graph": False,
	}


def _train_one_epoch(
	*,
	args: argparse.Namespace,
	training_config: TrainingConfig,
	training_plan: OneEpochTrainingPlan,
	training_model: RecurrentTrainingModel,
	processor: Any,
	loader: DataLoader[dict[str, Any]],
	sampler: BatchOffsetSampler,
	rank: int,
	world_size: int,
	local_rank: int,
	device: torch.device,
	output_dir: Path,
	checkpoint_metadata: dict[str, Any],
	resume_checkpoint: Path | None,
	training_precision: TrainingPrecision,
) -> TrainingCursor:
	parameter_groups = configure_trainable_parameters(training_model.encoder)
	trainable_names = parameter_groups.all
	aligned_names = align_trainable_parameter_dtype(
		training_model.encoder,
		training_precision.trainable_parameter_dtype,
	)
	if set(aligned_names) != set(trainable_names):
		raise RuntimeError("Trainable dtype alignment did not match the optimizer allowlist")
	_set_training_modes(training_model)
	gradient_accumulation_steps = training_config.gradient_accumulation_steps(
		args.per_device_batch_size,
		world_size,
	)
	if args.smoke_optimizer_steps:
		gradient_accumulation_steps = args.smoke_gradient_accumulation_steps
	optimizer, scheduler = build_optimizer_and_scheduler(
		training_model,
		training_config,
		recurrent_core_parameter_names=tuple(
			f"encoder.{name}" for name in parameter_groups.recurrent_core
		),
		final_fusion_parameter_names=tuple(
			f"encoder.{name}" for name in parameter_groups.final_fusion
		),
		total_steps=training_plan.optimizer_steps,
	)
	gradient_scaler = torch.cuda.amp.GradScaler(
		enabled=training_precision.gradient_scaling_enabled,
		init_scale=args.initial_gradient_scale,
	)
	cursor = TrainingCursor(
		stage=1,
		global_step=0,
		sampler_epoch=0,
		batch_in_epoch=training_plan.start_batch,
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
			gradient_scaler=gradient_scaler,
			expected_training_protocol=PURE_RECURRENT_TRAINING_PROTOCOL,
		)
		if cursor.stage != 1:
			raise ValueError("Resume checkpoint is not from the single training stage")
		source_batch_size = args.resume_per_device_batch_size
		if source_batch_size is None:
			metadata_batch_size = resume_metadata.get("per_device_batch_size")
			source_batch_size = (
				int(metadata_batch_size)
				if metadata_batch_size is not None
				else args.per_device_batch_size
			)
		if source_batch_size != args.per_device_batch_size:
			raise ValueError(
				"Single-stage exact resume requires the original per-device batch size",
			)
		if resume_metadata.get("training_plan") != asdict(training_plan):
			raise ValueError("Resume checkpoint training plan does not match this run")
		validate_checkpoint_metadata(
			resume_metadata,
			expected=checkpoint_metadata,
		)
		if rank == 0:
			truncate_metric_log(
				output_dir / "train_metrics.jsonl",
				maximum_global_step=cursor.global_step,
			)
	if cursor.sampler_epoch != 0:
		raise ValueError("One-epoch training checkpoints must remain in sampler epoch zero")
	if not training_plan.start_batch <= cursor.batch_in_epoch <= training_plan.end_batch:
		raise ValueError("Checkpoint data cursor is outside the one-epoch batch range")
	optimizer_step_limit = _optimizer_step_limit(
		configured_steps=training_plan.optimizer_steps,
		resumed_global_step=cursor.global_step,
		smoke_optimizer_steps=args.smoke_optimizer_steps,
		max_additional_optimizer_steps=args.max_additional_optimizer_steps,
	)
	if cursor.global_step >= optimizer_step_limit:
		raise ValueError(
			"Resolved optimizer step limit must be greater than the resumed global step",
		)
	ddp_model = DistributedDataParallel(
		training_model,
		device_ids=[local_rank],
		output_device=local_rank,
		**_distributed_data_parallel_options(),
	)
	if rank == 0:
		_write_json(
			output_dir / "trainable_parameters.json",
			{
				"formal_training_stages": 1,
				"recurrent_core_names": parameter_groups.recurrent_core,
				"final_fusion_names": parameter_groups.final_fusion,
				"trainable_names": parameter_groups.all,
				"trainable_parameter_count": sum(
					parameter.numel()
					for parameter in training_model.parameters()
					if parameter.requires_grad
				),
				"gradient_accumulation_steps": gradient_accumulation_steps,
				"contrastive_global_batch_size": (
					args.per_device_batch_size * world_size
				),
				"optimizer_global_batch_size": (
					args.per_device_batch_size * world_size * gradient_accumulation_steps
				),
				"epoch_batch_start": training_plan.start_batch,
				"epoch_batch_end": training_plan.end_batch,
				"optimizer_steps": training_plan.optimizer_steps,
			},
		)
	optimizer.zero_grad(set_to_none=True)
	metric_keys = (
		"total_loss",
		"final_infonce",
		"loop_infonce",
		"slot_diversity",
		"fusion_gate",
		"late_fusion_attention_entropy",
		"slot_pairwise_cosine",
	)
	accumulator: dict[str, torch.Tensor] = {}
	accumulated_local_samples = 0
	accumulated_global_samples = 0
	contrastive_batch_sizes: Counter[int] = Counter()
	source_counts: Counter[str] = Counter()
	direction_counts: Counter[str] = Counter()
	optimizer_steps_since_log = 0
	interval_gradient_scope_audited = False
	step_start = time.perf_counter()
	epoch = 0
	sampler.set_epoch(epoch)
	start_batch = cursor.batch_in_epoch
	sampler.set_batch_range(start_batch, training_plan.end_batch)
	status_phase: str | None = None
	for relative_batch_index, batch in enumerate(loader):
		batch_index = start_batch + relative_batch_index
		phase = _training_phase(cursor.global_step, training_plan)
		if rank == 0 and phase != status_phase:
			_write_json(
				output_dir / "status.json",
				{"status": "training", "phase": phase, "global_step": cursor.global_step},
			)
			status_phase = phase
		group_start = training_plan.start_batch + (
			(batch_index - training_plan.start_batch) // gradient_accumulation_steps
		) * gradient_accumulation_steps
		group_end = min(
			training_plan.end_batch,
			group_start + gradient_accumulation_steps,
		)
		group_size = group_end - group_start
		is_accumulation_boundary = batch_index + 1 == group_end
		local_batch_size = len(batch["pairs"])
		contrastive_global_batch_size = local_batch_size * world_size
		try:
			input_groups = group_model_inputs_by_modality(
				batch["query_inputs"],
				batch["candidate_inputs"],
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
			close_training_batch_images(batch)
		synchronization_context = (
			nullcontext() if is_accumulation_boundary else ddp_model.no_sync()
		)
		with synchronization_context:
			with torch.autocast(
				device_type="cuda",
				dtype=training_precision.autocast_dtype,
				enabled=training_precision.autocast_enabled,
			):
				step_output = ddp_model(
					local_batch_size=local_batch_size,
					positive_ids=batch["positive_ids"],
					processed_batches=processed_batches,
					original_indices=tuple(
						group.original_indices
						for group in input_groups
					),
				)
				loss = step_output["total_loss"] / group_size
			gradient_scaler.scale(loss).backward()
		_accumulate_metric_tensors(
			accumulator,
			step_output,
			metric_keys,
			sample_count=local_batch_size,
		)
		for diagnostic_name in (
			"recurrent_pass_cosine",
			"recurrent_pass_relative_update",
			"loop_infonce_by_pass",
		):
			for pass_index, value in enumerate(step_output[diagnostic_name], start=1):
				key = f"{diagnostic_name}_pass{pass_index}"
				accumulator[key] = (
					accumulator.get(key, torch.zeros_like(value.detach().float()))
					+ value.detach().float() * local_batch_size
				)
		accumulated_local_samples += local_batch_size
		accumulated_global_samples += contrastive_global_batch_size
		contrastive_batch_sizes[contrastive_global_batch_size] += 1
		source_counts.update(batch["sources"])
		direction_counts.update(batch["directions"])
		if not is_accumulation_boundary:
			cursor = TrainingCursor(
				stage=1,
				global_step=cursor.global_step,
				sampler_epoch=epoch,
				batch_in_epoch=batch_index + 1,
				gradient_accumulation_step=cursor.gradient_accumulation_step + 1,
				processed_samples=cursor.processed_samples + contrastive_global_batch_size,
			)
			continue
		gradient_scaler.unscale_(optimizer)
		gradient_audit = None
		if cursor.global_step == 0:
			gradient_audit = audit_gradient_scope(
				training_model.encoder,
				allowed_names=parameter_groups.all,
			)
			if rank == 0:
				_write_json(
					output_dir / f"{phase}_gradient_audit.json",
					{"phase": phase, **gradient_audit},
				)
		gradient_norm = torch.nn.utils.clip_grad_norm_(
			(
				parameter
				for parameter in training_model.parameters()
				if parameter.requires_grad
			),
			training_config.gradient_clip_norm,
		)
		used_learning_rates = tuple(
			float(group["lr"]) for group in optimizer.param_groups
		)
		scale_before_step = gradient_scaler.get_scale()
		gradient_scaler.step(optimizer)
		gradient_scaler.update()
		optimizer_step_skipped = (
			gradient_scaler.is_enabled()
			and gradient_scaler.get_scale() < scale_before_step
		)
		optimizer.zero_grad(set_to_none=True)
		processed_samples = cursor.processed_samples + contrastive_global_batch_size
		if optimizer_step_skipped:
			if rank == 0:
				LOGGER.warning(
					"Skipped non-finite FP16 optimizer step at data batch %d",
					batch_index,
				)
			cursor = TrainingCursor(
				stage=1,
				global_step=cursor.global_step,
				sampler_epoch=epoch,
				batch_in_epoch=batch_index + 1,
				gradient_accumulation_step=0,
				processed_samples=processed_samples,
			)
			raise FloatingPointError(
				"Non-finite FP16 gradients skipped an optimizer step; "
				"resume from the latest checkpoint instead of silently losing samples",
			)
		scheduler.step()
		global_step = cursor.global_step + 1
		optimizer_steps_since_log += 1
		interval_gradient_scope_audited = (
			interval_gradient_scope_audited or gradient_audit is not None
		)
		if should_log_training_metrics(
			optimizer_steps_since_log=optimizer_steps_since_log,
			global_step=global_step,
			optimizer_step_limit=optimizer_step_limit,
			force_every_step=bool(
				args.smoke_optimizer_steps or args.max_additional_optimizer_steps
			),
			force_boundary=False,
		):
			torch.cuda.synchronize(device)
			reduced_accumulator, global_metric_samples = _reduce_metric_tensors(
				accumulator,
				local_sample_count=accumulated_local_samples,
			)
			averages = _finalize_metric_tensors(
				reduced_accumulator,
				global_metric_samples,
			)
			finalized_diagnostics = {
				key: [
					averages.pop(f"{key}_pass{pass_index}")
					for pass_index in range(
						1,
						training_model.encoder.config.num_total_loop_passes + 1,
					)
				]
				for key in (
					"recurrent_pass_cosine",
					"recurrent_pass_relative_update",
					"loop_infonce_by_pass",
				)
			}
			global_source_counts, global_direction_counts = _gather_training_counters(
				source_counts=source_counts,
				direction_counts=direction_counts,
				world_size=world_size,
			)
			elapsed = time.perf_counter() - step_start
			log_record = {
				"formal_training_stage": 1,
				"phase": phase,
				"global_step": global_step,
				**averages,
				**finalized_diagnostics,
				"gradient_norm": float(gradient_norm.detach().float().item()),
				"recurrent_core_learning_rate": used_learning_rates[0],
				"final_fusion_learning_rate": used_learning_rates[1],
				"next_recurrent_core_learning_rate": float(scheduler.get_last_lr()[0]),
				"next_final_fusion_learning_rate": float(scheduler.get_last_lr()[1]),
				"gradient_scale": float(gradient_scaler.get_scale()),
				"gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
				"gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
				"samples_per_second": accumulated_global_samples / elapsed,
				"optimizer_global_batch_size": (
					global_metric_samples / optimizer_steps_since_log
				),
				"logged_global_samples": global_metric_samples,
				"logged_optimizer_steps": optimizer_steps_since_log,
				"contrastive_global_batch_size_min": min(contrastive_batch_sizes),
				"contrastive_global_batch_size_max": max(contrastive_batch_sizes),
				"contrastive_microbatch_count": sum(contrastive_batch_sizes.values()),
				"source_counts": dict(global_source_counts),
				"direction_counts": dict(global_direction_counts),
				"slot_collapse": averages["slot_pairwise_cosine"] > 0.98,
				"pooling_collapse": (
					training_model.encoder.config.num_latent_slots > 1
					and averages["late_fusion_attention_entropy"] < 0.1
				),
				"recurrence_unused": (
					global_step > 100
					and max(finalized_diagnostics["recurrent_pass_relative_update"]) < 1e-6
				),
				"gradient_scope_audited": interval_gradient_scope_audited,
			}
			if rank == 0:
				_append_json_line(output_dir / "train_metrics.jsonl", log_record)
				LOGGER.info(
					"phase=%s step=%d/%d loss=%.5f grad=%.5f samples/s=%.3f",
					phase,
					global_step,
					optimizer_step_limit,
					averages["total_loss"],
					float(gradient_norm.detach().float().item()),
					accumulated_global_samples / elapsed,
				)
			accumulator = {}
			accumulated_local_samples = 0
			accumulated_global_samples = 0
			contrastive_batch_sizes = Counter()
			source_counts = Counter()
			direction_counts = Counter()
			optimizer_steps_since_log = 0
			interval_gradient_scope_audited = False
			step_start = time.perf_counter()
		cursor = TrainingCursor(
			stage=1,
			global_step=global_step,
			sampler_epoch=epoch,
			batch_in_epoch=batch_index + 1,
			gradient_accumulation_step=0,
			processed_samples=processed_samples,
		)
		if _should_save_checkpoint(
			global_step=cursor.global_step,
			optimizer_step_limit=optimizer_step_limit,
			checkpoint_every=args.checkpoint_every,
			smoke_optimizer_steps=args.smoke_optimizer_steps,
			smoke_save_final_checkpoint=args.smoke_save_final_checkpoint,
		):
			_save_checkpoint(
				output_dir=output_dir,
				model=training_model,
				optimizer=optimizer,
				scheduler=scheduler,
				cursor=cursor,
				rank=rank,
				world_size=world_size,
				metadata=checkpoint_metadata,
				gradient_scaler=gradient_scaler,
				max_checkpoints=args.max_checkpoints,
			)
		if cursor.global_step >= optimizer_step_limit:
			break
	if (
		not args.smoke_optimizer_steps
		and not args.max_additional_optimizer_steps
		and (
			cursor.global_step != training_plan.optimizer_steps
			or cursor.batch_in_epoch != training_plan.end_batch
		)
	):
		raise RuntimeError(
			"Recurrent one-epoch run did not consume every batch with one optimizer update",
		)
	del ddp_model
	dist.barrier()
	return cursor


def run_training(args: argparse.Namespace) -> dict[str, Any] | None:
	"""Run one continuous epoch with the final objective active throughout."""
	if args.checkpoint_every <= 0:
		raise ValueError("checkpoint_every must be positive")
	if args.max_checkpoints != 1:
		raise ValueError("max_checkpoints must be exactly 1")
	if args.resume_per_device_batch_size is not None and args.resume_checkpoint is None:
		raise ValueError("resume_per_device_batch_size requires resume_checkpoint")
	if (
		args.resume_per_device_batch_size is not None
		and args.resume_per_device_batch_size <= 0
	):
		raise ValueError("resume_per_device_batch_size must be positive")
	if args.max_additional_optimizer_steps < 0:
		raise ValueError("max_additional_optimizer_steps cannot be negative")
	if args.initial_gradient_scale <= 0:
		raise ValueError("initial_gradient_scale must be positive")
	if args.smoke_optimizer_steps and args.max_additional_optimizer_steps:
		raise ValueError(
			"smoke_optimizer_steps and max_additional_optimizer_steps are mutually exclusive",
		)
	if args.smoke_save_final_checkpoint and not args.smoke_optimizer_steps:
		raise ValueError(
			"smoke_save_final_checkpoint requires smoke_optimizer_steps",
		)
	rank, world_size, local_rank, device = _initialize_distributed(args.expected_world_size)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	generator = seed_everything(42)
	resolved_attention_implementation = resolve_attention_implementation(
		args.attention_implementation,
	)
	training_precision = resolve_training_precision(args.runtime_precision)
	output_dir = Path(args.output_dir)
	resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
	if rank == 0:
		output_mode = prepare_training_output_directory(
			output_dir,
			resume_checkpoint=resume_checkpoint,
		)
		_write_json(
			output_dir / "status.json",
			{
				"status": "initializing" if output_mode == "fresh" else "resuming",
				"checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
			},
		)
	dist.barrier()
	project_root = Path(args.project_root)
	git_commit = _resolve_git_commit(project_root, args.code_commit)
	model_config = RecurrentModelConfig.from_yaml(args.model_config)
	if args.num_latent_slots is not None:
		model_config = model_config.with_variant(
			num_latent_slots=args.num_latent_slots,
		)
	training_config = TrainingConfig.from_yaml(args.training_config)
	model_checkpoint_path = Path(args.model_root) / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(model_checkpoint_path) if rank == 0 else None
	if rank == 0:
		create_or_load_master_slot_initialization(
			path=args.master_slot_path,
			max_num_latent_slots=model_config.max_num_latent_slots,
			hidden_size=model_config.hidden_size,
			seed=model_config.seed,
			mean=model_config.latent_init_mean,
			std=model_config.latent_init_std,
		)
	dist.barrier()
	components = load_recurrent_components(
		model_root=args.model_root,
		master_slot_path=args.master_slot_path,
		config=model_config,
		device=device,
		dtype=training_precision.parameter_dtype,
		attention_implementation=resolved_attention_implementation,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	training_model = RecurrentTrainingModel(components.model)
	training_model.encoder.set_activation_checkpointing(args.gradient_checkpointing)
	loader, sampler = _build_loader(args, rank, world_size, generator)
	gradient_accumulation_steps = training_config.gradient_accumulation_steps(
		args.per_device_batch_size,
		world_size,
	)
	if args.smoke_optimizer_steps:
		gradient_accumulation_steps = args.smoke_gradient_accumulation_steps
	parallel_batch_sizes = resolve_parallel_batch_sizes(
		per_device_batch_size=args.per_device_batch_size,
		world_size=world_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
	)
	training_plan = resolve_one_epoch_training_plan(
		train_rows=len(loader.dataset),
		loader_batches=len(loader),
		gradient_accumulation_steps=gradient_accumulation_steps,
		optimizer_global_batch_size=parallel_batch_sizes.optimizer_global_batch_size,
	)
	if (
		parallel_batch_sizes.contrastive_global_batch_size
		!= args.expected_contrastive_global_batch_size
	):
		raise ValueError(
			"Contrastive global batch is "
			f"{parallel_batch_sizes.contrastive_global_batch_size}, expected "
			f"{args.expected_contrastive_global_batch_size}; gradient accumulation "
			"does not add in-batch negatives",
		)
	manifest = {
		"scope": "recurrent_latent_slot_qwen3vl_v1",
		**pure_recurrent_result_identity(),
		"formal_training_stages": 1,
		"hostname": socket.gethostname(),
		"git_commit": git_commit,
		"command": sys.argv,
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"world_size": world_size,
		"specification_precision": "bf16",
		"runtime_precision": args.runtime_precision,
		"parameter_dtype": str(training_precision.parameter_dtype),
		"trainable_parameter_dtype": str(training_precision.trainable_parameter_dtype),
		"autocast_dtype": str(training_precision.autocast_dtype),
		"autocast_enabled": training_precision.autocast_enabled,
		"gradient_scaling_enabled": training_precision.gradient_scaling_enabled,
		"initial_gradient_scale": args.initial_gradient_scale,
		"requested_attention_implementation": args.attention_implementation,
		"resolved_attention_implementation": resolved_attention_implementation,
		"resolved_backbone_attention_implementation": (
			components.model.language_model.config._attn_implementation
		),
		"semantic_decoder_enabled": False,
		"modality_grouped_padding": True,
		"gradient_checkpointing": args.gradient_checkpointing,
		"ddp_gradient_as_bucket_view": True,
		"ddp_static_graph": False,
		"fused_adamw": True,
		"seed": 42,
		"model_config": asdict(model_config),
		"training_config": asdict(training_config),
		"schedule": {
			"epochs": 1,
			"policy": "single_stage_fixed_full_objective",
			"resolved_training_plan": asdict(training_plan),
			"loader_batches_per_rank": len(loader),
			"distributed_sampler_total_rows": sampler.total_size,
			"distributed_sampler_padding_rows": sampler.total_size - len(loader.dataset),
		},
		"dataset_root": str(args.dataset_root),
		"train_rows": len(loader.dataset),
		"per_device_batch_size": args.per_device_batch_size,
		"gradient_accumulation_steps": gradient_accumulation_steps,
		**asdict(parallel_batch_sizes),
		"expected_contrastive_global_batch_size": (
			args.expected_contrastive_global_batch_size
		),
		"multi_positive_contrastive_loss": True,
		"combined_contrastive_all_gather": True,
		"final_infonce_weight_all_steps": 1.0,
		"loop_infonce_weight_all_steps": 0.1,
		"slot_diversity_weight_all_steps": 0.05,
		"all_parameter_groups_active_from_step_one": True,
		"resume_per_device_batch_size": args.resume_per_device_batch_size,
		"max_additional_optimizer_steps": args.max_additional_optimizer_steps,
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
		"model_checkpoint_sha256_before": checkpoint_hash_before,
		"gpus": [
			{
				"rank": rank,
				"logical_device": local_rank,
				"name": torch.cuda.get_device_name(local_rank),
			}
		],
		"smoke_optimizer_steps": args.smoke_optimizer_steps,
		"checkpoint_every": args.checkpoint_every,
		"formal_training_log_interval": FORMAL_TRAINING_LOG_INTERVAL,
		"max_checkpoints": args.max_checkpoints,
		"visual_length_buckets": args.visual_length_buckets,
		"min_visual_bucket_size": args.min_visual_bucket_size,
	}
	all_manifests: list[dict[str, Any] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(all_manifests, manifest)
	checkpoint_metadata = {
		**pure_recurrent_result_identity(),
		"git_commit": manifest["git_commit"],
		"model_checkpoint_sha256": checkpoint_hash_before,
		"model_config": manifest["model_config"],
		"training_config": manifest["training_config"],
		"training_plan": asdict(training_plan),
		"dataset_root": str(Path(args.dataset_root)),
		"train_rows": len(loader.dataset),
		"per_device_batch_size": args.per_device_batch_size,
		"world_size": world_size,
		"expected_contrastive_global_batch_size": (
			args.expected_contrastive_global_batch_size
		),
		"runtime_precision": args.runtime_precision,
		"attention_implementation": resolved_attention_implementation,
		"gradient_checkpointing": args.gradient_checkpointing,
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
	}
	if rank == 0:
		manifest["gpus"] = [item["gpus"][0] for item in all_manifests if item is not None]
		if resume_checkpoint is None:
			_write_json(output_dir / "run_manifest.json", manifest)
		else:
			_write_json(
				output_dir / f"resume_manifest_{resume_checkpoint.stem}.json",
				manifest,
			)
		_write_json(
			output_dir / "status.json",
			{"status": "training", "phase": "single_stage"},
		)
	training_start = time.perf_counter()
	final_cursor = _train_one_epoch(
		args=args,
		training_config=training_config,
		training_plan=training_plan,
		training_model=training_model,
		processor=components.processor,
		loader=loader,
		sampler=sampler,
		rank=rank,
		world_size=world_size,
		local_rank=local_rank,
		device=device,
		output_dir=output_dir,
		checkpoint_metadata=checkpoint_metadata,
		resume_checkpoint=resume_checkpoint,
		training_precision=training_precision,
	)
	checkpoint_hash_after = checkpoint_sha256(model_checkpoint_path) if rank == 0 else None
	if rank == 0 and checkpoint_hash_after != checkpoint_hash_before:
		raise RuntimeError("Original Qwen checkpoint changed during training")
	result = None
	if rank == 0:
		trainable_parameter_count = sum(
			parameter.numel()
			for parameter in training_model.parameters()
			if parameter.requires_grad
		)
		result = {
			"status": "passed",
			**pure_recurrent_result_identity(),
			"trainable_parameter_count": trainable_parameter_count,
			"final_cursor": asdict(final_cursor),
			"runtime_seconds": time.perf_counter() - training_start,
			"model_checkpoint_sha256_before": checkpoint_hash_before,
			"model_checkpoint_sha256_after": checkpoint_hash_after,
		}
		_write_json(output_dir / "training_result.json", result)
		_write_json(output_dir / "status.json", {"status": "passed"})
	dist.barrier()
	dist.destroy_process_group()
	return result


def parse_args() -> argparse.Namespace:
	"""Parse the complete single-stage training command."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/loopedTransformer"),
	)
	parser.add_argument("--code-commit")
	parser.add_argument("--model-config", type=Path, default=Path("configs/base.yaml"))
	parser.add_argument(
		"--num-latent-slots",
		type=int,
		choices=ALLOWED_SLOT_COUNTS,
	)
	parser.add_argument("--training-config", type=Path, default=Path("configs/train.yaml"))
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument(
		"--master-slot-path",
		type=Path,
		default=Path(
			"/home/mnt/liyiwei/loopedTransformer/artifacts/master_slot_init_seed42.pt",
		),
	)
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=8)
	parser.add_argument("--per-device-batch-size", type=int, default=8)
	parser.add_argument(
		"--expected-contrastive-global-batch-size",
		type=int,
		default=64,
	)
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
	parser.add_argument("--initial-gradient-scale", type=float, default=32.0)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--max-checkpoints", type=int, choices=(1,), default=1)
	parser.add_argument("--resume-checkpoint", type=Path)
	parser.add_argument("--resume-per-device-batch-size", type=int)
	parser.add_argument("--max-additional-optimizer-steps", type=int, default=0)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument(
		"--gradient-checkpointing",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument(
		"--visual-length-buckets",
		type=int,
		default=1,
		help=(
			"Number of visual-length encoding buckets. One disables bucketing. "
			"Bucketing changes padding only, never contrastive batch composition."
		),
	)
	parser.add_argument(
		"--min-visual-bucket-size",
		type=int,
		default=DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	)
	parser.add_argument("--smoke-optimizer-steps", type=int, default=0)
	parser.add_argument("--smoke-gradient-accumulation-steps", type=int, default=1)
	parser.add_argument("--smoke-save-final-checkpoint", action="store_true")
	return parser.parse_args()


def main() -> int:
	"""Run distributed training and write failure status when possible."""
	args = parse_args()
	try:
		run_training(args)
		return 0
	except KeyboardInterrupt:
		logging.basicConfig(level=logging.INFO)
		LOGGER.warning("Recurrent training interrupted")
		if int(os.environ.get("RANK", "0")) == 0:
			output_dir = Path(args.output_dir)
			if output_dir.exists():
				_write_json(output_dir / "status.json", {"status": "interrupted"})
		if dist.is_available() and dist.is_initialized():
			dist.destroy_process_group()
		return 130
	except Exception:
		logging.basicConfig(level=logging.INFO)
		LOGGER.exception("Recurrent training failed")
		if int(os.environ.get("RANK", "0")) == 0:
			output_dir = Path(args.output_dir)
			if output_dir.exists():
				_write_json(output_dir / "status.json", {"status": "failed"})
		if dist.is_available() and dist.is_initialized():
			dist.destroy_process_group()
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
