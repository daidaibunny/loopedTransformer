"""Eight-GPU query-only recurrent training against immutable candidate banks."""

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
from looped_vl.baseline.data import BASELINE_DATASETS
from looped_vl.baseline.model import BaselineInputProcessor, load_frozen_evaluation_model
from looped_vl.candidate_bank import CandidateBankSpec, sha256_file
from looped_vl.query_recurrent.backbone import (
	FrozenQueryBackbone,
)
from looped_vl.query_recurrent.candidate_store import CandidateStoreCollection
from looped_vl.query_recurrent.config import (
	MAX_QUERY_RECURRENT_PARAMETERS,
	QueryRecurrentConfig,
)
from looped_vl.query_recurrent.data import (
	QueryOnlyManifestDataset,
	close_query_only_images,
	query_only_collate,
)
from looped_vl.query_recurrent.losses import query_recurrent_loss
from looped_vl.query_recurrent.model import (
	GroupedQueryRecurrentHead,
	QueryRecurrentHead,
	query_recurrent_diagnostics,
	recurrent_fp32_context,
	recurrent_gradient_group_norms,
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

LOGGER = logging.getLogger("query_recurrent_train")


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_json_line(path: Path, value: Any) -> None:
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, sort_keys=True) + "\n")


def _resolve_resume_source_commit(
	*,
	current_git_commit: str,
	checkpoint_git_commit: str,
	authorized_source_git_commit: str | None,
) -> str:
	"""Require an explicit exact commit when recovery code differs from a checkpoint."""
	if not checkpoint_git_commit:
		raise ValueError("Resume checkpoint is missing its source Git commit")
	if checkpoint_git_commit == current_git_commit:
		if (
			authorized_source_git_commit is not None
			and authorized_source_git_commit != checkpoint_git_commit
		):
			raise ValueError("Authorized source Git commit does not match the checkpoint")
		return checkpoint_git_commit
	if authorized_source_git_commit != checkpoint_git_commit:
		raise ValueError(
			"Resume recovery requires the exact checkpoint source Git commit",
		)
	return checkpoint_git_commit


def _lower_loaded_gradient_scale(
	scaler: Any,
	*,
	new_scale: float,
) -> float:
	"""Lower a restored gradient scale without changing model or optimizer state."""
	if not math.isfinite(new_scale) or new_scale <= 0:
		raise ValueError("Resume gradient scale must be finite and positive")
	state = scaler.state_dict()
	if not state or "scale" not in state:
		raise RuntimeError("Loaded gradient scaler state is unavailable")
	previous_scale = float(state["scale"])
	if new_scale >= previous_scale:
		raise ValueError(
			"Resume gradient scale must be strictly below the loaded scale",
		)
	state["scale"] = float(new_scale)
	state["_growth_tracker"] = 0
	scaler.load_state_dict(state)
	return previous_scale


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


def _required_bank_specs(dataset: str, split: str) -> tuple[CandidateBankSpec, ...]:
	if dataset == "coco":
		return (
			CandidateBankSpec("coco", split, "image"),
			CandidateBankSpec("coco", split, "text"),
		)
	return (CandidateBankSpec(dataset, "shared", "answer"),)


def _candidate_bank_identities(
	candidate_root: Path,
	specs: tuple[CandidateBankSpec, ...],
) -> dict[str, str]:
	identities = {}
	for spec in specs:
		root = candidate_root / spec.relative_path
		manifest_path = root / "bank_manifest.json"
		ready_path = root / "READY"
		if not manifest_path.is_file() or not ready_path.is_file():
			raise FileNotFoundError(f"Candidate bank is not ready: {spec.key}")
		manifest_hash = sha256_file(manifest_path)
		if ready_path.read_text(encoding="utf-8").strip() != manifest_hash:
			raise ValueError(f"Candidate bank READY checksum mismatch: {spec.key}")
		identities[spec.key] = manifest_hash
	return identities


def _build_loader(
	args: argparse.Namespace,
	*,
	rank: int,
	world_size: int,
	generator: torch.Generator,
) -> tuple[DataLoader[dict[str, Any]], BatchOffsetSampler]:
	dataset = QueryOnlyManifestDataset(
		args.dataset_root,
		args.dataset,
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
		"collate_fn": query_only_collate,
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
	model: QueryRecurrentHead,
	*,
	learning_rate: float,
	weight_decay: float,
) -> torch.optim.AdamW:
	parameters = list(model.parameters())
	kwargs = {
		"lr": learning_rate,
		"weight_decay": weight_decay,
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


def _encode_query_batch(
	*,
	batch: dict[str, Any],
	processor: BaselineInputProcessor,
	backbone: FrozenQueryBackbone,
	args: argparse.Namespace,
	device: torch.device,
) -> tuple[tuple[tuple[int, ...], torch.Tensor], ...]:
	try:
		groups = group_baseline_model_inputs(
			batch["query_inputs"],
			min_pixels=args.min_pixels,
			max_pixels=args.max_pixels,
			max_visual_buckets=args.visual_length_buckets,
			min_visual_bucket_size=args.min_visual_bucket_size,
		)
		processed = tuple(
			processor.prepare(list(group.model_inputs), device=device) for group in groups
		)
	finally:
		close_query_only_images(batch)
	features = []
	with torch.autocast(device_type="cuda", dtype=torch.float16):
		for group, inputs in zip(groups, processed, strict=True):
			frozen = backbone(inputs)
			features.append(
				(
					group.original_indices,
					frozen.base_embeddings,
				),
			)
	return tuple(features)


def _restore_base_embeddings(
	feature_groups: tuple[
		tuple[tuple[int, ...], torch.Tensor],
		...,
	],
	*,
	total_rows: int,
) -> torch.Tensor:
	"""Restore frozen query embeddings from padding groups to logical batch order."""
	flat_indices = tuple(index for group in feature_groups for index in group[0])
	if tuple(sorted(flat_indices)) != tuple(range(total_rows)):
		raise ValueError("Frozen feature groups must cover every logical query row")
	values = torch.cat(tuple(group[1] for group in feature_groups), dim=0)
	restore_order = torch.argsort(torch.tensor(flat_indices, device=values.device))
	return values.index_select(0, restore_order)


def _reduce_metrics(
	accumulator: dict[str, torch.Tensor],
	*,
	local_samples: int,
) -> tuple[dict[str, float], int]:
	if local_samples <= 0:
		raise ValueError("Metric sample count must be positive")
	reduced = {key: value.clone() for key, value in accumulator.items()}
	count = torch.tensor(
		local_samples,
		device=next(iter(reduced.values())).device,
		dtype=torch.long,
	)
	for value in reduced.values():
		dist.all_reduce(value, op=dist.ReduceOp.SUM)
	dist.all_reduce(count, op=dist.ReduceOp.SUM)
	global_samples = int(count.item())
	return (
		{key: float((value / global_samples).item()) for key, value in reduced.items()},
		global_samples,
	)


def _gather_rank_rng_states(world_size: int) -> list[dict[str, Any]]:
	states: list[dict[str, Any] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(states, capture_rng_state())
	if any(state is None for state in states):
		raise RuntimeError("Failed to gather every rank RNG state")
	return [state for state in states if state is not None]


def _save_checkpoint(
	*,
	output_dir: Path,
	model: QueryRecurrentHead,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	scaler: torch.cuda.amp.GradScaler,
	cursor: TrainingCursor,
	metadata: dict[str, Any],
	rank: int,
	world_size: int,
) -> None:
	states = _gather_rank_rng_states(world_size)
	path = output_dir / "checkpoints" / f"step{cursor.global_step:06d}.pt"
	if rank == 0 and not path.exists():
		save_training_checkpoint(
			path,
			model,
			optimizer,
			scheduler,
			cursor,
			states,
			metadata,
			gradient_scaler=scaler,
		)
		publish_latest_training_checkpoint(path, cursor, max_checkpoints=1)
	dist.barrier()


def _save_final_head(path: Path, model: QueryRecurrentHead, manifest: dict[str, Any]) -> str:
	payload = {
		"format_version": 1,
		"config": model.config.identity(),
		"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
		"run_manifest": manifest,
	}
	temporary = path.with_suffix(path.suffix + ".tmp")
	torch.save(payload, temporary)
	temporary.replace(path)
	return checkpoint_sha256(path)


def run_training(args: argparse.Namespace) -> dict[str, Any] | None:
	if args.dataset not in BASELINE_DATASETS:
		raise ValueError(f"Unsupported dataset: {args.dataset}")
	if args.epochs != 1:
		raise ValueError("Query-only recurrent training must use exactly one epoch")
	if args.per_device_batch_size * args.expected_world_size != 256:
		raise ValueError("Formal query-only training requires a true contrastive batch of 256")
	if args.gradient_accumulation_steps != 1:
		raise ValueError("Query-only recurrent training uses one optimizer batch per data batch")
	if args.max_checkpoints != 1 or args.checkpoint_every <= 0:
		raise ValueError("Exactly one rolling checkpoint and a positive interval are required")
	config = QueryRecurrentConfig(
		num_worlds=args.num_worlds,
		max_recurrent_steps=args.max_recurrent_steps,
		perturbation_scale=args.perturbation_scale,
		temperature=args.temperature,
		hard_negative_count=args.hard_negative_count,
		seed=args.seed,
	)
	config.validate()
	rank, world_size, local_rank, device = _initialize_distributed(args.expected_world_size)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	generator = _seed_everything(args.seed, rank)
	output_dir = Path(args.output_dir)
	resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
	if resume_checkpoint is None and (
		args.resume_source_git_commit is not None
		or args.resume_gradient_scale is not None
	):
		raise ValueError("Resume recovery controls require --resume-checkpoint")
	if rank == 0:
		mode = prepare_training_output_directory(
			output_dir,
			resume_checkpoint=resume_checkpoint,
		)
		_write_json(
			output_dir / "status.json",
			{"status": "initializing" if mode == "fresh" else "resuming"},
		)
	dist.barrier()

	model_root = Path(args.model_root)
	base_checkpoint_path = model_root / "model.safetensors"
	base_hash = checkpoint_sha256(base_checkpoint_path) if rank == 0 else None
	base_hash_values = [base_hash]
	dist.broadcast_object_list(base_hash_values, src=0)
	base_hash = str(base_hash_values[0])
	required_specs = _required_bank_specs(args.dataset, "train")
	bank_identities = (
		_candidate_bank_identities(Path(args.candidate_root), required_specs)
		if rank == 0
		else None
	)
	bank_values = [bank_identities]
	dist.broadcast_object_list(bank_values, src=0)
	bank_identities = dict(bank_values[0])
	if rank == 0:
		validator = CandidateStoreCollection(
			candidate_root=args.candidate_root,
			model_checkpoint_sha256=base_hash,
			validate_checksums=True,
		)
		for spec in required_specs:
			validator.get(spec)
		del validator
	dist.barrier()
	candidate_stores = CandidateStoreCollection(
		candidate_root=args.candidate_root,
		model_checkpoint_sha256=base_hash,
		validate_checksums=False,
	)
	for spec in required_specs:
		candidate_stores.get(spec)
	resolved_hard_negative_count = candidate_stores.resolved_hard_negative_count(
		required_specs,
		config.hard_negative_count,
	)

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
	head = QueryRecurrentHead(config).to(device)
	training_model = GroupedQueryRecurrentHead(head)
	if head.trainable_parameter_count > MAX_QUERY_RECURRENT_PARAMETERS:
		raise RuntimeError("Query recurrent parameter cap was violated")
	if any(parameter.requires_grad for parameter in base_model.parameters()):
		raise RuntimeError("Frozen Qwen unexpectedly contains trainable parameters")
	loader, sampler = _build_loader(
		args,
		rank=rank,
		world_size=world_size,
		generator=generator,
	)
	full_loader_batches = len(loader)
	full_total_steps = math.ceil(full_loader_batches / args.gradient_accumulation_steps)
	total_steps = (
		min(full_total_steps, args.max_optimizer_steps)
		if args.max_optimizer_steps
		else full_total_steps
	)
	optimizer = _build_optimizer(
		head,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
	)
	scheduler = _build_scheduler(
		optimizer,
		total_steps=total_steps,
		warmup_ratio=args.warmup_ratio,
	)
	scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=args.initial_gradient_scale)
	git_commit = _resolve_git_commit(Path(args.project_root))
	checkpoint_metadata = {
		"training_protocol": config.identity()["protocol"],
		"architecture": config.identity(),
		"dataset": args.dataset,
		"dataset_root": str(Path(args.dataset_root)),
		"candidate_root": str(Path(args.candidate_root)),
		"candidate_bank_manifest_sha256": bank_identities,
		"model_checkpoint_sha256": base_hash,
		"git_commit": git_commit,
		"world_size": world_size,
		"per_device_batch_size": args.per_device_batch_size,
		"gradient_accumulation_steps": args.gradient_accumulation_steps,
		"total_optimizer_steps": total_steps,
		"train_rows": len(loader.dataset),
		"seed": args.seed,
	}
	cursor = TrainingCursor(0, 0, 0, 0, 0, 0)
	resume_source_git_commit: str | None = None
	loaded_gradient_scale: float | None = None
	if resume_checkpoint is not None:
		cursor, metadata = load_training_checkpoint(
			resume_checkpoint,
			head,
			optimizer,
			scheduler,
			rank,
			gradient_scaler=scaler,
			expected_training_protocol=config.identity()["protocol"],
		)
		resume_source_git_commit = _resolve_resume_source_commit(
			current_git_commit=git_commit,
			checkpoint_git_commit=str(metadata.get("git_commit", "")),
			authorized_source_git_commit=args.resume_source_git_commit,
		)
		expected_resume_metadata = dict(checkpoint_metadata)
		expected_resume_metadata["git_commit"] = resume_source_git_commit
		validate_checkpoint_metadata(metadata, expected=expected_resume_metadata)
		if args.resume_gradient_scale is not None:
			loaded_gradient_scale = _lower_loaded_gradient_scale(
				scaler,
				new_scale=args.resume_gradient_scale,
			)
		if cursor.stage != 0 or cursor.gradient_accumulation_step != 0:
			raise ValueError("Resume requires a complete single-stage optimizer step")
		if cursor.global_step >= total_steps:
			raise ValueError("Resume checkpoint already reached the training limit")
		if rank == 0:
			truncate_metric_log(
				output_dir / "train_metrics.jsonl",
				maximum_global_step=cursor.global_step,
			)
	ddp_head = DistributedDataParallel(
		training_model,
		device_ids=[local_rank],
		output_device=local_rank,
		broadcast_buffers=False,
		find_unused_parameters=False,
		gradient_as_bucket_view=True,
	)
	manifest = {
		"scope": "query_only_parallel_world_recurrent_no_lora",
		"dataset": args.dataset,
		"dataset_root": str(args.dataset_root),
		"candidate_root": str(args.candidate_root),
		"candidate_bank_manifest_sha256": bank_identities,
		"train_rows": len(loader.dataset),
		"hostname": socket.gethostname(),
		"git_commit": git_commit,
		"command": sys.argv,
		"resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
		"resume_source_git_commit": resume_source_git_commit,
		"resume_gradient_scale_before_override": loaded_gradient_scale,
		"resume_gradient_scale": args.resume_gradient_scale,
		"world_size": world_size,
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"runtime_precision": "frozen_qwen_fp16_recurrent_fp32_loss_fp32",
		"initial_gradient_scale": args.initial_gradient_scale,
		"attention_implementation": args.attention_implementation,
		"per_device_batch_size": args.per_device_batch_size,
		"contrastive_global_batch_size": args.per_device_batch_size * world_size,
		"gradient_accumulation_steps": args.gradient_accumulation_steps,
		"epochs": args.epochs,
		"total_optimizer_steps": total_steps,
		"learning_rate": args.learning_rate,
		"weight_decay": args.weight_decay,
		"warmup_ratio": args.warmup_ratio,
		"formal_training_log_interval": FORMAL_TRAINING_LOG_INTERVAL,
		"checkpoint_every": args.checkpoint_every,
		"max_checkpoints": 1,
		"no_validation": True,
		"candidate_qwen_forward_calls": 0,
		"hard_negative_mining": {
			"source": "full_immutable_same_gallery_candidate_bank",
			"requested_count": config.hard_negative_count,
			"resolved_count": resolved_hard_negative_count,
			"positive_exclusion": "all_matching_positive_id",
		},
		"query_qwen_forward_calls_per_batch": 1,
		"trainable_parameter_count": head.trainable_parameter_count,
		"base_checkpoint_sha256_before": base_hash,
		"architecture": config.identity(),
		"visual_length_bucketing": {
			"enabled": args.visual_length_buckets > 1,
			"maximum_buckets": args.visual_length_buckets,
			"minimum_bucket_size": args.min_visual_bucket_size,
			"logical_contrastive_batch_unchanged": True,
		},
	}
	if rank == 0:
		manifest_name = (
			"run_manifest.json"
			if resume_checkpoint is None
			else f"resume_manifest_step{cursor.global_step:06d}.json"
		)
		_write_json(
			output_dir / manifest_name,
			manifest,
		)
		_write_json(output_dir / "status.json", {"status": "training"})

	optimizer.zero_grad(set_to_none=True)
	global_step = cursor.global_step
	total_samples = cursor.processed_samples
	training_start = time.perf_counter()
	log_start = training_start
	log_samples = 0
	metric_samples = 0
	optimizer_steps_since_log = 0
	metric_accumulator: dict[str, torch.Tensor] = {}
	direction_counts: Counter[str] = Counter()
	torch.cuda.reset_peak_memory_stats(device)
	for epoch in range(cursor.sampler_epoch, args.epochs):
		sampler.set_epoch(epoch)
		start_batch = cursor.batch_in_epoch if epoch == cursor.sampler_epoch else 0
		sampler.set_batch_range(start_batch, full_loader_batches)
		for relative_batch_index, batch in enumerate(loader):
			batch_index = start_batch + relative_batch_index
			feature_groups = _encode_query_batch(
				batch=batch,
				processor=processor,
				backbone=backbone,
				args=args,
				device=device,
			)
			candidate_embeddings = candidate_stores.lookup(
				batch["candidate_references"],
				device=device,
			).detach()
			base_embeddings = _restore_base_embeddings(
				feature_groups,
				total_rows=len(batch["positive_ids"]),
			)
			hard_negative_embeddings = candidate_stores.mine_hard_negatives(
				base_embeddings.detach(),
				batch["candidate_references"],
				count=resolved_hard_negative_count,
				device=device,
			)
			with recurrent_fp32_context(device.type):
				output = ddp_head(
					feature_groups=feature_groups,
					total_rows=len(batch["positive_ids"]),
				)
				losses = query_recurrent_loss(
					output,
					candidate_embeddings,
					batch["positive_ids"],
					batch["directions"],
					config,
					hard_negative_embeddings=hard_negative_embeddings,
				)
			batch_metrics = {
				**losses,
				**query_recurrent_diagnostics(output, base_embeddings),
				"attention_residual_scale": (
					head.recurrent_cell.attention_residual_scale.float()
				),
				"feed_forward_residual_scale": (
					head.recurrent_cell.feed_forward_residual_scale.float()
				),
			}
			scaler.scale(losses["loss"]).backward()
			scaler.unscale_(optimizer)
			batch_metrics.update(recurrent_gradient_group_norms(head))
			gradient_norm = torch.nn.utils.clip_grad_norm_(
				head.parameters(),
				args.gradient_clip_norm,
			)
			scale_before_step = scaler.get_scale()
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)
			if scaler.get_scale() < scale_before_step:
				raise FloatingPointError(
					"Non-finite recurrent gradients skipped an optimizer step; "
					"resume from checkpoint",
				)
			scheduler.step()
			global_step += 1
			optimizer_steps_since_log += 1
			batch_global_samples = len(batch["positive_ids"]) * world_size
			total_samples += batch_global_samples
			log_samples += batch_global_samples
			metric_samples += len(batch["positive_ids"])
			for key, value in batch_metrics.items():
				detached = value.detach().float()
				metric_accumulator[key] = (
					metric_accumulator.get(key, torch.zeros_like(detached))
					+ detached * len(batch["positive_ids"])
				)
			direction_counts.update(batch["directions"])
			cursor = TrainingCursor(
				0,
				global_step,
				epoch,
				batch_index + 1,
				0,
				total_samples,
			)
			if should_log_training_metrics(
				optimizer_steps_since_log=optimizer_steps_since_log,
				global_step=global_step,
				optimizer_step_limit=total_steps,
				force_every_step=0 < total_steps <= 2,
			):
				torch.cuda.synchronize(device)
				averages, global_metric_samples = _reduce_metrics(
					metric_accumulator,
					local_samples=metric_samples,
				)
				gathered_directions: list[dict[str, int] | None] = [None] * world_size
				dist.all_gather_object(gathered_directions, dict(direction_counts))
				global_directions: Counter[str] = Counter()
				for counts in gathered_directions:
					if counts is None:
						raise RuntimeError("Failed to gather direction counts")
					global_directions.update(counts)
				record = {
					"epoch": epoch,
					"global_step": global_step,
					**averages,
					"gradient_norm": float(gradient_norm.detach().float().item()),
					"learning_rate": float(scheduler.get_last_lr()[0]),
					"samples_per_second": log_samples / (time.perf_counter() - log_start),
					"total_samples": total_samples,
					"gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
					"gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
					"contrastive_global_batch_size": batch_global_samples,
					"logged_global_samples": global_metric_samples,
					"logged_optimizer_steps": optimizer_steps_since_log,
					"direction_counts": dict(global_directions),
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
				optimizer_steps_since_log = 0
				metric_accumulator = {}
				direction_counts = Counter()
			if not args.skip_checkpoint_save and (
				global_step % args.checkpoint_every == 0 or global_step == total_steps
			):
				_save_checkpoint(
					output_dir=output_dir,
					model=head,
					optimizer=optimizer,
					scheduler=scheduler,
					scaler=scaler,
					cursor=cursor,
					metadata=checkpoint_metadata,
					rank=rank,
					world_size=world_size,
				)
			if global_step >= total_steps:
				break
		if global_step >= total_steps:
			break
	if not args.max_optimizer_steps and (
		cursor.batch_in_epoch != full_loader_batches or global_step != full_total_steps
	):
		raise RuntimeError("One-epoch recurrent run did not consume every distributed batch")
	dist.barrier()
	final_hash = None
	if rank == 0 and not args.skip_final_save:
		final_hash = _save_final_head(output_dir / "query_recurrent_model.pt", head, manifest)
	dist.barrier()
	base_hash_after = checkpoint_sha256(base_checkpoint_path) if rank == 0 else None
	if rank == 0 and base_hash_after != base_hash:
		raise RuntimeError("Immutable Qwen checkpoint changed during recurrent training")
	result = None
	if rank == 0:
		result = {
			"status": "passed",
			"dataset": args.dataset,
			"optimizer_steps": global_step,
			"processed_samples": total_samples,
			"runtime_seconds": time.perf_counter() - training_start,
			"trainable_parameter_count": head.trainable_parameter_count,
			"model_checkpoint_sha256_before": base_hash,
			"model_checkpoint_sha256_after": base_hash_after,
			"query_recurrent_model_sha256": final_hash,
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
	parser.add_argument("--candidate-root", type=Path, required=True)
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
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
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--max-optimizer-steps", type=int, default=0)
	parser.add_argument("--max-train-rows", type=int, default=0)
	parser.add_argument("--checkpoint-every", type=int, default=100)
	parser.add_argument("--max-checkpoints", type=int, choices=(1,), default=1)
	parser.add_argument("--resume-checkpoint", type=Path)
	parser.add_argument("--resume-source-git-commit")
	parser.add_argument("--resume-gradient-scale", type=float)
	parser.add_argument("--skip-checkpoint-save", action="store_true")
	parser.add_argument("--skip-final-save", action="store_true")
	parser.add_argument("--num-worlds", type=int, choices=(1, 2, 4), default=4)
	parser.add_argument("--max-recurrent-steps", type=int, choices=(1, 2, 3, 4), default=4)
	parser.add_argument("--perturbation-scale", type=float, default=0.02)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.02)
	parser.add_argument("--temperature", type=float, default=0.02)
	parser.add_argument("--hard-negative-count", type=int, default=32)
	parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
	parser.add_argument("--initial-gradient-scale", type=float, default=4096.0)
	parser.add_argument("--seed", type=int, default=42)
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
		run_training(args)
		return 0
	except KeyboardInterrupt:
		LOGGER.warning("Query recurrent training interrupted")
		status = "interrupted"
		code = 130
	except Exception:
		LOGGER.exception("Query recurrent training failed")
		status = "failed"
		code = 1
	if int(os.environ.get("RANK", "0")) == 0 and Path(args.output_dir).exists():
		_write_json(Path(args.output_dir) / "status.json", {"status": status})
	if dist.is_available() and dist.is_initialized():
		dist.destroy_process_group()
	return code


if __name__ == "__main__":
	raise SystemExit(main())
