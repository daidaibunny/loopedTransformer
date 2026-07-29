"""Two-GPU Stage 1 then Stage 2 training for recurrent Qwen3-VL embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from looped_vl.data import DEFAULT_DATASET_ROOT, LoopedVLMixtureDataset
from looped_vl.models.config import RecurrentModelConfig
from looped_vl.models.latent_slot_inserter import create_or_load_master_slot_initialization
from looped_vl.models.loading import load_recurrent_components
from looped_vl.smoke import checkpoint_sha256
from looped_vl.training.checkpointing import (
	TrainingCursor,
	capture_rng_state,
	load_training_checkpoint,
	save_training_checkpoint,
)
from looped_vl.training.config import TrainingStageConfig
from looped_vl.training.data import close_training_batch_images, paired_training_collate
from looped_vl.training.model import RecurrentTrainingModel
from looped_vl.training.optimizer import build_optimizer_and_scheduler
from looped_vl.training.reproducibility import seed_everything
from looped_vl.training.trainability import audit_gradient_scope, configure_trainable_parameters

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: Any) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_json_line(path: Path, value: Any) -> None:
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(value, sort_keys=True) + "\n")


def _git_commit(project_root: Path) -> str:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.strip()


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


def _build_loader(
	args: argparse.Namespace,
	rank: int,
	world_size: int,
	generator: torch.Generator,
) -> tuple[DataLoader[dict[str, Any]], DistributedSampler[Any]]:
	dataset = LoopedVLMixtureDataset(
		args.dataset_root,
		"train",
		args.gqa_materialized_root,
	)
	sampler = DistributedSampler(
		dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True,
		seed=42,
		drop_last=True,
	)
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": args.per_device_batch_size,
		"sampler": sampler,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": paired_training_collate,
		"drop_last": True,
		"pin_memory": False,
		"generator": generator,
	}
	if args.num_workers > 0:
		loader_kwargs["prefetch_factor"] = args.prefetch_factor
		loader_kwargs["persistent_workers"] = True
	return DataLoader(**loader_kwargs), sampler


def _set_training_modes(model: RecurrentTrainingModel) -> None:
	model.train()
	model.encoder.base_embedding_model.eval()
	model.encoder.warmup_embedding_head.train()
	model.encoder.warmup_semantic_head.train()
	model.encoder.recurrent_connector.train()
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
) -> Path:
	rank_rng_states = _gather_rank_rng_states(world_size)
	path = output_dir / "checkpoints" / (
		f"stage{cursor.stage}_step{cursor.global_step:06d}.pt"
	)
	if rank == 0 and not path.exists():
		save_training_checkpoint(
			path=path,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			cursor=cursor,
			rank_rng_states=rank_rng_states,
			metadata=metadata,
		)
	dist.barrier()
	return path


def _mean_accumulator(accumulator: dict[str, float], count: int) -> dict[str, float]:
	return {key: value / count for key, value in accumulator.items()}


def _train_stage(
	*,
	args: argparse.Namespace,
	stage_config: TrainingStageConfig,
	training_model: RecurrentTrainingModel,
	processor: Any,
	loader: DataLoader[dict[str, Any]],
	sampler: DistributedSampler[Any],
	rank: int,
	world_size: int,
	local_rank: int,
	device: torch.device,
	output_dir: Path,
	checkpoint_metadata: dict[str, Any],
	resume_checkpoint: Path | None,
) -> TrainingCursor:
	trainable_names = configure_trainable_parameters(training_model.encoder, stage_config.stage)
	_set_training_modes(training_model)
	optimizer, scheduler = build_optimizer_and_scheduler(training_model, stage_config)
	gradient_accumulation_steps = stage_config.gradient_accumulation_steps(
		args.per_device_batch_size,
		world_size,
	)
	optimizer_step_limit = stage_config.steps
	if args.smoke_optimizer_steps:
		optimizer_step_limit = min(optimizer_step_limit, args.smoke_optimizer_steps)
		gradient_accumulation_steps = args.smoke_gradient_accumulation_steps
	cursor = TrainingCursor(
		stage=stage_config.stage,
		global_step=0,
		sampler_epoch=0,
		batch_in_epoch=0,
		gradient_accumulation_step=0,
	)
	if resume_checkpoint is not None:
		cursor, _ = load_training_checkpoint(
			path=resume_checkpoint,
			model=training_model,
			optimizer=optimizer,
			scheduler=scheduler,
			rank=rank,
		)
		if cursor.stage != stage_config.stage:
			raise ValueError("Resume checkpoint stage does not match active stage")
	ddp_model = DistributedDataParallel(
		training_model,
		device_ids=[local_rank],
		output_device=local_rank,
		broadcast_buffers=False,
		find_unused_parameters=False,
	)
	if rank == 0:
		_write_json(
			output_dir / f"stage{stage_config.stage}_trainable_parameters.json",
			{
				"stage": stage_config.stage,
				"trainable_names": trainable_names,
				"trainable_parameter_count": sum(
					parameter.numel()
					for parameter in training_model.parameters()
					if parameter.requires_grad
				),
				"gradient_accumulation_steps": gradient_accumulation_steps,
				"effective_batch_size": (
					args.per_device_batch_size * world_size * gradient_accumulation_steps
				),
			},
		)
	optimizer.zero_grad(set_to_none=True)
	accumulator: dict[str, float] = defaultdict(float)
	accumulated_batches = 0
	source_counts: Counter[str] = Counter()
	direction_counts: Counter[str] = Counter()
	step_start = time.perf_counter()
	latest_diagnostics: dict[str, Any] = {}
	epoch = cursor.sampler_epoch
	while cursor.global_step < optimizer_step_limit:
		sampler.set_epoch(epoch)
		for batch_index, batch in enumerate(loader):
			if epoch == cursor.sampler_epoch and batch_index < cursor.batch_in_epoch:
				close_training_batch_images(batch)
				continue
			try:
				combined_inputs = batch["query_inputs"] + batch["candidate_inputs"]
				processed_inputs = processor.prepare(combined_inputs, device=device)
			finally:
				close_training_batch_images(batch)
			is_accumulation_boundary = (
				cursor.gradient_accumulation_step + 1 == gradient_accumulation_steps
			)
			synchronization_context = (
				nullcontext() if is_accumulation_boundary else ddp_model.no_sync()
			)
			with synchronization_context:
				step_output = ddp_model(
					local_batch_size=args.per_device_batch_size,
					semantic_targets=batch["semantic_targets"],
					sources=batch["sources"],
					stage=stage_config.stage,
					**processed_inputs,
				)
				loss = step_output["total_loss"] / gradient_accumulation_steps
				loss.backward()
			for key in (
				"total_loss",
				"final_infonce",
				"slot_infonce",
				"semantic_decoder_ce",
				"slot_diversity",
				"fusion_gate",
				"late_fusion_attention_entropy",
				"slot_pairwise_cosine",
				"connector_output_norm",
			):
				accumulator[key] += float(step_output[key].detach().float().item())
			accumulated_batches += 1
			source_counts.update(batch["sources"])
			direction_counts.update(batch["directions"])
			latest_diagnostics = {
				"recurrent_pass_cosine": [
					float(value.detach().float().item())
					for value in step_output["recurrent_pass_cosine"]
				],
				"recurrent_pass_relative_update": [
					float(value.detach().float().item())
					for value in step_output["recurrent_pass_relative_update"]
				],
			}
			if is_accumulation_boundary:
				gradient_audit = None
				if cursor.global_step == 0:
					gradient_audit = audit_gradient_scope(
						training_model.encoder,
						allowed_names=trainable_names,
					)
					if rank == 0:
						_write_json(
							output_dir / f"stage{stage_config.stage}_gradient_audit.json",
							gradient_audit,
						)
				gradient_norm = torch.nn.utils.clip_grad_norm_(
					(
						parameter
						for parameter in training_model.parameters()
						if parameter.requires_grad
					),
					stage_config.gradient_clip_norm,
				)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)
				global_step = cursor.global_step + 1
				averages = _mean_accumulator(accumulator, accumulated_batches)
				elapsed = time.perf_counter() - step_start
				global_samples = (
					args.per_device_batch_size * world_size * accumulated_batches
				)
				log_record = {
					"stage": stage_config.stage,
					"global_step": global_step,
					**averages,
					**latest_diagnostics,
					"gradient_norm": float(gradient_norm.detach().float().item()),
					"learning_rate": float(scheduler.get_last_lr()[0]),
					"gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
					"gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
					"samples_per_second": global_samples / elapsed,
					"source_counts": dict(source_counts),
					"direction_counts": dict(direction_counts),
					"slot_collapse": averages["slot_pairwise_cosine"] > 0.98,
					"pooling_collapse": (
						training_model.encoder.config.num_latent_slots > 1
						and averages["late_fusion_attention_entropy"] < 0.1
					),
					"recurrence_unused": (
						global_step > 100 and averages["connector_output_norm"] < 1e-6
					),
					"gradient_scope_audited": gradient_audit is not None,
				}
				if rank == 0:
					_append_json_line(output_dir / "train_metrics.jsonl", log_record)
					LOGGER.info(
						"stage=%d step=%d/%d loss=%.5f grad=%.5f samples/s=%.3f",
						stage_config.stage,
						global_step,
						optimizer_step_limit,
						averages["total_loss"],
						float(gradient_norm.detach().float().item()),
						global_samples / elapsed,
					)
				cursor = TrainingCursor(
					stage=stage_config.stage,
					global_step=global_step,
					sampler_epoch=epoch,
					batch_in_epoch=batch_index + 1,
					gradient_accumulation_step=0,
				)
				accumulator = defaultdict(float)
				accumulated_batches = 0
				source_counts = Counter()
				direction_counts = Counter()
				step_start = time.perf_counter()
				if (
					cursor.global_step % args.checkpoint_every == 0
					or cursor.global_step == optimizer_step_limit
				):
					checkpoint_path = _save_checkpoint(
						output_dir=output_dir,
						model=training_model,
						optimizer=optimizer,
						scheduler=scheduler,
						cursor=cursor,
						rank=rank,
						world_size=world_size,
						metadata=checkpoint_metadata,
					)
					if rank == 0:
						_write_json(
							output_dir / "latest_checkpoint.json",
							{"path": str(checkpoint_path), "cursor": asdict(cursor)},
						)
				if cursor.global_step >= optimizer_step_limit:
					break
			else:
				cursor = TrainingCursor(
					stage=stage_config.stage,
					global_step=cursor.global_step,
					sampler_epoch=epoch,
					batch_in_epoch=batch_index + 1,
					gradient_accumulation_step=cursor.gradient_accumulation_step + 1,
				)
		if cursor.global_step >= optimizer_step_limit:
			break
		epoch += 1
		cursor = TrainingCursor(
			stage=stage_config.stage,
			global_step=cursor.global_step,
			sampler_epoch=epoch,
			batch_in_epoch=0,
			gradient_accumulation_step=cursor.gradient_accumulation_step,
		)
	del ddp_model
	dist.barrier()
	return cursor


def run_training(args: argparse.Namespace) -> dict[str, Any] | None:
	"""Run Stage 1 and Stage 2 under torchrun with immutable base checkpoints."""
	if args.start_stage > args.end_stage:
		raise ValueError("start_stage cannot be greater than end_stage")
	if args.checkpoint_every <= 0:
		raise ValueError("checkpoint_every must be positive")
	rank, world_size, local_rank, device = _initialize_distributed(args.expected_world_size)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	generator = seed_everything(42)
	output_dir = Path(args.output_dir)
	if rank == 0:
		if output_dir.exists():
			raise FileExistsError(f"Output directory already exists: {output_dir}")
		(output_dir / "checkpoints").mkdir(parents=True)
		_write_json(output_dir / "status.json", {"status": "initializing"})
	dist.barrier()
	project_root = Path(args.project_root)
	model_config = RecurrentModelConfig.from_yaml(args.model_config)
	model_checkpoint_path = Path(args.model_root) / "model.safetensors"
	semantic_checkpoint_path = Path(args.semantic_decoder_root) / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(model_checkpoint_path) if rank == 0 else None
	semantic_checkpoint_hash = (
		checkpoint_sha256(semantic_checkpoint_path) if rank == 0 else None
	)
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
		enable_lora=True,
		semantic_decoder_root=args.semantic_decoder_root,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	training_model = RecurrentTrainingModel(components.model)
	loader, sampler = _build_loader(args, rank, world_size, generator)
	stage_configs = {
		1: TrainingStageConfig.from_yaml(args.stage1_config),
		2: TrainingStageConfig.from_yaml(args.stage2_config),
	}
	manifest = {
		"scope": "recurrent_latent_slot_qwen3vl_v1",
		"hostname": socket.gethostname(),
		"git_commit": _git_commit(project_root),
		"command": sys.argv,
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"world_size": world_size,
		"precision": "bf16",
		"seed": 42,
		"model_config": asdict(model_config),
		"stage_configs": {stage: asdict(config) for stage, config in stage_configs.items()},
		"dataset_root": str(args.dataset_root),
		"train_rows": len(loader.dataset),
		"per_device_batch_size": args.per_device_batch_size,
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
		"model_checkpoint_sha256_before": checkpoint_hash_before,
		"semantic_decoder_checkpoint_sha256": semantic_checkpoint_hash,
		"semantic_decoder_root": str(args.semantic_decoder_root),
		"gpus": [
			{
				"rank": rank,
				"logical_device": local_rank,
				"name": torch.cuda.get_device_name(local_rank),
			}
		],
		"smoke_optimizer_steps": args.smoke_optimizer_steps,
	}
	all_manifests: list[dict[str, Any] | None] = [None for _ in range(world_size)]
	dist.all_gather_object(all_manifests, manifest)
	checkpoint_metadata = {
		"git_commit": manifest["git_commit"],
		"model_checkpoint_sha256": checkpoint_hash_before,
		"semantic_decoder_checkpoint_sha256": semantic_checkpoint_hash,
		"model_config": manifest["model_config"],
	}
	if rank == 0:
		manifest["gpus"] = [item["gpus"][0] for item in all_manifests if item is not None]
		_write_json(output_dir / "run_manifest.json", manifest)
		_write_json(output_dir / "status.json", {"status": "training", "stage": 1})
	training_start = time.perf_counter()
	final_cursor = None
	for stage in range(args.start_stage, args.end_stage + 1):
		if rank == 0:
			_write_json(output_dir / "status.json", {"status": "training", "stage": stage})
		final_cursor = _train_stage(
			args=args,
			stage_config=stage_configs[stage],
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
			resume_checkpoint=args.resume_checkpoint if stage == args.start_stage else None,
		)
	checkpoint_hash_after = checkpoint_sha256(model_checkpoint_path) if rank == 0 else None
	if rank == 0 and checkpoint_hash_after != checkpoint_hash_before:
		raise RuntimeError("Original Qwen checkpoint changed during training")
	semantic_checkpoint_hash_after = (
		checkpoint_sha256(semantic_checkpoint_path) if rank == 0 else None
	)
	if rank == 0 and semantic_checkpoint_hash_after != semantic_checkpoint_hash:
		raise RuntimeError("Original semantic decoder checkpoint changed during training")
	result = None
	if rank == 0:
		result = {
			"status": "passed",
			"final_cursor": asdict(final_cursor) if final_cursor is not None else None,
			"runtime_seconds": time.perf_counter() - training_start,
			"model_checkpoint_sha256_before": checkpoint_hash_before,
			"model_checkpoint_sha256_after": checkpoint_hash_after,
			"semantic_decoder_checkpoint_sha256_before": semantic_checkpoint_hash,
			"semantic_decoder_checkpoint_sha256_after": semantic_checkpoint_hash_after,
		}
		_write_json(output_dir / "training_result.json", result)
		_write_json(output_dir / "status.json", {"status": "passed"})
	dist.barrier()
	dist.destroy_process_group()
	return result


def parse_args() -> argparse.Namespace:
	"""Parse the complete two-stage training command."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--project-root", type=Path, default=Path("/mnt/afs/liyiwei/looped_vl"))
	parser.add_argument("--model-config", type=Path, default=Path("configs/base.yaml"))
	parser.add_argument("--stage1-config", type=Path, default=Path("configs/stage1.yaml"))
	parser.add_argument("--stage2-config", type=Path, default=Path("configs/stage2.yaml"))
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument(
		"--semantic-decoder-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/models/Qwen3-0.6B"),
	)
	parser.add_argument(
		"--master-slot-path",
		type=Path,
		default=Path("/mnt/afs/liyiwei/looped_vl/artifacts/master_slot_init_seed42.pt"),
	)
	parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
	parser.add_argument(
		"--gqa-materialized-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced"),
	)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=2)
	parser.add_argument("--start-stage", type=int, choices=(1, 2), default=1)
	parser.add_argument("--end-stage", type=int, choices=(1, 2), default=2)
	parser.add_argument("--per-device-batch-size", type=int, default=1)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--checkpoint-every", type=int, default=500)
	parser.add_argument("--resume-checkpoint", type=Path)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument("--smoke-optimizer-steps", type=int, default=0)
	parser.add_argument("--smoke-gradient-accumulation-steps", type=int, default=1)
	return parser.parse_args()


def main() -> int:
	"""Run distributed training and write failure status when possible."""
	args = parse_args()
	try:
		run_training(args)
		return 0
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
