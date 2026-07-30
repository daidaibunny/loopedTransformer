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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from looped_vl.baseline.data import (
	BASELINE_DATASETS,
	BaselineManifestDataset,
	baseline_pair_collate,
	close_baseline_batch_images,
)
from looped_vl.baseline.model import (
	BASELINE_LORA_ALPHA,
	BASELINE_LORA_RANK,
	BASELINE_LORA_TARGETS,
	BaselineInputProcessor,
	BaselineLoRATrainingModel,
	load_lora_training_model,
)
from looped_vl.smoke import checkpoint_sha256

LOGGER = logging.getLogger("baseline_train")


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


def _build_loader(
	args: argparse.Namespace,
	*,
	rank: int,
	world_size: int,
	generator: torch.Generator,
) -> tuple[DataLoader[dict[str, Any]], DistributedSampler[Any]]:
	dataset = BaselineManifestDataset(
		args.dataset_root,
		"train",
		max_rows=args.max_train_rows,
	)
	sampler = DistributedSampler(
		dataset,
		num_replicas=world_size,
		rank=rank,
		shuffle=True,
		seed=args.seed,
		drop_last=False,
	)
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
	if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
		raise ValueError("Batch size and gradient accumulation must be positive")
	rank, world_size, local_rank, device = _initialize_distributed(
		args.expected_world_size,
	)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	generator = _seed_everything(args.seed, rank)
	output_dir = Path(args.output_dir)
	if rank == 0:
		if output_dir.exists():
			raise FileExistsError(f"Training output already exists: {output_dir}")
		output_dir.mkdir(parents=True)
		_write_json(output_dir / "status.json", {"status": "initializing"})
	dist.barrier()

	model_root = Path(args.model_root)
	checkpoint_path = model_root / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(checkpoint_path) if rank == 0 else None
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
	).to(device)
	training_model = BaselineLoRATrainingModel(
		peft_model,
		temperature=args.temperature,
	)
	training_model.train()
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
	scaler = torch.cuda.amp.GradScaler(enabled=True)
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
		"hostname": socket.gethostname(),
		"git_commit": _resolve_git_commit(Path(args.project_root)),
		"command": sys.argv,
		"world_size": world_size,
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"runtime_precision": "fp16",
		"attention_implementation": args.attention_implementation,
		"gradient_checkpointing": args.gradient_checkpointing,
		"per_device_batch_size": args.per_device_batch_size,
		"gradient_accumulation_steps": args.gradient_accumulation_steps,
		"effective_global_batch_size": (
			args.per_device_batch_size
			* world_size
			* args.gradient_accumulation_steps
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
		"seed": args.seed,
		"lora": {
			"rank": BASELINE_LORA_RANK,
			"alpha": BASELINE_LORA_ALPHA,
			"dropout": 0.0,
			"target_modules": BASELINE_LORA_TARGETS,
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
		_write_json(output_dir / "run_manifest.json", manifest)
		_write_json(output_dir / "status.json", {"status": "training"})
	optimizer.zero_grad(set_to_none=True)
	global_step = 0
	total_samples = 0
	training_start = time.perf_counter()
	log_start = training_start
	log_samples = 0
	torch.cuda.reset_peak_memory_stats(device)
	for epoch in range(args.epochs):
		sampler.set_epoch(epoch)
		for batch_index, batch in enumerate(loader):
			group_start = (
				batch_index // args.gradient_accumulation_steps
			) * args.gradient_accumulation_steps
			group_size = min(
				args.gradient_accumulation_steps,
				len(loader) - group_start,
			)
			is_boundary = (
				(batch_index + 1) % args.gradient_accumulation_steps == 0
				or batch_index + 1 == len(loader)
			)
			try:
				query_inputs = processor.prepare(
					batch["query_inputs"],
					device=device,
				)
				candidate_inputs = processor.prepare(
					batch["candidate_inputs"],
					device=device,
				)
			finally:
				close_baseline_batch_images(batch)
			synchronization_context = nullcontext() if is_boundary else ddp_model.no_sync()
			with synchronization_context:
				with torch.autocast(device_type="cuda", dtype=torch.float16):
					output = ddp_model(
						query_inputs=query_inputs,
						candidate_inputs=candidate_inputs,
						positive_ids=batch["positive_ids"],
					)
					loss = output["loss"] / group_size
				scaler.scale(loss).backward()
			batch_global_samples = len(batch["positive_ids"]) * world_size
			total_samples += batch_global_samples
			log_samples += batch_global_samples
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
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)
			scheduler.step()
			global_step += 1
			torch.cuda.synchronize(device)
			elapsed = time.perf_counter() - log_start
			record = {
				"epoch": epoch,
				"global_step": global_step,
				"loss": float(output["loss"].detach().float().item()),
				"gradient_norm": float(gradient_norm.detach().float().item()),
				"learning_rate": float(scheduler.get_last_lr()[0]),
				"samples_per_second": log_samples / elapsed,
				"total_samples": total_samples,
				"gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
				"gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
				"query_norm": float(output["query_norm"].detach().float().item()),
				"candidate_norm": float(output["candidate_norm"].detach().float().item()),
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
			if global_step >= total_steps:
				break
		if global_step >= total_steps:
			break
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
	parser.add_argument("--per-device-batch-size", type=int, default=4)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--max-optimizer-steps", type=int, default=0)
	parser.add_argument("--max-train-rows", type=int, default=0)
	parser.add_argument("--skip-adapter-save", action="store_true")
	parser.add_argument("--learning-rate", type=float, default=5e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.02)
	parser.add_argument("--temperature", type=float, default=0.02)
	parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
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
	except Exception:
		logging.basicConfig(level=logging.INFO)
		LOGGER.exception("Baseline LoRA training failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
