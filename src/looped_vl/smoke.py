"""Frozen Qwen3-VL-Embedding smoke test for the mixture DataLoader."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch.utils.data import DataLoader

from looped_vl.data import (
	DEFAULT_DATASET_ROOT,
	LoopedVLMixtureDataset,
	mixture_collate,
	select_source_balanced_indices,
)
from looped_vl.runtime import (
	ATTENTION_IMPLEMENTATIONS,
	RUNTIME_PRECISIONS,
	resolve_attention_implementation,
	resolve_torch_dtype,
)

LOGGER = logging.getLogger("smoke")


def freeze_model(model: torch.nn.Module) -> None:
	"""Put a model in evaluation mode and disable gradients for every parameter."""
	model.eval()
	model.requires_grad_(False)


def assert_model_frozen(model: torch.nn.Module) -> None:
	"""Fail unless the model is in evaluation mode with no trainable parameters."""
	trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
	if trainable:
		raise RuntimeError(f"Model still has trainable parameters: {trainable[:5]}")
	if model.training:
		raise RuntimeError("Model is still in training mode")


def checkpoint_sha256(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
	"""Stream a checkpoint file and return its SHA-256 digest."""
	digest = hashlib.sha256()
	with Path(path).open("rb") as handle:
		while chunk := handle.read(chunk_size):
			digest.update(chunk)
	return digest.hexdigest()


def load_local_embedding_module(model_root: Path) -> ModuleType:
	"""Load the checkpoint's pinned embedding implementation by absolute path."""
	script_path = model_root / "scripts/qwen3_vl_embedding.py"
	if not script_path.is_file():
		raise FileNotFoundError(f"Missing embedding implementation: {script_path}")
	module_name = "looped_vl_local_qwen3_vl_embedding"
	spec = importlib.util.spec_from_file_location(module_name, script_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load embedding implementation: {script_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
	"""Read a balanced batch and run one frozen embedding forward pass."""
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required; CPU fallback is disabled")
	if torch.cuda.device_count() != 1:
		raise RuntimeError(
			"Smoke expects exactly one visible GPU; set CUDA_VISIBLE_DEVICES to one idle GPU",
		)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	resolved_attention_implementation = resolve_attention_implementation(
		args.attention_implementation,
	)
	runtime_dtype = resolve_torch_dtype(args.runtime_precision)

	model_root = Path(args.model_root)
	checkpoint_path = model_root / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(checkpoint_path)
	dataset = LoopedVLMixtureDataset(
		args.dataset_root,
		args.split,
		args.gqa_materialized_root,
	)
	indices = select_source_balanced_indices(dataset, args.per_source)
	loader = DataLoader(
		dataset,
		batch_size=len(indices),
		sampler=indices,
		num_workers=args.num_workers,
		collate_fn=mixture_collate,
		pin_memory=False,
	)
	load_start = time.perf_counter()
	batch = next(iter(loader))
	data_load_seconds = time.perf_counter() - load_start

	module = load_local_embedding_module(model_root)
	torch.cuda.reset_peak_memory_stats()
	model_load_start = time.perf_counter()
	embedder = module.Qwen3VLEmbedder(
		model_name_or_path=str(model_root),
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
		torch_dtype=runtime_dtype,
		attn_implementation=resolved_attention_implementation,
	)
	freeze_model(embedder.model)
	assert_model_frozen(embedder.model)
	model_load_seconds = time.perf_counter() - model_load_start

	forward_start = time.perf_counter()
	with torch.inference_mode():
		embeddings = embedder.process(batch["model_inputs"])
	forward_seconds = time.perf_counter() - forward_start
	assert_model_frozen(embedder.model)

	if embeddings.ndim != 2 or embeddings.shape[0] != len(indices):
		raise RuntimeError(f"Unexpected embedding shape: {tuple(embeddings.shape)}")
	if not torch.isfinite(embeddings).all():
		raise RuntimeError("Embeddings contain non-finite values")
	norms = torch.linalg.vector_norm(embeddings.float(), dim=1)
	if not torch.allclose(norms, torch.ones_like(norms), atol=5e-3, rtol=5e-3):
		raise RuntimeError(f"Embeddings are not unit normalized: {norms.tolist()}")
	if embeddings.requires_grad or embeddings.grad_fn is not None:
		raise RuntimeError("Inference embeddings unexpectedly track gradients")
	similarity = embeddings.float() @ embeddings.float().T
	if not torch.isfinite(similarity).all():
		raise RuntimeError("Similarity matrix contains non-finite values")

	checkpoint_hash_after = checkpoint_sha256(checkpoint_path)
	if checkpoint_hash_after != checkpoint_hash_before:
		raise RuntimeError("Model checkpoint hash changed during frozen smoke")
	parameter_count = sum(parameter.numel() for parameter in embedder.model.parameters())
	trainable_parameter_count = sum(
		parameter.numel()
		for parameter in embedder.model.parameters()
		if parameter.requires_grad
	)
	properties = torch.cuda.get_device_properties(0)
	return {
		"status": "passed",
		"dataset_split": args.split,
		"dataset_length": len(dataset),
		"sample_ids": batch["sample_ids"],
		"sources": batch["sources"],
		"reasoning_depths": batch["reasoning_depths"],
		"embedding_shape": list(embeddings.shape),
		"embedding_norms": norms.tolist(),
		"similarity_matrix": similarity.tolist(),
		"parameter_count": parameter_count,
		"trainable_parameter_count": trainable_parameter_count,
		"checkpoint_sha256_before": checkpoint_hash_before,
		"checkpoint_sha256_after": checkpoint_hash_after,
		"cuda_visible_device_count": torch.cuda.device_count(),
		"cuda_device_name": properties.name,
		"peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
		"data_load_seconds": data_load_seconds,
		"model_load_seconds": model_load_seconds,
		"forward_seconds": forward_seconds,
		"max_length": args.max_length,
		"min_pixels": args.min_pixels,
		"max_pixels": args.max_pixels,
		"runtime_precision": args.runtime_precision,
		"requested_attention_implementation": args.attention_implementation,
		"resolved_attention_implementation": resolved_attention_implementation,
		"seed": args.seed,
	}


def parse_args() -> argparse.Namespace:
	"""Parse smoke-test arguments."""
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
	parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
	parser.add_argument("--per-source", type=int, default=1)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=20260729)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--min-pixels", type=int, default=64 * 64)
	parser.add_argument("--max-pixels", type=int, default=512 * 512)
	parser.add_argument(
		"--attention-implementation",
		choices=ATTENTION_IMPLEMENTATIONS,
		default="auto",
	)
	parser.add_argument(
		"--runtime-precision",
		choices=RUNTIME_PRECISIONS,
		default="bf16",
	)
	parser.add_argument("--output-json")
	return parser.parse_args()


def main() -> int:
	"""Execute the smoke test and emit JSON to stdout and optionally a file."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = parse_args()
	try:
		result = run_smoke(args)
		serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
		print(serialized, end="")
		if args.output_json:
			Path(args.output_json).write_text(serialized, encoding="utf-8")
		return 0
	except Exception:
		LOGGER.exception("Frozen Qwen smoke failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
