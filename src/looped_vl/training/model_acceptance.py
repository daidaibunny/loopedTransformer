"""GPU acceptance checks for official equivalence and recurrent forward structure."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from looped_vl.data import DEFAULT_DATASET_ROOT, LoopedVLMixtureDataset
from looped_vl.models.config import RecurrentModelConfig
from looped_vl.models.loading import load_recurrent_components
from looped_vl.smoke import checkpoint_sha256

LOGGER = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
	if isinstance(value, torch.Tensor):
		if value.numel() == 1:
			return float(value.detach().float().item())
		return value.detach().float().cpu().tolist()
	if isinstance(value, tuple):
		return [_json_value(item) for item in value]
	return value


def _prepare_sample(
	dataset: LoopedVLMixtureDataset,
	index: int,
) -> tuple[dict[str, Any], Any]:
	sample = dataset[index]
	instruction = (
		"Retrieve the image that best matches the caption."
		if sample.source == "coco"
		else "Retrieve the correct answer to the visual question."
	)
	return (
		{
			"text": sample.text,
			"image": sample.image,
			"instruction": instruction,
		},
		sample,
	)


def run_model_acceptance(args: argparse.Namespace) -> dict[str, Any]:
	"""Run one exact official baseline or full recurrent forward acceptance check."""
	if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
		raise RuntimeError("Acceptance check requires exactly one visible CUDA device")
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)
	device = torch.device("cuda", 0)
	base_config = RecurrentModelConfig.from_yaml(args.config)
	if args.mode == "base_equivalence":
		config = base_config.with_variant(num_latent_slots=0, num_total_loop_passes=1)
	else:
		config = base_config
	dataset = LoopedVLMixtureDataset(
		args.dataset_root,
		args.split,
		args.gqa_materialized_root,
	)
	model_input, sample = _prepare_sample(dataset, args.index)
	checkpoint_path = Path(args.model_root) / "model.safetensors"
	checkpoint_hash_before = checkpoint_sha256(checkpoint_path)
	components = load_recurrent_components(
		model_root=args.model_root,
		master_slot_path=args.master_slot_path,
		config=config,
		device=device,
		enable_lora=args.enable_lora,
		attention_implementation=args.attention_implementation,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	components.model.eval()
	processed = components.processor.prepare([model_input], device=device)
	with torch.inference_mode():
		output = components.model(**processed)
		repeated_output = components.model(**processed)
	if not torch.equal(output.embeddings, repeated_output.embeddings):
		raise RuntimeError("Repeated forward output changed under the fixed seed")
	if output.embeddings.shape != (1, config.hidden_size):
		raise RuntimeError(f"Unexpected embedding shape: {output.embeddings.shape}")
	if not torch.isfinite(output.embeddings).all():
		raise RuntimeError("Embedding contains non-finite values")
	norm_error = float((output.embeddings.float().norm(dim=-1) - 1).abs().max().item())
	if norm_error >= 5e-3:
		raise RuntimeError(f"Embedding norm error is too large: {norm_error}")
	result: dict[str, Any] = {
		"status": "passed",
		"mode": args.mode,
		"sample_id": sample.sample_id,
		"source": sample.source,
		"embedding_shape": list(output.embeddings.shape),
		"slot_shape": list(output.slot_hidden_states.shape),
		"norm_error": norm_error,
		"repeated_forward_max_absolute_error": float(
			(output.embeddings.float() - repeated_output.embeddings.float()).abs().max().item(),
		),
		"diagnostics": {key: _json_value(value) for key, value in output.diagnostics.items()},
	}
	if args.mode == "base_equivalence":
		with torch.inference_mode():
			official_output = components.model.base_embedding_model(**processed)
			official_positions = processed["attention_mask"].sum(dim=-1) - 1
			batch_index = torch.arange(official_output.last_hidden_state.shape[0], device=device)
			official_embedding = F.normalize(
				official_output.last_hidden_state[batch_index, official_positions],
				p=2,
				dim=-1,
			)
		max_absolute_error = float(
			(output.embeddings.float() - official_embedding.float()).abs().max().item(),
		)
		if max_absolute_error >= 1e-2:
			raise RuntimeError(
				f"BF16 official-equivalence error {max_absolute_error} is not below 1e-2",
			)
		result["official_max_absolute_error"] = max_absolute_error
	else:
		expected_dynamic_tokens = config.num_latent_slots + 1
		expected_counts = (expected_dynamic_tokens,) * config.num_extra_loop_passes
		if tuple(output.diagnostics["extra_pass_dynamic_token_counts"]) != expected_counts:
			raise RuntimeError("Extra passes did not update exactly slots plus EOS")
		if any(output.diagnostics["prefix_cache_requires_grad"]):
			raise RuntimeError("A recurrent prefix K/V cache still tracks gradients")
		if tuple(output.diagnostics["deepstack_layer_indices"]) != (0, 1, 2):
			raise RuntimeError("DeepStack was not restricted to language layers 0, 1, and 2")
	checkpoint_hash_after = checkpoint_sha256(checkpoint_path)
	if checkpoint_hash_after != checkpoint_hash_before:
		raise RuntimeError("Original Qwen checkpoint changed during acceptance")
	result["checkpoint_sha256_before"] = checkpoint_hash_before
	result["checkpoint_sha256_after"] = checkpoint_hash_after
	sample.image.close()
	return result


def parse_args() -> argparse.Namespace:
	"""Parse model acceptance arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--mode", choices=("base_equivalence", "full_forward"), required=True)
	parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument(
		"--master-slot-path",
		type=Path,
		default=Path("/mnt/afs/liyiwei/loopedTransformer/artifacts/master_slot_init_seed42.pt"),
	)
	parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
	parser.add_argument(
		"--gqa-materialized-root",
		type=Path,
		default=Path("/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced"),
	)
	parser.add_argument("--split", choices=("train", "validation"), default="train")
	parser.add_argument("--index", type=int, default=17)
	parser.add_argument("--enable-lora", action="store_true")
	parser.add_argument(
		"--attention-implementation",
		choices=("flash_attention_2", "sdpa", "eager"),
		default="flash_attention_2",
	)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument("--output-json", type=Path, required=True)
	return parser.parse_args()


def main() -> int:
	"""Execute and persist one model acceptance result."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = parse_args()
	try:
		result = run_model_acceptance(args)
		args.output_json.parent.mkdir(parents=True, exist_ok=True)
		args.output_json.write_text(
			json.dumps(result, indent=2, sort_keys=True) + "\n",
			encoding="utf-8",
		)
		print(json.dumps(result, sort_keys=True))
		return 0
	except Exception:
		LOGGER.exception("Model acceptance failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
