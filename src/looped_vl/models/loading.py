"""Load the immutable official checkpoint and recurrent trainable modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from looped_vl.models.config import RecurrentModelConfig
from looped_vl.models.input_processing import RecurrentInputProcessor
from looped_vl.models.latent_slot_inserter import create_or_load_master_slot_initialization
from looped_vl.models.recurrent_qwen3vl_embedding import RecurrentQwen3VLEmbedding
from looped_vl.smoke import load_local_embedding_module


@dataclass(frozen=True)
class LoadedRecurrentComponents:
	"""The recurrent network and its matching official input processor."""

	model: RecurrentQwen3VLEmbedding
	processor: RecurrentInputProcessor


def load_recurrent_components(
	model_root: str | Path,
	master_slot_path: str | Path,
	config: RecurrentModelConfig,
	device: torch.device,
	*,
	enable_lora: bool,
	dtype: torch.dtype = torch.bfloat16,
	attention_implementation: str = "sdpa",
	max_length: int = 8192,
	min_pixels: int = 4 * 32 * 32,
	max_pixels: int = 1800 * 32 * 32,
) -> LoadedRecurrentComponents:
	"""Load local weights without ever writing into the original checkpoint directory."""
	model_path = Path(model_root)
	processor = RecurrentInputProcessor.from_pretrained(
		model_path,
		max_length=max_length,
		min_pixels=min_pixels,
		max_pixels=max_pixels,
	)
	official_module = load_local_embedding_module(model_path)
	base_model = official_module.Qwen3VLForEmbedding.from_pretrained(
		str(model_path),
		trust_remote_code=True,
		dtype=dtype,
		attn_implementation=attention_implementation,
	)
	current_embedding_count = base_model.get_input_embeddings().num_embeddings
	required_embedding_count = max(current_embedding_count, len(processor.processor.tokenizer))
	base_model.resize_token_embeddings(required_embedding_count)
	base_model.requires_grad_(False)
	master_slots = create_or_load_master_slot_initialization(
		path=master_slot_path,
		max_num_latent_slots=config.max_num_latent_slots,
		hidden_size=config.hidden_size,
		seed=config.seed,
		mean=config.latent_init_mean,
		std=config.latent_init_std,
	)
	model = RecurrentQwen3VLEmbedding(
		base_embedding_model=base_model,
		config=config,
		master_slot_initialization=master_slots,
		latent_placeholder_id=processor.latent_placeholder_id,
		pad_token_id=processor.pad_token_id,
		enable_lora=enable_lora,
	).to(device=device, dtype=dtype)
	return LoadedRecurrentComponents(model=model, processor=processor)
