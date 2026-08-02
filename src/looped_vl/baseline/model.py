"""Official Qwen3-VL embedding forward path with a PEFT LoRA overlay."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from looped_vl.baseline.losses import multi_positive_symmetric_info_nce
from looped_vl.query_recurrent.candidate_store import (
	CandidateReference,
	CandidateStoreCollection,
)
from looped_vl.query_recurrent.losses import multi_query_symmetric_info_nce
from looped_vl.smoke import load_local_embedding_module

BASELINE_LORA_RANK = 32
BASELINE_LORA_ALPHA = 32
BASELINE_LORA_LAST_FOUR_DECODER_LAYERS = (24, 25, 26, 27)
BASELINE_LORA_TARGETS = (
	"q_proj",
	"v_proj",
	"k_proj",
	"up_proj",
	"down_proj",
	"gate_proj",
)


def build_lora_config(
	*,
	decoder_layer_indices: tuple[int, ...] | None = None,
) -> LoraConfig:
	"""Return the model-specific LoRA configuration published by Qwen."""
	if decoder_layer_indices is not None:
		if not decoder_layer_indices:
			raise ValueError("decoder_layer_indices must not be empty")
		if any(index < 0 for index in decoder_layer_indices):
			raise ValueError("decoder_layer_indices must be non-negative")
		if tuple(sorted(set(decoder_layer_indices))) != decoder_layer_indices:
			raise ValueError("decoder_layer_indices must be sorted and unique")
	return LoraConfig(
		r=BASELINE_LORA_RANK,
		lora_alpha=BASELINE_LORA_ALPHA,
		target_modules=list(BASELINE_LORA_TARGETS),
		lora_dropout=0.0,
		bias="none",
		task_type=TaskType.FEATURE_EXTRACTION,
		layers_to_transform=(
			list(decoder_layer_indices) if decoder_layer_indices is not None else None
		),
		layers_pattern="layers" if decoder_layer_indices is not None else None,
	)


def describe_lora_decoder_scope(config: LoraConfig) -> dict[str, Any]:
	"""Return a stable result identity for an all-layer or selected-layer adapter."""
	indices = config.layers_to_transform
	if indices is None:
		return {"scope": "all_decoder_layers", "decoder_layer_indices": None}
	resolved_indices = [indices] if isinstance(indices, int) else list(indices)
	scope = (
		"last_4_decoder_layers"
		if tuple(resolved_indices) == BASELINE_LORA_LAST_FOUR_DECODER_LAYERS
		else "selected_decoder_layers"
	)
	return {"scope": scope, "decoder_layer_indices": resolved_indices}


class BaselineInputProcessor:
	"""Apply the official instruction-aware Qwen3-VL embedding input format."""

	def __init__(
		self,
		processor: Qwen3VLProcessor,
		*,
		max_length: int,
		min_pixels: int,
		max_pixels: int,
	) -> None:
		self.processor = processor
		self.max_length = max_length
		self.min_pixels = min_pixels
		self.max_pixels = max_pixels

	@classmethod
	def from_pretrained(
		cls,
		model_root: str | Path,
		*,
		max_length: int = 8192,
		min_pixels: int = 4 * 32 * 32,
		max_pixels: int = 1800 * 32 * 32,
	) -> BaselineInputProcessor:
		processor = Qwen3VLProcessor.from_pretrained(
			str(model_root),
			padding_side="right",
		)
		return cls(
			processor,
			max_length=max_length,
			min_pixels=min_pixels,
			max_pixels=max_pixels,
		)

	def _format(self, model_input: dict[str, Any]) -> list[dict[str, Any]]:
		instruction = str(model_input.get("instruction", "")).strip()
		if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
			instruction += "."
		content: list[dict[str, Any]] = []
		conversation = [
			{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": instruction or "Represent the user's input.",
					},
				],
			},
			{"role": "user", "content": content},
		]
		image = model_input.get("image")
		if image is not None:
			if isinstance(image, Image.Image):
				image_value: str | Image.Image = image
			elif isinstance(image, str):
				image_value = image if image.startswith(("http://", "https://")) else (
					f"file://{image}"
				)
			else:
				raise TypeError(f"Unsupported image input: {type(image)}")
			content.append(
				{
					"type": "image",
					"image": image_value,
					"min_pixels": self.min_pixels,
					"max_pixels": self.max_pixels,
				},
			)
		text = model_input.get("text")
		if text:
			content.append({"type": "text", "text": str(text)})
		if not content:
			content.append({"type": "text", "text": "NULL"})
		return conversation

	def prepare(
		self,
		model_inputs: list[dict[str, Any]],
		*,
		device: torch.device,
	) -> dict[str, torch.Tensor]:
		"""Format and batch one homogeneous query or candidate tower."""
		conversations = [self._format(model_input) for model_input in model_inputs]
		text = self.processor.apply_chat_template(
			conversations,
			add_generation_prompt=True,
			tokenize=False,
		)
		images, video_inputs, video_kwargs = process_vision_info(
			conversations,
			image_patch_size=16,
			return_video_metadata=True,
			return_video_kwargs=True,
		)
		if video_inputs is not None:
			videos, video_metadata = zip(*video_inputs, strict=True)
		else:
			videos, video_metadata = None, None
		processed = self.processor(
			text=text,
			images=images,
			videos=list(videos) if videos is not None else None,
			video_metadata=list(video_metadata) if video_metadata is not None else None,
			truncation=True,
			max_length=self.max_length,
			padding=True,
			do_resize=False,
			return_tensors="pt",
			**video_kwargs,
		)
		return {
			key: value.to(device=device, non_blocking=True)
			for key, value in processed.items()
			if isinstance(value, torch.Tensor)
		}


def pool_last_token(
	last_hidden_state: torch.Tensor,
	attention_mask: torch.Tensor,
) -> torch.Tensor:
	"""Extract the final unpadded token exactly as the official embedder does."""
	last_positions = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
	rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
	return F.normalize(last_hidden_state[rows, last_positions].float(), p=2, dim=-1)


def encode_grouped_baseline_batches(
	*,
	model: nn.Module,
	processed_batches: tuple[dict[str, torch.Tensor], ...],
	original_indices: tuple[tuple[int, ...], ...],
	total_rows: int,
) -> torch.Tensor:
	"""Encode separate padding groups and restore their shared logical batch order."""
	if len(processed_batches) != len(original_indices):
		raise ValueError("Processed batches and index groups must match")
	flat_indices = tuple(index for indices in original_indices for index in indices)
	if tuple(sorted(flat_indices)) != tuple(range(total_rows)):
		raise ValueError("Grouped indices must cover every logical row exactly once")
	group_embeddings = []
	for processed_inputs in processed_batches:
		output = model(**processed_inputs)
		group_embeddings.append(
			pool_last_token(
				output.last_hidden_state,
				processed_inputs["attention_mask"],
			),
		)
	grouped_embeddings = torch.cat(group_embeddings, dim=0)
	restore_order = torch.argsort(
		torch.tensor(flat_indices, device=grouped_embeddings.device),
	)
	return grouped_embeddings[restore_order]


def load_lora_training_model(
	model_root: str | Path,
	*,
	dtype: torch.dtype,
	attention_implementation: str,
	gradient_checkpointing: bool,
	decoder_layer_indices: tuple[int, ...] | None = None,
) -> PeftModel:
	"""Load the immutable base checkpoint and expose only LoRA parameters as trainable."""
	module = load_local_embedding_module(Path(model_root))
	base_model = module.Qwen3VLForEmbedding.from_pretrained(
		str(model_root),
		trust_remote_code=True,
		dtype=dtype,
		attn_implementation=attention_implementation,
	)
	base_model.requires_grad_(False)
	base_model.config.use_cache = False
	if gradient_checkpointing:
		base_model.gradient_checkpointing_enable(
			gradient_checkpointing_kwargs={"use_reentrant": False},
		)
		base_model.enable_input_require_grads()
	model = get_peft_model(
		base_model,
		build_lora_config(decoder_layer_indices=decoder_layer_indices),
	)
	return model


def load_lora_evaluation_model(
	model_root: str | Path,
	adapter_root: str | Path,
	*,
	dtype: torch.dtype,
	attention_implementation: str,
) -> PeftModel:
	"""Load a saved adapter without making either it or the base checkpoint trainable."""
	module = load_local_embedding_module(Path(model_root))
	base_model = module.Qwen3VLForEmbedding.from_pretrained(
		str(model_root),
		trust_remote_code=True,
		dtype=dtype,
		attn_implementation=attention_implementation,
	)
	model = PeftModel.from_pretrained(base_model, str(adapter_root), is_trainable=False)
	model.eval()
	model.requires_grad_(False)
	return model


def load_frozen_evaluation_model(
	model_root: str | Path,
	*,
	dtype: torch.dtype,
	attention_implementation: str,
) -> nn.Module:
	"""Load the untouched base checkpoint with every parameter frozen."""
	module = load_local_embedding_module(Path(model_root))
	model = module.Qwen3VLForEmbedding.from_pretrained(
		str(model_root),
		trust_remote_code=True,
		dtype=dtype,
		attn_implementation=attention_implementation,
	)
	model.config.use_cache = False
	model.eval()
	model.requires_grad_(False)
	return model


class BaselineLoRATrainingModel(nn.Module):
	"""Keep both embedding towers and multi-positive loss inside the DDP forward."""

	def __init__(self, model: PeftModel, temperature: float = 0.02) -> None:
		super().__init__()
		self.model = model
		self.temperature = temperature

	def forward(
		self,
		*,
		local_batch_size: int,
		processed_batches: tuple[dict[str, torch.Tensor], ...],
		original_indices: tuple[tuple[int, ...], ...],
		positive_ids: list[str],
	) -> dict[str, torch.Tensor]:
		if local_batch_size <= 0:
			raise ValueError("local_batch_size must be positive")
		combined_embeddings = encode_grouped_baseline_batches(
			model=self.model,
			processed_batches=processed_batches,
			original_indices=original_indices,
			total_rows=2 * local_batch_size,
		)
		query_embeddings, candidate_embeddings = combined_embeddings.split(local_batch_size)
		loss = multi_positive_symmetric_info_nce(
			query_embeddings,
			candidate_embeddings,
			positive_ids,
			self.temperature,
		)
		return {
			"loss": loss,
			"query_norm": query_embeddings.norm(dim=1).mean(),
			"candidate_norm": candidate_embeddings.norm(dim=1).mean(),
		}


class QueryOnlyLoRATrainingModel(nn.Module):
	"""Train only the LoRA query tower against immutable frozen candidate banks."""

	def __init__(
		self,
		model: PeftModel,
		candidate_stores: CandidateStoreCollection,
		*,
		temperature: float = 0.02,
		hard_negative_count: int = 32,
	) -> None:
		super().__init__()
		if hard_negative_count < 0:
			raise ValueError("hard_negative_count cannot be negative")
		self.model = model
		self.candidate_stores = candidate_stores
		self.temperature = temperature
		self.hard_negative_count = hard_negative_count

	def forward(
		self,
		*,
		local_batch_size: int,
		processed_batches: tuple[dict[str, torch.Tensor], ...],
		original_indices: tuple[tuple[int, ...], ...],
		candidate_embeddings: torch.Tensor,
		candidate_references: list[CandidateReference],
		positive_ids: list[str],
		directions: list[str],
	) -> dict[str, torch.Tensor]:
		"""Encode query inputs once; candidate Qwen is never called."""
		if local_batch_size <= 0:
			raise ValueError("local_batch_size must be positive")
		if candidate_embeddings.shape != (local_batch_size, 2048):
			raise ValueError("Candidate embeddings must have shape [local_batch_size, 2048]")
		if len(candidate_references) != local_batch_size:
			raise ValueError("Candidate references must match the local batch")
		query_embeddings = encode_grouped_baseline_batches(
			model=self.model,
			processed_batches=processed_batches,
			original_indices=original_indices,
			total_rows=local_batch_size,
		)
		hard_negatives = self.candidate_stores.mine_hard_negatives(
			query_embeddings.detach(),
			candidate_references,
			count=self.hard_negative_count,
			device=query_embeddings.device,
		)
		(loss,) = multi_query_symmetric_info_nce(
			(query_embeddings,),
			candidate_embeddings,
			positive_ids,
			directions,
			temperature=self.temperature,
			hard_negative_embeddings=hard_negatives,
		)
		return {
			"loss": loss,
			"query_norm": query_embeddings.norm(dim=1).mean(),
			"candidate_norm": candidate_embeddings.norm(dim=1).mean(),
		}
