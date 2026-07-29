"""Warm-up retrieval and semantic supervision heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_slot_groups(slot_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return reasoning slots first and retrieval slots second, sharing K=1."""
	if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
		raise ValueError("At least one contextual slot is required")
	slot_count = slot_hidden_states.shape[1]
	if slot_count == 1:
		return slot_hidden_states, slot_hidden_states
	reasoning_count = slot_count // 2
	return (
		slot_hidden_states[:, :reasoning_count],
		slot_hidden_states[:, reasoning_count:],
	)


class WarmupEmbeddingHead(nn.Module):
	"""Mean-pool retrieval slots, project to 2048, and L2 normalize."""

	def __init__(self, hidden_size: int = 2048) -> None:
		super().__init__()
		self.projection = nn.Linear(hidden_size, hidden_size, bias=True)

	def forward(self, slot_hidden_states: torch.Tensor) -> torch.Tensor:
		"""Produce the auxiliary normalized slot embedding."""
		_, embedding_slots = split_slot_groups(slot_hidden_states)
		pooled = embedding_slots.mean(dim=1)
		return F.normalize(self.projection(pooled), p=2, dim=-1)


@dataclass(frozen=True)
class SemanticDecoderOutput:
	"""Token-level semantic supervision result."""

	loss: torch.Tensor
	token_count: int


class WarmupSemanticDecoderHead(nn.Module):
	"""Qwen3-0.6B teacher-forced decoder conditioned on reasoning slots."""

	def __init__(self, decoder_model: nn.Module, tokenizer: Any, encoder_hidden_size: int) -> None:
		super().__init__()
		decoder_hidden_size = int(decoder_model.config.hidden_size)
		if decoder_hidden_size != 1024:
			raise ValueError(
				f"Semantic decoder hidden size must be 1024, found {decoder_hidden_size}",
			)
		self.decoder_model = decoder_model
		self.tokenizer = tokenizer
		self.slot_projector = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=True)
		self.decoder_model.config.use_cache = False

	@classmethod
	def from_pretrained(
		cls,
		model_root: str | Path,
		device: torch.device,
		dtype: torch.dtype,
		encoder_hidden_size: int = 2048,
	) -> WarmupSemanticDecoderHead:
		"""Load the local Qwen3-0.6B training-only semantic decoder."""
		tokenizer = AutoTokenizer.from_pretrained(str(model_root), padding_side="right")
		decoder_model = AutoModelForCausalLM.from_pretrained(
			str(model_root),
			dtype=dtype,
			attn_implementation="sdpa",
		).to(device)
		decoder_model.gradient_checkpointing_enable()
		return cls(
			decoder_model=decoder_model,
			tokenizer=tokenizer,
			encoder_hidden_size=encoder_hidden_size,
		).to(device=device, dtype=dtype)

	def forward(
		self,
		slot_hidden_states: torch.Tensor,
		targets: list[str],
		sources: list[str],
	) -> SemanticDecoderOutput:
		"""Compute token cross entropy for projected reasoning-slot prefixes."""
		if len(targets) != slot_hidden_states.shape[0] or len(sources) != len(targets):
			raise ValueError("Semantic targets and sources must match the slot batch")
		reasoning_slots, _ = split_slot_groups(slot_hidden_states)
		projected_slots = self.slot_projector(reasoning_slots)
		target_ids = self._tokenize_targets(targets, sources, projected_slots.device)
		batch_size, max_target_length = target_ids.shape
		bos_token_id = self.tokenizer.bos_token_id
		if bos_token_id is None:
			bos_token_id = self.tokenizer.eos_token_id
		if bos_token_id is None:
			raise RuntimeError("Semantic decoder tokenizer needs BOS or EOS")
		teacher_input_ids = torch.full_like(target_ids, self.tokenizer.pad_token_id)
		teacher_input_ids[:, 0] = bos_token_id
		if max_target_length > 1:
			teacher_input_ids[:, 1:] = target_ids[:, :-1].clamp_min(0)
		target_mask = target_ids != -100
		teacher_input_mask = torch.zeros_like(target_mask)
		teacher_input_mask[:, 0] = True
		if max_target_length > 1:
			teacher_input_mask[:, 1:] = target_mask[:, :-1]
		teacher_embeddings = self.decoder_model.get_input_embeddings()(
			teacher_input_ids,
		)
		inputs_embeds = torch.cat((projected_slots, teacher_embeddings), dim=1)
		attention_mask = torch.cat(
			(
				torch.ones(
					(batch_size, projected_slots.shape[1]),
					dtype=teacher_input_mask.dtype,
					device=teacher_input_mask.device,
				),
				teacher_input_mask,
			),
			dim=1,
		)
		outputs = self.decoder_model(
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			use_cache=False,
		)
		prediction_logits = outputs.logits[
			:,
			projected_slots.shape[1] : projected_slots.shape[1] + max_target_length,
		]
		loss = F.cross_entropy(
			prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
			target_ids.reshape(-1),
			ignore_index=-100,
		)
		return SemanticDecoderOutput(loss=loss, token_count=int(target_mask.sum().item()))

	def _tokenize_targets(
		self,
		targets: list[str],
		sources: list[str],
		device: torch.device,
	) -> torch.Tensor:
		encoded: list[list[int]] = []
		for target, source in zip(targets, sources, strict=True):
			max_length = 64 if source == "coco" else 32
			token_ids = self.tokenizer.encode(
				target,
				add_special_tokens=False,
				truncation=True,
				max_length=max_length,
			)
			if not token_ids:
				raise ValueError(f"Semantic target tokenized to empty text for {source}")
			encoded.append(token_ids)
		max_length = max(len(token_ids) for token_ids in encoded)
		padded = torch.full((len(encoded), max_length), -100, dtype=torch.long, device=device)
		for index, token_ids in enumerate(encoded):
			padded[index, : len(token_ids)] = torch.tensor(token_ids, device=device)
		return padded
