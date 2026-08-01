"""Shared per-round auxiliary slot retrieval head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _parameter_free_rms_norm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""RMS-normalize activations without introducing a learned scale."""
	float_states = hidden_states.float()
	variance = float_states.square().mean(dim=-1, keepdim=True)
	return float_states * torch.rsqrt(variance + eps)


def eos_conditioned_slot_attention_embedding(
	slot_hidden_states: torch.Tensor,
	conditioning_eos_hidden_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return a parameter-free hidden-size slot embedding and EOS attention weights."""
	if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
		raise ValueError("At least one contextual slot is required")
	if conditioning_eos_hidden_state.shape != (
		slot_hidden_states.shape[0],
		slot_hidden_states.shape[2],
	):
		raise ValueError("Conditioning EOS shape must match the slot batch and hidden size")
	normalized_eos = _parameter_free_rms_norm(conditioning_eos_hidden_state)
	normalized_slots = _parameter_free_rms_norm(slot_hidden_states)
	scores = torch.einsum("bd,bkd->bk", normalized_eos, normalized_slots)
	scores = scores / math.sqrt(slot_hidden_states.shape[-1])
	attention_weights = F.softmax(scores, dim=-1)
	pooled_slots = torch.einsum(
		"bk,bkd->bd",
		attention_weights.to(slot_hidden_states.dtype),
		slot_hidden_states,
	)
	return F.normalize(_parameter_free_rms_norm(pooled_slots), p=2, dim=-1), attention_weights


class RMSNorm(nn.Module):
	"""Parameter-efficient RMS normalization for auxiliary slot states."""

	def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.eps = eps

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"""Normalize in float32 and restore the activation dtype."""
		input_dtype = hidden_states.dtype
		float_states = hidden_states.float()
		variance = float_states.square().mean(dim=-1, keepdim=True)
		normalized = float_states * torch.rsqrt(variance + self.eps)
		return self.weight * normalized.to(input_dtype)


class AuxiliarySlotRetrievalHead(nn.Module):
	"""Use fixed layer-20 EOS to softly select slots for auxiliary retrieval."""

	def __init__(self, hidden_size: int = 2048, output_size: int = 256) -> None:
		super().__init__()
		self.normalization = RMSNorm(hidden_size)
		self.projection = nn.Linear(hidden_size, output_size, bias=False)

	def pool_slots(
		self,
		slot_hidden_states: torch.Tensor,
		conditioning_eos_hidden_state: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return the EOS-weighted raw slot sum and its normalized attention weights."""
		if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
			raise ValueError("At least one contextual slot is required")
		if (
			conditioning_eos_hidden_state.ndim != 2
			or conditioning_eos_hidden_state.shape[0] != slot_hidden_states.shape[0]
			or conditioning_eos_hidden_state.shape[1] != slot_hidden_states.shape[2]
		):
			raise ValueError(
				"Conditioning EOS must have shape [batch, hidden] matching the slots",
			)
		normalized_eos = self.normalization(conditioning_eos_hidden_state).float()
		normalized_slots = self.normalization(slot_hidden_states).float()
		scores = torch.einsum("bd,bkd->bk", normalized_eos, normalized_slots)
		scores = scores / math.sqrt(slot_hidden_states.shape[-1])
		attention_weights = F.softmax(scores, dim=-1)
		pooled = torch.einsum(
			"bk,bkd->bd",
			attention_weights.to(slot_hidden_states.dtype),
			slot_hidden_states,
		)
		return pooled, attention_weights

	def forward(
		self,
		slot_hidden_states: torch.Tensor,
		conditioning_eos_hidden_state: torch.Tensor,
	) -> torch.Tensor:
		"""Produce one EOS-conditioned auxiliary embedding for a recurrent round."""
		pooled, _ = self.pool_slots(
			slot_hidden_states,
			conditioning_eos_hidden_state,
		)
		normalized = self.normalization(pooled)
		return F.normalize(self.projection(normalized), p=2, dim=-1)
