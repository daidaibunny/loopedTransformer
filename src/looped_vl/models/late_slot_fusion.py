"""EOS-conditioned pooling over final contextual latent slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LateFusionOutput:
	"""Fused pre-normalization embedding and pooling diagnostics."""

	fused_embedding: torch.Tensor
	attention_weights: torch.Tensor
	attention_entropy: torch.Tensor
	gate: torch.Tensor


class EOSConditionedSlotFusion(nn.Module):
	"""Pool slots using the final valid token as the query and a zero gate."""

	def __init__(self, hidden_size: int = 2048, attention_dim: int = 256) -> None:
		super().__init__()
		self.attention_dim = attention_dim
		self.query_projection = nn.Linear(hidden_size, attention_dim, bias=False)
		self.key_projection = nn.Linear(hidden_size, attention_dim, bias=False)
		self.value_projection = nn.Linear(hidden_size, attention_dim, bias=False)
		self.output_projection = nn.Linear(attention_dim, hidden_size, bias=False)
		self.gamma = nn.Parameter(torch.zeros(()))
		for layer in (
			self.query_projection,
			self.key_projection,
			self.value_projection,
			self.output_projection,
		):
			nn.init.xavier_uniform_(layer.weight)

	def forward(
		self,
		eos_hidden_state: torch.Tensor,
		slot_hidden_states: torch.Tensor,
	) -> LateFusionOutput:
		"""Fuse all slots after the final Qwen decoder normalization."""
		if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
			raise ValueError("Late fusion requires at least one latent slot")
		query = self.query_projection(eos_hidden_state)
		keys = self.key_projection(slot_hidden_states)
		values = self.value_projection(slot_hidden_states)
		logits = torch.einsum("bd,bkd->bk", query, keys) / self.attention_dim**0.5
		weights_fp32 = torch.softmax(logits.float(), dim=-1)
		attention_weights = weights_fp32.to(values.dtype)
		pooled = torch.einsum("bk,bkd->bd", attention_weights, values)
		delta = self.output_projection(pooled)
		gate = torch.tanh(self.gamma)
		fused = eos_hidden_state + gate * delta
		entropy = -(weights_fp32 * weights_fp32.clamp_min(1e-12).log()).sum(dim=-1)
		return LateFusionOutput(
			fused_embedding=fused,
			attention_weights=attention_weights,
			attention_entropy=entropy,
			gate=gate,
		)
