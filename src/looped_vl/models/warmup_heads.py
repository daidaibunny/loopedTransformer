"""Shared per-round auxiliary slot retrieval head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
	"""Mean-pool all slots, RMS-normalize, project to 256, and L2-normalize."""

	def __init__(self, hidden_size: int = 2048, output_size: int = 256) -> None:
		super().__init__()
		self.normalization = RMSNorm(hidden_size)
		self.projection = nn.Linear(hidden_size, output_size, bias=False)

	def forward(self, slot_hidden_states: torch.Tensor) -> torch.Tensor:
		"""Produce one shared auxiliary retrieval embedding for a recurrent round."""
		if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
			raise ValueError("At least one contextual slot is required")
		pooled = slot_hidden_states.mean(dim=1)
		normalized = self.normalization(pooled)
		return F.normalize(self.projection(normalized), p=2, dim=-1)
