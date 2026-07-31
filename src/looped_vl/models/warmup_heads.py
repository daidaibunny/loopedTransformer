"""Warm-start retrieval supervision head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WarmupEmbeddingHead(nn.Module):
	"""Mean-pool every latent slot, project to 2048, and L2 normalize."""

	def __init__(self, hidden_size: int = 2048) -> None:
		super().__init__()
		self.projection = nn.Linear(hidden_size, hidden_size, bias=True)

	def forward(self, slot_hidden_states: torch.Tensor) -> torch.Tensor:
		"""Produce the auxiliary normalized slot embedding."""
		if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
			raise ValueError("At least one contextual slot is required")
		pooled = slot_hidden_states.mean(dim=1)
		return F.normalize(self.projection(pooled), p=2, dim=-1)
