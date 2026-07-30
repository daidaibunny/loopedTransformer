"""Shared zero-output connector between recurrent passes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
	"""RMS normalization compatible with PyTorch releases before 2.4."""

	def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.eps = eps

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"""Normalize in float32, then restore the model activation precision."""
		input_dtype = hidden_states.dtype
		float_states = hidden_states.to(torch.float32)
		variance = float_states.pow(2).mean(dim=-1, keepdim=True)
		normalized = float_states * torch.rsqrt(variance + self.eps)
		return self.weight * normalized.to(input_dtype)


class RecurrentConnector(nn.Module):
	"""Map layer-20 dynamic states back to the stable layer-12 anchor space."""

	def __init__(self, hidden_size: int = 2048, bottleneck_dim: int = 512) -> None:
		super().__init__()
		self.normalization = RMSNorm(hidden_size, eps=1e-6)
		self.down_projection = nn.Linear(hidden_size, bottleneck_dim, bias=True)
		self.up_projection = nn.Linear(bottleneck_dim, hidden_size, bias=True)
		nn.init.normal_(self.down_projection.weight, mean=0.0, std=0.02)
		nn.init.zeros_(self.down_projection.bias)
		nn.init.zeros_(self.up_projection.weight)
		nn.init.zeros_(self.up_projection.bias)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		"""Return the shared token-wise recurrent residual."""
		normalized = self.normalization(hidden_states)
		return self.up_projection(F.silu(self.down_projection(normalized)))
