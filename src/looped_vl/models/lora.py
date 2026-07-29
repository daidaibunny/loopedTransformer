"""Minimal LoRA implementation restricted to Decoder Layers 13–20."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

LORA_TARGET_PATHS = (
	("self_attn", "q_proj"),
	("self_attn", "k_proj"),
	("self_attn", "v_proj"),
	("mlp", "up_proj"),
	("mlp", "down_proj"),
	("mlp", "gate_proj"),
)


class LoRALinear(nn.Module):
	"""Frozen base linear layer plus a rank-decomposed trainable residual."""

	def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
		super().__init__()
		if rank <= 0 or alpha <= 0:
			raise ValueError("LoRA rank and alpha must be positive")
		self.base_layer = base_layer
		self.base_layer.requires_grad_(False)
		self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
		self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.scaling = alpha / rank
		nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
		nn.init.zeros_(self.lora_b.weight)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Add the LoRA residual without modifying the frozen base weight."""
		base_output = self.base_layer(inputs)
		lora_output = self.lora_b(self.lora_a(self.dropout(inputs)))
		return base_output + self.scaling * lora_output


def inject_loop_layer_lora(
	layers: Sequence[nn.Module],
	layer_start: int,
	layer_end: int,
	rank: int,
	alpha: int,
	dropout: float,
) -> tuple[str, ...]:
	"""Wrap only q/k/v and MLP projections in layers with indexes [12, 20)."""
	if (layer_start, layer_end) != (12, 20):
		raise ValueError("v1.0 LoRA injection must target layers [12, 20)")
	injected: list[str] = []
	for layer_index in range(layer_start, layer_end):
		layer = layers[layer_index]
		for parent_name, child_name in LORA_TARGET_PATHS:
			parent = getattr(layer, parent_name)
			base_layer = getattr(parent, child_name)
			if not isinstance(base_layer, nn.Linear):
				raise TypeError(
					f"Expected Linear at layer {layer_index} {parent_name}.{child_name}",
				)
			setattr(
				parent,
				child_name,
				LoRALinear(base_layer, rank=rank, alpha=alpha, dropout=dropout),
			)
			injected.append(f"layers.{layer_index}.{parent_name}.{child_name}")
	return tuple(injected)
