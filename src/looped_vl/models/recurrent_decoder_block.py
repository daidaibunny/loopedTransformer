"""Mask and cache helpers for dynamic-only recurrent decoder passes."""

from __future__ import annotations

import torch


def build_dynamic_attention_mask(
	prefix_attention_mask: torch.Tensor,
	dynamic_token_count: int,
	dtype: torch.dtype,
) -> torch.Tensor:
	"""Build an additive mask for prefix evidence plus bidirectional slots."""
	if prefix_attention_mask.ndim != 2:
		raise ValueError("prefix_attention_mask must be rank two")
	if dynamic_token_count <= 0:
		raise ValueError("dynamic_token_count must be positive")
	batch_size = prefix_attention_mask.shape[0]
	prefix_visible = prefix_attention_mask.to(torch.bool)[:, None, None, :].expand(
		batch_size,
		1,
		dynamic_token_count,
		-1,
	)
	dynamic_visible = torch.ones(
		(batch_size, 1, dynamic_token_count, dynamic_token_count),
		dtype=torch.bool,
		device=prefix_attention_mask.device,
	)
	visible = torch.cat((prefix_visible, dynamic_visible), dim=-1)
	mask = torch.full(
		visible.shape,
		torch.finfo(dtype).min,
		dtype=dtype,
		device=prefix_attention_mask.device,
	)
	return mask.masked_fill(visible, 0.0)


def build_full_sequence_bidirectional_slot_mask(
	*,
	attention_mask: torch.Tensor,
	slot_positions: torch.Tensor,
	dtype: torch.dtype,
) -> torch.Tensor:
	"""Keep Qwen causality except that every slot can read every other slot."""
	if attention_mask.ndim != 2:
		raise ValueError("attention_mask must be rank two")
	if slot_positions.ndim != 2 or slot_positions.shape[0] != attention_mask.shape[0]:
		raise ValueError("slot_positions must have shape [batch, slots]")
	if slot_positions.shape[1] == 0:
		raise ValueError("At least one slot is required for bidirectional slot attention")
	batch_size, sequence_length = attention_mask.shape
	if (
		slot_positions.min().item() < 0
		or slot_positions.max().item() >= sequence_length
	):
		raise ValueError("slot_positions must lie inside the full sequence")
	if not attention_mask.to(torch.bool).gather(1, slot_positions).all():
		raise ValueError("Every slot position must identify a valid token")
	position = torch.arange(sequence_length, device=attention_mask.device)
	causal_visible = position[None, :] <= position[:, None]
	valid_keys = attention_mask.to(torch.bool)[:, None, :]
	visible = causal_visible[None].expand(batch_size, -1, -1) & valid_keys
	is_slot = (
		position[None, None, :] == slot_positions[:, :, None]
	).any(dim=1)
	slot_to_slot = is_slot[:, :, None] & is_slot[:, None, :]
	visible = visible | slot_to_slot
	mask = torch.full(
		(batch_size, 1, sequence_length, sequence_length),
		torch.finfo(dtype).min,
		dtype=dtype,
		device=attention_mask.device,
	)
	return mask.masked_fill(visible[:, None], 0.0)


def detach_prefix_key_values(
	prefix_key: torch.Tensor,
	prefix_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Turn cached prefix evidence into a strict stop-gradient boundary."""
	return prefix_key.detach(), prefix_value.detach()
