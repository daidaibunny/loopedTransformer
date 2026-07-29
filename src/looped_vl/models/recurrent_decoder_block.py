"""Mask and cache helpers for dynamic-only recurrent decoder passes."""

from __future__ import annotations

import torch


def build_dynamic_attention_mask(
	prefix_attention_mask: torch.Tensor,
	dynamic_token_count: int,
	dtype: torch.dtype,
) -> torch.Tensor:
	"""Build an additive mask for prefix evidence plus causal slots and EOS."""
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
	dynamic_visible = torch.tril(
		torch.ones(
			(dynamic_token_count, dynamic_token_count),
			dtype=torch.bool,
			device=prefix_attention_mask.device,
		),
	)[None, None].expand(batch_size, 1, -1, -1)
	visible = torch.cat((prefix_visible, dynamic_visible), dim=-1)
	mask = torch.full(
		visible.shape,
		torch.finfo(dtype).min,
		dtype=dtype,
		device=prefix_attention_mask.device,
	)
	return mask.masked_fill(visible, 0.0)


def detach_prefix_key_values(
	prefix_key: torch.Tensor,
	prefix_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Turn cached prefix evidence into a strict stop-gradient boundary."""
	return prefix_key.detach(), prefix_value.detach()
