"""Insert learnable latent slots before each sample's final valid token."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class AugmentedSequence:
	"""Token tensors and exact dynamic-token positions after slot insertion."""

	input_ids: torch.Tensor
	attention_mask: torch.Tensor
	prefix_lengths: torch.Tensor
	slot_positions: torch.Tensor
	eos_positions: torch.Tensor


def augment_before_last_valid_token(
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	num_latent_slots: int,
	latent_placeholder_id: int,
	pad_token_id: int,
) -> AugmentedSequence:
	"""Insert placeholders immediately before every sample's last valid token."""
	if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
		raise ValueError("input_ids and attention_mask must have the same rank-two shape")
	if num_latent_slots < 0:
		raise ValueError("num_latent_slots cannot be negative")
	valid_lengths = attention_mask.to(torch.long).sum(dim=-1)
	if (valid_lengths <= 0).any():
		raise ValueError("Every sequence must contain at least one valid token")
	prefix_lengths = valid_lengths - 1
	batch_size, old_width = input_ids.shape
	new_width = old_width + num_latent_slots
	eos_positions = prefix_lengths + num_latent_slots
	new_positions = torch.arange(new_width, device=input_ids.device)[None, :]
	prefix_boundary = prefix_lengths[:, None]
	slot_boundary = prefix_boundary + num_latent_slots
	source_positions = torch.where(
		new_positions < prefix_boundary,
		new_positions,
		new_positions - num_latent_slots,
	).clamp(min=0, max=old_width - 1)
	augmented_ids = input_ids.gather(1, source_positions.expand(batch_size, -1))
	slot_offsets = torch.arange(num_latent_slots, device=input_ids.device)[None, :]
	slot_positions = prefix_boundary + slot_offsets
	if num_latent_slots:
		slot_mask = (new_positions >= prefix_boundary) & (new_positions < slot_boundary)
		augmented_ids = augmented_ids.masked_fill(slot_mask, latent_placeholder_id)
	valid_augmented_mask = new_positions < (valid_lengths + num_latent_slots)[:, None]
	augmented_ids = augmented_ids.masked_fill(~valid_augmented_mask, pad_token_id)
	augmented_mask = valid_augmented_mask.to(attention_mask.dtype)
	return AugmentedSequence(
		input_ids=augmented_ids,
		attention_mask=augmented_mask,
		prefix_lengths=prefix_lengths,
		slot_positions=slot_positions,
		eos_positions=eos_positions,
	)


def create_or_load_master_slot_initialization(
	path: str | Path,
	max_num_latent_slots: int,
	hidden_size: int,
	seed: int,
	mean: float,
	std: float,
) -> torch.Tensor:
	"""Create the shared seed-42 slot tensor once, then validate every later load."""
	target = Path(path)
	expected_shape = (1, max_num_latent_slots, hidden_size)
	if target.exists():
		tensor = torch.load(target, map_location="cpu", weights_only=True)
		if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
			raise ValueError(f"Invalid master slot initialization at {target}")
		return tensor
	target.parent.mkdir(parents=True, exist_ok=True)
	generator = torch.Generator(device="cpu")
	generator.manual_seed(seed)
	tensor = torch.empty(expected_shape, dtype=torch.float32)
	tensor.normal_(mean=mean, std=std, generator=generator)
	torch.save(tensor, target)
	return tensor
