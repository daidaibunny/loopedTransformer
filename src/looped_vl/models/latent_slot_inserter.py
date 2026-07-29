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
	augmented_ids = torch.full(
		(batch_size, new_width),
		pad_token_id,
		dtype=input_ids.dtype,
		device=input_ids.device,
	)
	augmented_mask = torch.zeros(
		(batch_size, new_width),
		dtype=attention_mask.dtype,
		device=attention_mask.device,
	)
	slot_positions = torch.empty(
		(batch_size, num_latent_slots),
		dtype=torch.long,
		device=input_ids.device,
	)
	eos_positions = prefix_lengths + num_latent_slots
	for batch_index in range(batch_size):
		prefix_length = int(prefix_lengths[batch_index].item())
		valid_length = int(valid_lengths[batch_index].item())
		augmented_ids[batch_index, :prefix_length] = input_ids[
			batch_index,
			:prefix_length,
		]
		if num_latent_slots:
			positions = torch.arange(
				prefix_length,
				prefix_length + num_latent_slots,
				device=input_ids.device,
			)
			slot_positions[batch_index] = positions
			augmented_ids[batch_index, positions] = latent_placeholder_id
		new_eos_position = int(eos_positions[batch_index].item())
		augmented_ids[batch_index, new_eos_position] = input_ids[
			batch_index,
			valid_length - 1,
		]
		augmented_mask[batch_index, : valid_length + num_latent_slots] = 1
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
