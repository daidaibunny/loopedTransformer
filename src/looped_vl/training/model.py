"""One DDP forward containing the encoder, auxiliary heads, and all losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from looped_vl.models.recurrent_qwen3vl_embedding import RecurrentQwen3VLEmbedding
from looped_vl.training.losses import slot_diversity_loss
from looped_vl.training.step import (
	compose_training_loss,
	distributed_multi_positive_info_nce_losses,
)


class _EncoderOutput(Protocol):
	embeddings: torch.Tensor
	loop_slot_hidden_states: tuple[torch.Tensor, ...]
	slot_hidden_states: torch.Tensor
	diagnostics: dict[str, Any]


@dataclass(frozen=True)
class GroupedEncoderOutput:
	"""Encoder tensors restored to query-then-candidate order."""

	embeddings: torch.Tensor
	loop_slot_hidden_states: tuple[torch.Tensor, ...]
	slot_hidden_states: torch.Tensor
	diagnostics: dict[str, Any]


def _weighted_diagnostic(
	outputs: tuple[_EncoderOutput, ...],
	counts: tuple[int, ...],
	key: str,
) -> torch.Tensor:
	"""Average a scalar diagnostic across differently sized encoder groups."""
	return sum(
		(output.diagnostics[key] * count for output, count in zip(outputs, counts, strict=True)),
		start=outputs[0].diagnostics[key].new_zeros(()),
	) / sum(counts)


def _encode_grouped_batches(
	*,
	encoder: nn.Module,
	processed_batches: tuple[dict[str, torch.Tensor], ...],
	original_indices: tuple[tuple[int, ...], ...],
	total_rows: int,
) -> GroupedEncoderOutput:
	"""Encode homogeneous padding groups and restore the original logical row order."""
	if not processed_batches or len(processed_batches) != len(original_indices):
		raise ValueError("Processed batches and original indices must be non-empty and aligned")
	flat_indices = tuple(index for group in original_indices for index in group)
	if sorted(flat_indices) != list(range(total_rows)):
		raise ValueError("Grouped encoder indices must cover every logical row exactly once")
	outputs = tuple(encoder(**batch) for batch in processed_batches)
	counts = tuple(len(indices) for indices in original_indices)
	for output, count in zip(outputs, counts, strict=True):
		if output.embeddings.shape[0] != count or output.slot_hidden_states.shape[0] != count:
			raise ValueError("Encoder group output size does not match its original indices")
	device = outputs[0].embeddings.device
	restore_order = torch.argsort(torch.tensor(flat_indices, device=device))
	embeddings = torch.cat(tuple(output.embeddings for output in outputs), dim=0).index_select(
		0,
		restore_order,
	)
	slot_hidden_states = torch.cat(
		tuple(output.slot_hidden_states for output in outputs),
		dim=0,
	).index_select(0, restore_order)
	loop_pass_count = len(outputs[0].loop_slot_hidden_states)
	if loop_pass_count == 0 or any(
		len(output.loop_slot_hidden_states) != loop_pass_count
		for output in outputs
	):
		raise ValueError("Grouped encoder outputs must contain the same recurrent rounds")
	loop_slot_hidden_states = tuple(
		torch.cat(
			tuple(output.loop_slot_hidden_states[pass_index] for output in outputs),
			dim=0,
		).index_select(0, restore_order)
		for pass_index in range(loop_pass_count)
	)
	scalar_keys = (
		"fusion_gate",
		"late_fusion_attention_entropy",
		"slot_pairwise_cosine",
	)
	diagnostics: dict[str, Any] = {
		key: _weighted_diagnostic(outputs, counts, key)
		for key in scalar_keys
	}
	for key in ("recurrent_pass_cosine", "recurrent_pass_relative_update"):
		value_count = len(outputs[0].diagnostics[key])
		if any(len(output.diagnostics[key]) != value_count for output in outputs):
			raise ValueError(f"Grouped diagnostic length mismatch for {key}")
		diagnostics[key] = tuple(
			sum(
				(
					output.diagnostics[key][value_index] * count
					for output, count in zip(outputs, counts, strict=True)
				),
				start=outputs[0].diagnostics[key][value_index].new_zeros(()),
			)
			/ total_rows
			for value_index in range(value_count)
		)
	return GroupedEncoderOutput(
		embeddings=embeddings,
		loop_slot_hidden_states=loop_slot_hidden_states,
		slot_hidden_states=slot_hidden_states,
		diagnostics=diagnostics,
	)


class RecurrentTrainingModel(nn.Module):
	"""Keep every trainable head inside the distributed forward graph."""

	def __init__(self, encoder: RecurrentQwen3VLEmbedding) -> None:
		super().__init__()
		self.encoder = encoder

	def forward(
		self,
		*,
		local_batch_size: int,
		positive_ids: list[str],
		processed_batches: tuple[dict[str, torch.Tensor], ...],
		original_indices: tuple[tuple[int, ...], ...],
	) -> dict[str, Any]:
		"""Encode padding-homogeneous groups and calculate every locked loss component."""
		output = _encode_grouped_batches(
			encoder=self.encoder,
			processed_batches=processed_batches,
			original_indices=original_indices,
			total_rows=2 * local_batch_size,
		)
		if output.embeddings.shape[0] != 2 * local_batch_size:
			raise ValueError("Combined encoder batch must contain query then candidate rows")
		query_embeddings, candidate_embeddings = output.embeddings.split(local_batch_size)
		embedding_pairs = {
			"final": (query_embeddings, candidate_embeddings),
		}
		for pass_index, pass_slot_states in enumerate(
			output.loop_slot_hidden_states,
			start=1,
		):
			query_slots, candidate_slots = pass_slot_states.split(local_batch_size)
			embedding_pairs[f"loop_pass_{pass_index}"] = (
				self.encoder.auxiliary_embedding_head(query_slots),
				self.encoder.auxiliary_embedding_head(candidate_slots),
			)
		contrastive_losses = distributed_multi_positive_info_nce_losses(
			embedding_pairs=embedding_pairs,
			positive_ids=positive_ids,
			temperature=self.encoder.config.temperature,
		)
		final_infonce = contrastive_losses["final"]
		loop_infonce_by_pass = tuple(
			contrastive_losses[f"loop_pass_{pass_index}"]
			for pass_index in range(1, len(output.loop_slot_hidden_states) + 1)
		)
		loop_infonce = torch.stack(loop_infonce_by_pass).mean()
		final_recurrent_slots = output.loop_slot_hidden_states[-1]
		query_slots, candidate_slots = final_recurrent_slots.split(local_batch_size)
		diversity = 0.5 * (
			slot_diversity_loss(query_slots) + slot_diversity_loss(candidate_slots)
		)
		total_loss = compose_training_loss(
			final_infonce=final_infonce,
			loop_infonce=loop_infonce,
			slot_diversity=diversity,
		)
		return {
			"total_loss": total_loss,
			"final_infonce": final_infonce,
			"loop_infonce": loop_infonce,
			"loop_infonce_by_pass": loop_infonce_by_pass,
			"slot_diversity": diversity,
			"fusion_gate": output.diagnostics["fusion_gate"],
			"late_fusion_attention_entropy": output.diagnostics[
				"late_fusion_attention_entropy"
			],
			"slot_pairwise_cosine": output.diagnostics["slot_pairwise_cosine"],
			"recurrent_pass_cosine": output.diagnostics["recurrent_pass_cosine"],
			"recurrent_pass_relative_update": output.diagnostics[
				"recurrent_pass_relative_update"
			],
		}
