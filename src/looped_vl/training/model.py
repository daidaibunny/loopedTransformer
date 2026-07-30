"""One DDP forward containing the encoder, auxiliary heads, and all losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from looped_vl.models.recurrent_qwen3vl_embedding import RecurrentQwen3VLEmbedding
from looped_vl.training.losses import slot_diversity_loss
from looped_vl.training.step import (
	compose_stage_loss,
	distributed_multi_positive_info_nce_losses,
)


class _EncoderOutput(Protocol):
	embeddings: torch.Tensor
	slot_hidden_states: torch.Tensor
	diagnostics: dict[str, Any]


@dataclass(frozen=True)
class GroupedEncoderOutput:
	"""Encoder tensors restored to query-then-candidate order."""

	embeddings: torch.Tensor
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
	scalar_keys = (
		"fusion_gate",
		"late_fusion_attention_entropy",
		"slot_pairwise_cosine",
		"connector_output_norm",
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
		semantic_targets: list[str],
		positive_ids: list[str],
		sources: list[str],
		stage: int,
		processed_batches: tuple[dict[str, torch.Tensor], ...],
		original_indices: tuple[tuple[int, ...], ...],
	) -> dict[str, Any]:
		"""Encode padding-homogeneous groups and calculate every v1.0 loss component."""
		output = _encode_grouped_batches(
			encoder=self.encoder,
			processed_batches=processed_batches,
			original_indices=original_indices,
			total_rows=2 * local_batch_size,
		)
		if output.embeddings.shape[0] != 2 * local_batch_size:
			raise ValueError("Combined encoder batch must contain query then candidate rows")
		query_embeddings, candidate_embeddings = output.embeddings.split(local_batch_size)
		query_slots, candidate_slots = output.slot_hidden_states.split(local_batch_size)
		query_slot_embeddings = self.encoder.warmup_embedding_head(query_slots)
		candidate_slot_embeddings = self.encoder.warmup_embedding_head(candidate_slots)
		embedding_pairs = {
			"slot": (query_slot_embeddings, candidate_slot_embeddings),
		}
		if stage == 2:
			embedding_pairs = {
				"final": (query_embeddings, candidate_embeddings),
				**embedding_pairs,
			}
		contrastive_losses = distributed_multi_positive_info_nce_losses(
			embedding_pairs=embedding_pairs,
			positive_ids=positive_ids,
			temperature=self.encoder.config.temperature,
		)
		final_infonce = contrastive_losses.get("final", query_embeddings.new_zeros(()))
		slot_infonce = contrastive_losses["slot"]
		semantic_output = self.encoder.warmup_semantic_head(
			query_slots,
			semantic_targets,
			sources,
		)
		diversity = 0.5 * (
			slot_diversity_loss(query_slots) + slot_diversity_loss(candidate_slots)
		)
		total_loss = compose_stage_loss(
			stage=stage,
			final_infonce=final_infonce,
			slot_infonce=slot_infonce,
			semantic_decoder_ce=semantic_output.loss,
			slot_diversity=diversity,
		)
		return {
			"total_loss": total_loss,
			"final_infonce": final_infonce,
			"slot_infonce": slot_infonce,
			"semantic_decoder_ce": semantic_output.loss,
			"semantic_token_count": semantic_output.token_count,
			"slot_diversity": diversity,
			"fusion_gate": output.diagnostics["fusion_gate"],
			"late_fusion_attention_entropy": output.diagnostics[
				"late_fusion_attention_entropy"
			],
			"slot_pairwise_cosine": output.diagnostics["slot_pairwise_cosine"],
			"connector_output_norm": output.diagnostics["connector_output_norm"],
			"recurrent_pass_cosine": output.diagnostics["recurrent_pass_cosine"],
			"recurrent_pass_relative_update": output.diagnostics[
				"recurrent_pass_relative_update"
			],
		}
