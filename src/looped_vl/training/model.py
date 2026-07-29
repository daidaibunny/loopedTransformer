"""One DDP forward containing the encoder, auxiliary heads, and all losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from looped_vl.models.recurrent_qwen3vl_embedding import RecurrentQwen3VLEmbedding
from looped_vl.training.losses import slot_diversity_loss
from looped_vl.training.step import compose_stage_loss, distributed_symmetric_info_nce


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
		sources: list[str],
		stage: int,
		**processed_inputs: torch.Tensor,
	) -> dict[str, Any]:
		"""Encode both towers together and calculate every v1.0 loss component."""
		output = self.encoder(**processed_inputs)
		if output.embeddings.shape[0] != 2 * local_batch_size:
			raise ValueError("Combined encoder batch must contain query then candidate rows")
		query_embeddings, candidate_embeddings = output.embeddings.split(local_batch_size)
		query_slots, candidate_slots = output.slot_hidden_states.split(local_batch_size)
		query_slot_embeddings = self.encoder.warmup_embedding_head(query_slots)
		candidate_slot_embeddings = self.encoder.warmup_embedding_head(candidate_slots)
		if stage == 1:
			with torch.no_grad():
				final_infonce = distributed_symmetric_info_nce(
					query_embeddings.detach(),
					candidate_embeddings.detach(),
					self.encoder.config.temperature,
				)
		else:
			final_infonce = distributed_symmetric_info_nce(
				query_embeddings,
				candidate_embeddings,
				self.encoder.config.temperature,
			)
		slot_infonce = distributed_symmetric_info_nce(
			query_slot_embeddings,
			candidate_slot_embeddings,
			self.encoder.config.temperature,
		)
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
