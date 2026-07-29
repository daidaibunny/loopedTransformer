"""Distributed contrastive losses and exact stage loss composition."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def distributed_symmetric_info_nce(
	query_embeddings: torch.Tensor,
	candidate_embeddings: torch.Tensor,
	temperature: float,
) -> torch.Tensor:
	"""Contrast each rank's paired rows against differentiably gathered global negatives."""
	if query_embeddings.shape != candidate_embeddings.shape:
		raise ValueError("Query and candidate embeddings must have equal shapes")
	if dist.is_available() and dist.is_initialized():
		world_size = dist.get_world_size()
		rank = dist.get_rank()
		gathered_queries = torch.cat(all_gather(query_embeddings), dim=0)
		gathered_candidates = torch.cat(all_gather(candidate_embeddings), dim=0)
	else:
		world_size = 1
		rank = 0
		gathered_queries = query_embeddings
		gathered_candidates = candidate_embeddings
	local_batch_size = query_embeddings.shape[0]
	if gathered_queries.shape[0] != local_batch_size * world_size:
		raise RuntimeError("Distributed ranks produced unequal contrastive batch sizes")
	labels = torch.arange(local_batch_size, device=query_embeddings.device)
	labels = labels + rank * local_batch_size
	query_logits = query_embeddings @ gathered_candidates.T / temperature
	candidate_logits = candidate_embeddings @ gathered_queries.T / temperature
	return 0.5 * (
		F.cross_entropy(query_logits, labels)
		+ F.cross_entropy(candidate_logits, labels)
	)


def compose_stage_loss(
	*,
	stage: int,
	final_infonce: torch.Tensor,
	slot_infonce: torch.Tensor,
	semantic_decoder_ce: torch.Tensor,
	slot_diversity: torch.Tensor,
) -> torch.Tensor:
	"""Apply the fixed v1.0 Stage 1 or Stage 2 scalar loss weights."""
	if stage == 1:
		return slot_infonce + semantic_decoder_ce + 0.05 * slot_diversity
	if stage == 2:
		return (
			final_infonce
			+ 0.2 * slot_infonce
			+ 0.2 * semantic_decoder_ce
			+ 0.05 * slot_diversity
		)
	raise ValueError("stage must be 1 or 2")
