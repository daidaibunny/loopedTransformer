"""Losses fixed by recurrent embedding specification v1.0."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_info_nce(
	query_embeddings: torch.Tensor,
	candidate_embeddings: torch.Tensor,
	temperature: float = 0.02,
) -> torch.Tensor:
	"""Compute symmetric diagonal InfoNCE for a paired retrieval batch."""
	if query_embeddings.shape != candidate_embeddings.shape:
		raise ValueError("Query and candidate embedding batches must have equal shapes")
	if query_embeddings.ndim != 2 or query_embeddings.shape[0] == 0:
		raise ValueError("Embeddings must have non-empty rank-two shapes")
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	similarity = query_embeddings @ candidate_embeddings.T / temperature
	labels = torch.arange(similarity.shape[0], device=similarity.device)
	return 0.5 * (
		F.cross_entropy(similarity, labels)
		+ F.cross_entropy(similarity.T, labels)
	)


def slot_diversity_loss(slot_hidden_states: torch.Tensor) -> torch.Tensor:
	"""Average off-diagonal cosine similarity of final contextual slots."""
	if slot_hidden_states.ndim != 3 or slot_hidden_states.shape[1] == 0:
		raise ValueError("slot_hidden_states must have shape [batch, slots, hidden]")
	slot_count = slot_hidden_states.shape[1]
	if slot_count == 1:
		return slot_hidden_states.sum() * 0.0
	normalized = F.normalize(slot_hidden_states.float(), p=2, dim=-1)
	cosine = normalized @ normalized.transpose(1, 2)
	off_diagonal = ~torch.eye(
		slot_count,
		dtype=torch.bool,
		device=slot_hidden_states.device,
	)
	return cosine[:, off_diagonal].mean()
