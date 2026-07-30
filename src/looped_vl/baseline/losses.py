"""Distributed multi-positive contrastive loss for retrieval baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def _gather_positive_ids(local_ids: tuple[str, ...]) -> tuple[str, ...]:
	if not (dist.is_available() and dist.is_initialized()):
		return local_ids
	gathered: list[tuple[str, ...] | None] = [None for _ in range(dist.get_world_size())]
	dist.all_gather_object(gathered, local_ids)
	if any(ids is None for ids in gathered):
		raise RuntimeError("Failed to gather positive identifiers from every rank")
	return tuple(identifier for ids in gathered if ids is not None for identifier in ids)


def _gather_embeddings(local_embeddings: torch.Tensor) -> torch.Tensor:
	if not (dist.is_available() and dist.is_initialized()):
		return local_embeddings
	gathered = all_gather(local_embeddings)
	if any(item.shape != local_embeddings.shape for item in gathered):
		raise RuntimeError("Every rank must use the same local contrastive batch size")
	return torch.cat(gathered, dim=0)


def _multi_positive_directional_loss(
	local_embeddings: torch.Tensor,
	global_targets: torch.Tensor,
	local_ids: tuple[str, ...],
	global_ids: tuple[str, ...],
	temperature: float,
) -> torch.Tensor:
	logits = local_embeddings.float() @ global_targets.float().T / temperature
	positive_mask = torch.tensor(
		[
			[local_id == global_id for global_id in global_ids]
			for local_id in local_ids
		],
		dtype=torch.bool,
		device=logits.device,
	)
	if not positive_mask.any(dim=1).all():
		raise RuntimeError("Every local sample must have at least one global positive")
	positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
	return (
		torch.logsumexp(logits, dim=1)
		- torch.logsumexp(positive_logits, dim=1)
	).mean()


def multi_positive_symmetric_info_nce(
	query_embeddings: torch.Tensor,
	candidate_embeddings: torch.Tensor,
	positive_ids: Sequence[str],
	temperature: float = 0.02,
) -> torch.Tensor:
	"""Treat every globally gathered row with the same semantic ID as a positive."""
	if query_embeddings.shape != candidate_embeddings.shape:
		raise ValueError("Query and candidate embedding shapes must match")
	if query_embeddings.ndim != 2 or query_embeddings.shape[0] == 0:
		raise ValueError("Embeddings must have a non-empty rank-two shape")
	if len(positive_ids) != query_embeddings.shape[0]:
		raise ValueError("positive_ids must contain one identifier per local row")
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	local_ids = tuple(str(identifier) for identifier in positive_ids)
	if any(not identifier for identifier in local_ids):
		raise ValueError("positive_ids cannot contain empty identifiers")
	global_ids = _gather_positive_ids(local_ids)
	global_queries = _gather_embeddings(query_embeddings)
	global_candidates = _gather_embeddings(candidate_embeddings)
	query_loss = _multi_positive_directional_loss(
		query_embeddings,
		global_candidates,
		local_ids,
		global_ids,
		temperature,
	)
	candidate_loss = _multi_positive_directional_loss(
		candidate_embeddings,
		global_queries,
		local_ids,
		global_ids,
		temperature,
	)
	return 0.5 * (query_loss + candidate_loss)
