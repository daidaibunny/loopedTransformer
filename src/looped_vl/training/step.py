"""Distributed contrastive losses and exact stage loss composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def _gather_positive_ids(local_ids: tuple[str, ...]) -> tuple[str, ...]:
	if not (dist.is_available() and dist.is_initialized()):
		return local_ids
	gathered: list[tuple[str, ...] | None] = [None for _ in range(dist.get_world_size())]
	dist.all_gather_object(gathered, local_ids)
	if any(identifiers is None for identifiers in gathered):
		raise RuntimeError("Failed to gather positive identifiers from every rank")
	return tuple(
		identifier
		for identifiers in gathered
		if identifiers is not None
		for identifier in identifiers
	)


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


def distributed_multi_positive_info_nce_losses(
	*,
	embedding_pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
	positive_ids: Sequence[str],
	temperature: float,
) -> dict[str, torch.Tensor]:
	"""Calculate multi-positive losses with one differentiable gather per width."""
	if not embedding_pairs:
		raise ValueError("At least one embedding pair is required")
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	names = tuple(embedding_pairs)
	local_tensors = tuple(
		embedding
		for name in names
		for embedding in embedding_pairs[name]
	)
	reference_batch_size = local_tensors[0].shape[0]
	if local_tensors[0].ndim != 2 or reference_batch_size == 0:
		raise ValueError("Embeddings must have a non-empty rank-two shape")
	for name in names:
		query_embeddings, candidate_embeddings = embedding_pairs[name]
		if (
			query_embeddings.ndim != 2
			or candidate_embeddings.ndim != 2
			or query_embeddings.shape != candidate_embeddings.shape
			or query_embeddings.shape[0] != reference_batch_size
		):
			raise ValueError(
				"Every embedding pair must have equal query/candidate shapes and "
				"the shared local batch size",
			)
	if len(positive_ids) != reference_batch_size:
		raise ValueError("positive_ids must contain one identifier per local row")
	local_ids = tuple(str(identifier) for identifier in positive_ids)
	if any(not identifier for identifier in local_ids):
		raise ValueError("positive_ids cannot contain empty identifiers")
	global_ids = _gather_positive_ids(local_ids)
	tensor_groups: dict[
		tuple[int, torch.dtype, torch.device],
		list[tuple[int, torch.Tensor]],
	] = {}
	for tensor_index, tensor in enumerate(local_tensors):
		key = (tensor.shape[1], tensor.dtype, tensor.device)
		tensor_groups.setdefault(key, []).append((tensor_index, tensor))
	global_tensors: dict[int, torch.Tensor] = {}
	for group in tensor_groups.values():
		packed_local = torch.stack([tensor for _, tensor in group], dim=0)
		if dist.is_available() and dist.is_initialized():
			gathered = all_gather(packed_local)
			if any(item.shape != packed_local.shape for item in gathered):
				raise RuntimeError("Every rank must use equal local contrastive batch sizes")
			packed_global = torch.cat(gathered, dim=1)
		else:
			packed_global = packed_local
		for group_index, (tensor_index, _) in enumerate(group):
			global_tensors[tensor_index] = packed_global[group_index]
	losses: dict[str, torch.Tensor] = {}
	for index, name in enumerate(names):
		local_query = local_tensors[2 * index]
		local_candidate = local_tensors[2 * index + 1]
		global_query = global_tensors[2 * index]
		global_candidate = global_tensors[2 * index + 1]
		losses[name] = 0.5 * (
			_multi_positive_directional_loss(
				local_query,
				global_candidate,
				local_ids,
				global_ids,
				temperature,
			)
			+ _multi_positive_directional_loss(
				local_candidate,
				global_query,
				local_ids,
				global_ids,
				temperature,
			)
		)
	return losses


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


def compose_training_loss(
	*,
	final_infonce: torch.Tensor,
	loop_infonce: torch.Tensor,
	progressive_loss: torch.Tensor,
	slot_diversity: torch.Tensor,
) -> torch.Tensor:
	"""Train final retrieval, final slots, and non-degrading later recurrent passes."""
	return final_infonce + 0.1 * loop_infonce + 0.1 * progressive_loss + 0.0 * slot_diversity


def progressive_non_degradation_loss(
	loop_losses: Sequence[torch.Tensor],
) -> torch.Tensor:
	"""Penalize a later recurrent pass only when its retrieval loss becomes worse."""
	if len(loop_losses) < 2:
		return loop_losses[0].new_zeros(()) if loop_losses else torch.tensor(0.0)
	penalties = tuple(
		F.relu(current - previous.detach())
		for previous, current in zip(loop_losses[:-1], loop_losses[1:], strict=True)
	)
	return torch.stack(penalties).mean()
