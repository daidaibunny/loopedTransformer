"""One-collective multi-step losses for query-only recurrent retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from looped_vl.query_recurrent.config import QueryRecurrentConfig
from looped_vl.query_recurrent.model import QueryRecurrentOutput


def _gather_ids(local_ids: tuple[str, ...]) -> tuple[str, ...]:
	if not (dist.is_available() and dist.is_initialized()):
		return local_ids
	gathered: list[tuple[str, ...] | None] = [None for _ in range(dist.get_world_size())]
	dist.all_gather_object(gathered, local_ids)
	if any(ids is None for ids in gathered):
		raise RuntimeError("Failed to gather positive identifiers")
	return tuple(identifier for ids in gathered if ids is not None for identifier in ids)


def _gather_candidates(local_candidates: torch.Tensor) -> torch.Tensor:
	if not (dist.is_available() and dist.is_initialized()):
		return local_candidates
	return torch.cat(all_gather(local_candidates), dim=0)


def _gather_query_stack(local_queries: torch.Tensor) -> torch.Tensor:
	if not (dist.is_available() and dist.is_initialized()):
		return local_queries
	return torch.cat(all_gather(local_queries), dim=1)


def _directional_loss(
	local_embeddings: torch.Tensor,
	global_targets: torch.Tensor,
	positive_mask: torch.Tensor,
	valid_target_mask: torch.Tensor,
	*,
	temperature: float,
	hard_negative_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
	logits = local_embeddings.float() @ global_targets.float().T / temperature
	logits = logits.masked_fill(~valid_target_mask, float("-inf"))
	positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
	denominator_logits = logits
	if hard_negative_embeddings is not None:
		if (
			hard_negative_embeddings.ndim != 3
			or hard_negative_embeddings.shape[0] != local_embeddings.shape[0]
			or hard_negative_embeddings.shape[2] != local_embeddings.shape[1]
		):
			raise ValueError("Hard-negative embeddings must have shape [batch, negatives, hidden]")
		hard_logits = torch.einsum(
			"bd,bnd->bn",
			local_embeddings.float(),
			hard_negative_embeddings.float(),
		) / temperature
		denominator_logits = torch.cat((logits, hard_logits), dim=1)
	return (
		torch.logsumexp(denominator_logits, dim=1)
		- torch.logsumexp(positive_logits, dim=1)
	).mean()


def multi_query_symmetric_info_nce(
	query_embeddings: Sequence[torch.Tensor],
	candidate_embeddings: torch.Tensor,
	positive_ids: Sequence[str],
	directions: Sequence[str],
	*,
	temperature: float,
	hard_negative_embeddings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
	"""Gather candidates and all recurrent query outputs exactly once per microbatch."""
	if not query_embeddings:
		raise ValueError("At least one query embedding tensor is required")
	if candidate_embeddings.ndim != 2 or candidate_embeddings.shape[0] == 0:
		raise ValueError("Candidate embeddings must have a non-empty rank-two shape")
	batch_size, hidden_size = candidate_embeddings.shape
	if len(positive_ids) != batch_size:
		raise ValueError("positive_ids must contain one identifier per local candidate")
	if len(directions) != batch_size:
		raise ValueError("directions must contain one gallery identity per local candidate")
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	if any(tensor.shape != (batch_size, hidden_size) for tensor in query_embeddings):
		raise ValueError("Every query tensor must match candidate shape")
	local_ids = tuple(str(identifier) for identifier in positive_ids)
	global_ids = _gather_ids(local_ids)
	local_directions = tuple(str(direction) for direction in directions)
	global_directions = _gather_ids(local_directions)
	global_candidates = _gather_candidates(candidate_embeddings)
	local_query_stack = torch.stack(tuple(query_embeddings), dim=0)
	global_query_stack = _gather_query_stack(local_query_stack)
	valid_target_mask = torch.tensor(
		[
			[local_direction == global_direction for global_direction in global_directions]
			for local_direction in local_directions
		],
		device=candidate_embeddings.device,
		dtype=torch.bool,
	)
	positive_mask = torch.tensor(
		[
			[
				local_id == global_id and local_direction == global_direction
				for global_id, global_direction in zip(
					global_ids,
					global_directions,
					strict=True,
				)
			]
			for local_id, local_direction in zip(local_ids, local_directions, strict=True)
		],
		device=candidate_embeddings.device,
		dtype=torch.bool,
	)
	if not positive_mask.any(dim=1).all():
		raise RuntimeError("Every local query must have at least one global positive")
	losses = []
	for query_index, local_queries in enumerate(query_embeddings):
		query_loss = _directional_loss(
			local_queries,
			global_candidates,
			positive_mask,
			valid_target_mask,
			temperature=temperature,
			hard_negative_embeddings=hard_negative_embeddings,
		)
		candidate_loss = _directional_loss(
			candidate_embeddings,
			global_query_stack[query_index],
			positive_mask,
			valid_target_mask,
			temperature=temperature,
		)
		losses.append(0.5 * (query_loss + candidate_loss))
	return tuple(losses)


def slot_pairwise_absolute_cosine(slot_states: torch.Tensor) -> torch.Tensor:
	"""Log slot collapse without forcing potentially meaningless diversity."""
	if slot_states.shape[1] <= 1:
		return slot_states.new_zeros(())
	normalized = torch.nn.functional.normalize(slot_states.float(), dim=-1)
	cosine = normalized @ normalized.transpose(1, 2)
	off_diagonal = ~torch.eye(
		slot_states.shape[1],
		device=slot_states.device,
		dtype=torch.bool,
	)
	return cosine[:, off_diagonal].abs().mean()


def query_recurrent_loss(
	output: QueryRecurrentOutput,
	candidate_embeddings: torch.Tensor,
	positive_ids: Sequence[str],
	directions: Sequence[str],
	config: QueryRecurrentConfig,
	*,
	hard_negative_embeddings: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
	"""Train every fused pass directly within its own candidate gallery."""
	losses = multi_query_symmetric_info_nce(
		(*output.step_embeddings, *output.slot_proposal_embeddings),
		candidate_embeddings,
		positive_ids,
		directions,
		temperature=config.temperature,
		hard_negative_embeddings=hard_negative_embeddings,
	)
	step_count = len(output.step_embeddings)
	step_losses = losses[:step_count]
	proposal_losses = losses[step_count:]
	if len(proposal_losses) != step_count:
		raise RuntimeError("Every recurrent pass must expose one slot proposal embedding")
	pass_weights = torch.arange(
		1,
		step_count + 1,
		device=step_losses[0].device,
		dtype=step_losses[0].dtype,
	)
	pass_weights = pass_weights / pass_weights.sum()
	direct_pass_loss = (torch.stack(step_losses) * pass_weights).sum()
	slot_proposal_loss = (torch.stack(proposal_losses) * pass_weights).sum()
	main_loss = step_losses[-1]
	if len(step_losses) > 1:
		progressive_loss = torch.stack(
			[
				torch.relu(current - previous + config.progressive_margin)
				for previous, current in zip(step_losses[:-1], step_losses[1:], strict=True)
			],
		).mean()
	else:
		progressive_loss = main_loss.new_zeros(())
	total = (
		config.direct_pass_loss_weight * direct_pass_loss
		+ config.slot_proposal_loss_weight * slot_proposal_loss
		+ config.progressive_loss_weight * progressive_loss
	)
	return {
		"loss": total,
		"main_info_nce": main_loss,
		"direct_pass_info_nce": direct_pass_loss,
		"slot_proposal_info_nce": slot_proposal_loss,
		"progressive_margin_loss": progressive_loss,
		"slot_pairwise_absolute_cosine": slot_pairwise_absolute_cosine(
			output.slot_states[-1],
		),
		**{
			f"step_{index}_info_nce": loss
			for index, loss in enumerate(step_losses, start=1)
		},
		**{
			f"step_{index}_slot_proposal_info_nce": loss
			for index, loss in enumerate(proposal_losses, start=1)
		},
	}
