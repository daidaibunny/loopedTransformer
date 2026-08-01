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
	*,
	temperature: float,
) -> torch.Tensor:
	logits = local_embeddings.float() @ global_targets.float().T / temperature
	positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
	return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def multi_query_symmetric_info_nce(
	query_embeddings: Sequence[torch.Tensor],
	candidate_embeddings: torch.Tensor,
	positive_ids: Sequence[str],
	*,
	temperature: float,
) -> tuple[torch.Tensor, ...]:
	"""Gather candidates and all recurrent query outputs exactly once per microbatch."""
	if not query_embeddings:
		raise ValueError("At least one query embedding tensor is required")
	if candidate_embeddings.ndim != 2 or candidate_embeddings.shape[0] == 0:
		raise ValueError("Candidate embeddings must have a non-empty rank-two shape")
	batch_size, hidden_size = candidate_embeddings.shape
	if len(positive_ids) != batch_size:
		raise ValueError("positive_ids must contain one identifier per local candidate")
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	if any(tensor.shape != (batch_size, hidden_size) for tensor in query_embeddings):
		raise ValueError("Every query tensor must match candidate shape")
	local_ids = tuple(str(identifier) for identifier in positive_ids)
	global_ids = _gather_ids(local_ids)
	global_candidates = _gather_candidates(candidate_embeddings)
	local_query_stack = torch.stack(tuple(query_embeddings), dim=0)
	global_query_stack = _gather_query_stack(local_query_stack)
	positive_mask = torch.tensor(
		[[local_id == global_id for global_id in global_ids] for local_id in local_ids],
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
			temperature=temperature,
		)
		candidate_loss = _directional_loss(
			candidate_embeddings,
			global_query_stack[query_index],
			positive_mask,
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
	config: QueryRecurrentConfig,
) -> dict[str, torch.Tensor]:
	"""Train retrieval, recurrent improvement, slot signal, and differentiable exit jointly."""
	main_embeddings = (
		output.soft_embeddings if config.exit_mode == "dynamic" else output.embeddings
	)
	queries = (
		main_embeddings,
		*output.step_embeddings,
		*output.auxiliary_embeddings,
	)
	losses = multi_query_symmetric_info_nce(
		queries,
		candidate_embeddings,
		positive_ids,
		temperature=config.temperature,
	)
	step_count = len(output.step_embeddings)
	main_loss = losses[0]
	step_losses = losses[1 : 1 + step_count]
	auxiliary_losses = losses[1 + step_count :]
	auxiliary_loss = torch.stack(auxiliary_losses).mean()
	if len(step_losses) > 1:
		progressive_loss = torch.stack(
			[
				torch.relu(current - previous)
				for previous, current in zip(step_losses[:-1], step_losses[1:], strict=True)
			],
		).mean()
	else:
		progressive_loss = main_loss.new_zeros(())
	compute_penalty = output.expected_steps.float().mean() / config.max_recurrent_steps
	if config.exit_mode != "dynamic":
		compute_penalty = compute_penalty * 0.0
	total = (
		main_loss
		+ config.auxiliary_loss_weight * auxiliary_loss
		+ config.progressive_loss_weight * progressive_loss
		+ config.compute_penalty_weight * compute_penalty
		+ 0.0 * output.exit_probabilities.sum()
	)
	return {
		"loss": total,
		"main_info_nce": main_loss,
		"auxiliary_info_nce": auxiliary_loss,
		"progressive_non_degradation": progressive_loss,
		"compute_penalty": compute_penalty,
		"expected_steps": output.expected_steps.float().mean(),
		"slot_pairwise_absolute_cosine": slot_pairwise_absolute_cosine(
			output.slot_states[-1],
		),
		**{
			f"step_{index}_info_nce": loss
			for index, loss in enumerate(step_losses, start=1)
		},
	}
