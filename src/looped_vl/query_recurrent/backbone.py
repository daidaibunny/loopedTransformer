"""One frozen Qwen query pass that returns only the official final embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from looped_vl.baseline.model import pool_last_token


@dataclass(frozen=True)
class FrozenQueryFeatures:
	"""Detached official final-valid-token retrieval embeddings."""

	base_embeddings: torch.Tensor


class FrozenQueryBackbone(nn.Module):
	"""Run the immutable Qwen backbone exactly once without retaining histories."""

	def __init__(self, base_embedding_model: nn.Module) -> None:
		super().__init__()
		self.base_embedding_model = base_embedding_model
		self.base_embedding_model.eval()
		self.base_embedding_model.requires_grad_(False)

	def forward(self, processed_inputs: dict[str, torch.Tensor]) -> FrozenQueryFeatures:
		"""Return one detached official embedding without building a Qwen graph."""
		attention_mask = processed_inputs["attention_mask"]
		with torch.no_grad():
			outputs = self.base_embedding_model.model(
				**processed_inputs,
				output_hidden_states=False,
				return_dict=True,
				use_cache=False,
			)
		base_embeddings = pool_last_token(outputs.last_hidden_state, attention_mask)
		return FrozenQueryFeatures(base_embeddings=base_embeddings.detach())


def combine_frozen_query_groups(
	groups: list[tuple[tuple[int, ...], FrozenQueryFeatures]],
	*,
	total_rows: int,
) -> FrozenQueryFeatures:
	"""Restore grouped Qwen outputs to their logical contrastive order."""
	if not groups or total_rows <= 0:
		raise ValueError("At least one non-empty frozen query group is required")
	flat_indices = tuple(index for indices, _features in groups for index in indices)
	if tuple(sorted(flat_indices)) != tuple(range(total_rows)):
		raise ValueError("Frozen query groups must cover every logical row exactly once")
	first = groups[0][1]
	hidden_size = first.base_embeddings.shape[-1]
	base_embeddings = first.base_embeddings.new_empty((total_rows, hidden_size))
	for indices, features in groups:
		if len(indices) != features.base_embeddings.shape[0]:
			raise ValueError("Frozen query group indexes and feature rows must match")
		positions = torch.tensor(indices, device=base_embeddings.device)
		base_embeddings[positions] = features.base_embeddings
	return FrozenQueryFeatures(base_embeddings=base_embeddings)
