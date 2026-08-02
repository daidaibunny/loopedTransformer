from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from looped_vl.query_recurrent.backbone import (
	FrozenQueryBackbone,
	FrozenQueryFeatures,
	combine_frozen_query_groups,
)
from looped_vl.query_recurrent.config import QueryRecurrentConfig
from looped_vl.query_recurrent.losses import (
	multi_query_symmetric_info_nce,
	query_recurrent_loss,
)
from looped_vl.query_recurrent.model import QueryRecurrentHead


class _FakeInnerQwen(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.tensor(1.0))
		self.call_count = 0
		self.requested_hidden_states: bool | None = None

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		*,
		output_hidden_states: bool,
		**_kwargs: object,
	) -> SimpleNamespace:
		del attention_mask
		self.call_count += 1
		self.requested_hidden_states = output_hidden_states
		hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048)
		return SimpleNamespace(last_hidden_state=hidden)


class _FakeQwenWrapper(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.model = _FakeInnerQwen()


def test_frozen_query_backbone_runs_once_and_returns_only_final_embedding() -> None:
	model = _FakeQwenWrapper()
	backbone = FrozenQueryBackbone(model)
	inputs = {
		"input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
		"attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
	}

	features = backbone(inputs)

	assert model.model.call_count == 1
	assert model.model.requested_hidden_states is False
	assert features.base_embeddings.shape == (2, 2048)
	assert torch.allclose(features.base_embeddings.norm(dim=1), torch.ones(2))
	assert features.base_embeddings.requires_grad is False
	assert all(parameter.requires_grad is False for parameter in backbone.parameters())
	assert not hasattr(features, "history_hidden_states")


def test_frozen_feature_groups_restore_order_without_history_padding() -> None:
	first = FrozenQueryFeatures(base_embeddings=torch.full((1, 4), 2.0))
	second = FrozenQueryFeatures(
		base_embeddings=torch.tensor([[5.0] * 4, [7.0] * 4]),
	)

	combined = combine_frozen_query_groups(
		[((1,), first), ((0, 2), second)],
		total_rows=3,
	)

	assert combined.base_embeddings[:, 0].tolist() == [5.0, 2.0, 7.0]


def test_final_mean_loss_backpropagates_without_candidate_gradients() -> None:
	config = QueryRecurrentConfig(max_recurrent_steps=2)
	head = QueryRecurrentHead(config)
	base = torch.nn.functional.normalize(torch.randn(4, 2048), dim=-1)
	candidates = torch.nn.functional.normalize(torch.randn(4, 2048), dim=-1)
	output = head(base_embeddings=base)
	for embedding in output.step_embeddings:
		embedding.retain_grad()

	losses = query_recurrent_loss(
		output,
		candidates,
		["a", "b", "c", "d"],
		["text_to_image"] * 4,
		config,
	)
	losses["loss"].backward()

	assert torch.isfinite(losses["loss"])
	assert candidates.grad is None
	assert output.step_embeddings[0].grad is None
	assert output.step_embeddings[-1].grad is not None
	assert output.step_embeddings[-1].grad.abs().sum() > 0
	assert torch.equal(losses["loss"], losses["final_mean_info_nce"])
	assert "step_1_info_nce" not in losses
	assert all(
		parameter.grad is not None and parameter.grad.abs().sum() > 0
		for parameter in head.parameters()
	)


def test_contrastive_loss_never_uses_candidates_from_another_gallery() -> None:
	queries = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
	candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

	(loss,) = multi_query_symmetric_info_nce(
		(queries,),
		candidates,
		["image:1", "image:2"],
		["text_to_image", "image_to_text"],
		temperature=1.0,
	)

	assert torch.allclose(loss, torch.zeros_like(loss))
