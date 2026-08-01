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
from looped_vl.query_recurrent.losses import query_recurrent_loss
from looped_vl.query_recurrent.model import QueryRecurrentHead


class _FakeInnerQwen(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.tensor(1.0))
		self.call_count = 0

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_kwargs: object):
		self.call_count += 1
		base = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048)
		hidden_states = tuple(base + layer for layer in range(29))
		return SimpleNamespace(hidden_states=hidden_states, last_hidden_state=hidden_states[-1])


class _FakeQwenWrapper(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.model = _FakeInnerQwen()


class _AddOneDecoderLayer(nn.Module):
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return hidden_states + 1


class _FakeLanguageModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.layers = nn.ModuleList(_AddOneDecoderLayer() for _ in range(28))


class _HookedInnerQwen(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.language_model = _FakeLanguageModel()
		self.requested_all_hidden_states: bool | None = None

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		*,
		output_hidden_states: bool,
		**_kwargs: object,
	):
		del attention_mask
		self.requested_all_hidden_states = output_hidden_states
		hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048)
		for layer in self.language_model.layers:
			hidden = layer(hidden)
		return SimpleNamespace(hidden_states=None, last_hidden_state=hidden + 100)


class _HookedQwenWrapper(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.model = _HookedInnerQwen()


def test_frozen_query_backbone_runs_once_and_returns_only_selected_histories() -> None:
	model = _FakeQwenWrapper()
	backbone = FrozenQueryBackbone(model, (7, 14, 21, 28))
	inputs = {
		"input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
		"attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
	}

	features = backbone(inputs)

	assert model.model.call_count == 1
	assert features.history_hidden_states.shape == (2, 4, 3, 2048)
	assert features.base_embeddings.shape == (2, 2048)
	assert all(parameter.requires_grad is False for parameter in backbone.parameters())
	assert features.history_hidden_states.requires_grad is False
	assert torch.allclose(features.base_embeddings.norm(dim=1), torch.ones(2))


def test_frozen_histories_can_feed_a_trainable_recurrent_backward() -> None:
	model = _FakeQwenWrapper()
	config = QueryRecurrentConfig(num_slots=1)
	backbone = FrozenQueryBackbone(model, config.history_layers)
	head = QueryRecurrentHead(config)
	features = backbone(
		{
			"input_ids": torch.tensor([[1, 2, 3]]),
			"attention_mask": torch.ones(1, 3, dtype=torch.long),
		},
	)

	output = head(
		history_hidden_states=features.history_hidden_states,
		attention_mask=features.attention_mask,
		base_embeddings=features.base_embeddings,
	)
	output.auxiliary_embeddings[-1].sum().backward()

	assert head.memory_projection.weight.grad is not None
	assert model.model.weight.grad is None


def test_frozen_backbone_hooks_only_requested_layers_and_uses_final_norm_for_28() -> None:
	model = _HookedQwenWrapper()
	backbone = FrozenQueryBackbone(model, (7, 14, 21, 28))
	inputs = {
		"input_ids": torch.tensor([[1, 2]]),
		"attention_mask": torch.ones(1, 2, dtype=torch.long),
	}

	features = backbone(inputs)

	assert model.model.requested_all_hidden_states is False
	assert features.history_hidden_states[0, :, 0, 0].tolist() == [8.0, 15.0, 22.0, 129.0]


def test_frozen_feature_groups_restore_order_and_pad_only_head_memory() -> None:
	first = FrozenQueryFeatures(
		history_hidden_states=torch.full((1, 1, 2, 4), 2.0),
		attention_mask=torch.tensor([[1, 1]]),
		base_embeddings=torch.full((1, 4), 2.0),
	)
	second = FrozenQueryFeatures(
		history_hidden_states=torch.full((2, 1, 3, 4), 5.0),
		attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
		base_embeddings=torch.tensor([[5.0] * 4, [7.0] * 4]),
	)

	combined = combine_frozen_query_groups(
		[((1,), first), ((0, 2), second)],
		total_rows=3,
	)

	assert combined.history_hidden_states.shape == (3, 1, 3, 4)
	assert combined.base_embeddings[:, 0].tolist() == [5.0, 2.0, 7.0]
	assert combined.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0], [1, 1, 0]]


def test_query_recurrent_loss_backpropagates_without_candidate_gradients() -> None:
	config = QueryRecurrentConfig(num_slots=4)
	head = QueryRecurrentHead(config)
	history = torch.randn(3, 4, 5, 2048)
	mask = torch.ones(3, 5, dtype=torch.long)
	base = torch.nn.functional.normalize(torch.randn(3, 2048), dim=-1)
	candidates = torch.nn.functional.normalize(torch.randn(3, 2048), dim=-1)
	output = head(
		history_hidden_states=history,
		attention_mask=mask,
		base_embeddings=base,
	)

	losses = query_recurrent_loss(
		output,
		candidates,
		["same", "same", "different"],
		config,
	)
	losses["loss"].backward()

	assert torch.isfinite(losses["loss"])
	assert candidates.grad is None
	assert head.output_projection.weight.grad is not None
	assert head.output_projection.weight.grad.abs().sum() > 0
	attention = head.recurrent_layers[0].self_attention
	assert attention.in_proj_weight.grad is not None
	assert attention.in_proj_weight.grad.abs().sum() > 0
