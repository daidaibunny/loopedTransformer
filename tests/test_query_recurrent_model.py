from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from looped_vl.query_recurrent.config import (
	DEFAULT_HISTORY_LAYERS,
	MAX_QUERY_RECURRENT_PARAMETERS,
	QUERY_RECURRENT_ARCHITECTURE,
	QUERY_RECURRENT_PROTOCOL,
	QueryRecurrentConfig,
)
from looped_vl.query_recurrent.model import (
	GroupedQueryRecurrentHead,
	QueryRecurrentHead,
	RecurrentHistoryLayer,
	query_recurrent_diagnostics,
	recurrent_gradient_group_norms,
)


def _inputs(config: QueryRecurrentConfig, batch_size: int = 3) -> dict[str, torch.Tensor]:
	history = torch.randn(
		batch_size,
		len(config.history_layers),
		6,
		config.hidden_size,
	)
	mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * batch_size)
	base = torch.nn.functional.normalize(torch.randn(batch_size, config.hidden_size), dim=-1)
	return {
		"history_hidden_states": history,
		"attention_mask": mask,
		"base_embeddings": base,
	}


class _RecordingAttention(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.query: torch.Tensor | None = None

	def forward(
		self,
		query: torch.Tensor,
		_key: torch.Tensor,
		_value: torch.Tensor,
		**_kwargs: object,
	) -> tuple[torch.Tensor, None]:
		self.query = query.detach().clone()
		return torch.zeros_like(query), None


def test_locked_query_recurrent_identity_is_frozen_candidate_no_lora() -> None:
	identity = QueryRecurrentConfig().identity()

	assert identity["architecture"] == QUERY_RECURRENT_ARCHITECTURE
	assert identity["protocol"] == QUERY_RECURRENT_PROTOCOL
	assert identity["backbone_frozen"] is True
	assert identity["candidate_backbone_executed"] is False
	assert identity["lora_enabled"] is False
	assert identity["formal_training_stages"] == 1
	assert identity["history_layers"] == DEFAULT_HISTORY_LAYERS


@pytest.mark.parametrize("slot_count", [1, 4, 8])
def test_slot_ablation_stays_below_five_million_parameters(slot_count: int) -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig(num_slots=slot_count))

	assert 4_800_000 < head.trainable_parameter_count < MAX_QUERY_RECURRENT_PARAMETERS


def test_zero_gate_starts_every_recurrent_step_at_the_frozen_embedding() -> None:
	config = QueryRecurrentConfig()
	head = QueryRecurrentHead(config)
	inputs = _inputs(config)

	output = head(**inputs)

	for embedding in output.step_embeddings:
		assert torch.allclose(embedding, inputs["base_embeddings"], atol=1e-6)
	assert torch.allclose(output.embeddings, inputs["base_embeddings"], atol=1e-6)
	assert torch.count_nonzero(head.output_projection.weight) > 0
	assert head.residual_gate.item() == 0.0
	assert not hasattr(head, "exit_controller")


def test_zero_gated_residual_scale_is_independent_of_projection_magnitude() -> None:
	config = QueryRecurrentConfig(max_recurrent_steps=1)
	head = QueryRecurrentHead(config)
	inputs = _inputs(config)
	with torch.no_grad():
		head.output_projection.weight.mul_(1_000.0)
		head.residual_gate.fill_(math.atanh(0.1))

	output = head(**inputs)
	movement = torch.linalg.vector_norm(
		output.embeddings - inputs["base_embeddings"],
		dim=-1,
	)

	assert movement.max().item() < 0.12


def test_recurrent_history_attention_keeps_persistent_slot_identities() -> None:
	config = QueryRecurrentConfig()
	layer = RecurrentHistoryLayer(config)
	self_attention = _RecordingAttention()
	history_attention = _RecordingAttention()
	layer.self_attention = self_attention
	layer.history_attention = history_attention
	slots = torch.zeros(1, 2, config.state_size)
	memory = torch.zeros(1, 3, config.state_size)
	memory_padding_mask = torch.zeros(1, 3, dtype=torch.bool)
	identity = torch.stack(
		(
			torch.tensor([1.0, -1.0] * (config.state_size // 2)),
			torch.tensor([-1.0, 1.0] * (config.state_size // 2)),
		),
	)

	layer(
		slots,
		memory,
		memory_padding_mask,
		slot_identity=identity,
	)

	assert history_attention.query is not None
	assert not torch.equal(
		history_attention.query[:, 0],
		history_attention.query[:, 1],
	)


def test_fixed_recurrence_returns_normalized_pass_outputs_and_final_pass() -> None:
	config = QueryRecurrentConfig()
	head = QueryRecurrentHead(config)
	output = head(**_inputs(config))

	assert len(output.step_embeddings) == 4
	assert len(output.slot_states) == 4
	assert len(output.slot_attention_weights) == 4
	assert torch.allclose(output.embeddings.norm(dim=1), torch.ones(3), atol=1e-5)
	assert torch.equal(output.embeddings, output.step_embeddings[-1])
	for weights in output.slot_attention_weights:
		assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_recurrent_diagnostics_expose_each_pass_movement_attention_and_collapse() -> None:
	base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
	pass_one = torch.nn.functional.normalize(
		torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
		dim=-1,
	)
	pass_two = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
	collapsed_slots = torch.tensor(
		[
			[[1.0, 0.0], [1.0, 0.0]],
			[[0.0, 1.0], [0.0, 1.0]],
		],
	)
	diverse_slots = torch.tensor(
		[
			[[1.0, 0.0], [0.0, 1.0]],
			[[1.0, 0.0], [0.0, 1.0]],
		],
	)
	output = SimpleNamespace(
		step_embeddings=(pass_one, pass_two),
		slot_states=(collapsed_slots, diverse_slots),
		slot_attention_weights=(
			torch.full((2, 2), 0.5),
			torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
		),
	)

	diagnostics = query_recurrent_diagnostics(output, base)

	assert diagnostics["step_1_embedding_delta_from_base_l2"].item() > 0
	assert diagnostics["step_1_embedding_delta_from_previous_l2"].item() > 0
	assert diagnostics["step_2_embedding_delta_from_previous_l2"].item() > 0
	assert diagnostics["step_1_slot_pairwise_absolute_cosine"].item() == pytest.approx(1.0)
	assert diagnostics["step_2_slot_pairwise_absolute_cosine"].item() == pytest.approx(0.0)
	assert diagnostics["step_1_slot_attention_normalized_entropy"].item() == pytest.approx(1.0)
	assert diagnostics["step_2_slot_attention_normalized_entropy"].item() == pytest.approx(0.0)
	assert diagnostics["step_1_slot_attention_max_weight"].item() == pytest.approx(0.5)
	assert diagnostics["step_2_slot_attention_max_weight"].item() == pytest.approx(1.0)


def test_zero_initialized_readout_exposes_first_step_gradient_starvation() -> None:
	config = QueryRecurrentConfig(num_slots=4, max_recurrent_steps=1)
	head = QueryRecurrentHead(config)
	inputs = _inputs(config, batch_size=2)
	optimizer = torch.optim.SGD(head.parameters(), lr=0.1)

	first_output = head(**inputs)
	first_output.embeddings.sum().backward()
	first_norms = recurrent_gradient_group_norms(head)

	assert first_norms["gradient_norm_residual_gate"].item() > 0
	assert first_norms["gradient_norm_output_projection"].item() == 0
	assert first_norms["gradient_norm_recurrent_layers"].item() == 0
	assert first_norms["gradient_norm_initializer"].item() == 0
	assert first_norms["gradient_norm_memory_projection"].item() == 0
	optimizer.step()
	optimizer.zero_grad(set_to_none=True)

	second_output = head(**inputs)
	second_output.embeddings.sum().backward()
	second_norms = recurrent_gradient_group_norms(head)

	assert second_norms["gradient_norm_output_projection"].item() > 0
	assert second_norms["gradient_norm_recurrent_layers"].item() > 0
	assert second_norms["gradient_norm_initializer"].item() > 0
	assert second_norms["gradient_norm_memory_projection"].item() > 0


def test_fixed_one_step_and_final_layer_history_are_supported_ablations() -> None:
	config = QueryRecurrentConfig(
		history_layers=(28,),
		max_recurrent_steps=1,
	)
	head = QueryRecurrentHead(config)
	output = head(**_inputs(config, batch_size=2))

	assert len(output.step_embeddings) == 1
	assert torch.equal(output.embeddings, output.step_embeddings[0])


def test_grouped_head_preserves_results_and_avoids_cross_bucket_padding() -> None:
	config = QueryRecurrentConfig(num_slots=4)
	head = QueryRecurrentHead(config)
	grouped_head = GroupedQueryRecurrentHead(head)
	short = _inputs(config, batch_size=1)
	long = _inputs(config, batch_size=2)
	long["history_hidden_states"] = torch.randn(2, len(config.history_layers), 9, 2048)
	long["attention_mask"] = torch.ones(2, 9, dtype=torch.long)

	output = grouped_head(
		feature_groups=(
			(
				(1,),
				short["history_hidden_states"],
				short["attention_mask"],
				short["base_embeddings"],
			),
			(
				(0, 2),
				long["history_hidden_states"],
				long["attention_mask"],
				long["base_embeddings"],
			),
		),
		total_rows=3,
	)
	short_output = head(**short)
	long_output = head(**long)

	assert torch.allclose(output.step_embeddings[-1][1], short_output.step_embeddings[-1][0])
	assert torch.allclose(output.step_embeddings[-1][0], long_output.step_embeddings[-1][0])
	assert torch.allclose(output.step_embeddings[-1][2], long_output.step_embeddings[-1][1])
	output.slot_states[-1].sum().backward()
	assert head.memory_projection.weight.grad is not None


@pytest.mark.parametrize(
	"changes,match",
	[
		({"num_slots": 2}, "num_slots"),
		({"max_recurrent_steps": 3}, "max_recurrent_steps"),
		({"history_layers": ()}, "history"),
		({"history_layers": (0, 28)}, "1 through 28"),
	],
)
def test_invalid_formal_variants_are_rejected(changes: dict[str, object], match: str) -> None:
	with pytest.raises(ValueError, match=match):
		QueryRecurrentConfig(**changes).validate()
