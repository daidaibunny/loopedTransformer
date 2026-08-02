from __future__ import annotations

import pytest
import torch

from looped_vl.query_recurrent.config import (
	DEFAULT_HISTORY_LAYERS,
	MAX_QUERY_RECURRENT_PARAMETERS,
	QUERY_RECURRENT_ARCHITECTURE,
	QUERY_RECURRENT_PROTOCOL,
	QueryRecurrentConfig,
)
from looped_vl.query_recurrent.model import GroupedQueryRecurrentHead, QueryRecurrentHead


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
	config = QueryRecurrentConfig(exit_mode="fixed")
	head = QueryRecurrentHead(config)
	inputs = _inputs(config)

	output = head(**inputs)

	for embedding in output.step_embeddings:
		assert torch.allclose(embedding, inputs["base_embeddings"], atol=1e-6)
	assert torch.allclose(output.embeddings, inputs["base_embeddings"], atol=1e-6)
	assert torch.count_nonzero(head.output_projection.weight) == 0
	assert not hasattr(head, "residual_gate")


def test_dynamic_exit_returns_normalized_step_outputs_and_valid_halting_weights() -> None:
	config = QueryRecurrentConfig()
	head = QueryRecurrentHead(config)
	output = head(**_inputs(config))

	assert len(output.step_embeddings) == 4
	assert len(output.slot_states) == 4
	assert len(output.slot_attention_weights) == 4
	assert output.exit_probabilities.shape == (3, 4)
	assert output.halting_weights.shape == (3, 4)
	assert torch.allclose(output.halting_weights.sum(dim=1), torch.ones(3))
	assert torch.allclose(output.embeddings.norm(dim=1), torch.ones(3), atol=1e-5)
	assert torch.allclose(output.soft_embeddings.norm(dim=1), torch.ones(3), atol=1e-5)
	assert torch.equal(output.selected_steps, torch.full((3,), 4))
	assert (output.expected_steps >= 1).all()
	assert (output.expected_steps <= 4).all()
	for weights in output.slot_attention_weights:
		assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_fixed_one_step_and_final_layer_history_are_supported_ablations() -> None:
	config = QueryRecurrentConfig(
		history_layers=(28,),
		max_recurrent_steps=1,
		exit_mode="fixed",
	)
	head = QueryRecurrentHead(config)
	output = head(**_inputs(config, batch_size=2))

	assert len(output.step_embeddings) == 1
	assert torch.equal(output.selected_steps, torch.ones(2, dtype=torch.long))


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
		({"max_recurrent_steps": 1, "exit_mode": "dynamic"}, "Dynamic exit"),
	],
)
def test_invalid_formal_variants_are_rejected(changes: dict[str, object], match: str) -> None:
	with pytest.raises(ValueError, match=match):
		QueryRecurrentConfig(**changes).validate()
