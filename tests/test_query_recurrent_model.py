from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from looped_vl.query_recurrent.config import (
	MAX_QUERY_RECURRENT_PARAMETERS,
	QUERY_RECURRENT_ARCHITECTURE,
	QUERY_RECURRENT_PROTOCOL,
	QueryRecurrentConfig,
)
from looped_vl.query_recurrent.model import (
	GroupedQueryRecurrentHead,
	QueryRecurrentHead,
	query_recurrent_diagnostics,
	recurrent_fp32_context,
	recurrent_gradient_group_norms,
)


def _base_embeddings(batch_size: int = 3) -> torch.Tensor:
	return torch.nn.functional.normalize(torch.randn(batch_size, 2048), dim=-1)


def test_locked_identity_is_parallel_population_no_lora_final_mean() -> None:
	identity = QueryRecurrentConfig().identity()

	assert identity["architecture"] == QUERY_RECURRENT_ARCHITECTURE
	assert identity["protocol"] == QUERY_RECURRENT_PROTOCOL
	assert identity["backbone_frozen"] is True
	assert identity["candidate_backbone_executed"] is False
	assert identity["lora_enabled"] is False
	assert identity["formal_training_stages"] == 1
	assert identity["num_worlds"] == 4
	assert identity["max_recurrent_steps"] == 4
	assert identity["initialization"] == "query_conditioned_antithetic_zero_mean"
	assert identity["world_interaction"] == "shared_centered_bidirectional_attention"
	assert identity["readout"] == "final_world_mean_l2_normalized"
	assert identity["pass_supervision"] == "final_only"
	assert identity["dynamic_exit"] is False
	assert identity["recurrent_step_embeddings"] is False


def test_parameter_count_matches_last_four_layer_lora_budget() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig())

	assert head.trainable_parameter_count == 4_391_554
	assert head.trainable_parameter_count < MAX_QUERY_RECURRENT_PARAMETERS
	assert head.trainable_parameter_count < 4_456_448


def test_antithetic_initial_worlds_preserve_mean_and_two_axes() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig())
	base = _base_embeddings(batch_size=2)

	worlds = head.initialize_worlds(base)
	deviations = worlds - base[:, None, :]

	assert worlds.shape == (2, 4, 2048)
	assert torch.allclose(worlds.mean(dim=1), base, atol=1e-7)
	assert torch.allclose(deviations[:, 0], -deviations[:, 1], atol=1e-7)
	assert torch.allclose(deviations[:, 2], -deviations[:, 3], atol=1e-7)
	assert torch.allclose(
		(deviations[:, 0] * deviations[:, 2]).sum(dim=-1),
		torch.zeros(2),
		atol=1e-5,
	)
	assert torch.allclose(
		deviations[:, 0].norm(dim=-1),
		base.norm(dim=-1) * head.config.perturbation_scale,
		atol=1e-5,
	)


def test_shared_cell_is_permutation_equivariant_over_worlds() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig(max_recurrent_steps=1))
	worlds = head.initialize_worlds(_base_embeddings(batch_size=2))
	permutation = torch.tensor([2, 0, 3, 1])

	original, original_attention = head.recurrent_cell(worlds)
	permuted, permuted_attention = head.recurrent_cell(worlds[:, permutation])

	assert torch.allclose(permuted, original[:, permutation], atol=1e-5)
	assert torch.allclose(
		permuted_attention,
		original_attention[:, permutation][:, :, permutation],
		atol=1e-5,
	)


def test_fixed_recurrence_returns_only_mean_pooled_unit_embeddings() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig())
	base = _base_embeddings()

	output = head(base_embeddings=base)

	assert len(output.step_embeddings) == 4
	assert len(output.world_states) == 4
	assert len(output.interaction_weights) == 4
	assert output.initial_world_states.shape == (3, 4, 2048)
	assert torch.allclose(output.initial_world_states.mean(dim=1), base, atol=1e-7)
	assert torch.equal(output.embeddings, output.step_embeddings[-1])
	for step, worlds in zip(output.step_embeddings, output.world_states, strict=True):
		assert torch.allclose(step.norm(dim=-1), torch.ones(3), atol=1e-5)
		assert torch.allclose(
			step,
			torch.nn.functional.normalize(worlds.mean(dim=1).float(), dim=-1),
			atol=1e-6,
		)
	assert not hasattr(head, "recurrent_step_embeddings")
	assert not hasattr(head, "exit_controller")
	assert not hasattr(output, "slot_bridge_embeddings")


def test_recurrent_block_stays_float32_inside_outer_mixed_precision() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig(max_recurrent_steps=1))
	base = _base_embeddings(batch_size=2)

	with (
		torch.autocast(device_type="cpu", dtype=torch.bfloat16),
		recurrent_fp32_context("cpu"),
	):
		output = head(base_embeddings=base)

	assert output.embeddings.dtype == torch.float32
	assert output.world_states[0].dtype == torch.float32


def test_one_shared_cell_is_reused_for_every_recurrent_step() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig(max_recurrent_steps=4))

	assert hasattr(head, "recurrent_cell")
	assert not hasattr(head, "recurrent_layers")
	assert sum(1 for name, _module in head.named_modules() if name == "recurrent_cell") == 1


def test_grouped_head_restores_logical_order_before_one_population_forward() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig())
	grouped_head = GroupedQueryRecurrentHead(head)
	short = _base_embeddings(batch_size=1)
	long = _base_embeddings(batch_size=2)

	output = grouped_head(
		feature_groups=(
			((1,), short),
			((0, 2), long),
		),
		total_rows=3,
	)
	direct_base = torch.stack((long[0], short[0], long[1]))
	direct = head(base_embeddings=direct_base)

	assert torch.allclose(output.embeddings, direct.embeddings)
	assert torch.allclose(output.initial_world_states, direct.initial_world_states)


def test_final_loss_reaches_every_trainable_component_on_first_step() -> None:
	head = QueryRecurrentHead(QueryRecurrentConfig(max_recurrent_steps=2))
	output = head(base_embeddings=_base_embeddings(batch_size=4))

	output.embeddings.square().sum().backward()
	gradient_norms = recurrent_gradient_group_norms(head)

	assert all(value.item() > 0 for value in gradient_norms.values())


def test_diagnostics_measure_population_spread_interaction_and_mean_motion() -> None:
	base = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
	initial = torch.stack((base + 0.1, base - 0.1), dim=1)
	step_one_worlds = initial + 0.05
	step_two_worlds = step_one_worlds + torch.tensor([0.1, 0.0, 0.0, 0.0])
	step_one = torch.nn.functional.normalize(step_one_worlds.mean(dim=1), dim=-1)
	step_two = torch.nn.functional.normalize(step_two_worlds.mean(dim=1), dim=-1)
	weights = torch.full((2, 2, 2), 0.5)
	output = SimpleNamespace(
		step_embeddings=(step_one, step_two),
		world_states=(step_one_worlds, step_two_worlds),
		interaction_weights=(weights, weights),
		initial_world_states=initial,
	)

	diagnostics = query_recurrent_diagnostics(output, base)

	assert diagnostics["initial_population_mean_error_l2"].item() < 1e-7
	assert diagnostics["step_1_embedding_delta_from_base_l2"].item() > 0
	assert diagnostics["step_2_embedding_delta_from_previous_l2"].item() > 0
	assert diagnostics["step_1_population_spread_l2"].item() > 0
	assert diagnostics["step_1_interaction_normalized_entropy"].item() == pytest.approx(1.0)
	assert diagnostics["step_1_interaction_off_diagonal_mass"].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
	("changes", "match"),
	[
		({"hidden_size": 1024}, "2048"),
		({"attention_size": 318}, "attention_size"),
		({"num_worlds": 3}, "num_worlds"),
		({"max_recurrent_steps": 5}, "max_recurrent_steps"),
		({"perturbation_scale": 0.0}, "perturbation_scale"),
		({"pass_supervision": "every_pass"}, "pass_supervision"),
	],
)
def test_invalid_formal_variants_are_rejected(changes: dict[str, object], match: str) -> None:
	with pytest.raises(ValueError, match=match):
		QueryRecurrentConfig(**changes).validate()
