from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from looped_vl.models.config import (
	DAMPED_RECURRENT_ARCHITECTURE,
	DAMPED_RECURRENT_TRAINING_PROTOCOL,
	RecurrentModelConfig,
	pure_recurrent_result_identity,
)
from looped_vl.models.recurrent_qwen3vl_embedding import (
	RecurrentQwen3VLEmbedding,
	damped_recurrent_update,
)
from looped_vl.models.warmup_heads import AuxiliarySlotRetrievalHead
from looped_vl.training.losses import slot_diversity_loss
from looped_vl.training.model import RecurrentTrainingModel
from looped_vl.training.step import compose_training_loss
from looped_vl.training.train import _set_training_modes
from looped_vl.training.trainability import configure_trainable_parameters


class _TinyBaseEmbeddingModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.frozen_weight = nn.Parameter(torch.zeros(1))


def _build_encoder() -> RecurrentQwen3VLEmbedding:
	config = RecurrentModelConfig()
	return RecurrentQwen3VLEmbedding(
		base_embedding_model=_TinyBaseEmbeddingModel(),
		config=config,
		master_slot_initialization=torch.zeros(1, 16, 2048),
		latent_placeholder_id=7,
		pad_token_id=0,
	)


def test_locked_configuration_uses_eight_slots_and_full_recurrent_updates() -> None:
	config = RecurrentModelConfig.from_yaml(Path("configs/base.yaml"))

	assert config.num_latent_slots == 8
	assert config.num_total_loop_passes == 4
	assert config.recurrent_step_size == pytest.approx(1.0)
	assert config.use_recurrent_layer_scale is True
	assert not any("connector" in name or name.startswith("lora_") for name in vars(config))


def test_result_identity_is_single_stage_and_explicitly_excludes_lora() -> None:
	assert DAMPED_RECURRENT_ARCHITECTURE == (
		"direct_eos_layerscale_mid_decoder_recurrence_no_lora_v5"
	)
	assert DAMPED_RECURRENT_TRAINING_PROTOCOL == (
		"single_stage_progressive_slot_attention_no_lora_v5"
	)
	assert pure_recurrent_result_identity() == {
		"architecture": DAMPED_RECURRENT_ARCHITECTURE,
		"training_protocol": DAMPED_RECURRENT_TRAINING_PROTOCOL,
		"backbone_frozen": True,
		"lora_enabled": False,
		"formal_training_stages": 1,
	}


def test_encoder_keeps_only_active_slots_and_has_no_recurrent_connector() -> None:
	encoder = _build_encoder()

	assert encoder.latent_slots.shape == (1, 8, 2048)
	assert not hasattr(encoder, "recurrent_connector")
	assert encoder.recurrent_layer_scales.shape == (8, 2048)
	assert not hasattr(encoder, "auxiliary_embedding_head")
	assert not hasattr(encoder, "late_fusion")


def test_v5_training_mode_does_not_require_removed_auxiliary_or_fusion_modules() -> None:
	encoder = _build_encoder()
	training_model = RecurrentTrainingModel(encoder)

	_set_training_modes(training_model)

	assert training_model.training is True
	assert encoder.base_embedding_model.training is False


def test_auxiliary_head_matches_shared_256_dimensional_specification() -> None:
	head = AuxiliarySlotRetrievalHead(hidden_size=32, output_size=8)
	slots = torch.randn(3, 4, 32)
	eos = torch.randn(3, 32)

	embeddings = head(slots, eos)

	assert embeddings.shape == (3, 8)
	assert head.projection.bias is None
	assert torch.allclose(embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)
	assert sum(parameter.numel() for parameter in head.parameters()) == 32 + 32 * 8


def test_damped_update_uses_inverse_total_pass_count() -> None:
	previous = torch.tensor([[[0.0, 4.0]]])
	proposal = torch.tensor([[[4.0, 0.0]]])

	assert torch.equal(
		damped_recurrent_update(previous, proposal, total_passes=4),
		torch.tensor([[[1.0, 3.0]]]),
	)
	assert torch.equal(
		damped_recurrent_update(previous, proposal, total_passes=1),
		proposal,
	)


def test_slot_diversity_penalizes_negative_and_positive_correlation_equally() -> None:
	slots = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])

	assert slot_diversity_loss(slots).item() == pytest.approx(1.0)


def test_single_stage_loss_has_fixed_weights_from_the_first_step() -> None:
	total = compose_training_loss(
		final_infonce=torch.tensor(10.0),
		loop_infonce=torch.tensor(2.0),
		progressive_loss=torch.tensor(3.0),
		slot_diversity=torch.tensor(4.0),
	)

	assert total.item() == pytest.approx(10.0 + 0.1 * 2.0 + 0.1 * 3.0)


def test_trainable_parameter_count_matches_no_lora_connector_free_design() -> None:
	encoder = _build_encoder()

	groups = configure_trainable_parameters(encoder)
	trainable_count = sum(
		parameter.numel()
		for parameter in encoder.parameters()
		if parameter.requires_grad
	)

	assert not any("lora_" in name or "connector" in name for name in groups.all)
	assert set(groups.all) == {"latent_slots", "recurrent_layer_scales"}
	assert trainable_count == 32_768


def test_active_latent_slot_parameter_has_standard_contiguous_strides() -> None:
	encoder = _build_encoder()

	assert encoder.latent_slots.shape == (1, 8, 2048)
	assert encoder.latent_slots.stride() == (8 * 2048, 2048, 1)


def test_training_uses_final_slot_loss_and_progressive_non_degradation() -> None:
	class _FakeEncoder(nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.config = SimpleNamespace(temperature=0.02)
			self.auxiliary_embedding_head = AuxiliarySlotRetrievalHead(
				hidden_size=4,
				output_size=2,
			)

		def forward(self, values: torch.Tensor) -> SimpleNamespace:
			final_embeddings = torch.nn.functional.normalize(values[:, :2], dim=-1)
			loop_slots = tuple(
				values[:, None, :] + pass_index
				for pass_index in range(4)
			)
			zero = values.new_zeros(())
			return SimpleNamespace(
				embeddings=final_embeddings,
				loop_slot_hidden_states=loop_slots,
				conditioning_eos_hidden_state=values,
				slot_hidden_states=loop_slots[-1],
			diagnostics={
					"eos_conditioned_slot_attention_entropy": zero,
					"slot_pairwise_cosine": zero,
					"recurrent_pass_cosine": (zero, zero, zero, zero),
					"recurrent_pass_relative_update": (zero, zero, zero, zero),
				},
			)

	model = RecurrentTrainingModel(_FakeEncoder())  # type: ignore[arg-type]
	output = model(
		local_batch_size=2,
		positive_ids=["first", "second"],
		processed_batches=(
			{
				"values": torch.tensor(
					[
						[1.0, 0.0, 0.0, 0.0],
						[0.0, 1.0, 0.0, 0.0],
						[1.0, 0.0, 0.0, 0.0],
						[0.0, 1.0, 0.0, 0.0],
					],
				),
			},
		),
		original_indices=((0, 1, 2, 3),),
	)

	assert len(output["loop_infonce_by_pass"]) == 4
	assert torch.equal(
		output["loop_infonce"],
		output["loop_infonce_by_pass"][-1],
	)
	assert output["progressive_loss"].ndim == 0
