from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from looped_vl.models.config import (
	PURE_RECURRENT_ARCHITECTURE,
	PURE_RECURRENT_TRAINING_PROTOCOL,
	RecurrentModelConfig,
)
from looped_vl.models.recurrent_qwen3vl_embedding import (
	RecurrentQwen3VLEmbedding,
	recurrent_update,
)
from looped_vl.models.warmup_heads import eos_conditioned_slot_attention_embedding
from looped_vl.training.step import compose_training_loss, progressive_non_degradation_loss
from looped_vl.training.trainability import configure_trainable_parameters


class _TinyBaseEmbeddingModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.frozen_weight = nn.Parameter(torch.zeros(1))


def _build_encoder(config: RecurrentModelConfig | None = None) -> RecurrentQwen3VLEmbedding:
	resolved = config or RecurrentModelConfig()
	return RecurrentQwen3VLEmbedding(
		base_embedding_model=_TinyBaseEmbeddingModel(),
		config=resolved,
		master_slot_initialization=torch.zeros(
			1,
			resolved.max_num_latent_slots,
			resolved.hidden_size,
		),
		latent_placeholder_id=7,
		pad_token_id=0,
	)


def test_v5_configuration_adds_effective_depth_without_lora_or_fusion_shortcuts() -> None:
	config = RecurrentModelConfig.from_yaml(Path("configs/base.yaml"))

	assert PURE_RECURRENT_ARCHITECTURE == (
		"direct_eos_layerscale_mid_decoder_recurrence_no_lora_v5"
	)
	assert PURE_RECURRENT_TRAINING_PROTOCOL == (
		"single_stage_progressive_slot_attention_no_lora_v5"
	)
	assert config.recurrent_step_size == 1.0
	assert config.use_recurrent_layer_scale is True
	assert config.final_readout == "direct_eos_after_suffix"
	assert config.auxiliary_output_dim == 2048
	assert config.slot_diversity_weight == 0.0


def test_v5_recurrent_update_uses_an_r_independent_step_size() -> None:
	previous = torch.tensor([[[0.0, 4.0]]])
	proposal = torch.tensor([[[4.0, 0.0]]])

	assert torch.equal(
		recurrent_update(previous, proposal, step_size=1.0),
		proposal,
	)
	assert torch.equal(
		recurrent_update(previous, proposal, step_size=0.5),
		torch.tensor([[[2.0, 2.0]]]),
	)


def test_v5_has_only_slots_and_shared_recurrent_layer_scales_trainable() -> None:
	encoder = _build_encoder()
	groups = configure_trainable_parameters(encoder)

	assert encoder.recurrent_layer_scales.shape == (8, 2048)
	assert not hasattr(encoder, "eos_delta")
	assert not hasattr(encoder, "late_fusion")
	assert not hasattr(encoder, "auxiliary_embedding_head")
	assert groups.final_fusion == ()
	assert set(groups.recurrent_core) == {"latent_slots", "recurrent_layer_scales"}
	assert sum(
		parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
	) == 32_768


def test_parameter_free_slot_attention_embedding_keeps_2048_dimensional_signal() -> None:
	slots = torch.randn(3, 4, 32, requires_grad=True)
	eos = torch.randn(3, 32, requires_grad=True)

	embeddings, weights = eos_conditioned_slot_attention_embedding(slots, eos)

	assert embeddings.shape == (3, 32)
	assert weights.shape == (3, 4)
	assert torch.allclose(embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)
	assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_progressive_loss_only_pushes_a_worse_later_pass() -> None:
	pass_one = torch.tensor(1.0, requires_grad=True)
	pass_two = torch.tensor(1.5, requires_grad=True)
	pass_three = torch.tensor(1.2, requires_grad=True)

	progressive = progressive_non_degradation_loss((pass_one, pass_two, pass_three))
	total = compose_training_loss(
		final_infonce=torch.tensor(10.0),
		loop_infonce=pass_three,
		progressive_loss=progressive,
		slot_diversity=torch.tensor(9.0),
	)
	total.backward()

	assert progressive.item() == pytest.approx((0.5 + 0.0) / 2)
	assert total.item() == pytest.approx(10.0 + 0.1 * 1.2 + 0.1 * 0.25)
	assert pass_one.grad is None
	assert pass_two.grad is not None
	assert pass_three.grad is not None
