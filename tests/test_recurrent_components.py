from pathlib import Path

import pytest
import torch

from looped_vl.models.config import RecurrentModelConfig
from looped_vl.models.late_slot_fusion import EOSConditionedSlotFusion
from looped_vl.models.latent_slot_inserter import (
	augment_before_last_valid_token,
	create_or_load_master_slot_initialization,
)
from looped_vl.models.recurrent_connector import RecurrentConnector
from looped_vl.models.recurrent_decoder_block import (
	build_dynamic_attention_mask,
	detach_prefix_key_values,
)
from looped_vl.models.recurrent_qwen3vl_embedding import RecurrentQwen3VLEmbedding
from looped_vl.training.losses import slot_diversity_loss, symmetric_info_nce


class _CountingLinear(torch.nn.Linear):
	def __init__(self, features: int) -> None:
		super().__init__(features, features, bias=False)
		self.call_count = 0

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		self.call_count += 1
		return super().forward(inputs)


class _ProjectionCaptureAttention(torch.nn.Module):
	def __init__(self, features: int, head_dim: int) -> None:
		super().__init__()
		self.head_dim = head_dim
		self.k_proj = _CountingLinear(features)
		self.v_proj = _CountingLinear(features)
		self.k_norm = torch.nn.Identity()


class _ProjectionCaptureLayer(torch.nn.Module):
	def __init__(self, features: int, head_dim: int) -> None:
		super().__init__()
		self.self_attn = _ProjectionCaptureAttention(features, head_dim)

	def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
		batch_size, sequence_length, _ = hidden_states.shape
		attention = self.self_attn
		attention.k_norm(
			attention.k_proj(hidden_states).view(
				batch_size,
				sequence_length,
				-1,
				attention.head_dim,
			),
		)
		attention.v_proj(hidden_states)
		return hidden_states + 1


def test_base_configuration_matches_v1_specification() -> None:
	config = RecurrentModelConfig.from_yaml(Path("configs/base.yaml"))

	assert config.seed == 42
	assert config.hidden_size == 2048
	assert config.max_num_latent_slots == 16
	assert config.num_latent_slots == 4
	assert config.loop_start_layer == 12
	assert config.loop_end_layer == 20
	assert config.num_total_loop_passes == 4
	assert config.detach_prefix_kv_cache is True
	assert config.recurrent_bottleneck_dim == 512
	assert config.fusion_attention_dim == 256
	assert config.lora_rank == 32
	assert config.lora_alpha == 32
	assert config.lora_target_modules == (
		"q_proj",
		"k_proj",
		"v_proj",
		"up_proj",
		"down_proj",
		"gate_proj",
	)


def test_configuration_accepts_only_required_slot_and_loop_sweeps() -> None:
	config = RecurrentModelConfig()

	for slot_count in (1, 2, 4, 8, 16):
		config.with_variant(num_latent_slots=slot_count)
	for loop_count in (1, 2, 3, 4):
		config.with_variant(num_total_loop_passes=loop_count)

	with pytest.raises(ValueError, match="num_latent_slots"):
		config.with_variant(num_latent_slots=3)
	with pytest.raises(ValueError, match="num_total_loop_passes"):
		config.with_variant(num_total_loop_passes=5)


def test_slots_are_inserted_immediately_before_each_last_valid_token() -> None:
	input_ids = torch.tensor(
		[
			[10, 11, 99, 0, 0],
			[20, 21, 22, 23, 99],
		],
	)
	attention_mask = torch.tensor(
		[
			[1, 1, 1, 0, 0],
			[1, 1, 1, 1, 1],
		],
	)

	augmented = augment_before_last_valid_token(
		input_ids=input_ids,
		attention_mask=attention_mask,
		num_latent_slots=2,
		latent_placeholder_id=777,
		pad_token_id=0,
	)

	assert augmented.input_ids.tolist() == [
		[10, 11, 777, 777, 99, 0, 0],
		[20, 21, 22, 23, 777, 777, 99],
	]
	assert augmented.attention_mask.tolist() == [
		[1, 1, 1, 1, 1, 0, 0],
		[1, 1, 1, 1, 1, 1, 1],
	]
	assert augmented.prefix_lengths.tolist() == [2, 4]
	assert augmented.slot_positions.tolist() == [[2, 3], [4, 5]]
	assert augmented.eos_positions.tolist() == [4, 6]


def test_slot_insertion_handles_zero_slots_and_extra_trailing_padding() -> None:
	input_ids = torch.tensor(
		[
			[10, 99, 0, 0],
			[20, 21, 99, 0],
		],
	)
	attention_mask = torch.tensor(
		[
			[1, 1, 0, 0],
			[1, 1, 1, 0],
		],
	)

	augmented = augment_before_last_valid_token(
		input_ids=input_ids,
		attention_mask=attention_mask,
		num_latent_slots=0,
		latent_placeholder_id=777,
		pad_token_id=0,
	)

	assert torch.equal(augmented.input_ids, input_ids)
	assert torch.equal(augmented.attention_mask, attention_mask)
	assert augmented.slot_positions.shape == (2, 0)
	assert augmented.eos_positions.tolist() == [1, 2]


def test_master_slot_initialization_is_seeded_once_and_sliced(tmp_path: Path) -> None:
	path = tmp_path / "master_slot_init_seed42.pt"
	full = create_or_load_master_slot_initialization(
		path=path,
		max_num_latent_slots=16,
		hidden_size=32,
		seed=42,
		mean=0.0,
		std=0.02,
	)
	second_load = create_or_load_master_slot_initialization(
		path=path,
		max_num_latent_slots=16,
		hidden_size=32,
		seed=42,
		mean=0.0,
		std=0.02,
	)

	assert path.is_file()
	assert full.shape == (1, 16, 32)
	assert torch.equal(full, second_load)
	assert torch.equal(full[:, :4], second_load[:, :4])


def test_recurrent_connector_starts_as_an_exact_zero_residual() -> None:
	connector = RecurrentConnector(hidden_size=32, bottleneck_dim=8)
	hidden_states = torch.randn(2, 5, 32)

	output = connector(hidden_states)

	assert output.abs().max().item() < 1e-7
	assert connector.up_projection.weight.count_nonzero().item() == 0


def test_recurrent_components_run_after_bfloat16_precision_alignment() -> None:
	connector = RecurrentConnector(hidden_size=32, bottleneck_dim=8).to(torch.bfloat16)
	fusion = EOSConditionedSlotFusion(hidden_size=32, attention_dim=8).to(torch.bfloat16)
	hidden_states = torch.randn(2, 5, 32, dtype=torch.bfloat16)

	connector_output = connector(hidden_states)
	fusion_output = fusion(hidden_states[:, -1], hidden_states[:, :4])

	assert connector_output.dtype == torch.bfloat16
	assert fusion_output.fused_embedding.dtype == torch.bfloat16


def test_late_fusion_starts_as_identity_and_k1_attention_is_one() -> None:
	fusion = EOSConditionedSlotFusion(hidden_size=32, attention_dim=8)
	eos = torch.randn(2, 32)
	slots = torch.randn(2, 1, 32)

	result = fusion(eos, slots)

	assert (result.fused_embedding - eos).abs().max().item() < 1e-7
	assert torch.equal(result.attention_weights, torch.ones(2, 1))
	assert result.gate.item() == pytest.approx(0.0)


def test_dynamic_attention_mask_preserves_prefix_padding_and_slot_causality() -> None:
	prefix_attention_mask = torch.tensor(
		[
			[1, 1, 0],
			[1, 1, 1],
		],
		dtype=torch.bool,
	)

	mask = build_dynamic_attention_mask(
		prefix_attention_mask,
		dynamic_token_count=3,
		dtype=torch.float32,
	)
	visible = mask == 0

	assert visible.shape == (2, 1, 3, 6)
	assert visible[0, 0].tolist() == [
		[True, True, False, True, False, False],
		[True, True, False, True, True, False],
		[True, True, False, True, True, True],
	]


def test_prefix_key_value_cache_is_detached() -> None:
	key = torch.randn(2, 8, 5, 4, requires_grad=True)
	value = torch.randn(2, 8, 5, 4, requires_grad=True)

	detached_key, detached_value = detach_prefix_key_values(key, value)

	assert detached_key.requires_grad is False
	assert detached_value.requires_grad is False
	assert detached_key.grad_fn is None
	assert detached_value.grad_fn is None


def test_pass_one_prefix_cache_reuses_projected_key_and_value() -> None:
	layer = _ProjectionCaptureLayer(features=8, head_dim=4)
	hidden_states = torch.randn(2, 6, 8, requires_grad=True)
	cos = torch.ones(2, 6, 4)
	sin = torch.zeros(2, 6, 4)

	output, cache = RecurrentQwen3VLEmbedding._run_full_layer_and_capture_prefix(
		layer=layer,
		hidden_states=hidden_states,
		position_embeddings=(cos, sin),
		max_prefix_length=3,
		attention_mask=None,
		position_ids=torch.arange(6).expand(2, -1),
		cache_position=torch.arange(6),
	)

	assert torch.equal(output, hidden_states + 1)
	assert layer.self_attn.k_proj.call_count == 1
	assert layer.self_attn.v_proj.call_count == 1
	assert cache.key.shape == (2, 2, 3, 4)
	assert cache.value.shape == (2, 2, 3, 4)
	assert cache.key.requires_grad is False
	assert cache.value.requires_grad is False


def test_slot_losses_cover_k1_and_symmetric_contrastive_learning() -> None:
	single_slot = torch.randn(3, 1, 16)
	query = torch.eye(4)
	candidate = torch.eye(4)

	assert slot_diversity_loss(single_slot).item() == pytest.approx(0.0)
	assert symmetric_info_nce(query, candidate, temperature=0.02).item() < 1e-6
