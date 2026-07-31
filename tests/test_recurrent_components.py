from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from looped_vl.models.config import RecurrentModelConfig
from looped_vl.models.late_slot_fusion import EOSConditionedSlotFusion
from looped_vl.models.latent_slot_inserter import (
	augment_before_last_valid_token,
	create_or_load_master_slot_initialization,
)
from looped_vl.models.loading import load_recurrent_components
from looped_vl.models.recurrent_connector import RecurrentConnector, RMSNorm
from looped_vl.models.recurrent_decoder_block import (
	build_dynamic_attention_mask,
	detach_prefix_key_values,
)
from looped_vl.models.recurrent_qwen3vl_embedding import (
	RecurrentQwen3VLEmbedding,
	_run_full_sequence_decoder_layer,
)
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


class _RecordedAddLayer(torch.nn.Module):
	def __init__(self, delta: torch.Tensor) -> None:
		super().__init__()
		self.register_buffer("delta", delta)
		self.inputs: list[torch.Tensor] = []

	def forward(self, hidden_states: torch.Tensor, **_kwargs: object) -> torch.Tensor:
		self.inputs.append(hidden_states.detach().clone())
		return hidden_states + self.delta


class _TinyLanguageModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.config = SimpleNamespace()
		self.prefix_layer = _RecordedAddLayer(torch.tensor([1.0, 0.0]))
		self.loop_layer = torch.nn.Identity()
		self.suffix_layer = _RecordedAddLayer(torch.tensor([0.0, 1.0]))
		self.layers = torch.nn.ModuleList(
			(self.prefix_layer, self.loop_layer, self.suffix_layer),
		)
		self.norm = torch.nn.Identity()

	def rotary_emb(
		self,
		hidden_states: torch.Tensor,
		_position_ids: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		return torch.ones_like(hidden_states), torch.zeros_like(hidden_states)


class _TinyMultimodalModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.language_model = _TinyLanguageModel()


class _TinyEmbeddingModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.model = _TinyMultimodalModel()


class _IdentityConnector(torch.nn.Module):
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return hidden_states


class _IdentitySlotFusion(torch.nn.Module):
	def forward(
		self,
		eos_hidden_state: torch.Tensor,
		slot_hidden_states: torch.Tensor,
	) -> SimpleNamespace:
		return SimpleNamespace(
			fused_embedding=eos_hidden_state,
			attention_weights=torch.ones(
				eos_hidden_state.shape[0],
				slot_hidden_states.shape[1],
			),
			attention_entropy=eos_hidden_state.new_zeros(eos_hidden_state.shape[0]),
			gate=eos_hidden_state.new_zeros(()),
		)


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
	assert not any(name.startswith("lora_") for name in vars(config))


def test_recurrent_loader_and_model_expose_no_lora_switch() -> None:
	assert "enable_lora" not in signature(load_recurrent_components).parameters
	assert "enable_lora" not in signature(RecurrentQwen3VLEmbedding).parameters


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


def test_project_rms_norm_matches_torch_reference() -> None:
	inputs = torch.randn(2, 3, 32)
	reference = torch.nn.RMSNorm(32, eps=1e-6)
	project = RMSNorm(32, eps=1e-6)
	project.weight.data.copy_(reference.weight)

	assert torch.allclose(project(inputs), reference(inputs), atol=1e-6, rtol=1e-6)


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


def test_full_sequence_activation_checkpointing_preserves_gradients() -> None:
	layer = _RecordedAddLayer(torch.tensor([1.0, -1.0]))
	plain_input = torch.randn(2, 3, 2, requires_grad=True)
	checkpointed_input = plain_input.detach().clone().requires_grad_(True)
	arguments = {
		"layer": layer,
		"attention_mask": None,
		"position_ids": torch.arange(3).expand(2, -1),
		"cache_position": torch.arange(3),
		"position_embeddings": (
			torch.ones(2, 3, 2),
			torch.zeros(2, 3, 2),
		),
	}

	plain = _run_full_sequence_decoder_layer(
		hidden_states=plain_input,
		activation_checkpointing=False,
		**arguments,
	)
	checkpointed = _run_full_sequence_decoder_layer(
		hidden_states=checkpointed_input,
		activation_checkpointing=True,
		**arguments,
	)
	plain.square().sum().backward()
	checkpointed.square().sum().backward()

	assert torch.equal(plain, checkpointed)
	assert torch.equal(plain_input.grad, checkpointed_input.grad)


def test_each_reported_pass_runs_its_loop_count_then_the_shared_suffix(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = RecurrentQwen3VLEmbedding.__new__(RecurrentQwen3VLEmbedding)
	torch.nn.Module.__init__(model)
	model.config = SimpleNamespace(
		loop_start_layer=1,
		loop_end_layer=2,
		num_extra_loop_passes=2,
		num_total_loop_passes=3,
		num_latent_slots=1,
	)
	model.base_embedding_model = _TinyEmbeddingModel()
	model.recurrent_connector = _IdentityConnector()
	model.late_fusion = _IdentitySlotFusion()
	model.activation_checkpointing_enabled = False
	full_loop_calls: list[torch.Tensor] = []
	dynamic_loop_calls: list[torch.Tensor] = []

	def fake_full_loop(
		**kwargs: object,
	) -> tuple[torch.Tensor, SimpleNamespace]:
		hidden_states = kwargs["hidden_states"]
		assert isinstance(hidden_states, torch.Tensor)
		full_loop_calls.append(hidden_states.detach().clone())
		cache = SimpleNamespace(
			key=hidden_states.new_zeros((1, 1, 2, 2)),
			value=hidden_states.new_zeros((1, 1, 2, 2)),
		)
		return hidden_states + torch.tensor([1.0, 0.0]), cache

	def fake_dynamic_loop(**kwargs: object) -> torch.Tensor:
		hidden_states = kwargs["dynamic_hidden_states"]
		assert isinstance(hidden_states, torch.Tensor)
		dynamic_loop_calls.append(hidden_states.detach().clone())
		return hidden_states + torch.tensor([1.0, 0.0])

	monkeypatch.setattr(
		"looped_vl.models.recurrent_qwen3vl_embedding.create_causal_mask",
		lambda **_kwargs: torch.zeros(1, 1, 4, 4),
	)
	monkeypatch.setattr(
		model,
		"_run_full_layer_and_capture_prefix",
		fake_full_loop,
	)
	monkeypatch.setattr(model, "_run_dynamic_layer", fake_dynamic_loop)
	augmented = SimpleNamespace(
		attention_mask=torch.ones(1, 4, dtype=torch.long),
		prefix_lengths=torch.tensor([2]),
		slot_positions=torch.tensor([[2]]),
		eos_positions=torch.tensor([3]),
	)

	output = model._run_recurrent_decoder(
		hidden_states=torch.zeros(1, 4, 2),
		augmented=augmented,
		position_ids=torch.arange(4).view(1, 1, 4),
		visual_position_mask=None,
		deepstack_visual_embeddings=None,
		return_all_loop_embeddings=True,
	)

	assert output.loop_embeddings is not None
	assert len(output.loop_embeddings) == 3
	expected = (
		torch.nn.functional.normalize(torch.tensor([[2.0, 1.0]]), dim=-1),
		torch.nn.functional.normalize(torch.tensor([[4.0, 1.0]]), dim=-1),
		torch.nn.functional.normalize(torch.tensor([[6.0, 1.0]]), dim=-1),
	)
	for actual, expected_embedding in zip(output.loop_embeddings, expected, strict=True):
		assert torch.allclose(actual, expected_embedding)
	assert torch.equal(output.embeddings, output.loop_embeddings[-1])
	assert len(full_loop_calls) == 1
	assert len(dynamic_loop_calls) == 2
	suffix_inputs = model.language_model.suffix_layer.inputs
	assert len(suffix_inputs) == 3
	for suffix_input in suffix_inputs:
		assert torch.equal(
			suffix_input[:, :2],
			torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]),
		)
	assert [value[0, 3, 0].item() for value in suffix_inputs] == [2.0, 4.0, 6.0]


def test_slot_losses_cover_k1_and_symmetric_contrastive_learning() -> None:
	single_slot = torch.randn(3, 1, 16)
	query = torch.eye(4)
	candidate = torch.eye(4)

	assert slot_diversity_loss(single_slot).item() == pytest.approx(0.0)
	assert symmetric_info_nce(query, candidate, temperature=0.02).item() < 1e-6
