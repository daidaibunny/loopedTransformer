from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from looped_vl.baseline.model import (
	BASELINE_LORA_ALPHA,
	BASELINE_LORA_LAST_FOUR_DECODER_LAYERS,
	BASELINE_LORA_RANK,
	BASELINE_LORA_TARGETS,
	BaselineLoRATrainingModel,
	QueryOnlyLoRATrainingModel,
	build_lora_config,
	describe_lora_decoder_scope,
	encode_grouped_baseline_batches,
	load_frozen_evaluation_model,
)


def test_baseline_lora_matches_official_qwen_embedding_configuration() -> None:
	config = build_lora_config()

	assert BASELINE_LORA_RANK == 32
	assert BASELINE_LORA_ALPHA == 32
	assert BASELINE_LORA_TARGETS == (
		"q_proj",
		"v_proj",
		"k_proj",
		"up_proj",
		"down_proj",
		"gate_proj",
	)
	assert config.r == 32
	assert config.lora_alpha == 32
	assert config.target_modules == set(BASELINE_LORA_TARGETS)
	assert describe_lora_decoder_scope(config) == {
		"scope": "all_decoder_layers",
		"decoder_layer_indices": None,
	}
	assert config.lora_dropout == 0.0
	assert config.layers_to_transform is None


def test_baseline_lora_can_target_only_the_last_four_decoder_layers() -> None:
	config = build_lora_config(
		decoder_layer_indices=BASELINE_LORA_LAST_FOUR_DECODER_LAYERS,
	)

	assert BASELINE_LORA_LAST_FOUR_DECODER_LAYERS == (24, 25, 26, 27)
	assert config.layers_to_transform == [24, 25, 26, 27]
	assert config.layers_pattern == "layers"
	assert config.target_modules == set(BASELINE_LORA_TARGETS)
	assert describe_lora_decoder_scope(config) == {
		"scope": "last_4_decoder_layers",
		"decoder_layer_indices": [24, 25, 26, 27],
	}


class _FakeEmbeddingModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.forward_calls = 0

	def forward(
		self,
		input_values: torch.Tensor,
		attention_mask: torch.Tensor,
	) -> object:
		self.forward_calls += 1
		return type(
			"Output",
			(),
			{"last_hidden_state": input_values.unsqueeze(1)},
		)()


def test_baseline_grouped_encoder_restores_original_order() -> None:
	model = _FakeEmbeddingModel()
	text_group = {
		"input_values": torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
		"attention_mask": torch.ones(2, 1, dtype=torch.long),
	}
	vision_group = {
		"input_values": torch.tensor([[2.0, 2.0], [-3.0, 3.0]]),
		"attention_mask": torch.ones(2, 1, dtype=torch.long),
	}

	embeddings = encode_grouped_baseline_batches(
		model=model,
		processed_batches=(text_group, vision_group),
		original_indices=((0, 3), (1, 2)),
		total_rows=4,
	)

	diagonal = 2**-0.5
	assert torch.allclose(
		embeddings,
		torch.tensor(
			[
				[1.0, 0.0],
				[diagonal, diagonal],
				[-diagonal, diagonal],
				[0.0, 1.0],
			],
		),
	)


class _FakeFrozenEmbeddingModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(1))
		self.config = type("Config", (), {"use_cache": True})()


class _FakeEmbeddingModule:
	class Qwen3VLForEmbedding:
		loaded_arguments: dict[str, object] = {}

		@classmethod
		def from_pretrained(cls, model_root: str, **kwargs: object) -> nn.Module:
			cls.loaded_arguments = {"model_root": model_root, **kwargs}
			return _FakeFrozenEmbeddingModel()


def test_frozen_evaluation_model_has_no_trainable_parameters(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setattr(
		"looped_vl.baseline.model.load_local_embedding_module",
		lambda _model_root: _FakeEmbeddingModule,
	)

	model = load_frozen_evaluation_model(
		str(tmp_path),
		dtype=torch.float16,
		attention_implementation="sdpa",
	)

	assert model.training is False
	assert not any(parameter.requires_grad for parameter in model.parameters())
	assert model.config.use_cache is False  # type: ignore[attr-defined]
	assert _FakeEmbeddingModule.Qwen3VLForEmbedding.loaded_arguments == {
		"model_root": str(tmp_path),
		"trust_remote_code": True,
		"dtype": torch.float16,
		"attn_implementation": "sdpa",
	}


def test_baseline_grouped_forward_restores_query_and_candidate_order(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	training_model = BaselineLoRATrainingModel(_FakeEmbeddingModel())  # type: ignore[arg-type]
	captured: dict[str, torch.Tensor] = {}

	def fake_loss(
		query_embeddings: torch.Tensor,
		candidate_embeddings: torch.Tensor,
		positive_ids: list[str],
		temperature: float,
	) -> torch.Tensor:
		captured["query"] = query_embeddings
		captured["candidate"] = candidate_embeddings
		return query_embeddings.sum() * 0 + candidate_embeddings.sum() * 0

	monkeypatch.setattr("looped_vl.baseline.model.multi_positive_symmetric_info_nce", fake_loss)
	text_group = {
		"input_values": torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
		"attention_mask": torch.ones(2, 1, dtype=torch.long),
	}
	vision_group = {
		"input_values": torch.tensor([[2.0, 2.0], [-3.0, 3.0]]),
		"attention_mask": torch.ones(2, 1, dtype=torch.long),
	}

	training_model(
		local_batch_size=2,
		processed_batches=(text_group, vision_group),
		original_indices=((0, 3), (1, 2)),
		positive_ids=["first", "second"],
	)

	diagonal = 2**-0.5
	assert torch.allclose(
		captured["query"],
		torch.tensor([[1.0, 0.0], [diagonal, diagonal]]),
	)
	assert torch.allclose(
		captured["candidate"],
		torch.tensor([[-diagonal, diagonal], [0.0, 1.0]]),
	)


class _FakeCandidateStores:
	def __init__(self) -> None:
		self.mined_queries: torch.Tensor | None = None

	def mine_hard_negatives(
		self,
		query_embeddings: torch.Tensor,
		_references: list[object],
		*,
		count: int,
		device: torch.device,
	) -> None:
		assert count == 32
		assert device == query_embeddings.device
		self.mined_queries = query_embeddings
		return None


def test_query_only_lora_encodes_no_candidate_inputs(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _FakeEmbeddingModel()
	stores = _FakeCandidateStores()
	training_model = QueryOnlyLoRATrainingModel(  # type: ignore[arg-type]
		model,
		stores,  # type: ignore[arg-type]
	)
	captured: dict[str, object] = {}

	def fake_loss(
		query_embeddings: tuple[torch.Tensor, ...],
		candidate_embeddings: torch.Tensor,
		positive_ids: list[str],
		directions: list[str],
		**_kwargs: object,
	) -> tuple[torch.Tensor]:
		captured.update(
			queries=query_embeddings,
			candidates=candidate_embeddings,
			positive_ids=positive_ids,
			directions=directions,
		)
		return (query_embeddings[0].sum() * 0.0,)

	monkeypatch.setattr("looped_vl.baseline.model.multi_query_symmetric_info_nce", fake_loss)
	query_values = torch.zeros(2, 2048)
	query_values[0, 0] = 1.0
	query_values[1, 1] = 2.0
	query_batch = {
		"input_values": query_values,
		"attention_mask": torch.ones(2, 1, dtype=torch.long),
	}
	candidates = torch.nn.functional.normalize(torch.randn(2, 2048), dim=1)

	result = training_model(
		local_batch_size=2,
		processed_batches=(query_batch,),
		original_indices=((0, 1),),
		candidate_embeddings=candidates,
		candidate_references=[object(), object()],  # type: ignore[list-item]
		positive_ids=["p0", "p1"],
		directions=["image_to_text", "image_to_text"],
	)

	assert model.forward_calls == 1
	assert stores.mined_queries is not None
	assert stores.mined_queries.requires_grad is False
	assert captured["candidates"] is candidates
	assert result["loss"].shape == ()
