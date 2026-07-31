from types import SimpleNamespace

import pytest
import torch

from looped_vl.baseline.data import build_coco_retrieval_inputs
from looped_vl.models.warmup_heads import WarmupEmbeddingHead
from looped_vl.training.data import build_training_pair, group_model_inputs_by_modality


def _sample(**overrides: object) -> SimpleNamespace:
	values = {
		"source": "coco",
		"mixture_position": 0,
		"text": "a caption",
		"answer": "",
		"image": object(),
	}
	values.update(overrides)
	return SimpleNamespace(**values)


def test_coco_pair_directions_are_deterministic_and_alternate() -> None:
	text_to_image = build_training_pair(_sample(mixture_position=0))
	image_to_text = build_training_pair(_sample(mixture_position=1))

	assert text_to_image.direction == "text_to_image"
	assert text_to_image.query_input == {
		"text": "a caption",
		"instruction": "Retrieve the image that best matches the caption.",
	}
	assert text_to_image.candidate_input["image"] is not None
	assert image_to_text.direction == "image_to_text"
	assert image_to_text.query_input["image"] is not None
	assert image_to_text.candidate_input == {"text": "a caption"}
	assert not hasattr(text_to_image, "semantic_target")


def test_recurrent_and_baseline_coco_directions_use_the_same_builder() -> None:
	for position in range(4):
		image = object()
		recurrent = build_training_pair(
			_sample(mixture_position=position, image=image),
		)
		baseline = build_coco_retrieval_inputs(
			text="a caption",
			image=image,
			position=position,
		)

		assert recurrent.direction == baseline.direction
		assert recurrent.query_input == baseline.query_input
		assert recurrent.candidate_input == baseline.candidate_input


@pytest.mark.parametrize("source", ["gqa_balanced", "clevr"])
def test_reasoning_pair_uses_image_question_and_answer(source: str) -> None:
	pair = build_training_pair(
		_sample(source=source, text="what color?", answer="red", mixture_position=10),
	)

	assert pair.direction == "visual_question_answering"
	assert pair.query_input["text"] == "what color?"
	assert pair.query_input["image"] is not None
	assert pair.candidate_input == {"text": "red"}
	assert not hasattr(pair, "semantic_target")


@pytest.mark.parametrize("slot_count", [1, 2, 4, 8, 16])
def test_warmup_embedding_head_outputs_unit_normalized_vectors(slot_count: int) -> None:
	head = WarmupEmbeddingHead(hidden_size=32)
	slots = torch.randn(3, slot_count, 32)

	embeddings = head(slots)

	assert embeddings.shape == (3, 32)
	assert torch.allclose(embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_warmup_embedding_head_pools_every_latent_slot() -> None:
	head = WarmupEmbeddingHead(hidden_size=2)
	with torch.no_grad():
		head.projection.weight.copy_(torch.eye(2))
		head.projection.bias.zero_()
	slots = torch.tensor(
		[
			[
				[2.0, 0.0],
				[2.0, 0.0],
				[0.0, 0.0],
				[0.0, 0.0],
			],
		],
	)

	embeddings = head(slots)

	assert torch.allclose(embeddings, torch.tensor([[1.0, 0.0]]))


def test_model_inputs_are_grouped_by_modality_and_keep_original_indices() -> None:
	image_a = object()
	image_b = object()
	query_inputs = [
		{"text": "caption"},
		{"text": "question", "image": image_a},
	]
	candidate_inputs = [
		{"image": image_b},
		{"text": "answer"},
	]

	groups = group_model_inputs_by_modality(query_inputs, candidate_inputs)

	assert [group.name for group in groups] == ["text", "vision"]
	assert groups[0].original_indices == (0, 3)
	assert groups[0].model_inputs == (query_inputs[0], candidate_inputs[1])
	assert groups[1].original_indices == (1, 2)
	assert groups[1].model_inputs == (query_inputs[1], candidate_inputs[0])
