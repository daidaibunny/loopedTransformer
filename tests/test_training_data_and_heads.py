from types import SimpleNamespace

import pytest
import torch

from looped_vl.models.warmup_heads import WarmupEmbeddingHead, split_slot_groups
from looped_vl.training.data import build_training_pair


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
	assert text_to_image.semantic_target == "a caption"


@pytest.mark.parametrize("source", ["gqa_balanced", "clevr"])
def test_reasoning_pair_uses_image_question_and_answer(source: str) -> None:
	pair = build_training_pair(
		_sample(source=source, text="what color?", answer="red", mixture_position=10),
	)

	assert pair.direction == "visual_question_answering"
	assert pair.query_input["text"] == "what color?"
	assert pair.query_input["image"] is not None
	assert pair.candidate_input == {"text": "red"}
	assert pair.semantic_target == "red"


def test_slot_grouping_matches_every_required_k_value() -> None:
	for slot_count, expected_reasoning, expected_embedding in (
		(1, 1, 1),
		(2, 1, 1),
		(4, 2, 2),
		(8, 4, 4),
		(16, 8, 8),
	):
		slots = torch.randn(2, slot_count, 32)
		reasoning, embedding = split_slot_groups(slots)
		assert reasoning.shape[1] == expected_reasoning
		assert embedding.shape[1] == expected_embedding
		if slot_count == 1:
			assert reasoning.data_ptr() == embedding.data_ptr()


def test_warmup_embedding_head_outputs_unit_normalized_vectors() -> None:
	head = WarmupEmbeddingHead(hidden_size=32)
	slots = torch.randn(3, 4, 32)

	embeddings = head(slots)

	assert embeddings.shape == (3, 32)
	assert torch.allclose(embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)
