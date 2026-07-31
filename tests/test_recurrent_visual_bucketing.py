"""Recurrent visual-length bucketing must not change contrastive batch composition."""

from __future__ import annotations

import pytest
from PIL import Image

from looped_vl.training.data import group_model_inputs_by_modality


def _vision_input(size: tuple[int, int]) -> dict[str, object]:
	return {"image": Image.new("RGB", size)}


def _text_input(index: int) -> dict[str, object]:
	return {"text": f"answer-{index}"}


def test_recurrent_bucketing_splits_only_vision_encoding_and_keeps_every_row() -> None:
	query_inputs = (
		[_vision_input((64, 64)) for _ in range(8)]
		+ [_vision_input((320, 320)) for _ in range(8)]
		+ [_vision_input((640, 640)) for _ in range(8)]
	)
	candidate_inputs = [_text_input(index) for index in range(24)]

	groups = group_model_inputs_by_modality(
		query_inputs,
		candidate_inputs,
		min_pixels=4 * 32 * 32,
		max_pixels=1800 * 32 * 32,
		max_visual_buckets=3,
		min_visual_bucket_size=8,
	)

	assert groups[0].name == "text"
	assert groups[0].original_indices == tuple(range(24, 48))
	vision_groups = groups[1:]
	assert len(vision_groups) == 3
	assert [len(group.model_inputs) for group in vision_groups] == [8, 8, 8]
	assert tuple(
		sorted(index for group in groups for index in group.original_indices)
	) == tuple(range(48))


def test_recurrent_bucketing_defaults_to_one_vision_group() -> None:
	query_inputs = [_vision_input((320, 320)) for _ in range(4)]
	candidate_inputs = [_text_input(index) for index in range(4)]

	groups = group_model_inputs_by_modality(query_inputs, candidate_inputs)

	assert [group.name for group in groups] == ["text", "vision"]
	assert groups[1].original_indices == (0, 1, 2, 3)


def test_recurrent_bucketing_never_splits_below_the_minimum_bucket_size() -> None:
	query_inputs = [
		_vision_input((64, 64)),
		_vision_input((320, 320)),
		_vision_input((640, 640)),
		_vision_input((640, 640)),
	]
	candidate_inputs = [_text_input(index) for index in range(4)]

	groups = group_model_inputs_by_modality(
		query_inputs,
		candidate_inputs,
		min_pixels=4 * 32 * 32,
		max_pixels=1800 * 32 * 32,
		max_visual_buckets=3,
		min_visual_bucket_size=8,
	)

	vision_groups = [group for group in groups if group.name != "text"]
	assert len(vision_groups) == 1
	assert vision_groups[0].original_indices == (0, 1, 2, 3)


def test_recurrent_bucketing_rejects_invalid_bucket_settings() -> None:
	query_inputs = [_vision_input((320, 320))]
	candidate_inputs = [_text_input(0)]

	with pytest.raises(ValueError, match="max_visual_buckets"):
		group_model_inputs_by_modality(
			query_inputs,
			candidate_inputs,
			max_visual_buckets=0,
		)
	with pytest.raises(ValueError, match="min_visual_bucket_size"):
		group_model_inputs_by_modality(
			query_inputs,
			candidate_inputs,
			min_visual_bucket_size=0,
		)
