from __future__ import annotations

from PIL import Image

from looped_vl.baseline.bucketing import (
	estimate_visual_token_count,
	group_baseline_model_inputs,
)


def _vision_input(size: tuple[int, int]) -> dict[str, object]:
	return {"image": Image.new("RGB", size)}


def test_visual_token_estimate_matches_qwen_smart_resize() -> None:
	small = _vision_input((32, 32))
	medium = _vision_input((320, 320))
	large = _vision_input((640, 640))

	assert estimate_visual_token_count(
		small,
		min_pixels=4 * 32 * 32,
		max_pixels=1800 * 32 * 32,
	) == 4
	assert estimate_visual_token_count(
		medium,
		min_pixels=4 * 32 * 32,
		max_pixels=1800 * 32 * 32,
	) == 100
	assert estimate_visual_token_count(
		large,
		min_pixels=4 * 32 * 32,
		max_pixels=1800 * 32 * 32,
	) == 400


def test_baseline_bucketing_splits_only_vision_encoding_and_preserves_indices() -> None:
	vision_inputs = (
		[_vision_input((64, 64)) for _ in range(8)]
		+ [_vision_input((320, 320)) for _ in range(8)]
		+ [_vision_input((640, 640)) for _ in range(8)]
	)
	model_inputs = vision_inputs + [{"text": f"answer-{index}"} for index in range(24)]

	groups = group_baseline_model_inputs(
		model_inputs,
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


def test_baseline_bucketing_avoids_extra_model_calls_for_equal_lengths() -> None:
	model_inputs = [_vision_input((320, 320)) for _ in range(32)]

	groups = group_baseline_model_inputs(
		model_inputs,
		min_pixels=4 * 32 * 32,
		max_pixels=1800 * 32 * 32,
		max_visual_buckets=3,
		min_visual_bucket_size=8,
	)

	assert len(groups) == 1
	assert groups[0].name == "vision"
	assert groups[0].original_indices == tuple(range(32))
