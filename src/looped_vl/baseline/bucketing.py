"""Baseline-only visual-length bucketing without changing contrastive batches."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

from PIL import Image
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE, smart_resize

from looped_vl.training.data import ModalityInputGroup

QWEN3_VL_IMAGE_PATCH_SIZE = 16
DEFAULT_VISUAL_LENGTH_BUCKETS = 3
DEFAULT_MIN_VISUAL_BUCKET_SIZE = 8


def estimate_visual_token_count(
	model_input: dict[str, Any],
	*,
	min_pixels: int,
	max_pixels: int,
) -> int:
	"""Return the exact post-smart-resize visual token count used for padding."""
	image = model_input.get("image")
	if not isinstance(image, Image.Image):
		raise TypeError("Baseline visual bucketing requires an already decoded PIL image")
	width, height = image.size
	factor = QWEN3_VL_IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE
	resized_height, resized_width = smart_resize(
		height,
		width,
		factor=factor,
		min_pixels=min_pixels,
		max_pixels=max_pixels,
	)
	return (resized_height // factor) * (resized_width // factor)


def _balanced_boundaries(
	lengths: tuple[int, ...],
	*,
	max_buckets: int,
	min_bucket_size: int,
) -> tuple[int, ...]:
	"""Choose deterministic near-quantile boundaries without splitting equal lengths."""
	row_count = len(lengths)
	maximum_bucket_count = min(max_buckets, row_count // min_bucket_size)
	change_positions = tuple(
		position
		for position in range(1, row_count)
		if lengths[position - 1] != lengths[position]
	)
	for bucket_count in range(maximum_bucket_count, 1, -1):
		ideal_boundaries = tuple(
			round(row_count * bucket_index / bucket_count)
			for bucket_index in range(1, bucket_count)
		)
		valid_options: list[tuple[int, tuple[int, ...]]] = []
		for boundaries in combinations(change_positions, bucket_count - 1):
			segments = (0, *boundaries, row_count)
			if any(
				segments[index + 1] - segments[index] < min_bucket_size
				for index in range(bucket_count)
			):
				continue
			cost = sum(
				abs(boundary - ideal)
				for boundary, ideal in zip(boundaries, ideal_boundaries, strict=True)
			)
			valid_options.append((cost, boundaries))
		if valid_options:
			return min(valid_options)[1]
	return ()


def group_baseline_model_inputs(
	model_inputs: Sequence[dict[str, Any]],
	*,
	min_pixels: int,
	max_pixels: int,
	max_visual_buckets: int,
	min_visual_bucket_size: int,
	original_indices: Sequence[int] | None = None,
) -> tuple[ModalityInputGroup, ...]:
	"""Split only visual encoding calls and retain every original logical position."""
	if max_visual_buckets <= 0:
		raise ValueError("max_visual_buckets must be positive")
	if min_visual_bucket_size <= 0:
		raise ValueError("min_visual_bucket_size must be positive")
	resolved_indices = (
		tuple(range(len(model_inputs)))
		if original_indices is None
		else tuple(original_indices)
	)
	if len(resolved_indices) != len(model_inputs):
		raise ValueError("Original index and model input counts must match")
	if len(set(resolved_indices)) != len(resolved_indices):
		raise ValueError("Original indices must be unique")
	text_rows: list[tuple[int, dict[str, Any]]] = []
	vision_rows: list[tuple[int, int, dict[str, Any]]] = []
	for original_index, model_input in zip(
		resolved_indices,
		model_inputs,
		strict=True,
	):
		if "image" in model_input:
			vision_rows.append(
				(
					estimate_visual_token_count(
						model_input,
						min_pixels=min_pixels,
						max_pixels=max_pixels,
					),
					original_index,
					model_input,
				),
			)
		elif "video" in model_input:
			raise ValueError("Baseline visual bucketing does not support videos")
		else:
			text_rows.append((original_index, model_input))
	groups: list[ModalityInputGroup] = []
	if text_rows:
		groups.append(
			ModalityInputGroup(
				name="text",
				original_indices=tuple(index for index, _model_input in text_rows),
				model_inputs=tuple(model_input for _index, model_input in text_rows),
			),
		)
	if vision_rows:
		vision_rows.sort(key=lambda row: (row[0], row[1]))
		lengths = tuple(row[0] for row in vision_rows)
		boundaries = _balanced_boundaries(
			lengths,
			max_buckets=max_visual_buckets,
			min_bucket_size=min_visual_bucket_size,
		)
		start_positions = (0, *boundaries)
		end_positions = (*boundaries, len(vision_rows))
		for start, end in zip(start_positions, end_positions, strict=True):
			bucket = vision_rows[start:end]
			name = (
				"vision"
				if not boundaries
				else f"vision_tokens_{bucket[0][0]}_{bucket[-1][0]}"
			)
			groups.append(
				ModalityInputGroup(
					name=name,
					original_indices=tuple(row[1] for row in bucket),
					model_inputs=tuple(row[2] for row in bucket),
				),
			)
	if sum(len(group.model_inputs) for group in groups) != len(model_inputs):
		raise RuntimeError("Baseline bucketing lost one or more model inputs")
	return tuple(groups)
