"""Shared visual-length bucketing used by both the baseline and recurrent trainers.

Bucketing changes only how rows are padded and encoded. Callers must preserve every
original logical position so that contrastive batch composition, the in-batch negative
pool, and the resulting loss stay identical to unbucketed encoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

from PIL import Image
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE, smart_resize

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
		raise TypeError("Visual bucketing requires an already decoded PIL image")
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


def balanced_boundaries(
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


def split_visual_rows_into_buckets(
	model_inputs: Sequence[dict[str, Any]],
	*,
	min_pixels: int,
	max_pixels: int,
	max_visual_buckets: int,
	min_visual_bucket_size: int,
	original_indices: Sequence[int],
) -> tuple[tuple[str, tuple[int, ...], tuple[dict[str, Any], ...]], ...]:
	"""Return `(name, original_indices, model_inputs)` per visual-length bucket."""
	if max_visual_buckets <= 0:
		raise ValueError("max_visual_buckets must be positive")
	if min_visual_bucket_size <= 0:
		raise ValueError("min_visual_bucket_size must be positive")
	if len(original_indices) != len(model_inputs):
		raise ValueError("Original index and model input counts must match")
	if len(set(original_indices)) != len(original_indices):
		raise ValueError("Original indices must be unique")
	rows: list[tuple[int, int, dict[str, Any]]] = []
	for original_index, model_input in zip(original_indices, model_inputs, strict=True):
		if "video" in model_input:
			raise ValueError("Visual bucketing does not support videos")
		rows.append(
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
	rows.sort(key=lambda row: (row[0], row[1]))
	lengths = tuple(row[0] for row in rows)
	boundaries = balanced_boundaries(
		lengths,
		max_buckets=max_visual_buckets,
		min_bucket_size=min_visual_bucket_size,
	)
	start_positions = (0, *boundaries)
	end_positions = (*boundaries, len(rows))
	buckets: list[tuple[str, tuple[int, ...], tuple[dict[str, Any], ...]]] = []
	for start, end in zip(start_positions, end_positions, strict=True):
		bucket = rows[start:end]
		name = (
			"vision"
			if not boundaries
			else f"vision_tokens_{bucket[0][0]}_{bucket[-1][0]}"
		)
		buckets.append(
			(
				name,
				tuple(row[1] for row in bucket),
				tuple(row[2] for row in bucket),
			),
		)
	if sum(len(bucket[2]) for bucket in buckets) != len(model_inputs):
		raise RuntimeError("Visual bucketing lost one or more model inputs")
	return tuple(buckets)
