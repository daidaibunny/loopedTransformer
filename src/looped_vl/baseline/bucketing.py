"""Baseline visual-length bucketing without changing contrastive batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from looped_vl.training.data import ModalityInputGroup
from looped_vl.visual_bucketing import (
	DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	DEFAULT_VISUAL_LENGTH_BUCKETS,
	QWEN3_VL_IMAGE_PATCH_SIZE,
	estimate_visual_token_count,
	split_visual_rows_into_buckets,
)

__all__ = [
	"DEFAULT_MIN_VISUAL_BUCKET_SIZE",
	"DEFAULT_VISUAL_LENGTH_BUCKETS",
	"QWEN3_VL_IMAGE_PATCH_SIZE",
	"estimate_visual_token_count",
	"group_baseline_model_inputs",
]


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
	vision_indices: list[int] = []
	vision_inputs: list[dict[str, Any]] = []
	for original_index, model_input in zip(resolved_indices, model_inputs, strict=True):
		if "image" in model_input:
			vision_indices.append(original_index)
			vision_inputs.append(model_input)
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
	if vision_inputs:
		groups.extend(
			ModalityInputGroup(
				name=bucket_name,
				original_indices=bucket_indices,
				model_inputs=bucket_inputs,
			)
			for bucket_name, bucket_indices, bucket_inputs in split_visual_rows_into_buckets(
				tuple(vision_inputs),
				min_pixels=min_pixels,
				max_pixels=max_pixels,
				max_visual_buckets=max_visual_buckets,
				min_visual_bucket_size=min_visual_bucket_size,
				original_indices=tuple(vision_indices),
			)
		)
	if sum(len(group.model_inputs) for group in groups) != len(model_inputs):
		raise RuntimeError("Baseline bucketing lost one or more model inputs")
	return tuple(groups)
