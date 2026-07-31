"""Deterministic paired retrieval inputs for COCO, GQA Balanced, and CLEVR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from looped_vl.baseline.data import VQA_INSTRUCTION, build_coco_retrieval_inputs
from looped_vl.data import MixtureSample
from looped_vl.visual_bucketing import (
	DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	split_visual_rows_into_buckets,
)


@dataclass(frozen=True)
class TrainingPair:
	"""One shared-encoder query/candidate pair."""

	source: str
	direction: str
	query_input: dict[str, Any]
	candidate_input: dict[str, Any]
	positive_id: str
	reasoning_depth: int
	sample_id: str
	image: Image.Image


@dataclass(frozen=True)
class ModalityInputGroup:
	"""Inputs sharing a padding profile plus their positions in the combined towers."""

	name: str
	original_indices: tuple[int, ...]
	model_inputs: tuple[dict[str, Any], ...]


def build_training_pair(sample: MixtureSample | Any) -> TrainingPair:
	"""Map one mixture row to the exact v1.0 dual-tower training structure."""
	if sample.source == "coco":
		coco_inputs = build_coco_retrieval_inputs(
			text=sample.text,
			image=sample.image,
			position=sample.mixture_position,
		)
		direction = coco_inputs.direction
		query_input = coco_inputs.query_input
		candidate_input = coco_inputs.candidate_input
	elif sample.source in {"gqa_balanced", "clevr"}:
		direction = "visual_question_answering"
		query_input = {
			"text": sample.text,
			"image": sample.image,
			"instruction": VQA_INSTRUCTION,
		}
		candidate_input = {"text": sample.answer}
	else:
		raise ValueError(f"Unsupported training source: {sample.source}")
	return TrainingPair(
		source=sample.source,
		direction=direction,
		query_input=query_input,
		candidate_input=candidate_input,
		positive_id=str(getattr(sample, "positive_id", "")),
		reasoning_depth=int(getattr(sample, "reasoning_depth", 0)),
		sample_id=str(getattr(sample, "sample_id", sample.mixture_position)),
		image=sample.image,
	)


def paired_training_collate(samples: list[MixtureSample]) -> dict[str, Any]:
	"""Keep PIL images open until both towers have been processed in the trainer."""
	pairs = [build_training_pair(sample) for sample in samples]
	if any(not pair.positive_id for pair in pairs):
		raise ValueError("Every recurrent training pair must have a positive_id")
	return {
		"pairs": pairs,
		"query_inputs": [pair.query_input for pair in pairs],
		"candidate_inputs": [pair.candidate_input for pair in pairs],
		"positive_ids": [pair.positive_id for pair in pairs],
		"sources": [pair.source for pair in pairs],
		"directions": [pair.direction for pair in pairs],
		"sample_ids": [pair.sample_id for pair in pairs],
		"reasoning_depths": [pair.reasoning_depth for pair in pairs],
	}


def group_model_inputs_by_modality(
	query_inputs: list[dict[str, Any]],
	candidate_inputs: list[dict[str, Any]],
	*,
	min_pixels: int | None = None,
	max_pixels: int | None = None,
	max_visual_buckets: int = 1,
	min_visual_bucket_size: int = DEFAULT_MIN_VISUAL_BUCKET_SIZE,
) -> tuple[ModalityInputGroup, ...]:
	"""Separate pure text from visual rows without changing their logical batch order.

	Passing `max_visual_buckets` above one additionally splits the visual encoding call
	into visual-length buckets. Bucketing only changes how rows are padded and encoded;
	every logical position is preserved, so the contrastive batch composition, negative
	pool, and loss are unchanged.
	"""
	if len(query_inputs) != len(candidate_inputs):
		raise ValueError("Query and candidate input counts must match")
	if max_visual_buckets <= 0:
		raise ValueError("max_visual_buckets must be positive")
	if min_visual_bucket_size <= 0:
		raise ValueError("min_visual_bucket_size must be positive")
	combined_inputs = query_inputs + candidate_inputs
	grouped: list[ModalityInputGroup] = []
	for name, has_vision in (("text", False), ("vision", True)):
		indexed_inputs = tuple(
			(index, model_input)
			for index, model_input in enumerate(combined_inputs)
			if ("image" in model_input or "video" in model_input) is has_vision
		)
		if not indexed_inputs:
			continue
		if has_vision and max_visual_buckets > 1:
			if min_pixels is None or max_pixels is None:
				raise ValueError(
					"Visual-length bucketing requires min_pixels and max_pixels",
				)
			grouped.extend(
				ModalityInputGroup(
					name=bucket_name,
					original_indices=bucket_indices,
					model_inputs=bucket_inputs,
				)
				for bucket_name, bucket_indices, bucket_inputs in (
					split_visual_rows_into_buckets(
						tuple(model_input for _index, model_input in indexed_inputs),
						min_pixels=min_pixels,
						max_pixels=max_pixels,
						max_visual_buckets=max_visual_buckets,
						min_visual_bucket_size=min_visual_bucket_size,
						original_indices=tuple(index for index, _ in indexed_inputs),
					)
				)
			)
			continue
		grouped.append(
			ModalityInputGroup(
				name=name,
				original_indices=tuple(index for index, _ in indexed_inputs),
				model_inputs=tuple(model_input for _, model_input in indexed_inputs),
			),
		)
	if sum(len(group.model_inputs) for group in grouped) != len(combined_inputs):
		raise RuntimeError("Modality grouping lost one or more model inputs")
	return tuple(grouped)


def close_training_batch_images(batch: dict[str, Any]) -> None:
	"""Close every decoded image exactly once after processor conversion."""
	seen: set[int] = set()
	for pair in batch["pairs"]:
		image_identity = id(pair.image)
		if image_identity not in seen:
			pair.image.close()
			seen.add(image_identity)
