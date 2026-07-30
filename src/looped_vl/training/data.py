"""Deterministic paired retrieval inputs for COCO, GQA Balanced, and CLEVR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from looped_vl.data import MixtureSample

COCO_TEXT_TO_IMAGE_INSTRUCTION = "Retrieve the image that best matches the caption."
COCO_IMAGE_TO_TEXT_INSTRUCTION = "Retrieve the caption that best describes the image."
VQA_INSTRUCTION = "Retrieve the correct answer to the visual question."


@dataclass(frozen=True)
class TrainingPair:
	"""One shared-encoder query/candidate pair plus its semantic target."""

	source: str
	direction: str
	query_input: dict[str, Any]
	candidate_input: dict[str, Any]
	semantic_target: str
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
		if sample.mixture_position % 2 == 0:
			direction = "text_to_image"
			query_input = {
				"text": sample.text,
				"instruction": COCO_TEXT_TO_IMAGE_INSTRUCTION,
			}
			candidate_input = {"image": sample.image}
		else:
			direction = "image_to_text"
			query_input = {
				"image": sample.image,
				"instruction": COCO_IMAGE_TO_TEXT_INSTRUCTION,
			}
			candidate_input = {"text": sample.text}
		semantic_target = sample.text
	elif sample.source in {"gqa_balanced", "clevr"}:
		direction = "visual_question_answering"
		query_input = {
			"text": sample.text,
			"image": sample.image,
			"instruction": VQA_INSTRUCTION,
		}
		candidate_input = {"text": sample.answer}
		semantic_target = sample.answer
	else:
		raise ValueError(f"Unsupported training source: {sample.source}")
	if not semantic_target.strip():
		raise ValueError(f"Empty semantic target for {sample.source}")
	return TrainingPair(
		source=sample.source,
		direction=direction,
		query_input=query_input,
		candidate_input=candidate_input,
		semantic_target=semantic_target,
		reasoning_depth=int(getattr(sample, "reasoning_depth", 0)),
		sample_id=str(getattr(sample, "sample_id", sample.mixture_position)),
		image=sample.image,
	)


def paired_training_collate(samples: list[MixtureSample]) -> dict[str, Any]:
	"""Keep PIL images open until both towers have been processed in the trainer."""
	pairs = [build_training_pair(sample) for sample in samples]
	return {
		"pairs": pairs,
		"query_inputs": [pair.query_input for pair in pairs],
		"candidate_inputs": [pair.candidate_input for pair in pairs],
		"semantic_targets": [pair.semantic_target for pair in pairs],
		"sources": [pair.source for pair in pairs],
		"directions": [pair.direction for pair in pairs],
		"sample_ids": [pair.sample_id for pair in pairs],
		"reasoning_depths": [pair.reasoning_depth for pair in pairs],
	}


def group_model_inputs_by_modality(
	query_inputs: list[dict[str, Any]],
	candidate_inputs: list[dict[str, Any]],
) -> tuple[ModalityInputGroup, ...]:
	"""Separate pure text from visual rows without changing their logical batch order."""
	if len(query_inputs) != len(candidate_inputs):
		raise ValueError("Query and candidate input counts must match")
	combined_inputs = query_inputs + candidate_inputs
	grouped: list[ModalityInputGroup] = []
	for name, has_vision in (("text", False), ("vision", True)):
		indexed_inputs = tuple(
			(index, model_input)
			for index, model_input in enumerate(combined_inputs)
			if ("image" in model_input or "video" in model_input) is has_vision
		)
		if indexed_inputs:
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
