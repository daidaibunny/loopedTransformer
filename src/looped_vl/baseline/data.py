"""Independent COCO, GQA Balanced, and CLEVR retrieval manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from looped_vl.data import ParquetShardIndex, _read_row_group

COCO_TEXT_TO_IMAGE_INSTRUCTION = "Retrieve the image that best matches the caption."
COCO_IMAGE_TO_TEXT_INSTRUCTION = "Retrieve the caption that best describes the image."
VQA_INSTRUCTION = "Retrieve the correct answer to the visual question."
BASELINE_DATASETS = ("coco", "gqa_balanced", "clevr")
BASELINE_SPLITS = ("train", "validation", "test")


def normalize_answer(answer: str) -> str:
	"""Normalize an answer key without changing the candidate text shown to the model."""
	return " ".join(answer.strip().lower().split())


@dataclass(frozen=True)
class BaselinePairSample:
	"""One query/candidate training pair from exactly one source dataset."""

	sample_id: str
	dataset: str
	image_id: str
	query_input: dict[str, Any]
	candidate_input: dict[str, Any]
	positive_id: str
	candidate_text: str
	image: Image.Image


class BaselineManifestDataset(Dataset[BaselinePairSample]):
	"""Read one normalized single-dataset split without mixing data sources."""

	def __init__(
		self,
		dataset_root: str | Path,
		split: str,
		*,
		max_rows: int = 0,
	) -> None:
		if split not in BASELINE_SPLITS:
			raise ValueError(f"Unsupported baseline split: {split}")
		if max_rows < 0:
			raise ValueError("max_rows cannot be negative")
		self.dataset_root = Path(dataset_root)
		self.split = split
		self.index = ParquetShardIndex(self.dataset_root / split)
		self.length = min(len(self.index), max_rows) if max_rows else len(self.index)

	def __len__(self) -> int:
		return self.length

	def get_record(self, index: int) -> dict[str, Any]:
		"""Read one manifest record without decoding its image."""
		if index < 0:
			index += self.length
		if index < 0 or index >= self.length:
			raise IndexError(index)
		location = self.index.locate(index)
		table = _read_row_group(str(location.path), location.row_group)
		return table.slice(location.offset_in_row_group, 1).to_pylist()[0]

	def __getitem__(self, index: int) -> BaselinePairSample:
		record = self.get_record(index)
		dataset = str(record["dataset"])
		if dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported baseline dataset: {dataset}")
		image_path = Path(record["image_path"])
		if not image_path.is_file():
			raise FileNotFoundError(f"Missing baseline image: {image_path}")
		with Image.open(image_path) as source_image:
			image = source_image.convert("RGB")
			image.load()
		if dataset == "coco":
			query_input = {
				"text": str(record["query_text"]),
				"instruction": COCO_TEXT_TO_IMAGE_INSTRUCTION,
			}
			candidate_input = {"image": image}
		else:
			query_input = {
				"text": str(record["query_text"]),
				"image": image,
				"instruction": VQA_INSTRUCTION,
			}
			candidate_input = {"text": str(record["candidate_text"])}
		return BaselinePairSample(
			sample_id=str(record["sample_id"]),
			dataset=dataset,
			image_id=str(record["image_id"]),
			query_input=query_input,
			candidate_input=candidate_input,
			positive_id=str(record["positive_id"]),
			candidate_text=str(record["candidate_text"]),
			image=image,
		)


def baseline_pair_collate(samples: list[BaselinePairSample]) -> dict[str, Any]:
	"""Keep decoded images alive until the model processor has consumed the batch."""
	if not samples:
		raise ValueError("Cannot collate an empty baseline batch")
	return {
		"samples": samples,
		"sample_ids": [sample.sample_id for sample in samples],
		"query_inputs": [sample.query_input for sample in samples],
		"candidate_inputs": [sample.candidate_input for sample in samples],
		"positive_ids": [sample.positive_id for sample in samples],
	}


def close_baseline_batch_images(batch: dict[str, Any]) -> None:
	"""Close each image once after processor conversion."""
	seen: set[int] = set()
	for sample in batch["samples"]:
		image_identity = id(sample.image)
		if image_identity not in seen:
			sample.image.close()
			seen.add(image_identity)
