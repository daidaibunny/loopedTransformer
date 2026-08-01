"""Query-only training samples paired with immutable candidate references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from looped_vl.baseline.data import (
	BASELINE_DATASETS,
	BASELINE_SPLITS,
	COCO_IMAGE_TO_TEXT_INSTRUCTION,
	COCO_TEXT_TO_IMAGE_INSTRUCTION,
	VQA_INSTRUCTION,
)
from looped_vl.candidate_bank import CandidateBankSpec
from looped_vl.data import ParquetShardIndex, _read_row_group
from looped_vl.query_recurrent.candidate_store import CandidateReference


@dataclass(frozen=True)
class QueryOnlySample:
	"""One raw query plus the identity of its pre-encoded positive candidate."""

	sample_id: str
	dataset: str
	direction: str
	query_input: dict[str, Any]
	candidate_reference: CandidateReference
	image: Image.Image | None


class QueryOnlyManifestDataset(Dataset[QueryOnlySample]):
	"""Read baseline manifests without decoding images used only by frozen candidates."""

	def __init__(
		self,
		dataset_root: str | Path,
		dataset: str,
		split: str,
		*,
		max_rows: int = 0,
	) -> None:
		if dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported query-only dataset: {dataset}")
		if split not in BASELINE_SPLITS:
			raise ValueError(f"Unsupported query-only split: {split}")
		if max_rows < 0:
			raise ValueError("max_rows cannot be negative")
		self.dataset_root = Path(dataset_root)
		self.dataset = dataset
		self.split = split
		self.index = ParquetShardIndex(self.dataset_root / split)
		self.length = min(len(self.index), max_rows) if max_rows else len(self.index)

	def __len__(self) -> int:
		return self.length

	def _record(self, index: int) -> dict[str, Any]:
		if index < 0:
			index += self.length
		if index < 0 or index >= self.length:
			raise IndexError(index)
		location = self.index.locate(index)
		table = _read_row_group(str(location.path), location.row_group)
		return table.slice(location.offset_in_row_group, 1).to_pylist()[0]

	@staticmethod
	def _load_image(path_value: object) -> Image.Image:
		path = Path(str(path_value))
		if not path.is_file():
			raise FileNotFoundError(f"Missing query image: {path}")
		with Image.open(path) as source_image:
			image = source_image.convert("RGB")
			image.load()
		return image

	def __getitem__(self, index: int) -> QueryOnlySample:
		record = self._record(index)
		row_dataset = str(record["dataset"])
		row_split = str(record["split"])
		if row_dataset != self.dataset or row_split != self.split:
			raise ValueError(
				f"Manifest row is {row_dataset}/{row_split}; expected {self.dataset}/{self.split}",
			)
		sample_id = str(record["sample_id"])
		positive_id = str(record["positive_id"])
		if self.dataset == "coco" and index % 2 == 0:
			return QueryOnlySample(
				sample_id=sample_id,
				dataset=self.dataset,
				direction="text_to_image",
				query_input={
					"text": str(record["query_text"]),
					"instruction": COCO_TEXT_TO_IMAGE_INSTRUCTION,
				},
				candidate_reference=CandidateReference(
					CandidateBankSpec("coco", self.split, "image"),
					item_id=positive_id,
					positive_id=positive_id,
				),
				image=None,
			)
		image = self._load_image(record["image_path"])
		if self.dataset == "coco":
			return QueryOnlySample(
				sample_id=sample_id,
				dataset=self.dataset,
				direction="image_to_text",
				query_input={
					"image": image,
					"instruction": COCO_IMAGE_TO_TEXT_INSTRUCTION,
				},
				candidate_reference=CandidateReference(
					CandidateBankSpec("coco", self.split, "text"),
					item_id=sample_id,
					positive_id=positive_id,
				),
				image=image,
			)
		return QueryOnlySample(
			sample_id=sample_id,
			dataset=self.dataset,
			direction="visual_question_answering",
			query_input={
				"text": str(record["query_text"]),
				"image": image,
				"instruction": VQA_INSTRUCTION,
			},
			candidate_reference=CandidateReference(
				CandidateBankSpec(self.dataset, "shared", "answer"),
				item_id=positive_id,
				positive_id=positive_id,
			),
			image=image,
		)


def query_only_collate(samples: list[QueryOnlySample]) -> dict[str, Any]:
	"""Preserve logical row order until candidate lookup and contrastive learning."""
	if not samples:
		raise ValueError("Cannot collate an empty query-only batch")
	return {
		"samples": samples,
		"sample_ids": [sample.sample_id for sample in samples],
		"directions": [sample.direction for sample in samples],
		"query_inputs": [sample.query_input for sample in samples],
		"candidate_references": [sample.candidate_reference for sample in samples],
		"positive_ids": [sample.candidate_reference.positive_id for sample in samples],
	}


def close_query_only_images(batch: dict[str, Any]) -> None:
	"""Close each decoded query image after official Qwen preprocessing."""
	for sample in batch["samples"]:
		if sample.image is not None:
			sample.image.close()
