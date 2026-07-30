"""Recurrent training data backed directly by the frozen baseline split manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from looped_vl.baseline.data import BASELINE_DATASETS, BASELINE_SPLITS
from looped_vl.data import MixtureSample, ParquetShardIndex, _read_row_group

BASELINE_SPLIT_POLICIES = {
	"coco": "karpathy_113287_5000_5000_image_disjoint",
	"gqa_balanced": "official_train_val_testdev_balanced",
	"clevr": "official_train_plus_seed42_image_disjoint_halves_of_official_validation",
}
BASELINE_MANIFEST_COLUMNS = (
	"sample_id",
	"dataset",
	"split",
	"image_id",
	"image_path",
	"query_text",
	"candidate_text",
	"positive_id",
)


def _read_json(path: Path) -> dict[str, Any]:
	if not path.is_file():
		raise ValueError(f"Missing baseline-aligned metadata: {path}")
	value = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(value, dict):
		raise ValueError(f"Baseline-aligned metadata must be a mapping: {path}")
	return value


def _load_split_contract(
	dataset_root: Path,
	split: str,
) -> tuple[str, int]:
	"""Validate that a recurrent run points at one frozen baseline dataset root."""
	if split not in BASELINE_SPLITS:
		raise ValueError(f"Unsupported baseline-aligned split: {split}")
	if not (dataset_root / ".ready").is_file():
		raise ValueError(f"Baseline-aligned dataset is not ready: {dataset_root}")
	config = _read_json(dataset_root / "config.json")
	dataset = config.get("dataset")
	if dataset not in BASELINE_DATASETS:
		raise ValueError(
			f"Recurrent data must use a baseline-aligned dataset root, found {dataset!r}",
		)
	expected_policy = BASELINE_SPLIT_POLICIES[dataset]
	if config.get("split_policy") != expected_policy:
		raise ValueError(
			f"Baseline-aligned split policy mismatch for {dataset}: "
			f"{config.get('split_policy')!r} != {expected_policy!r}",
		)
	split_counts = config.get("split_counts")
	if not isinstance(split_counts, dict) or split not in split_counts:
		raise ValueError(f"Baseline-aligned config is missing split count for {split}")
	expected_rows = int(split_counts[split])
	stats = _read_json(dataset_root / "stats.json")
	split_stats = stats.get(split)
	if not isinstance(split_stats, dict) or int(split_stats.get("rows", -1)) != expected_rows:
		raise ValueError(
			f"Baseline-aligned split count in stats disagrees with config for "
			f"{dataset} {split}",
		)
	return str(dataset), expected_rows


def _normalize_record(
	record: dict[str, Any],
	*,
	position: int,
	expected_dataset: str,
	expected_split: str,
) -> dict[str, Any]:
	missing = set(BASELINE_MANIFEST_COLUMNS) - record.keys()
	if missing:
		raise ValueError(f"Baseline manifest row is missing columns: {sorted(missing)}")
	dataset = str(record["dataset"])
	split = str(record["split"])
	if dataset != expected_dataset or split != expected_split:
		raise ValueError(
			"Baseline manifest row drifted from its dataset contract: "
			f"{dataset}/{split} != {expected_dataset}/{expected_split}",
		)
	return {
		"mixture_position": position,
		"sample_id": str(record["sample_id"]),
		"source": dataset,
		"source_split": split,
		"task_type": (
			"image_text_matching"
			if dataset == "coco"
			else "visual_question_answering"
		),
		"image_storage": "filesystem",
		"image_path": str(record["image_path"]),
		"image_id": str(record["image_id"]),
		"text": str(record["query_text"]),
		"answer": str(record["candidate_text"]),
		"full_answer": str(record["candidate_text"]),
		"reasoning_trace_json": "[]",
		"reasoning_depth": 0,
		"metadata_json": json.dumps(
			{"positive_id": str(record["positive_id"])},
			sort_keys=True,
		),
	}


class RecurrentAlignedDataset(Dataset[MixtureSample]):
	"""Expose the exact baseline rows through the recurrent training sample contract."""

	def __init__(self, dataset_root: str | Path, split: str) -> None:
		self.dataset_root = Path(dataset_root)
		self.split = split
		self.dataset, expected_rows = _load_split_contract(self.dataset_root, split)
		self.index = ParquetShardIndex(self.dataset_root / split)
		if len(self.index) != expected_rows:
			raise ValueError(
				f"Baseline-aligned split count drift for {self.dataset} {split}: "
				f"{len(self.index)} != {expected_rows}",
			)

	def __len__(self) -> int:
		return len(self.index)

	def get_record(self, index: int) -> dict[str, Any]:
		"""Read and normalize one baseline manifest row without decoding its image."""
		location = self.index.locate(index)
		table = _read_row_group(str(location.path), location.row_group)
		record = table.slice(location.offset_in_row_group, 1).to_pylist()[0]
		return _normalize_record(
			record,
			position=index,
			expected_dataset=self.dataset,
			expected_split=self.split,
		)

	@staticmethod
	def resolve_image_path(record: dict[str, Any]) -> Path:
		"""Resolve the materialized image path embedded in the baseline manifest."""
		path = Path(record["image_path"])
		if not path.is_file():
			raise FileNotFoundError(f"Missing baseline-aligned image: {path}")
		return path

	def __getitem__(self, index: int) -> MixtureSample:
		record = self.get_record(index)
		image_path = self.resolve_image_path(record)
		with Image.open(image_path) as source_image:
			image = source_image.convert("RGB")
			image.load()
		return MixtureSample(
			mixture_position=int(record["mixture_position"]),
			sample_id=str(record["sample_id"]),
			source=str(record["source"]),
			source_split=str(record["source_split"]),
			task_type=str(record["task_type"]),
			image_id=str(record["image_id"]),
			image_path=image_path,
			image=image,
			text=str(record["text"]),
			answer=str(record["answer"]),
			full_answer=str(record["full_answer"]),
			reasoning_trace_json=str(record["reasoning_trace_json"]),
			reasoning_depth=int(record["reasoning_depth"]),
			metadata_json=str(record["metadata_json"]),
		)


def load_aligned_records(
	dataset_root: str | Path,
	split: str,
	max_rows: int = 0,
) -> list[dict[str, Any]]:
	"""Load normalized evaluation rows from the exact baseline Parquet split."""
	if max_rows < 0:
		raise ValueError("max_rows cannot be negative")
	dataset = RecurrentAlignedDataset(dataset_root, split)
	records: list[dict[str, Any]] = []
	for path in sorted((dataset.dataset_root / split).glob("*.parquet")):
		for batch in pq.ParquetFile(path).iter_batches(
			columns=list(BASELINE_MANIFEST_COLUMNS),
			batch_size=8192,
		):
			for record in batch.to_pylist():
				records.append(
					_normalize_record(
						record,
						position=len(records),
						expected_dataset=dataset.dataset,
						expected_split=split,
					),
				)
				if max_rows and len(records) >= max_rows:
					return records
	if len(records) != len(dataset):
		raise ValueError(
			f"Baseline-aligned row loading drift: {len(records)} != {len(dataset)}",
		)
	return records
