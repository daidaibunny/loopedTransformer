"""Parquet-backed dataset and image resolution for the Looped VL mixture."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

SOURCE_ORDER = ("coco", "gqa_balanced", "clevr")
DEFAULT_TRAIN_SAMPLES = 100_000
DEFAULT_VALIDATION_SAMPLES = 10_000
DEFAULT_TEST_SAMPLES = 10_000
DEFAULT_DATASET_ROOT = Path(
	f"/mnt/afs/liyiwei/datasets/looped_vl_mix_v1_train{DEFAULT_TRAIN_SAMPLES}"
	f"_val{DEFAULT_VALIDATION_SAMPLES}_test{DEFAULT_TEST_SAMPLES}",
)
SOURCE_INSTRUCTIONS = {
	"coco": "Represent the image and caption for multimodal retrieval.",
	"gqa_balanced": "Represent the image and question for visual reasoning.",
	"clevr": "Represent the image and question for visual reasoning.",
}


@dataclass(frozen=True)
class RowLocation:
	"""Physical location of one logical row in a Parquet dataset."""

	path: Path
	row_group: int
	offset_in_row_group: int


@dataclass(frozen=True)
class ParquetShard:
	"""Row boundaries for one Parquet shard."""

	path: Path
	start: int
	end: int
	row_group_ends: tuple[int, ...]


class ParquetShardIndex:
	"""Map global row indexes to Parquet row groups without loading the dataset."""

	def __init__(self, split_root: str | Path) -> None:
		self.split_root = Path(split_root)
		files = sorted(self.split_root.glob("*.parquet"))
		if not files:
			raise FileNotFoundError(f"No Parquet shards found under {self.split_root}")

		shards: list[ParquetShard] = []
		shard_ends: list[int] = []
		start = 0
		for path in files:
			metadata = pq.ParquetFile(path).metadata
			row_group_ends: list[int] = []
			rows_in_shard = 0
			for row_group in range(metadata.num_row_groups):
				rows_in_shard += metadata.row_group(row_group).num_rows
				row_group_ends.append(rows_in_shard)
			end = start + rows_in_shard
			shards.append(
				ParquetShard(
					path=path,
					start=start,
					end=end,
					row_group_ends=tuple(row_group_ends),
				)
			)
			shard_ends.append(end)
			start = end
		self._shards = tuple(shards)
		self._shard_ends = tuple(shard_ends)
		self._length = start

	def __len__(self) -> int:
		return self._length

	def locate(self, index: int) -> RowLocation:
		"""Return the Parquet row group and local offset for a global index."""
		normalized_index = index
		if normalized_index < 0:
			normalized_index += self._length
		if normalized_index < 0 or normalized_index >= self._length:
			raise IndexError(f"Dataset index {index} is outside [0, {self._length})")

		shard_index = bisect_right(self._shard_ends, normalized_index)
		shard = self._shards[shard_index]
		index_in_shard = normalized_index - shard.start
		row_group = bisect_right(shard.row_group_ends, index_in_shard)
		row_group_start = 0 if row_group == 0 else shard.row_group_ends[row_group - 1]
		return RowLocation(
			path=shard.path,
			row_group=row_group,
			offset_in_row_group=index_in_shard - row_group_start,
		)


@lru_cache(maxsize=16)
def _read_row_group(path: str, row_group: int) -> pa.Table:
	"""Read and cache a manifest row group inside each DataLoader process."""
	return pq.ParquetFile(path).read_row_group(row_group)


class GQAImageResolver:
	"""Resolve a GQA image identifier against the materialized JPEG cache."""

	def __init__(self, root: str | Path) -> None:
		self.root = Path(root)

	def resolve(self, source_split: str, image_id: str) -> Path:
		"""Return an existing materialized JPEG path or fail with the exact target."""
		normalized_split = "val" if source_split == "validation" else source_split
		path = self.root / normalized_split / f"{image_id}.jpg"
		if not path.is_file():
			raise FileNotFoundError(
				f"Missing materialized GQA image: {path}. Run looped_vl.materialize_gqa.",
			)
		return path


@dataclass(frozen=True)
class MixtureSample:
	"""One decoded mixture sample with its normalized training metadata."""

	mixture_position: int
	sample_id: str
	source: str
	source_split: str
	task_type: str
	image_id: str
	image_path: Path
	image: Image.Image
	text: str
	answer: str
	full_answer: str
	reasoning_trace_json: str
	reasoning_depth: int
	metadata_json: str


class LoopedVLMixtureDataset(Dataset[MixtureSample]):
	"""Map-style loader for the normalized Looped VL mixture manifests."""

	def __init__(
		self,
		dataset_root: str | Path,
		split: str,
		gqa_materialized_root: str | Path,
	) -> None:
		if split not in {"train", "validation", "test"}:
			raise ValueError(f"Unsupported split: {split}")
		self.dataset_root = Path(dataset_root)
		self.split = split
		self.index = ParquetShardIndex(self.dataset_root / split)
		self.gqa_resolver = GQAImageResolver(gqa_materialized_root)

	def __len__(self) -> int:
		return len(self.index)

	def get_record(self, index: int) -> dict[str, Any]:
		"""Read normalized metadata without decoding the referenced image."""
		location = self.index.locate(index)
		table = _read_row_group(str(location.path), location.row_group)
		return table.slice(location.offset_in_row_group, 1).to_pylist()[0]

	def resolve_image_path(self, record: dict[str, Any]) -> Path:
		"""Resolve either a filesystem image or a materialized GQA image."""
		storage = record["image_storage"]
		if storage == "filesystem":
			path = Path(record["image_path"])
			if not path.is_file():
				raise FileNotFoundError(f"Missing source image: {path}")
			return path
		if storage == "hf_parquet":
			return self.gqa_resolver.resolve(record["source_split"], record["image_id"])
		raise ValueError(f"Unsupported image storage backend: {storage}")

	def __getitem__(self, index: int) -> MixtureSample:
		record = self.get_record(index)
		image_path = self.resolve_image_path(record)
		with Image.open(image_path) as source_image:
			image = source_image.convert("RGB")
			image.load()
		return MixtureSample(
			mixture_position=record["mixture_position"],
			sample_id=record["sample_id"],
			source=record["source"],
			source_split=record["source_split"],
			task_type=record["task_type"],
			image_id=record["image_id"],
			image_path=image_path,
			image=image,
			text=record["text"],
			answer=record["answer"],
			full_answer=record["full_answer"],
			reasoning_trace_json=record["reasoning_trace_json"],
			reasoning_depth=record["reasoning_depth"],
			metadata_json=record["metadata_json"],
		)


def select_source_balanced_indices(
	dataset: LoopedVLMixtureDataset,
	per_source: int,
) -> list[int]:
	"""Select an equal small smoke subset without decoding images."""
	if per_source <= 0:
		raise ValueError("per_source must be positive")
	selected = {source: [] for source in SOURCE_ORDER}
	for index in range(len(dataset)):
		source = dataset.get_record(index)["source"]
		if source in selected and len(selected[source]) < per_source:
			selected[source].append(index)
		if all(len(indices) == per_source for indices in selected.values()):
			break
	missing = {
		source: per_source - len(indices)
		for source, indices in selected.items()
		if len(indices) != per_source
	}
	if missing:
		raise ValueError(f"Dataset cannot provide the requested source balance: {missing}")
	return [index for source in SOURCE_ORDER for index in selected[source]]


def mixture_collate(samples: list[MixtureSample]) -> dict[str, Any]:
	"""Preserve PIL images and produce inputs accepted by Qwen3VLEmbedder.process."""
	model_inputs = [
		{
			"text": sample.text,
			"image": sample.image,
			"instruction": SOURCE_INSTRUCTIONS[sample.source],
		}
		for sample in samples
	]
	return {
		"samples": samples,
		"sample_ids": [sample.sample_id for sample in samples],
		"sources": [sample.source for sample in samples],
		"texts": [sample.text for sample in samples],
		"answers": [sample.answer for sample in samples],
		"reasoning_depths": [sample.reasoning_depth for sample in samples],
		"model_inputs": model_inputs,
	}
