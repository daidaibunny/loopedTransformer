"""Immutable candidate galleries shared by every recurrent retrieval experiment."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from looped_vl.baseline.data import BASELINE_DATASETS, BASELINE_SPLITS

CANDIDATE_BANK_VERSION = "frozen_qwen3vl_candidate_bank_v1"
EMBEDDING_DIMENSION = 2048
DEFAULT_EMBEDDING_SHARD_ROWS = 8192


@dataclass(frozen=True, order=True)
class CandidateBankSpec:
	"""One dataset gallery with a stable split and modality identity."""

	dataset: str
	split: str
	gallery: str

	def __post_init__(self) -> None:
		if self.dataset not in BASELINE_DATASETS:
			raise ValueError(f"Unsupported candidate-bank dataset: {self.dataset}")
		if self.dataset == "coco":
			if self.split not in BASELINE_SPLITS:
				raise ValueError(f"Unsupported COCO candidate split: {self.split}")
			if self.gallery not in {"image", "text"}:
				raise ValueError(f"Unsupported COCO candidate gallery: {self.gallery}")
		elif self.split != "shared" or self.gallery != "answer":
			raise ValueError(
				f"{self.dataset} must use the shared training-answer gallery",
			)

	@property
	def key(self) -> str:
		"""Return the stable manifest key used by training and evaluation."""
		return f"{self.dataset}/{self.split}/{self.gallery}"

	@property
	def relative_path(self) -> Path:
		"""Return the canonical directory below one candidate-bank root."""
		return Path(self.dataset) / self.split / self.gallery


CANDIDATE_BANK_SPECS = tuple(
	[
		CandidateBankSpec("coco", split, gallery)
		for split in BASELINE_SPLITS
		for gallery in ("image", "text")
	]
	+ [
		CandidateBankSpec("gqa_balanced", "shared", "answer"),
		CandidateBankSpec("clevr", "shared", "answer"),
	],
)


@dataclass(frozen=True)
class CandidateItem:
	"""One independently encoded candidate with its retrieval identity."""

	item_id: str
	positive_id: str
	text: str | None = None
	image_path: Path | None = None

	def __post_init__(self) -> None:
		if not self.item_id:
			raise ValueError("Candidate item_id cannot be empty")
		if not self.positive_id:
			raise ValueError("Candidate positive_id cannot be empty")
		if (self.text is None) == (self.image_path is None):
			raise ValueError("Candidate must contain exactly one of text or image_path")
		if self.text is not None and not self.text.strip():
			raise ValueError("Candidate text cannot be blank")

	@property
	def input_kind(self) -> str:
		"""Return the official Qwen input modality for this candidate."""
		return "text" if self.text is not None else "image"

	def model_input(self) -> dict[str, str]:
		"""Return the instruction-free input used by the official target tower."""
		if self.text is not None:
			return {"text": self.text}
		if self.image_path is None:
			raise RuntimeError("Candidate image path unexpectedly missing")
		return {"image": str(self.image_path)}


ITEM_SCHEMA = pa.schema(
	[
		("item_index", pa.int64()),
		("item_id", pa.string()),
		("positive_id", pa.string()),
		("input_kind", pa.string()),
		("text", pa.string()),
		("image_path", pa.string()),
	],
)


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
	"""Return the SHA-256 checksum of one file without loading it into memory."""
	digest = hashlib.sha256()
	with Path(path).open("rb") as handle:
		while chunk := handle.read(chunk_size):
			digest.update(chunk)
	return digest.hexdigest()


def _iter_parquet_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
	for path in paths:
		parquet_file = pq.ParquetFile(path)
		for batch in parquet_file.iter_batches(batch_size=8192):
			yield from batch.to_pylist()


def source_files_for_spec(
	dataset_root: str | Path,
	spec: CandidateBankSpec,
) -> tuple[Path, ...]:
	"""Return every split-authority file that determines one candidate gallery."""
	root = Path(dataset_root)
	config_path = root / "config.json"
	if not config_path.is_file():
		raise FileNotFoundError(f"Missing dataset configuration: {config_path}")
	if spec.dataset == "coco":
		parquet_paths = tuple(sorted((root / spec.split).glob("*.parquet")))
		if not parquet_paths:
			raise FileNotFoundError(f"No COCO manifest shards under {root / spec.split}")
		return (config_path, *parquet_paths)
	answer_path = root / "answer_gallery.json"
	if not answer_path.is_file():
		raise FileNotFoundError(f"Missing training answer gallery: {answer_path}")
	return config_path, answer_path


def source_checksums_for_spec(
	dataset_root: str | Path,
	spec: CandidateBankSpec,
) -> dict[str, str]:
	"""Hash the exact source files that define one bank."""
	root = Path(dataset_root)
	return {
		str(path.relative_to(root)): sha256_file(path)
		for path in source_files_for_spec(root, spec)
	}


def iter_candidate_items(
	dataset_root: str | Path,
	spec: CandidateBankSpec,
) -> Iterator[CandidateItem]:
	"""Yield candidates in their canonical retrieval-gallery order."""
	root = Path(dataset_root)
	if spec.dataset == "coco":
		paths = sorted((root / spec.split).glob("*.parquet"))
		if not paths:
			raise FileNotFoundError(f"No COCO manifest shards under {root / spec.split}")
		seen_images: set[str] = set()
		for row in _iter_parquet_rows(paths):
			row_dataset = str(row["dataset"])
			row_split = str(row["split"])
			if row_dataset != spec.dataset or row_split != spec.split:
				raise ValueError(
					f"Candidate source row has identity {row_dataset}/{row_split}; "
					f"expected {spec.dataset}/{spec.split}",
				)
			positive_id = str(row["positive_id"])
			if spec.gallery == "image":
				if positive_id in seen_images:
					continue
				seen_images.add(positive_id)
				yield CandidateItem(
					item_id=positive_id,
					positive_id=positive_id,
					image_path=Path(str(row["image_path"])),
				)
			else:
				yield CandidateItem(
					item_id=str(row["sample_id"]),
					positive_id=positive_id,
					text=str(row["query_text"]),
				)
		return

	gallery_path = root / "answer_gallery.json"
	if not gallery_path.is_file():
		raise FileNotFoundError(f"Missing training answer gallery: {gallery_path}")
	gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
	if not isinstance(gallery, list):
		raise TypeError(f"Answer gallery must be a list: {gallery_path}")
	for row in gallery:
		positive_id = str(row["positive_id"])
		yield CandidateItem(
			item_id=positive_id,
			positive_id=positive_id,
			text=str(row["text"]),
		)


def build_candidate_items(
	dataset_root: str | Path,
	spec: CandidateBankSpec,
) -> list[CandidateItem]:
	"""Materialize candidate definitions for tests and small galleries."""
	return list(iter_candidate_items(dataset_root, spec))


def _item_record(item: CandidateItem, item_index: int) -> dict[str, Any]:
	return {
		"item_index": item_index,
		"item_id": item.item_id,
		"positive_id": item.positive_id,
		"input_kind": item.input_kind,
		"text": item.text,
		"image_path": str(item.image_path) if item.image_path is not None else None,
	}


def write_candidate_item_manifest(
	*,
	dataset_root: str | Path,
	spec: CandidateBankSpec,
	output_path: str | Path,
	batch_rows: int = 8192,
) -> dict[str, Any]:
	"""Atomically write an indexed candidate manifest and validate unique identities."""
	if batch_rows <= 0:
		raise ValueError("batch_rows must be positive")
	path = Path(output_path)
	if path.exists():
		raise FileExistsError(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = path.with_suffix(path.suffix + ".partial")
	if temporary_path.exists():
		raise FileExistsError(temporary_path)
	writer = pq.ParquetWriter(temporary_path, ITEM_SCHEMA, compression="zstd")
	buffer: list[dict[str, Any]] = []
	seen_item_ids: set[str] = set()
	positive_ids: set[str] = set()
	row_count = 0
	input_kind: str | None = None
	try:
		for item_index, item in enumerate(iter_candidate_items(dataset_root, spec)):
			if item.item_id in seen_item_ids:
				raise ValueError(f"Duplicate candidate item_id: {item.item_id}")
			seen_item_ids.add(item.item_id)
			positive_ids.add(item.positive_id)
			if input_kind is None:
				input_kind = item.input_kind
			elif item.input_kind != input_kind:
				raise ValueError(f"Candidate bank mixes {input_kind} and {item.input_kind}")
			buffer.append(_item_record(item, item_index))
			if len(buffer) >= batch_rows:
				writer.write_table(pa.Table.from_pylist(buffer, schema=ITEM_SCHEMA))
				row_count += len(buffer)
				buffer.clear()
		if buffer:
			writer.write_table(pa.Table.from_pylist(buffer, schema=ITEM_SCHEMA))
			row_count += len(buffer)
	finally:
		writer.close()
	if row_count == 0 or input_kind is None:
		raise ValueError(f"Candidate bank is empty: {spec.key}")
	os.replace(temporary_path, path)
	return {
		"rows": row_count,
		"unique_item_ids": len(seen_item_ids),
		"unique_positive_ids": len(positive_ids),
		"input_kind": input_kind,
		"sha256": sha256_file(path),
	}


def embedding_shard_ranges(item_count: int, shard_rows: int) -> tuple[tuple[int, int], ...]:
	"""Partition one bank into deterministic contiguous embedding ranges."""
	if item_count <= 0:
		raise ValueError("item_count must be positive")
	if shard_rows <= 0:
		raise ValueError("shard_rows must be positive")
	return tuple(
		(start, min(start + shard_rows, item_count))
		for start in range(0, item_count, shard_rows)
	)


def validate_embedding_shard(
	embeddings: torch.Tensor,
	*,
	expected_rows: int,
	embedding_dimension: int = EMBEDDING_DIMENSION,
) -> dict[str, Any]:
	"""Reject incomplete, corrupt, or non-normalized candidate embeddings."""
	if embeddings.ndim != 2:
		raise ValueError(f"Embedding shard must be rank 2, found {embeddings.ndim}")
	if embeddings.shape[0] != expected_rows:
		raise ValueError(
			f"Embedding shard row count is {embeddings.shape[0]}; expected {expected_rows}",
		)
	if embeddings.shape[1] != embedding_dimension:
		raise ValueError(
			f"Embedding dimension is {embeddings.shape[1]}; expected {embedding_dimension}",
		)
	if embeddings.dtype != torch.float16:
		raise ValueError(f"Candidate embeddings must use float16, found {embeddings.dtype}")
	if not torch.isfinite(embeddings).all():
		raise ValueError("Candidate embeddings contain non-finite values")
	norms = torch.linalg.vector_norm(embeddings.float(), dim=1)
	if not torch.allclose(norms, torch.ones_like(norms), atol=5e-3, rtol=5e-3):
		raise ValueError("Candidate embeddings are not unit normalized")
	return {
		"rows": expected_rows,
		"embedding_dimension": embedding_dimension,
		"dtype": "float16",
	}


def write_json_atomic(path: str | Path, value: Any) -> None:
	"""Publish JSON only after its complete temporary file is on disk."""
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
	temporary_path.write_text(
		json.dumps(value, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	os.replace(temporary_path, output_path)


def load_ready_candidate_bank(
	bank_root: str | Path,
	*,
	expected_spec: CandidateBankSpec,
	expected_model_sha256: str,
) -> dict[str, Any]:
	"""Load a published bank only when identity and every shard checksum match."""
	root = Path(bank_root)
	ready_path = root / "READY"
	manifest_path = root / "bank_manifest.json"
	if not ready_path.is_file() or not manifest_path.is_file():
		raise FileNotFoundError(f"Candidate bank is not ready: {root}")
	expected_manifest_sha256 = ready_path.read_text(encoding="utf-8").strip()
	if expected_manifest_sha256 != sha256_file(manifest_path):
		raise ValueError(f"Candidate bank READY checksum mismatch under {root}")
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	if manifest.get("version") != CANDIDATE_BANK_VERSION:
		raise ValueError(f"Unsupported candidate-bank version under {root}")
	if manifest.get("spec") != {
		"dataset": expected_spec.dataset,
		"split": expected_spec.split,
		"gallery": expected_spec.gallery,
	}:
		raise ValueError(f"Candidate-bank spec mismatch under {root}")
	if manifest.get("model", {}).get("checkpoint_sha256") != expected_model_sha256:
		raise ValueError(f"Candidate-bank model checksum mismatch under {root}")
	item_manifest = manifest["items"]
	item_path = root / str(item_manifest["path"])
	if sha256_file(item_path) != item_manifest["sha256"]:
		raise ValueError(f"Candidate item checksum mismatch under {root}")
	covered_rows = 0
	for shard in manifest["embedding_shards"]:
		shard_path = root / str(shard["path"])
		if sha256_file(shard_path) != shard["sha256"]:
			raise ValueError(f"Candidate embedding checksum mismatch: {shard_path}")
		payload = torch.load(shard_path, map_location="cpu", weights_only=True)
		start = int(payload["start"])
		end = int(payload["end"])
		if (
			start != covered_rows
			or start != int(shard["start"])
			or end != int(shard["end"])
		):
			raise ValueError(f"Candidate embedding range mismatch: {shard_path}")
		validate_embedding_shard(
			payload["embeddings"],
			expected_rows=end - start,
			embedding_dimension=int(manifest["embedding_dimension"]),
		)
		covered_rows = end
	if covered_rows != int(item_manifest["rows"]):
		raise ValueError(f"Candidate embedding coverage mismatch under {root}")
	return manifest
