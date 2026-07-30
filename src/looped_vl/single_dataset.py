"""Build full, source-pure manifests from the official dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_SCHEMA = pa.schema(
	[
		pa.field("mixture_position", pa.int64(), nullable=False),
		pa.field("selection_key", pa.uint64(), nullable=False),
		pa.field("sample_id", pa.string(), nullable=False),
		pa.field("source_sample_id", pa.string(), nullable=False),
		pa.field("source", pa.string(), nullable=False),
		pa.field("source_split", pa.string(), nullable=False),
		pa.field("task_type", pa.string(), nullable=False),
		pa.field("image_storage", pa.string(), nullable=False),
		pa.field("image_path", pa.string(), nullable=False),
		pa.field("image_config", pa.string(), nullable=False),
		pa.field("image_id", pa.string(), nullable=False),
		pa.field("text", pa.string(), nullable=False),
		pa.field("answer", pa.string(), nullable=False),
		pa.field("full_answer", pa.string(), nullable=False),
		pa.field("reasoning_trace_json", pa.string(), nullable=False),
		pa.field("reasoning_depth", pa.int16(), nullable=False),
		pa.field("metadata_json", pa.string(), nullable=False),
	],
)


def _compact_json(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _write_split(
	records: Iterable[dict[str, Any]],
	output_root: Path,
	rows_per_shard: int,
) -> dict[str, Any]:
	"""Write one complete split while keeping only one shard buffer in memory."""
	if rows_per_shard <= 0:
		raise ValueError("rows_per_shard must be positive")
	output_root.mkdir(parents=True)
	buffer: list[dict[str, Any]] = []
	shards: list[dict[str, Any]] = []
	sample_ids: set[str] = set()
	image_ids: set[str] = set()
	source_name: str | None = None
	row_count = 0

	def flush() -> None:
		nonlocal buffer
		if not buffer:
			return
		path = output_root / f"part-{len(shards):05d}.parquet"
		pq.write_table(
			pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA),
			path,
			compression="zstd",
			row_group_size=min(rows_per_shard, 8192),
		)
		shards.append(
			{
				"file": path.name,
				"rows": len(buffer),
				"sha256": _sha256(path),
			},
		)
		buffer = []

	for position, input_record in enumerate(records):
		record = {
			**input_record,
			"mixture_position": position,
			"selection_key": position,
		}
		sample_id = str(record["sample_id"])
		if sample_id in sample_ids:
			raise ValueError(f"Duplicate sample identifier: {sample_id}")
		sample_ids.add(sample_id)
		image_ids.add(str(record["image_id"]))
		if source_name is None:
			source_name = str(record["source"])
		elif record["source"] != source_name:
			raise ValueError("A single-dataset manifest cannot mix sources")
		buffer.append(record)
		row_count += 1
		if len(buffer) == rows_per_shard:
			flush()
	flush()
	if row_count == 0:
		raise ValueError(f"No eligible rows were found for {output_root}")
	return {
		"sample_rows": row_count,
		"unique_sample_ids": len(sample_ids),
		"unique_images": len(image_ids),
		"source": source_name,
		"shards": shards,
	}


def _finalize_dataset(
	*,
	output_root: Path,
	dataset_name: str,
	source: str,
	sample_unit: str,
	official_splits: dict[str, str],
	source_root: Path,
	split_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
	result = {
		"dataset_name": dataset_name,
		"source": source,
		"sample_unit": sample_unit,
		"selection": "all_eligible_official_rows_without_sampling",
		"official_splits": official_splits,
		"source_root": str(source_root),
		"splits": split_stats,
		"schema": str(OUTPUT_SCHEMA),
	}
	(output_root / "config.json").write_text(
		json.dumps(result, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	(output_root / "stats.json").write_text(
		json.dumps(result, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	checksum_lines = [
		f"{shard['sha256']}  {split}/{shard['file']}"
		for split, stats in split_stats.items()
		for shard in stats["shards"]
	]
	(output_root / "checksums.sha256").write_text(
		"\n".join(checksum_lines) + "\n",
		encoding="utf-8",
	)
	(output_root / ".ready").write_text("ready\n", encoding="utf-8")
	return result


def _prepare_output_root(output_root: str | Path) -> Path:
	path = Path(output_root)
	if path.exists():
		raise FileExistsError(f"Output dataset already exists: {path}")
	path.mkdir(parents=True)
	return path


def _coco_records(source_root: Path, split: str) -> Iterator[dict[str, Any]]:
	annotation_path = source_root / "annotations" / f"captions_{split}2017.json"
	payload = json.loads(annotation_path.read_text(encoding="utf-8"))
	image_names = {str(item["id"]): str(item["file_name"]) for item in payload["images"]}
	image_root = source_root / f"{split}2017"
	for annotation in payload["annotations"]:
		source_sample_id = str(annotation["id"])
		image_id = str(annotation["image_id"])
		image_path = image_root / image_names[image_id]
		if not image_path.is_file():
			raise FileNotFoundError(f"Missing COCO image: {image_path}")
		caption = str(annotation["caption"]).strip()
		if not caption:
			raise ValueError(f"Empty COCO caption: {source_sample_id}")
		yield {
			"sample_id": f"coco:{split}:{source_sample_id}",
			"source_sample_id": source_sample_id,
			"source": "coco",
			"source_split": split,
			"task_type": "image_text_matching",
			"image_storage": "filesystem",
			"image_path": str(image_path),
			"image_config": "",
			"image_id": image_id,
			"text": caption,
			"answer": "",
			"full_answer": "",
			"reasoning_trace_json": "[]",
			"reasoning_depth": 0,
			"metadata_json": _compact_json({"image_file_name": image_names[image_id]}),
		}


def build_coco_dataset(
	source_root: str | Path,
	output_root: str | Path,
	*,
	rows_per_shard: int = 100_000,
) -> dict[str, Any]:
	"""Build all COCO train2017 and val2017 caption-image pairs."""
	source_path = Path(source_root)
	output_path = _prepare_output_root(output_root)
	official_splits = {"train": "train2017", "validation": "val2017"}
	split_stats = {
		output_split: _write_split(
			_coco_records(source_path, source_split.removesuffix("2017")),
			output_path / output_split,
			rows_per_shard,
		)
		for output_split, source_split in official_splits.items()
	}
	return _finalize_dataset(
		output_root=output_path,
		dataset_name="coco_full_official",
		source="coco",
		sample_unit="caption_image_pair",
		official_splits=official_splits,
		source_root=source_path,
		split_stats=split_stats,
	)


def _gqa_records(source_root: Path, split: str) -> Iterator[dict[str, Any]]:
	paths = sorted((source_root / f"{split}_balanced_instructions").glob("*.parquet"))
	if not paths:
		raise FileNotFoundError(f"Missing GQA Balanced instructions for {split}")
	columns = [
		"id",
		"imageId",
		"question",
		"answer",
		"fullAnswer",
		"isBalanced",
		"semantic",
		"semanticStr",
	]
	for path in paths:
		parquet_file = pq.ParquetFile(path)
		available_columns = set(parquet_file.schema_arrow.names)
		missing = set(columns) - available_columns
		if missing:
			raise ValueError(f"GQA file {path} is missing columns: {sorted(missing)}")
		for batch in parquet_file.iter_batches(columns=columns, batch_size=8192):
			for row in batch.to_pylist():
				source_sample_id = str(row["id"])
				if row["isBalanced"] is not True:
					raise ValueError(f"GQA row is not balanced: {source_sample_id}")
				answer = str(row["answer"] or "").strip()
				if not answer:
					raise ValueError(f"GQA scoring row has no answer: {source_sample_id}")
				semantic = row["semantic"] or []
				yield {
					"sample_id": f"gqa_balanced:{split}:{source_sample_id}",
					"source_sample_id": source_sample_id,
					"source": "gqa_balanced",
					"source_split": split,
					"task_type": "visual_question_answering",
					"image_storage": "hf_parquet",
					"image_path": "",
					"image_config": f"{split}_balanced_images",
					"image_id": str(row["imageId"]),
					"text": str(row["question"]).strip(),
					"answer": answer,
					"full_answer": str(row["fullAnswer"] or "").strip(),
					"reasoning_trace_json": _compact_json(semantic),
					"reasoning_depth": len(semantic),
					"metadata_json": _compact_json({"semantic_str": row["semanticStr"] or ""}),
				}


def build_gqa_balanced_dataset(
	source_root: str | Path,
	output_root: str | Path,
	*,
	rows_per_shard: int = 100_000,
) -> dict[str, Any]:
	"""Build every answered row from the official GQA Balanced train and val splits."""
	source_path = Path(source_root)
	output_path = _prepare_output_root(output_root)
	official_splits = {"train": "train", "validation": "val"}
	split_stats = {
		output_split: _write_split(
			_gqa_records(source_path, source_split),
			output_path / output_split,
			rows_per_shard,
		)
		for output_split, source_split in official_splits.items()
	}
	return _finalize_dataset(
		output_root=output_path,
		dataset_name="gqa_balanced_full_official",
		source="gqa_balanced",
		sample_unit="balanced_visual_question",
		official_splits=official_splits,
		source_root=source_path,
		split_stats=split_stats,
	)


def _stream_json_array(path: Path, expression: str) -> Iterator[dict[str, Any]]:
	process = subprocess.Popen(
		["jq", "-c", expression, str(path)],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)
	if process.stdout is None or process.stderr is None:
		raise RuntimeError(f"Failed to stream JSON file: {path}")
	try:
		for line in process.stdout:
			yield json.loads(line)
	finally:
		process.stdout.close()
	stderr = process.stderr.read().decode(errors="replace")
	return_code = process.wait()
	if return_code != 0:
		raise RuntimeError(f"jq failed for {path}: {stderr.strip()}")


def _clevr_records(source_root: Path, split: str) -> Iterator[dict[str, Any]]:
	question_path = source_root / "questions" / f"CLEVR_{split}_questions.json"
	image_root = source_root / "images" / split
	for row in _stream_json_array(question_path, ".questions[]"):
		source_sample_id = str(row["question_index"])
		if "answer" not in row:
			raise ValueError(f"CLEVR scoring row has no answer: {source_sample_id}")
		image_name = str(row["image_filename"])
		image_path = image_root / image_name
		if not image_path.is_file():
			raise FileNotFoundError(f"Missing CLEVR image: {image_path}")
		program = row["program"] or []
		yield {
			"sample_id": f"clevr:{split}:{source_sample_id}",
			"source_sample_id": source_sample_id,
			"source": "clevr",
			"source_split": split,
			"task_type": "visual_question_answering",
			"image_storage": "filesystem",
			"image_path": str(image_path),
			"image_config": "",
			"image_id": str(row["image_index"]),
			"text": str(row["question"]).strip(),
			"answer": str(row["answer"]).strip(),
			"full_answer": str(row["answer"]).strip(),
			"reasoning_trace_json": _compact_json(program),
			"reasoning_depth": len(program),
			"metadata_json": _compact_json(
				{
					"image_file_name": image_name,
					"question_family_index": row["question_family_index"],
				},
			),
		}


def build_clevr_dataset(
	source_root: str | Path,
	output_root: str | Path,
	*,
	rows_per_shard: int = 100_000,
) -> dict[str, Any]:
	"""Build every answered question from the official CLEVR train and val splits."""
	source_path = Path(source_root)
	output_path = _prepare_output_root(output_root)
	official_splits = {"train": "train", "validation": "val"}
	split_stats = {
		output_split: _write_split(
			_clevr_records(source_path, source_split),
			output_path / output_split,
			rows_per_shard,
		)
		for output_split, source_split in official_splits.items()
	}
	return _finalize_dataset(
		output_root=output_path,
		dataset_name="clevr_full_official",
		source="clevr",
		sample_unit="visual_question",
		official_splits=official_splits,
		source_root=source_path,
		split_stats=split_stats,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source", choices=("coco", "gqa_balanced", "clevr"), required=True)
	parser.add_argument("--source-root", type=Path, required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--rows-per-shard", type=int, default=100_000)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	builders = {
		"coco": build_coco_dataset,
		"gqa_balanced": build_gqa_balanced_dataset,
		"clevr": build_clevr_dataset,
	}
	result = builders[args.source](
		args.source_root,
		args.output_root,
		rows_per_shard=args.rows_per_shard,
	)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
