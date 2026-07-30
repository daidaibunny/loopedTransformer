"""Create a deterministic prefix subset of an existing interleaved mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from looped_vl.data import DEFAULT_TEST_SAMPLES, DEFAULT_VALIDATION_SAMPLES

EXPECTED_BLOCK_COUNTS = {"coco": 10, "gqa_balanced": 7, "clevr": 3}
INTERLEAVE_BLOCK_SIZE = sum(EXPECTED_BLOCK_COUNTS.values())


def _read_prefix(split_root: Path, sample_count: int) -> pa.Table:
	"""Read exactly the requested prefix across ordered Parquet shards."""
	if sample_count <= 0:
		raise ValueError("sample_count must be positive")
	remaining = sample_count
	tables: list[pa.Table] = []
	for path in sorted(split_root.glob("*.parquet")):
		if remaining == 0:
			break
		table = pq.read_table(path)
		rows_to_take = min(remaining, table.num_rows)
		tables.append(table.slice(0, rows_to_take))
		remaining -= rows_to_take
	if remaining:
		raise ValueError(
			f"Requested {sample_count} rows but only found {sample_count - remaining} "
			f"under {split_root}",
		)
	return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _read_split(split_root: Path) -> pa.Table:
	"""Read all ordered Parquet shards for one split."""
	paths = sorted(split_root.glob("*.parquet"))
	if not paths:
		raise FileNotFoundError(f"No Parquet shards found under {split_root}")
	tables = [pq.read_table(path) for path in paths]
	return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _source_counts(table: pa.Table) -> dict[str, int]:
	"""Count source rows in a mixture table."""
	counts = Counter(table.column("source").to_pylist())
	return {source: counts[source] for source in EXPECTED_BLOCK_COUNTS}


def _validate_interleave(table: pa.Table) -> dict[str, int]:
	"""Verify total ratio and every contiguous 20-row mixture block."""
	if table.num_rows % INTERLEAVE_BLOCK_SIZE:
		raise ValueError(
			f"Row count {table.num_rows} is not divisible by {INTERLEAVE_BLOCK_SIZE}",
		)
	sources = table.column("source").to_pylist()
	for start in range(0, table.num_rows, INTERLEAVE_BLOCK_SIZE):
		block_counts = Counter(sources[start : start + INTERLEAVE_BLOCK_SIZE])
		if dict(block_counts) != EXPECTED_BLOCK_COUNTS:
			raise ValueError(f"Invalid source ratio in block starting at row {start}")
	return _source_counts(table)


def _source_stats(table: pa.Table) -> dict[str, dict[str, Any]]:
	"""Compute selected rows, unique images, and reasoning-depth histograms."""
	sources = table.column("source").to_pylist()
	image_ids = table.column("image_id").to_pylist()
	reasoning_depths = table.column("reasoning_depth").to_pylist()
	image_sets: dict[str, set[str]] = defaultdict(set)
	depth_counts: dict[str, Counter[int]] = defaultdict(Counter)
	row_counts: Counter[str] = Counter()
	for source, image_id, reasoning_depth in zip(
		sources,
		image_ids,
		reasoning_depths,
		strict=True,
	):
		row_counts[source] += 1
		image_sets[source].add(image_id)
		depth_counts[source][reasoning_depth] += 1
	return {
		source: {
			"selected_samples": row_counts[source],
			"selected_unique_images": len(image_sets[source]),
			"reasoning_depth_histogram": {
				str(depth): count for depth, count in sorted(depth_counts[source].items())
			},
		}
		for source in EXPECTED_BLOCK_COUNTS
	}


def _verification(table: pa.Table, source_counts: dict[str, int]) -> dict[str, Any]:
	"""Build exact row and identifier checks for one output split."""
	sample_ids = table.column("sample_id").to_pylist()
	return {
		"files": 1,
		"rows": table.num_rows,
		"source_counts": source_counts,
		"unique_sample_ids": len(set(sample_ids)),
	}


def _sha256(path: Path) -> str:
	"""Hash one generated data file."""
	digest = hashlib.sha256()
	with path.open("rb") as file_handle:
		for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def build_prefix_subset(
	source_root: str | Path,
	output_root: str | Path,
	train_samples: int,
	validation_samples: int = DEFAULT_VALIDATION_SAMPLES,
	test_samples: int = DEFAULT_TEST_SAMPLES,
) -> dict[str, Any]:
	"""Build train, validation, and disjoint test splits from a parent mixture."""
	requested_counts = {
		"train": train_samples,
		"validation": validation_samples,
		"test": test_samples,
	}
	invalid_counts = {
		split: count
		for split, count in requested_counts.items()
		if count <= 0 or count % INTERLEAVE_BLOCK_SIZE
	}
	if invalid_counts:
		raise ValueError(
			f"Every split count must be positive and divisible by {INTERLEAVE_BLOCK_SIZE}: "
			f"{invalid_counts}",
		)
	source_path = Path(source_root)
	output_path = Path(output_root)
	if output_path.exists():
		raise FileExistsError(f"Output dataset already exists: {output_path}")

	train_table = _read_prefix(source_path / "train", train_samples)
	parent_validation_table = _read_split(source_path / "validation")
	evaluation_samples = validation_samples + test_samples
	if evaluation_samples > parent_validation_table.num_rows:
		raise ValueError(
			f"Requested {evaluation_samples} validation and test rows but parent contains "
			f"{parent_validation_table.num_rows}",
		)
	validation_table = parent_validation_table.slice(0, validation_samples)
	test_table = parent_validation_table.slice(validation_samples, test_samples)
	train_counts = _validate_interleave(train_table)
	validation_counts = _validate_interleave(validation_table)
	test_counts = _validate_interleave(test_table)
	if len(set(train_table.column("sample_id").to_pylist())) != train_samples:
		raise ValueError("Train prefix contains duplicate sample identifiers")
	validation_ids = set(validation_table.column("sample_id").to_pylist())
	test_ids = set(test_table.column("sample_id").to_pylist())
	if len(validation_ids) != validation_samples or len(test_ids) != test_samples:
		raise ValueError("Validation or test split contains duplicate sample identifiers")
	if not validation_ids.isdisjoint(test_ids):
		raise ValueError("Validation and test sample identifiers overlap")

	train_root = output_path / "train"
	validation_root = output_path / "validation"
	test_root = output_path / "test"
	train_root.mkdir(parents=True)
	validation_root.mkdir(parents=True)
	test_root.mkdir(parents=True)
	train_file = train_root / "part-00000-of-00001.parquet"
	validation_file = validation_root / "part-00000-of-00001.parquet"
	test_file = test_root / "part-00000-of-00001.parquet"
	pq.write_table(train_table, train_file, compression="zstd", row_group_size=8192)
	pq.write_table(validation_table, validation_file, compression="zstd", row_group_size=8192)
	pq.write_table(test_table, test_file, compression="zstd", row_group_size=8192)

	parent_config_path = source_path / "config.json"
	parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
	dataset_name = (
		f"{parent_config.get('dataset_name', source_path.name)}_train{train_samples}"
		f"_val{validation_samples}_test{test_samples}"
	)
	config = {
		**parent_config,
		"dataset_name": dataset_name,
		"parent_dataset_root": str(source_path),
		"subset_method": "deterministic_prefix_of_interleaved_parent",
		"train_counts": train_counts,
		"validation_counts": validation_counts,
		"test_counts": test_counts,
		"schema": str(train_table.schema),
	}
	stats = {
		"dataset_name": dataset_name,
		"parent_dataset_root": str(source_path),
		"selection": "deterministic_prefix_of_interleaved_parent",
		"interleave_block": EXPECTED_BLOCK_COUNTS,
		"seed": parent_config.get("seed"),
		"splits": {
			"train": {
				"requested_counts": train_counts,
				"shards": [
					{
						"file": train_file.name,
						"rows": train_table.num_rows,
						"source_counts": train_counts,
					},
				],
				"source_stats": _source_stats(train_table),
				"verification": _verification(train_table, train_counts),
			},
			"validation": {
				"requested_counts": validation_counts,
				"shards": [
					{
						"file": validation_file.name,
						"rows": validation_table.num_rows,
						"source_counts": validation_counts,
					},
				],
				"source_stats": _source_stats(validation_table),
				"verification": _verification(validation_table, validation_counts),
			},
			"test": {
				"requested_counts": test_counts,
				"shards": [
					{
						"file": test_file.name,
						"rows": test_table.num_rows,
						"source_counts": test_counts,
					},
				],
				"source_stats": _source_stats(test_table),
				"verification": _verification(test_table, test_counts),
			},
		},
	}
	(output_path / "config.json").write_text(
		json.dumps(config, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	(output_path / "stats.json").write_text(
		json.dumps(stats, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	checksum_lines = [
		f"{_sha256(train_file)}  train/{train_file.name}",
		f"{_sha256(validation_file)}  validation/{validation_file.name}",
		f"{_sha256(test_file)}  test/{test_file.name}",
	]
	(output_path / "checksums.sha256").write_text(
		"\n".join(checksum_lines) + "\n",
		encoding="utf-8",
	)
	(output_path / ".ready").write_text("ready\n", encoding="utf-8")
	return {
		"dataset_root": str(output_path),
		"train": _verification(train_table, train_counts),
		"validation": _verification(validation_table, validation_counts),
		"test": _verification(test_table, test_counts),
	}


def parse_args() -> argparse.Namespace:
	"""Parse subset creation arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-root", type=Path, required=True)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--train-samples", type=int, default=100_000)
	parser.add_argument(
		"--validation-samples",
		type=int,
		default=DEFAULT_VALIDATION_SAMPLES,
	)
	parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
	return parser.parse_args()


def main() -> None:
	"""Build a subset and print its exact verification summary."""
	args = parse_args()
	result = build_prefix_subset(
		args.source_root,
		args.output_root,
		args.train_samples,
		args.validation_samples,
		args.test_samples,
	)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
