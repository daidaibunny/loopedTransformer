"""Materialize GQA Balanced image Parquet rows into a reusable JPEG cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

LOGGER = logging.getLogger("materialize_gqa")


def materialize_split(
	source_root: str | Path,
	output_root: str | Path,
	split: str,
) -> dict[str, Any]:
	"""Write one GQA Balanced image split and verify every expected identifier."""
	if split not in {"train", "val", "testdev"}:
		raise ValueError(f"Unsupported GQA split: {split}")
	source_root = Path(source_root)
	output_root = Path(output_root)
	config = f"{split}_balanced_images"
	parquet_paths = sorted((source_root / config).glob("*.parquet"))
	if not parquet_paths:
		raise FileNotFoundError(f"No image Parquet files under {source_root / config}")
	expected_images = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
	target_root = output_root / split
	target_root.mkdir(parents=True, exist_ok=True)

	seen_ids: set[str] = set()
	written_images = 0
	reused_images = 0
	total_bytes = 0
	for parquet_path in parquet_paths:
		parquet_file = pq.ParquetFile(parquet_path)
		for batch in parquet_file.iter_batches(columns=["id", "image"], batch_size=64):
			for row in batch.to_pylist():
				image_id = row["id"]
				if image_id in seen_ids:
					raise ValueError(f"Duplicate GQA image id: {image_id}")
				seen_ids.add(image_id)
				image_bytes = row["image"]["bytes"]
				if not image_bytes or not image_bytes.startswith(b"\xff\xd8"):
					raise ValueError(f"GQA image {image_id} is not JPEG encoded")
				total_bytes += len(image_bytes)
				target_path = target_root / f"{image_id}.jpg"
				if target_path.exists():
					if target_path.stat().st_size != len(image_bytes):
						raise RuntimeError(f"Existing file size mismatch: {target_path}")
					reused_images += 1
					continue
				temporary_path = target_path.with_suffix(".jpg.partial")
				with temporary_path.open("xb") as handle:
					handle.write(image_bytes)
				os.replace(temporary_path, target_path)
				written_images += 1

	if len(seen_ids) != expected_images:
		raise RuntimeError(
			f"Materialized {len(seen_ids)} unique images, expected {expected_images}",
		)
	materialized_count = sum(1 for path in target_root.glob("*.jpg") if path.is_file())
	if materialized_count != expected_images:
		raise RuntimeError(
			f"Found {materialized_count} materialized files, expected {expected_images}",
		)
	stats = {
		"split": split,
		"source_config": config,
		"expected_images": expected_images,
		"written_images": written_images,
		"reused_images": reused_images,
		"encoded_bytes": total_bytes,
	}
	(output_root / f".{split}_ready").write_text(
		json.dumps(stats, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	return stats


def parse_args() -> argparse.Namespace:
	"""Parse materialization arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--source-root",
		default="/mnt/afs/liyiwei/datasets/gqa_hf_full",
	)
	parser.add_argument(
		"--output-root",
		default="/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced",
	)
	parser.add_argument("--splits", nargs="+", default=["train", "val", "testdev"])
	return parser.parse_args()


def main() -> int:
	"""Materialize requested splits and emit machine-readable statistics."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = parse_args()
	try:
		for split in args.splits:
			stats = materialize_split(args.source_root, args.output_root, split)
			LOGGER.info("Completed %s", json.dumps(stats, sort_keys=True))
		return 0
	except Exception:
		LOGGER.exception("GQA image materialization failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
