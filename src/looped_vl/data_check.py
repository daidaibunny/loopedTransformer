"""Real-data integration checks for both Looped VL mixture splits."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Subset

from looped_vl.data import DEFAULT_DATASET_ROOT, LoopedVLMixtureDataset, mixture_collate

LOGGER = logging.getLogger("data_check")


def representative_indices(dataset_length: int) -> list[int]:
	"""Select COCO, GQA, and CLEVR rows from first, middle, and last blocks."""
	if dataset_length < 20 or dataset_length % 20 != 0:
		raise ValueError("Dataset length must be positive and divisible by 20")
	middle_block = dataset_length // 2
	middle_block -= middle_block % 20
	last_block = dataset_length - 20
	return [
		block_start + source_offset
		for block_start in (0, middle_block, last_block)
		for source_offset in (0, 10, 17)
	]


def check_split(
	dataset_root: Path,
	gqa_materialized_root: Path,
	split: str,
	num_workers: int,
) -> dict[str, Any]:
	"""Read representative rows through DataLoader workers and validate decoded images."""
	dataset = LoopedVLMixtureDataset(dataset_root, split, gqa_materialized_root)
	indices = representative_indices(len(dataset))
	subset = Subset(dataset, indices)
	loader = DataLoader(
		subset,
		batch_size=3,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=mixture_collate,
		persistent_workers=num_workers > 0,
	)
	start = time.perf_counter()
	sources: Counter[str] = Counter()
	sample_ids: list[str] = []
	image_sizes: list[list[int]] = []
	for batch in loader:
		for sample in batch["samples"]:
			if sample.image.mode != "RGB":
				raise RuntimeError(f"Image {sample.sample_id} is not RGB")
			if sample.image.width <= 0 or sample.image.height <= 0:
				raise RuntimeError(f"Image {sample.sample_id} has an invalid size")
			sources[sample.source] += 1
			sample_ids.append(sample.sample_id)
			image_sizes.append([sample.image.width, sample.image.height])
	elapsed = time.perf_counter() - start
	if sources != Counter({"coco": 3, "gqa_balanced": 3, "clevr": 3}):
		raise RuntimeError(f"Unexpected source counts for {split}: {sources}")
	return {
		"split": split,
		"dataset_length": len(dataset),
		"checked_indices": indices,
		"checked_sample_ids": sample_ids,
		"source_counts": dict(sources),
		"image_sizes": image_sizes,
		"num_workers": num_workers,
		"elapsed_seconds": elapsed,
	}


def parse_args() -> argparse.Namespace:
	"""Parse data integration arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset-root",
		default=DEFAULT_DATASET_ROOT,
	)
	parser.add_argument(
		"--gqa-materialized-root",
		default="/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced",
	)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--output-json")
	return parser.parse_args()


def main() -> int:
	"""Run train and validation data integration checks."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = parse_args()
	try:
		results = [
			check_split(
				Path(args.dataset_root),
				Path(args.gqa_materialized_root),
				split,
				args.num_workers,
			)
			for split in ("train", "validation")
		]
		serialized = json.dumps({"status": "passed", "splits": results}, indent=2) + "\n"
		print(serialized, end="")
		if args.output_json:
			Path(args.output_json).write_text(serialized, encoding="utf-8")
		return 0
	except Exception:
		LOGGER.exception("Data integration check failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
