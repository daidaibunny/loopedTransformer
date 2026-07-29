"""Audit raw and official Qwen-processed image resolutions for a dataset slice."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

from looped_vl.data import LoopedVLMixtureDataset

QWEN_IMAGE_FACTOR = 32


def _percentile(values: list[int], fraction: float) -> float:
	"""Return an interpolated percentile for a non-empty integer sequence."""
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	lower_index = int(position)
	upper_index = min(lower_index + 1, len(ordered) - 1)
	weight = position - lower_index
	return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def _distribution(values: list[int]) -> dict[str, float | int]:
	"""Summarize one non-empty resolution-derived integer measurement."""
	if not values:
		raise ValueError("Cannot summarize an empty measurement")
	return {
		"minimum": min(values),
		"p10": _percentile(values, 0.10),
		"median": _percentile(values, 0.50),
		"p90": _percentile(values, 0.90),
		"maximum": max(values),
		"mean": statistics.fmean(values),
	}


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
	"""Summarize raw and processed dimensions for one source group."""
	if not records:
		raise ValueError("Cannot summarize an empty source group")
	return {
		"count": len(records),
		"unique_raw_resolutions": len(
			{(record["raw_width"], record["raw_height"]) for record in records},
		),
		"unique_processed_resolutions": len(
			{
				(record["processed_width"], record["processed_height"])
				for record in records
			},
		),
		"raw_width": _distribution([record["raw_width"] for record in records]),
		"raw_height": _distribution([record["raw_height"] for record in records]),
		"raw_pixels": _distribution([record["raw_pixels"] for record in records]),
		"processed_pixels": _distribution(
			[record["processed_pixels"] for record in records],
		),
		"visual_tokens": _distribution([record["visual_tokens"] for record in records]),
	}


def summarize_resolutions(
	records: list[dict[str, Any]],
	min_pixels: int,
	max_pixels: int,
) -> dict[str, Any]:
	"""Apply Qwen's official smart resize and summarize raw/processed image sizes."""
	if not records:
		raise ValueError("At least one resolution record is required")
	processed_records: list[dict[str, Any]] = []
	for record in records:
		raw_width = int(record["raw_width"])
		raw_height = int(record["raw_height"])
		processed_height, processed_width = smart_resize(
			height=raw_height,
			width=raw_width,
			factor=QWEN_IMAGE_FACTOR,
			min_pixels=min_pixels,
			max_pixels=max_pixels,
		)
		processed_records.append(
			{
				**record,
				"raw_width": raw_width,
				"raw_height": raw_height,
				"raw_pixels": raw_width * raw_height,
				"processed_width": processed_width,
				"processed_height": processed_height,
				"processed_pixels": processed_width * processed_height,
				"visual_tokens": (
					processed_width * processed_height // (QWEN_IMAGE_FACTOR**2)
				),
			},
		)

	by_source: dict[str, Any] = {}
	for source in sorted({record["source"] for record in processed_records}):
		by_source[source] = _summarize_group(
			[record for record in processed_records if record["source"] == source],
		)
	return {
		"official_preprocessing": {
			"factor": QWEN_IMAGE_FACTOR,
			"min_pixels": min_pixels,
			"max_pixels": max_pixels,
			"visual_token_definition": "processed_pixels / factor^2",
		},
		"source_counts": dict(Counter(record["source"] for record in processed_records)),
		"overall": _summarize_group(processed_records),
		"by_source": by_source,
	}


def audit_dataset_slice(args: argparse.Namespace) -> dict[str, Any]:
	"""Read image headers for an exact contiguous mixture slice and summarize sizes."""
	dataset = LoopedVLMixtureDataset(
		args.dataset_root,
		args.split,
		args.gqa_materialized_root,
	)
	end_index = args.start_index + args.sample_count
	if args.start_index < 0 or end_index > len(dataset):
		raise ValueError(
			f"Requested indexes [{args.start_index}, {end_index}) exceed dataset length "
			f"{len(dataset)}",
		)

	records: list[dict[str, Any]] = []
	for index in range(args.start_index, end_index):
		dataset_record = dataset.get_record(index)
		image_path = dataset.resolve_image_path(dataset_record)
		with Image.open(image_path) as image:
			raw_width, raw_height = image.size
		records.append(
			{
				"index": index,
				"source": dataset_record["source"],
				"raw_width": raw_width,
				"raw_height": raw_height,
			},
		)

	result = summarize_resolutions(records, args.min_pixels, args.max_pixels)
	result.update(
		{
			"status": "passed",
			"dataset_root": str(Path(args.dataset_root)),
			"split": args.split,
			"start_index": args.start_index,
			"end_index_exclusive": end_index,
			"sample_count": args.sample_count,
		}
	)
	return result


def parse_args() -> argparse.Namespace:
	"""Parse dataset resolution audit arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--dataset-root",
		default="/mnt/afs/liyiwei/datasets/looped_vl_mix_v1",
	)
	parser.add_argument(
		"--gqa-materialized-root",
		default="/mnt/afs/liyiwei/datasets/gqa_hf_full/materialized_balanced",
	)
	parser.add_argument("--split", choices=("train", "validation"), default="train")
	parser.add_argument("--start-index", type=int, default=100)
	parser.add_argument("--sample-count", type=int, default=500)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument("--output-json", type=Path, required=True)
	return parser.parse_args()


def main() -> None:
	"""Run the resolution audit and save the result as JSON."""
	args = parse_args()
	result = audit_dataset_slice(args)
	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(
		json.dumps(result, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
