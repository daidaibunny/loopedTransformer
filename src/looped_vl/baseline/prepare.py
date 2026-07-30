"""Build three independent full-data retrieval manifests without mixing sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from looped_vl.baseline.data import BASELINE_DATASETS, normalize_answer

MANIFEST_SCHEMA = pa.schema(
	[
		("sample_id", pa.string()),
		("dataset", pa.string()),
		("split", pa.string()),
		("image_id", pa.string()),
		("image_path", pa.string()),
		("query_text", pa.string()),
		("candidate_text", pa.string()),
		("positive_id", pa.string()),
	],
)
EXPECTED_COCO_IMAGES = {"train": 113_287, "validation": 5_000, "test": 5_000}
EXPECTED_GQA_ROWS = {"train": 943_000, "validation": 132_062, "test": 12_578}
EXPECTED_CLEVR_TRAIN_ROWS = 699_989
EXPECTED_CLEVR_VALIDATION_IMAGES = 15_000


def split_clevr_validation_images(
	image_names: Iterable[str],
	*,
	seed: int,
	validation_images: int,
) -> tuple[set[str], set[str]]:
	"""Deterministically split official CLEVR validation images into dev and test."""
	ordered = sorted(set(image_names))
	if validation_images <= 0 or validation_images >= len(ordered):
		raise ValueError("validation_images must leave non-empty validation and test sets")
	random.Random(seed).shuffle(ordered)
	validation = set(ordered[:validation_images])
	test = set(ordered[validation_images:])
	if not validation.isdisjoint(test):
		raise RuntimeError("CLEVR validation and test image identifiers overlap")
	return validation, test


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _write_rows(
	rows: Iterator[dict[str, str]],
	output_path: Path,
	*,
	batch_rows: int = 8192,
) -> dict[str, Any]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.exists():
		raise FileExistsError(output_path)
	temporary_path = output_path.with_suffix(".parquet.partial")
	if temporary_path.exists():
		raise FileExistsError(temporary_path)
	writer = pq.ParquetWriter(temporary_path, MANIFEST_SCHEMA, compression="zstd")
	buffer: list[dict[str, str]] = []
	row_count = 0
	sample_ids: set[str] = set()
	image_ids: set[str] = set()
	try:
		for row in rows:
			sample_id = row["sample_id"]
			if sample_id in sample_ids:
				raise ValueError(f"Duplicate sample ID: {sample_id}")
			sample_ids.add(sample_id)
			image_ids.add(row["image_id"])
			buffer.append(row)
			if len(buffer) >= batch_rows:
				writer.write_table(pa.Table.from_pylist(buffer, schema=MANIFEST_SCHEMA))
				row_count += len(buffer)
				buffer.clear()
		if buffer:
			writer.write_table(pa.Table.from_pylist(buffer, schema=MANIFEST_SCHEMA))
			row_count += len(buffer)
	finally:
		writer.close()
	if row_count == 0:
		raise ValueError(f"No rows produced for {output_path}")
	os.replace(temporary_path, output_path)
	return {
		"rows": row_count,
		"unique_sample_ids": len(sample_ids),
		"unique_images": len(image_ids),
		"sha256": _sha256(output_path),
	}


def _resolve_coco_image(coco_root: Path, image_id: int) -> Path:
	filename = f"{image_id:012d}.jpg"
	for split_root in ("train2017", "val2017"):
		path = coco_root / split_root / filename
		if path.is_file():
			return path
	raise FileNotFoundError(f"COCO image {image_id} is missing from train2017 and val2017")


def _iter_karpathy_images(path: Path) -> Iterator[dict[str, Any]]:
	with path.open("rb") as handle:
		yield from ijson.items(handle, "images.item")


def prepare_coco(
	*,
	coco_root: Path,
	karpathy_json: Path,
	output_root: Path,
) -> dict[str, Any]:
	"""Materialize the standard Karpathy image-disjoint retrieval split."""
	split_alias = {"train": "train", "restval": "train", "val": "validation", "test": "test"}
	images_by_split: dict[str, list[dict[str, Any]]] = {
		"train": [],
		"validation": [],
		"test": [],
	}
	for image in _iter_karpathy_images(karpathy_json):
		source_split = str(image["split"])
		if source_split not in split_alias:
			raise ValueError(f"Unsupported Karpathy split: {source_split}")
		images_by_split[split_alias[source_split]].append(image)

	stats: dict[str, Any] = {}
	image_sets: dict[str, set[str]] = {}
	for split, images in images_by_split.items():
		image_sets[split] = {str(image["cocoid"]) for image in images}
		if len(image_sets[split]) != EXPECTED_COCO_IMAGES[split]:
			raise RuntimeError(
				f"COCO {split} has {len(image_sets[split])} images; "
				f"expected {EXPECTED_COCO_IMAGES[split]}",
			)

		def rows(
			active_images: list[dict[str, Any]] = images,
			active_split: str = split,
		) -> Iterator[dict[str, str]]:
			for image in active_images:
				image_id = int(image["cocoid"])
				image_path = _resolve_coco_image(coco_root, image_id)
				for sentence_index, sentence in enumerate(image["sentences"]):
					caption = str(sentence["raw"]).strip()
					if not caption:
						raise ValueError(f"Empty COCO caption for image {image_id}")
					sentence_id = sentence.get("sentid", sentence_index)
					yield {
						"sample_id": f"coco:{active_split}:{sentence_id}",
						"dataset": "coco",
						"split": active_split,
						"image_id": str(image_id),
						"image_path": str(image_path),
						"query_text": caption,
						"candidate_text": "",
						"positive_id": f"image:{image_id}",
					}

		stats[split] = _write_rows(
			rows(),
			output_root / split / "part-00000-of-00001.parquet",
		)

	for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
		if not image_sets[left].isdisjoint(image_sets[right]):
			raise RuntimeError(f"COCO image leakage between {left} and {right}")
	return _finalize_dataset(
		output_root,
		dataset="coco",
		split_policy="karpathy_113287_5000_5000_image_disjoint",
		stats=stats,
		sources={"coco_root": str(coco_root), "karpathy_json": str(karpathy_json)},
	)


def _iter_parquet_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
	for path in paths:
		parquet_file = pq.ParquetFile(path)
		for batch in parquet_file.iter_batches(batch_size=8192):
			yield from batch.to_pylist()


def _resolve_materialized_gqa_image(
	materialized_root: Path,
	source_split: str,
	image_id: str,
	verified_images: set[tuple[str, str]],
) -> Path:
	"""Verify each unique GQA image once instead of once per question."""
	image_key = (source_split, image_id)
	image_path = materialized_root / source_split / f"{image_id}.jpg"
	if image_key in verified_images:
		return image_path
	if not image_path.is_file():
		raise FileNotFoundError(f"Missing materialized GQA image: {image_path}")
	verified_images.add(image_key)
	return image_path


def prepare_gqa(
	*,
	gqa_root: Path,
	materialized_root: Path,
	output_root: Path,
) -> dict[str, Any]:
	"""Build official GQA Balanced train, validation, and labeled testdev manifests."""
	source_splits = {
		"train": "train",
		"validation": "val",
		"test": "testdev",
	}
	answer_gallery: dict[str, str] = {}
	verified_images: set[tuple[str, str]] = set()
	stats: dict[str, Any] = {}
	for split, source_split in source_splits.items():
		paths = sorted((gqa_root / f"{source_split}_balanced_instructions").glob("*.parquet"))
		if not paths:
			raise FileNotFoundError(f"Missing GQA instruction Parquet for {source_split}")

		def rows(
			active_paths: list[Path] = paths,
			active_split: str = split,
			active_source_split: str = source_split,
		) -> Iterator[dict[str, str]]:
			for row in _iter_parquet_rows(active_paths):
				answer = str(row["answer"]).strip()
				answer_key = normalize_answer(answer)
				if not answer_key:
					raise ValueError(f"Empty GQA answer for {row['id']}")
				if active_split == "train":
					answer_gallery.setdefault(answer_key, answer)
				image_id = str(row["imageId"])
				image_path = _resolve_materialized_gqa_image(
					materialized_root,
					active_source_split,
					image_id,
					verified_images,
				)
				yield {
					"sample_id": f"gqa:{active_split}:{row['id']}",
					"dataset": "gqa_balanced",
					"split": active_split,
					"image_id": image_id,
					"image_path": str(image_path),
					"query_text": str(row["question"]).strip(),
					"candidate_text": answer,
					"positive_id": f"answer:{answer_key}",
				}

		stats[split] = _write_rows(
			rows(),
			output_root / split / "part-00000-of-00001.parquet",
		)
		if stats[split]["rows"] != EXPECTED_GQA_ROWS[split]:
			raise RuntimeError(
				f"GQA {split} has {stats[split]['rows']} rows; "
				f"expected {EXPECTED_GQA_ROWS[split]}",
			)
	_write_answer_gallery(output_root, answer_gallery)
	return _finalize_dataset(
		output_root,
		dataset="gqa_balanced",
		split_policy="official_train_val_testdev_balanced",
		stats=stats,
		sources={
			"gqa_root": str(gqa_root),
			"materialized_root": str(materialized_root),
		},
	)


def _iter_clevr_questions(path: Path) -> Iterator[dict[str, Any]]:
	with path.open("rb") as handle:
		yield from ijson.items(handle, "questions.item")


def prepare_clevr(
	*,
	clevr_root: Path,
	output_root: Path,
	seed: int = 42,
) -> dict[str, Any]:
	"""Keep official train full and split labeled validation by image with seed 42."""
	questions_root = clevr_root / "questions"
	images_root = clevr_root / "images"
	validation_image_names = sorted(
		path.name for path in (images_root / "val").glob("*.png") if path.is_file()
	)
	if len(validation_image_names) != EXPECTED_CLEVR_VALIDATION_IMAGES:
		raise RuntimeError(
			f"CLEVR validation has {len(validation_image_names)} images; "
			f"expected {EXPECTED_CLEVR_VALIDATION_IMAGES}",
		)
	validation_images, test_images = split_clevr_validation_images(
		validation_image_names,
		seed=seed,
		validation_images=EXPECTED_CLEVR_VALIDATION_IMAGES // 2,
	)
	answer_gallery: dict[str, str] = {}
	stats: dict[str, Any] = {}

	def train_rows() -> Iterator[dict[str, str]]:
		path = questions_root / "CLEVR_train_questions.json"
		for row in _iter_clevr_questions(path):
			answer = str(row["answer"]).strip()
			answer_key = normalize_answer(answer)
			answer_gallery.setdefault(answer_key, answer)
			filename = str(row["image_filename"])
			yield {
				"sample_id": f"clevr:train:{row['question_index']}",
				"dataset": "clevr",
				"split": "train",
				"image_id": filename,
				"image_path": str(images_root / "train" / filename),
				"query_text": str(row["question"]).strip(),
				"candidate_text": answer,
				"positive_id": f"answer:{answer_key}",
			}

	stats["train"] = _write_rows(
		train_rows(),
		output_root / "train" / "part-00000-of-00001.parquet",
	)
	if stats["train"]["rows"] != EXPECTED_CLEVR_TRAIN_ROWS:
		raise RuntimeError(
			f"CLEVR train has {stats['train']['rows']} rows; "
			f"expected {EXPECTED_CLEVR_TRAIN_ROWS}",
		)

	for split, selected_images in (
		("validation", validation_images),
		("test", test_images),
	):
		def held_out_rows(
			active_split: str = split,
			active_images: set[str] = selected_images,
		) -> Iterator[dict[str, str]]:
			path = questions_root / "CLEVR_val_questions.json"
			for row in _iter_clevr_questions(path):
				filename = str(row["image_filename"])
				if filename not in active_images:
					continue
				answer = str(row["answer"]).strip()
				yield {
					"sample_id": f"clevr:{active_split}:{row['question_index']}",
					"dataset": "clevr",
					"split": active_split,
					"image_id": filename,
					"image_path": str(images_root / "val" / filename),
					"query_text": str(row["question"]).strip(),
					"candidate_text": answer,
					"positive_id": f"answer:{normalize_answer(answer)}",
				}

		stats[split] = _write_rows(
			held_out_rows(),
			output_root / split / "part-00000-of-00001.parquet",
		)
		if stats[split]["unique_images"] != len(selected_images):
			raise RuntimeError(f"CLEVR {split} lost one or more selected images")
	_write_answer_gallery(output_root, answer_gallery)
	return _finalize_dataset(
		output_root,
		dataset="clevr",
		split_policy="official_train_plus_seed42_image_disjoint_halves_of_official_validation",
		stats=stats,
		sources={"clevr_root": str(clevr_root)},
		extra={
			"seed": seed,
			"validation_images": len(validation_images),
			"test_images": len(test_images),
		},
	)


def _write_answer_gallery(output_root: Path, gallery: dict[str, str]) -> None:
	if not gallery:
		raise ValueError("Answer gallery cannot be empty")
	payload = [
		{"positive_id": f"answer:{key}", "text": gallery[key]}
		for key in sorted(gallery)
	]
	(output_root / "answer_gallery.json").write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)


def _finalize_dataset(
	output_root: Path,
	*,
	dataset: str,
	split_policy: str,
	stats: dict[str, Any],
	sources: dict[str, str],
	extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
	if dataset not in BASELINE_DATASETS:
		raise ValueError(dataset)
	config = {
		"dataset": dataset,
		"split_policy": split_policy,
		"manifest_schema": str(MANIFEST_SCHEMA),
		"split_counts": {split: int(value["rows"]) for split, value in stats.items()},
		"sources": sources,
		**(extra or {}),
	}
	(output_root / "config.json").write_text(
		json.dumps(config, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	(output_root / "stats.json").write_text(
		json.dumps(stats, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	(output_root / ".ready").write_text("ready\n", encoding="utf-8")
	return {"root": str(output_root), "config": config, "stats": stats}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset", choices=BASELINE_DATASETS, required=True)
	parser.add_argument(
		"--output-base",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/looped_vl_single_baselines_v1"),
	)
	parser.add_argument(
		"--coco-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/coco"),
	)
	parser.add_argument("--karpathy-json", type=Path)
	parser.add_argument(
		"--gqa-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/gqa_hf_full"),
	)
	parser.add_argument(
		"--gqa-materialized-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/gqa_hf_full/materialized_balanced"),
	)
	parser.add_argument(
		"--clevr-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/clevr/CLEVR_v1.0"),
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	output_root = args.output_base / args.dataset
	if output_root.exists():
		raise FileExistsError(f"Baseline manifest root already exists: {output_root}")
	output_root.mkdir(parents=True)
	if args.dataset == "coco":
		if args.karpathy_json is None:
			raise ValueError("--karpathy-json is required for COCO")
		result = prepare_coco(
			coco_root=args.coco_root,
			karpathy_json=args.karpathy_json,
			output_root=output_root,
		)
	elif args.dataset == "gqa_balanced":
		result = prepare_gqa(
			gqa_root=args.gqa_root,
			materialized_root=args.gqa_materialized_root,
			output_root=output_root,
		)
	else:
		result = prepare_clevr(clevr_root=args.clevr_root, output_root=output_root)
	print(json.dumps(result, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
