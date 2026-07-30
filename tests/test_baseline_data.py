from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from looped_vl.baseline.data import (
	BaselineManifestDataset,
	baseline_pair_collate,
	normalize_answer,
)
from looped_vl.baseline.prepare import split_clevr_validation_images


def test_clevr_validation_split_is_seeded_exact_and_image_disjoint() -> None:
	image_names = [f"CLEVR_val_{index:06d}.png" for index in range(10)]

	first_validation, first_test = split_clevr_validation_images(
		image_names,
		seed=42,
		validation_images=5,
	)
	second_validation, second_test = split_clevr_validation_images(
		reversed(image_names),
		seed=42,
		validation_images=5,
	)

	assert first_validation == second_validation
	assert first_test == second_test
	assert len(first_validation) == len(first_test) == 5
	assert first_validation.isdisjoint(first_test)
	assert first_validation | first_test == set(image_names)


def test_manifest_dataset_builds_coco_and_vqa_pairs(tmp_path: Path) -> None:
	image_path = tmp_path / "image.jpg"
	Image.new("RGB", (8, 8), color=(20, 40, 60)).save(image_path)
	split_root = tmp_path / "train"
	split_root.mkdir()
	table = pa.Table.from_pylist(
		[
			{
				"sample_id": "coco:1",
				"dataset": "coco",
				"split": "train",
				"image_id": "1",
				"image_path": str(image_path),
				"query_text": "A small image.",
				"candidate_text": "",
				"positive_id": "image:1",
			},
			{
				"sample_id": "gqa:1",
				"dataset": "gqa_balanced",
				"split": "train",
				"image_id": "1",
				"image_path": str(image_path),
				"query_text": "What color is it?",
				"candidate_text": "Blue",
				"positive_id": "answer:blue",
			},
		],
	)
	pq.write_table(table, split_root / "part-00000.parquet", row_group_size=1)
	(tmp_path / "config.json").write_text(
		json.dumps({"dataset": "test", "split_counts": {"train": 2}}),
		encoding="utf-8",
	)

	dataset = BaselineManifestDataset(tmp_path, "train")
	coco = dataset[0]
	gqa = dataset[1]
	batch = baseline_pair_collate([coco, gqa])

	assert coco.query_input == {
		"text": "A small image.",
		"instruction": "Retrieve the image that best matches the caption.",
	}
	assert "image" in coco.candidate_input
	assert "image" in gqa.query_input
	assert gqa.candidate_input == {"text": "Blue"}
	assert batch["positive_ids"] == ["image:1", "answer:blue"]
	assert normalize_answer("  Light   BLUE ") == "light blue"
	for sample in batch["samples"]:
		sample.image.close()
