from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from looped_vl.baseline.data import (
	BaselineManifestDataset,
	baseline_pair_collate,
	build_coco_retrieval_inputs,
	count_coco_retrieval_directions,
	normalize_answer,
)
from looped_vl.baseline.prepare import (
	_resolve_materialized_gqa_image,
	split_clevr_validation_images,
)


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


def test_gqa_image_existence_is_checked_once_per_unique_image(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	verified_images: set[tuple[str, str]] = set()
	call_count = 0

	def fake_is_file(path: Path) -> bool:
		nonlocal call_count
		call_count += 1
		return True

	monkeypatch.setattr(Path, "is_file", fake_is_file)
	first = _resolve_materialized_gqa_image(
		tmp_path,
		"train",
		"image-1",
		verified_images,
	)
	second = _resolve_materialized_gqa_image(
		tmp_path,
		"train",
		"image-1",
		verified_images,
	)

	assert first == second
	assert call_count == 1
	assert verified_images == {("train", "image-1")}


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
				"sample_id": "coco:2",
				"dataset": "coco",
				"split": "train",
				"image_id": "2",
				"image_path": str(image_path),
				"query_text": "A second image.",
				"candidate_text": "",
				"positive_id": "image:2",
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
		json.dumps({"dataset": "test", "split_counts": {"train": 3}}),
		encoding="utf-8",
	)

	dataset = BaselineManifestDataset(tmp_path, "train")
	coco = dataset[0]
	reverse_coco = dataset[1]
	gqa = dataset[2]
	batch = baseline_pair_collate([coco, reverse_coco, gqa])

	assert coco.direction == "text_to_image"
	assert coco.query_input == {
		"text": "A small image.",
		"instruction": "Retrieve the image that best matches the caption.",
	}
	assert "image" in coco.candidate_input
	assert reverse_coco.direction == "image_to_text"
	assert "image" in reverse_coco.query_input
	assert reverse_coco.candidate_input == {"text": "A second image."}
	assert "image" in gqa.query_input
	assert gqa.candidate_input == {"text": "Blue"}
	assert batch["directions"] == [
		"text_to_image",
		"image_to_text",
		"visual_question_answering",
	]
	assert batch["positive_ids"] == ["image:1", "image:2", "answer:blue"]
	assert normalize_answer("  Light   BLUE ") == "light blue"
	for sample in batch["samples"]:
		sample.image.close()


def test_coco_direction_helper_is_balanced_and_keeps_the_sample_count() -> None:
	image = object()
	pairs = [
		build_coco_retrieval_inputs(
			text=f"caption {position}",
			image=image,
			position=position,
		)
		for position in range(10)
	]

	assert len(pairs) == 10
	assert [pair.direction for pair in pairs].count("text_to_image") == 5
	assert [pair.direction for pair in pairs].count("image_to_text") == 5
	assert count_coco_retrieval_directions(10) == {
		"text_to_image": 5,
		"image_to_text": 5,
	}
	assert count_coco_retrieval_directions(11) == {
		"text_to_image": 6,
		"image_to_text": 5,
	}
