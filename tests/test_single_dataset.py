import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from looped_vl.single_dataset import (
	build_clevr_dataset,
	build_coco_dataset,
	build_gqa_balanced_dataset,
)


def _read_split(root: Path, split: str) -> pa.Table:
	return pa.concat_tables(
		[pq.read_table(path) for path in sorted((root / split).glob("*.parquet"))],
	)


def test_build_coco_dataset_keeps_every_caption_and_reports_unique_images(
	tmp_path: Path,
) -> None:
	source_root = tmp_path / "coco"
	for split in ("train", "val"):
		(source_root / f"{split}2017").mkdir(parents=True)
		(source_root / "annotations").mkdir(exist_ok=True)
		for name in ("a.jpg", "b.jpg"):
			(source_root / f"{split}2017" / name).write_bytes(b"image")
		payload = {
			"images": [
				{"id": 1, "file_name": "a.jpg"},
				{"id": 2, "file_name": "b.jpg"},
			],
			"annotations": [
				{"id": 10, "image_id": 1, "caption": "first"},
				{"id": 11, "image_id": 1, "caption": "second"},
				{"id": 12, "image_id": 2, "caption": "third"},
			],
		}
		(source_root / "annotations" / f"captions_{split}2017.json").write_text(
			json.dumps(payload),
			encoding="utf-8",
		)

	output_root = tmp_path / "coco-full"
	result = build_coco_dataset(source_root, output_root, rows_per_shard=2)

	train = _read_split(output_root, "train")
	validation = _read_split(output_root, "validation")
	assert train.num_rows == 3
	assert validation.num_rows == 3
	assert set(train.column("source").to_pylist()) == {"coco"}
	assert result["splits"]["train"]["sample_rows"] == 3
	assert result["splits"]["train"]["unique_images"] == 2
	assert result["sample_unit"] == "caption_image_pair"
	assert result["official_splits"] == {"train": "train2017", "validation": "val2017"}
	assert not (output_root / "test").exists()
	assert (output_root / ".ready").is_file()


def test_build_gqa_balanced_dataset_keeps_all_answered_rows(tmp_path: Path) -> None:
	source_root = tmp_path / "gqa"
	for split in ("train", "val"):
		table = pa.Table.from_pylist(
			[
				{
					"id": f"{split}-1",
					"imageId": "image-a",
					"question": "what color?",
					"answer": "red",
					"fullAnswer": "It is red.",
					"isBalanced": True,
					"semantic": [{"operation": "select"}],
					"semanticStr": "select",
				},
				{
					"id": f"{split}-2",
					"imageId": "image-a",
					"question": "is it red?",
					"answer": "yes",
					"fullAnswer": "Yes.",
					"isBalanced": True,
					"semantic": [],
					"semanticStr": "",
				},
			],
		)
		path = source_root / f"{split}_balanced_instructions" / "part.parquet"
		path.parent.mkdir(parents=True)
		pq.write_table(table, path)

	output_root = tmp_path / "gqa-full"
	result = build_gqa_balanced_dataset(source_root, output_root, rows_per_shard=1)

	assert _read_split(output_root, "train").num_rows == 2
	assert _read_split(output_root, "validation").num_rows == 2
	assert result["splits"]["train"]["unique_images"] == 1
	assert result["sample_unit"] == "balanced_visual_question"
	assert result["official_splits"] == {"train": "train", "validation": "val"}


def test_build_clevr_dataset_keeps_every_answered_question(tmp_path: Path) -> None:
	source_root = tmp_path / "CLEVR_v1.0"
	for split in ("train", "val"):
		(source_root / "questions").mkdir(parents=True, exist_ok=True)
		(source_root / "images" / split).mkdir(parents=True)
		(source_root / "images" / split / f"CLEVR_{split}_000000.png").write_bytes(b"image")
		payload = {
			"questions": [
				{
					"question_index": 1,
					"image_index": 0,
					"image_filename": f"CLEVR_{split}_000000.png",
					"question": "How many?",
					"answer": "2",
					"program": [{"function": "count"}],
					"question_family_index": 3,
				},
			],
		}
		(source_root / "questions" / f"CLEVR_{split}_questions.json").write_text(
			json.dumps(payload),
			encoding="utf-8",
		)

	output_root = tmp_path / "clevr-full"
	result = build_clevr_dataset(source_root, output_root, rows_per_shard=1)

	assert _read_split(output_root, "train").num_rows == 1
	assert _read_split(output_root, "validation").num_rows == 1
	assert result["splits"]["validation"]["unique_images"] == 1
	assert result["sample_unit"] == "visual_question"
	assert result["official_splits"] == {"train": "train", "validation": "val"}
