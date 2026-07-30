from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from looped_vl.recurrent_data import (
	BASELINE_SPLIT_POLICIES,
	RecurrentAlignedDataset,
	load_aligned_records,
)


def _write_aligned_dataset(
	root: Path,
	*,
	dataset: str,
	split: str = "train",
	row_count: int = 2,
) -> list[dict[str, str]]:
	image_path = root / "images" / f"{dataset}.jpg"
	image_path.parent.mkdir(parents=True)
	Image.new("RGB", (8, 8), color=(20, 40, 60)).save(image_path)
	rows = [
		{
			"sample_id": f"{dataset}:{split}:{index}",
			"dataset": dataset,
			"split": split,
			"image_id": f"image-{index}",
			"image_path": str(image_path),
			"query_text": f"query {index}",
			"candidate_text": "" if dataset == "coco" else f"answer {index}",
			"positive_id": (
				f"image:image-{index}" if dataset == "coco" else f"answer:answer {index}"
			),
		}
		for index in range(row_count)
	]
	split_root = root / split
	split_root.mkdir(parents=True)
	pq.write_table(
		pa.Table.from_pylist(rows),
		split_root / "part-00000-of-00001.parquet",
		row_group_size=1,
	)
	(root / "config.json").write_text(
		json.dumps(
			{
				"dataset": dataset,
				"split_policy": BASELINE_SPLIT_POLICIES[dataset],
				"split_counts": {split: row_count},
			},
		),
		encoding="utf-8",
	)
	(root / "stats.json").write_text(
		json.dumps({split: {"rows": row_count}}),
		encoding="utf-8",
	)
	(root / ".ready").write_text("ready\n", encoding="utf-8")
	return rows


@pytest.mark.parametrize("dataset", ["coco", "gqa_balanced", "clevr"])
def test_recurrent_dataset_reads_the_exact_baseline_manifest(
	tmp_path: Path,
	dataset: str,
) -> None:
	root = tmp_path / dataset
	baseline_rows = _write_aligned_dataset(root, dataset=dataset)

	aligned = RecurrentAlignedDataset(root, "train")
	records = load_aligned_records(root, "train")
	sample = aligned[0]

	assert len(aligned) == len(baseline_rows)
	assert [record["sample_id"] for record in records] == [
		row["sample_id"] for row in baseline_rows
	]
	assert sample.sample_id == baseline_rows[0]["sample_id"]
	assert sample.source == dataset
	assert sample.source_split == "train"
	assert sample.text == baseline_rows[0]["query_text"]
	assert sample.answer == baseline_rows[0]["candidate_text"]
	assert sample.image_path == Path(baseline_rows[0]["image_path"])
	sample.image.close()


def test_recurrent_dataset_rejects_the_legacy_full_official_split(tmp_path: Path) -> None:
	root = tmp_path / "legacy"
	_write_aligned_dataset(root, dataset="coco")
	(root / "config.json").write_text(
		json.dumps(
			{
				"dataset_name": "coco_full_official",
				"official_splits": {"train": "train2017", "validation": "val2017"},
			},
		),
		encoding="utf-8",
	)

	with pytest.raises(ValueError, match="baseline-aligned"):
		RecurrentAlignedDataset(root, "train")


def test_recurrent_dataset_rejects_split_count_drift(tmp_path: Path) -> None:
	root = tmp_path / "coco"
	_write_aligned_dataset(root, dataset="coco")
	(root / "config.json").write_text(
		json.dumps(
			{
				"dataset": "coco",
				"split_policy": BASELINE_SPLIT_POLICIES["coco"],
				"split_counts": {"train": 3},
			},
		),
		encoding="utf-8",
	)

	with pytest.raises(ValueError, match="split count"):
		RecurrentAlignedDataset(root, "train")
