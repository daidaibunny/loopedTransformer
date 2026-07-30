import json
from inspect import signature
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from looped_vl.subset_dataset import build_prefix_subset


def test_subset_defaults_define_splits_by_sample_count() -> None:
	parameters = signature(build_prefix_subset).parameters

	assert parameters["train_samples"].default is parameters["train_samples"].empty
	assert parameters["validation_samples"].default == 10_000
	assert parameters["test_samples"].default == 10_000


def _write_split(path: Path, repetitions: int) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	rows = []
	for block in range(repetitions):
		for offset, source in enumerate(
			["coco"] * 10 + ["gqa_balanced"] * 7 + ["clevr"] * 3,
		):
			position = block * 20 + offset
			rows.append(
				{
					"mixture_position": position,
					"sample_id": f"{source}:{position}",
					"source": source,
					"image_id": f"image-{position % 5}",
					"reasoning_depth": position % 3,
				},
			)
	pq.write_table(pa.Table.from_pylist(rows), path)


def test_build_prefix_subset_preserves_ratio_and_validation(tmp_path: Path) -> None:
	source_root = tmp_path / "full"
	output_root = tmp_path / "subset"
	_write_split(source_root / "train/part-00000-of-00001.parquet", repetitions=2)
	_write_split(source_root / "validation/part-00000-of-00001.parquet", repetitions=2)
	(source_root / "config.json").write_text(
		json.dumps({"dataset_name": "full", "seed": 7}),
		encoding="utf-8",
	)

	result = build_prefix_subset(
		source_root,
		output_root,
		train_samples=20,
		validation_samples=20,
		test_samples=20,
	)

	train_table = pq.read_table(output_root / "train/part-00000-of-00001.parquet")
	validation_table = pq.read_table(output_root / "validation/part-00000-of-00001.parquet")
	test_table = pq.read_table(output_root / "test/part-00000-of-00001.parquet")
	config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
	assert train_table.num_rows == 20
	assert validation_table.num_rows == 20
	assert test_table.num_rows == 20
	assert set(validation_table.column("sample_id").to_pylist()).isdisjoint(
		set(test_table.column("sample_id").to_pylist()),
	)
	assert result["train"]["source_counts"] == {
		"coco": 10,
		"gqa_balanced": 7,
		"clevr": 3,
	}
	assert config["train_counts"] == {
		"coco": 10,
		"gqa_balanced": 7,
		"clevr": 3,
	}
	assert (output_root / "checksums.sha256").is_file()
	assert (output_root / ".ready").is_file()
	assert result["test"]["source_counts"] == {
		"coco": 10,
		"gqa_balanced": 7,
		"clevr": 3,
	}
