from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from looped_vl.materialize_gqa import materialize_split


def jpeg_bytes(color: tuple[int, int, int]) -> bytes:
	buffer = BytesIO()
	Image.new("RGB", (8, 8), color=color).save(buffer, format="JPEG")
	return buffer.getvalue()


def write_gqa_parquet(root: Path, split: str = "train") -> None:
	config_root = root / f"{split}_balanced_images"
	config_root.mkdir(parents=True)
	table = pa.Table.from_pylist(
		[
			{"id": "gqa-1", "image": {"bytes": jpeg_bytes((255, 0, 0)), "path": None}},
			{"id": "gqa-2", "image": {"bytes": jpeg_bytes((0, 255, 0)), "path": None}},
		]
	)
	pq.write_table(table, config_root / "train-00000.parquet", row_group_size=1)


def test_materialize_split_writes_and_revalidates_exact_jpeg_bytes(tmp_path: Path) -> None:
	source_root = tmp_path / "gqa"
	output_root = tmp_path / "materialized"
	write_gqa_parquet(source_root)

	first = materialize_split(source_root, output_root, "train")
	second = materialize_split(source_root, output_root, "train")

	assert first["expected_images"] == 2
	assert first["written_images"] == 2
	assert second["written_images"] == 0
	assert second["reused_images"] == 2
	assert (output_root / "train/gqa-1.jpg").read_bytes().startswith(b"\xff\xd8")
	assert (output_root / ".train_ready").is_file()


def test_materialize_split_rejects_existing_file_with_wrong_size(tmp_path: Path) -> None:
	source_root = tmp_path / "gqa"
	output_root = tmp_path / "materialized"
	write_gqa_parquet(source_root)
	bad_path = output_root / "train/gqa-1.jpg"
	bad_path.parent.mkdir(parents=True)
	bad_path.write_bytes(b"wrong")

	with pytest.raises(RuntimeError, match="size mismatch"):
		materialize_split(source_root, output_root, "train")


def test_materialize_split_supports_labeled_testdev_images(tmp_path: Path) -> None:
	source_root = tmp_path / "gqa"
	output_root = tmp_path / "materialized"
	write_gqa_parquet(source_root, split="testdev")

	stats = materialize_split(source_root, output_root, "testdev")

	assert stats["source_config"] == "testdev_balanced_images"
	assert (output_root / "testdev/gqa-1.jpg").is_file()
