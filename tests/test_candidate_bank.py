from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from looped_vl.candidate_bank import (
	CANDIDATE_BANK_SPECS,
	CandidateBankSpec,
	build_candidate_items,
	embedding_shard_ranges,
	load_ready_candidate_bank,
	sha256_file,
	validate_embedding_shard,
	write_candidate_item_manifest,
)
from looped_vl.encode_candidate_banks import CandidateItemDataset


def _write_manifest(root: Path, split: str, rows: list[dict[str, str]]) -> None:
	split_root = root / split
	split_root.mkdir(parents=True)
	pq.write_table(pa.Table.from_pylist(rows), split_root / "part-00000.parquet")


def test_candidate_bank_specs_cover_exactly_eight_immutable_galleries() -> None:
	assert (
		CandidateBankSpec("coco", "train", "image"),
		CandidateBankSpec("coco", "train", "text"),
		CandidateBankSpec("coco", "validation", "image"),
		CandidateBankSpec("coco", "validation", "text"),
		CandidateBankSpec("coco", "test", "image"),
		CandidateBankSpec("coco", "test", "text"),
		CandidateBankSpec("gqa_balanced", "shared", "answer"),
		CandidateBankSpec("clevr", "shared", "answer"),
	) == CANDIDATE_BANK_SPECS


def test_coco_candidate_items_deduplicate_images_and_keep_every_caption(
	tmp_path: Path,
) -> None:
	rows = [
		{
			"sample_id": "coco:train:caption-1",
			"dataset": "coco",
			"split": "train",
			"image_id": "10",
			"image_path": "/images/10.jpg",
			"query_text": "First caption.",
			"candidate_text": "",
			"positive_id": "image:10",
		},
		{
			"sample_id": "coco:train:caption-2",
			"dataset": "coco",
			"split": "train",
			"image_id": "10",
			"image_path": "/images/10.jpg",
			"query_text": "Second caption.",
			"candidate_text": "",
			"positive_id": "image:10",
		},
		{
			"sample_id": "coco:train:caption-3",
			"dataset": "coco",
			"split": "train",
			"image_id": "11",
			"image_path": "/images/11.jpg",
			"query_text": "Third caption.",
			"candidate_text": "",
			"positive_id": "image:11",
		},
	]
	_write_manifest(tmp_path, "train", rows)

	images = build_candidate_items(
		tmp_path,
		CandidateBankSpec("coco", "train", "image"),
	)
	texts = build_candidate_items(
		tmp_path,
		CandidateBankSpec("coco", "train", "text"),
	)

	assert [item.item_id for item in images] == ["image:10", "image:11"]
	assert [item.image_path for item in images] == [
		Path("/images/10.jpg"),
		Path("/images/11.jpg"),
	]
	assert [item.item_id for item in texts] == [
		"coco:train:caption-1",
		"coco:train:caption-2",
		"coco:train:caption-3",
	]
	assert [item.positive_id for item in texts] == ["image:10", "image:10", "image:11"]
	assert [item.text for item in texts] == [
		"First caption.",
		"Second caption.",
		"Third caption.",
	]


@pytest.mark.parametrize("dataset", ["gqa_balanced", "clevr"])
def test_answer_candidate_bank_uses_one_training_gallery_for_every_split(
	tmp_path: Path,
	dataset: str,
) -> None:
	(tmp_path / "answer_gallery.json").write_text(
		json.dumps(
			[
				{"positive_id": "answer:no", "text": "no"},
				{"positive_id": "answer:yes", "text": "yes"},
			],
		),
		encoding="utf-8",
	)

	items = build_candidate_items(
		tmp_path,
		CandidateBankSpec(dataset, "shared", "answer"),
	)

	assert [item.item_id for item in items] == ["answer:no", "answer:yes"]
	assert [item.positive_id for item in items] == ["answer:no", "answer:yes"]
	assert [item.text for item in items] == ["no", "yes"]
	assert all(item.image_path is None for item in items)


def test_embedding_shard_validation_requires_exact_unit_normalized_rows() -> None:
	embeddings = torch.nn.functional.normalize(torch.randn(3, 2048), dim=1).half()

	validated = validate_embedding_shard(
		embeddings,
		expected_rows=3,
		embedding_dimension=2048,
	)

	assert validated == {"rows": 3, "embedding_dimension": 2048, "dtype": "float16"}

	with pytest.raises(ValueError, match="row count"):
		validate_embedding_shard(
			embeddings,
			expected_rows=4,
			embedding_dimension=2048,
		)
	with pytest.raises(ValueError, match="unit normalized"):
		validate_embedding_shard(
			embeddings * 2,
			expected_rows=3,
			embedding_dimension=2048,
		)
	bad = embeddings.clone()
	bad[0, 0] = torch.nan
	with pytest.raises(ValueError, match="non-finite"):
		validate_embedding_shard(
			bad,
			expected_rows=3,
			embedding_dimension=2048,
		)


def test_embedding_shard_ranges_cover_every_item_once() -> None:
	assert embedding_shard_ranges(10, 4) == ((0, 4), (4, 8), (8, 10))
	with pytest.raises(ValueError, match="item_count"):
		embedding_shard_ranges(0, 4)
	with pytest.raises(ValueError, match="shard_rows"):
		embedding_shard_ranges(10, 0)


def test_ready_bank_loader_rejects_changed_model_and_validates_every_shard(
	tmp_path: Path,
) -> None:
	root = tmp_path / "bank"
	items_root = root / "items"
	shards_root = root / "embedding_shards"
	items_root.mkdir(parents=True)
	shards_root.mkdir()
	items_path = items_root / "part-00000.parquet"
	pq.write_table(
		pa.Table.from_pylist(
			[
				{
					"item_index": 0,
					"item_id": "answer:yes",
					"positive_id": "answer:yes",
					"input_kind": "text",
					"text": "yes",
					"image_path": None,
				},
			],
		),
		items_path,
	)
	embeddings = torch.nn.functional.normalize(torch.randn(1, 2048), dim=1).half()
	shard_path = shards_root / "part-00000.pt"
	torch.save({"start": 0, "end": 1, "embeddings": embeddings}, shard_path)
	manifest = {
		"version": "frozen_qwen3vl_candidate_bank_v1",
		"spec": {
			"dataset": "gqa_balanced",
			"split": "shared",
			"gallery": "answer",
		},
		"model": {"checkpoint_sha256": "model-hash"},
		"embedding_dimension": 2048,
		"items": {
			"path": "items/part-00000.parquet",
			"rows": 1,
			"sha256": sha256_file(items_path),
		},
		"embedding_shards": [
			{
				"path": "embedding_shards/part-00000.pt",
				"start": 0,
				"end": 1,
				"sha256": sha256_file(shard_path),
			},
		],
	}
	manifest_path = root / "bank_manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
	(root / "READY").write_text(f"{sha256_file(manifest_path)}\n", encoding="utf-8")
	spec = CandidateBankSpec("gqa_balanced", "shared", "answer")

	loaded = load_ready_candidate_bank(
		root,
		expected_spec=spec,
		expected_model_sha256="model-hash",
	)

	assert loaded == manifest
	with pytest.raises(ValueError, match="model checksum"):
		load_ready_candidate_bank(
			root,
			expected_spec=spec,
			expected_model_sha256="different-model",
		)
	(root / "READY").write_text("changed\n", encoding="utf-8")
	with pytest.raises(ValueError, match="READY checksum"):
		load_ready_candidate_bank(
			root,
			expected_spec=spec,
			expected_model_sha256="model-hash",
		)


def test_candidate_item_manifest_is_indexed_and_checksum_stable(tmp_path: Path) -> None:
	dataset_root = tmp_path / "dataset"
	(dataset_root / "config.json").parent.mkdir(parents=True)
	(dataset_root / "config.json").write_text("{}\n", encoding="utf-8")
	(dataset_root / "answer_gallery.json").write_text(
		json.dumps(
			[
				{"positive_id": "answer:no", "text": "no"},
				{"positive_id": "answer:yes", "text": "yes"},
			],
		),
		encoding="utf-8",
	)
	output_path = tmp_path / "items" / "part-00000.parquet"

	result = write_candidate_item_manifest(
		dataset_root=dataset_root,
		spec=CandidateBankSpec("clevr", "shared", "answer"),
		output_path=output_path,
		batch_rows=1,
	)

	assert result == {
		"rows": 2,
		"unique_item_ids": 2,
		"unique_positive_ids": 2,
		"input_kind": "text",
		"sha256": sha256_file(output_path),
	}
	assert pq.read_table(output_path).column("item_index").to_pylist() == [0, 1]


def test_candidate_item_dataset_reads_text_and_image_rows_with_global_indices(
	tmp_path: Path,
) -> None:
	image_path = tmp_path / "candidate.png"
	Image.new("L", (3, 2), color=127).save(image_path)
	items_root = tmp_path / "items"
	items_root.mkdir()
	pq.write_table(
		pa.Table.from_pylist(
			[
				{
					"item_index": 0,
					"item_id": "unused",
					"positive_id": "unused",
					"input_kind": "text",
					"text": "unused",
					"image_path": None,
				},
				{
					"item_index": 1,
					"item_id": "answer:yes",
					"positive_id": "answer:yes",
					"input_kind": "text",
					"text": "yes",
					"image_path": None,
				},
				{
					"item_index": 2,
					"item_id": "image:10",
					"positive_id": "image:10",
					"input_kind": "image",
					"text": None,
					"image_path": str(image_path),
				},
			],
		),
		items_root / "part-00000.parquet",
		row_group_size=1,
	)
	dataset = CandidateItemDataset(items_root, start=1, end=3)

	text_sample = dataset[0]
	image_sample = dataset[-1]

	assert len(dataset) == 2
	assert text_sample.item_index == 1
	assert text_sample.model_input == {"text": "yes"}
	assert text_sample.image is None
	assert image_sample.item_index == 2
	assert image_sample.image is image_sample.model_input["image"]
	assert image_sample.image is not None
	assert image_sample.image.mode == "RGB"
	assert image_sample.image.size == (3, 2)
	image_sample.image.close()
	with pytest.raises(IndexError):
		_ = dataset[2]
