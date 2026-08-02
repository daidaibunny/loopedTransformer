from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from looped_vl.candidate_bank import CandidateBankSpec, sha256_file
from looped_vl.query_recurrent.candidate_store import (
	CandidateReference,
	CandidateStoreCollection,
	ImmutableCandidateStore,
)
from looped_vl.query_recurrent.data import QueryOnlyManifestDataset


def _write_ready_bank(
	root: Path,
	spec: CandidateBankSpec,
	items: list[tuple[str, str]],
	model_hash: str = "model-hash",
) -> None:
	bank_root = root / spec.relative_path
	items_root = bank_root / "items"
	shards_root = bank_root / "embedding_shards"
	items_root.mkdir(parents=True)
	shards_root.mkdir()
	items_path = items_root / "part-00000.parquet"
	pq.write_table(
		pa.Table.from_pylist(
			[
				{
					"item_index": index,
					"item_id": item_id,
					"positive_id": positive_id,
				}
				for index, (item_id, positive_id) in enumerate(items)
			],
		),
		items_path,
	)
	embeddings = torch.nn.functional.normalize(torch.randn(len(items), 2048), dim=1).half()
	shard_path = shards_root / "part-00000.pt"
	torch.save(
		{"start": 0, "end": len(items), "embeddings": embeddings},
		shard_path,
	)
	manifest = {
		"version": "frozen_qwen3vl_candidate_bank_v1",
		"spec": {
			"dataset": spec.dataset,
			"split": spec.split,
			"gallery": spec.gallery,
		},
		"model": {"checkpoint_sha256": model_hash},
		"embedding_dimension": 2048,
		"items": {
			"path": "items/part-00000.parquet",
			"rows": len(items),
			"sha256": sha256_file(items_path),
		},
		"embedding_shards": [
			{
				"path": "embedding_shards/part-00000.pt",
				"start": 0,
				"end": len(items),
				"sha256": sha256_file(shard_path),
			},
		],
	}
	manifest_path = bank_root / "bank_manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
	(bank_root / "READY").write_text(f"{sha256_file(manifest_path)}\n", encoding="utf-8")


def _write_coco_rows(root: Path, image_path: Path) -> None:
	(root / "train").mkdir(parents=True)
	pq.write_table(
		pa.Table.from_pylist(
			[
				{
					"sample_id": "caption:0",
					"dataset": "coco",
					"split": "train",
					"image_id": "10",
					"image_path": str(image_path),
					"query_text": "A caption.",
					"candidate_text": "",
					"positive_id": "image:10",
				},
				{
					"sample_id": "caption:1",
					"dataset": "coco",
					"split": "train",
					"image_id": "10",
					"image_path": str(image_path),
					"query_text": "Another caption.",
					"candidate_text": "",
					"positive_id": "image:10",
				},
			],
		),
		root / "train" / "part-00000.parquet",
	)


def test_candidate_store_validates_and_preserves_mixed_gallery_order(tmp_path: Path) -> None:
	image_spec = CandidateBankSpec("coco", "train", "image")
	text_spec = CandidateBankSpec("coco", "train", "text")
	_write_ready_bank(tmp_path, image_spec, [("image:10", "image:10")])
	_write_ready_bank(
		tmp_path,
		text_spec,
		[("caption:0", "image:10"), ("caption:1", "image:10")],
	)
	collection = CandidateStoreCollection(
		candidate_root=tmp_path,
		model_checkpoint_sha256="model-hash",
		validate_checksums=True,
	)
	references = [
		CandidateReference(text_spec, "caption:1", "image:10"),
		CandidateReference(image_spec, "image:10", "image:10"),
		CandidateReference(text_spec, "caption:0", "image:10"),
	]

	resolved = collection.lookup(references, device=torch.device("cpu"))

	text_store = collection.get(text_spec)
	image_store = collection.get(image_spec)
	assert torch.equal(resolved[0], text_store.embeddings[1])
	assert torch.equal(resolved[1], image_store.embeddings[0])
	assert torch.equal(resolved[2], text_store.embeddings[0])
	assert resolved.requires_grad is False


def test_query_dataset_never_decodes_candidate_only_coco_image(tmp_path: Path) -> None:
	image_path = tmp_path / "image.png"
	Image.new("RGB", (4, 3), color=(1, 2, 3)).save(image_path)
	_write_coco_rows(tmp_path, image_path)
	dataset = QueryOnlyManifestDataset(tmp_path, "coco", "train")

	text_query = dataset[0]
	image_query = dataset[1]

	assert text_query.direction == "text_to_image"
	assert text_query.image is None
	assert "image" not in text_query.query_input
	assert text_query.candidate_reference.spec.gallery == "image"
	assert text_query.candidate_reference.item_id == "image:10"
	assert image_query.direction == "image_to_text"
	assert image_query.image is image_query.query_input["image"]
	assert image_query.candidate_reference.spec.gallery == "text"
	assert image_query.candidate_reference.item_id == "caption:1"
	if image_query.image is not None:
		image_query.image.close()


def test_candidate_store_rejects_wrong_base_model(tmp_path: Path) -> None:
	spec = CandidateBankSpec("clevr", "shared", "answer")
	_write_ready_bank(tmp_path, spec, [("answer:yes", "answer:yes")])

	with pytest.raises(ValueError, match="model checksum"):
		ImmutableCandidateStore(
			candidate_root=tmp_path,
			spec=spec,
			model_checkpoint_sha256="wrong",
		)


def test_candidate_store_mines_same_gallery_non_positive_neighbors(tmp_path: Path) -> None:
	spec = CandidateBankSpec("coco", "train", "text")
	_write_ready_bank(
		tmp_path,
		spec,
		[("positive", "image:1"), ("hard", "image:2"), ("easy", "image:3")],
	)
	store = ImmutableCandidateStore(
		candidate_root=tmp_path,
		spec=spec,
		model_checkpoint_sha256="model-hash",
	)
	values = torch.zeros(3, 2048, dtype=torch.float16)
	values[0, 0] = 1.0
	values[1, :2] = torch.tensor([0.9, 0.1], dtype=torch.float16)
	values[2, 0] = -1.0
	values[1] = torch.nn.functional.normalize(values[1].float(), dim=0).half()
	store._shard_tensors = [values]
	query = torch.zeros(1, 2048)
	query[0, 0] = 1.0

	embeddings, item_indices = store.mine_hard_negatives(
		query,
		positive_ids=["image:1"],
		count=1,
		device=torch.device("cpu"),
	)

	assert item_indices.tolist() == [[1]]
	assert torch.equal(embeddings[0, 0], values[1])
