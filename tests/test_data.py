from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from looped_vl.data import (
	GQAImageResolver,
	LoopedVLMixtureDataset,
	mixture_collate,
	select_source_balanced_indices,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.new("RGB", (8, 8), color=color).save(path)


def make_row(
	position: int,
	source: str,
	image_path: Path | None,
	image_id: str,
	image_config: str = "",
) -> dict[str, object]:
	storage = "filesystem" if image_path is not None else "hf_parquet"
	return {
		"mixture_position": position,
		"selection_key": position,
		"sample_id": f"{source}:train:{position}",
		"source_sample_id": str(position),
		"source": source,
		"source_split": "train",
		"task_type": (
			"image_text_matching" if source == "coco" else "visual_question_answering"
		),
		"image_storage": storage,
		"image_path": str(image_path) if image_path is not None else "",
		"image_config": image_config,
		"image_id": image_id,
		"text": f"question or caption {position}",
		"answer": "yes" if source != "coco" else "",
		"full_answer": "Yes." if source != "coco" else "",
		"reasoning_trace_json": "[]",
		"reasoning_depth": 0,
		"metadata_json": "{}",
	}


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
	data_root = tmp_path / "mixture"
	train_root = data_root / "train"
	train_root.mkdir(parents=True)
	gqa_root = tmp_path / "gqa_materialized"

	coco_image = tmp_path / "coco.jpg"
	clevr_image = tmp_path / "clevr.png"
	gqa_image = gqa_root / "train" / "gqa-1.jpg"
	write_image(coco_image, (255, 0, 0))
	write_image(clevr_image, (0, 255, 0))
	write_image(gqa_image, (0, 0, 255))

	rows = []
	for position in range(20):
		if position < 10:
			rows.append(make_row(position, "coco", coco_image, f"coco-{position}"))
		elif position < 17:
			rows.append(
				make_row(
					position,
					"gqa_balanced",
					None,
					"gqa-1",
					"train_balanced_images",
				)
			)
		else:
			rows.append(make_row(position, "clevr", clevr_image, f"clevr-{position}"))
	table = pa.Table.from_pylist(rows)
	pq.write_table(table.slice(0, 12), train_root / "part-00000.parquet", row_group_size=4)
	pq.write_table(table.slice(12), train_root / "part-00001.parquet", row_group_size=3)
	return data_root, gqa_root


def test_dataset_reads_across_shards_and_resolves_every_image_backend(
	tiny_dataset: tuple[Path, Path],
) -> None:
	data_root, gqa_root = tiny_dataset
	dataset = LoopedVLMixtureDataset(data_root, "train", gqa_root)

	assert len(dataset) == 20
	assert dataset[0].source == "coco"
	assert dataset[0].image.getpixel((0, 0))[0] > 200
	assert dataset[10].source == "gqa_balanced"
	assert dataset[10].image.getpixel((0, 0))[2] > 200
	assert dataset[19].source == "clevr"
	assert dataset[19].image.getpixel((0, 0))[1] > 200


def test_source_balanced_indices_and_collate_preserve_one_of_each_source(
	tiny_dataset: tuple[Path, Path],
) -> None:
	data_root, gqa_root = tiny_dataset
	dataset = LoopedVLMixtureDataset(data_root, "train", gqa_root)
	indices = select_source_balanced_indices(dataset, per_source=1)
	loader = DataLoader(
		dataset,
		batch_size=3,
		sampler=indices,
		collate_fn=mixture_collate,
		num_workers=0,
	)
	batch = next(iter(loader))

	assert batch["sources"] == ["coco", "gqa_balanced", "clevr"]
	assert len(batch["model_inputs"]) == 3
	assert all("image" in model_input for model_input in batch["model_inputs"])
	assert batch["model_inputs"][0]["text"].startswith("question or caption")


def test_missing_gqa_materialized_image_fails_with_actionable_path(tmp_path: Path) -> None:
	resolver = GQAImageResolver(tmp_path)

	with pytest.raises(FileNotFoundError, match="gqa-404.jpg"):
		resolver.resolve("train", "gqa-404")
