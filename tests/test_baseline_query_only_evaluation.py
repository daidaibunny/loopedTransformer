from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from looped_vl.baseline.evaluate import (
	_build_groups,
	_evaluation_group_names,
	_validate_candidate_store_order,
)


def test_query_only_lora_evaluation_encodes_queries_but_not_candidates() -> None:
	assert _evaluation_group_names("coco", query_only=True) == (
		"text_query",
		"image_query",
	)
	assert _evaluation_group_names("gqa_balanced", query_only=True) == ("query",)
	assert _evaluation_group_names("coco", query_only=False) == (
		"text_query",
		"image_target",
		"image_query",
		"text_target",
	)


def test_query_only_candidate_order_must_match_the_test_gallery() -> None:
	store = SimpleNamespace(item_ids=("a", "b"), spec=SimpleNamespace(key="test-bank"))
	items = [SimpleNamespace(item_id="a"), SimpleNamespace(item_id="b")]

	_validate_candidate_store_order(store, items)
	with pytest.raises(ValueError, match="ordering"):
		_validate_candidate_store_order(store, list(reversed(items)))


def test_coco_image_gallery_uses_candidate_bank_positive_identifiers(tmp_path: Path) -> None:
	rows = [
		{
			"sample_id": "caption:1",
			"image_id": "391895",
			"positive_id": "image:391895",
			"query_text": "first caption",
			"image_path": str(tmp_path / "image.jpg"),
		},
		{
			"sample_id": "caption:2",
			"image_id": "391895",
			"positive_id": "image:391895",
			"query_text": "second caption",
			"image_path": str(tmp_path / "image.jpg"),
		},
	]

	groups, relevance = _build_groups("coco", tmp_path, rows)
	store = SimpleNamespace(
		item_ids=("image:391895",),
		spec=SimpleNamespace(key="coco/test/image"),
	)

	_validate_candidate_store_order(store, groups["image_target"])
	assert groups["image_query"][0].item_id == "image:391895"
	assert relevance["text_to_image"] == [(0,), (0,)]
	assert relevance["image_to_text"] == [(0, 1)]
