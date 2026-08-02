from __future__ import annotations

from types import SimpleNamespace

import pytest

from looped_vl.baseline.evaluate import (
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
