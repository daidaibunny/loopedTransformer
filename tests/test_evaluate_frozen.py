import pytest
import torch

from looped_vl.evaluate_frozen import (
	build_answer_gallery,
	build_coco_relevance,
	compute_ranking_metrics,
)


def test_compute_ranking_metrics_handles_single_positive_queries() -> None:
	targets = torch.eye(20, dtype=torch.float32)
	queries = torch.stack(
		[
			targets[0],
			0.8 * targets[1] + 0.7 * targets[2] + 0.6 * targets[0],
		],
	)

	metrics = compute_ranking_metrics(
		queries,
		targets,
		positive_indices=[(0,), (0,)],
		device=torch.device("cpu"),
		score_batch_size=2,
	)

	assert metrics["map"] == pytest.approx(100 * (1 + 1 / 3) / 2)
	assert metrics["p_at_1"] == pytest.approx(50.0)
	assert metrics["p_at_5"] == pytest.approx(20.0)
	assert metrics["p_at_10"] == pytest.approx(10.0)
	assert metrics["p_at_20"] == pytest.approx(5.0)
	assert metrics["r_at_1"] == pytest.approx(50.0)
	assert metrics["r_at_5"] == pytest.approx(100.0)
	assert metrics["r_at_10"] == pytest.approx(100.0)
	assert metrics["r_at_20"] == pytest.approx(100.0)
	assert metrics["mrr"] == pytest.approx(100 * (1 + 1 / 3) / 2)
	assert metrics["ndcg_at_10"] == pytest.approx(75.0)


def test_compute_ranking_metrics_uses_all_coco_positives() -> None:
	targets = torch.eye(20, dtype=torch.float32)
	query = 0.9 * targets[0] + 0.8 * targets[1] + 0.7 * targets[2]

	metrics = compute_ranking_metrics(
		query.unsqueeze(0),
		targets,
		positive_indices=[(0, 2)],
		device=torch.device("cpu"),
		score_batch_size=1,
	)

	assert metrics["map"] == pytest.approx(100 * (1 + 2 / 3) / 2)
	assert metrics["p_at_1"] == pytest.approx(100.0)
	assert metrics["p_at_5"] == pytest.approx(40.0)
	assert metrics["r_at_1"] == pytest.approx(50.0)
	assert metrics["r_at_5"] == pytest.approx(100.0)


def test_answer_gallery_normalizes_and_deduplicates_answers() -> None:
	answers, positive_indices = build_answer_gallery([" Yes ", "yes", "NO"])

	assert answers == ["no", "yes"]
	assert positive_indices == [(1,), (1,), (0,)]


def test_coco_relevance_keeps_both_retrieval_directions() -> None:
	rows = [
		{"image_id": "image-a", "text": "caption one"},
		{"image_id": "image-a", "text": "caption two"},
		{"image_id": "image-b", "text": "caption three"},
	]

	result = build_coco_relevance(rows)

	assert result["image_ids"] == ["image-a", "image-b"]
	assert result["text_to_image_positive_indices"] == [(0,), (0,), (1,)]
	assert result["image_to_text_positive_indices"] == [(0, 1), (2,)]
