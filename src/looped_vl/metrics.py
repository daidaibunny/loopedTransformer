"""Required evaluation metrics and aggregation rules for every project experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

METRIC_SCALE = "percentage_0_to_100"
REQUIRED_RANKING_METRICS = (
	"map",
	"p_at_1",
	"p_at_5",
	"p_at_10",
	"p_at_20",
	"r_at_1",
	"r_at_5",
	"r_at_10",
	"r_at_20",
	"mrr",
	"ndcg_at_10",
)
REQUIRED_DATASETS = ("coco", "gqa_balanced", "clevr")
COCO_DIRECTIONS = ("text_to_image", "image_to_text")
MIXTURE_WEIGHTS = MappingProxyType(
	{
		"coco": 0.50,
		"gqa_balanced": 0.35,
		"clevr": 0.15,
	},
)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
	"""Return a mapping or fail with its report location."""
	if not isinstance(value, Mapping):
		raise ValueError(f"{path} must be a mapping")
	return value


def _validate_metric_set(metrics: object, path: str) -> Mapping[str, Any]:
	"""Validate the complete required metric set at one report location."""
	metric_mapping = _require_mapping(metrics, path)
	missing = [metric for metric in REQUIRED_RANKING_METRICS if metric not in metric_mapping]
	if missing:
		raise ValueError(
			"Missing required metrics: "
			+ ", ".join(f"{path}_{metric}" for metric in missing),
		)
	for metric in REQUIRED_RANKING_METRICS:
		value = metric_mapping[metric]
		metric_path = f"{path}_{metric}"
		if isinstance(value, bool) or not isinstance(value, int | float):
			raise ValueError(f"{metric_path} must be numeric")
		if not math.isfinite(value) or not 0.0 <= value <= 100.0:
			raise ValueError(f"{metric_path} must be finite and within [0, 100]")
	return metric_mapping


def aggregate_coco_directions(
	text_to_image: Mapping[str, float],
	image_to_text: Mapping[str, float],
) -> dict[str, float]:
	"""Average COCO text-to-image and image-to-text metrics with equal weights."""
	_validate_metric_set(text_to_image, "coco_text_to_image")
	_validate_metric_set(image_to_text, "coco_image_to_text")
	return {
		metric: (float(text_to_image[metric]) + float(image_to_text[metric])) / 2.0
		for metric in REQUIRED_RANKING_METRICS
	}


def aggregate_mixture_metrics(
	coco: Mapping[str, float],
	gqa_balanced: Mapping[str, float],
	clevr: Mapping[str, float],
) -> dict[str, float]:
	"""Aggregate dataset metrics with the frozen 50:35:15 mixture weights."""
	dataset_metrics = {
		"coco": coco,
		"gqa_balanced": gqa_balanced,
		"clevr": clevr,
	}
	for dataset, metrics in dataset_metrics.items():
		_validate_metric_set(metrics, dataset)
	return {
		metric: sum(
			MIXTURE_WEIGHTS[dataset] * float(dataset_metrics[dataset][metric])
			for dataset in REQUIRED_DATASETS
		)
		for metric in REQUIRED_RANKING_METRICS
	}


def validate_evaluation_report(report: Mapping[str, Any]) -> None:
	"""Fail unless a report contains every required mix and per-dataset metric."""
	if report.get("metric_scale") != METRIC_SCALE:
		raise ValueError(f"metric_scale must be {METRIC_SCALE}")
	_validate_metric_set(report.get("mix"), "mix")
	datasets = _require_mapping(report.get("datasets"), "datasets")
	for dataset in REQUIRED_DATASETS:
		if dataset not in datasets:
			raise ValueError(f"Missing required dataset metrics: {dataset}")

	coco = _require_mapping(datasets["coco"], "coco")
	_validate_metric_set(coco.get("aggregate"), "coco_aggregate")
	for direction in COCO_DIRECTIONS:
		if direction not in coco:
			raise ValueError(f"Missing required dataset metrics: coco_{direction}")
		_validate_metric_set(coco[direction], f"coco_{direction}")
	_validate_metric_set(datasets["gqa_balanced"], "gqa_balanced")
	_validate_metric_set(datasets["clevr"], "clevr")
