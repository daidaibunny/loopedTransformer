import pytest

from looped_vl.metrics import (
	COCO_DIRECTIONS,
	METRIC_SCALE,
	MIXTURE_WEIGHTS,
	REQUIRED_DATASETS,
	REQUIRED_RANKING_METRICS,
	aggregate_coco_directions,
	aggregate_mixture_metrics,
	validate_evaluation_report,
)

EXPECTED_METRICS = (
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


def _complete_metrics() -> dict[str, float]:
	return {metric: 50.0 for metric in REQUIRED_RANKING_METRICS}


def _complete_report() -> dict[str, object]:
	return {
		"metric_scale": METRIC_SCALE,
		"mix": _complete_metrics(),
		"datasets": {
			"coco": {
				"aggregate": _complete_metrics(),
				"text_to_image": _complete_metrics(),
				"image_to_text": _complete_metrics(),
			},
			"gqa_balanced": _complete_metrics(),
			"clevr": _complete_metrics(),
		},
	}


def test_required_evaluation_metrics_and_datasets_are_frozen() -> None:
	assert REQUIRED_RANKING_METRICS == EXPECTED_METRICS
	assert METRIC_SCALE == "percentage_0_to_100"
	assert REQUIRED_DATASETS == ("coco", "gqa_balanced", "clevr")
	assert COCO_DIRECTIONS == ("text_to_image", "image_to_text")
	assert MIXTURE_WEIGHTS == {
		"coco": 0.50,
		"gqa_balanced": 0.35,
		"clevr": 0.15,
	}


def test_validate_evaluation_report_accepts_mix_and_every_dataset() -> None:
	validate_evaluation_report(_complete_report())


def test_metric_aggregation_uses_equal_coco_directions_and_mixture_weights() -> None:
	text_to_image = {metric: 20.0 for metric in REQUIRED_RANKING_METRICS}
	image_to_text = {metric: 40.0 for metric in REQUIRED_RANKING_METRICS}
	coco = aggregate_coco_directions(text_to_image, image_to_text)
	gqa = {metric: 50.0 for metric in REQUIRED_RANKING_METRICS}
	clevr = {metric: 70.0 for metric in REQUIRED_RANKING_METRICS}

	mix = aggregate_mixture_metrics(coco, gqa, clevr)

	assert coco["map"] == pytest.approx(30.0)
	assert mix["map"] == pytest.approx(43.0)


def test_validate_evaluation_report_rejects_missing_mix_metric() -> None:
	report = _complete_report()
	del report["mix"]["map"]  # type: ignore[index]

	with pytest.raises(ValueError, match="mix_map"):
		validate_evaluation_report(report)


def test_validate_evaluation_report_rejects_missing_dataset_metric() -> None:
	report = _complete_report()
	del report["datasets"]["gqa_balanced"]["r_at_20"]  # type: ignore[index]

	with pytest.raises(ValueError, match="gqa_balanced_r_at_20"):
		validate_evaluation_report(report)


def test_validate_evaluation_report_rejects_missing_coco_direction() -> None:
	report = _complete_report()
	del report["datasets"]["coco"]["image_to_text"]  # type: ignore[index]

	with pytest.raises(ValueError, match="coco_image_to_text"):
		validate_evaluation_report(report)


@pytest.mark.parametrize("invalid_value", [-0.1, 100.1, float("nan"), "50.0"])
def test_validate_evaluation_report_rejects_invalid_values(invalid_value: object) -> None:
	report = _complete_report()
	report["mix"]["map"] = invalid_value  # type: ignore[index]

	with pytest.raises(ValueError, match="mix_map"):
		validate_evaluation_report(report)
