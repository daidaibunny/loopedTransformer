from __future__ import annotations

import sys

import pytest

from looped_vl.throughput import parse_args, summarize_timings


def test_throughput_defaults_use_10k_validation_and_test_samples(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(sys, "argv", ["looped-vl-throughput"])

	args = parse_args()

	assert args.validation_samples == 10_000
	assert args.test_samples == 10_000


def test_summarize_timings_projects_train_and_full_dataset_runtime() -> None:
	result = summarize_timings(
		batch_size=20,
		batch_total_seconds=[2.0, 2.0, 2.0, 2.0],
		batch_load_seconds=[0.2, 0.2, 0.2, 0.2],
		batch_process_seconds=[1.8, 1.8, 1.8, 1.8],
		train_samples=1_000_000,
		full_samples=1_050_000,
	)

	assert result["measured_samples"] == 80
	assert result["end_to_end_samples_per_second"] == pytest.approx(10.0)
	assert result["process_samples_per_second"] == pytest.approx(100 / 9)
	assert result["batch_time_coefficient_of_variation"] == 0.0
	assert result["projected_train_seconds"] == pytest.approx(100_000.0)
	assert result["projected_full_seconds"] == pytest.approx(105_000.0)
