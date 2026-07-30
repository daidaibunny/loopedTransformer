from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from looped_vl.baseline.train import (
	_accumulate_logging_metrics,
	_finalize_logging_metrics,
	_validate_parallel_batch_sizes,
	parse_args,
)


def test_parallel_batch_validation_requires_a_true_256_pair_contrastive_batch() -> None:
	_validate_parallel_batch_sizes(
		per_device_batch_size=32,
		world_size=8,
		gradient_accumulation_steps=1,
		expected_contrastive_global_batch_size=256,
	)

	with pytest.raises(ValueError, match="Contrastive global batch"):
		_validate_parallel_batch_sizes(
			per_device_batch_size=8,
			world_size=8,
			gradient_accumulation_steps=4,
			expected_contrastive_global_batch_size=256,
		)


def test_logging_metrics_average_every_microbatch_by_sample_count() -> None:
	accumulator: dict[str, torch.Tensor] = {}
	_accumulate_logging_metrics(
		accumulator,
		{
			"loss": torch.tensor(2.0),
			"query_norm": torch.tensor(4.0),
			"candidate_norm": torch.tensor(3.0),
		},
		sample_count=3,
	)
	_accumulate_logging_metrics(
		accumulator,
		{
			"loss": torch.tensor(6.0),
			"query_norm": torch.tensor(8.0),
			"candidate_norm": torch.tensor(7.0),
		},
		sample_count=1,
	)

	assert _finalize_logging_metrics(accumulator, sample_count=4) == {
		"loss": pytest.approx(3.0),
		"query_norm": pytest.approx(5.0),
		"candidate_norm": pytest.approx(4.0),
	}


def test_baseline_cli_defaults_to_the_true_256_pair_batch_and_four_checkpoints(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setattr(
		sys,
		"argv",
		[
			"baseline-train",
			"--dataset",
			"coco",
			"--dataset-root",
			str(tmp_path / "coco"),
			"--output-dir",
			str(tmp_path / "output"),
		],
	)

	args = parse_args()

	assert args.per_device_batch_size == 32
	assert args.gradient_accumulation_steps == 1
	assert args.expected_contrastive_global_batch_size == 256
	assert args.checkpoint_every == 100
	assert args.max_checkpoints == 4
