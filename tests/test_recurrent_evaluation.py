import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from looped_vl.evaluate_recurrent import (
	build_loop_metric_series,
	load_recurrent_inference_checkpoint,
	parse_args,
)


class _TinyInferenceModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.latent_slots = nn.Parameter(torch.zeros(1, 2))
		self.eos_delta = nn.Parameter(torch.zeros(1, 2))
		self.recurrent_connector = nn.Linear(2, 2)
		self.late_fusion = nn.Linear(2, 2)
		self.base_embedding_model = nn.Module()
		self.base_embedding_model.lora_a = nn.Linear(2, 1, bias=False)
		self.warmup_embedding_head = nn.Linear(2, 2)


def _checkpoint_state(model: nn.Module) -> dict[str, torch.Tensor]:
	return {
		f"encoder.{name}": torch.full_like(parameter, 3)
		for name, parameter in model.named_parameters()
	}


def test_inference_checkpoint_loads_only_recurrent_and_lora_parameters(
	tmp_path: Path,
) -> None:
	model = _TinyInferenceModel()
	state = _checkpoint_state(model)
	path = tmp_path / "checkpoint.pt"
	torch.save(
		{
			"format_version": 1,
			"trainable_parameter_state": state,
			"metadata": {
				"model_checkpoint_sha256": "base-hash",
				"model_config": {"num_total_loop_passes": 4},
			},
		},
		path,
	)

	metadata = load_recurrent_inference_checkpoint(
		model,
		path,
		expected_base_hash="base-hash",
		expected_model_config={"num_total_loop_passes": 4},
	)

	assert metadata["model_checkpoint_sha256"] == "base-hash"
	for name, parameter in model.named_parameters():
		if not name.startswith("warmup_"):
			assert torch.equal(parameter, torch.full_like(parameter, 3))


def test_inference_checkpoint_rejects_wrong_base_hash_and_missing_parameter(
	tmp_path: Path,
) -> None:
	model = _TinyInferenceModel()
	state = _checkpoint_state(model)
	state.pop("encoder.eos_delta")
	path = tmp_path / "checkpoint.pt"
	torch.save(
		{
			"format_version": 1,
			"trainable_parameter_state": state,
			"metadata": {
				"model_checkpoint_sha256": "wrong-hash",
				"model_config": {"num_total_loop_passes": 4},
			},
		},
		path,
	)

	with pytest.raises(ValueError, match="base checkpoint hash"):
		load_recurrent_inference_checkpoint(
			model,
			path,
			expected_base_hash="base-hash",
			expected_model_config={"num_total_loop_passes": 4},
		)

	payload = torch.load(path, weights_only=False)
	payload["metadata"]["model_checkpoint_sha256"] = "base-hash"
	torch.save(payload, path)
	with pytest.raises(ValueError, match="Missing inference parameters"):
		load_recurrent_inference_checkpoint(
			model,
			path,
			expected_base_hash="base-hash",
			expected_model_config={"num_total_loop_passes": 4},
		)


def test_loop_metric_series_reports_previous_and_r1_percentage_point_deltas() -> None:
	series = build_loop_metric_series(
		{
			1: {"map": 20.0, "p_at_1": 30.0},
			2: {"map": 21.5, "p_at_1": 29.0},
			3: {"map": 24.0, "p_at_1": 31.0},
		},
	)

	assert series["1"]["delta_from_previous_percentage_points"] == {
		"map": 0.0,
		"p_at_1": 0.0,
	}
	assert series["2"]["delta_from_previous_percentage_points"] == {
		"map": pytest.approx(1.5),
		"p_at_1": pytest.approx(-1.0),
	}
	assert series["3"]["delta_from_r1_percentage_points"] == {
		"map": pytest.approx(4.0),
		"p_at_1": pytest.approx(1.0),
	}


def test_recurrent_evaluation_uses_aligned_manifest_without_legacy_gqa_root(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	monkeypatch.setattr(
		sys,
		"argv",
		[
			"evaluate-recurrent",
			"--source",
			"gqa_balanced",
			"--dataset-root",
			str(tmp_path / "gqa_balanced"),
			"--model-root",
			str(tmp_path / "model"),
			"--master-slot-path",
			str(tmp_path / "slots.pt"),
			"--checkpoint",
			str(tmp_path / "checkpoint.pt"),
			"--output-dir",
			str(tmp_path / "evaluation"),
			"--expected-world-size",
			"8",
		],
	)

	args = parse_args()

	assert args.dataset_root == tmp_path / "gqa_balanced"
	assert args.split == "test"
	assert args.runtime_precision == "fp16"
	assert not hasattr(args, "gqa_materialized_root")
