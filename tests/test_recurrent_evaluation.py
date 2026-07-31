import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from looped_vl import evaluate_recurrent
from looped_vl.evaluate_recurrent import (
	_initialize_evaluation_distributed,
	_primary_final_pass_metrics,
	_summarize_evaluation_runtime,
	build_loop_metric_series,
	load_recurrent_inference_checkpoint,
	parse_args,
)
from looped_vl.models.config import pure_recurrent_result_identity


class _TinyInferenceModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.latent_slots = nn.Parameter(torch.zeros(1, 2))
		self.eos_delta = nn.Parameter(torch.zeros(1, 2))
		self.recurrent_connector = nn.Linear(2, 2)
		self.late_fusion = nn.Linear(2, 2)
		self.base_embedding_model = nn.Module()
		self.base_embedding_model.backbone_weight = nn.Parameter(torch.zeros(2, 2))
		self.warmup_embedding_head = nn.Linear(2, 2)


def _checkpoint_state(model: nn.Module) -> dict[str, torch.Tensor]:
	return {
		f"encoder.{name}": torch.full_like(parameter, 3)
		for name, parameter in model.named_parameters()
	}


def _checkpoint_metadata() -> dict[str, object]:
	return {
		**pure_recurrent_result_identity(),
		"model_checkpoint_sha256": "base-hash",
		"model_config": {"num_total_loop_passes": 4},
	}


def test_inference_checkpoint_loads_only_pure_recurrent_parameters(
	tmp_path: Path,
) -> None:
	model = _TinyInferenceModel()
	state = _checkpoint_state(model)
	path = tmp_path / "checkpoint.pt"
	torch.save(
		{
			"format_version": 1,
			"trainable_parameter_state": state,
			"metadata": _checkpoint_metadata(),
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
		if not name.startswith(("warmup_", "base_embedding_model")):
			assert torch.equal(parameter, torch.full_like(parameter, 3))
	assert torch.equal(
		model.base_embedding_model.backbone_weight,
		torch.zeros_like(model.base_embedding_model.backbone_weight),
	)


def test_inference_checkpoint_rejects_lora_parameters(tmp_path: Path) -> None:
	model = _TinyInferenceModel()
	state = _checkpoint_state(model)
	state["encoder.base_embedding_model.layers.12.self_attn.q_proj.lora_a.weight"] = (
		torch.zeros(2, 2)
	)
	path = tmp_path / "checkpoint.pt"
	torch.save(
		{
			"format_version": 1,
			"trainable_parameter_state": state,
			"metadata": _checkpoint_metadata(),
		},
		path,
	)

	with pytest.raises(ValueError, match="LoRA"):
		load_recurrent_inference_checkpoint(
			model,
			path,
			expected_base_hash="base-hash",
			expected_model_config={"num_total_loop_passes": 4},
		)


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
				**_checkpoint_metadata(),
				"model_checkpoint_sha256": "wrong-hash",
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


def test_inference_checkpoint_rejects_missing_no_lora_result_identity(
	tmp_path: Path,
) -> None:
	model = _TinyInferenceModel()
	path = tmp_path / "checkpoint.pt"
	torch.save(
		{
			"format_version": 1,
			"trainable_parameter_state": _checkpoint_state(model),
			"metadata": {
				"model_checkpoint_sha256": "base-hash",
				"model_config": {"num_total_loop_passes": 4},
			},
		},
		path,
	)

	with pytest.raises(ValueError, match="pure recurrent result identity"):
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


def test_recurrent_evaluation_uses_cpu_collectives(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	events: list[tuple[str, object]] = []
	monkeypatch.setenv("LOCAL_RANK", "3")
	monkeypatch.setattr(
		evaluate_recurrent.torch.cuda,
		"set_device",
		lambda rank: events.append(("set_device", rank)),
	)
	monkeypatch.setattr(
		evaluate_recurrent.dist,
		"init_process_group",
		lambda **kwargs: events.append(("init_process_group", kwargs)),
	)
	monkeypatch.setattr(evaluate_recurrent.dist, "get_rank", lambda: 2)
	monkeypatch.setattr(evaluate_recurrent.dist, "get_world_size", lambda: 8)

	rank, world_size, local_rank, device = _initialize_evaluation_distributed(8)

	assert events == [
		("set_device", 3),
		("init_process_group", {"backend": "gloo"}),
	]
	assert (rank, world_size, local_rank, device) == (
		2,
		8,
		3,
		torch.device("cuda", 3),
	)


def test_recurrent_runtime_summary_reports_global_throughput_and_peak_memory() -> None:
	summary = _summarize_evaluation_runtime(
		runtimes=[
			{
				"encoding_seconds": 10.0,
				"encoded_items": 50,
				"peak_gpu_memory_bytes": 12_000,
			},
			{
				"encoding_seconds": 12.5,
				"encoded_items": 50,
				"peak_gpu_memory_bytes": 15_000,
			},
		],
		total_encoded_items=100,
		total_seconds=20.0,
	)

	assert summary == {
		"total_seconds": 20.0,
		"encoding_wall_seconds": 12.5,
		"encoded_items": 100,
		"encoding_items_per_second": 8.0,
		"peak_gpu_memory_bytes": 15_000,
	}


def test_final_pass_primary_metrics_use_coco_equal_direction_mean() -> None:
	metrics = {"map": 40.0}
	assert _primary_final_pass_metrics(
		source="coco",
		loop_metrics={"4": {"aggregate": {"metrics": metrics}}},
		final_pass=4,
	) == metrics
	assert _primary_final_pass_metrics(
		source="gqa_balanced",
		loop_metrics={"4": {"metrics": metrics}},
		final_pass=4,
	) == metrics


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
