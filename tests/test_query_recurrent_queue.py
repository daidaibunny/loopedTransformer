from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from looped_vl.candidate_bank import CANDIDATE_BANK_SPECS, sha256_file
from looped_vl.query_recurrent.launch import _queue_command, validate_gpu_inventory
from looped_vl.query_recurrent.queue import (
	FORMAL_QUERY_RECURRENT_RUNS,
	_child_process_environment,
	_next_evaluation_output,
	_queue_manifests_match,
	build_evaluation_command,
	build_training_command,
	validate_all_candidate_banks,
)
from looped_vl.query_recurrent.train import (
	_lower_loaded_gradient_scale,
	_resolve_resume_source_commit,
)


def _args(tmp_path: Path) -> SimpleNamespace:
	return SimpleNamespace(
		world_size=8,
		per_device_batch_size=32,
		evaluation_batch_size=32,
		num_workers=4,
		checkpoint_every=100,
		smoke_rows=512,
		smoke_steps=2,
		project_root=tmp_path / "project",
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		candidate_root=tmp_path / "banks",
		output_root=tmp_path / "outputs",
	)


def test_formal_queue_contains_only_the_focused_coco_v2_controls() -> None:
	assert [run.name for run in FORMAL_QUERY_RECURRENT_RUNS] == [
		"coco_v2_k8_r1_fixed",
		"coco_v2_k8_r4_fixed",
	]
	assert all(run.dataset == "coco" for run in FORMAL_QUERY_RECURRENT_RUNS)
	assert all(run.exit_mode == "fixed" for run in FORMAL_QUERY_RECURRENT_RUNS)


def test_commands_lock_no_lora_one_epoch_no_validation_and_every_pass_test(
	tmp_path: Path,
) -> None:
	args = _args(tmp_path)
	run = FORMAL_QUERY_RECURRENT_RUNS[1]
	train = build_training_command(run, args=args, output_dir=tmp_path / "train")
	test = build_evaluation_command(
		run,
		args=args,
		training_output=tmp_path / "train",
		evaluation_output=tmp_path / "test",
	)
	test_text = " ".join(test).lower()

	assert all("lora" not in argument for argument in train if argument.startswith("--"))
	assert all("validation" not in argument for argument in train if argument.startswith("--"))
	assert train[train.index("--epochs") + 1] == "1"
	assert train[train.index("--max-checkpoints") + 1] == "1"
	assert train[train.index("--per-device-batch-size") + 1] == "32"
	assert train[train.index("--hard-negative-count") + 1] == "32"
	assert "looped_vl.query_recurrent.evaluate" in test_text
	assert test[test.index("--recurrent-checkpoint") + 1].endswith(
		"query_recurrent_model.pt",
	)


def test_all_eight_ready_markers_are_required(tmp_path: Path) -> None:
	for spec in CANDIDATE_BANK_SPECS:
		bank_root = tmp_path / spec.relative_path
		bank_root.mkdir(parents=True)
		manifest_path = bank_root / "bank_manifest.json"
		manifest_path.write_text(json.dumps({"spec": spec.key}), encoding="utf-8")
		(bank_root / "READY").write_text(f"{sha256_file(manifest_path)}\n", encoding="utf-8")

	identities = validate_all_candidate_banks(tmp_path)

	assert set(identities) == {spec.key for spec in CANDIDATE_BANK_SPECS}


def test_failed_test_is_never_overwritten_and_gets_a_retry_directory(tmp_path: Path) -> None:
	run_root = tmp_path / "run"
	(run_root / "test").mkdir(parents=True)
	(run_root / "test" / "status.json").write_text(
		json.dumps({"status": "failed"}),
		encoding="utf-8",
	)

	next_output = _next_evaluation_output(run_root)

	assert next_output == run_root / "test_retry_01"


def test_launcher_requires_exactly_eight_v100s_and_preserves_batch_32(tmp_path: Path) -> None:
	args = _args(tmp_path)
	command = _queue_command(args)

	assert validate_gpu_inventory("\n".join(["Tesla V100-SXM2-32GB"] * 8), expected_count=8)
	with pytest.raises(RuntimeError, match="Expected 8 V100"):
		validate_gpu_inventory("\n".join(["Tesla V100-SXM2-32GB"] * 7), expected_count=8)
	assert command[command.index("--per-device-batch-size") + 1] == "32"
	assert command[command.index("--world-size") + 1] == "8"


def test_recovery_requires_the_exact_checkpoint_source_commit() -> None:
	assert (
		_resolve_resume_source_commit(
			current_git_commit="new",
			checkpoint_git_commit="new",
			authorized_source_git_commit=None,
		)
		== "new"
	)
	with pytest.raises(ValueError, match="source Git commit"):
		_resolve_resume_source_commit(
			current_git_commit="new",
			checkpoint_git_commit="old",
			authorized_source_git_commit=None,
		)
	assert (
		_resolve_resume_source_commit(
			current_git_commit="new",
			checkpoint_git_commit="old",
			authorized_source_git_commit="old",
		)
		== "old"
	)


def test_recovery_can_only_lower_the_loaded_fp16_gradient_scale() -> None:
	scaler = torch.amp.GradScaler("cpu", init_scale=4096.0)

	previous_scale = _lower_loaded_gradient_scale(scaler, new_scale=2048.0)

	assert previous_scale == 4096.0
	assert scaler.state_dict()["scale"] == 2048.0
	assert scaler.state_dict()["_growth_tracker"] == 0
	with pytest.raises(ValueError, match="strictly below"):
		_lower_loaded_gradient_scale(scaler, new_scale=2048.0)


def test_resume_command_records_the_authorized_commit_and_lower_scale(
	tmp_path: Path,
) -> None:
	args = _args(tmp_path)
	args.resume_source_git_commit = "old-commit"
	args.resume_gradient_scale = 2048.0
	checkpoint = tmp_path / "step001000.pt"

	command = build_training_command(
		FORMAL_QUERY_RECURRENT_RUNS[-1],
		args=args,
		output_dir=tmp_path / "train",
		resume_checkpoint=checkpoint,
	)

	assert command[command.index("--resume-checkpoint") + 1] == str(checkpoint)
	assert command[command.index("--resume-source-git-commit") + 1] == "old-commit"
	assert command[command.index("--resume-gradient-scale") + 1] == "2048.0"


def test_existing_queue_manifest_normalizes_json_tuple_round_trip() -> None:
	current = {"runs": [{"history_layers": (7, 14, 21, 28)}]}
	written_and_loaded = {"runs": [{"history_layers": [7, 14, 21, 28]}]}

	assert _queue_manifests_match(written_and_loaded, current)


def test_distributed_children_use_the_compatible_protobuf_runtime(tmp_path: Path) -> None:
	environment = _child_process_environment(_args(tmp_path))

	assert environment["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] == "python"
	assert environment["TOKENIZERS_PARALLELISM"] == "false"
	assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
	assert environment["PYTHONPATH"] == str(tmp_path / "project" / "src")
