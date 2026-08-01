from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from looped_vl.candidate_bank import CANDIDATE_BANK_SPECS, sha256_file
from looped_vl.query_recurrent.launch import _queue_command, validate_gpu_inventory
from looped_vl.query_recurrent.queue import (
	FORMAL_QUERY_RECURRENT_RUNS,
	_next_evaluation_output,
	build_evaluation_command,
	build_training_command,
	validate_all_candidate_banks,
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


def test_formal_queue_contains_the_locked_eight_experiments() -> None:
	assert [run.name for run in FORMAL_QUERY_RECURRENT_RUNS] == [
		"coco_k8_r1_fixed",
		"coco_k8_r4_fixed",
		"coco_k8_r4_dynamic",
		"coco_k1_r4_dynamic",
		"coco_k4_r4_dynamic",
		"coco_k8_r4_dynamic_layer28",
		"gqa_k8_r4_dynamic",
		"clevr_k8_r4_dynamic",
	]
	assert FORMAL_QUERY_RECURRENT_RUNS[5].history_layers == (28,)
	assert sum(run.dataset == "coco" for run in FORMAL_QUERY_RECURRENT_RUNS) == 6


def test_commands_lock_no_lora_one_epoch_no_validation_and_every_pass_test(
	tmp_path: Path,
) -> None:
	args = _args(tmp_path)
	run = FORMAL_QUERY_RECURRENT_RUNS[2]
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
