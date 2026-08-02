from __future__ import annotations

import json
from pathlib import Path

import pytest

from looped_vl.query_recurrent.monitor import (
	_latest_evaluation_path,
	expected_stage_paths,
	read_stage_statuses,
	validate_stage_order,
)


def test_monitor_locks_recurrent_before_three_lora_controls(tmp_path: Path) -> None:
	stages = expected_stage_paths(
		output_root=tmp_path / "recurrent",
		control_output_root=tmp_path / "controls",
	)

	assert [name for name, _path in stages] == [
		"smoke_coco_v11_p4_r4_final_mean",
		"coco_v11_p4_r4_final_mean_train",
		"coco_v11_p4_r4_final_mean_test",
		"coco_query_only_last4_lora_frozen_candidates_train",
		"coco_query_only_last4_lora_frozen_candidates_test",
		"gqa_balanced_query_only_last4_lora_frozen_candidates_train",
		"gqa_balanced_query_only_last4_lora_frozen_candidates_test",
		"clevr_query_only_last4_lora_frozen_candidates_train",
		"clevr_query_only_last4_lora_frozen_candidates_test",
	]


def test_monitor_uses_explicit_existing_coco_control_root(tmp_path: Path) -> None:
	existing_coco_root = tmp_path / "historical" / "coco"
	stages = expected_stage_paths(
		output_root=tmp_path / "recurrent",
		control_output_root=tmp_path / "controls",
		existing_coco_control_run_root=existing_coco_root,
	)

	assert stages[3][1] == existing_coco_root / "train"
	assert stages[4][1] == existing_coco_root / "test"


def test_monitor_follows_the_latest_evaluation_retry(tmp_path: Path) -> None:
	run_root = tmp_path / "run"
	(run_root / "test").mkdir(parents=True)
	(run_root / "test_retry_01").mkdir()

	assert _latest_evaluation_path(run_root) == run_root / "test_retry_01"
	(run_root / "test_retry_02").mkdir()
	(run_root / "latest_test.json").write_text(
		json.dumps({"path": str(run_root / "test_retry_01")}),
		encoding="utf-8",
	)
	assert _latest_evaluation_path(run_root) == run_root / "test_retry_01"


def test_monitor_rejects_evaluation_pointer_outside_run_root(tmp_path: Path) -> None:
	run_root = tmp_path / "run"
	run_root.mkdir()
	(run_root / "latest_test.json").write_text(
		json.dumps({"path": str(tmp_path / "other" / "test")}),
		encoding="utf-8",
	)

	with pytest.raises(ValueError, match="escapes"):
		_latest_evaluation_path(run_root)


def test_monitor_rejects_later_stage_before_recurrent_test_passes(tmp_path: Path) -> None:
	stages = expected_stage_paths(
		output_root=tmp_path / "recurrent",
		control_output_root=tmp_path / "controls",
	)
	for _name, path in stages[:2]:
		path.mkdir(parents=True)
		(path / "status.json").write_text(json.dumps({"status": "passed"}))
	later_path = stages[3][1]
	later_path.mkdir(parents=True)
	(later_path / "status.json").write_text(json.dumps({"status": "training"}))

	statuses = read_stage_statuses(stages)

	with pytest.raises(RuntimeError, match="before prior stage"):
		validate_stage_order(statuses)


def test_monitor_rejects_more_than_one_rolling_checkpoint(tmp_path: Path) -> None:
	stages = expected_stage_paths(
		output_root=tmp_path / "recurrent",
		control_output_root=tmp_path / "controls",
	)
	training_path = stages[1][1]
	(training_path / "checkpoints").mkdir(parents=True)
	(training_path / "checkpoints" / "step000100.pt").touch()
	(training_path / "checkpoints" / "step000200.pt").touch()

	with pytest.raises(RuntimeError, match="rolling checkpoints"):
		read_stage_statuses(stages)
