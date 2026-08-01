from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from looped_vl.baseline.frozen_queue import (
	FrozenEvaluationRun,
	build_frozen_queue_commands,
	parse_v100_names,
)
from looped_vl.baseline.model import BASELINE_LORA_LAST_FOUR_DECODER_LAYERS
from looped_vl.baseline.queue import (
	BaselineRun,
	build_frozen_evaluation_command,
	build_training_command,
	run_queue,
)
from looped_vl.baseline.smoke_search import _project_pythonpath
from looped_vl.training.wait_and_launch import parse_gpu_snapshot


def test_gpu_snapshot_can_require_all_eight_v100_devices() -> None:
	output = "\n".join(f"{index}, 0, 0" for index in range(8))

	snapshot = parse_gpu_snapshot(
		output,
		compute_process_count=0,
		expected_indexes=tuple(range(8)),
	)

	assert snapshot.is_idle is True
	assert len(snapshot.gpus) == 8


def test_training_command_keeps_dataset_specific_parallel_parameters(tmp_path: Path) -> None:
	run = BaselineRun(
		dataset="clevr",
		per_device_batch_size=32,
		gradient_accumulation_steps=1,
		num_workers=6,
		gradient_checkpointing=False,
	)

	command = build_training_command(
		run,
		project_root=tmp_path,
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
	)

	assert command[1:3] == ["-m", "torch.distributed.run"]
	assert command[command.index("--dataset") + 1] == "clevr"
	assert command[command.index("--per-device-batch-size") + 1] == "32"
	assert command[command.index("--gradient-accumulation-steps") + 1] == "1"
	assert command[command.index("--expected-contrastive-global-batch-size") + 1] == "256"
	assert command[command.index("--num-workers") + 1] == "6"
	assert command[command.index("--expected-world-size") + 1] == "8"
	assert command[command.index("--visual-length-buckets") + 1] == "3"
	assert command[command.index("--min-visual-bucket-size") + 1] == "8"
	assert "--no-gradient-checkpointing" in command


def test_training_command_can_select_only_the_last_four_decoder_layers(
	tmp_path: Path,
) -> None:
	run = BaselineRun("coco", 32, 1, 4)

	command = build_training_command(
		run,
		project_root=tmp_path,
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
		lora_decoder_layer_indices=BASELINE_LORA_LAST_FOUR_DECODER_LAYERS,
	)

	assert command[command.index("--lora-decoder-layer-indices") + 1] == "24,25,26,27"


def test_zero_idle_seconds_submits_all_three_runs_without_an_idle_gate(
	tmp_path: Path,
	monkeypatch,
) -> None:
	def reject_idle_wait(**_kwargs: object) -> None:
		raise AssertionError("zero-second direct submission must not call the idle gate")

	monkeypatch.setattr("looped_vl.baseline.queue.wait_for_idle_window", reject_idle_wait)
	monkeypatch.setattr(
		"looped_vl.baseline.queue.subprocess.run",
		lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
	)
	args = argparse.Namespace(
		output_root=tmp_path / "outputs",
		world_size=8,
		required_idle_seconds=0.0,
		poll_seconds=5.0,
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		project_root=tmp_path,
		checkpoint_every=100,
		max_checkpoints=1,
		lora_scope="last_4_decoder_layers",
	)
	runs = [
		BaselineRun("coco", 32, 1, 4),
		BaselineRun("gqa_balanced", 32, 1, 4),
		BaselineRun("clevr", 32, 1, 4),
	]

	run_queue(args, runs)

	assert not list((tmp_path / "outputs").glob("*_idle_gate.jsonl"))


def test_frozen_evaluation_command_uses_all_eight_v100_ranks(tmp_path: Path) -> None:
	command = build_frozen_evaluation_command(
		"gqa_balanced",
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
		batch_size=32,
		num_workers=4,
	)

	assert command[1:3] == ["-m", "torch.distributed.run"]
	assert "--nproc_per_node=8" in command
	assert command[command.index("--dataset") + 1] == "gqa_balanced"
	assert command[command.index("--dataset-root") + 1] == str(
		tmp_path / "datasets" / "gqa_balanced",
	)
	assert command[command.index("--expected-world-size") + 1] == "8"
	assert command[command.index("--batch-size") + 1] == "32"
	assert command[command.index("--num-workers") + 1] == "4"
	assert command[command.index("--visual-length-buckets") + 1] == "3"
	assert command[command.index("--min-visual-bucket-size") + 1] == "8"
	assert "--adapter-root" not in command


def test_frozen_queue_runs_each_latest_single_dataset_split_once(tmp_path: Path) -> None:
	commands = build_frozen_queue_commands(
		(
			FrozenEvaluationRun("coco", 32, 4),
			FrozenEvaluationRun("gqa_balanced", 32, 4),
			FrozenEvaluationRun("clevr", 32, 4),
		),
		dataset_root=tmp_path / "datasets",
		model_root=tmp_path / "model",
		output_root=tmp_path / "outputs",
		world_size=8,
	)

	assert [
		command[command.index("--dataset") + 1] for command in commands
	] == ["coco", "gqa_balanced", "clevr"]
	assert [
		command[command.index("--output-dir") + 1] for command in commands
	] == [
		str(tmp_path / "outputs" / "coco"),
		str(tmp_path / "outputs" / "gqa_balanced"),
		str(tmp_path / "outputs" / "clevr"),
	]


def test_frozen_queue_requires_exactly_eight_v100_devices() -> None:
	names = parse_v100_names(
		"\n".join(f"{index}, Tesla V100-SXM2-32GB" for index in range(8)),
		world_size=8,
	)

	assert names == tuple("Tesla V100-SXM2-32GB" for _ in range(8))


def test_smoke_search_preserves_the_selected_runtime_pythonpath(tmp_path: Path) -> None:
	pythonpath = _project_pythonpath(
		tmp_path,
		"/usr/local/lib/python3.10/dist-packages",
	)

	assert pythonpath == (
		f"{tmp_path / 'src'}:/usr/local/lib/python3.10/dist-packages"
	)
