from pathlib import Path

from looped_vl.training.resume_throughput_pipeline import (
	TrainingBenchmark,
	build_frozen_evaluation_command,
	build_training_command,
	select_best_training_benchmark,
)


def test_training_benchmark_command_resumes_exact_checkpoint_with_batch_four(
	tmp_path: Path,
) -> None:
	command = build_training_command(
		torchrun=Path("/env/bin/torchrun"),
		output_dir=tmp_path / "batch4",
		resume_checkpoint=tmp_path / "stage1_step000500.pt",
		per_device_batch_size=4,
		resume_per_device_batch_size=1,
		code_commit="a" * 40,
		max_additional_optimizer_steps=3,
		num_workers=8,
		prefetch_factor=4,
		end_stage=1,
	)

	assert command[:4] == [
		"/env/bin/torchrun",
		"--standalone",
		"--nproc_per_node=2",
		"-m",
	]
	assert command[command.index("--per-device-batch-size") + 1] == "4"
	assert command[command.index("--resume-per-device-batch-size") + 1] == "1"
	assert command[command.index("--max-additional-optimizer-steps") + 1] == "3"
	assert command[command.index("--end-stage") + 1] == "1"


def test_frozen_full_evaluation_command_does_not_apply_smoke_limit(tmp_path: Path) -> None:
	command = build_frozen_evaluation_command(
		torchrun=Path("/env/bin/torchrun"),
		output_dir=tmp_path / "full",
		batch_size=128,
		num_workers=8,
		prefetch_factor=4,
		score_batch_size=1024,
		max_test_rows=0,
	)

	assert "--max-test-rows" not in command
	assert command[command.index("--batch-size") + 1] == "128"
	assert command[command.index("--expected-world-size") + 1] == "2"


def test_training_benchmark_selection_prefers_throughput_with_memory_headroom() -> None:
	benchmarks = [
		TrainingBenchmark(
			per_device_batch_size=4,
			samples_per_second=11.0,
			peak_memory_bytes=40_000_000_000,
			output_dir=Path("/batch4"),
		),
		TrainingBenchmark(
			per_device_batch_size=8,
			samples_per_second=14.0,
			peak_memory_bytes=75_000_000_000,
			output_dir=Path("/batch8"),
		),
		TrainingBenchmark(
			per_device_batch_size=16,
			samples_per_second=13.0,
			peak_memory_bytes=60_000_000_000,
			output_dir=Path("/batch16"),
		),
	]

	selected = select_best_training_benchmark(
		benchmarks,
		memory_limit_bytes=72_000_000_000,
	)

	assert selected.per_device_batch_size == 16
	assert selected.samples_per_second == 13.0
