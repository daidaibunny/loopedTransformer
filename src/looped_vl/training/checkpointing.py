"""Trainable-only checkpoints with full per-rank reproducibility state."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn


class GradientScaler(Protocol):
	"""Checkpoint-facing subset of the CUDA gradient scaler API."""

	def state_dict(self) -> dict[str, Any]:
		"""Return the scaler state."""
		...

	def load_state_dict(self, state_dict: dict[str, Any]) -> None:
		"""Restore the scaler state."""
		...


@dataclass(frozen=True)
class TrainingCursor:
	"""Exact position needed to resume the deterministic training stream."""

	stage: int
	global_step: int
	sampler_epoch: int
	batch_in_epoch: int
	gradient_accumulation_step: int
	processed_samples: int = 0


def prepare_training_output_directory(
	output_dir: str | Path,
	*,
	resume_checkpoint: str | Path | None,
) -> str:
	"""Create a fresh output or validate an in-place checkpoint resume."""
	output = Path(output_dir)
	if resume_checkpoint is None:
		if output.exists():
			raise FileExistsError(f"Training output already exists: {output}")
		(output / "checkpoints").mkdir(parents=True)
		return "fresh"
	checkpoint = Path(resume_checkpoint)
	if not output.is_dir():
		raise FileNotFoundError(f"Resume output directory does not exist: {output}")
	if not checkpoint.resolve().is_relative_to(output.resolve()):
		raise ValueError("Resume checkpoint must belong to the output directory")
	if not checkpoint.is_file():
		raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
	return "resume"


def validate_checkpoint_metadata(
	metadata: dict[str, Any],
	*,
	expected: dict[str, Any],
) -> None:
	"""Reject a checkpoint whose fixed data, model, or runtime identity changed."""
	for key, value in expected.items():
		if metadata.get(key) != value:
			raise ValueError(
				f"Resume checkpoint {key} mismatch: {metadata.get(key)!r} != {value!r}",
			)


def truncate_metric_log(path: str | Path, *, maximum_global_step: int) -> int:
	"""Atomically remove log records written after the resumed checkpoint."""
	if maximum_global_step < 0:
		raise ValueError("maximum_global_step cannot be negative")
	target = Path(path)
	if not target.is_file():
		return 0
	kept_lines: list[str] = []
	removed = 0
	for line_number, line in enumerate(
		target.read_text(encoding="utf-8").splitlines(),
		start=1,
	):
		if not line.strip():
			continue
		record = json.loads(line)
		if "global_step" not in record:
			raise ValueError(f"Metric log line {line_number} has no global_step")
		if int(record["global_step"]) <= maximum_global_step:
			kept_lines.append(line)
		else:
			removed += 1
	temporary = target.with_suffix(target.suffix + ".resume.tmp")
	if temporary.exists():
		raise FileExistsError(f"Temporary metric log already exists: {temporary}")
	temporary.write_text(
		"".join(f"{line}\n" for line in kept_lines),
		encoding="utf-8",
	)
	temporary.replace(target)
	return removed


def rebase_training_cursor_batch_size(
	cursor: TrainingCursor,
	*,
	source_per_device_batch_size: int,
	target_per_device_batch_size: int,
) -> TrainingCursor:
	"""Preserve the next local sample when changing the per-device batch size."""
	if source_per_device_batch_size <= 0 or target_per_device_batch_size <= 0:
		raise ValueError("Per-device batch sizes must be positive")
	if cursor.gradient_accumulation_step != 0:
		raise ValueError("Batch-size rebase requires an optimizer accumulation boundary")
	local_samples_consumed = cursor.batch_in_epoch * source_per_device_batch_size
	if local_samples_consumed % target_per_device_batch_size:
		raise ValueError(
			"Consumed local sample count is not divisible by the target batch size",
		)
	return TrainingCursor(
		stage=cursor.stage,
		global_step=cursor.global_step,
		sampler_epoch=cursor.sampler_epoch,
		batch_in_epoch=local_samples_consumed // target_per_device_batch_size,
		gradient_accumulation_step=0,
		processed_samples=cursor.processed_samples,
	)


def capture_rng_state() -> dict[str, Any]:
	"""Capture Python, NumPy, CPU Torch, and every visible CUDA RNG state."""
	return {
		"python": random.getstate(),
		"numpy": np.random.get_state(),
		"torch_cpu": torch.get_rng_state(),
		"torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
	}


def restore_rng_state(state: dict[str, Any]) -> None:
	"""Restore every RNG stream captured by :func:`capture_rng_state`."""
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])
	torch.set_rng_state(state["torch_cpu"])
	if torch.cuda.is_available() and state["torch_cuda_all"]:
		torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
	"""Sort named step checkpoints deterministically, then fall back to modification time."""
	match = re.search(r"step(\d+)", path.stem)
	step = int(match.group(1)) if match else -1
	stage_match = re.search(r"stage(\d+)", path.stem)
	stage = int(stage_match.group(1)) if stage_match else 0
	return stage, step, path.name


def prune_training_checkpoints(
	checkpoint_root: str | Path,
	*,
	max_checkpoints: int,
) -> list[Path]:
	"""Keep only the latest training checkpoint file."""
	if max_checkpoints != 1:
		raise ValueError("max_checkpoints must be exactly 1")
	root = Path(checkpoint_root)
	checkpoints = sorted(root.glob("*.pt"), key=_checkpoint_sort_key)
	removed = checkpoints[:-max_checkpoints]
	for path in removed:
		path.unlink()
	return removed


def publish_latest_training_checkpoint(
	checkpoint_path: str | Path,
	cursor: TrainingCursor,
	*,
	max_checkpoints: int,
) -> list[Path]:
	"""Atomically publish the newest checkpoint pointer, then remove older files."""
	checkpoint = Path(checkpoint_path)
	if not checkpoint.is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
	pointer = checkpoint.parent.parent / "latest_checkpoint.json"
	temporary = pointer.with_suffix(pointer.suffix + ".tmp")
	if temporary.exists():
		raise FileExistsError(f"Temporary checkpoint pointer already exists: {temporary}")
	temporary.write_text(
		json.dumps(
			{"path": str(checkpoint), "cursor": asdict(cursor)},
			indent=2,
			sort_keys=True,
		)
		+ "\n",
		encoding="utf-8",
	)
	temporary.replace(pointer)
	return prune_training_checkpoints(
		checkpoint.parent,
		max_checkpoints=max_checkpoints,
	)


def _trainable_parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
	return {
		name: parameter.detach().cpu()
		for name, parameter in model.named_parameters()
		if parameter.requires_grad
	}


def save_training_checkpoint(
	path: str | Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	cursor: TrainingCursor,
	rank_rng_states: list[dict[str, Any]],
	metadata: dict[str, Any],
	gradient_scaler: GradientScaler | None = None,
) -> None:
	"""Save all trainable values, optimizer state, cursor, and rank RNG streams."""
	target = Path(path)
	if target.exists():
		raise FileExistsError(f"Checkpoint already exists: {target}")
	target.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"format_version": 1,
		"trainable_parameter_state": _trainable_parameter_state(model),
		"optimizer_state": optimizer.state_dict(),
		"scheduler_state": scheduler.state_dict(),
		"cursor": asdict(cursor),
		"rank_rng_states": rank_rng_states,
		"metadata": metadata,
		"gradient_scaler_state": (
			gradient_scaler.state_dict() if gradient_scaler is not None else None
		),
	}
	temporary = target.with_suffix(target.suffix + ".tmp")
	if temporary.exists():
		raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
	torch.save(payload, temporary)
	temporary.replace(target)


def load_training_checkpoint(
	path: str | Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	rank: int,
	gradient_scaler: GradientScaler | None = None,
	expected_training_protocol: str | None = None,
) -> tuple[TrainingCursor, dict[str, Any]]:
	"""Restore trainable values, optimizer, scheduler, cursor, and this rank's RNG."""
	payload = torch.load(path, map_location="cpu", weights_only=False)
	metadata = payload["metadata"]
	if (
		expected_training_protocol is not None
		and metadata.get("training_protocol") != expected_training_protocol
	):
		raise ValueError(
			"Checkpoint training protocol does not match the active single-stage run",
		)
	model_parameters = dict(model.named_parameters())
	for name, value in payload["trainable_parameter_state"].items():
		if name not in model_parameters:
			raise KeyError(f"Checkpoint parameter is missing from model: {name}")
		model_parameters[name].data.copy_(
			value.to(
				device=model_parameters[name].device,
				dtype=model_parameters[name].dtype,
			),
		)
	optimizer.load_state_dict(payload["optimizer_state"])
	scheduler.load_state_dict(payload["scheduler_state"])
	gradient_scaler_state = payload.get("gradient_scaler_state")
	if gradient_scaler is not None and gradient_scaler_state is not None:
		gradient_scaler.load_state_dict(gradient_scaler_state)
	rank_rng_states = payload["rank_rng_states"]
	if rank >= len(rank_rng_states):
		raise ValueError(f"Checkpoint does not contain RNG state for rank {rank}")
	restore_rng_state(rank_rng_states[rank])
	return TrainingCursor(**payload["cursor"]), metadata
