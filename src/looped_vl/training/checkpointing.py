"""Trainable-only checkpoints with full per-rank reproducibility state."""

from __future__ import annotations

import random
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
) -> tuple[TrainingCursor, dict[str, Any]]:
	"""Restore trainable values, optimizer, scheduler, cursor, and this rank's RNG."""
	payload = torch.load(path, map_location="cpu", weights_only=False)
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
	return TrainingCursor(**payload["cursor"]), payload["metadata"]
