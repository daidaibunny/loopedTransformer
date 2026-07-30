"""Strict parameter groups for warm-start and joint optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

WARM_START_PREFIXES = (
	"latent_slots",
	"recurrent_connector.",
	"warmup_embedding_head.",
)
JOINT_ONLY_PREFIXES = (
	"eos_delta",
	"late_fusion.",
)


@dataclass(frozen=True)
class TrainableParameterGroups:
	"""Names updated throughout training or only after the warm-start window."""

	warm_start: tuple[str, ...]
	joint_only: tuple[str, ...]

	@property
	def all(self) -> tuple[str, ...]:
		"""Return every parameter owned by the single optimizer."""
		return self.warm_start + self.joint_only


def align_trainable_parameter_dtype(model: nn.Module, dtype: torch.dtype) -> tuple[str, ...]:
	"""Move only trainable parameter storage to the optimizer-safe dtype."""
	aligned_names: list[str] = []
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		parameter.data = parameter.data.to(dtype=dtype)
		if parameter.grad is not None:
			parameter.grad.data = parameter.grad.data.to(dtype=dtype)
		aligned_names.append(name)
	return tuple(aligned_names)


def configure_trainable_parameters(model: nn.Module) -> TrainableParameterGroups:
	"""Freeze the backbone and enable both optimizer groups exactly once."""
	model.requires_grad_(False)
	warm_start: list[str] = []
	joint_only: list[str] = []
	for name, parameter in model.named_parameters():
		if name.startswith(WARM_START_PREFIXES):
			parameter.requires_grad_(True)
			warm_start.append(name)
			continue
		if name.startswith(JOINT_ONLY_PREFIXES) or (
			".lora_a." in name or ".lora_b." in name
		):
			parameter.requires_grad_(True)
			joint_only.append(name)
	if not warm_start or not joint_only:
		raise RuntimeError("Warm-start and joint-only parameter groups must both be non-empty")
	return TrainableParameterGroups(
		warm_start=tuple(warm_start),
		joint_only=tuple(joint_only),
	)


def audit_gradient_scope(
	model: nn.Module,
	allowed_names: tuple[str, ...],
) -> dict[str, object]:
	"""Reject every nonzero gradient outside the active stage's exact allowlist."""
	allowed = set(allowed_names)
	nonzero_names: list[str] = []
	for name, parameter in model.named_parameters():
		if parameter.grad is None or not torch.count_nonzero(parameter.grad).item():
			continue
		if name not in allowed:
			raise RuntimeError(f"Forbidden parameter received a gradient: {name}")
		nonzero_names.append(name)
	return {
		"nonzero_gradient_parameter_count": len(nonzero_names),
		"nonzero_gradient_parameter_names": tuple(nonzero_names),
	}
