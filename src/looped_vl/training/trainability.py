"""Strict trainable groups for a full-objective recurrent optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

RECURRENT_CORE_PREFIXES = (
	"latent_slots",
	"recurrent_connector.",
	"warmup_embedding_head.",
)
FINAL_FUSION_PREFIXES = (
	"eos_delta",
	"late_fusion.",
)


@dataclass(frozen=True)
class TrainableParameterGroups:
	"""Recurrent-core and final-fusion names updated throughout the full epoch."""

	recurrent_core: tuple[str, ...]
	final_fusion: tuple[str, ...]

	@property
	def all(self) -> tuple[str, ...]:
		"""Return every parameter owned by the single optimizer."""
		return self.recurrent_core + self.final_fusion


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
	recurrent_core: list[str] = []
	final_fusion: list[str] = []
	for name, parameter in model.named_parameters():
		if name.startswith(RECURRENT_CORE_PREFIXES):
			parameter.requires_grad_(True)
			recurrent_core.append(name)
			continue
		if name.startswith(FINAL_FUSION_PREFIXES):
			parameter.requires_grad_(True)
			final_fusion.append(name)
	if not recurrent_core or not final_fusion:
		raise RuntimeError("Recurrent-core and final-fusion groups must both be non-empty")
	return TrainableParameterGroups(
		recurrent_core=tuple(recurrent_core),
		final_fusion=tuple(final_fusion),
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
