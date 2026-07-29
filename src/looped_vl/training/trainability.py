"""Strict Stage 1 and Stage 2 trainable-parameter allowlists."""

from __future__ import annotations

import torch
from torch import nn

STAGE1_PREFIXES = (
	"latent_slots",
	"recurrent_connector.",
	"warmup_embedding_head.",
	"warmup_semantic_head.",
)
STAGE2_PREFIXES = STAGE1_PREFIXES + (
	"eos_delta",
	"late_fusion.",
)


def configure_trainable_parameters(model: nn.Module, stage: int) -> tuple[str, ...]:
	"""Freeze everything, then enable exactly the v1.0 parameter set for one stage."""
	if stage not in (1, 2):
		raise ValueError("stage must be 1 or 2")
	model.requires_grad_(False)
	allowed_prefixes = STAGE1_PREFIXES if stage == 1 else STAGE2_PREFIXES
	trainable: list[str] = []
	for name, parameter in model.named_parameters():
		is_allowed_component = name.startswith(allowed_prefixes)
		is_stage2_lora = stage == 2 and (".lora_a." in name or ".lora_b." in name)
		if is_allowed_component or is_stage2_lora:
			parameter.requires_grad_(True)
			trainable.append(name)
	if not trainable:
		raise RuntimeError(f"No trainable parameters were selected for Stage {stage}")
	return tuple(trainable)


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
