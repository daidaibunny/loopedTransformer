"""Strict Stage 1 and Stage 2 trainable-parameter allowlists."""

from __future__ import annotations

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
