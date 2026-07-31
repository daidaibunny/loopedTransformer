"""One AdamW optimizer with phase-aware parameter-group scheduling."""

from __future__ import annotations

import math

import torch
from torch import nn

from looped_vl.training.config import TrainingConfig


def build_optimizer_and_scheduler(
	model: nn.Module,
	config: TrainingConfig,
	*,
	recurrent_core_parameter_names: tuple[str, ...],
	final_fusion_parameter_names: tuple[str, ...],
	total_steps: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
	"""Create one optimizer whose parameter groups are active from the first step."""
	if total_steps <= 1:
		raise ValueError("total_steps must be greater than one")
	named_parameters = dict(model.named_parameters())
	recurrent_core_parameters = [
		named_parameters[name] for name in recurrent_core_parameter_names
	]
	final_fusion_parameters = [
		named_parameters[name] for name in final_fusion_parameter_names
	]
	selected_names = set(recurrent_core_parameter_names) | set(
		final_fusion_parameter_names,
	)
	trainable_names = {
		name for name, parameter in named_parameters.items() if parameter.requires_grad
	}
	if selected_names != trainable_names:
		raise ValueError("Optimizer parameter groups do not match trainable parameters")
	if not recurrent_core_parameters:
		raise RuntimeError("Recurrent-core optimizer group cannot be empty")
	parameter_groups: list[dict[str, object]] = [
		{"params": recurrent_core_parameters, "group_name": "recurrent_core"},
	]
	if final_fusion_parameters:
		parameter_groups.append(
			{"params": final_fusion_parameters, "group_name": "final_fusion"},
		)
	optimizer = torch.optim.AdamW(
		parameter_groups,
		lr=config.learning_rate,
		weight_decay=config.weight_decay,
		betas=config.betas,
		eps=config.eps,
		fused=recurrent_core_parameters[0].is_cuda,
	)
	warmup_steps = max(1, round(total_steps * config.warmup_ratio))

	def learning_rate_multiplier(step: int) -> float:
		if step < warmup_steps:
			return float(step + 1) / warmup_steps
		progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
		return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

	lambdas = [learning_rate_multiplier]
	if final_fusion_parameters:
		lambdas.append(learning_rate_multiplier)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
	return optimizer, scheduler
