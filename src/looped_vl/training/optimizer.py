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
	warm_start_parameter_names: tuple[str, ...],
	joint_parameter_names: tuple[str, ...],
	total_steps: int,
	warm_start_steps: int,
	joint_activation_steps: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
	"""Create one optimizer; keep joint parameters at zero LR during warm-start."""
	if total_steps <= 1:
		raise ValueError("total_steps must be greater than one")
	if not 0 < warm_start_steps < total_steps:
		raise ValueError("warm_start_steps must leave at least one joint step")
	joint_steps = total_steps - warm_start_steps
	if not 0 < joint_activation_steps <= joint_steps:
		raise ValueError("joint_activation_steps must fit inside joint training")
	named_parameters = dict(model.named_parameters())
	warm_start_parameters = [
		named_parameters[name] for name in warm_start_parameter_names
	]
	joint_parameters = [named_parameters[name] for name in joint_parameter_names]
	selected_names = set(warm_start_parameter_names) | set(joint_parameter_names)
	trainable_names = {
		name for name, parameter in named_parameters.items() if parameter.requires_grad
	}
	if selected_names != trainable_names:
		raise ValueError("Optimizer parameter groups do not match trainable parameters")
	if not warm_start_parameters:
		raise RuntimeError("Warm-start optimizer group cannot be empty")
	parameter_groups: list[dict[str, object]] = [
		{"params": warm_start_parameters, "group_name": "warm_start"},
	]
	if joint_parameters:
		parameter_groups.append(
			{"params": joint_parameters, "group_name": "joint_only"},
		)
	optimizer = torch.optim.AdamW(
		parameter_groups,
		lr=config.learning_rate,
		weight_decay=config.weight_decay,
		betas=config.betas,
		eps=config.eps,
		fused=warm_start_parameters[0].is_cuda,
	)
	warmup_steps = max(1, round(total_steps * config.warmup_ratio))

	def learning_rate_multiplier(step: int) -> float:
		if step < warmup_steps:
			return float(step + 1) / warmup_steps
		progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
		return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

	def joint_learning_rate_multiplier(step: int) -> float:
		if step < warm_start_steps:
			return 0.0
		joint_step = step - warm_start_steps
		if joint_step < joint_activation_steps:
			return float(joint_step + 1) / joint_activation_steps
		progress = (joint_step - joint_activation_steps) / max(
			1,
			joint_steps - joint_activation_steps,
		)
		return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

	lambdas = [learning_rate_multiplier]
	if joint_parameters:
		lambdas.append(joint_learning_rate_multiplier)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
	return optimizer, scheduler
