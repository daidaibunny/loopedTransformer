"""AdamW and warmup-cosine scheduler construction."""

from __future__ import annotations

import math

import torch
from torch import nn

from looped_vl.training.config import TrainingStageConfig


def build_optimizer_and_scheduler(
	model: nn.Module,
	config: TrainingStageConfig,
	*,
	total_steps: int | None = None,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
	"""Create the exact optimizer over only currently trainable parameters."""
	resolved_total_steps = config.schedule_weight if total_steps is None else total_steps
	if resolved_total_steps <= 0:
		raise ValueError("total_steps must be positive")
	parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
	if not parameters:
		raise RuntimeError("Cannot build an optimizer without trainable parameters")
	optimizer = torch.optim.AdamW(
		parameters,
		lr=config.learning_rate,
		weight_decay=config.weight_decay,
		betas=config.betas,
		eps=config.eps,
		fused=parameters[0].is_cuda,
	)
	warmup_steps = max(1, round(resolved_total_steps * config.warmup_ratio))

	def learning_rate_multiplier(step: int) -> float:
		if step < warmup_steps:
			return float(step + 1) / warmup_steps
		progress = (step - warmup_steps) / max(1, resolved_total_steps - warmup_steps)
		return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
	return optimizer, scheduler
