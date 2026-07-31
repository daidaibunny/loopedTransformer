"""One-epoch scheduling for continuous recurrent training."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

from torch.utils.data import Sampler

FORMAL_TRAINING_LOG_INTERVAL = 50


def should_log_training_metrics(
	*,
	optimizer_steps_since_log: int,
	global_step: int,
	optimizer_step_limit: int,
	force_every_step: bool = False,
	force_boundary: bool = False,
) -> bool:
	"""Log smokes per step and formal runs every 50 steps or at a required boundary."""
	if optimizer_steps_since_log <= 0:
		raise ValueError("optimizer_steps_since_log must be positive")
	if global_step <= 0 or optimizer_step_limit <= 0:
		raise ValueError("optimizer step values must be positive")
	if global_step > optimizer_step_limit:
		raise ValueError("global_step cannot exceed optimizer_step_limit")
	return (
		force_every_step
		or force_boundary
		or optimizer_steps_since_log >= FORMAL_TRAINING_LOG_INTERVAL
		or global_step == optimizer_step_limit
	)


@dataclass(frozen=True)
class OneEpochTrainingPlan:
	"""A full epoch with a warm-start prefix and continuous joint suffix."""

	start_batch: int
	end_batch: int
	optimizer_steps: int
	warm_start_optimizer_steps: int
	joint_optimizer_steps: int
	joint_activation_optimizer_steps: int

	@property
	def loader_batches(self) -> int:
		"""Return the number of per-rank DataLoader batches in the epoch."""
		return self.end_batch - self.start_batch


@dataclass(frozen=True)
class ParallelBatchSizes:
	"""Separate the contrastive negative pool from the optimizer accumulation."""

	contrastive_global_batch_size: int
	optimizer_global_batch_size: int


def resolve_parallel_batch_sizes(
	*,
	per_device_batch_size: int,
	world_size: int,
	gradient_accumulation_steps: int,
) -> ParallelBatchSizes:
	"""Calculate both non-equivalent global batch cardinalities."""
	if per_device_batch_size <= 0 or world_size <= 0 or gradient_accumulation_steps <= 0:
		raise ValueError("Parallel batch dimensions must be positive")
	contrastive_size = per_device_batch_size * world_size
	return ParallelBatchSizes(
		contrastive_global_batch_size=contrastive_size,
		optimizer_global_batch_size=contrastive_size * gradient_accumulation_steps,
	)


class BatchOffsetSampler(Sampler[int]):
	"""Skip consumed batches before the DataLoader decodes any samples."""

	def __init__(self, sampler: Sampler[int], batch_size: int) -> None:
		if batch_size <= 0:
			raise ValueError("batch_size must be positive")
		self.sampler = sampler
		self.batch_size = batch_size
		self.start_batch = 0
		self.end_batch: int | None = None

	def set_batch_range(self, start_batch: int, end_batch: int) -> None:
		"""Restrict iteration to an absolute, half-open batch range."""
		max_batches = math.ceil(len(self.sampler) / self.batch_size)
		if start_batch < 0 or end_batch < start_batch or end_batch > max_batches:
			raise ValueError("batch range is outside the sampler epoch")
		self.start_batch = start_batch
		self.end_batch = end_batch

	def set_epoch(self, epoch: int) -> None:
		"""Forward deterministic epoch selection to a distributed sampler."""
		set_epoch = getattr(self.sampler, "set_epoch", None)
		if set_epoch is not None:
			set_epoch(epoch)

	def __iter__(self) -> Iterator[int]:
		stop_sample = (
			None
			if self.end_batch is None
			else min(len(self.sampler), self.end_batch * self.batch_size)
		)
		return islice(
			iter(self.sampler),
			self.start_batch * self.batch_size,
			stop_sample,
		)

	def __len__(self) -> int:
		stop_sample = (
			len(self.sampler)
			if self.end_batch is None
			else min(len(self.sampler), self.end_batch * self.batch_size)
		)
		return max(0, stop_sample - self.start_batch * self.batch_size)

	def __getattr__(self, name: str) -> Any:
		"""Expose distributed sampler cardinalities for run manifests."""
		return getattr(self.sampler, name)


def resolve_one_epoch_training_plan(
	*,
	train_rows: int,
	loader_batches: int,
	gradient_accumulation_steps: int,
	optimizer_global_batch_size: int,
	warm_start_epoch_fraction: float,
	joint_activation_warmup_ratio: float,
) -> OneEpochTrainingPlan:
	"""Resolve the dynamic warm-start prefix inside one continuous epoch."""
	if train_rows <= 0:
		raise ValueError("train_rows must be positive")
	if loader_batches <= 0:
		raise ValueError("loader_batches must be positive")
	if gradient_accumulation_steps <= 0:
		raise ValueError("gradient_accumulation_steps must be positive")
	if optimizer_global_batch_size <= 0:
		raise ValueError("optimizer_global_batch_size must be positive")
	if not 0 < warm_start_epoch_fraction < 1:
		raise ValueError("warm_start_epoch_fraction must be between zero and one")
	if not 0 < joint_activation_warmup_ratio < 1:
		raise ValueError("joint_activation_warmup_ratio must be between zero and one")
	total_optimizer_steps = math.ceil(loader_batches / gradient_accumulation_steps)
	if total_optimizer_steps < 2:
		raise ValueError("One-epoch training needs at least one joint optimizer step")
	warm_start_steps = math.ceil(
		warm_start_epoch_fraction * train_rows / optimizer_global_batch_size,
	)
	warm_start_steps = min(max(warm_start_steps, 1), total_optimizer_steps - 1)
	joint_steps = total_optimizer_steps - warm_start_steps
	joint_activation_steps = min(
		joint_steps,
		max(1, math.ceil(joint_steps * joint_activation_warmup_ratio)),
	)
	return OneEpochTrainingPlan(
		start_batch=0,
		end_batch=loader_batches,
		optimizer_steps=total_optimizer_steps,
		warm_start_optimizer_steps=warm_start_steps,
		joint_optimizer_steps=joint_steps,
		joint_activation_optimizer_steps=joint_activation_steps,
	)
