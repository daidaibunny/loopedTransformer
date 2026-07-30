"""One-epoch scheduling shared by the two recurrent training stages."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

from torch.utils.data import Sampler


@dataclass(frozen=True)
class EpochStagePlan:
	"""One contiguous batch range trained under a single parameter scope."""

	stage: int
	start_batch: int
	end_batch: int
	optimizer_steps: int

	@property
	def loader_batches(self) -> int:
		"""Return the number of per-rank DataLoader batches in this stage."""
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


def resolve_one_epoch_stage_plans(
	*,
	loader_batches: int,
	gradient_accumulation_steps: int,
	stage_step_weights: dict[int, int],
) -> tuple[EpochStagePlan, EpochStagePlan]:
	"""Split one epoch into contiguous Stage 1 and Stage 2 optimizer groups."""
	if loader_batches <= 0:
		raise ValueError("loader_batches must be positive")
	if gradient_accumulation_steps <= 0:
		raise ValueError("gradient_accumulation_steps must be positive")
	if set(stage_step_weights) != {1, 2}:
		raise ValueError("stage_step_weights must contain exactly Stage 1 and Stage 2")
	if any(weight <= 0 for weight in stage_step_weights.values()):
		raise ValueError("stage step weights must be positive")
	total_optimizer_steps = math.ceil(loader_batches / gradient_accumulation_steps)
	if total_optimizer_steps < 2:
		raise ValueError("One-epoch two-stage training needs at least two optimizer steps")
	stage1_weight = stage_step_weights[1]
	weight_total = sum(stage_step_weights.values())
	stage1_steps = round(total_optimizer_steps * stage1_weight / weight_total)
	stage1_steps = min(max(stage1_steps, 1), total_optimizer_steps - 1)
	stage1_end_batch = min(
		loader_batches,
		stage1_steps * gradient_accumulation_steps,
	)
	stage2_batches = loader_batches - stage1_end_batch
	stage2_steps = math.ceil(stage2_batches / gradient_accumulation_steps)
	return (
		EpochStagePlan(
			stage=1,
			start_batch=0,
			end_batch=stage1_end_batch,
			optimizer_steps=stage1_steps,
		),
		EpochStagePlan(
			stage=2,
			start_batch=stage1_end_batch,
			end_batch=loader_batches,
			optimizer_steps=stage2_steps,
		),
	)
