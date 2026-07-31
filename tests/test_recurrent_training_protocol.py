from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from looped_vl.baseline.losses import multi_positive_symmetric_info_nce
from looped_vl.training.data import paired_training_collate
from looped_vl.training.schedule import (
	BatchOffsetSampler,
	resolve_one_epoch_training_plan,
	resolve_parallel_batch_sizes,
)
from looped_vl.training.step import distributed_multi_positive_info_nce_losses


def _sample(
	*,
	position: int,
	positive_id: str,
	answer: str = "yes",
) -> SimpleNamespace:
	return SimpleNamespace(
		source="clevr",
		mixture_position=position,
		text=f"question {position}",
		answer=answer,
		image=object(),
		positive_id=positive_id,
		reasoning_depth=3,
		sample_id=f"clevr:{position}",
	)


def test_recurrent_collate_preserves_positive_ids() -> None:
	batch = paired_training_collate(
		[
			_sample(position=0, positive_id="answer:yes"),
			_sample(position=1, positive_id="answer:yes"),
			_sample(position=2, positive_id="answer:no", answer="no"),
		],
	)

	assert batch["positive_ids"] == ["answer:yes", "answer:yes", "answer:no"]


def test_recurrent_losses_match_independent_multi_positive_losses() -> None:
	query = torch.tensor(
		[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
		requires_grad=True,
	)
	candidate = torch.tensor(
		[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
		requires_grad=True,
	)
	slot_query = torch.tensor(
		[[0.8, 0.2], [1.0, 0.0], [0.1, 0.9]],
		requires_grad=True,
	)
	slot_candidate = candidate.detach().clone().requires_grad_(True)
	positive_ids = ("answer:yes", "answer:yes", "answer:no")

	actual = distributed_multi_positive_info_nce_losses(
		embedding_pairs={
			"final": (query, candidate),
			"slot": (slot_query, slot_candidate),
		},
		positive_ids=positive_ids,
		temperature=0.02,
	)
	expected_final = multi_positive_symmetric_info_nce(
		query,
		candidate,
		positive_ids,
		temperature=0.02,
	)
	expected_slot = multi_positive_symmetric_info_nce(
		slot_query,
		slot_candidate,
		positive_ids,
		temperature=0.02,
	)

	assert torch.equal(actual["final"], expected_final)
	assert torch.equal(actual["slot"], expected_slot)
	embedding_tensors = (query, candidate, slot_query, slot_candidate)
	actual_gradients = torch.autograd.grad(
		sum(actual.values()),
		embedding_tensors,
		retain_graph=True,
	)
	expected_gradients = torch.autograd.grad(
		expected_final + expected_slot,
		embedding_tensors,
	)
	for actual_gradient, expected_gradient in zip(
		actual_gradients,
		expected_gradients,
		strict=True,
	):
		assert torch.equal(actual_gradient, expected_gradient)
		assert torch.isfinite(actual_gradient).all()


@pytest.mark.parametrize(
	(
		"train_rows",
		"optimizer_global_batch_size",
		"loader_batches",
		"expected_optimizer_steps",
	),
	(
		(100_000, 256, 1_563, 391),
		(100_000, 512, 1_563, 196),
		(566_747, 512, 8_856, 1_107),
		(943_000, 512, 14_735, 1_842),
		(699_989, 512, 10_938, 1_368),
	),
)
def test_single_stage_plan_covers_every_loader_batch_once(
	train_rows: int,
	optimizer_global_batch_size: int,
	loader_batches: int,
	expected_optimizer_steps: int,
) -> None:
	gradient_accumulation_steps = 4 if optimizer_global_batch_size == 256 else 8
	plan = resolve_one_epoch_training_plan(
		train_rows=train_rows,
		loader_batches=loader_batches,
		gradient_accumulation_steps=gradient_accumulation_steps,
		optimizer_global_batch_size=optimizer_global_batch_size,
	)

	assert plan.start_batch == 0
	assert plan.end_batch == loader_batches
	assert plan.optimizer_steps == expected_optimizer_steps
	assert not hasattr(plan, "auxiliary_emphasis_optimizer_steps")


def test_one_epoch_training_plan_accepts_one_optimizer_step() -> None:
	plan = resolve_one_epoch_training_plan(
		train_rows=8,
		loader_batches=1,
		gradient_accumulation_steps=1,
		optimizer_global_batch_size=8,
	)

	assert plan.optimizer_steps == 1


def test_batch_sizes_distinguish_negative_pool_from_optimizer_accumulation() -> None:
	sizes = resolve_parallel_batch_sizes(
		per_device_batch_size=8,
		world_size=8,
		gradient_accumulation_steps=8,
	)

	assert sizes.contrastive_global_batch_size == 64
	assert sizes.optimizer_global_batch_size == 512


def test_batch_offset_sampler_skips_indices_before_dataset_loading() -> None:
	sampler = BatchOffsetSampler(list(range(11)), batch_size=2)
	sampler.set_batch_range(start_batch=2, end_batch=5)

	assert list(sampler) == [4, 5, 6, 7, 8, 9]
	assert len(sampler) == 6


def test_resume_batch_range_preserves_each_rank_one_epoch_stream() -> None:
	dataset = list(range(23))
	for rank in range(2):
		distributed = DistributedSampler(
			dataset,
			num_replicas=2,
			rank=rank,
			shuffle=True,
			seed=42,
			drop_last=False,
		)
		distributed.set_epoch(0)
		expected = list(distributed)
		sampler = BatchOffsetSampler(distributed, batch_size=2)
		sampler.set_batch_range(0, 3)
		before_resume = list(sampler)
		sampler.set_batch_range(3, 6)
		after_resume = list(sampler)

		assert before_resume + after_resume == expected
