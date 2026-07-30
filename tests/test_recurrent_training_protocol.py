from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from looped_vl.baseline.losses import multi_positive_symmetric_info_nce
from looped_vl.training.data import paired_training_collate
from looped_vl.training.schedule import (
	BatchOffsetSampler,
	resolve_one_epoch_stage_plans,
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


def test_stage1_can_skip_unused_final_contrastive_loss() -> None:
	query = torch.eye(2, requires_grad=True)
	candidate = torch.eye(2, requires_grad=True)

	losses = distributed_multi_positive_info_nce_losses(
		embedding_pairs={"slot": (query, candidate)},
		positive_ids=("first", "second"),
		temperature=0.02,
	)

	assert set(losses) == {"slot"}


@pytest.mark.parametrize(
	("loader_batches", "expected_stage1_steps", "expected_stage2_steps"),
	(
		(8_856, 426, 681),
		(14_735, 708, 1_134),
		(10_938, 526, 842),
	),
)
def test_one_epoch_stage_plan_covers_every_batch_once(
	loader_batches: int,
	expected_stage1_steps: int,
	expected_stage2_steps: int,
) -> None:
	stage1, stage2 = resolve_one_epoch_stage_plans(
		loader_batches=loader_batches,
		gradient_accumulation_steps=8,
		stage_step_weights={1: 2_000, 2: 3_200},
	)

	assert stage1.start_batch == 0
	assert stage1.end_batch == stage2.start_batch
	assert stage2.end_batch == loader_batches
	assert stage1.optimizer_steps == expected_stage1_steps
	assert stage2.optimizer_steps == expected_stage2_steps
	assert stage1.end_batch - stage1.start_batch <= 8 * stage1.optimizer_steps
	assert stage2.end_batch - stage2.start_batch <= 8 * stage2.optimizer_steps


def test_one_epoch_stage_plan_rejects_too_few_optimizer_groups() -> None:
	with pytest.raises(ValueError, match="at least two optimizer steps"):
		resolve_one_epoch_stage_plans(
			loader_batches=3,
			gradient_accumulation_steps=8,
			stage_step_weights={1: 2_000, 2: 3_200},
		)


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


def test_stage_batch_ranges_reconstruct_each_rank_one_epoch_stream() -> None:
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
		stage1, stage2 = resolve_one_epoch_stage_plans(
			loader_batches=6,
			gradient_accumulation_steps=2,
			stage_step_weights={1: 2_000, 2: 3_200},
		)
		sampler.set_batch_range(stage1.start_batch, stage1.end_batch)
		stage1_indices = list(sampler)
		sampler.set_batch_range(stage2.start_batch, stage2.end_batch)
		stage2_indices = list(sampler)

		assert stage1_indices + stage2_indices == expected
