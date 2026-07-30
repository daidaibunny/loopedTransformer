from __future__ import annotations

import torch

from looped_vl.baseline.losses import multi_positive_symmetric_info_nce


def test_multi_positive_loss_treats_duplicate_answer_labels_as_positives() -> None:
	query_embeddings = torch.tensor(
		[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
		requires_grad=True,
	)
	candidate_embeddings = torch.tensor(
		[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
		requires_grad=True,
	)

	loss = multi_positive_symmetric_info_nce(
		query_embeddings,
		candidate_embeddings,
		positive_ids=("answer:yes", "answer:yes", "answer:no"),
		temperature=0.02,
	)
	loss.backward()

	assert torch.isfinite(loss)
	assert loss.item() < 0.1
	assert query_embeddings.grad is not None
	assert candidate_embeddings.grad is not None


def test_multi_positive_loss_rejects_missing_local_labels() -> None:
	embeddings = torch.eye(2)

	try:
		multi_positive_symmetric_info_nce(
			embeddings,
			embeddings,
			positive_ids=("only-one",),
			temperature=0.02,
		)
	except ValueError as error:
		assert "positive_ids" in str(error)
	else:
		raise AssertionError("Expected mismatched positive IDs to fail")
