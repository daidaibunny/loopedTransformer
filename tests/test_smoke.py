from __future__ import annotations

import pytest
import torch

from looped_vl.smoke import assert_model_frozen, freeze_model


def test_freeze_model_disables_gradients_and_sets_evaluation_mode() -> None:
	model = torch.nn.Sequential(
		torch.nn.Linear(4, 8),
		torch.nn.Dropout(p=0.5),
		torch.nn.Linear(8, 2),
	)
	model.train()

	freeze_model(model)

	assert model.training is False
	assert all(parameter.requires_grad is False for parameter in model.parameters())
	assert_model_frozen(model)


def test_assert_model_frozen_rejects_trainable_parameter() -> None:
	model = torch.nn.Linear(2, 2)

	with pytest.raises(RuntimeError, match="trainable"):
		assert_model_frozen(model)
