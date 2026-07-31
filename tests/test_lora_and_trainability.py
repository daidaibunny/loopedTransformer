import torch
from torch import nn

from looped_vl.training.trainability import (
	align_trainable_parameter_dtype,
	audit_gradient_scope,
	configure_trainable_parameters,
)


class TinyRecurrentModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.base_embedding_model = nn.Linear(4, 4)
		self.latent_slots = nn.Parameter(torch.randn(1, 16, 4))
		self.eos_delta = nn.Parameter(torch.zeros(1, 1, 4))
		self.late_fusion = nn.Linear(4, 4)
		self.auxiliary_embedding_head = nn.Linear(4, 4)


def test_pure_recurrent_trainability_freezes_the_entire_backbone() -> None:
	model = TinyRecurrentModel()

	groups = configure_trainable_parameters(model)
	assert all(
		name.startswith(
			(
				"latent_slots",
				"auxiliary_embedding_head",
			)
		)
		for name in groups.recurrent_core
	)
	assert any(name.startswith("eos_delta") for name in groups.final_fusion)
	assert any(name.startswith("late_fusion") for name in groups.final_fusion)
	assert not any(name.startswith("base_embedding_model") for name in groups.all)
	assert not any("lora_" in name.lower() for name in groups.all)
	assert not any(
		parameter.requires_grad
		for parameter in model.base_embedding_model.parameters()
	)
	assert set(groups.all) == {
		name for name, parameter in model.named_parameters() if parameter.requires_grad
	}


def test_fp16_storage_promotes_only_trainable_parameters_to_fp32() -> None:
	model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)).to(torch.float16)
	model[1].requires_grad_(False)

	aligned_names = align_trainable_parameter_dtype(model, torch.float32)

	assert aligned_names == ("0.weight", "0.bias")
	assert model[0].weight.dtype is torch.float32
	assert model[0].bias.dtype is torch.float32
	assert model[1].weight.dtype is torch.float16
	assert model[1].bias.dtype is torch.float16


def test_gradient_audit_rejects_any_gradient_outside_allowlist() -> None:
	model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
	allowed_name = "0.weight"
	model[0].weight.grad = torch.ones_like(model[0].weight)
	model[1].weight.grad = torch.ones_like(model[1].weight)

	try:
		audit_gradient_scope(model, allowed_names=(allowed_name,))
		raised = False
	except RuntimeError:
		raised = True

	assert raised is True
