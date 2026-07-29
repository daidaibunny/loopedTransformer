import torch
from torch import nn

from looped_vl.models.lora import LoRALinear, inject_loop_layer_lora
from looped_vl.training.trainability import configure_trainable_parameters


class TinyAttention(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.q_proj = nn.Linear(4, 4, bias=False)
		self.k_proj = nn.Linear(4, 4, bias=False)
		self.v_proj = nn.Linear(4, 4, bias=False)
		self.o_proj = nn.Linear(4, 4, bias=False)


class TinyMLP(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.up_proj = nn.Linear(4, 8, bias=False)
		self.down_proj = nn.Linear(8, 4, bias=False)
		self.gate_proj = nn.Linear(4, 8, bias=False)


class TinyLayer(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.self_attn = TinyAttention()
		self.mlp = TinyMLP()


class TinyRecurrentModel(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.base_embedding_model = nn.Module()
		self.base_embedding_model.layers = nn.ModuleList([TinyLayer() for _ in range(28)])
		self.latent_slots = nn.Parameter(torch.randn(1, 16, 4))
		self.eos_delta = nn.Parameter(torch.zeros(1, 1, 4))
		self.recurrent_connector = nn.Linear(4, 4)
		self.late_fusion = nn.Linear(4, 4)
		self.warmup_embedding_head = nn.Linear(4, 4)
		self.warmup_semantic_head = nn.Linear(4, 4)


def test_lora_zero_initialization_preserves_base_linear_output() -> None:
	base = nn.Linear(4, 3, bias=False)
	lora = LoRALinear(base, rank=2, alpha=2, dropout=0.0)
	inputs = torch.randn(5, 4)

	assert torch.equal(lora(inputs), base(inputs))
	assert lora.lora_b.weight.count_nonzero().item() == 0


def test_lora_is_injected_only_into_required_modules_of_layers_13_to_20() -> None:
	layers = nn.ModuleList([TinyLayer() for _ in range(28)])

	injected_names = inject_loop_layer_lora(
		layers=layers,
		layer_start=12,
		layer_end=20,
		rank=32,
		alpha=32,
		dropout=0.0,
	)

	assert len(injected_names) == 8 * 6
	assert not isinstance(layers[11].self_attn.q_proj, LoRALinear)
	assert isinstance(layers[12].self_attn.q_proj, LoRALinear)
	assert isinstance(layers[19].mlp.gate_proj, LoRALinear)
	assert not isinstance(layers[20].self_attn.q_proj, LoRALinear)
	assert not isinstance(layers[12].self_attn.o_proj, LoRALinear)


def test_stage_trainability_matches_strict_parameter_allowlists() -> None:
	model = TinyRecurrentModel()
	inject_loop_layer_lora(model.base_embedding_model.layers, 12, 20, 2, 2, 0.0)

	stage1 = configure_trainable_parameters(model, stage=1)
	assert all(
		name.startswith(
			(
				"latent_slots",
				"recurrent_connector",
				"warmup_embedding_head",
				"warmup_semantic_head",
			)
		)
		for name in stage1
	)
	assert not any("lora_" in name for name in stage1)

	stage2 = configure_trainable_parameters(model, stage=2)
	assert any(name.startswith("eos_delta") for name in stage2)
	assert any(name.startswith("late_fusion") for name in stage2)
	assert any("lora_" in name for name in stage2)
	assert not any("o_proj" in name for name in stage2)
	assert not any("layers.11" in name or "layers.20" in name for name in stage2)
