"""Validated configuration for Recurrent Latent-Slot Qwen3-VL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SLOT_COUNTS = (0, 1, 2, 4, 8, 16)
ALLOWED_LOOP_PASSES = (1, 2, 3, 4)
REQUIRED_LORA_TARGETS = (
	"q_proj",
	"k_proj",
	"v_proj",
	"up_proj",
	"down_proj",
	"gate_proj",
)


@dataclass(frozen=True)
class RecurrentModelConfig:
	"""Every structural constant fixed by implementation specification v1.0."""

	model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
	seed: int = 42
	hidden_size: int = 2048
	max_num_latent_slots: int = 16
	num_latent_slots: int = 4
	latent_init_mean: float = 0.0
	latent_init_std: float = 0.02
	loop_start_layer: int = 12
	loop_end_layer: int = 20
	num_total_loop_passes: int = 4
	update_prefix_in_extra_loops: bool = False
	detach_prefix_kv_cache: bool = True
	recurrent_bottleneck_dim: int = 512
	recurrent_activation: str = "silu"
	recurrent_dropout: float = 0.0
	recurrent_shared_across_passes: bool = True
	recurrent_output_init: str = "zero"
	fusion_type: str = "eos_conditioned_slot_attention"
	fusion_attention_dim: int = 256
	fusion_num_heads: int = 1
	fusion_dropout: float = 0.0
	fusion_residual_gate_init: float = 0.0
	fuse_after_final_decoder_norm: bool = True
	lora_layer_start: int = 12
	lora_layer_end: int = 20
	lora_rank: int = 32
	lora_alpha: int = 32
	lora_dropout: float = 0.0
	lora_target_modules: tuple[str, ...] = REQUIRED_LORA_TARGETS
	temperature: float = 0.02
	stage1_slot_weight: float = 1.0
	stage1_decoder_weight: float = 1.0
	stage1_diversity_weight: float = 0.05
	stage2_final_weight: float = 1.0
	stage2_slot_weight: float = 0.2
	stage2_decoder_weight: float = 0.2
	stage2_diversity_weight: float = 0.05

	@property
	def num_extra_loop_passes(self) -> int:
		"""Return the recurrent-only pass count after the full first pass."""
		return self.num_total_loop_passes - 1

	def validate(self) -> None:
		"""Reject any structural value outside the fixed v1.0 experiment space."""
		if self.seed != 42:
			raise ValueError("seed must remain 42")
		if self.hidden_size != 2048:
			raise ValueError("hidden_size must remain 2048")
		if self.max_num_latent_slots != 16:
			raise ValueError("max_num_latent_slots must remain 16")
		if self.num_latent_slots not in ALLOWED_SLOT_COUNTS:
			raise ValueError(f"num_latent_slots must be one of {ALLOWED_SLOT_COUNTS}")
		if self.num_total_loop_passes not in ALLOWED_LOOP_PASSES:
			raise ValueError(
				f"num_total_loop_passes must be one of {ALLOWED_LOOP_PASSES}",
			)
		if (self.loop_start_layer, self.loop_end_layer) != (12, 20):
			raise ValueError("loop layers must use Python indexes [12, 20)")
		if self.update_prefix_in_extra_loops:
			raise ValueError("prefix updates in extra loops are forbidden")
		if not self.detach_prefix_kv_cache:
			raise ValueError("prefix K/V cache must be detached")
		if self.recurrent_bottleneck_dim != 512:
			raise ValueError("recurrent bottleneck must remain 512")
		if self.recurrent_activation != "silu" or self.recurrent_dropout != 0.0:
			raise ValueError("recurrent connector must use SiLU with zero dropout")
		if not self.recurrent_shared_across_passes or self.recurrent_output_init != "zero":
			raise ValueError("connector must be shared and zero-output initialized")
		if self.fusion_attention_dim != 256 or self.fusion_num_heads != 1:
			raise ValueError("late fusion must use one 256-dimensional head")
		if self.fusion_dropout != 0.0 or self.fusion_residual_gate_init != 0.0:
			raise ValueError("late fusion must use zero dropout and a zero gate")
		if not self.fuse_after_final_decoder_norm:
			raise ValueError("late fusion must run after the final decoder norm")
		if (self.lora_layer_start, self.lora_layer_end) != (12, 20):
			raise ValueError("LoRA layers must use Python indexes [12, 20)")
		if (self.lora_rank, self.lora_alpha, self.lora_dropout) != (32, 32, 0.0):
			raise ValueError("LoRA must use rank 32, alpha 32, and zero dropout")
		if self.lora_target_modules != REQUIRED_LORA_TARGETS:
			raise ValueError("LoRA target modules do not match v1.0")
		if self.temperature != 0.02:
			raise ValueError("InfoNCE temperature must remain 0.02")

	def with_variant(
		self,
		*,
		num_latent_slots: int | None = None,
		num_total_loop_passes: int | None = None,
	) -> RecurrentModelConfig:
		"""Create one allowed baseline or ablation variant."""
		updated = replace(
			self,
			num_latent_slots=(
				self.num_latent_slots if num_latent_slots is None else num_latent_slots
			),
			num_total_loop_passes=(
				self.num_total_loop_passes
				if num_total_loop_passes is None
				else num_total_loop_passes
			),
		)
		updated.validate()
		return updated

	@classmethod
	def from_yaml(cls, path: str | Path) -> RecurrentModelConfig:
		"""Load the nested project YAML into the immutable runtime configuration."""
		value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
		if not isinstance(value, dict):
			raise ValueError("Model configuration must be a YAML mapping")
		recurrent = _mapping(value, "recurrent_connector")
		fusion = _mapping(value, "fusion")
		lora = _mapping(value, "lora")
		loss = _mapping(value, "loss")
		config = cls(
			model_name=str(value["model_name"]),
			seed=int(value["seed"]),
			hidden_size=int(value["hidden_size"]),
			max_num_latent_slots=int(value["max_num_latent_slots"]),
			num_latent_slots=int(value["num_latent_slots"]),
			latent_init_mean=float(value["latent_init_mean"]),
			latent_init_std=float(value["latent_init_std"]),
			loop_start_layer=int(value["loop_start_layer"]),
			loop_end_layer=int(value["loop_end_layer"]),
			num_total_loop_passes=int(value["num_total_loop_passes"]),
			update_prefix_in_extra_loops=bool(value["update_prefix_in_extra_loops"]),
			detach_prefix_kv_cache=bool(value["detach_prefix_kv_cache"]),
			recurrent_bottleneck_dim=int(recurrent["bottleneck_dim"]),
			recurrent_activation=str(recurrent["activation"]),
			recurrent_dropout=float(recurrent["dropout"]),
			recurrent_shared_across_passes=bool(recurrent["share_across_passes"]),
			recurrent_output_init=str(recurrent["output_init"]),
			fusion_type=str(fusion["type"]),
			fusion_attention_dim=int(fusion["attention_dim"]),
			fusion_num_heads=int(fusion["num_heads"]),
			fusion_dropout=float(fusion["dropout"]),
			fusion_residual_gate_init=float(fusion["residual_gate_init"]),
			fuse_after_final_decoder_norm=bool(fusion["fuse_after_final_decoder_norm"]),
			lora_layer_start=int(lora["layer_start"]),
			lora_layer_end=int(lora["layer_end"]),
			lora_rank=int(lora["rank"]),
			lora_alpha=int(lora["alpha"]),
			lora_dropout=float(lora["dropout"]),
			lora_target_modules=tuple(lora["target_modules"]),
			temperature=float(loss["temperature"]),
			stage1_slot_weight=float(loss["stage1_slot_weight"]),
			stage1_decoder_weight=float(loss["stage1_decoder_weight"]),
			stage1_diversity_weight=float(loss["stage1_diversity_weight"]),
			stage2_final_weight=float(loss["stage2_final_weight"]),
			stage2_slot_weight=float(loss["stage2_slot_weight"]),
			stage2_decoder_weight=float(loss["stage2_decoder_weight"]),
			stage2_diversity_weight=float(loss["stage2_diversity_weight"]),
		)
		config.validate()
		return config


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
	nested = value.get(key)
	if not isinstance(nested, dict):
		raise ValueError(f"Configuration key {key} must be a mapping")
	return nested
