"""Validated configuration for Recurrent Latent-Slot Qwen3-VL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SLOT_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
ALLOWED_MASTER_SLOT_COUNTS = (16, 64)
ALLOWED_LOOP_PASSES = (1, 2, 3, 4)
DAMPED_RECURRENT_ARCHITECTURE = "damped_mid_decoder_latent_slot_recurrence_no_lora_v3"
DAMPED_RECURRENT_TRAINING_PROTOCOL = "pure_recurrent_single_stage_eos_weighted_aux_v4"
PURE_RECURRENT_ARCHITECTURE = DAMPED_RECURRENT_ARCHITECTURE
PURE_RECURRENT_TRAINING_PROTOCOL = DAMPED_RECURRENT_TRAINING_PROTOCOL


def pure_recurrent_result_identity() -> dict[str, object]:
	"""Return the immutable architecture fields required in every recurrent result."""
	return {
		"architecture": PURE_RECURRENT_ARCHITECTURE,
		"training_protocol": PURE_RECURRENT_TRAINING_PROTOCOL,
		"backbone_frozen": True,
		"lora_enabled": False,
		"formal_training_stages": 1,
	}


@dataclass(frozen=True)
class RecurrentModelConfig:
	"""Every structural constant fixed by the damped no-LoRA v3 specification."""

	model_name: str = "Qwen/Qwen3-VL-Embedding-2B"
	seed: int = 42
	hidden_size: int = 2048
	max_num_latent_slots: int = 16
	num_latent_slots: int = 8
	latent_init_mean: float = 0.0
	latent_init_std: float = 0.02
	loop_start_layer: int = 12
	loop_end_layer: int = 20
	num_total_loop_passes: int = 4
	update_prefix_in_extra_loops: bool = False
	detach_prefix_kv_cache: bool = True
	fusion_type: str = "eos_conditioned_slot_attention"
	fusion_attention_dim: int = 256
	fusion_num_heads: int = 1
	fusion_dropout: float = 0.0
	fusion_residual_gate_init: float = 0.0
	fuse_after_final_decoder_norm: bool = True
	auxiliary_output_dim: int = 256
	auxiliary_pooling: str = "eos_conditioned_weighted_slots"
	auxiliary_normalization: str = "rmsnorm"
	auxiliary_bias: bool = False
	temperature: float = 0.02
	final_infonce_weight: float = 1.0
	loop_infonce_weight: float = 0.1
	slot_diversity_weight: float = 0.05

	@property
	def num_extra_loop_passes(self) -> int:
		"""Return the recurrent-only pass count after the full first pass."""
		return self.num_total_loop_passes - 1

	@property
	def recurrent_step_size(self) -> float:
		"""Return the fixed damping coefficient alpha = 1 / R."""
		return 1.0 / self.num_total_loop_passes

	def validate(self) -> None:
		"""Reject any structural value outside the locked experiment space."""
		if self.seed != 42:
			raise ValueError("seed must remain 42")
		if self.hidden_size != 2048:
			raise ValueError("hidden_size must remain 2048")
		if self.max_num_latent_slots not in ALLOWED_MASTER_SLOT_COUNTS:
			raise ValueError(
				f"max_num_latent_slots must be one of {ALLOWED_MASTER_SLOT_COUNTS}",
			)
		if self.num_latent_slots not in ALLOWED_SLOT_COUNTS:
			raise ValueError(f"num_latent_slots must be one of {ALLOWED_SLOT_COUNTS}")
		if self.num_latent_slots > self.max_num_latent_slots:
			raise ValueError("num_latent_slots cannot exceed max_num_latent_slots")
		if self.num_total_loop_passes not in ALLOWED_LOOP_PASSES:
			raise ValueError(
				f"num_total_loop_passes must be one of {ALLOWED_LOOP_PASSES}",
			)
		if self.num_latent_slots == 0 and self.num_total_loop_passes != 1:
			raise ValueError("The zero-slot base variant requires exactly one pass")
		if (self.loop_start_layer, self.loop_end_layer) != (12, 20):
			raise ValueError("loop layers must use Python indexes [12, 20)")
		if self.update_prefix_in_extra_loops:
			raise ValueError("prefix updates in extra loops are forbidden")
		if not self.detach_prefix_kv_cache:
			raise ValueError("prefix K/V cache must be detached")
		if self.fusion_attention_dim != 256 or self.fusion_num_heads != 1:
			raise ValueError("late fusion must use one 256-dimensional head")
		if self.fusion_dropout != 0.0 or self.fusion_residual_gate_init != 0.0:
			raise ValueError("late fusion must use zero dropout and a zero gate")
		if not self.fuse_after_final_decoder_norm:
			raise ValueError("late fusion must run after the final decoder norm")
		if self.auxiliary_output_dim != 256:
			raise ValueError("auxiliary retrieval embeddings must have dimension 256")
		if (
			self.auxiliary_pooling != "eos_conditioned_weighted_slots"
			or self.auxiliary_normalization != "rmsnorm"
			or self.auxiliary_bias
		):
			raise ValueError(
				"auxiliary head must use EOS-conditioned weighted slots, RMSNorm, "
				"and no bias",
			)
		if self.temperature != 0.02:
			raise ValueError("InfoNCE temperature must remain 0.02")
		if (
			self.final_infonce_weight,
			self.loop_infonce_weight,
			self.slot_diversity_weight,
		) != (1.0, 0.1, 0.05):
			raise ValueError("loss weights must remain 1.0, 0.1, and 0.05")

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
		fusion = _mapping(value, "fusion")
		auxiliary = _mapping(value, "auxiliary_head")
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
			fusion_type=str(fusion["type"]),
			fusion_attention_dim=int(fusion["attention_dim"]),
			fusion_num_heads=int(fusion["num_heads"]),
			fusion_dropout=float(fusion["dropout"]),
			fusion_residual_gate_init=float(fusion["residual_gate_init"]),
			fuse_after_final_decoder_norm=bool(fusion["fuse_after_final_decoder_norm"]),
			auxiliary_output_dim=int(auxiliary["output_dim"]),
			auxiliary_pooling=str(auxiliary["pooling"]),
			auxiliary_normalization=str(auxiliary["normalization"]),
			auxiliary_bias=bool(auxiliary["bias"]),
			temperature=float(loss["temperature"]),
			final_infonce_weight=float(loss["final_infonce_weight"]),
			loop_infonce_weight=float(loss["loop_infonce_weight"]),
			slot_diversity_weight=float(loss["slot_diversity_weight"]),
		)
		config.validate()
		return config


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
	nested = value.get(key)
	if not isinstance(nested, dict):
		raise ValueError(f"Configuration key {key} must be a mapping")
	return nested
