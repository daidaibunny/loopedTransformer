"""Locked configuration for parallel-world query recurrence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

QUERY_RECURRENT_ARCHITECTURE = "query_only_parallel_world_recurrent_no_lora_v11"
QUERY_RECURRENT_PROTOCOL = "single_stage_antithetic_final_mean_v11"
MAX_QUERY_RECURRENT_PARAMETERS = 5_000_000
SUPPORTED_WORLD_COUNTS = (1, 2, 4)
SUPPORTED_RECURRENT_STEPS = (1, 2, 3, 4)


@dataclass(frozen=True)
class QueryRecurrentConfig:
	"""Define one shared recurrent population Block over the frozen final embedding."""

	hidden_size: int = 2048
	attention_size: int = 320
	num_attention_heads: int = 5
	feed_forward_size: int = 288
	num_worlds: int = 4
	max_recurrent_steps: int = 4
	perturbation_scale: float = 0.02
	maximum_residual_scale: float = 0.25
	initial_residual_scale: float = 0.01
	temperature: float = 0.02
	pass_supervision: str = "final_only"
	hard_negative_count: int = 32
	seed: int = 42

	def validate(self) -> None:
		"""Reject settings outside the first parallel-world experiment contract."""
		if self.hidden_size != 2048:
			raise ValueError("Parallel-world hidden size must be 2048")
		if self.attention_size <= 0 or self.attention_size % self.num_attention_heads:
			raise ValueError(
				"attention_size must be positive and divisible by num_attention_heads",
			)
		if self.feed_forward_size <= 0:
			raise ValueError("feed_forward_size must be positive")
		if self.num_worlds not in SUPPORTED_WORLD_COUNTS:
			raise ValueError(f"num_worlds must be one of {SUPPORTED_WORLD_COUNTS}")
		if self.max_recurrent_steps not in SUPPORTED_RECURRENT_STEPS:
			raise ValueError(
				f"max_recurrent_steps must be one of {SUPPORTED_RECURRENT_STEPS}",
			)
		if not 0 < self.perturbation_scale < 1:
			raise ValueError("perturbation_scale must be in (0, 1)")
		if not 0 < self.maximum_residual_scale <= 1:
			raise ValueError("maximum_residual_scale must be in (0, 1]")
		if not 0 < self.initial_residual_scale < self.maximum_residual_scale:
			raise ValueError(
				"initial_residual_scale must be below maximum_residual_scale",
			)
		if self.temperature <= 0:
			raise ValueError("temperature must be positive")
		if self.pass_supervision != "final_only":
			raise ValueError("pass_supervision must be final_only")
		if self.hard_negative_count < 0:
			raise ValueError("hard_negative_count cannot be negative")

	def with_variant(self, **changes: Any) -> QueryRecurrentConfig:
		"""Return one validated immutable ablation variant."""
		variant = replace(self, **changes)
		variant.validate()
		return variant

	def identity(self) -> dict[str, Any]:
		"""Return every result-affecting architecture field for manifests."""
		self.validate()
		return {
			"architecture": QUERY_RECURRENT_ARCHITECTURE,
			"protocol": QUERY_RECURRENT_PROTOCOL,
			"backbone_frozen": True,
			"candidate_backbone_executed": False,
			"lora_enabled": False,
			"formal_training_stages": 1,
			"initialization": "query_conditioned_antithetic_zero_mean",
			"world_interaction": "shared_centered_bidirectional_attention",
			"readout": "final_world_mean_l2_normalized",
			"dynamic_exit": False,
			"recurrent_step_embeddings": False,
			**asdict(self),
		}
