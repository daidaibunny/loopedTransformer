"""Locked configuration for the query-only recurrent retrieval model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

QUERY_RECURRENT_ARCHITECTURE = "query_only_history_recurrent_no_lora_v2"
QUERY_RECURRENT_PROTOCOL = "single_stage_directional_hard_negative_pass_supervision_v2"
MAX_QUERY_RECURRENT_PARAMETERS = 5_000_000
DEFAULT_HISTORY_LAYERS = (7, 14, 21, 28)
SUPPORTED_SLOT_COUNTS = (1, 4, 8)
SUPPORTED_RECURRENT_STEPS = (1, 4)
SUPPORTED_EXIT_MODES = ("fixed", "dynamic")


@dataclass(frozen=True)
class QueryRecurrentConfig:
	"""Define one parameter-bounded recurrent head over frozen Qwen histories."""

	hidden_size: int = 2048
	state_size: int = 288
	num_attention_heads: int = 8
	feed_forward_multiplier: int = 4
	recurrent_block_layers: int = 2
	num_slots: int = 8
	max_recurrent_steps: int = 4
	history_layers: tuple[int, ...] = DEFAULT_HISTORY_LAYERS
	exit_mode: str = "fixed"
	exit_threshold: float = 0.5
	temperature: float = 0.02
	direct_pass_loss_weight: float = 1.0
	dynamic_loss_weight: float = 0.5
	progressive_loss_weight: float = 0.1
	progressive_margin: float = 0.02
	compute_penalty_weight: float = 0.01
	hard_negative_count: int = 32
	seed: int = 42

	def validate(self) -> None:
		"""Reject settings outside the first formal query-only ablation contract."""
		if self.hidden_size != 2048:
			raise ValueError("Query-only recurrent hidden size must be 2048")
		if self.state_size <= 0 or self.state_size % self.num_attention_heads:
			raise ValueError("state_size must be positive and divisible by attention heads")
		if self.feed_forward_multiplier <= 0 or self.recurrent_block_layers <= 0:
			raise ValueError("Recurrent block dimensions must be positive")
		if self.num_slots not in SUPPORTED_SLOT_COUNTS:
			raise ValueError(f"num_slots must be one of {SUPPORTED_SLOT_COUNTS}")
		if self.max_recurrent_steps not in SUPPORTED_RECURRENT_STEPS:
			raise ValueError(
				f"max_recurrent_steps must be one of {SUPPORTED_RECURRENT_STEPS}",
			)
		if self.exit_mode not in SUPPORTED_EXIT_MODES:
			raise ValueError(f"exit_mode must be one of {SUPPORTED_EXIT_MODES}")
		if self.exit_mode == "dynamic" and self.max_recurrent_steps <= 1:
			raise ValueError("Dynamic exit requires more than one recurrent step")
		if not self.history_layers:
			raise ValueError("At least one frozen Qwen history layer is required")
		if tuple(sorted(set(self.history_layers))) != self.history_layers:
			raise ValueError("history_layers must be sorted and unique")
		if self.history_layers[0] < 1 or self.history_layers[-1] > 28:
			raise ValueError("history_layers must use one-indexed decoder layers 1 through 28")
		if not 0.0 < self.exit_threshold < 1.0:
			raise ValueError("exit_threshold must be in (0, 1)")
		if self.temperature <= 0:
			raise ValueError("temperature must be positive")
		for name in (
			"direct_pass_loss_weight",
			"dynamic_loss_weight",
			"progressive_loss_weight",
			"progressive_margin",
			"compute_penalty_weight",
		):
			if getattr(self, name) < 0:
				raise ValueError(f"{name} cannot be negative")
		if self.direct_pass_loss_weight == 0:
			raise ValueError("direct_pass_loss_weight must be positive")
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
			**asdict(self),
		}
