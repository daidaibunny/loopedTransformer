"""Validated optimizer configuration for each fixed training stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingStageConfig:
	"""The exact v1.0 schedule for Stage 1 or Stage 2."""

	stage: int
	steps: int
	optimizer: str
	learning_rate: float
	weight_decay: float
	betas: tuple[float, float]
	eps: float
	effective_batch_size: int
	gradient_clip_norm: float
	precision: str
	lr_scheduler: str
	warmup_ratio: float

	@classmethod
	def from_yaml(cls, path: str | Path) -> TrainingStageConfig:
		"""Load and validate one stage file."""
		value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
		if not isinstance(value, dict):
			raise ValueError("Stage configuration must be a YAML mapping")
		config = cls(
			stage=int(value["stage"]),
			steps=int(value["steps"]),
			optimizer=str(value["optimizer"]),
			learning_rate=float(value["learning_rate"]),
			weight_decay=float(value["weight_decay"]),
			betas=tuple(float(item) for item in value["betas"]),
			eps=float(value["eps"]),
			effective_batch_size=int(value["effective_batch_size"]),
			gradient_clip_norm=float(value["gradient_clip_norm"]),
			precision=str(value["precision"]),
			lr_scheduler=str(value["lr_scheduler"]),
			warmup_ratio=float(value["warmup_ratio"]),
		)
		config.validate()
		return config

	def validate(self) -> None:
		"""Enforce all fixed optimizer values from specification v1.0."""
		expected_steps = {1: 2000, 2: 3200}
		if self.stage not in expected_steps or self.steps != expected_steps[self.stage]:
			raise ValueError("Stage 1/2 must use exactly 2000/3200 optimizer steps")
		if self.optimizer != "AdamW":
			raise ValueError("optimizer must be AdamW")
		if self.learning_rate != 1e-5 or self.weight_decay != 0.01:
			raise ValueError("learning rate and weight decay must be 1e-5 and 0.01")
		if self.betas != (0.9, 0.95) or self.eps != 1e-8:
			raise ValueError("AdamW betas and epsilon do not match v1.0")
		if self.effective_batch_size != 512:
			raise ValueError("effective batch size must remain 512")
		if self.gradient_clip_norm != 1.0:
			raise ValueError("gradient clip norm must remain 1.0")
		if self.precision != "bf16":
			raise ValueError("precision must remain bf16")
		if self.lr_scheduler != "cosine" or self.warmup_ratio != 0.03:
			raise ValueError("scheduler must be cosine with warmup ratio 0.03")

	def gradient_accumulation_steps(
		self,
		per_device_batch_size: int,
		world_size: int,
	) -> int:
		"""Derive exact accumulation required to reach effective batch size 512."""
		global_micro_batch = per_device_batch_size * world_size
		if global_micro_batch <= 0 or self.effective_batch_size % global_micro_batch:
			raise ValueError(
				"per-device batch size times world size must divide effective batch size 512",
			)
		return self.effective_batch_size // global_micro_batch
