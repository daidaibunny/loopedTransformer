"""Validated optimizer configuration for one continuous training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
	"""Optimization values for a one-epoch run with an initial warm-start window."""

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
	warm_start_epoch_fraction: float
	joint_activation_warmup_ratio: float

	@classmethod
	def from_yaml(cls, path: str | Path) -> TrainingConfig:
		"""Load and validate the single training configuration."""
		value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
		if not isinstance(value, dict):
			raise ValueError("Training configuration must be a YAML mapping")
		config = cls(
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
			warm_start_epoch_fraction=float(value["warm_start_epoch_fraction"]),
			joint_activation_warmup_ratio=float(
				value["joint_activation_warmup_ratio"],
			),
		)
		config.validate()
		return config

	def validate(self) -> None:
		"""Enforce the fixed optimizer and dynamic warm-start protocol."""
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
		if self.warm_start_epoch_fraction != 0.35:
			raise ValueError("warm-start window must cover exactly 0.35 epoch")
		if self.joint_activation_warmup_ratio != 0.03:
			raise ValueError("joint parameter activation warmup ratio must be 0.03")

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
