"""Hardware-aware attention and precision selection."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch

ATTENTION_IMPLEMENTATIONS = ("auto", "flash_attention_2", "sdpa", "eager")
RUNTIME_PRECISIONS = ("bf16", "fp16")


@dataclass(frozen=True)
class TrainingPrecision:
	"""Parameter storage and automatic mixed-precision settings for training."""

	parameter_dtype: torch.dtype
	trainable_parameter_dtype: torch.dtype
	autocast_dtype: torch.dtype
	autocast_enabled: bool
	gradient_scaling_enabled: bool


def resolve_attention_implementation(
	requested: str,
	*,
	compute_capability: tuple[int, int] | None = None,
	flash_attention_available: bool | None = None,
) -> str:
	"""Resolve a supported attention backend without selecting FlashAttention on Volta."""
	if requested not in ATTENTION_IMPLEMENTATIONS:
		raise ValueError(f"Unsupported attention implementation: {requested}")
	if compute_capability is None:
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA is required to resolve the attention implementation")
		compute_capability = torch.cuda.get_device_capability()
	if flash_attention_available is None:
		flash_attention_available = importlib.util.find_spec("flash_attn") is not None
	flash_attention_supported = compute_capability[0] >= 8
	if requested == "flash_attention_2":
		if not flash_attention_supported:
			raise ValueError(
				"FlashAttention 2 requires an NVIDIA Ampere or newer GPU; "
				f"found compute capability {compute_capability[0]}.{compute_capability[1]}",
			)
		if not flash_attention_available:
			raise RuntimeError("FlashAttention 2 was requested but flash_attn is not installed")
		return requested
	if requested == "auto":
		if flash_attention_supported and flash_attention_available:
			return "flash_attention_2"
		return "sdpa"
	return requested


def resolve_torch_dtype(precision: str) -> torch.dtype:
	"""Map the explicit runtime precision name to its Torch dtype."""
	if precision == "bf16":
		return torch.bfloat16
	if precision == "fp16":
		return torch.float16
	raise ValueError(f"Unsupported runtime precision: {precision}")


def resolve_training_precision(precision: str) -> TrainingPrecision:
	"""Keep FP16 trainable weights in FP32 while using Volta Tensor Cores."""
	if precision == "bf16":
		return TrainingPrecision(
			parameter_dtype=torch.bfloat16,
			trainable_parameter_dtype=torch.bfloat16,
			autocast_dtype=torch.bfloat16,
			autocast_enabled=False,
			gradient_scaling_enabled=False,
		)
	if precision == "fp16":
		return TrainingPrecision(
			parameter_dtype=torch.float16,
			trainable_parameter_dtype=torch.float32,
			autocast_dtype=torch.float16,
			autocast_enabled=True,
			gradient_scaling_enabled=True,
		)
	raise ValueError(f"Unsupported runtime precision: {precision}")
