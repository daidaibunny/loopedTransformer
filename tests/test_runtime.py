import pytest
import torch

from looped_vl.runtime import (
	resolve_attention_implementation,
	resolve_torch_dtype,
)


def test_v100_auto_attention_uses_sdpa() -> None:
	assert resolve_attention_implementation(
		"auto",
		compute_capability=(7, 0),
		flash_attention_available=True,
	) == "sdpa"


def test_ampere_auto_attention_uses_flash_attention_when_installed() -> None:
	assert resolve_attention_implementation(
		"auto",
		compute_capability=(8, 0),
		flash_attention_available=True,
	) == "flash_attention_2"


def test_auto_attention_falls_back_to_sdpa_without_flash_attention() -> None:
	assert resolve_attention_implementation(
		"auto",
		compute_capability=(9, 0),
		flash_attention_available=False,
	) == "sdpa"


def test_explicit_flash_attention_rejects_v100() -> None:
	with pytest.raises(ValueError, match="Ampere or newer"):
		resolve_attention_implementation(
			"flash_attention_2",
			compute_capability=(7, 0),
			flash_attention_available=True,
		)


def test_runtime_precision_names_resolve_to_torch_dtypes() -> None:
	assert resolve_torch_dtype("bf16") is torch.bfloat16
	assert resolve_torch_dtype("fp16") is torch.float16

	with pytest.raises(ValueError, match="Unsupported runtime precision"):
		resolve_torch_dtype("fp32")
