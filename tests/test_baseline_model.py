from __future__ import annotations

from looped_vl.baseline.model import (
	BASELINE_LORA_ALPHA,
	BASELINE_LORA_RANK,
	BASELINE_LORA_TARGETS,
	build_lora_config,
)


def test_baseline_lora_matches_official_qwen_embedding_configuration() -> None:
	config = build_lora_config()

	assert BASELINE_LORA_RANK == 32
	assert BASELINE_LORA_ALPHA == 32
	assert BASELINE_LORA_TARGETS == (
		"q_proj",
		"v_proj",
		"k_proj",
		"up_proj",
		"down_proj",
		"gate_proj",
	)
	assert config.r == 32
	assert config.lora_alpha == 32
	assert config.target_modules == set(BASELINE_LORA_TARGETS)
	assert config.lora_dropout == 0.0
