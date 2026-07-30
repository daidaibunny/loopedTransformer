"""Official Qwen3-VL-Embedding formatting and preprocessing with a latent token."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

LOGGER = logging.getLogger(__name__)
LATENT_SLOT_TOKEN = "<|latent_slot|>"


class RecurrentInputProcessor:
	"""Match official embedding preprocessing and expose the latent placeholder ID."""

	def __init__(
		self,
		processor: Qwen3VLProcessor,
		max_length: int = 8192,
		min_pixels: int = 4 * 32 * 32,
		max_pixels: int = 1800 * 32 * 32,
		default_instruction: str = "Represent the user's input.",
	) -> None:
		self.processor = processor
		self.max_length = max_length
		self.min_pixels = min_pixels
		self.max_pixels = max_pixels
		self.default_instruction = default_instruction
		self.processor.tokenizer.add_special_tokens(
			{"additional_special_tokens": [LATENT_SLOT_TOKEN]},
		)
		self.latent_placeholder_id = self.processor.tokenizer.convert_tokens_to_ids(
			LATENT_SLOT_TOKEN,
		)
		if self.latent_placeholder_id == self.processor.tokenizer.unk_token_id:
			raise RuntimeError("Latent placeholder token was not registered")
		if self.processor.tokenizer.pad_token_id is None:
			raise RuntimeError("Qwen tokenizer must define a pad token")

	@property
	def pad_token_id(self) -> int:
		"""Return the official tokenizer padding ID."""
		return int(self.processor.tokenizer.pad_token_id)

	@classmethod
	def from_pretrained(
		cls,
		model_root: str | Path,
		max_length: int = 8192,
		min_pixels: int = 4 * 32 * 32,
		max_pixels: int = 1800 * 32 * 32,
	) -> RecurrentInputProcessor:
		"""Load the pinned local Qwen processor with right padding."""
		processor = Qwen3VLProcessor.from_pretrained(
			str(model_root),
			padding_side="right",
		)
		return cls(
			processor=processor,
			max_length=max_length,
			min_pixels=min_pixels,
			max_pixels=max_pixels,
		)

	def format_model_input(
		self,
		*,
		text: str | None = None,
		image: str | Image.Image | None = None,
		instruction: str | None = None,
	) -> list[dict[str, Any]]:
		"""Create the same system/user conversation as the official embedder."""
		normalized_instruction = instruction
		if normalized_instruction:
			normalized_instruction = normalized_instruction.strip()
			if normalized_instruction and not unicodedata.category(
				normalized_instruction[-1],
			).startswith("P"):
				normalized_instruction += "."
		content: list[dict[str, Any]] = []
		conversation = [
			{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": normalized_instruction or self.default_instruction,
					},
				],
			},
			{"role": "user", "content": content},
		]
		if image is not None:
			if isinstance(image, Image.Image):
				image_content: str | Image.Image = image
			elif isinstance(image, str):
				image_content = image if image.startswith(("http", "oss")) else f"file://{image}"
			else:
				raise TypeError(f"Unsupported image type: {type(image)}")
			content.append(
				{
					"type": "image",
					"image": image_content,
					"min_pixels": self.min_pixels,
					"max_pixels": self.max_pixels,
				},
			)
		if text:
			content.append({"type": "text", "text": text})
		if not content:
			content.append({"type": "text", "text": "NULL"})
		return conversation

	def prepare(
		self,
		model_inputs: list[dict[str, Any]],
		device: torch.device | None = None,
	) -> dict[str, torch.Tensor]:
		"""Process a batch with official chat, vision, resize, and tokenization logic."""
		conversations = [self.format_model_input(**model_input) for model_input in model_inputs]
		texts = self.processor.apply_chat_template(
			conversations,
			add_generation_prompt=True,
			tokenize=False,
		)
		images, video_inputs, video_kwargs = process_vision_info(
			conversations,
			image_patch_size=16,
			return_video_metadata=True,
			return_video_kwargs=True,
		)
		if video_inputs is not None:
			videos, video_metadata = zip(*video_inputs, strict=True)
			videos = list(videos)
			video_metadata = list(video_metadata)
		else:
			videos, video_metadata = None, None
		processed = self.processor(
			text=texts,
			images=images,
			videos=videos,
			video_metadata=video_metadata,
			truncation=True,
			max_length=self.max_length,
			padding=True,
			do_resize=False,
			return_tensors="pt",
			**video_kwargs,
		)
		result = {key: value for key, value in processed.items() if isinstance(value, torch.Tensor)}
		if device is not None:
			result = {
				key: value.to(device=device, non_blocking=True)
				for key, value in result.items()
			}
		return result
