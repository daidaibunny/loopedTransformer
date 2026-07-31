"""Exact recurrent latent-slot forward path for Qwen3-VL-Embedding-2B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
	apply_rotary_pos_emb,
	repeat_kv,
)

from looped_vl.models.config import RecurrentModelConfig
from looped_vl.models.late_slot_fusion import EOSConditionedSlotFusion
from looped_vl.models.latent_slot_inserter import (
	AugmentedSequence,
	augment_before_last_valid_token,
)
from looped_vl.models.recurrent_decoder_block import (
	build_dynamic_attention_mask,
	detach_prefix_key_values,
)
from looped_vl.models.warmup_heads import AuxiliarySlotRetrievalHead


@dataclass(frozen=True)
class PrefixKeyValue:
	"""One loop layer's detached, already-rotated prefix evidence."""

	key: torch.Tensor
	value: torch.Tensor


@dataclass(frozen=True)
class RecurrentEmbeddingOutput:
	"""Normalized retrieval output plus contextual slots and diagnostics."""

	embeddings: torch.Tensor
	loop_embeddings: tuple[torch.Tensor, ...] | None
	loop_slot_hidden_states: tuple[torch.Tensor, ...]
	slot_hidden_states: torch.Tensor
	eos_hidden_state: torch.Tensor
	attention_weights: torch.Tensor | None
	diagnostics: dict[str, Any]


def _dynamic_scaled_dot_product_attention(
	*,
	query: torch.Tensor,
	key: torch.Tensor,
	value: torch.Tensor,
	attention_mask: torch.Tensor,
	scale: float,
) -> torch.Tensor:
	"""Use PyTorch's fused attention dispatcher for the recurrent query block."""
	return F.scaled_dot_product_attention(
		query,
		key,
		value,
		attn_mask=attention_mask,
		dropout_p=0.0,
		is_causal=False,
		scale=scale,
	)


def damped_recurrent_update(
	previous_states: torch.Tensor,
	proposed_states: torch.Tensor,
	*,
	total_passes: int,
) -> torch.Tensor:
	"""Advance recurrent slots with the fixed step size alpha = 1 / R."""
	if previous_states.shape != proposed_states.shape:
		raise ValueError("Previous and proposed recurrent slot states must have equal shapes")
	if total_passes <= 0:
		raise ValueError("total_passes must be positive")
	step_size = 1.0 / total_passes
	return previous_states + step_size * (proposed_states - previous_states)


def _gather_sequence_positions(
	hidden_states: torch.Tensor,
	positions: torch.Tensor,
) -> torch.Tensor:
	"""Gather per-sample sequence positions without assuming equal valid lengths."""
	batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
	if positions.ndim == 1:
		return hidden_states[batch_index, positions]
	return hidden_states[batch_index[:, None], positions]


def _scatter_sequence_positions(
	hidden_states: torch.Tensor,
	positions: torch.Tensor,
	values: torch.Tensor,
) -> torch.Tensor:
	"""Return a cloned sequence with per-sample positions replaced by values."""
	result = hidden_states.clone()
	batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
	if positions.ndim == 1:
		result[batch_index, positions] = values
	else:
		result[batch_index[:, None], positions] = values
	return result


def _pairwise_slot_cosine(slot_hidden_states: torch.Tensor) -> torch.Tensor:
	"""Return the mean absolute off-diagonal slot cosine for logging."""
	if slot_hidden_states.shape[1] <= 1:
		return slot_hidden_states.new_zeros(())
	normalized = F.normalize(slot_hidden_states.float(), p=2, dim=-1)
	cosine = normalized @ normalized.transpose(1, 2)
	slot_count = slot_hidden_states.shape[1]
	off_diagonal = ~torch.eye(slot_count, dtype=torch.bool, device=cosine.device)
	return cosine[:, off_diagonal].abs().mean()


def _run_full_sequence_decoder_layer(
	*,
	layer: nn.Module,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor | None,
	position_ids: torch.Tensor,
	cache_position: torch.Tensor,
	position_embeddings: tuple[torch.Tensor, torch.Tensor],
	activation_checkpointing: bool,
) -> torch.Tensor:
	"""Run one full-sequence layer, optionally recomputing it during backward."""

	def layer_forward(states: torch.Tensor) -> torch.Tensor:
		return layer(
			states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=None,
			cache_position=cache_position,
			position_embeddings=position_embeddings,
		)

	if activation_checkpointing and torch.is_grad_enabled():
		return checkpoint(layer_forward, hidden_states, use_reentrant=False)
	return layer_forward(hidden_states)


class RecurrentQwen3VLEmbedding(nn.Module):
	"""Wrap the official embedding model with dynamic-only middle-layer recurrence."""

	def __init__(
		self,
		base_embedding_model: nn.Module,
		config: RecurrentModelConfig,
		master_slot_initialization: torch.Tensor,
		latent_placeholder_id: int,
		pad_token_id: int,
	) -> None:
		super().__init__()
		config.validate()
		if tuple(master_slot_initialization.shape) != (
			1,
			config.max_num_latent_slots,
			config.hidden_size,
		):
			raise ValueError("Master slot tensor does not match [1, 16, 2048]")
		self.config = config
		self.base_embedding_model = base_embedding_model
		self.latent_placeholder_id = latent_placeholder_id
		self.pad_token_id = pad_token_id
		self.latent_slots = nn.Parameter(
			master_slot_initialization[:, : config.num_latent_slots].clone(),
		)
		self.eos_delta = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
		self.late_fusion = EOSConditionedSlotFusion(
			hidden_size=config.hidden_size,
			attention_dim=config.fusion_attention_dim,
		)
		self.auxiliary_embedding_head = AuxiliarySlotRetrievalHead(
			hidden_size=config.hidden_size,
			output_size=config.auxiliary_output_dim,
		)
		self.activation_checkpointing_enabled = False

	@property
	def multimodal_model(self) -> nn.Module:
		"""Return the official Qwen3-VL multimodal backbone."""
		return self.base_embedding_model.model

	@property
	def language_model(self) -> nn.Module:
		"""Return the official 28-layer language decoder."""
		return self.multimodal_model.language_model

	def set_activation_checkpointing(self, enabled: bool) -> None:
		"""Enable full-sequence decoder recomputation for memory-safe training."""
		self.activation_checkpointing_enabled = enabled

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		pixel_values: torch.Tensor | None = None,
		pixel_values_videos: torch.Tensor | None = None,
		image_grid_thw: torch.Tensor | None = None,
		video_grid_thw: torch.Tensor | None = None,
		return_all_loop_embeddings: bool = False,
	) -> RecurrentEmbeddingOutput:
		"""Encode one tower with either the exact base or damped recurrent path."""
		if self.config.num_latent_slots == 0 and self.config.num_total_loop_passes == 1:
			output = self._official_base_forward(
				input_ids=input_ids,
				attention_mask=attention_mask,
				pixel_values=pixel_values,
				pixel_values_videos=pixel_values_videos,
				image_grid_thw=image_grid_thw,
				video_grid_thw=video_grid_thw,
			)
			if return_all_loop_embeddings:
				return RecurrentEmbeddingOutput(
					embeddings=output.embeddings,
					loop_embeddings=(output.embeddings,),
					loop_slot_hidden_states=output.loop_slot_hidden_states,
					slot_hidden_states=output.slot_hidden_states,
					eos_hidden_state=output.eos_hidden_state,
					attention_weights=output.attention_weights,
					diagnostics=output.diagnostics,
				)
			return output
		augmented = augment_before_last_valid_token(
			input_ids=input_ids,
			attention_mask=attention_mask,
			num_latent_slots=self.config.num_latent_slots,
			latent_placeholder_id=self.latent_placeholder_id,
			pad_token_id=self.pad_token_id,
		)
		(
			hidden_states,
			position_ids,
			visual_position_mask,
			deepstack_visual_embeddings,
		) = self._prepare_augmented_embeddings(
			augmented=augmented,
			pixel_values=pixel_values,
			pixel_values_videos=pixel_values_videos,
			image_grid_thw=image_grid_thw,
			video_grid_thw=video_grid_thw,
		)
		return self._run_recurrent_decoder(
			hidden_states=hidden_states,
			augmented=augmented,
			position_ids=position_ids,
			visual_position_mask=visual_position_mask,
			deepstack_visual_embeddings=deepstack_visual_embeddings,
			return_all_loop_embeddings=return_all_loop_embeddings,
		)

	def _official_base_forward(self, **inputs: torch.Tensor | None) -> RecurrentEmbeddingOutput:
		"""Delegate K=0, R=1 to the official code path for numerical equivalence."""
		attention_mask = inputs["attention_mask"]
		if attention_mask is None:
			raise ValueError("attention_mask is required")
		outputs = self.base_embedding_model(**inputs)
		hidden_states = outputs.last_hidden_state
		eos_positions = attention_mask.to(torch.long).sum(dim=-1) - 1
		eos_hidden_state = _gather_sequence_positions(hidden_states, eos_positions)
		embeddings = F.normalize(eos_hidden_state, p=2, dim=-1)
		return RecurrentEmbeddingOutput(
			embeddings=embeddings,
			loop_embeddings=None,
			loop_slot_hidden_states=(),
			slot_hidden_states=hidden_states[:, :0],
			eos_hidden_state=eos_hidden_state,
			attention_weights=None,
			diagnostics={
				"variant": "base",
				"deepstack_layer_indices": (0, 1, 2),
				"extra_pass_dynamic_token_counts": (),
			},
		)

	def _prepare_augmented_embeddings(
		self,
		augmented: AugmentedSequence,
		pixel_values: torch.Tensor | None,
		pixel_values_videos: torch.Tensor | None,
		image_grid_thw: torch.Tensor | None,
		video_grid_thw: torch.Tensor | None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
		"""Use official vision, placeholder, and MRoPE routines on the augmented sequence."""
		model = self.multimodal_model
		input_ids = augmented.input_ids
		inputs_embeds = model.get_input_embeddings()(input_ids)
		image_mask = None
		video_mask = None
		deepstack_image_embeddings = None
		deepstack_video_embeddings = None
		if pixel_values is not None:
			image_embeddings, deepstack_image_embeddings = model.get_image_features(
				pixel_values,
				image_grid_thw,
			)
			image_embeddings = torch.cat(image_embeddings, dim=0).to(
				inputs_embeds.device,
				inputs_embeds.dtype,
			)
			image_mask, _ = model.get_placeholder_mask(
				input_ids,
				inputs_embeds=inputs_embeds,
				image_features=image_embeddings,
			)
			inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeddings)
		if pixel_values_videos is not None:
			video_embeddings, deepstack_video_embeddings = model.get_video_features(
				pixel_values_videos,
				video_grid_thw,
			)
			video_embeddings = torch.cat(video_embeddings, dim=0).to(
				inputs_embeds.device,
				inputs_embeds.dtype,
			)
			_, video_mask = model.get_placeholder_mask(
				input_ids,
				inputs_embeds=inputs_embeds,
				video_features=video_embeddings,
			)
			inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeddings)
		visual_position_mask, deepstack_embeddings = self._merge_deepstack_inputs(
			inputs_embeds,
			image_mask,
			video_mask,
			deepstack_image_embeddings,
			deepstack_video_embeddings,
		)
		position_ids, _ = model.get_rope_index(
			input_ids,
			image_grid_thw,
			video_grid_thw,
			attention_mask=augmented.attention_mask,
		)
		if self.config.num_latent_slots:
			slots = self.latent_slots[:, : self.config.num_latent_slots].expand(
				input_ids.shape[0],
				-1,
				-1,
			).to(inputs_embeds.dtype)
			inputs_embeds = _scatter_sequence_positions(
				inputs_embeds,
				augmented.slot_positions,
				slots,
			)
		eos_values = _gather_sequence_positions(inputs_embeds, augmented.eos_positions)
		eos_values = eos_values + self.eos_delta[0, 0].to(inputs_embeds.dtype)
		inputs_embeds = _scatter_sequence_positions(
			inputs_embeds,
			augmented.eos_positions,
			eos_values,
		)
		return inputs_embeds, position_ids, visual_position_mask, deepstack_embeddings

	@staticmethod
	def _merge_deepstack_inputs(
		inputs_embeds: torch.Tensor,
		image_mask: torch.Tensor | None,
		video_mask: torch.Tensor | None,
		deepstack_image_embeddings: list[torch.Tensor] | None,
		deepstack_video_embeddings: list[torch.Tensor] | None,
	) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
		"""Match the official image/video DeepStack merge without moving its layers."""
		if image_mask is not None and video_mask is not None:
			flat_image_mask = image_mask[..., 0]
			flat_video_mask = video_mask[..., 0]
			visual_position_mask = flat_image_mask | flat_video_mask
			if deepstack_image_embeddings is None or deepstack_video_embeddings is None:
				raise RuntimeError("Missing DeepStack image or video embeddings")
			joint_embeddings: list[torch.Tensor] = []
			image_joint_mask = flat_image_mask[visual_position_mask]
			video_joint_mask = flat_video_mask[visual_position_mask]
			for image_embedding, video_embedding in zip(
				deepstack_image_embeddings,
				deepstack_video_embeddings,
				strict=True,
			):
				joint = image_embedding.new_zeros(
					visual_position_mask.sum(),
					image_embedding.shape[-1],
				).to(inputs_embeds.device)
				joint[image_joint_mask] = image_embedding
				joint[video_joint_mask] = video_embedding
				joint_embeddings.append(joint)
			return visual_position_mask, joint_embeddings
		if image_mask is not None:
			return image_mask[..., 0], deepstack_image_embeddings
		if video_mask is not None:
			return video_mask[..., 0], deepstack_video_embeddings
		return None, None

	def _run_recurrent_decoder(
		self,
		hidden_states: torch.Tensor,
		augmented: AugmentedSequence,
		position_ids: torch.Tensor,
		visual_position_mask: torch.Tensor | None,
		deepstack_visual_embeddings: list[torch.Tensor] | None,
		return_all_loop_embeddings: bool,
	) -> RecurrentEmbeddingOutput:
		"""Run prefix, full first pass, dynamic-only passes, suffix, norm, and fusion."""
		language_model = self.language_model
		sequence_length = hidden_states.shape[1]
		cache_position = torch.arange(sequence_length, device=hidden_states.device)
		text_position_ids = position_ids[0]
		causal_mask = create_causal_mask(
			config=language_model.config,
			input_embeds=hidden_states,
			attention_mask=augmented.attention_mask,
			cache_position=cache_position,
			past_key_values=None,
			position_ids=text_position_ids,
		)
		position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
		deepstack_layers_executed: list[int] = []
		for layer_index in range(self.config.loop_start_layer):
			hidden_states = _run_full_sequence_decoder_layer(
				layer=language_model.layers[layer_index],
				hidden_states=hidden_states,
				attention_mask=causal_mask,
				position_ids=text_position_ids,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				activation_checkpointing=(
					self.activation_checkpointing_enabled and self.training
				),
			)
			if (
				deepstack_visual_embeddings is not None
				and layer_index < len(deepstack_visual_embeddings)
			):
				hidden_states = language_model._deepstack_process(
					hidden_states,
					visual_position_mask,
					deepstack_visual_embeddings[layer_index],
				)
				deepstack_layers_executed.append(layer_index)

		slot_positions = augmented.slot_positions
		initial_slot_states = _gather_sequence_positions(hidden_states, slot_positions)
		max_prefix_length = sequence_length - slot_positions.shape[1] - 1
		prefix_caches: list[PrefixKeyValue] = []
		for layer_index in range(self.config.loop_start_layer, self.config.loop_end_layer):
			layer = language_model.layers[layer_index]
			if self.config.num_extra_loop_passes:
				hidden_states, prefix_cache = self._run_full_layer_and_capture_prefix(
					layer=layer,
					hidden_states=hidden_states,
					position_embeddings=position_embeddings,
					max_prefix_length=max_prefix_length,
					attention_mask=causal_mask,
					position_ids=text_position_ids,
					cache_position=cache_position,
					activation_checkpointing=(
						self.activation_checkpointing_enabled and self.training
					),
				)
				prefix_caches.append(prefix_cache)
			else:
				hidden_states = _run_full_sequence_decoder_layer(
					layer=layer,
					hidden_states=hidden_states,
					attention_mask=causal_mask,
					position_ids=text_position_ids,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
					activation_checkpointing=(
						self.activation_checkpointing_enabled and self.training
					),
				)
		pass_one_full_hidden_states = hidden_states
		pass_one_proposed_slots = _gather_sequence_positions(hidden_states, slot_positions)
		slot_states = damped_recurrent_update(
			initial_slot_states,
			pass_one_proposed_slots,
			total_passes=self.config.num_total_loop_passes,
		)
		loop_slot_hidden_states = [slot_states]

		slot_cos, slot_sin = self._gather_dynamic_position_embeddings(
			position_embeddings,
			slot_positions,
		)
		prefix_mask = (
			torch.arange(max_prefix_length, device=hidden_states.device)[None, :]
			< augmented.prefix_lengths[:, None]
		)
		slot_attention_mask = build_dynamic_attention_mask(
			prefix_mask,
			dynamic_token_count=slot_positions.shape[1],
			dtype=hidden_states.dtype,
		)
		pass_cosines: list[torch.Tensor] = [
			F.cosine_similarity(
				slot_states.float().flatten(1),
				initial_slot_states.float().flatten(1),
				dim=-1,
			).mean(),
		]
		pass_relative_updates: list[torch.Tensor] = [
			(
				(slot_states.float() - initial_slot_states.float()).flatten(1).norm(dim=-1)
				/ initial_slot_states.float().flatten(1).norm(dim=-1).clamp_min(1e-12)
			).mean(),
		]
		for _ in range(self.config.num_extra_loop_passes):
			previous_slot_states = slot_states
			proposed_slot_states = previous_slot_states
			for offset, layer_index in enumerate(
				range(self.config.loop_start_layer, self.config.loop_end_layer),
			):
				proposed_slot_states = self._run_dynamic_layer(
					layer=language_model.layers[layer_index],
					dynamic_hidden_states=proposed_slot_states,
					prefix_key_value=prefix_caches[offset],
					position_embeddings=(slot_cos, slot_sin),
					attention_mask=slot_attention_mask,
				)
			slot_states = damped_recurrent_update(
				previous_slot_states,
				proposed_slot_states,
				total_passes=self.config.num_total_loop_passes,
			)
			loop_slot_hidden_states.append(slot_states)
			pass_cosines.append(
				F.cosine_similarity(
					slot_states.float().flatten(1),
					previous_slot_states.float().flatten(1),
					dim=-1,
				).mean(),
			)
			pass_relative_updates.append(
				(
					(slot_states.float() - previous_slot_states.float())
					.flatten(1)
					.norm(dim=-1)
					/ previous_slot_states.float().flatten(1).norm(dim=-1).clamp_min(1e-12)
				).mean(),
			)

		finalized_outputs = (
			tuple(
				self._run_suffix_and_fusion(
					pass_one_full_hidden_states=pass_one_full_hidden_states,
					slot_positions=slot_positions,
					slot_states=pass_slots,
					augmented=augmented,
					causal_mask=causal_mask,
					text_position_ids=text_position_ids,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
				)
				for pass_slots in loop_slot_hidden_states
			)
			if return_all_loop_embeddings
			else (
				self._run_suffix_and_fusion(
					pass_one_full_hidden_states=pass_one_full_hidden_states,
					slot_positions=slot_positions,
					slot_states=slot_states,
					augmented=augmented,
					causal_mask=causal_mask,
					text_position_ids=text_position_ids,
					cache_position=cache_position,
					position_embeddings=position_embeddings,
				),
			)
		)
		final_output = finalized_outputs[-1]
		return RecurrentEmbeddingOutput(
			embeddings=final_output["embeddings"],
			loop_embeddings=(
				tuple(output["embeddings"] for output in finalized_outputs)
				if return_all_loop_embeddings
				else None
			),
			loop_slot_hidden_states=tuple(loop_slot_hidden_states),
			slot_hidden_states=final_output["slot_hidden_states"],
			eos_hidden_state=final_output["eos_hidden_state"],
			attention_weights=final_output["attention_weights"],
			diagnostics={
				"variant": self._variant_name(),
				"deepstack_layer_indices": tuple(deepstack_layers_executed),
				"extra_pass_dynamic_token_counts": tuple(
					slot_positions.shape[1]
					for _ in range(self.config.num_extra_loop_passes)
				),
				"prefix_cache_requires_grad": tuple(
					cache.key.requires_grad or cache.value.requires_grad
					for cache in prefix_caches
				),
				"recurrent_pass_cosine": tuple(pass_cosines),
				"recurrent_pass_relative_update": tuple(pass_relative_updates),
				"recurrent_step_size": self.config.recurrent_step_size,
				"fusion_gate": final_output["fusion_gate"],
				"late_fusion_attention_entropy": final_output["attention_entropy"].mean(),
				"slot_pairwise_cosine": _pairwise_slot_cosine(
					slot_states,
				),
			},
		)

	def _run_suffix_and_fusion(
		self,
		*,
		pass_one_full_hidden_states: torch.Tensor,
		slot_positions: torch.Tensor,
		slot_states: torch.Tensor,
		augmented: AugmentedSequence,
		causal_mask: torch.Tensor,
		text_position_ids: torch.Tensor,
		cache_position: torch.Tensor,
		position_embeddings: tuple[torch.Tensor, torch.Tensor],
	) -> dict[str, torch.Tensor | None]:
		"""Finalize one recurrent pass through Layers 21–28, norm, and late fusion."""
		language_model = self.language_model
		hidden_states = _scatter_sequence_positions(
			pass_one_full_hidden_states,
			slot_positions,
			slot_states,
		)
		for layer_index in range(self.config.loop_end_layer, len(language_model.layers)):
			hidden_states = _run_full_sequence_decoder_layer(
				layer=language_model.layers[layer_index],
				hidden_states=hidden_states,
				attention_mask=causal_mask,
				position_ids=text_position_ids,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				activation_checkpointing=(
					self.activation_checkpointing_enabled and self.training
				),
			)
		hidden_states = language_model.norm(hidden_states)
		eos_hidden_state = _gather_sequence_positions(hidden_states, augmented.eos_positions)
		slot_hidden_states = _gather_sequence_positions(
			hidden_states,
			augmented.slot_positions,
		)
		attention_weights = None
		attention_entropy = eos_hidden_state.new_zeros(eos_hidden_state.shape[0])
		fusion_gate = eos_hidden_state.new_zeros(())
		if self.config.num_latent_slots:
			fusion_output = self.late_fusion(eos_hidden_state, slot_hidden_states)
			pre_normalized_embedding = fusion_output.fused_embedding
			attention_weights = fusion_output.attention_weights
			attention_entropy = fusion_output.attention_entropy
			fusion_gate = fusion_output.gate
		else:
			pre_normalized_embedding = eos_hidden_state
		return {
			"embeddings": F.normalize(pre_normalized_embedding, p=2, dim=-1),
			"slot_hidden_states": slot_hidden_states,
			"eos_hidden_state": eos_hidden_state,
			"attention_weights": attention_weights,
			"attention_entropy": attention_entropy,
			"fusion_gate": fusion_gate,
		}

	@staticmethod
	def _run_full_layer_and_capture_prefix(
		layer: nn.Module,
		hidden_states: torch.Tensor,
		position_embeddings: tuple[torch.Tensor, torch.Tensor],
		max_prefix_length: int,
		attention_mask: torch.Tensor | None,
		position_ids: torch.Tensor,
		cache_position: torch.Tensor,
		activation_checkpointing: bool = False,
	) -> tuple[torch.Tensor, PrefixKeyValue]:
		"""Reuse Pass-1 projections when constructing the detached prefix cache."""
		attention = layer.self_attn
		captured: dict[str, torch.Tensor] = {}

		def capture_key(
			_module: nn.Module,
			_inputs: tuple[torch.Tensor, ...],
			output: torch.Tensor,
		) -> None:
			captured["key"] = output

		def capture_value(
			_module: nn.Module,
			_inputs: tuple[torch.Tensor, ...],
			output: torch.Tensor,
		) -> None:
			captured["value"] = output

		key_hook = attention.k_norm.register_forward_hook(capture_key)
		value_hook = attention.v_proj.register_forward_hook(capture_value)
		try:
			output_hidden_states = _run_full_sequence_decoder_layer(
				layer=layer,
				hidden_states=hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				activation_checkpointing=activation_checkpointing,
			)
		finally:
			key_hook.remove()
			value_hook.remove()
		if set(captured) != {"key", "value"}:
			raise RuntimeError("Pass-1 attention did not expose both prefix projections")
		batch_size, sequence_length, _ = hidden_states.shape
		key = captured["key"].transpose(1, 2)
		value = captured["value"].view(
			batch_size,
			sequence_length,
			-1,
			attention.head_dim,
		).transpose(1, 2)
		cos, sin = position_embeddings
		dummy_query = key
		_, rotated_key = apply_rotary_pos_emb(dummy_query, key, cos, sin)
		prefix_key, prefix_value = detach_prefix_key_values(
			rotated_key[:, :, :max_prefix_length],
			value[:, :, :max_prefix_length],
		)
		return output_hidden_states, PrefixKeyValue(key=prefix_key, value=prefix_value)

	@staticmethod
	def _gather_dynamic_position_embeddings(
		position_embeddings: tuple[torch.Tensor, torch.Tensor],
		dynamic_positions: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		cos, sin = position_embeddings
		return (
			_gather_sequence_positions(cos, dynamic_positions),
			_gather_sequence_positions(sin, dynamic_positions),
		)

	@staticmethod
	def _run_dynamic_layer(
		layer: nn.Module,
		dynamic_hidden_states: torch.Tensor,
		prefix_key_value: PrefixKeyValue,
		position_embeddings: tuple[torch.Tensor, torch.Tensor],
		attention_mask: torch.Tensor,
	) -> torch.Tensor:
		"""Run one loop layer on dynamic tokens only, with detached prefix evidence."""
		residual = dynamic_hidden_states
		normalized = layer.input_layernorm(dynamic_hidden_states)
		attention = layer.self_attn
		batch_size, token_count, _ = normalized.shape
		hidden_shape = (batch_size, token_count, -1, attention.head_dim)
		query = attention.q_norm(attention.q_proj(normalized).view(hidden_shape)).transpose(1, 2)
		key = attention.k_norm(attention.k_proj(normalized).view(hidden_shape)).transpose(1, 2)
		value = attention.v_proj(normalized).view(hidden_shape).transpose(1, 2)
		query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
		key = torch.cat((prefix_key_value.key, key), dim=2)
		value = torch.cat((prefix_key_value.value, value), dim=2)
		repeated_key = repeat_kv(key, attention.num_key_value_groups)
		repeated_value = repeat_kv(value, attention.num_key_value_groups)
		attention_output = _dynamic_scaled_dot_product_attention(
			query=query,
			key=repeated_key,
			value=repeated_value,
			attention_mask=attention_mask,
			scale=attention.scaling,
		)
		attention_output = attention_output.transpose(1, 2).reshape(
			batch_size,
			token_count,
			-1,
		).contiguous()
		hidden_states = residual + attention.o_proj(attention_output)
		residual = hidden_states
		hidden_states = layer.post_attention_layernorm(hidden_states)
		return residual + layer.mlp(hidden_states)

	def _variant_name(self) -> str:
		if self.config.num_latent_slots == 0:
			return "base"
		if self.config.num_total_loop_passes == 1:
			return "slots_without_recurrence"
		return "full_proposed_model"
