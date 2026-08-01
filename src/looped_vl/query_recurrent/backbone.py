"""One frozen Qwen query pass that exposes selected decoder histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from looped_vl.baseline.model import pool_last_token


@dataclass(frozen=True)
class FrozenQueryFeatures:
	"""Detached selected histories and the exact official base retrieval embedding."""

	history_hidden_states: torch.Tensor
	attention_mask: torch.Tensor
	base_embeddings: torch.Tensor


class FrozenQueryBackbone(nn.Module):
	"""Run the immutable Qwen backbone once and retain only requested layer outputs."""

	def __init__(self, base_embedding_model: nn.Module, history_layers: tuple[int, ...]) -> None:
		super().__init__()
		if not history_layers or history_layers[-1] > 28:
			raise ValueError("Frozen query history layers must be within 1 through 28")
		self.base_embedding_model = base_embedding_model
		self.history_layers = history_layers
		self.base_embedding_model.eval()
		self.base_embedding_model.requires_grad_(False)

	def _decoder_layers(self) -> nn.ModuleList | None:
		inner_model = getattr(self.base_embedding_model, "model", None)
		language_model = getattr(inner_model, "language_model", None)
		layers = getattr(language_model, "layers", None)
		if isinstance(layers, nn.ModuleList) and len(layers) >= 28:
			return layers
		return None

	@staticmethod
	def _layer_hidden_state(output: object) -> torch.Tensor:
		if isinstance(output, torch.Tensor):
			return output
		if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
			return output[0]
		raise TypeError("Qwen decoder layer hook did not receive hidden states")

	def forward(self, processed_inputs: dict[str, torch.Tensor]) -> FrozenQueryFeatures:
		"""Return detached histories without building a Qwen backward graph."""
		attention_mask = processed_inputs["attention_mask"]
		decoder_layers = self._decoder_layers()
		# These frozen tensors feed trainable projections, so use no_grad instead of
		# inference_mode; inference tensors cannot be saved for the recurrent backward.
		if decoder_layers is None:
			with torch.no_grad():
				outputs = self.base_embedding_model.model(
					**processed_inputs,
					output_hidden_states=True,
					return_dict=True,
					use_cache=False,
				)
				hidden_states = outputs.hidden_states
				if hidden_states is None or len(hidden_states) < 29:
					raise RuntimeError("Frozen Qwen did not return all 28 decoder histories")
				history = torch.stack(
					[hidden_states[layer] for layer in self.history_layers],
					dim=1,
				)
		else:
			captured: dict[int, torch.Tensor] = {}
			handles = []
			for layer_number in self.history_layers:
				if layer_number == 28:
					continue

				def capture_layer(
					_module: nn.Module,
					_inputs: tuple[object, ...],
					output: object,
					*,
					captured_layer: int = layer_number,
				) -> None:
					captured[captured_layer] = self._layer_hidden_state(output)

				handles.append(
					decoder_layers[layer_number - 1].register_forward_hook(capture_layer),
				)
			try:
				with torch.no_grad():
					outputs = self.base_embedding_model.model(
						**processed_inputs,
						output_hidden_states=False,
						return_dict=True,
						use_cache=False,
					)
			finally:
				for handle in handles:
					handle.remove()
			captured[28] = outputs.last_hidden_state
			missing = [layer for layer in self.history_layers if layer not in captured]
			if missing:
				raise RuntimeError(f"Frozen Qwen hooks missed decoder layers: {missing}")
			history = torch.stack(
				[captured[layer] for layer in self.history_layers],
				dim=1,
			)
		base_embeddings = pool_last_token(outputs.last_hidden_state, attention_mask)
		return FrozenQueryFeatures(
			history_hidden_states=history.detach(),
			attention_mask=attention_mask.detach(),
			base_embeddings=base_embeddings.detach(),
		)


def combine_frozen_query_groups(
	groups: list[tuple[tuple[int, ...], FrozenQueryFeatures]],
	*,
	total_rows: int,
) -> FrozenQueryFeatures:
	"""Pad only the small recurrent head input and restore the logical contrastive order."""
	if not groups or total_rows <= 0:
		raise ValueError("At least one non-empty frozen query group is required")
	flat_indices = tuple(index for indices, _features in groups for index in indices)
	if tuple(sorted(flat_indices)) != tuple(range(total_rows)):
		raise ValueError("Frozen query groups must cover every logical row exactly once")
	first = groups[0][1]
	history_count = first.history_hidden_states.shape[1]
	hidden_size = first.history_hidden_states.shape[-1]
	maximum_tokens = max(features.attention_mask.shape[1] for _indices, features in groups)
	history = first.history_hidden_states.new_zeros(
		(total_rows, history_count, maximum_tokens, hidden_size),
	)
	attention_mask = first.attention_mask.new_zeros((total_rows, maximum_tokens))
	base_embeddings = first.base_embeddings.new_empty((total_rows, hidden_size))
	for indices, features in groups:
		if len(indices) != features.base_embeddings.shape[0]:
			raise ValueError("Frozen query group indexes and feature rows must match")
		positions = torch.tensor(indices, device=history.device)
		token_count = features.attention_mask.shape[1]
		history[positions, :, :token_count] = features.history_hidden_states
		attention_mask[positions, :token_count] = features.attention_mask
		base_embeddings[positions] = features.base_embeddings
	return FrozenQueryFeatures(history, attention_mask, base_embeddings)
