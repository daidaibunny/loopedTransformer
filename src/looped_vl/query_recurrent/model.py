"""Small shared recurrent block over frozen multi-layer Qwen query histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from looped_vl.query_recurrent.config import (
	MAX_QUERY_RECURRENT_PARAMETERS,
	QueryRecurrentConfig,
)


def parameter_free_rms_norm(values: torch.Tensor) -> torch.Tensor:
	"""Normalize the final dimension without adding trainable affine parameters."""
	return values * torch.rsqrt(values.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(
		values.dtype,
	)


class RecurrentHistoryLayer(nn.Module):
	"""Update slots through self-attention, frozen-history attention, and one feed-forward path."""

	def __init__(self, config: QueryRecurrentConfig) -> None:
		super().__init__()
		dimension = config.state_size
		feed_forward_size = dimension * config.feed_forward_multiplier
		self.self_norm = nn.LayerNorm(dimension)
		self.self_attention = nn.MultiheadAttention(
			dimension,
			config.num_attention_heads,
			batch_first=True,
		)
		self.history_norm = nn.LayerNorm(dimension)
		self.history_attention = nn.MultiheadAttention(
			dimension,
			config.num_attention_heads,
			batch_first=True,
		)
		self.feed_forward_norm = nn.LayerNorm(dimension)
		self.feed_forward = nn.Sequential(
			nn.Linear(dimension, feed_forward_size),
			nn.GELU(approximate="tanh"),
			nn.Linear(feed_forward_size, dimension),
		)

	def forward(
		self,
		slots: torch.Tensor,
		memory: torch.Tensor,
		memory_padding_mask: torch.Tensor,
	) -> torch.Tensor:
		"""Apply one shared state update while the Qwen memory remains fixed."""
		normalized_slots = self.self_norm(slots)
		self_update = self.self_attention(
			normalized_slots,
			normalized_slots,
			normalized_slots,
			need_weights=False,
		)[0]
		slots = slots + self_update
		history_query = self.history_norm(slots)
		history_update = self.history_attention(
			history_query,
			memory,
			memory,
			key_padding_mask=memory_padding_mask,
			need_weights=False,
		)[0]
		slots = slots + history_update
		return slots + self.feed_forward(self.feed_forward_norm(slots))


@dataclass(frozen=True)
class QueryRecurrentOutput:
	"""Every recurrent result needed for training, dynamic exit, and pass-wise evaluation."""

	embeddings: torch.Tensor
	soft_embeddings: torch.Tensor
	step_embeddings: tuple[torch.Tensor, ...]
	auxiliary_embeddings: tuple[torch.Tensor, ...]
	slot_states: tuple[torch.Tensor, ...]
	exit_probabilities: torch.Tensor
	halting_weights: torch.Tensor
	selected_steps: torch.Tensor
	expected_steps: torch.Tensor
	slot_attention_weights: tuple[torch.Tensor, ...]


def combine_query_recurrent_outputs(
	groups: tuple[tuple[tuple[int, ...], QueryRecurrentOutput], ...],
	*,
	total_rows: int,
) -> QueryRecurrentOutput:
	"""Restore grouped encoding order without padding histories across visual buckets."""
	if not groups or total_rows <= 0:
		raise ValueError("At least one non-empty recurrent group is required")
	flat_indices = tuple(index for indices, _output in groups for index in indices)
	if tuple(sorted(flat_indices)) != tuple(range(total_rows)):
		raise ValueError("Recurrent groups must cover every logical row exactly once")
	restore_order = torch.argsort(
		torch.tensor(flat_indices, device=groups[0][1].embeddings.device),
	)

	def combine_tensors(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
		return torch.cat(values, dim=0).index_select(0, restore_order)

	step_count = len(groups[0][1].step_embeddings)
	if any(len(output.step_embeddings) != step_count for _indices, output in groups):
		raise ValueError("Grouped recurrent outputs have different pass counts")
	return QueryRecurrentOutput(
		embeddings=combine_tensors(tuple(output.embeddings for _indices, output in groups)),
		soft_embeddings=combine_tensors(
			tuple(output.soft_embeddings for _indices, output in groups),
		),
		step_embeddings=tuple(
			combine_tensors(
				tuple(output.step_embeddings[step] for _indices, output in groups),
			)
			for step in range(step_count)
		),
		auxiliary_embeddings=tuple(
			combine_tensors(
				tuple(output.auxiliary_embeddings[step] for _indices, output in groups),
			)
			for step in range(step_count)
		),
		slot_states=tuple(
			combine_tensors(
				tuple(output.slot_states[step] for _indices, output in groups),
			)
			for step in range(step_count)
		),
		exit_probabilities=combine_tensors(
			tuple(output.exit_probabilities for _indices, output in groups),
		),
		halting_weights=combine_tensors(
			tuple(output.halting_weights for _indices, output in groups),
		),
		selected_steps=combine_tensors(
			tuple(output.selected_steps for _indices, output in groups),
		),
		expected_steps=combine_tensors(
			tuple(output.expected_steps for _indices, output in groups),
		),
		slot_attention_weights=tuple(
			combine_tensors(
				tuple(output.slot_attention_weights[step] for _indices, output in groups),
			)
			for step in range(step_count)
		),
	)


class QueryRecurrentHead(nn.Module):
	"""Convert frozen Qwen histories into a retrieval embedding with a shared recurrent Block."""

	def __init__(self, config: QueryRecurrentConfig) -> None:
		super().__init__()
		config.validate()
		self.config = config
		dimension = config.state_size
		feed_forward_size = dimension * config.feed_forward_multiplier
		self.memory_projection = nn.Linear(config.hidden_size, dimension, bias=False)
		self.layer_embeddings = nn.Parameter(torch.empty(28, dimension))
		self.slot_queries = nn.Parameter(torch.empty(config.num_slots, dimension))
		self.initializer_norm = nn.LayerNorm(dimension)
		self.initializer_attention = nn.MultiheadAttention(
			dimension,
			config.num_attention_heads,
			batch_first=True,
		)
		self.initializer_feed_forward_norm = nn.LayerNorm(dimension)
		self.initializer_feed_forward = nn.Sequential(
			nn.Linear(dimension, feed_forward_size),
			nn.GELU(approximate="tanh"),
			nn.Linear(feed_forward_size, dimension),
		)
		self.recurrent_layers = nn.ModuleList(
			RecurrentHistoryLayer(config) for _ in range(config.recurrent_block_layers)
		)
		self.output_norm = nn.LayerNorm(dimension)
		self.output_projection = nn.Linear(dimension, config.hidden_size, bias=False)
		self.residual_gate = nn.Parameter(torch.zeros(config.hidden_size))
		self.exit_controller = nn.Sequential(
			nn.LayerNorm(dimension),
			nn.Linear(dimension, dimension // 4),
			nn.GELU(approximate="tanh"),
			nn.Linear(dimension // 4, 1),
		)
		self._reset_parameters()
		trainable_count = sum(parameter.numel() for parameter in self.parameters())
		if trainable_count > MAX_QUERY_RECURRENT_PARAMETERS:
			raise ValueError(
				f"Query-only recurrent head has {trainable_count:,} parameters; "
				f"limit is {MAX_QUERY_RECURRENT_PARAMETERS:,}",
			)

	def _reset_parameters(self) -> None:
		generator = torch.Generator(device="cpu")
		generator.manual_seed(self.config.seed)
		with torch.no_grad():
			self.layer_embeddings.normal_(mean=0.0, std=0.02, generator=generator)
			self.slot_queries.normal_(mean=0.0, std=0.02, generator=generator)
			final_exit_layer = self.exit_controller[-1]
			if not isinstance(final_exit_layer, nn.Linear):
				raise TypeError("Exit controller must end with a linear layer")
			final_exit_layer.weight.zero_()
			final_exit_layer.bias.fill_(-2.0)

	@property
	def trainable_parameter_count(self) -> int:
		"""Return the exact parameter count used for comparison with last-four-layer LoRA."""
		return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

	def _project_history(
		self,
		history_hidden_states: torch.Tensor,
		attention_mask: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if history_hidden_states.ndim != 4:
			raise ValueError("History states must have shape [batch, layers, tokens, hidden]")
		batch_size, history_count, token_count, hidden_size = history_hidden_states.shape
		if hidden_size != self.config.hidden_size:
			raise ValueError("Frozen history hidden size does not match the configuration")
		if history_count != len(self.config.history_layers):
			raise ValueError("Frozen history layer count does not match the configuration")
		if attention_mask.shape != (batch_size, token_count):
			raise ValueError("History attention mask shape does not match frozen tokens")
		normalized = parameter_free_rms_norm(history_hidden_states)
		memory = self.memory_projection(normalized)
		layer_indices = torch.tensor(
			[layer - 1 for layer in self.config.history_layers],
			device=memory.device,
		)
		memory = memory + self.layer_embeddings[layer_indices][None, :, None, :]
		memory = memory.reshape(batch_size, history_count * token_count, -1)
		valid = attention_mask.to(torch.bool)
		memory_padding_mask = (~valid[:, None, :]).expand(
			batch_size,
			history_count,
			token_count,
		).reshape(batch_size, history_count * token_count)
		return memory, memory_padding_mask

	def _initialize_slots(
		self,
		memory: torch.Tensor,
		memory_padding_mask: torch.Tensor,
	) -> torch.Tensor:
		queries = self.slot_queries[None].expand(memory.shape[0], -1, -1)
		normalized_queries = self.initializer_norm(queries)
		update = self.initializer_attention(
			normalized_queries,
			memory,
			memory,
			key_padding_mask=memory_padding_mask,
			need_weights=False,
		)[0]
		slots = queries + update
		return slots + self.initializer_feed_forward(
			self.initializer_feed_forward_norm(slots),
		)

	def _read_step(
		self,
		slots: torch.Tensor,
		base_embeddings: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		condition = parameter_free_rms_norm(
			self.memory_projection(base_embeddings.to(slots.dtype)),
		)
		normalized_slots = parameter_free_rms_norm(slots)
		scores = torch.einsum("bd,bkd->bk", condition, normalized_slots)
		scores = scores / (self.config.state_size**0.5)
		weights = torch.softmax(scores, dim=1)
		pooled = torch.einsum("bk,bkd->bd", weights.to(slots.dtype), slots)
		residual = F.normalize(
			self.output_projection(self.output_norm(pooled)).float(),
			p=2,
			dim=-1,
		)
		gate = torch.tanh(self.residual_gate.float())
		fused = F.normalize(base_embeddings.float() + gate * residual, p=2, dim=-1)
		return fused, residual, weights

	def _halting_weights(self, probabilities: torch.Tensor) -> torch.Tensor:
		remaining = torch.ones_like(probabilities[:, 0])
		weights = []
		for step in range(probabilities.shape[1]):
			if step == probabilities.shape[1] - 1:
				weight = remaining
			else:
				weight = remaining * probabilities[:, step]
				remaining = remaining * (1.0 - probabilities[:, step])
			weights.append(weight)
		return torch.stack(weights, dim=1)

	def forward(
		self,
		*,
		history_hidden_states: torch.Tensor,
		attention_mask: torch.Tensor,
		base_embeddings: torch.Tensor,
	) -> QueryRecurrentOutput:
		"""Run a fixed number of shared updates and optionally select a dynamic exit."""
		if base_embeddings.ndim != 2 or base_embeddings.shape[-1] != self.config.hidden_size:
			raise ValueError("Base embeddings must have shape [batch, 2048]")
		memory, memory_padding_mask = self._project_history(
			history_hidden_states,
			attention_mask,
		)
		slots = self._initialize_slots(memory, memory_padding_mask)
		step_embeddings = []
		auxiliary_embeddings = []
		slot_states = []
		exit_probabilities = []
		slot_attention_weights = []
		for _step in range(self.config.max_recurrent_steps):
			for layer in self.recurrent_layers:
				slots = layer(slots, memory, memory_padding_mask)
			fused, auxiliary, slot_weights = self._read_step(slots, base_embeddings)
			step_embeddings.append(fused)
			auxiliary_embeddings.append(auxiliary)
			slot_states.append(slots)
			exit_probabilities.append(torch.sigmoid(self.exit_controller(slots.mean(dim=1))))
			slot_attention_weights.append(slot_weights)
		probabilities = torch.cat(exit_probabilities, dim=1)
		halting_weights = self._halting_weights(probabilities)
		stacked_steps = torch.stack(step_embeddings, dim=1)
		soft_embeddings = F.normalize(
			(stacked_steps * halting_weights[:, :, None]).sum(dim=1),
			p=2,
			dim=-1,
		)
		if self.config.exit_mode == "dynamic":
			threshold_met = probabilities >= self.config.exit_threshold
			threshold_met[:, -1] = True
			selected_steps = threshold_met.to(torch.int64).argmax(dim=1) + 1
			embeddings = stacked_steps[
				torch.arange(stacked_steps.shape[0], device=stacked_steps.device),
				selected_steps - 1,
			]
		else:
			selected_steps = torch.full(
				(base_embeddings.shape[0],),
				self.config.max_recurrent_steps,
				device=base_embeddings.device,
				dtype=torch.long,
			)
			embeddings = step_embeddings[-1]
		step_numbers = torch.arange(
			1,
			self.config.max_recurrent_steps + 1,
			device=halting_weights.device,
			dtype=halting_weights.dtype,
		)
		expected_steps = (halting_weights * step_numbers[None]).sum(dim=1)
		return QueryRecurrentOutput(
			embeddings=embeddings,
			soft_embeddings=soft_embeddings,
			step_embeddings=tuple(step_embeddings),
			auxiliary_embeddings=tuple(auxiliary_embeddings),
			slot_states=tuple(slot_states),
			exit_probabilities=probabilities,
			halting_weights=halting_weights,
			selected_steps=selected_steps,
			expected_steps=expected_steps,
			slot_attention_weights=tuple(slot_attention_weights),
		)


class GroupedQueryRecurrentHead(nn.Module):
	"""Run one shared head per padding bucket while keeping one logical contrastive batch."""

	def __init__(self, head: QueryRecurrentHead) -> None:
		super().__init__()
		self.head = head

	def forward(
		self,
		*,
		feature_groups: tuple[
			tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor],
			...,
		],
		total_rows: int,
	) -> QueryRecurrentOutput:
		"""Apply the head independently to each bucket and restore logical row order."""
		outputs = tuple(
			(
				indices,
				self.head(
					history_hidden_states=history,
					attention_mask=attention_mask,
					base_embeddings=base_embeddings,
				),
			)
			for indices, history, attention_mask, base_embeddings in feature_groups
		)
		return combine_query_recurrent_outputs(outputs, total_rows=total_rows)
