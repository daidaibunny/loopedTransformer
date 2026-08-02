"""Small shared recurrent block over frozen multi-layer Qwen query histories."""

from __future__ import annotations

import math
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
		*,
		slot_identity: torch.Tensor,
	) -> torch.Tensor:
		"""Apply one shared state update while the Qwen memory remains fixed."""
		if slot_identity.shape[-2:] != slots.shape[-2:]:
			raise ValueError("Slot identity must match the slot count and state dimension")
		identity = parameter_free_rms_norm(slot_identity).to(slots.dtype)
		normalized_slots = self.self_norm(slots)
		self_update = self.self_attention(
			normalized_slots + identity,
			normalized_slots,
			normalized_slots,
			need_weights=False,
		)[0]
		slots = slots + self_update
		history_query = self.history_norm(slots) + identity
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
	"""Every fixed-pass recurrent result needed for training and evaluation."""

	embeddings: torch.Tensor
	step_embeddings: tuple[torch.Tensor, ...]
	slot_states: tuple[torch.Tensor, ...]
	slot_attention_weights: tuple[torch.Tensor, ...]


def _mean_l2_norm(values: torch.Tensor) -> torch.Tensor:
	return torch.linalg.vector_norm(values.float(), dim=-1).mean()


def _normalized_attention_entropy(weights: torch.Tensor) -> torch.Tensor:
	if weights.shape[-1] <= 1:
		return weights.new_zeros((), dtype=torch.float32)
	probabilities = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
	entropy = -(probabilities * probabilities.log()).sum(dim=-1)
	return (entropy / math.log(weights.shape[-1])).mean()


def _slot_pairwise_absolute_cosine(slot_states: torch.Tensor) -> torch.Tensor:
	if slot_states.shape[1] <= 1:
		return slot_states.new_zeros((), dtype=torch.float32)
	normalized = F.normalize(slot_states.float(), dim=-1)
	cosine = normalized @ normalized.transpose(1, 2)
	off_diagonal = ~torch.eye(
		slot_states.shape[1],
		device=slot_states.device,
		dtype=torch.bool,
	)
	return cosine[:, off_diagonal].abs().mean()


def query_recurrent_diagnostics(
	output: QueryRecurrentOutput,
	base_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
	"""Measure whether each pass moves, specializes slots, and changes the readout."""
	if not output.step_embeddings:
		raise ValueError("At least one recurrent pass is required for diagnostics")
	if not (
		len(output.step_embeddings)
		== len(output.slot_states)
		== len(output.slot_attention_weights)
	):
		raise ValueError("Recurrent diagnostic tensors must have the same pass count")
	diagnostics: dict[str, torch.Tensor] = {}
	previous = base_embeddings.float()
	for step, (embedding, slots, weights) in enumerate(
		zip(
			output.step_embeddings,
			output.slot_states,
			output.slot_attention_weights,
			strict=True,
		),
		start=1,
	):
		embedding_float = embedding.float()
		prefix = f"step_{step}"
		diagnostics[f"{prefix}_embedding_delta_from_base_l2"] = _mean_l2_norm(
			embedding_float - base_embeddings.float(),
		)
		diagnostics[f"{prefix}_embedding_delta_from_previous_l2"] = _mean_l2_norm(
			embedding_float - previous,
		)
		diagnostics[f"{prefix}_embedding_cosine_to_base"] = F.cosine_similarity(
			embedding_float,
			base_embeddings.float(),
			dim=-1,
		).mean()
		diagnostics[f"{prefix}_slot_pairwise_absolute_cosine"] = (
			_slot_pairwise_absolute_cosine(slots)
		)
		diagnostics[f"{prefix}_slot_attention_normalized_entropy"] = (
			_normalized_attention_entropy(weights)
		)
		diagnostics[f"{prefix}_slot_attention_max_weight"] = (
			weights.float().amax(dim=-1).mean()
		)
		previous = embedding_float
	return diagnostics


def _gradient_l2_norm(parameters: tuple[nn.Parameter, ...]) -> torch.Tensor:
	if not parameters:
		raise ValueError("A gradient group must contain at least one parameter")
	total = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
	for parameter in parameters:
		if parameter.grad is not None:
			total = total + parameter.grad.detach().float().pow(2).sum()
	return total.sqrt()


def recurrent_gradient_group_norms(head: QueryRecurrentHead) -> dict[str, torch.Tensor]:
	"""Expose gradient reachability for every distinct recurrent-head component."""
	initializer_parameters = (
		*tuple(head.initializer_norm.parameters()),
		*tuple(head.initializer_attention.parameters()),
		*tuple(head.initializer_feed_forward_norm.parameters()),
		*tuple(head.initializer_feed_forward.parameters()),
	)
	groups = {
		"residual_gate": (head.residual_gate,),
		"output_projection": tuple(head.output_projection.parameters()),
		"output_norm": tuple(head.output_norm.parameters()),
		"recurrent_layers": tuple(head.recurrent_layers.parameters()),
		"initializer": initializer_parameters,
		"memory_projection": tuple(head.memory_projection.parameters()),
		"slot_queries": (head.slot_queries,),
		"layer_embeddings": (head.layer_embeddings,),
	}
	return {
		f"gradient_norm_{name}": _gradient_l2_norm(parameters)
		for name, parameters in groups.items()
	}


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
		step_embeddings=tuple(
			combine_tensors(
				tuple(output.step_embeddings[step] for _indices, output in groups),
			)
			for step in range(step_count)
		),
		slot_states=tuple(
			combine_tensors(
				tuple(output.slot_states[step] for _indices, output in groups),
			)
			for step in range(step_count)
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
		self.residual_gate = nn.Parameter(torch.zeros(()))
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
			nn.init.orthogonal_(self.slot_queries)
			self.slot_queries.mul_(self.config.state_size**0.5 * 0.02)
			nn.init.xavier_uniform_(self.output_projection.weight, generator=generator)
			self.residual_gate.zero_()

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
	) -> tuple[torch.Tensor, torch.Tensor]:
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
		fused = F.normalize(
			base_embeddings.float() + torch.tanh(self.residual_gate.float()) * residual,
			p=2,
			dim=-1,
		)
		return fused, weights

	def forward(
		self,
		*,
		history_hidden_states: torch.Tensor,
		attention_mask: torch.Tensor,
		base_embeddings: torch.Tensor,
	) -> QueryRecurrentOutput:
		"""Run exactly the configured number of shared recurrent updates."""
		if base_embeddings.ndim != 2 or base_embeddings.shape[-1] != self.config.hidden_size:
			raise ValueError("Base embeddings must have shape [batch, 2048]")
		memory, memory_padding_mask = self._project_history(
			history_hidden_states,
			attention_mask,
		)
		slots = self._initialize_slots(memory, memory_padding_mask)
		step_embeddings = []
		slot_states = []
		slot_attention_weights = []
		for _step in range(self.config.max_recurrent_steps):
			for layer in self.recurrent_layers:
				slots = layer(
					slots,
					memory,
					memory_padding_mask,
					slot_identity=self.slot_queries,
				)
			fused, slot_weights = self._read_step(slots, base_embeddings)
			step_embeddings.append(fused)
			slot_states.append(slots)
			slot_attention_weights.append(slot_weights)
		return QueryRecurrentOutput(
			embeddings=step_embeddings[-1],
			step_embeddings=tuple(step_embeddings),
			slot_states=tuple(slot_states),
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
