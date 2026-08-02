"""Shared parallel-world recurrent Block over frozen Qwen query embeddings."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from looped_vl.query_recurrent.config import (
	MAX_QUERY_RECURRENT_PARAMETERS,
	QueryRecurrentConfig,
)


def recurrent_fp32_context(device_type: str) -> AbstractContextManager[None]:
	"""Disable an outer autocast region for the small trainable recurrent Block."""
	return torch.autocast(device_type=device_type, enabled=False)


def parameter_free_rms_norm(values: torch.Tensor) -> torch.Tensor:
	"""Normalize the final dimension without adding trainable affine parameters."""
	return values * torch.rsqrt(values.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(
		values.dtype,
	)


@dataclass(frozen=True)
class QueryRecurrentOutput:
	"""Final mean embedding and every fixed recurrent-pass diagnostic state."""

	embeddings: torch.Tensor
	step_embeddings: tuple[torch.Tensor, ...]
	world_states: tuple[torch.Tensor, ...]
	interaction_weights: tuple[torch.Tensor, ...]
	initial_world_states: torch.Tensor


class ParallelWorldRecurrentCell(nn.Module):
	"""Compare centered worlds and update all of them with one shared nonlinear cell."""

	def __init__(self, config: QueryRecurrentConfig) -> None:
		super().__init__()
		self.config = config
		self.query_projection = nn.Linear(
			config.hidden_size,
			config.attention_size,
			bias=False,
		)
		self.key_projection = nn.Linear(
			config.hidden_size,
			config.attention_size,
			bias=False,
		)
		self.value_projection = nn.Linear(
			config.hidden_size,
			config.attention_size,
			bias=False,
		)
		self.attention_output = nn.Linear(
			config.attention_size,
			config.hidden_size,
			bias=False,
		)
		self.feed_forward_gate = nn.Linear(
			config.hidden_size,
			config.feed_forward_size,
			bias=False,
		)
		self.feed_forward_up = nn.Linear(
			config.hidden_size,
			config.feed_forward_size,
			bias=False,
		)
		self.feed_forward_down = nn.Linear(
			config.feed_forward_size,
			config.hidden_size,
			bias=False,
		)
		initial_logit = math.atanh(
			config.initial_residual_scale / config.maximum_residual_scale,
		)
		self.attention_residual_logit = nn.Parameter(torch.tensor(initial_logit))
		self.feed_forward_residual_logit = nn.Parameter(torch.tensor(initial_logit))

	@property
	def attention_residual_scale(self) -> torch.Tensor:
		"""Return one bounded attention scale shared by every recurrent pass."""
		return self.config.maximum_residual_scale * torch.tanh(
			self.attention_residual_logit,
		)

	@property
	def feed_forward_residual_scale(self) -> torch.Tensor:
		"""Return one bounded feed-forward scale shared by every recurrent pass."""
		return self.config.maximum_residual_scale * torch.tanh(
			self.feed_forward_residual_logit,
		)

	def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
		batch_size, world_count, _dimension = values.shape
		head_dimension = self.config.attention_size // self.config.num_attention_heads
		return values.reshape(
			batch_size,
			world_count,
			self.config.num_attention_heads,
			head_dimension,
		).transpose(1, 2)

	def forward(self, world_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Advance every world simultaneously using centered bidirectional interaction."""
		if world_states.ndim != 3:
			raise ValueError("World states must have shape [batch, worlds, hidden]")
		if world_states.shape[1:] != (
			self.config.num_worlds,
			self.config.hidden_size,
		):
			raise ValueError("World state shape does not match the configuration")
		world_mean = world_states.mean(dim=1, keepdim=True)
		deviations = world_states - world_mean
		state_rms = torch.sqrt(
			world_states.float().pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-6,
		)
		normalized_states = parameter_free_rms_norm(world_states)
		normalized_deviations = (deviations.float() / state_rms).to(world_states.dtype)
		queries = self._split_heads(self.query_projection(normalized_states))
		keys = self._split_heads(self.key_projection(normalized_deviations))
		values = self._split_heads(self.value_projection(normalized_deviations))
		head_dimension = queries.shape[-1]
		scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2))
		weights = torch.softmax(scores / math.sqrt(head_dimension), dim=-1)
		attended = torch.matmul(weights.to(values.dtype), values)
		attended = attended.transpose(1, 2).reshape(
			world_states.shape[0],
			self.config.num_worlds,
			self.config.attention_size,
		)
		interaction = self.attention_output(attended)
		interaction = interaction - interaction.mean(dim=1, keepdim=True)
		interacted = world_states + self.attention_residual_scale * interaction
		normalized_interacted = parameter_free_rms_norm(interacted)
		feed_forward = self.feed_forward_down(
			F.silu(self.feed_forward_gate(normalized_interacted))
			* self.feed_forward_up(normalized_interacted),
		)
		updated = interacted + self.feed_forward_residual_scale * feed_forward
		return updated, weights.mean(dim=1)


def _mean_l2_norm(values: torch.Tensor) -> torch.Tensor:
	return torch.linalg.vector_norm(values.float(), dim=-1).mean()


def _normalized_attention_entropy(weights: torch.Tensor) -> torch.Tensor:
	if weights.shape[-1] <= 1:
		return weights.new_zeros((), dtype=torch.float32)
	probabilities = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
	entropy = -(probabilities * probabilities.log()).sum(dim=-1)
	return (entropy / math.log(weights.shape[-1])).mean()


def _interaction_off_diagonal_mass(weights: torch.Tensor) -> torch.Tensor:
	if weights.shape[-1] <= 1:
		return weights.new_zeros((), dtype=torch.float32)
	diagonal = weights.float().diagonal(dim1=-2, dim2=-1).sum(dim=-1)
	return (1 - diagonal / weights.shape[-1]).mean()


def _population_spread(world_states: torch.Tensor) -> torch.Tensor:
	deviations = world_states.float() - world_states.float().mean(dim=1, keepdim=True)
	return torch.linalg.vector_norm(deviations, dim=-1).mean()


def query_recurrent_diagnostics(
	output: QueryRecurrentOutput,
	base_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
	"""Measure population spread, interaction, and mean movement at every pass."""
	if not output.step_embeddings:
		raise ValueError("At least one recurrent pass is required for diagnostics")
	if not (
		len(output.step_embeddings)
		== len(output.world_states)
		== len(output.interaction_weights)
	):
		raise ValueError("Recurrent diagnostic tensors must have the same pass count")
	initial_mean = output.initial_world_states.float().mean(dim=1)
	diagnostics = {
		"initial_population_mean_error_l2": _mean_l2_norm(
			initial_mean - base_embeddings.float(),
		),
		"initial_population_spread_l2": _population_spread(
			output.initial_world_states,
		),
	}
	previous = base_embeddings.float()
	for step, (embedding, worlds, weights) in enumerate(
		zip(
			output.step_embeddings,
			output.world_states,
			output.interaction_weights,
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
		diagnostics[f"{prefix}_population_spread_l2"] = _population_spread(worlds)
		diagnostics[f"{prefix}_interaction_normalized_entropy"] = (
			_normalized_attention_entropy(weights)
		)
		diagnostics[f"{prefix}_interaction_off_diagonal_mass"] = (
			_interaction_off_diagonal_mass(weights)
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
	"""Expose first-step gradient reachability for each population component."""
	cell = head.recurrent_cell
	groups = {
		"perturbation_directions": (head.perturbation_direction_codes,),
		"world_attention": (
			*tuple(cell.query_projection.parameters()),
			*tuple(cell.key_projection.parameters()),
			*tuple(cell.value_projection.parameters()),
			*tuple(cell.attention_output.parameters()),
		),
		"feed_forward": (
			*tuple(cell.feed_forward_gate.parameters()),
			*tuple(cell.feed_forward_up.parameters()),
			*tuple(cell.feed_forward_down.parameters()),
		),
		"residual_scales": (
			cell.attention_residual_logit,
			cell.feed_forward_residual_logit,
		),
	}
	return {
		f"gradient_norm_{name}": _gradient_l2_norm(parameters)
		for name, parameters in groups.items()
	}


class QueryRecurrentHead(nn.Module):
	"""Evolve antithetic 2,048-dimensional worlds with one shared recurrent Block."""

	def __init__(self, config: QueryRecurrentConfig) -> None:
		super().__init__()
		config.validate()
		self.config = config
		self.recurrent_cell = ParallelWorldRecurrentCell(config)
		self.perturbation_direction_codes = nn.Parameter(
			torch.empty(2, config.attention_size),
		)
		self._reset_parameters()
		if self.trainable_parameter_count > MAX_QUERY_RECURRENT_PARAMETERS:
			raise ValueError(
				f"Query-only recurrent head has {self.trainable_parameter_count:,} parameters; "
				f"limit is {MAX_QUERY_RECURRENT_PARAMETERS:,}",
			)

	def _reset_parameters(self) -> None:
		with torch.random.fork_rng(devices=[]):
			torch.manual_seed(self.config.seed)
			for module in self.modules():
				if isinstance(module, nn.Linear):
					nn.init.xavier_uniform_(module.weight)
			nn.init.orthogonal_(self.perturbation_direction_codes)

	@property
	def trainable_parameter_count(self) -> int:
		"""Return the exact parameter count for the last-four-layer LoRA comparison."""
		return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

	def initialize_worlds(self, base_embeddings: torch.Tensor) -> torch.Tensor:
		"""Create deterministic antithetic worlds whose mean is exactly the Qwen embedding."""
		if base_embeddings.ndim != 2 or base_embeddings.shape[-1] != self.config.hidden_size:
			raise ValueError("Base embeddings must have shape [batch, 2048]")
		base = base_embeddings.float()
		if self.config.num_worlds == 1:
			return base[:, None, :]
		normalized = parameter_free_rms_norm(base_embeddings)
		features = torch.tanh(self.recurrent_cell.value_projection(normalized))
		raw_directions = tuple(
			self.recurrent_cell.attention_output(
				features * code[None].to(features.dtype),
			).float()
			for code in self.perturbation_direction_codes
		)
		first = F.normalize(raw_directions[0], dim=-1, eps=1e-6)
		base_norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
		first_delta = self.config.perturbation_scale * base_norm * first
		worlds = [base + first_delta, base - first_delta]
		if self.config.num_worlds == 4:
			second_raw = raw_directions[1] - (
				raw_directions[1] * first
			).sum(dim=-1, keepdim=True) * first
			second = F.normalize(second_raw, dim=-1, eps=1e-6)
			second_delta = self.config.perturbation_scale * base_norm * second
			worlds.extend((base + second_delta, base - second_delta))
		stacked = torch.stack(worlds, dim=1)
		return stacked - stacked.mean(dim=1, keepdim=True) + base[:, None, :]

	def forward(self, *, base_embeddings: torch.Tensor) -> QueryRecurrentOutput:
		"""Run a fixed number of simultaneous shared-parameter population updates."""
		worlds = self.initialize_worlds(base_embeddings)
		initial_worlds = worlds
		step_embeddings = []
		world_states = []
		interaction_weights = []
		for _step in range(self.config.max_recurrent_steps):
			worlds, weights = self.recurrent_cell(worlds)
			step_embeddings.append(F.normalize(worlds.mean(dim=1).float(), dim=-1))
			world_states.append(worlds)
			interaction_weights.append(weights)
		return QueryRecurrentOutput(
			embeddings=step_embeddings[-1],
			step_embeddings=tuple(step_embeddings),
			world_states=tuple(world_states),
			interaction_weights=tuple(interaction_weights),
			initial_world_states=initial_worlds,
		)


class GroupedQueryRecurrentHead(nn.Module):
	"""Restore grouped frozen embeddings before one logical population forward."""

	def __init__(self, head: QueryRecurrentHead) -> None:
		super().__init__()
		self.head = head

	def forward(
		self,
		*,
		feature_groups: tuple[tuple[tuple[int, ...], torch.Tensor], ...],
		total_rows: int,
	) -> QueryRecurrentOutput:
		"""Restore the contrastive batch order, then evolve every Query once."""
		if not feature_groups or total_rows <= 0:
			raise ValueError("At least one non-empty feature group is required")
		flat_indices = tuple(index for indices, _base in feature_groups for index in indices)
		if tuple(sorted(flat_indices)) != tuple(range(total_rows)):
			raise ValueError("Feature groups must cover every logical row exactly once")
		first = feature_groups[0][1]
		base_embeddings = first.new_empty((total_rows, first.shape[-1]))
		for indices, values in feature_groups:
			if len(indices) != values.shape[0]:
				raise ValueError("Feature group indexes and rows must match")
			positions = torch.tensor(indices, device=values.device)
			base_embeddings[positions] = values
		return self.head(base_embeddings=base_embeddings)
