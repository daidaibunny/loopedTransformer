"""Validated in-memory access to immutable candidate embedding banks."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch

from looped_vl.candidate_bank import (
	CandidateBankSpec,
	load_ready_candidate_bank,
	sha256_file,
	validate_embedding_shard,
)


@dataclass(frozen=True)
class CandidateReference:
	"""One training target resolved against a specific immutable bank."""

	spec: CandidateBankSpec
	item_id: str
	positive_id: str


class ImmutableCandidateStore:
	"""Load candidate vectors once and resolve stable item identifiers without Qwen calls."""

	def __init__(
		self,
		*,
		candidate_root: str | Path,
		spec: CandidateBankSpec,
		model_checkpoint_sha256: str,
		validate_checksums: bool = True,
	) -> None:
		self.root = Path(candidate_root) / spec.relative_path
		self.spec = spec
		if validate_checksums:
			manifest = load_ready_candidate_bank(
				self.root,
				expected_spec=spec,
				expected_model_sha256=model_checkpoint_sha256,
			)
		else:
			manifest_path = self.root / "bank_manifest.json"
			ready_path = self.root / "READY"
			if not manifest_path.is_file() or not ready_path.is_file():
				raise FileNotFoundError(f"Candidate bank is not ready: {self.root}")
			if ready_path.read_text(encoding="utf-8").strip() != sha256_file(manifest_path):
				raise ValueError(f"Candidate bank READY checksum mismatch under {self.root}")
			manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
			if manifest.get("spec") != {
				"dataset": spec.dataset,
				"split": spec.split,
				"gallery": spec.gallery,
			}:
				raise ValueError(f"Candidate-bank spec mismatch under {self.root}")
			if manifest.get("model", {}).get("checkpoint_sha256") != model_checkpoint_sha256:
				raise ValueError(f"Candidate-bank model checksum mismatch under {self.root}")
		self.manifest = manifest
		items_path = self.root / str(manifest["items"]["path"])
		items = pq.read_table(items_path, columns=["item_index", "item_id", "positive_id"])
		item_indices = [int(value) for value in items.column("item_index").to_pylist()]
		if item_indices != list(range(len(item_indices))):
			raise ValueError(f"Candidate item indexes are not contiguous under {self.root}")
		self.item_ids = tuple(str(value) for value in items.column("item_id").to_pylist())
		self.positive_ids = tuple(
			str(value) for value in items.column("positive_id").to_pylist()
		)
		if len(set(self.item_ids)) != len(self.item_ids):
			raise ValueError(f"Candidate item identifiers are not unique under {self.root}")
		self.index_by_item_id = {
			item_id: index for index, item_id in enumerate(self.item_ids)
		}
		self._shard_starts: list[int] = []
		self._shard_ends: list[int] = []
		self._shard_tensors: list[torch.Tensor] = []
		covered_until = 0
		for shard in manifest["embedding_shards"]:
			shard_path = self.root / str(shard["path"])
			try:
				payload = torch.load(
					shard_path,
					map_location="cpu",
					weights_only=True,
					mmap=True,
				)
			except TypeError:
				payload = torch.load(
					shard_path,
					map_location="cpu",
					weights_only=True,
				)
			start = int(payload["start"])
			end = int(payload["end"])
			if start != covered_until or end != int(shard["end"]):
				raise ValueError(f"Candidate shard range mismatch under {self.root}")
			embeddings = payload["embeddings"]
			if validate_checksums:
				validate_embedding_shard(embeddings, expected_rows=end - start)
			elif embeddings.shape != (end - start, 2048) or embeddings.dtype != torch.float16:
				raise ValueError(f"Candidate shard tensor mismatch under {self.root}")
			self._shard_starts.append(start)
			self._shard_ends.append(end)
			self._shard_tensors.append(embeddings)
			covered_until = end
		if covered_until != len(self.item_ids):
			raise ValueError(f"Candidate embeddings do not cover every item under {self.root}")

	@property
	def embeddings(self) -> torch.Tensor:
		"""Materialize a full gallery only for rank-zero exact evaluation."""
		return torch.cat(self._shard_tensors, dim=0)

	def indices_for_item_ids(self, item_ids: list[str]) -> torch.Tensor:
		"""Resolve item identifiers in logical batch order or fail on the first unknown item."""
		indices = []
		for item_id in item_ids:
			if item_id not in self.index_by_item_id:
				raise KeyError(f"Candidate item {item_id!r} is absent from {self.spec.key}")
			indices.append(self.index_by_item_id[item_id])
		return torch.tensor(indices, dtype=torch.long)

	def lookup(self, item_ids: list[str], *, device: torch.device) -> torch.Tensor:
		"""Copy only requested vectors from shared file-backed shards to one GPU."""
		indices = self.indices_for_item_ids(item_ids)
		result = torch.empty((len(item_ids), 2048), dtype=torch.float16)
		positions_by_shard: dict[int, list[int]] = {}
		for position, item_index in enumerate(indices.tolist()):
			shard_index = bisect_right(self._shard_starts, item_index) - 1
			if shard_index < 0 or item_index >= self._shard_ends[shard_index]:
				raise RuntimeError(f"Candidate item index {item_index} is outside all shards")
			positions_by_shard.setdefault(shard_index, []).append(position)
		for shard_index, positions in positions_by_shard.items():
			item_indexes = indices[positions] - self._shard_starts[shard_index]
			result[positions] = self._shard_tensors[shard_index][item_indexes]
		return result.to(device=device, non_blocking=True)


class CandidateStoreCollection:
	"""Lazily open the exact candidate banks referenced by one dataset batch."""

	def __init__(
		self,
		*,
		candidate_root: str | Path,
		model_checkpoint_sha256: str,
		validate_checksums: bool,
	) -> None:
		self.candidate_root = Path(candidate_root)
		self.model_checkpoint_sha256 = model_checkpoint_sha256
		self.validate_checksums = validate_checksums
		self._stores: dict[str, ImmutableCandidateStore] = {}

	def get(self, spec: CandidateBankSpec) -> ImmutableCandidateStore:
		"""Return one cached store after validating its immutable identity."""
		if spec.key not in self._stores:
			self._stores[spec.key] = ImmutableCandidateStore(
				candidate_root=self.candidate_root,
				spec=spec,
				model_checkpoint_sha256=self.model_checkpoint_sha256,
				validate_checksums=self.validate_checksums,
			)
		return self._stores[spec.key]

	def lookup(
		self,
		references: list[CandidateReference],
		*,
		device: torch.device,
	) -> torch.Tensor:
		"""Resolve a mixed-gallery logical batch while preserving its exact row order."""
		if not references:
			raise ValueError("Candidate references cannot be empty")
		result = torch.empty(
			(len(references), 2048),
			device=device,
			dtype=torch.float16,
		)
		positions_by_key: dict[str, list[int]] = {}
		spec_by_key: dict[str, CandidateBankSpec] = {}
		for position, reference in enumerate(references):
			positions_by_key.setdefault(reference.spec.key, []).append(position)
			spec_by_key[reference.spec.key] = reference.spec
		for key, positions in positions_by_key.items():
			store = self.get(spec_by_key[key])
			item_ids = [references[position].item_id for position in positions]
			values = store.lookup(item_ids, device=device)
			result[torch.tensor(positions, device=device)] = values
		return result
