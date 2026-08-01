"""Pre-encode every immutable single-dataset candidate gallery with frozen Qwen."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from looped_vl.baseline.bucketing import (
	DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	DEFAULT_VISUAL_LENGTH_BUCKETS,
	group_baseline_model_inputs,
)
from looped_vl.baseline.model import (
	BaselineInputProcessor,
	encode_grouped_baseline_batches,
	load_frozen_evaluation_model,
)
from looped_vl.candidate_bank import (
	CANDIDATE_BANK_SPECS,
	CANDIDATE_BANK_VERSION,
	DEFAULT_EMBEDDING_SHARD_ROWS,
	EMBEDDING_DIMENSION,
	CandidateBankSpec,
	embedding_shard_ranges,
	load_ready_candidate_bank,
	sha256_file,
	source_checksums_for_spec,
	validate_embedding_shard,
	write_candidate_item_manifest,
	write_json_atomic,
)
from looped_vl.data import ParquetShardIndex, _read_row_group
from looped_vl.smoke import checkpoint_sha256

LOGGER = logging.getLogger("encode_candidate_banks")


@dataclass(frozen=True)
class CandidateEncodingSample:
	"""One indexed candidate input with an optional decoded image."""

	item_index: int
	model_input: dict[str, Any]
	image: Image.Image | None = None


class CandidateItemDataset(Dataset[CandidateEncodingSample]):
	"""Read one contiguous candidate-manifest range without loading all metadata."""

	def __init__(self, items_root: str | Path, start: int, end: int) -> None:
		if start < 0 or end <= start:
			raise ValueError(f"Invalid candidate item range: [{start}, {end})")
		self.index = ParquetShardIndex(items_root)
		if end > len(self.index):
			raise ValueError(
				f"Candidate item range ends at {end}, but manifest has {len(self.index)}",
			)
		self.start = start
		self.end = end

	def __len__(self) -> int:
		return self.end - self.start

	def __getitem__(self, index: int) -> CandidateEncodingSample:
		if index < 0:
			index += len(self)
		if index < 0 or index >= len(self):
			raise IndexError(index)
		global_index = self.start + index
		location = self.index.locate(global_index)
		table = _read_row_group(str(location.path), location.row_group)
		record = table.slice(location.offset_in_row_group, 1).to_pylist()[0]
		if int(record["item_index"]) != global_index:
			raise RuntimeError(
				f"Candidate manifest index {record['item_index']} does not match {global_index}",
			)
		input_kind = str(record["input_kind"])
		if input_kind == "text":
			return CandidateEncodingSample(
				item_index=global_index,
				model_input={"text": str(record["text"])},
			)
		if input_kind != "image":
			raise ValueError(f"Unsupported candidate input kind: {input_kind}")
		image_path = Path(str(record["image_path"]))
		if not image_path.is_file():
			raise FileNotFoundError(f"Missing candidate image: {image_path}")
		with Image.open(image_path) as source_image:
			image = source_image.convert("RGB")
			image.load()
		return CandidateEncodingSample(
			item_index=global_index,
			model_input={"image": image},
			image=image,
		)


def _collate_candidate_samples(
	samples: list[CandidateEncodingSample],
) -> dict[str, Any]:
	if not samples:
		raise ValueError("Cannot collate an empty candidate batch")
	return {
		"item_indices": [sample.item_index for sample in samples],
		"model_inputs": [sample.model_input for sample in samples],
		"images": [sample.image for sample in samples if sample.image is not None],
	}


def _close_images(images: list[Image.Image]) -> None:
	for image in images:
		image.close()


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
	if not torch.cuda.is_available():
		raise RuntimeError("Candidate-bank encoding requires CUDA; CPU fallback is disabled")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="gloo")
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	if world_size != expected_world_size:
		raise RuntimeError(f"Expected {expected_world_size} ranks, found {world_size}")
	return rank, world_size, torch.device("cuda", local_rank)


def _git_commit(project_root: Path) -> str:
	result = subprocess.run(
		["git", "rev-parse", "HEAD"],
		cwd=project_root,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.strip()


def _load_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def _prepare_item_manifest(
	*,
	dataset_root: Path,
	bank_root: Path,
	spec: CandidateBankSpec,
) -> dict[str, Any]:
	"""Create or validate the immutable candidate order before GPU encoding."""
	source_checksums = source_checksums_for_spec(dataset_root, spec)
	items_path = bank_root / "items" / "part-00000.parquet"
	metadata_path = bank_root / "items_manifest.json"
	if metadata_path.exists():
		metadata = _load_json(metadata_path)
		if metadata.get("source_checksums") != source_checksums:
			raise ValueError(f"Candidate source files changed for {spec.key}")
		if not items_path.is_file() or sha256_file(items_path) != metadata.get("sha256"):
			raise ValueError(f"Candidate item manifest changed for {spec.key}")
		return metadata
	if items_path.exists():
		raise FileExistsError(
			f"Candidate item manifest exists without metadata and cannot be resumed: {items_path}",
		)
	item_metadata = write_candidate_item_manifest(
		dataset_root=dataset_root,
		spec=spec,
		output_path=items_path,
	)
	metadata = {
		**item_metadata,
		"path": "items/part-00000.parquet",
		"source_checksums": source_checksums,
	}
	write_json_atomic(metadata_path, metadata)
	return metadata


def _immutable_bank_config(
	*,
	args: argparse.Namespace,
	spec: CandidateBankSpec,
	item_metadata: dict[str, Any],
	model_sha256: str,
	code_commit: str,
) -> dict[str, Any]:
	return {
		"version": CANDIDATE_BANK_VERSION,
		"spec": asdict(spec),
		"items": item_metadata,
		"model": {
			"checkpoint_sha256": model_sha256,
			"embedding_readout": "official_final_valid_token",
			"trainable_parameters": 0,
		},
		"preprocessing": {
			"max_length": args.max_length,
			"min_pixels": args.min_pixels,
			"max_pixels": args.max_pixels,
			"candidate_instruction": None,
			"runtime_precision": "float16",
			"attention_implementation": args.attention_implementation,
			"visual_length_buckets": args.visual_length_buckets,
			"minimum_visual_bucket_size": args.min_visual_bucket_size,
			"image_batch_size": args.image_batch_size,
			"text_batch_size": args.text_batch_size,
		},
		"storage": {
			"embedding_dimension": EMBEDDING_DIMENSION,
			"embedding_dtype": "float16",
			"unit_normalized": True,
			"embedding_shard_rows": args.embedding_shard_rows,
		},
		"code_commit": code_commit,
	}


def _publish_or_validate_bank_config(bank_root: Path, config: dict[str, Any]) -> None:
	config_path = bank_root / "bank_config.json"
	if config_path.exists():
		if _load_json(config_path) != config:
			raise ValueError(f"Candidate bank resume configuration changed: {bank_root}")
		return
	write_json_atomic(config_path, config)


def _encode_item_range(
	*,
	items_root: Path,
	start: int,
	end: int,
	input_kind: str,
	model: torch.nn.Module,
	processor: BaselineInputProcessor,
	device: torch.device,
	args: argparse.Namespace,
) -> torch.Tensor:
	dataset = CandidateItemDataset(items_root, start, end)
	batch_size = args.image_batch_size if input_kind == "image" else args.text_batch_size
	loader_kwargs: dict[str, Any] = {
		"dataset": dataset,
		"batch_size": batch_size,
		"shuffle": False,
		"num_workers": args.num_workers,
		"collate_fn": _collate_candidate_samples,
		"pin_memory": True,
	}
	if args.num_workers:
		loader_kwargs.update(
			{
				"multiprocessing_context": "spawn",
				"persistent_workers": True,
				"prefetch_factor": args.prefetch_factor,
			},
		)
	loader = DataLoader(**loader_kwargs)
	chunks: list[torch.Tensor] = []
	seen_indices: list[int] = []
	for batch in loader:
		try:
			input_groups = group_baseline_model_inputs(
				batch["model_inputs"],
				min_pixels=args.min_pixels,
				max_pixels=args.max_pixels,
				max_visual_buckets=args.visual_length_buckets,
				min_visual_bucket_size=args.min_visual_bucket_size,
			)
			processed_batches = tuple(
				processor.prepare(list(group.model_inputs), device=device)
				for group in input_groups
			)
		finally:
			_close_images(batch["images"])
		with torch.inference_mode(), torch.autocast(
			device_type="cuda",
			dtype=torch.float16,
		):
			embeddings = encode_grouped_baseline_batches(
				model=model,
				processed_batches=processed_batches,
				original_indices=tuple(group.original_indices for group in input_groups),
				total_rows=len(batch["item_indices"]),
			)
		chunks.append(embeddings.detach().to(device="cpu", dtype=torch.float16))
		seen_indices.extend(int(index) for index in batch["item_indices"])
	expected_indices = list(range(start, end))
	if seen_indices != expected_indices:
		raise RuntimeError(f"Candidate encoder covered unexpected indexes in [{start}, {end})")
	combined = torch.cat(chunks, dim=0)
	validate_embedding_shard(combined, expected_rows=end - start)
	return combined


def _validate_existing_embedding_shard(path: Path, start: int, end: int) -> bool:
	if not path.is_file():
		return False
	payload = torch.load(path, map_location="cpu", weights_only=True)
	if int(payload.get("start", -1)) != start or int(payload.get("end", -1)) != end:
		raise ValueError(f"Existing candidate shard has the wrong range: {path}")
	validate_embedding_shard(payload["embeddings"], expected_rows=end - start)
	return True


def _save_embedding_shard(
	path: Path,
	*,
	start: int,
	end: int,
	embeddings: torch.Tensor,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = path.with_suffix(path.suffix + ".partial")
	torch.save(
		{"start": start, "end": end, "embeddings": embeddings.contiguous()},
		temporary_path,
	)
	os.replace(temporary_path, path)


def _finalize_bank(
	*,
	bank_root: Path,
	config: dict[str, Any],
	item_metadata: dict[str, Any],
	started_at: str,
	world_size: int,
	gpu_names: list[str],
) -> dict[str, Any]:
	ranges = embedding_shard_ranges(
		int(item_metadata["rows"]),
		int(config["storage"]["embedding_shard_rows"]),
	)
	shard_manifests = []
	covered_until = 0
	for shard_index, (start, end) in enumerate(ranges):
		path = bank_root / "embedding_shards" / f"part-{shard_index:05d}.pt"
		if start != covered_until:
			raise RuntimeError(f"Candidate ranges are not contiguous under {bank_root}")
		if not _validate_existing_embedding_shard(path, start, end):
			raise FileNotFoundError(f"Missing candidate embedding shard: {path}")
		shard_manifests.append(
			{
				"path": str(path.relative_to(bank_root)),
				"start": start,
				"end": end,
				"sha256": sha256_file(path),
			},
		)
		covered_until = end
	if covered_until != int(item_metadata["rows"]):
		raise RuntimeError(f"Candidate embedding coverage is incomplete under {bank_root}")
	manifest = {
		**config,
		"status": "passed",
		"embedding_dimension": EMBEDDING_DIMENSION,
		"embedding_shards": shard_manifests,
		"runtime": {
			"started_at": started_at,
			"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"hostname": socket.gethostname(),
			"world_size": world_size,
			"gpu_names": gpu_names,
		},
	}
	write_json_atomic(bank_root / "bank_manifest.json", manifest)
	ready_partial = bank_root / "READY.partial"
	ready_partial.write_text(f"{sha256_file(bank_root / 'bank_manifest.json')}\n", encoding="utf-8")
	os.replace(ready_partial, bank_root / "READY")
	return manifest


def _broadcast_object(value: Any, rank: int) -> Any:
	values = [value if rank == 0 else None]
	dist.broadcast_object_list(values, src=0)
	return values[0]


def run_candidate_bank_encoding(args: argparse.Namespace) -> None:
	"""Encode all eight canonical candidate banks with one frozen model load per rank."""
	rank, world_size, device = _initialize_distributed(args.expected_world_size)
	logging.basicConfig(
		level=logging.INFO,
		format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
	)
	output_root = Path(args.output_root)
	if rank == 0:
		output_root.mkdir(parents=True, exist_ok=True)
		write_json_atomic(output_root / "status.json", {"status": "initializing"})
	dist.barrier()
	model_root = Path(args.model_root)
	checkpoint_path = model_root / "model.safetensors"
	model_sha256 = _broadcast_object(
		checkpoint_sha256(checkpoint_path) if rank == 0 else None,
		rank,
	)
	code_commit = _broadcast_object(
		_git_commit(Path(args.project_root)) if rank == 0 else None,
		rank,
	)
	processor = BaselineInputProcessor.from_pretrained(
		model_root,
		max_length=args.max_length,
		min_pixels=args.min_pixels,
		max_pixels=args.max_pixels,
	)
	model = load_frozen_evaluation_model(
		model_root,
		dtype=torch.float16,
		attention_implementation=args.attention_implementation,
	).to(device)
	if any(parameter.requires_grad for parameter in model.parameters()):
		raise RuntimeError("Candidate encoder unexpectedly contains trainable parameters")
	gpu_names = [None for _ in range(world_size)]
	dist.all_gather_object(gpu_names, torch.cuda.get_device_name(device))
	for spec in CANDIDATE_BANK_SPECS:
		bank_root = output_root / spec.relative_path
		dataset_root = Path(args.dataset_root) / spec.dataset
		started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
		preparation: dict[str, Any] | None = None
		if rank == 0:
			if (bank_root / "READY").is_file():
				load_ready_candidate_bank(
					bank_root,
					expected_spec=spec,
					expected_model_sha256=model_sha256,
				)
				preparation = {"skip": True}
			else:
				bank_root.mkdir(parents=True, exist_ok=True)
				item_metadata = _prepare_item_manifest(
					dataset_root=dataset_root,
					bank_root=bank_root,
					spec=spec,
				)
				config = _immutable_bank_config(
					args=args,
					spec=spec,
					item_metadata=item_metadata,
					model_sha256=model_sha256,
					code_commit=code_commit,
				)
				_publish_or_validate_bank_config(bank_root, config)
				preparation = {
					"skip": False,
					"items": item_metadata,
					"config": config,
				}
			write_json_atomic(
				output_root / "status.json",
				{"status": "encoding", "bank": spec.key, **preparation},
			)
		preparation = _broadcast_object(preparation, rank)
		if preparation["skip"]:
			LOGGER.info("bank=%s already ready and fully validated", spec.key)
			dist.barrier()
			continue
		item_metadata = preparation["items"]
		config = preparation["config"]
		ranges = embedding_shard_ranges(
			int(item_metadata["rows"]),
			args.embedding_shard_rows,
		)
		for shard_index, (start, end) in enumerate(ranges):
			if shard_index % world_size != rank:
				continue
			shard_path = bank_root / "embedding_shards" / f"part-{shard_index:05d}.pt"
			if _validate_existing_embedding_shard(shard_path, start, end):
				LOGGER.info("bank=%s shard=%d already valid", spec.key, shard_index)
				continue
			shard_start = time.perf_counter()
			embeddings = _encode_item_range(
				items_root=bank_root / "items",
				start=start,
				end=end,
				input_kind=str(item_metadata["input_kind"]),
				model=model,
				processor=processor,
				device=device,
				args=args,
			)
			_save_embedding_shard(
				shard_path,
				start=start,
				end=end,
				embeddings=embeddings,
			)
			LOGGER.info(
				"bank=%s shard=%d range=[%d,%d) items/s=%.2f",
				spec.key,
				shard_index,
				start,
				end,
				(end - start) / (time.perf_counter() - shard_start),
			)
		dist.barrier()
		if rank == 0:
			_finalize_bank(
				bank_root=bank_root,
				config=config,
				item_metadata=item_metadata,
				started_at=started_at,
				world_size=world_size,
				gpu_names=[str(name) for name in gpu_names],
			)
		dist.barrier()
	if rank == 0:
		checkpoint_hash_after = checkpoint_sha256(checkpoint_path)
		if checkpoint_hash_after != model_sha256:
			raise RuntimeError("Immutable Qwen checkpoint changed during candidate encoding")
		write_json_atomic(
			output_root / "status.json",
			{
				"status": "passed",
				"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
				"candidate_bank_count": len(CANDIDATE_BANK_SPECS),
				"model_checkpoint_sha256": model_sha256,
			},
		)
	dist.barrier()
	dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--project-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/loopedTransformer"),
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/datasets/looped_vl_single_baselines_v1"),
	)
	parser.add_argument(
		"--model-root",
		type=Path,
		default=Path("/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original"),
	)
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--expected-world-size", type=int, default=8)
	parser.add_argument("--image-batch-size", type=int, default=32)
	parser.add_argument("--text-batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--prefetch-factor", type=int, default=2)
	parser.add_argument("--embedding-shard-rows", type=int, default=DEFAULT_EMBEDDING_SHARD_ROWS)
	parser.add_argument("--max-length", type=int, default=8192)
	parser.add_argument("--min-pixels", type=int, default=4 * 32 * 32)
	parser.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
	parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
	parser.add_argument("--visual-length-buckets", type=int, default=DEFAULT_VISUAL_LENGTH_BUCKETS)
	parser.add_argument(
		"--min-visual-bucket-size",
		type=int,
		default=DEFAULT_MIN_VISUAL_BUCKET_SIZE,
	)
	args = parser.parse_args()
	for name in (
		"expected_world_size",
		"image_batch_size",
		"text_batch_size",
		"num_workers",
		"prefetch_factor",
		"embedding_shard_rows",
		"max_length",
		"min_pixels",
		"max_pixels",
		"visual_length_buckets",
		"min_visual_bucket_size",
	):
		if getattr(args, name) <= 0:
			parser.error(f"--{name.replace('_', '-')} must be positive")
	return args


def main() -> int:
	args = parse_args()
	try:
		run_candidate_bank_encoding(args)
		return 0
	except KeyboardInterrupt:
		return 130
	except Exception as error:
		if int(os.environ.get("RANK", "0")) == 0:
			output_root = Path(args.output_root)
			if output_root.exists():
				write_json_atomic(
					output_root / "status.json",
					{"status": "failed", "error": repr(error)},
				)
		print(f"candidate-bank encoding failed: {error!r}", file=sys.stderr, flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
