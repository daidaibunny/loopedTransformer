# TO-DO

- [x] Add failing tests for Parquet indexing, image resolution, batching, and frozen smoke guards.
- [x] Implement the mixture dataset and GQA materialized-image resolver.
- [x] Implement source-balanced smoke sample selection and batch collation.
- [x] Materialize and validate GQA Balanced train and validation images.
- [x] Run unit tests and linting locally or in the configured remote environment.
- [x] Run frozen Qwen3-VL-Embedding-2B smoke on GPU 1.
- [x] Verify embeddings and confirm the checkpoint hash is unchanged.
- [x] Benchmark 4,000 mixed-resolution samples with official Qwen preprocessing limits.
- [x] Audit raw and processed resolutions for the exact throughput sample window.
- [x] Create and validate the 100,000-row train subset with the exact 50:35:15 ratio.
- [x] Replace the oversized held-out split with disjoint 10,000-row validation and test
  sets, defined strictly by sample count.
- [x] Freeze and enforce the required Mix and per-dataset evaluation metric contract.
- [x] Verify the installed Qwen3-VL implementation and exact layer/module boundaries.
- [x] Add failing acceptance tests for sections 27–29 of the recurrent v1.0 specification.
- [x] Implement latent-slot insertion and the shared 16-slot seed-42 initialization file.
- [x] Implement four-pass Layers 13–20 recurrence with detached prefix K/V evidence.
- [x] Implement the zero-output recurrent connector and EOS-conditioned late fusion.
- [x] Implement Base, EOS-only, slots-only, full, slot-count, and loop-count variants.
- [x] Implement warm-up heads, losses, Stage 1, and Stage 2 LoRA trainability allowlists.
- [x] Implement deterministic paired training data, optimizer, scheduler, and full RNG resume.
- [x] Add required per-step and per-recurrent-pass diagnostics.
- [x] Pass unit, integration, equivalence, gradient, and two-GPU training smoke tests.
- [x] Monitor both GPUs idle for three continuous minutes before the full launch.
- [x] Launch the two-stage training in a unique tmux session and verify stable progress.
- [x] Add tests for modality-grouped padding, vectorized slot insertion, fused recurrent
  attention, and Pass-1 Key/Value reuse.
- [x] Make FlashAttention 2 the explicit default for backbone, semantic decoder, smoke,
  throughput, and frozen evaluation.
- [x] Add safe training optimizations: homogeneous modality batches, fused AdamW, static
  gradient buckets, target-token caching, and optional semantic gradient checkpointing.
- [x] Force DataLoader workers to use `spawn` so they cannot inherit and retain CUDA
  contexts after a rank failure.
- [x] Skip multi-gigabyte optimizer checkpoints during disposable smoke benchmarks.
- [ ] Run the optimized two-GPU Stage 1 and Stage 2 smoke benchmarks at batch size 8.
- [ ] Estimate one 100,000-row training epoch; create a 50,000-row train subset only if the
  measured upper estimate exceeds one hour.
- [ ] Audit the full official COCO, GQA Balanced, and CLEVR splits on `8XV100`.
- [ ] Profile one-dataset training on eight V100 GPUs and compare the measured COCO epoch
  time with the reported 20-minute reference.
- [ ] Train and evaluate COCO, GQA Balanced, and CLEVR independently on their full splits.
- [ ] Report retrieval metrics after recurrent passes 1, 2, 3, and 4, including the change
  from the previous pass and from pass 1.
- [x] Freeze the independent baseline split protocol: COCO Karpathy, GQA Balanced
  train/validation/testdev, and an image-disjoint seed-42 split of labeled CLEVR validation.
- [x] Add failing tests for full-data manifests, duplicate-positive contrastive loss,
  official Qwen LoRA configuration, and the eight-GPU idle queue.
- [x] Implement the unmodified Qwen3-VL-Embedding-2B LoRA trainer and single-dataset test
  evaluator without using the 50:35:15 mixture.
- [ ] Materialize and validate the three full single-dataset manifests on `8XV100`.
- [ ] Run eight-V100 smoke searches for COCO, GQA Balanced, and CLEVR.
- [ ] Queue all three full LoRA train/test runs after a continuous two-minute idle window.
