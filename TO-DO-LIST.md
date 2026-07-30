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
- [x] Select attention by GPU capability: FlashAttention 2 on supported Ampere-or-newer
  GPUs and PyTorch scaled dot-product attention on V100.
- [x] Add safe training optimizations: homogeneous modality batches, fused AdamW, static
  gradient buckets, target-token caching, and optional semantic gradient checkpointing.
- [x] Force DataLoader workers to use `spawn` so they cannot inherit and retain CUDA
  contexts after a rank failure.
- [x] Skip multi-gigabyte optimizer checkpoints during disposable smoke benchmarks.
- [x] Run optimized eight-V100 Stage 1 and Stage 2 smoke benchmarks at batch size 8.
- [x] Add V100-safe FP16 automatic mixed precision with FP32 trainable parameters,
  gradient scaling, and resumable scaler state.
- [x] Audit the full official COCO, GQA Balanced, and CLEVR splits on `8XV100`.
- [x] Profile one-dataset training on eight V100 GPUs and compare the measured COCO epoch
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
- [x] Materialize and validate the three full single-dataset manifests on `8XV100`.
- [x] Run eight-V100 smoke searches for COCO, GQA Balanced, and CLEVR.
- [x] Queue all three full LoRA train/test runs after a continuous two-minute idle window.
- [x] Make recurrent training, acceptance, and evaluation directly reuse the latest
  baseline train/validation/test manifests and reject the older recurrent-only splits.
- [x] Make baseline and recurrent COCO training share one deterministic 50:50
  text-to-image/image-to-text direction rule without changing manifest row counts.
- [x] Require the baseline negative pool itself to contain 256 pairs and search
  eight-V100 batch-32 throughput without treating gradient accumulation as negatives.
- [x] Add exact resumable baseline state and cap baseline/recurrent training checkpoints
  at four files per experiment, including the final checkpoint.
- [x] Average baseline loss and embedding norms over every microbatch and every rank,
  while skipping scheduler advancement after a non-finite FP16 optimizer step.
- [x] Group baseline inputs by text/vision modality before Qwen preprocessing and restore
  their exact logical pair order before the contrastive loss.
