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
- [x] (Superseded) Implement the former zero-output recurrent connector and
  EOS-conditioned late fusion.
- [x] Implement Base, EOS-only, slots-only, full, slot-count, and loop-count variants.
- [x] Supersede the original Stage 1/Stage 2 allowlists with one optimizer containing
  warm-start and joint-only parameter groups.
- [x] Implement deterministic paired training data, optimizer, scheduler, and full RNG resume.
- [x] Add required per-step and per-recurrent-pass diagnostics.
- [x] Pass unit, integration, equivalence, gradient, and two-GPU training smoke tests.
- [x] Monitor both GPUs idle for three continuous minutes before the full launch.
- [x] Record the historical two-stage launch and prevent its checkpoints from resuming
  under the new single-stage protocol.
- [x] Add tests for modality-grouped padding, vectorized slot insertion, fused recurrent
  attention, and Pass-1 Key/Value reuse.
- [x] Select attention by GPU capability: FlashAttention 2 on supported Ampere-or-newer
  GPUs and PyTorch scaled dot-product attention on V100.
- [x] Add safe training optimizations: homogeneous modality batches, fused AdamW, static
  gradient buckets, target-token caching, and optional semantic gradient checkpointing.
- [x] Force DataLoader workers to use `spawn` so they cannot inherit and retain CUDA
  contexts after a rank failure.
- [x] Skip multi-gigabyte optimizer checkpoints during disposable smoke benchmarks.
- [x] Record the historical eight-V100 two-stage smoke benchmarks at batch size 8.
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
- [x] Add exact resumable baseline state and make baseline/recurrent training dynamically
  retain only the latest checkpoint, including the end-of-epoch checkpoint.
- [x] Average baseline loss and embedding norms over every microbatch and every rank,
  while skipping scheduler advancement after a non-finite FP16 optimizer step.
- [x] (Superseded) Replace recurrent Stage 1/Stage 2 with one continuous one-epoch task whose
  auxiliary-emphasis length is `ceil(0.35 * train_rows / optimizer_global_batch_size)`.
- [x] (Superseded) Keep final InfoNCE at weight 1.0 for the full epoch and activate every recurrent
  parameter group from step one; use the first 35% only to raise auxiliary slot InfoNCE
  from weight 0.2 to 1.0.
- [x] Remove the Qwen3-0.6B semantic decoder and its loss, data, checkpoint, and command
  paths from recurrent training.
- [x] Replace auxiliary mean pooling with fixed layer-20 EOS-conditioned soft weighting
  over every current-pass slot; retain one shared training-only RMSNorm and projection.
- [x] Group baseline inputs by text/vision modality before Qwen preprocessing and restore
  their exact logical pair order before the contrastive loss.
- [x] Cross-audit baseline and recurrent runs as source-pure one-epoch training followed
  directly by held-out test, with no validation-based selection.
- [x] Unify exact in-place resume, metric-log rollback, 100-step checkpoint cadence, and
  rolling retention of only the latest checkpoint file for both trainers.
- [x] Make recurrent loss and diagnostic logs sample-weighted across every microbatch and
  every distributed rank, and fail instead of silently losing a non-finite update.
- [x] Add one restartable queue for the three baseline and three recurrent train/test
  experiments, with no idle-time gate and no validation command.
- [x] Re-run final eight-V100 safety smokes for both trainers and every dataset-specific
  recurrent memory policy.
- [x] (Superseded) Cancel the old recurrent half of the restartable six-run queue; retain
  completed baseline results and replace recurrent v5 with the query-only queue below.
- [x] Evaluate the frozen original Qwen3-VL-Embedding-2B serially on the full COCO, GQA
  Balanced, and CLEVR test splits before restarting unfinished training experiments.
- [x] Standardize the cross-model result table to one primary row per dataset: COCO uses
  the equal-direction mean; GQA Balanced and CLEVR use answer retrieval.
- [x] Remove the incorrectly coupled Layers 13–20 LoRA modules from recurrent
  configuration, training, evaluation, checkpoints, and parameter accounting.
- [x] Stamp the previous recurrent manifest, training result, checkpoint, and test report with
  the pure-recurrent/no-LoRA identity and reject incompatible result inputs.
- [x] Use CPU Gloo collectives for recurrent evaluation and record the final-pass primary
  metrics, per-pass metrics, exact peak GPU memory, and global encoding throughput.
- [x] Re-run recurrent safety and throughput smokes under the damped recurrent
  protocol; older recurrent smokes contained LoRA or the removed connector and cannot
  select the formal configuration.
- [x] (Superseded) Lock the recurrent v2 forward path to K=8, R=4, slots-only extra
  passes, fixed EOS after Pass 1, and parameter-free damping with step size `1 / R`.
- [x] Remove the learned recurrent connector from the model, trainable parameters,
  checkpoints, evaluation loader, and diagnostics.
- [x] Replace final-slot-only warm-up supervision with one shared 256-dimensional
  auxiliary retrieval head applied after every complete recurrent pass.
- [x] Lock one-stage training to fixed loss weights: final InfoNCE 1.0, mean per-round
  auxiliary InfoNCE 0.1, and absolute-cosine final-slot diversity 0.05.
- [x] Reject old recurrent checkpoints through the v3 architecture and v4 training
  protocol identity.
- [x] Run a new eight-V100 architecture, gradient, memory, and throughput smoke for the
  damped connector-free model before launching any formal recurrent experiment.
- [x] (Superseded) Retire the old K=8/16/32/64 GQA recurrent-v5 smoke plan.
- [x] (Superseded) Retire the old causal-attention recurrent-v5 COCO K=8/12/16/32 plan.
- [x] Train and evaluate rank-32 LoRA limited to decoder layers 24–27 on full COCO,
  GQA Balanced, and CLEVR, preserving the original all-layer LoRA baseline.
- [x] Replace the ineffective recurrent v3 path with the direct-EOS recurrent v5 path:
  no LoRA, shared per-layer recurrent channel scales, pass-count-independent step size,
  parameter-free slot attention supervision, and a 5,000,000-parameter hard limit.
- [x] Queue the three last-four-layer LoRA runs separately and cancel their obsolete
  recurrent-v5 continuation; the active query-only experiments use a new isolated queue.
- [x] Define the canonical eight-bank candidate layout: six split-specific COCO image and
  caption galleries plus one shared training-answer gallery for each of GQA Balanced and
  CLEVR.
- [x] Implement deterministic, resumable frozen-Qwen candidate encoding with immutable
  manifests, checksums, contiguous float16 embedding shards, and atomic `READY` publication.
- [x] Add local tests for candidate identity, ordering, image deduplication, shared answer
  galleries, indexed text/image reading, embedding validation, and published-bank loading.
- [x] Encode and validate all eight candidate banks on the explicitly selected compute
  route without overwriting any existing published bank.
- [x] Make the new query-only recurrent trainer and evaluator require the validated
  immutable candidate banks and prove that the candidate Qwen tower is never executed.
- [x] Implement the under-5M query-only history recurrent Block, zero-gated fusion,
  EOS-conditioned slot attention pooling, dynamic exit, and pass-wise loss.
- [x] Preserve logical contrastive batches while splitting only Qwen and recurrent-head
  encoding by modality and visual length, avoiding cross-bucket history padding.
- [x] Add one resumable eight-run queue for COCO loop/exit/slot/history ablations plus
  canonical GQA Balanced and CLEVR runs, all one epoch with no validation.
- [x] Complete one eight-V100 batch-32 smoke for the latest query-only code and verify all
  ranks, finite gradients, peak memory, throughput, and zero candidate-Qwen calls.
- [x] Train and fully test the COCO K=8/R=1 fixed-exit control; record its Pass 0 and
  Pass 1 metrics, runtime, memory, candidate-bank identity, and one-checkpoint proof.
- [x] Train and fully test the COCO K=8/R=4 fixed and dynamic controls. Record every
  pass and both exit outputs; the first dynamic controller selected Pass 4 for every
  test query and therefore did not provide dynamic compute savings.
- [x] Train and fully test the COCO K=1 and K=4 dynamic slot-count ablations. Both
  selected Pass 4 for every test query; neither improved meaningfully over K=8.
- [x] Train and fully test the COCO Layer-28-only-history ablation. It was within
  0.0051 mAP points of four histories and again selected Pass 4 for every query.
- [x] Train and fully test the canonical GQA Balanced query-only recurrent run; record
  its +13.1957-point primary mAP gain and all Pass 0–4 metrics.
- [x] Resume and complete the canonical CLEVR query-only recurrent run from its only
  rolling step-1000 checkpoint, lowering the restored FP16 gradient scale from 4,096
  to 2,048 and retaining only the final step-2,735 checkpoint.
- [x] Train and test all eight query-only recurrent runs after all candidate banks pass
  checksum validation; record Pass 0 through Pass 4 and dynamic-exit metrics in result.md.
- [x] Diagnose the completed COCO v1 runs: verify the data and banks, quantify slot collapse,
  ineffective recurrence/history/exit controls, zero-gate gradient starvation, mixed-gallery
  negatives, and the much smaller embedding movement than last-four-layer LoRA.
- [x] Implement the v2 direction-isolated InfoNCE, direct supervision for every fused pass,
  progressive improvement margin, zero-initialized full residual projection, recurrent slot
  identity injection, and full-bank same-gallery hard-negative mining.
- [x] Remove dynamic exit from the first v2 architecture after the v1 controller selected
  maximum Pass 4 for every COCO query; lock training and evaluation to explicit Pass R.
- [x] Implement and test the last-four-layer query-only LoRA control against the same frozen
  candidate banks without changing the ordinary two-tower LoRA baseline.
- [x] Stop the query-only LoRA control at its next rolling checkpoint (step 400), retain
  only that checkpoint, and postpone the control until the recurrent root cause is known.
- [x] Add pass-wise embedding movement, slot-collapse, slot-attention, and component-gradient
  diagnostics without changing the v2 forward result or loss.
- [x] Run the 200-step eight-V100 COCO v2 training diagnostic: it showed nearly uniform
  slot attention, approximately 0.999 slot cosine, and unbounded movement far from frozen
  Qwen; retain its exact step-200 checkpoint as root-cause evidence.
- [x] Fix the recurrent evaluator bug that incorrectly appended finite embeddings only
  inside the non-finite error branch, and add a regression test.
- [x] Re-test the v2 diagnostic checkpoint with the corrected evaluator; Pass 4 recovered
  only to 53.6398 mAP versus the frozen Pass-0 value of 61.2489, confirming destructive
  unbounded residual movement rather than an evaluator-only failure.
- [x] Restore the specified Xavier-projected, scalar `tanh` zero-gated residual fusion as
  the isolated v3 candidate fix without adding dynamic exit or LoRA.
- [x] Run the v3 200-step training diagnostic; its gate controlled the initial update, but
  the unnormalized 2,048-dimensional residual still reached L2 movement 0.44 by step 200
  while slots remained collapsed.
- [x] Complete the full Pass-0-to-Pass-4 test for the v3 zero-gated candidate; Pass 4
  reached 60.5458 mAP versus 61.2489 at Pass 0, so a scalar gate alone did not bound the
  unnormalized residual magnitude.
- [x] L2-normalize the residual direction before the scalar gate so projection magnitude
  cannot bypass zero-gated fusion; add a scale-invariance regression test.
- [x] Run the same 200-step COCO quality diagnostic for the v4 unit-residual candidate;
  Pass 2 reached 61.3239 mAP (+0.0751 over Pass 0) while unit residuals kept mean movement
  near 0.01, but Passes 2–4 remained nearly identical.
- [x] Diagnose that v4 step 50 still has approximately 0.999 slot cosine and near-uniform
  EOS-conditioned slot attention despite fixing residual scale.
- [x] Add RMS-normalized persistent slot identity to every recurrent self-attention and
  history-attention query, replacing the ineffective small raw-state reinjection.
- [x] Remove the recurrent launcher's eager PEFT import path and configure the V100 image's
  documented protobuf compatibility mode before unavoidable Qwen vision imports reach its
  legacy ONNX package.
- [x] Run the same 200-step COCO quality diagnostic for the v5 persistent-identity
  candidate. It increased slot collapse and remained flat from Pass 1 through Pass 4;
  its best mAP was 61.3232 versus v4's 61.3239, so reject the mechanism.
- [x] Add a no-parameter, training-only InfoNCE loss on every 2,048-dimensional unit slot
  proposal so the residual projection and recurrent Block receive gradients while the
  inference residual gate remains initialized to exactly zero; restore the v4 recurrent
  state path so this v6 diagnostic changes only the supervision mechanism relative to v4.
- [x] Run the same 200-step COCO quality diagnostic for the v6 slot-proposal-supervised
  candidate. It amplified recurrent gradients by over two orders of magnitude and reduced
  slot collapse, but bare-proposal InfoNCE optimized the wrong additive geometry: best
  mAP was only 61.2706 and later passes degraded.
- [x] Replace bare-proposal supervision with a training-only fixed-scale bridge equal to
  `L2Norm(frozen_embedding + 0.1 * proposal)`, preserving the zero-gated inference path,
  parameter count, data, and all other loss settings in the v7 candidate.
- [x] Run the same 200-step COCO quality diagnostic and full Pass-0-to-Pass-4 test for v7.
  The fixed-scale bridge restored moderate early gradients and the v4 quality level, but
  Pass 2 improved over Pass 1 by only 0.0003 mAP and Passes 3–4 regressed. The recurrent
  state reached a fixed point after approximately one full update.
- [x] Use the v4–v7 diagnostic evidence to isolate gradient reachability from recurrent
  update stability: v6 proved the former was broken, v7 fixed it in the correct additive
  geometry, and v7 then proved one-step convergence remained the dominant failure.
- [x] Add parameter-free inverse-R recurrent-state damping while preserving R=1 exactly;
  support fixed R=1/2/3/4 without LoRA or dynamic exit and stamp the new v8 identity.
- [x] Run the same 200-step COCO quality diagnostic and full Pass-0-to-Pass-4 test for the
  v8 damped candidate. Pass 1 reached 61.3266 mAP (+0.0777 over frozen), but Pass 4 fell
  to 61.3237 and no later pass beat Pass 1. At step 200, per-pass movement after Pass 1
  shrank from 0.00103 to 0.00056 and 0.00038 while final slot cosine reached 0.9987.
  Therefore update magnitude was not the root cause; every-pass supervision trained a
  one-step fixed-point mapping.
- [x] Replace every-pass and progressive supervision with final-pass-only fused and bridge
  InfoNCE in the v9 candidate. Keep the v8 damping, model, data, parameter count, and all
  other settings fixed; intermediate passes remain required evaluation outputs.
- [ ] Run the same 200-step COCO quality diagnostic and full Pass-0-to-Pass-4 test for the
  v9 final-pass-only candidate, checking whether later passes keep moving and beat both
  Pass 0 and Pass 1.
- [ ] Change one failed mechanism at a time and rerun the same fixed diagnostic window;
  promote no architecture to full training until a later pass beats both Pass 0 and Pass 1.
