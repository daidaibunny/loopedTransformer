# Looped VL

## Scope

- This repository is exclusively for Recurrent Latent-Slot Qwen3-VL-Embedding work.
- Do not add, import, or run PoLar code in this repository.
- The formal recurrent model is pure recurrent and must contain no LoRA modules or
  parameters. LoRA belongs only to the independent comparison code under
  `src/looped_vl/baseline/`.
- The active diagnostic candidate is
  `query_only_history_recurrent_no_lora_v3_candidate`. The completed v1 query-only runs,
  the rejected unbounded-residual v2 diagnostic, and former
  `direct_eos_layerscale_mid_decoder_recurrence_no_lora_v5` queue remain historical only;
  never launch them as current experiments.
- Use one training stage and one full-data epoch. All trainable recurrent parameters are
  active from the first optimizer step; do not use validation or checkpoint selection.
- The completely frozen Qwen query tower runs exactly once. It exposes the token states
  after decoder Layers 7, 14, 21, and 28 plus the official final-valid-token 2,048-
  dimensional embedding. The Layer-28-only history is an explicit ablation.
- Project the frozen histories into a 288-dimensional state. Initialize K latent slots by
  cross-attention, then apply one shared two-layer recurrent Block R times. Each Block
  layer contains slot self-attention, cross-attention to the unchanged Qwen histories,
  and a feed-forward network. The default is K=8 and R=4; formal ablations use K=1/4/8
  and R=1/4.
- Use `EOS-conditioned slot attention pooling` after every recurrent pass: the frozen
  final-valid-token embedding selects useful slots with soft attention. Project the
  selected state to 2,048 dimensions with Xavier initialization, multiply it by a shared
  `tanh` scalar gate initialized to zero, add it to the frozen Qwen embedding, then
  L2-normalize. This `zero-gated residual fusion` makes every pass exactly equal to frozen
  Qwen at initialization. The gate receives the first optimizer-step gradient; the
  residual projection and recurrent path receive gradients after the gate becomes nonzero.
- The first v2 uses an explicit fixed recurrent count only. It contains no exit controller,
  exit threshold, halting loss, or compute penalty. Reconsider dynamic exit only after a
  fixed Pass-4 model shows a measurable gain over a separately trained fixed Pass-1 model.
- Train only the slot initializer, shared recurrent Block, history projection,
  and zero-initialized residual readout. Enforce the 5,000,000-parameter limit. Exact v2
  candidate counts are 4,852,801 for K=1, 4,853,665 for K=4, and 4,854,817 for K=8.
- Keep the original Qwen3-VL checkpoint immutable. Save learned parameters and checkpoints
  only under experiment-specific output directories.
- Always call the single-query attention from the final valid token over latent slots
  `EOS-conditioned slot attention pooling` (`EOS 条件化槽位注意力池化`). Do not shorten
  it to ordinary pooling. Never use mean pooling in a formal run.

## Remote execution

- There are two independent compute routes: `8XV100` and `2XA800`. Follow the route
  separation, GPU guard, SSH, storage, and scheduling rules in the workspace-root
  `AGENTS.md`.
- Select one route explicitly for every remote task. Never mix code, data, environments,
  processes, checkpoints, logs, or results between the two routes.
- The currently selected route is `8XV100`. Use SSH alias `8XV100`, verify that the host
  has eight NVIDIA V100 GPUs before every launch, and use all eight GPUs when the program
  supports distributed execution.
- On `8XV100`, discover the project, data, model, and Python environment under
  `/home/mnt/liyiwei` before use. Do not reuse `/mnt/afs` paths or the shared LOCUS
  interpreter from `2XA800`.
- `2XA800` remains available for tasks that explicitly select it. Its operational target
  is SSH alias `gyy1`, with project root `/mnt/afs/liyiwei/loopedTransformer` and the two
  assigned physical GPUs 0 and 1.
- Run long jobs in uniquely named detached tmux sessions with separate logs.
- Before an `8XV100` launch, pause and verify the persistent idle GPU guard exactly as
  specified in the workspace-root `AGENTS.md`. Resume it after the real task finishes.
- Treat physical batch size and gradient accumulation as route-specific settings. Measure
  them on the selected hardware while preserving the required effective global batch size
  and all result-affecting model settings.
- After launch, poll every distributed rank, GPU use, progress logs, and checkpoints for
  several minutes.

## Data

- Resolve dataset roots under the selected route's storage mount. Do not copy a path from
  one compute route into commands for the other route.
- The old mixed subset on `2XA800` is
  `/mnt/afs/liyiwei/datasets/looped_vl_mix_v1_train100000_val10000_test10000`; it is not
  the input to the first single-dataset experiments.
- The first experiments train and evaluate COCO, GQA Balanced, and CLEVR independently,
  using the frozen baseline manifests rather than the 50:35:15 mixture or the older
  independently generated recurrent manifests.
- The baseline manifests are the only split authority for both the unmodified LoRA
  baseline and recurrent experiments:
  - COCO uses the Karpathy 113,287/5,000/5,000 image-disjoint train/validation/test split.
  - GQA Balanced uses official train/validation/testdev.
  - CLEVR uses full official train and the seed-42 image-disjoint halves of official
    validation for validation/test.
- Full-layer and last-four-decoder-layer LoRA baselines must read the same raw baseline
  Parquet rows for every dataset. They encode both query and candidate online with the
  active Qwen model; they must never read an immutable candidate bank. Candidate banks
  belong only to the query-only recurrent experiments below.
- Recurrent training, acceptance, and evaluation must read those exact baseline Parquet
  files directly. Do not copy rows into another recurrent dataset or use
  `looped_vl_single_v1/{coco_full,gqa_balanced_full,clevr_full}`.
- Keep the 50:35:15 mixture only for later mixed-dataset experiments and aggregate reports.
- For the later mixed subset only, define split sizes by sample rows: 100,000 train,
  10,000 validation, and 10,000 test.
- Resolve COCO and CLEVR with `image_path`.
- Resolve GQA Balanced from its materialized image cache by `image_id`.
- Preserve the exact source ratio in train, validation, and test.
- Never use test samples for tuning or checkpoint selection.

## Immutable candidate banks

- Every future recurrent experiment must use version
  `frozen_qwen3vl_candidate_bank_v1`; it must not encode candidates during training or
  evaluation.
- Maintain exactly eight candidate banks:
  - COCO train, validation, and test each have one deduplicated image gallery and one
    complete caption gallery.
  - GQA Balanced has one training-answer gallery shared by train, validation, and test.
  - CLEVR has one training-answer gallery shared by train, validation, and test.
- Candidate ordering comes only from the frozen baseline Parquet files and training-answer
  galleries. Preserve `item_index`, `item_id`, and `positive_id`; never reorder or silently
  drop candidates.
- Encode candidates with the completely frozen original Qwen3-VL-Embedding-2B checkpoint.
  Use its official final valid-token readout, 2,048 dimensions, L2 normalization, and
  float16 storage. Candidate inputs omit a task-specific instruction; the unchanged Qwen
  processor supplies its fixed generic system message.
- A candidate bank is usable only when its `READY` checksum matches `bank_manifest.json`,
  the base-model checksum matches, the item-manifest checksum matches, every embedding
  shard checksum and contiguous range matches, and every stored vector is finite and unit
  normalized.
- Candidate banks are immutable after `READY` publication. A changed source manifest,
  answer gallery, model checkpoint, preprocessing setting, candidate order, or code commit
  requires a new output root and a new bank version; never overwrite a published bank.
- Future query-only recurrent training keeps candidate tensors detached and loads them by
  stable candidate index or identifier. The candidate Qwen tower must have zero forward
  calls and zero trainable parameters.

## Checkpoint policy

- Baseline and recurrent full training save an exact resumable checkpoint every 100
  optimizer steps and at the end of the epoch.
- Each experiment dynamically retains only its newest resumable checkpoint. Saving a new
  checkpoint must remove the previous one only after the new file exists and the
  latest-checkpoint pointer has been atomically published.
- Keep the final adapter or final recurrent learned weights separately from the rolling
  resumable checkpoint. Evaluation reads those final learned weights, not an intermediate
  optimizer checkpoint.
- Reject recurrent checkpoints containing LoRA parameters or any superseded architecture
  or training protocol.
- Every new recurrent `run_manifest.json`, `training_result.json`, checkpoint metadata,
  and `report.json` must declare architecture
  `query_only_history_recurrent_no_lora_v3_candidate`, protocol
  `single_stage_fixed_recurrence_zero_gated_v3_candidate`, `backbone_frozen: true`,
  `candidate_backbone_executed: false`,
  `lora_enabled: false`, and `formal_training_stages: 1`. Reject missing or
  conflicting identities.
- Recurrent training runs for exactly one epoch. Compute InfoNCE separately inside each
  candidate gallery; a COCO text-to-image query must never use caption candidates, and an
  image-to-text query must never use image candidates. Directly supervise every fused
  recurrent pass with linearly increasing normalized pass weights. Use progressive margin
  0.02 with weight 0.1 and mine 32 same-gallery hard negatives from the complete immutable
  bank while excluding every matching `positive_id`. Slot absolute cosine is logged but
  has weight 0.0. All trainable groups use one learning-rate schedule from step one.
- The last-four-layer query-only LoRA control is separate from the ordinary two-tower LoRA
  baseline. It may read frozen candidate banks, but must declare the query-only control
  scope, keep candidate Qwen forward calls at zero, use the same gallery-isolated objective
  and 32 same-gallery hard negatives as recurrent v2, and never replace the existing
  baseline. It must target decoder-layer indices 24, 25, 26, and 27 only.

## Verification

- Add tests before implementation changes.
- Verify all dataset backends, batching, processor inputs, embedding shapes, finite values,
  unit norms, and model checkpoint hash stability.
- Enforce the locked no-LoRA, single-stage, connector-free architecture with structural,
  trainability, determinism, checkpoint-compatibility, and numerical tests.

## Required evaluation metrics

- Every evaluation report must pass `looped_vl.metrics.validate_evaluation_report`.
- For the first single-dataset experiments, report each dataset independently. COCO must
  report text-to-image and image-to-text separately.
- In every unified cross-model table, use exactly one primary row per model and dataset:
  the equal-direction mean for COCO, and the answer-retrieval row for GQA Balanced and
  CLEVR. Keep COCO direction-specific rows only as diagnostic details.
- Compare model-quality metrics only within the same dataset and exact test manifest and
  candidate gallery. Do not treat values from different retrieval tasks as directly
  comparable.
- Every current recurrent report must retain the complete required metric set for frozen
  Qwen Pass 0 and recurrent Pass 1 through the configured fixed Pass R. It must report each
  pass's metric change from frozen Qwen. For COCO, summaries use the equal-direction mean
  while direction-specific pass metrics remain required. Historical v1 reports may retain
  their obsolete dynamic hard/soft exit diagnostics.
- Report the weighted Mix result only for later mixed-dataset experiments.
- Required metrics are mAP, P@1/5/10/20, R@1/5/10/20, MRR, and nDCG@10.
- Use percentage values from 0 to 100. Aggregate COCO directions equally, then aggregate
  datasets with fixed weights COCO:GQA Balanced:CLEVR = 50:35:15.

## Experiment result registry

- The repository-root `result.md` is the mandatory central experiment comparison record.
  Raw `run_manifest.json`, `training_result.json`, and `report.json` files remain the
  source of truth.
- Update `result.md` after every formal experiment reaches passed, failed, or interrupted
  status. Also record a smoke run when it selects a production batch size, memory setting,
  checkpoint policy, or other result-affecting runtime choice.
- Never report a running experiment as completed. Use `Pending`, `Running`, `Failed`,
  `Interrupted`, or `N/A` explicitly, and never infer a missing value.
- Every record must contain:
  - experiment identity, objective, terminal status, start/end time, route, node, and code
    commit;
  - dataset and split, sample rows, unique images, and manifest checksum;
  - base-model checksum, architecture or recurrent pass count, trainable scope/count, and
    final checkpoint or adapter checksum;
  - seed, precision, attention implementation, GPU/world size, per-device batch,
    contrastive global batch, optimizer global batch, optimizer, learning rate, schedule,
    epochs/steps, and checkpoint policy;
  - query/candidate gallery protocol, validation-use status, full required test metrics,
    wall time, stable throughput, peak GPU memory, and evidence paths.
- COCO records must contain text-to-image, image-to-text, and their equal-direction mean.
  GQA Balanced and CLEVR records must contain their independent answer-retrieval results.
- Do not combine a smoke metric, partial-test metric, old mixed-data metric, or another
  project result with the current full single-dataset comparison.
- Keep the all-model horizontal comparison limited to the frozen reference, locked LoRA
  baselines, one final result from the locked recurrent architecture, and any single
  historical reference row explicitly requested by the user. Mark unavailable datasets
  as `N/A`. Record controls, pass outputs, slot/history ablations, and other detailed
  variants only in their corresponding experiment sections unless the user explicitly
  promotes another comparison row.
