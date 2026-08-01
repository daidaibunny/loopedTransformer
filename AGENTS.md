# Looped VL

## Scope

- This repository is exclusively for Recurrent Latent-Slot Qwen3-VL-Embedding work.
- Do not add, import, or run PoLar code in this repository.
- The formal recurrent model is pure recurrent and must contain no LoRA modules or
  parameters. LoRA belongs only to the independent comparison code under
  `src/looped_vl/baseline/`.
- The locked recurrent architecture is
  `direct_eos_layerscale_mid_decoder_recurrence_no_lora_v5`. The recurrent path contains
  no LoRA, recurrent connector, trainable auxiliary projection, EOS delta, or late fusion.
- Use one training stage and one full-data epoch. All trainable recurrent parameters are
  active from the first optimizer step.
- Default to 8 active slots from the shared seed-42 16-slot initialization bank and 4
  total recurrent passes. Extra passes update slots only; EOS is fixed after Pass 1.
- Slot-count ablations may use K=12, K=16, K=32, or K=64 without changing the formal
  K=8 default. Every K=8/12/16/32/64 comparison must use prefixes of the same seed-42
  64-slot bank, `artifacts/master_slot_init_seed42_kmax64.pt`; never overwrite the
  existing 16-slot bank.
- Pass 1 runs decoder Layers 13–20 on the full sequence and captures detached prefix
  Key/Value evidence. Passes 2–4 update only the latent slots. Their recurrent step size
  is independent of the number of passes and defaults to 1.0.
- The extra passes share one trainable 2048-channel scale vector per repeated decoder
  layer. Together with the latent slots, these are the only trainable recurrent
  parameters. Enforce a hard limit of 5,000,000 trainable parameters before optimization.
- Keep the original Qwen3-VL checkpoint immutable. Save learned parameters and checkpoints
  only under experiment-specific output directories.
- Always call the single-query attention from the final valid token over latent slots
  `EOS-conditioned slot attention pooling` (`EOS 条件化槽位注意力池化`). Do not shorten
  it to ordinary pooling. The v5 final retrieval readout is the direct Layer-28 EOS after
  the original final RMSNorm and L2 normalization; it has no residual fusion.
- The parameter-free training auxiliary readout uses the fixed layer-20 EOS from Pass 1
  as a query over each pass's slots. RMS-normalize EOS and slots without learned weights,
  scale their dot products by the square root of 2048, softmax across slots, sum the raw
  slots with those weights, apply parameter-free RMS normalization, and L2-normalize the
  resulting 2048-dimensional vector. It must never use mean pooling in a formal run.

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
- Reject recurrent checkpoints containing LoRA parameters, recurrent-connector
  parameters, or any superseded training protocol.
- Every recurrent `run_manifest.json`, `training_result.json`, checkpoint metadata, and
  `report.json` must declare architecture
  `direct_eos_layerscale_mid_decoder_recurrence_no_lora_v5`, protocol
  `single_stage_progressive_slot_attention_no_lora_v5`, `backbone_frozen: true`,
  `lora_enabled: false`, and `formal_training_stages: 1`. Reject missing or
  conflicting identities.
- Recurrent training runs for exactly one epoch with final InfoNCE weight 1.0 at every
  optimizer step. The final-pass parameter-free slot-attention InfoNCE has weight 0.1,
  and the progressive non-degradation loss has weight 0.1. Slot diversity is logged but
  has weight 0.0. All trainable groups use the same learning-rate schedule from step one.

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
- Every recurrent report must retain the complete required metric set for Pass 1 through
  Pass 4. It must also map Pass 1/2/3/4 to 0/1/2/3 completed recurrent updates and
  summarize the primary mAP change from the previous pass and from Pass 1. For COCO, this
  summary uses the equal-direction mean; direction-specific pass metrics remain required.
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
