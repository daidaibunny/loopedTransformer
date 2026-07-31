# Looped VL

## Scope

- This repository is exclusively for Recurrent Latent-Slot Qwen3-VL-Embedding work.
- Do not add, import, or run PoLar code in this repository.
- The formal recurrent model is pure recurrent and must contain no LoRA modules or
  parameters. LoRA belongs only to the independent comparison code under
  `src/looped_vl/baseline/`.
- Implement the attached v1.0 specification exactly, keeping forward activation updates
  separate from the trainable-parameter allowlist for each stage.
- Keep the original Qwen3-VL checkpoint immutable. Save learned parameters and checkpoints
  only under experiment-specific output directories.
- Always call the single-query attention from the final valid token over latent slots
  `EOS-conditioned slot attention pooling` (`EOS 条件化槽位注意力池化`). Do not shorten
  it to ordinary pooling. Call the subsequent gated residual addition
  `zero-gated residual fusion`.

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

## Checkpoint policy

- Baseline and recurrent full training save an exact resumable checkpoint every 100
  optimizer steps and at the end of the epoch.
- Each experiment dynamically retains only its newest resumable checkpoint. Saving a new
  checkpoint must remove the previous one only after the new file exists and the
  latest-checkpoint pointer has been atomically published.
- Keep the final adapter or final recurrent learned weights separately from the rolling
  resumable checkpoint. Evaluation reads those final learned weights, not an intermediate
  optimizer checkpoint.
- Reject recurrent checkpoints containing LoRA parameters or the superseded
  `single_stage_warm_start_v1` protocol.
- Every recurrent `run_manifest.json`, `training_result.json`, checkpoint metadata, and
  `report.json` must declare architecture `recurrent_latent_slot_qwen3vl_no_lora_v1`,
  protocol `pure_recurrent_single_stage_v1`, `backbone_frozen: true`, and
  `lora_enabled: false`. Reject missing or conflicting identities.

## Verification

- Add tests before implementation changes.
- Verify all dataset backends, batching, processor inputs, embedding shapes, finite values,
  unit norms, and model checkpoint hash stability.
- Enforce every structural, trainability, determinism, and numerical acceptance condition
  from sections 27–29 of the v1.0 implementation specification.

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
