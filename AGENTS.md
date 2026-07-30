# Looped VL

## Scope

- This repository is exclusively for Recurrent Latent-Slot Qwen3-VL-Embedding work.
- Do not add, import, or run PoLar code in this repository.
- Implement the attached v1.0 specification exactly, keeping forward activation updates
  separate from the trainable-parameter allowlist for each stage.
- Keep the original Qwen3-VL checkpoint immutable. Save learned parameters and checkpoints
  only under experiment-specific output directories.

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
  using each dataset's full official split rather than the 50:35:15 mixture.
- Keep the 50:35:15 mixture only for later mixed-dataset experiments and aggregate reports.
- For the later mixed subset only, define split sizes by sample rows: 100,000 train,
  10,000 validation, and 10,000 test.
- Resolve COCO and CLEVR with `image_path`.
- Resolve GQA Balanced from its materialized image cache by `image_id`.
- Preserve the exact source ratio in train, validation, and test.
- Never use test samples for tuning or checkpoint selection.

## Verification

- Add tests before implementation changes.
- Verify all dataset backends, batching, processor inputs, embedding shapes, finite values,
  unit norms, and model checkpoint hash stability.
- Enforce every structural, trainability, determinism, and numerical acceptance condition
  from sections 27–29 of the v1.0 implementation specification.

## Required evaluation metrics

- Every evaluation report must pass `looped_vl.metrics.validate_evaluation_report`.
- Report the weighted Mix result and complete per-dataset results for COCO, GQA Balanced,
  and CLEVR. COCO must also report text-to-image and image-to-text separately.
- Required metrics are mAP, P@1/5/10/20, R@1/5/10/20, MRR, and nDCG@10.
- Use percentage values from 0 to 100. Aggregate COCO directions equally, then aggregate
  datasets with fixed weights COCO:GQA Balanced:CLEVR = 50:35:15.
