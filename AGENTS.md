# Looped VL

## Scope

- This repository is exclusively for Recurrent Latent-Slot Qwen3-VL-Embedding work.
- Do not add, import, or run PoLar code in this repository.
- Implement the attached v1.0 specification exactly, keeping forward activation updates
  separate from the trainable-parameter allowlist for each stage.
- Keep the original Qwen3-VL checkpoint immutable. Save learned parameters and checkpoints
  only under experiment-specific output directories.

## Remote execution

- Use only SSH alias `gyy1` and its two assigned physical GPUs, 0 and 1.
- Use the official Git clone at `/mnt/afs/liyiwei/loopedTransformer` as the remote code root.
- Use `/mnt/afs/likangle/reserach/LOCUS-MLLM/envs/LOCUS/bin/python`.
- Run long jobs in uniquely named detached tmux sessions with separate logs.
- Before a two-GPU launch, inspect both GPUs once and submit immediately when neither GPU
  has an active compute process. Do not impose a continuous idle waiting period.
- Use per-device training batch size 8 on two GPUs, with gradient accumulation 32 and
  effective global batch size 512. Do not use batch size 16; it exhausted an 80 GB GPU
  under the official variable-resolution preprocessing.
- After launch, poll both ranks, GPU use, progress logs, and checkpoints for several minutes.

## Data

- Dataset root:
  `/mnt/afs/liyiwei/datasets/looped_vl_mix_v1_train100000_val10000_test10000`.
- Parent full dataset: `/mnt/afs/liyiwei/datasets/looped_vl_mix_v1`.
- Define split sizes by sample rows: 100,000 train, 10,000 validation, and 10,000 test.
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
