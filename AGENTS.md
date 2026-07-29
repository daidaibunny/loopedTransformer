# Looped VL

## Scope

- Build and validate data loading for the 100,000-row train subset.
- Use the frozen local Qwen3-VL-Embedding-2B checkpoint for smoke tests only.
- Do not train, create an optimizer, call backward, or modify model weights.

## Remote execution

- Use only SSH alias `gyy1` and physical GPU 1 when it is idle.
- Use `/mnt/afs/liyiwei/looped_vl` as the remote code root.
- Use `/mnt/afs/likangle/reserach/LOCUS-MLLM/envs/LOCUS/bin/python`.
- Run long jobs in uniquely named detached tmux sessions with separate logs.

## Data

- Dataset root:
  `/mnt/afs/liyiwei/datasets/looped_vl_mix_v1_train100000_val25000_test25000`.
- Parent full dataset: `/mnt/afs/liyiwei/datasets/looped_vl_mix_v1`.
- Resolve COCO and CLEVR with `image_path`.
- Resolve GQA Balanced from its materialized image cache by `image_id`.
- Preserve the exact source ratio in train, validation, and test.
- Never use test samples for tuning or checkpoint selection.

## Verification

- Add tests before implementation changes.
- Verify all dataset backends, batching, processor inputs, embedding shapes, finite values,
  unit norms, and model checkpoint hash stability.

## Required evaluation metrics

- Every evaluation report must pass `looped_vl.metrics.validate_evaluation_report`.
- Report the weighted Mix result and complete per-dataset results for COCO, GQA Balanced,
  and CLEVR. COCO must also report text-to-image and image-to-text separately.
- Required metrics are mAP, P@1/5/10/20, R@1/5/10/20, MRR, and nDCG@10.
- Use percentage values from 0 to 100. Aggregate COCO directions equally, then aggregate
  datasets with fixed weights COCO:GQA Balanced:CLEVR = 50:35:15.
