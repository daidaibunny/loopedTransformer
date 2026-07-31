# Deleted files

## 2026-07-30

The following remote smoke-test checkpoints were deleted from
`/mnt/afs/liyiwei/loopedTransformer/outputs` before migration to the 8×V100
environment:

- `recovery_batch4plus_fresh_direct_20260730/training_batch4/checkpoints/stage1_step000003.pt`
- `recovery_batch4plus_fresh_direct_20260730/training_batch8/checkpoints/stage1_step000003.pt`
- `recurrent_train_stage1_smoke1gpu_v2_20260729/checkpoints/stage1_step000001.pt`
- `recurrent_train_stage2_smoke1gpu_20260729/checkpoints/stage2_step000001.pt`
- `smoke2gpu_rls_k4_r4_seed42_20260729/checkpoints/stage1_step000001.pt`
- `smoke2gpu_rls_k4_r4_seed42_20260729/checkpoints/stage2_step000001.pt`

Reason: these files only preserve one- or three-step smoke-test optimizer states,
are incompatible with the new 8×V100 execution profile, and account for
21,896,216,430 bytes of unnecessary cross-cluster transfer. Their accompanying
run manifests, metrics, gradient audits, and status files remain preserved.

Recovery: rerun the corresponding smoke test from its preserved manifest and
the unchanged Qwen3-VL-Embedding-2B base checkpoint.

## 2026-07-31

- `src/looped_vl/models/lora.py`

Reason: this custom LoRA implementation was incorrectly coupled to the recurrent model.
The formal recurrent experiment must keep the entire Qwen backbone frozen and train only
the recurrent slots, connector, training-only warm-start head, EOS delta, and
EOS-conditioned slot attention pooling plus zero-gated residual fusion.

Recovery: the removed implementation remains available in Git history. The independent
LoRA baseline continues to use its separate PEFT implementation under
`src/looped_vl/baseline/`.
