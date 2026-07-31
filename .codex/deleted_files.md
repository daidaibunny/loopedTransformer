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

### Approved remote checkpoint cleanup

The following disposable checkpoints and their `latest_checkpoint.json` pointers were
deleted from the `8XV100` route after this deletion record was pushed:

- `/home/mnt/liyiwei/outputs/final_recurrent_smoke_1c30e73_20260730/checkpoints/step000003.pt`
  - Size: 208,378,826 bytes
  - SHA-256: `91c8eb2d79406695e8ae9e0633d708108ee92539925a5960e7809d2a4f8f163a`
  - Protocol: superseded `single_stage_warm_start_v1`
- `/home/mnt/liyiwei/outputs/rls_v2_gqa_train_eval_b24_bc9f232_v3_20260731/train/checkpoints/step000002.pt`
  - Size: 101,303,818 bytes
  - SHA-256: `1adbf8eedf556b4dab30aa888c67d9fce51e3f71940a19d862afcb2decb1f027`
  - Protocol: two-step `pure_recurrent_full_objective_v2` runtime smoke

Reason: both files preserve optimizer state for completed disposable smoke runs. The
newer smoke's parameters, throughput, memory, checkpoint checksum, and evidence paths are
already recorded in `result.md`. Neither file is eligible for a formal quality result or
needed by a queued experiment. Their manifests, metrics, gradient audits, status files,
and evaluation reports remain preserved. Including both pointer files, the cleanup
released 309,683,217 bytes.

Recovery: rerun the preserved smoke command from its run manifest against the immutable
Qwen3-VL-Embedding-2B base checkpoint. The three completed full LoRA experiments retain
their final resumable checkpoints and final adapters.
