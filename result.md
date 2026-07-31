# Experiment Results

This is the central comparison record for this repository. Raw `run_manifest.json`,
`training_result.json`, and `report.json` files remain the source of truth. `N/A` means
that a field was not measured; it must never be guessed.

The fixed fields follow the smallest useful intersection of the official
[MLflow run model](https://mlflow.org/docs/latest/ml/tracking/) and
[MLflow Tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api/):
experiment identity, code version, input data, model and parameters, metrics, runtime,
and output files. Hardware and resource use are also required because they determine
whether throughput comparisons are meaningful.

## Current split contract

| Dataset | Train rows | Train images | Validation rows | Validation images | Test rows | Test images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO | 566,747 | 113,287 | 25,010 | 5,000 | 25,010 | 5,000 |
| GQA Balanced | 943,000 | 72,140 | 132,062 | 10,234 | 12,578 | 398 |
| CLEVR | 699,989 | 70,000 | 74,991 | 7,500 | 75,000 | 7,500 |

Training and test counts are sample rows, not image counts. Validation is not used in
the current experiments.

## Formal comparison status

| Experiment | COCO | GQA Balanced | CLEVR | Comparable final metrics |
| --- | --- | --- | --- | --- |
| Frozen original Qwen3-VL-Embedding-2B | Pending | Pending | Pending | No |
| Independent LoRA baseline | Passed | Running | Pending | COCO only |
| Recurrent latent-slot model | Pending | Pending | Pending | No |

The frozen row cannot reuse the old 200-row smoke or an older mixed-data report. It must
be evaluated on the exact test splits above.

### Scheduled frozen evaluation

- Status: waiting for the current six-experiment queue; no metrics exist yet.
- Code: commit `fe4e1794acda4ed79a504d8b4f0238d9147e35ee` in isolated worktree
  `/home/mnt/liyiwei/loopedTransformer_worktrees/frozen_eval_fe4e179`.
- Runtime: 8 × V100, FP16, scaled dot-product attention, per-device batch 32,
  4 workers, zero trainable parameters, no validation.
- Dataset order: COCO → GQA Balanced → CLEVR, using the split contract above.
- Tmux: `frozen_base_full_test_after_six_fe4e179_v2_20260731`.
- Output:
  `/home/mnt/liyiwei/outputs/frozen_base_full_test_fe4e179_20260731`.
- Log:
  `/home/mnt/liyiwei/outputs/frozen_base_full_test_fe4e179_v2_20260731.tmux.log`.

## EXP-001 — COCO independent LoRA baseline

- Status/date: passed, 2026-07-30; exact start/end timestamps were not recorded (`N/A`).
- Objective: establish the unmodified-architecture LoRA retrieval baseline on full COCO.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `9930a8dcad3f764a80d31870773846266c15d2b2`.
- Data: 566,747 train rows; 25,010 held-out test rows; no validation use. Train
  manifest SHA-256 `555211fd08280e4e9ab72f040d64b002c1e2aa4b72f6cb6b42427adabb381ff8`;
  test manifest SHA-256
  `a36d9000f665bbaf6bc7d7e472f38e069bb201c0cf6770c0784b498e7193ec89`.
- Base model: `Qwen3-VL-Embedding-2B/base_original`;
  SHA-256 `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`
  before and after.
- Trainable parameters: 31,195,136 LoRA parameters; rank 32, alpha 32, dropout 0;
  all 28 decoder layers; `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`,
  and `gate_proj`.
- Optimization: seed 42, 1 epoch, 2,214 optimizer steps, learning rate 5e-5,
  weight decay 0.01, temperature 0.02, warmup ratio 0.02.
- Runtime: FP16, scaled dot-product attention, gradient checkpointing enabled,
  per-device batch 32, contrastive and optimizer global batch 256, 4 data workers.
- Checkpoints: every 100 steps, at most 4 retained; final adapter SHA-256
  `d8cd3e611c130a27f3c4e52949e33b64cff83f3459c87b0687e6be1b9e264646`.
- Performance: train 11,700.65 seconds; final-step throughput 48.20 samples/second;
  peak allocated memory 8.67 GiB per rank; test 300.53 seconds.
- Output:
  `/home/mnt/liyiwei/outputs/six_full_train_test_9930a8d_20260730/baseline/coco`.
- Evidence: tmux `six_full_train_test_9930a8d_20260730`; queue command is stored
  in the root `status.json` and the exact train command is stored in
  `train/run_manifest.json`; log
  `/home/mnt/liyiwei/outputs/six_full_train_test_9930a8d_20260730.tmux.log`.

### Test metrics (%)

| Direction | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text→image | 72.0247 | 62.0832 | 16.8581 | 9.0672 | 4.7439 | 62.0832 | 84.2903 | 90.6717 | 94.8780 | 72.0247 | 76.2017 |
| Image→text | 64.0410 | 78.4600 | 57.8000 | 35.8980 | 20.5160 | 15.6867 | 57.7787 | 71.7680 | 82.0300 | 85.0017 | 70.5028 |
| Equal-direction mean | 68.0328 | 70.2716 | 37.3290 | 22.4826 | 12.6300 | 38.8849 | 71.0345 | 81.2199 | 88.4540 | 78.5132 | 73.3522 |

Coverage was 100%. Similarity was the dot product of L2-normalized embeddings.

## Canonical performance-selection smokes

These runs select safe execution settings. Their short-run losses are not model-quality
results and must not be compared with full-test metrics.

| Experiment | Commit | Data | Per-device batch | Contrastive / optimizer batch | Checkpointing | Workers | Final throughput | Peak allocated memory | Status |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Baseline batch sweep | `52d2159` | COCO | 16 | 128 / 128 | on | 4 | 44.51 samples/s | 6.49 GiB | passed |
| Baseline batch sweep | `52d2159` | COCO | 24 | 192 / 192 | on | 4 | 48.13 samples/s | 7.50 GiB | passed |
| Baseline selected | `e467751` | COCO | 32 | 256 / 256 | on | 4 | 50.04 samples/s | 8.53 GiB | passed |
| Baseline selected | `e467751` | GQA Balanced | 32 | 256 / 256 | on | 4 | 25.94 samples/s | 11.88 GiB | passed |
| Baseline selected | `e467751` | CLEVR | 32 | 256 / 256 | on | 4 | 90.50 samples/s | 6.48 GiB | passed |
| Recurrent selected | `2db8533` | COCO | 8 | 64 / 512 | off | 4 | 46.68 samples/s | 14.32 GiB | passed |
| Recurrent selected | `0393bef` | GQA Balanced | 8 | 64 / 512 | on | 4 | 28.47 samples/s | 7.17 GiB | passed |
| Recurrent selected | `0393bef` | CLEVR | 8 | 64 / 512 | off | 4 | 59.62 samples/s | 9.36 GiB | passed |

The baseline batch-32 checkpoint-off variants for COCO and GQA failed from insufficient
memory and are not selected. Eight workers did not improve the selected batch-32 runs.

## Required record for every new experiment

1. Identity: unique ID, objective, terminal status, start/end time, route, node, exact code
   commit.
2. Data: dataset, split, sample-row count, unique-image count, and manifest checksum.
3. Model: base checkpoint checksum, architecture or loop passes, trainable scope and count,
   and final checkpoint or adapter checksum.
4. Parameters: seed, precision, attention implementation, GPU count, physical and global
   batch sizes, optimizer, learning rate, schedule, epochs/steps, and checkpoint policy.
5. Evaluation: query and candidate gallery definition, test-only policy, metric cutoffs,
   and all required metrics.
6. Efficiency: wall time, stable throughput, peak GPU memory, and noteworthy failures.
7. Evidence: tmux name, command, log path, output path, and comparability limitations.
