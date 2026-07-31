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
| Frozen original Qwen3-VL-Embedding-2B | Passed | Passed | Passed | All three datasets |
| Independent LoRA baseline | Passed | Passed | Interrupted | COCO and GQA |
| Recurrent latent-slot model | Pending | Pending | Pending | No |

The first CLEVR LoRA attempt was interrupted by the 2026-07-31 experiment reorder.
Its latest resumable state is step 100 (25,600 processed train rows); it has no final
adapter or test metrics and is not a completed result.

## Unified primary comparison metrics (%)

Each model contributes exactly one primary row per dataset. COCO is a bidirectional
image-text retrieval task, so its primary row is the arithmetic mean of text-to-image
and image-to-text metrics. The two direction-specific rows remain diagnostic details in
the corresponding experiment section; they are not three separate COCO tests. GQA
Balanced and CLEVR are question-to-answer retrieval tasks and therefore each have one
direction.

| Model | Dataset | Primary aggregation | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen original Qwen3-VL-Embedding-2B | COCO | Equal-direction mean | 61.2443 | 64.0153 | 33.9117 | 20.6115 | 11.7428 | 34.2366 | 64.9735 | 75.4542 | 83.6250 | 73.0671 | 66.9576 |
| Independent LoRA baseline | COCO | Equal-direction mean | 68.0328 | 70.2716 | 37.3290 | 22.4826 | 12.6300 | 38.8849 | 71.0345 | 81.2199 | 88.4540 | 78.5132 | 73.3522 |
| Frozen original Qwen3-VL-Embedding-2B | GQA Balanced | Answer retrieval | 52.1141 | 36.2697 | 14.4077 | 8.5292 | 4.6554 | 36.2697 | 72.0385 | 85.2918 | 93.1070 | 52.1141 | 59.5010 |
| Independent LoRA baseline | GQA Balanced | Answer retrieval | 74.9712 | 62.9035 | 17.9043 | 9.3473 | 4.8056 | 62.9035 | 89.5214 | 93.4727 | 96.1123 | 74.9712 | 79.3467 |
| Frozen original Qwen3-VL-Embedding-2B | CLEVR | Answer retrieval | 84.9359 | 73.1707 | 19.8499 | 9.9993 | 5.0000 | 73.1707 | 99.2493 | 99.9933 | 100.0000 | 84.9359 | 88.7750 |

Frozen-versus-LoRA rows are directly comparable within COCO and within GQA Balanced:
they use the same full test manifest checksum, query/candidate gallery, embedding
normalization, similarity function, and metric implementation. The current COCO full
test is 25,010 caption sample rows over 5,000 images; it is not a 25,010-image test.
CLEVR does not yet have a completed LoRA result. Values from different datasets measure
different retrieval tasks and must not be compared as if they shared one candidate
gallery. Runtime is also not a strict frozen-versus-LoRA comparison because frozen
evaluation used visual-length bucketing while the earlier LoRA evaluation did not.

## EXP-000A/B/C — Frozen original Qwen3-VL-Embedding-2B

- Status/date: all three passed, 2026-07-31. The serial full-test queue ran from
  2026-07-31T03:42:04Z to 2026-07-31T03:59:28Z; exact per-dataset start timestamps were
  not separately recorded.
- Objective: measure the untouched Qwen3-VL-Embedding-2B checkpoint on each full
  single-dataset test split before any LoRA or recurrent training.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `2877e70ad5394aa060b419823673fdcd02bad6d1`.
- Model: `Qwen3-VL-Embedding-2B/base_original`, zero trainable parameters, no adapter
  and no training checkpoint. Base-model SHA-256 was
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` before and
  after every test.
- Runtime: FP16, scaled dot-product attention, per-device batch 32, 4 data workers,
  8 independent encoding ranks, CPU control collectives, and baseline-only visual-length
  bucketing with at most 3 buckets and at least 8 items per bucket. Seed, optimizer,
  learning rate, epochs, and checkpoint policy are N/A because this is deterministic
  frozen inference. Validation was not used.
- Protocol: dot product of L2-normalized embeddings; P/R cutoffs 1/5/10/20 and nDCG
  cutoff 10. COCO uses its held-out image/caption galleries in both directions. GQA and
  CLEVR retrieve normalized answers observed in the corresponding training split.
- Evidence: tmux `frozen_qwen2b_three_2877e70_v2_20260731`; log
  `/home/mnt/liyiwei/outputs/frozen_qwen2b_three_2877e70_v2_20260731.tmux.log`;
  queue and reports under
  `/home/mnt/liyiwei/outputs/frozen_qwen2b_three_2877e70_v2_20260731`.

| Dataset | Test rows | Test images | Test manifest SHA-256 | Answer gallery | Runtime | Test rows/s | Coverage | Peak GPU memory |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| COCO | 25,010 | 5,000 | `a36d9000f665bbaf6bc7d7e472f38e069bb201c0cf6770c0784b498e7193ec89` | N/A | 289.93 s | 86.26 | 100.0000% | N/A |
| GQA Balanced | 12,578 | 398 | `ba1442bb782bb4627efc081111009e833dbbe6a5451a9f0264075cf662318b2b` | 1,833 | 202.83 s | 62.01 | 99.9841% | N/A |
| CLEVR | 75,000 | 7,500 | `d6d46dde537152465bb479684efd083ca6c6975d8a4d2fa3e3a1f1776395d094` | 28 | 416.81 s | 179.94 | 100.0000% | N/A |

Peak GPU memory is N/A because this version reset the PyTorch peak counter but did not
write its value into `report.json`; observed `nvidia-smi` snapshots are not substituted
for an exact peak.

### COCO frozen test metrics (%)

| Direction | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text→image | 64.5980 | 53.5906 | 15.5354 | 8.5610 | 4.5836 | 53.5906 | 77.6769 | 85.6098 | 91.6713 | 64.5980 | 69.1638 |
| Image→text | 57.8905 | 74.4400 | 52.2880 | 32.6620 | 18.9020 | 14.8827 | 52.2700 | 65.2987 | 75.5787 | 81.5361 | 64.7513 |
| Equal-direction mean | 61.2443 | 64.0153 | 33.9117 | 20.6115 | 11.7428 | 34.2366 | 64.9735 | 75.4542 | 83.6250 | 73.0671 | 66.9576 |

### GQA Balanced and CLEVR frozen test metrics (%)

| Dataset | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GQA Balanced | 52.1141 | 36.2697 | 14.4077 | 8.5292 | 4.6554 | 36.2697 | 72.0385 | 85.2918 | 93.1070 | 52.1141 | 59.5010 |
| CLEVR | 84.9359 | 73.1707 | 19.8499 | 9.9993 | 5.0000 | 73.1707 | 99.2493 | 99.9933 | 100.0000 | 84.9359 | 88.7750 |

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

## EXP-002 — GQA Balanced independent LoRA baseline

- Status/date: passed, 2026-07-31; exact start/end timestamps were not recorded (`N/A`).
- Objective: establish the unmodified-architecture LoRA answer-retrieval baseline on
  full GQA Balanced.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `9930a8dcad3f764a80d31870773846266c15d2b2`.
- Data: 943,000 train rows; 12,578 held-out test rows; no validation use. Train
  manifest SHA-256 `2022a835621ea4c072e1e09c5412b78d3322fdd1ac658485ac947859fb20abdb`;
  test manifest SHA-256
  `ba1442bb782bb4627efc081111009e833dbbe6a5451a9f0264075cf662318b2b`.
- Base model: `Qwen3-VL-Embedding-2B/base_original`;
  SHA-256 `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`
  before and after.
- Trainable parameters: 31,195,136 LoRA parameters; rank 32, alpha 32, dropout 0;
  all 28 decoder layers; `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`,
  and `gate_proj`.
- Optimization: seed 42, 1 epoch, 3,684 optimizer steps, learning rate 5e-5,
  weight decay 0.01, temperature 0.02, warmup ratio 0.02.
- Runtime: FP16, scaled dot-product attention, gradient checkpointing enabled,
  per-device batch 32, contrastive and optimizer global batch 256, 4 data workers.
- Checkpoints: every 100 steps, at most 4 retained; final adapter SHA-256
  `8434c82b66963c53db625643fce374d00d1e1fab549a2668c9feaff54ae0712d`.
- Performance: train 35,357.95 seconds; whole-run average 26.67 samples/second;
  peak allocated memory 19.17 GiB per rank; test 215.91 seconds.
- Output:
  `/home/mnt/liyiwei/outputs/six_full_train_test_9930a8d_20260730/baseline/gqa_balanced`.
- Evidence: tmux `six_full_train_test_9930a8d_20260730`; exact commands are stored
  in the queue `status.json` and train `run_manifest.json`; log
  `/home/mnt/liyiwei/outputs/six_full_train_test_9930a8d_20260730.tmux.log`.

### Test metrics (%)

| mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 74.9712 | 62.9035 | 17.9043 | 9.3473 | 4.8056 | 62.9035 | 89.5214 | 93.4727 | 96.1123 | 74.9712 | 79.3467 |

The answer gallery contained 1,833 normalized answers observed in training. Coverage
was 99.9841%; answer accuracy equals P@1.

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

## All-model horizontal comparison

This final table is the complete at-a-glance comparison matrix. It contains one primary
row for every model and dataset combination, including combinations that are interrupted
or still pending. Only full held-out test results appear as numbers; smoke losses,
partial-test metrics, and values from other candidate galleries are excluded.

For COCO, each metric is the equal-direction mean of text-to-image and image-to-text.
For GQA Balanced and CLEVR, each metric is answer retrieval. Results are directly
comparable only between rows from the same dataset. The recurrent parameter count is the
current training-time count on `main`; its training-only warm-start embedding head can be
discarded for inference.

| Dataset | Model | Status | Latent slots K | Loop passes R | Added trainable parameters | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO | Frozen original Qwen3-VL-Embedding-2B | Passed | 0 | 1 | 0 | 61.2443 | 64.0153 | 33.9117 | 20.6115 | 11.7428 | 34.2366 | 64.9735 | 75.4542 | 83.6250 | 73.0671 | 66.9576 |
| COCO | Independent LoRA baseline | Passed | 0 | 1 | 31,195,136 | 68.0328 | 70.2716 | 37.3290 | 22.4826 | 12.6300 | 38.8849 | 71.0345 | 81.2199 | 88.4540 | 78.5132 | 73.3522 |
| COCO | Recurrent latent-slot model | Pending | 4 | 4 | 17,342,977 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| GQA Balanced | Frozen original Qwen3-VL-Embedding-2B | Passed | 0 | 1 | 0 | 52.1141 | 36.2697 | 14.4077 | 8.5292 | 4.6554 | 36.2697 | 72.0385 | 85.2918 | 93.1070 | 52.1141 | 59.5010 |
| GQA Balanced | Independent LoRA baseline | Passed | 0 | 1 | 31,195,136 | 74.9712 | 62.9035 | 17.9043 | 9.3473 | 4.8056 | 62.9035 | 89.5214 | 93.4727 | 96.1123 | 74.9712 | 79.3467 |
| GQA Balanced | Recurrent latent-slot model | Pending | 4 | 4 | 17,342,977 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CLEVR | Frozen original Qwen3-VL-Embedding-2B | Passed | 0 | 1 | 0 | 84.9359 | 73.1707 | 19.8499 | 9.9993 | 5.0000 | 73.1707 | 99.2493 | 99.9933 | 100.0000 | 84.9359 | 88.7750 |
| CLEVR | Independent LoRA baseline | Interrupted | 0 | 1 | 31,195,136 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CLEVR | Recurrent latent-slot model | Pending | 4 | 4 | 17,342,977 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
