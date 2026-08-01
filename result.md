# Experiment Results

This is the central comparison record for this repository. Raw `run_manifest.json`,
`training_result.json`, and `report.json` files remain the source of truth. `N/A` means
that a field was not measured; it must never be guessed.

The pretrained model is Qwen3-VL-Embedding-2B at
`/home/mnt/liyiwei/models/Qwen3-VL-Embedding-2B/base_original`. In the remainder of this
file, `backbone` refers to that exact immutable pretrained checkpoint.

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
| Frozen backbone | Passed | Passed | Passed | All three datasets |
| Backbone + LoRA | Passed | Passed | Passed | All three datasets |
| Backbone + LoRA, decoder layers 24–27 only | Passed | Running | Pending | COCO only |
| Superseded damped mid-decoder recurrent slots (no LoRA) | Passed | N/A | N/A | Historical COCO only |
| Query-only history recurrent Block (no LoRA) | Running | Pending | Pending | COCO R=1 fixed control |

The active definition is `query_only_history_recurrent_no_lora_v1`, with protocol
`single_stage_frozen_candidate_dynamic_exit_v1`. Candidate embeddings come only from
the eight immutable frozen-Qwen banks. Candidate Qwen therefore has zero training and
test forward calls. Query Qwen is also frozen and runs once, exposing decoder histories
from Layers 7, 14, 21, and 28 to a trainable 288-dimensional shared recurrent Block.
The default uses K=8 slots, at most R=4 shared updates, EOS-conditioned slot attention
pooling, zero-gated residual fusion, and a sample-dependent exit controller. It contains
no LoRA.

Every active run uses one stage, one full-data epoch, no validation, one rolling
checkpoint, per-device batch 32 on eight V100 GPUs, and a true contrastive global batch
of 256. The first completed control is EXP-005: COCO K=8, R=1, fixed exit. The remaining
R=4 fixed/dynamic, K=1/4, Layer-28-only-history, GQA Balanced, and CLEVR experiments are
still running or queued. EXP-004A/B/C/D use the superseded damped mid-decoder design and
remain below only as historical evidence; they are not the active architecture.

For every active recurrent test, retain frozen-Qwen Pass 0, recurrent Pass 1 through
Pass 4, dynamic hard exit, and dynamic soft exit, with every required metric and the
change from the in-run Pass 0. COCO uses its equal-direction mean for the concise
comparison while retaining text-to-image and image-to-text details.

## Current recurrent parameter accounting

The original Qwen3-VL-Embedding-2B backbone and every candidate bank remain frozen and
unchanged. The active K=8 query recurrent head adds:

| Component | Parameters | Retained for inference |
| --- | ---: | --- |
| Frozen-history projection and layer identities | 597,888 | Yes |
| Slot queries and slot initializer | 1,001,376 | Yes |
| Two-layer shared recurrent Block | 2,665,152 | Yes |
| EOS-conditioned readout and zero-gated residual fusion | 592,448 | Yes |
| Dynamic exit controller | 21,457 | Yes |
| **Total trainable addition** | **4,878,321** | **Yes** |

All active recurrent parameters are retained for inference; there is no training-only
head. The exact totals are 4,876,305 for K=1, 4,877,169 for K=4, and 4,878,321 for K=8.
The independent last-four-decoder-layer LoRA baseline trains 4,456,448 parameters, so
K=8 recurrent adds 421,873 more parameters, or 1.0947 times that LoRA parameter count.
The full 28-layer LoRA baseline trains 31,195,136 parameters. Parameter counts alone do
not establish quality; use the held-out metrics below.

## EXP-000A/B/C — Frozen backbone

- Status/date: all three passed, 2026-07-31. The serial full-test queue ran from
  2026-07-31T03:42:04Z to 2026-07-31T03:59:28Z; exact per-dataset start timestamps were
  not separately recorded.
- Objective: measure the untouched backbone on each full single-dataset test split
  before any LoRA or recurrent training.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `2877e70ad5394aa060b419823673fdcd02bad6d1`.
- Backbone: zero trainable parameters, no adapter, and no training checkpoint.
  Backbone SHA-256 was
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

## EXP-001 — COCO backbone + LoRA

- Status/date: passed, 2026-07-30; exact start/end timestamps were not recorded (`N/A`).
- Objective: establish the unmodified-architecture LoRA retrieval baseline on full COCO.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `9930a8dcad3f764a80d31870773846266c15d2b2`.
- Data: 566,747 train rows; 25,010 held-out test rows; no validation use. Train
  manifest SHA-256 `555211fd08280e4e9ab72f040d64b002c1e2aa4b72f6cb6b42427adabb381ff8`;
  test manifest SHA-256
  `a36d9000f665bbaf6bc7d7e472f38e069bb201c0cf6770c0784b498e7193ec89`.
- Backbone SHA-256:
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` before and
  after.
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

## EXP-002 — GQA Balanced backbone + LoRA

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
- Backbone SHA-256:
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` before and
  after.
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

## EXP-003 — CLEVR backbone + LoRA

- Status/date: passed, 2026-07-31. Training resumed from step 100 at
  2026-07-31T04:30:19Z, finished at 2026-07-31T06:42:28Z, and full-test evaluation
  finished at 2026-07-31T06:50:34Z.
- Objective: establish the unmodified-architecture LoRA answer-retrieval baseline on
  full CLEVR.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `9930a8dcad3f764a80d31870773846266c15d2b2`.
- Data: 699,989 train rows; 75,000 held-out test rows; no validation use. Train
  manifest SHA-256
  `f9d000f6a5258da38fa31f9f75b1ac6a65756039d4db15c5c3e8109aa770bf71`;
  test manifest SHA-256
  `d6d46dde537152465bb479684efd083ca6c6975d8a4d2fa3e3a1f1776395d094`.
- Backbone SHA-256:
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` before and
  after.
- Trainable parameters: 31,195,136 LoRA parameters; rank 32, alpha 32, dropout 0;
  all 28 decoder layers; `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`,
  and `gate_proj`.
- Optimization: seed 42, 1 epoch, 2,735 optimizer steps, learning rate 5e-5,
  weight decay 0.01, temperature 0.02, warmup ratio 0.02.
- Runtime: FP16, scaled dot-product attention, gradient checkpointing enabled,
  per-device batch 32, contrastive and optimizer global batch 256, 4 data workers.
- Resume/checkpoints: resumed from step 100 (25,600 processed rows), checkpointed
  every 100 steps, and retained at most 1 checkpoint during the resumed segment.
  The final adapter SHA-256 is
  `501464adf13ea00dcb937ec046ae9ed4d640e035117fa41c08fb804db7269786`.
- Performance: the resumed segment took 7,929.74 seconds; the final full batch at
  step 2,734 reached 91.27 samples/second; peak allocated memory was 6.54 GiB on
  rank 0. Full-test evaluation took 432.40 seconds (173.45 rows/second).
- Output:
  `/home/mnt/liyiwei/outputs/six_full_train_test_9930a8d_20260730/baseline/clevr`.
- Evidence: resume log
  `/home/mnt/liyiwei/outputs/baseline_clevr_resume_9930a8d_20260731.log`; exact
  commands are in `train/run_manifest.json` and
  `train/resume_manifest_step000100.json`.

The distributed sampler padded three rows so that all eight ranks received equal
work. Therefore `training_result.json` records 699,992 processed rows while the
dataset still contains exactly 699,989 unique train rows.

### Test metrics (%)

| mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99.2076 | 98.4640 | 19.9997 | 10.0000 | 5.0000 | 98.4640 | 99.9987 | 100.0000 | 100.0000 | 99.2076 | 99.4138 |

The answer gallery contained 28 normalized answers observed in training. Coverage
was 100%; answer accuracy equals P@1.

## COCO bidirectional diagnostic metrics

COCO is the only current dataset with two protocol-defined retrieval directions.
Text-to-image uses each caption as a query over the image gallery. Image-to-text uses
each image as a query over the caption gallery. The equal-direction mean remains the
single primary COCO value in the all-model table; the two directional rows below are
diagnostics, not additional independent tests.

| Model | Status | Direction | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen backbone | Passed | Text→image | 64.5980 | 53.5906 | 15.5354 | 8.5610 | 4.5836 | 53.5906 | 77.6769 | 85.6098 | 91.6713 | 64.5980 | 69.1638 |
| Frozen backbone | Passed | Image→text | 57.8905 | 74.4400 | 52.2880 | 32.6620 | 18.9020 | 14.8827 | 52.2700 | 65.2987 | 75.5787 | 81.5361 | 64.7513 |
| Frozen backbone | Passed | Equal-direction mean | 61.2443 | 64.0153 | 33.9117 | 20.6115 | 11.7428 | 34.2366 | 64.9735 | 75.4542 | 83.6250 | 73.0671 | 66.9576 |
| Backbone + LoRA | Passed | Text→image | 72.0247 | 62.0832 | 16.8581 | 9.0672 | 4.7439 | 62.0832 | 84.2903 | 90.6717 | 94.8780 | 72.0247 | 76.2017 |
| Backbone + LoRA | Passed | Image→text | 64.0410 | 78.4600 | 57.8000 | 35.8980 | 20.5160 | 15.6867 | 57.7787 | 71.7680 | 82.0300 | 85.0017 | 70.5028 |
| Backbone + LoRA | Passed | Equal-direction mean | 68.0328 | 70.2716 | 37.3290 | 22.4826 | 12.6300 | 38.8849 | 71.0345 | 81.2199 | 88.4540 | 78.5132 | 73.3522 |
| Recurrent latent slots, K=8, Pass 4 (EXP-004A) | Passed | Text→image | 66.5165 | 55.8257 | 15.8561 | 8.6653 | 4.6180 | 55.8257 | 79.2803 | 86.6533 | 92.3591 | 66.5165 | 70.9131 |
| Recurrent latent slots, K=8, Pass 4 (EXP-004A) | Passed | Image→text | 55.5491 | 70.2600 | 50.2560 | 31.7980 | 18.5670 | 14.0467 | 50.2387 | 63.5727 | 74.2387 | 78.4199 | 62.3308 |
| Recurrent latent slots, K=8, Pass 4 (EXP-004A) | Passed | Equal-direction mean | 61.0328 | 63.0428 | 33.0560 | 20.2317 | 11.5925 | 34.9362 | 64.7595 | 75.1130 | 83.2989 | 72.4682 | 66.6219 |
| Recurrent latent slots, K=12, Pass 4 (EXP-004B) | Passed | Text→image | 65.8472 | 55.2339 | 15.7385 | 8.5958 | 4.5934 | 55.2339 | 78.6925 | 85.9576 | 91.8673 | 65.8472 | 70.2152 |
| Recurrent latent slots, K=12, Pass 4 (EXP-004B) | Passed | Image→text | 55.9960 | 70.8400 | 50.5640 | 31.9500 | 18.6420 | 14.1620 | 50.5453 | 63.8740 | 74.5387 | 78.9134 | 62.7404 |
| Recurrent latent slots, K=12, Pass 4 (EXP-004B) | Passed | Equal-direction mean | 60.9216 | 63.0370 | 33.1513 | 20.2729 | 11.6177 | 34.6980 | 64.6189 | 74.9158 | 83.2030 | 72.3803 | 66.4778 |
| Recurrent latent slots, K=16, Pass 4 (EXP-004C) | Passed | Text→image | 66.1807 | 55.6257 | 15.8329 | 8.6202 | 4.6034 | 55.6257 | 79.1643 | 86.2015 | 92.0672 | 66.1807 | 70.5342 |
| Recurrent latent slots, K=16, Pass 4 (EXP-004C) | Passed | Image→text | 56.5751 | 71.2000 | 51.0680 | 32.2440 | 18.8330 | 14.2340 | 51.0507 | 64.4627 | 75.3013 | 79.3803 | 63.2988 |
| Recurrent latent slots, K=16, Pass 4 (EXP-004C) | Passed | Equal-direction mean | 61.3779 | 63.4129 | 33.4504 | 20.4321 | 11.7182 | 34.9299 | 65.1075 | 75.3321 | 83.6843 | 72.7805 | 66.9165 |
| Recurrent latent slots, K=32, Pass 4 (EXP-004D) | Passed | Text→image | 66.6458 | 56.0176 | 15.8880 | 8.6829 | 4.6140 | 56.0176 | 79.4402 | 86.8293 | 92.2791 | 66.6458 | 71.0604 |
| Recurrent latent slots, K=32, Pass 4 (EXP-004D) | Passed | Image→text | 56.0733 | 70.6000 | 50.7040 | 31.9080 | 18.7420 | 14.1140 | 50.6847 | 63.7900 | 74.9387 | 78.7059 | 62.6542 |
| Recurrent latent slots, K=32, Pass 4 (EXP-004D) | Passed | Equal-direction mean | 61.3595 | 63.3088 | 33.2960 | 20.2955 | 11.6780 | 35.0658 | 65.0624 | 75.3096 | 83.6089 | 72.6758 | 66.8573 |

GQA Balanced and CLEVR remain one-way image-question-to-answer retrieval tasks under the
current manifests. Reverse answer-to-question/image retrieval would require a new
candidate gallery and relevance definition, so it is not reported as a symmetric metric.

## EXP-SMOKE-004 — Superseded connector-model safety and GQA batch search

- Status/date: passed, 2026-07-31. Training smoke ran from 07:05:59Z until the
  batch-24 evaluation started at 07:08:07Z; batch 24 passed at 07:09:34Z. The
  independent batch-32 evaluation ran from 07:10:41Z to 07:12:14Z.
- Objective: historically verify the then-current pure-recurrent/no-LoRA training path, rolling
  one-checkpoint policy, Pass 1–4 export, CPU Gloo evaluation collectives, and the
  largest tested GQA evaluation batch. This is a runtime smoke, not a formal quality
  result.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: `main` commit `bc9f232772f72d823bb225bab50655a36e163739`,
  executed from detached worktree
  `/home/mnt/liyiwei/loopedTransformer_worktrees/recurrent_smoke_bc9f232`.
  The code passed 170 local tests before submission.
- Model: `recurrent_latent_slot_qwen3vl_no_lora_v1`, K=4 latent slots, R=4
  total passes, frozen backbone, 8,430,081 training parameters and 4,233,729
  inference parameters. Backbone SHA-256:
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`.
  This model used the now-removed recurrent connector, so its throughput and selected
  batch sizes must be revalidated before use by the current damped architecture.
- Training smoke: GQA Balanced train manifest, 1,024 consumed sample rows, exact
  unique-image count N/A, no validation, 2 optimizer steps, FP16, scaled dot-product
  attention, gradient checkpointing enabled, seed 42, AdamW, cosine schedule,
  per-device batch 8, contrastive global batch 64, gradient accumulation 8, optimizer
  global batch 512, and 4 data workers. Final InfoNCE weight was 1.0 in both steps.
- Training efficiency: 68.72 seconds as recorded by `training_result.json`; the
  second step reached 32.00 samples/second; exact peak allocated GPU memory was
  7.07 GiB. Both steps had finite loss and gradient norm and passed recurrence,
  slot-collapse, and pooling-collapse guards.
- Checkpoint: the smoke explicitly saved one final resumable checkpoint,
  `step000002.pt`, 96.61 MiB, SHA-256
  `1adbf8eedf556b4dab30aa888c67d9fce51e3f71940a19d862afcb2decb1f027`.
  It was subsequently deleted after the connector architecture was superseded; the
  manifest, metrics, gradient audits, and report remain preserved.
- Evaluation protocol: partial GQA test prefixes from the authoritative full test
  manifest SHA-256
  `ba1442bb782bb4627efc081111009e833dbbe6a5451a9f0264075cf662318b2b`;
  FP16, scaled dot-product attention, 8 Gloo ranks, 4 workers, and all four loop-pass
  embeddings exported. The smoke checkpoint had only two optimizer updates, so its
  partial-prefix retrieval metrics are intentionally excluded from the formal model
  comparison.

| Per-device batch | Test-prefix rows | Unique images | Encoded items | Encoding wall time | Encoding throughput | Exact peak GPU memory | Total evaluation time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 384 | 204 | 523 | 39.03 s | 13.40 items/s | 5.29 GiB | 72.27 s |
| 32 | 512 | 245 | 686 | 41.22 s | 16.64 items/s | 5.71 GiB | 74.64 s |

Both settings ran two full query batches per rank. Batch 32 improved measured encoding
throughput by 24.19% over batch 24 while remaining far below the 32 GiB device limit.
Therefore GQA recurrent evaluation uses per-device batch 32; recurrent training remains
per-device batch 8.

- Evidence: tmux
  `rls_v2_gqa_train_eval_b24_bc9f232_v3_20260731`, log and outputs under
  `/home/mnt/liyiwei/outputs/rls_v2_gqa_train_eval_b24_bc9f232_v3_20260731`;
  tmux `rls_v2_gqa_eval_b32_bc9f232_20260731`, log and outputs under
  `/home/mnt/liyiwei/outputs/rls_v2_gqa_eval_b32_bc9f232_20260731`.
- Submission notes: two earlier attempts stopped before model execution—one used the
  system Python without `qwen_vl_utils`, and one detected that the shared checkout had
  advanced to a newer `main` commit. The successful run used the established project
  environment and a commit-pinned worktree. Neither failed attempt produced a
  checkpoint or quality result.

## EXP-SMOKE-005 — Current damped recurrence with EOS-weighted auxiliary slots

- Status/date: passed, 2026-07-31; exact start/end timestamps were not separately
  recorded (`N/A`).
- Objective: validate the current no-LoRA, no-connector architecture, select safe
  eight-V100 train/evaluation batch sizes, verify one-checkpoint retention, and exercise
  the Pass 1–4 metric and recurrent-improvement report. This is a runtime smoke and
  partial-prefix test, not a formal model-quality result.
- Route/node/code: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB,
  `main` commit `6b82132f83304ebfeeba24715f77bef3a77ab99b`.
- Model: `damped_mid_decoder_latent_slot_recurrence_no_lora_v3`,
  `pure_recurrent_single_stage_eos_weighted_aux_v4`, K=8, R=4, fixed
  `alpha=1/4`, frozen backbone, no LoRA, no recurrent connector, and one shared
  256-dimensional EOS-conditioned weighted-slot auxiliary head. Training has
  2,641,921 parameters; inference retains 2,115,585. Backbone SHA-256 remained
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`.
- Data: GQA Balanced train has 943,000 rows and 72,140 images; the smoke consumed
  1,536 rows. The authoritative train manifest SHA-256 is
  `2022a835621ea4c072e1e09c5412b78d3322fdd1ac658485ac947859fb20abdb`.
  Validation was not used.
- Training: FP16, scaled dot-product attention, gradient checkpointing, seed 42,
  AdamW, learning rate 1e-5, cosine schedule, weight decay 0.01, per-device batch
  32, contrastive global batch 256, gradient accumulation 2, optimizer global batch
  512, 4 workers, and 3 optimizer steps. The formal schedule remains exactly one
  full epoch; only this smoke stopped after three steps.
- Training efficiency: 84.22 seconds total. After the first compilation/warm-up step,
  steps 2 and 3 reached 36.28 and 35.29 samples/second, averaging 35.79
  samples/second. Exact peak PyTorch allocated memory was 16.25 GiB. Every rank
  completed, losses and gradients were finite, and collapse/unused-recurrence guards
  stayed false.
- Checkpoint: one rolling file,
  `checkpoints/step000003.pt` (31,839,738 bytes), SHA-256
  `f15711c7886433ef2082f7ccee40c521016ff9f62b39662e358e2460e1b6d4b3`.
  No older checkpoint remains in this experiment.
- Evaluation: partial GQA test prefix with 2,048 query rows, 381 unique query images,
  and a 379-answer gallery. The authoritative full-test manifest SHA-256 is
  `ba1442bb782bb4627efc081111009e833dbbe6a5451a9f0264075cf662318b2b`.
  Eight Gloo ranks used FP16, scaled dot-product attention, per-device batch 128,
  and 4 workers. Encoding 2,427 query/gallery items took 84.13 seconds
  (28.85 items/second); total evaluation time was 118.37 seconds and exact peak
  allocated GPU memory was 10.86 GiB.

### Partial GQA metrics by recurrent pass (%)

| Pass | Extra recurrent updates | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 49.3414 | 31.6406 | 14.6484 | 9.0381 | 4.8340 | 31.6406 | 73.2422 | 90.3809 | 96.6797 | 49.3414 | 58.6854 |
| 2 | 1 | 50.7671 | 33.3496 | 14.9707 | 9.1162 | 4.8340 | 33.3496 | 74.8535 | 91.1621 | 96.6797 | 50.7671 | 60.0042 |
| 3 | 2 | 51.3037 | 33.9355 | 15.0195 | 9.1504 | 4.8364 | 33.9355 | 75.0977 | 91.5039 | 96.7285 | 51.3037 | 60.5163 |
| 4 | 3 | 51.6104 | 34.1797 | 15.1367 | 9.1504 | 4.8413 | 34.1797 | 75.6836 | 91.5039 | 96.8262 | 51.6104 | 60.7645 |

| Pass | Extra recurrent updates | mAP change from previous pass | mAP change from Pass 1 |
| ---: | ---: | ---: | ---: |
| 1 | 0 | +0.0000 | +0.0000 |
| 2 | 1 | +1.4257 | +1.4257 |
| 3 | 2 | +0.5367 | +1.9623 |
| 4 | 3 | +0.3066 | +2.2689 |

These partial metrics show that the reporting path and all recurrent pass outputs work;
they do not establish model quality because the checkpoint saw only 1,536 training rows
and the test used only 2,048 rows. They are excluded from the formal comparison table.
Batch 32 training and batch 128 evaluation are the selected safe settings. Larger
batches were not attempted because measured memory plus long visual-token tails left
insufficient safety margin; they are safe selections, not proven mathematical maxima.

- Evidence: training tmux/log
  `rls_v4_eosaux_b32_6b82132_20260731` and
  `/home/mnt/liyiwei/outputs/rls_v4_eosaux_b32_6b82132_20260731.tmux.log`;
  training output
  `/home/mnt/liyiwei/outputs/rls_v4_eosaux_b32_6b82132_20260731`;
  evaluation tmux/log
  `rls_v4_eval_gqa_b128_6b82132_20260731` and
  `/home/mnt/liyiwei/outputs/rls_v4_eval_gqa_b128_6b82132_20260731.tmux.log`;
  evaluation output
  `/home/mnt/liyiwei/outputs/rls_v4_eval_gqa_b128_6b82132_20260731`.

## EXP-004A/B/C/D — Superseded COCO damped recurrent latent-slot sweep

- Status/date: all four passed, 2026-07-31. The serial queue script was written at
  10:19:54Z. Each entry lists the time its `training_result.json` and `report.json`
  were written: K=8 train 13:09:59Z / test 13:16:34Z; K=12 train 16:08:21Z / test
  16:15:03Z; K=16 train 19:09:16Z / test 19:16:01Z; K=32 train 22:20:49Z / test
  22:27:52Z. Exact per-run start timestamps were not recorded (`N/A`).
- Objective: historical full-COCO evaluation of the former damped mid-decoder design.
  This architecture was later superseded by the query-only history recurrent Block;
  these values remain evidence but do not select the active configuration.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `5b6207f266dd5b3cf63b2269716cdd379a00eb0b`, identical for all four runs.
  Visual-length bucketing did not exist at this commit, so all four runs used plain
  modality-grouped padding. The LoRA baselines in EXP-001/002/003 also ran without
  bucketing, so the comparison is not distorted by that setting.
- Data: `looped_vl_single_baselines_v1/coco`; 566,747 train rows and 25,010 held-out
  test rows; no validation use. Train manifest SHA-256
  `555211fd08280e4e9ab72f040d64b002c1e2aa4b72f6cb6b42427adabb381ff8`; test manifest
  SHA-256 `a36d9000f665bbaf6bc7d7e472f38e069bb201c0cf6770c0784b498e7193ec89`. Unique
  train image count `N/A`; test image count 5,000.
- Model: `damped_mid_decoder_latent_slot_recurrence_no_lora_v3`, training protocol
  `pure_recurrent_single_stage_eos_weighted_aux_v4`, `backbone_frozen` true,
  `lora_enabled` false, `formal_training_stages` 1. Loop layers 12–20, 4 total passes,
  slots-only extra passes, `update_prefix_in_extra_loops` false,
  `detach_prefix_kv_cache` true, `auxiliary_pooling`
  `eos_conditioned_weighted_slots`, `fusion_type` `eos_conditioned_slot_attention`,
  `fusion_residual_gate_init` 0.0. Backbone SHA-256
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` before and after
  in all four runs.
- Trainable scope, identical in structure for all four runs: `latent_slots`,
  `auxiliary_embedding_head.normalization.weight`,
  `auxiliary_embedding_head.projection.weight`, `eos_delta`, `late_fusion.gamma`,
  and the four `late_fusion` projection weights. Only the slot tensor changes size
  with K.
- Optimization: seed 42, 1 epoch, 2,214 loader batches per rank, 1,107 optimizer steps,
  AdamW with betas 0.9/0.95, learning rate 1e-5, cosine schedule, warmup ratio 0.03,
  weight decay 0.01, gradient clip 1.0, temperature 0.02, final InfoNCE weight 1.0 at
  every step, mean per-pass auxiliary InfoNCE weight 0.1, slot diversity weight 0.05.
- Runtime: FP16 autocast with float32 trainable parameters, resolved attention
  implementation `sdpa` from requested `auto`, gradient checkpointing enabled,
  per-device batch 32, contrastive global batch 256, gradient accumulation 2,
  optimizer global batch 512, 4 data workers, prefetch factor 2, world size 8,
  `modality_grouped_padding` true, `visual_length_buckets` unset.
- Checkpoints: written every 100 steps with at most 1 retained; every run resolved to
  `step001107.pt` at cursor `processed_samples` 566,752, confirming exactly one epoch.
- Evaluation: COCO two-direction retrieval. Text-to-image uses 25,010 caption queries
  over a 5,000-image gallery; image-to-text uses 5,000 image queries over a 25,010
  caption gallery. Test split only, cutoffs 1/5/10/20, nDCG cutoff 10, metrics on the
  0–100 percentage scale, 60,020 encoded items per run.
- Evidence: tmux `rls_coco_full_b32_k8_k12_k16_k32_5b6207f_20260731`; queue root
  `/home/mnt/liyiwei/outputs/rls_coco_full_b32_k8_k12_k16_k32_5b6207f_20260731`
  containing `queue.sh`, `queue_progress.log` ending in `queue_finished`, and the four
  `k8`, `k12`, `k16`, `k32` subdirectories. The queue tmux log file exists but is
  empty, so per-run console output must be read from each subdirectory instead.

### Per-run identity and efficiency

| ID | K | Trainable | Inference | Train seconds | Median train samples/second | Peak train memory | Test seconds | Encoding items/second | Peak test memory | Final total loss | Final checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EXP-004A | 8 | 2,641,921 | 2,115,585 | 10,157.71 | 56.03 | 11.70 GiB | 373.61 | 190.999 | 10.82 GiB | 0.4541 | `16f40fa21d06301400b540a5a20f19313cfb8943dfe82a6dece56c1cba29b1c0` |
| EXP-004B | 12 | 2,650,113 | 2,123,777 | 10,253.48 | 55.51 | 12.31 GiB | 381.53 | 186.390 | 10.82 GiB | 0.4214 | `b8195a8ef29a2b07d36de89564aaa5eb1e55f0e94de752adffcec988a7cdb80f` |
| EXP-004C | 16 | 2,658,305 | 2,131,969 | 10,402.19 | 54.67 | 12.84 GiB | 383.93 | 184.597 | 10.82 GiB | 0.4173 | `cdefdfc53a071d92dffa25e85e23aa97fce6adeefeeebf97600fbd40357b45d6` |
| EXP-004D | 32 | 2,691,073 | 2,164,737 | 11,034.04 | 51.57 | 15.11 GiB | 403.02 | 174.493 | 10.85 GiB | 0.3877 | `87ca154c17fa30f11db6febed2b40b56912bbcb8be46941e1ab5dbe9ce90232d` |

Training memory is the maximum `gpu_peak_memory_allocated_bytes` across logged steps.
Test memory is `peak_gpu_memory_bytes` from each `report.json`. Median throughput
excludes the first three logged points so that warmup is not counted.

### Pass 4 test metrics by K (%)

| ID | K | Direction | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EXP-004A | 8 | Text→image | 66.5165 | 55.8257 | 15.8561 | 8.6653 | 4.6180 | 55.8257 | 79.2803 | 86.6533 | 92.3591 | 66.5165 | 70.9131 |
| EXP-004A | 8 | Image→text | 55.5491 | 70.2600 | 50.2560 | 31.7980 | 18.5670 | 14.0467 | 50.2387 | 63.5727 | 74.2387 | 78.4199 | 62.3308 |
| EXP-004A | 8 | Equal-direction mean | 61.0328 | 63.0428 | 33.0560 | 20.2317 | 11.5925 | 34.9362 | 64.7595 | 75.1130 | 83.2989 | 72.4682 | 66.6219 |
| EXP-004B | 12 | Text→image | 65.8472 | 55.2339 | 15.7385 | 8.5958 | 4.5934 | 55.2339 | 78.6925 | 85.9576 | 91.8673 | 65.8472 | 70.2152 |
| EXP-004B | 12 | Image→text | 55.9960 | 70.8400 | 50.5640 | 31.9500 | 18.6420 | 14.1620 | 50.5453 | 63.8740 | 74.5387 | 78.9134 | 62.7404 |
| EXP-004B | 12 | Equal-direction mean | 60.9216 | 63.0370 | 33.1513 | 20.2729 | 11.6177 | 34.6980 | 64.6189 | 74.9158 | 83.2030 | 72.3803 | 66.4778 |
| EXP-004C | 16 | Text→image | 66.1807 | 55.6257 | 15.8329 | 8.6202 | 4.6034 | 55.6257 | 79.1643 | 86.2015 | 92.0672 | 66.1807 | 70.5342 |
| EXP-004C | 16 | Image→text | 56.5751 | 71.2000 | 51.0680 | 32.2440 | 18.8330 | 14.2340 | 51.0507 | 64.4627 | 75.3013 | 79.3803 | 63.2988 |
| EXP-004C | 16 | Equal-direction mean | 61.3779 | 63.4129 | 33.4504 | 20.4321 | 11.7182 | 34.9299 | 65.1075 | 75.3321 | 83.6843 | 72.7805 | 66.9165 |
| EXP-004D | 32 | Text→image | 66.6458 | 56.0176 | 15.8880 | 8.6829 | 4.6140 | 56.0176 | 79.4402 | 86.8293 | 92.2791 | 66.6458 | 71.0604 |
| EXP-004D | 32 | Image→text | 56.0733 | 70.6000 | 50.7040 | 31.9080 | 18.7420 | 14.1140 | 50.6847 | 63.7900 | 74.9387 | 78.7059 | 62.6542 |
| EXP-004D | 32 | Equal-direction mean | 61.3595 | 63.3088 | 33.2960 | 20.2955 | 11.6780 | 35.0658 | 65.0624 | 75.3096 | 83.6089 | 72.6758 | 66.8573 |

### Equal-direction mean by recurrent pass (%)

| ID | K | Pass | Extra recurrent updates | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EXP-004A | 8 | 1 | 0 | 61.4771 | 63.4608 | 33.3960 | 20.3653 | 11.6480 | 35.1784 | 65.2111 | 75.4646 | 83.5762 | 72.8736 | 67.0484 |
| EXP-004A | 8 | 2 | 1 | 61.2203 | 63.1348 | 33.2448 | 20.2737 | 11.6055 | 35.0521 | 65.0158 | 75.2683 | 83.3902 | 72.5717 | 66.7922 |
| EXP-004A | 8 | 3 | 2 | 61.0843 | 62.9468 | 33.1128 | 20.2533 | 11.5876 | 34.9522 | 64.8675 | 75.1923 | 83.2962 | 72.4343 | 66.6759 |
| EXP-004A | 8 | 4 | 3 | 61.0328 | 63.0428 | 33.0560 | 20.2317 | 11.5925 | 34.9362 | 64.7595 | 75.1130 | 83.2989 | 72.4682 | 66.6219 |
| EXP-004B | 12 | 1 | 0 | 61.0989 | 63.1629 | 33.1980 | 20.2861 | 11.6264 | 34.8159 | 64.7172 | 74.9998 | 83.2646 | 72.5060 | 66.6245 |
| EXP-004B | 12 | 2 | 1 | 60.9476 | 63.0609 | 33.1577 | 20.2295 | 11.6166 | 34.7219 | 64.6353 | 74.8665 | 83.1963 | 72.4192 | 66.4748 |
| EXP-004B | 12 | 3 | 2 | 60.9028 | 62.9630 | 33.0941 | 20.2511 | 11.6111 | 34.6720 | 64.5733 | 74.8745 | 83.1750 | 72.3425 | 66.4472 |
| EXP-004B | 12 | 4 | 3 | 60.9216 | 63.0370 | 33.1513 | 20.2729 | 11.6177 | 34.6980 | 64.6189 | 74.9158 | 83.2030 | 72.3803 | 66.4778 |
| EXP-004C | 16 | 1 | 0 | 61.3324 | 63.2068 | 33.3684 | 20.3751 | 11.6963 | 34.9398 | 65.0248 | 75.2744 | 83.6306 | 72.6086 | 66.8394 |
| EXP-004C | 16 | 2 | 1 | 61.3051 | 63.2529 | 33.3924 | 20.3819 | 11.6995 | 34.9219 | 65.0252 | 75.2707 | 83.6226 | 72.6420 | 66.8264 |
| EXP-004C | 16 | 3 | 2 | 61.3330 | 63.3409 | 33.3852 | 20.4211 | 11.7127 | 34.8979 | 65.0372 | 75.3184 | 83.6786 | 72.7196 | 66.8770 |
| EXP-004C | 16 | 4 | 3 | 61.3779 | 63.4129 | 33.4504 | 20.4321 | 11.7182 | 34.9299 | 65.1075 | 75.3321 | 83.6843 | 72.7805 | 66.9165 |
| EXP-004D | 32 | 1 | 0 | 61.1474 | 62.9568 | 33.1496 | 20.2531 | 11.6449 | 34.9222 | 64.8741 | 75.1733 | 83.5145 | 72.4286 | 66.6809 |
| EXP-004D | 32 | 2 | 1 | 61.1418 | 63.0388 | 33.1616 | 20.2231 | 11.6364 | 34.9558 | 64.9108 | 75.1297 | 83.4485 | 72.4735 | 66.6539 |
| EXP-004D | 32 | 3 | 2 | 61.2348 | 63.1248 | 33.3016 | 20.2561 | 11.6565 | 34.9618 | 65.0581 | 75.1957 | 83.5229 | 72.5457 | 66.7336 |
| EXP-004D | 32 | 4 | 3 | 61.3595 | 63.3088 | 33.2960 | 20.2955 | 11.6780 | 35.0658 | 65.0624 | 75.3096 | 83.6089 | 72.6758 | 66.8573 |

### Recurrent gain and comparison against the two existing COCO baselines

| ID | K | Pass 1 mAP | Pass 4 mAP | Pass 4 − Pass 1 | Pass 4 − frozen backbone | Pass 4 − LoRA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EXP-004A | 8 | 61.4771 | 61.0328 | −0.4443 | −0.2115 | −7.0000 |
| EXP-004B | 12 | 61.0989 | 60.9216 | −0.1773 | −0.3227 | −7.1112 |
| EXP-004C | 16 | 61.3324 | 61.3779 | +0.0455 | +0.1336 | −6.6549 |
| EXP-004D | 32 | 61.1474 | 61.3595 | +0.2121 | +0.1152 | −6.6733 |

All differences are percentage points of the equal-direction mean mAP. The frozen
backbone reference is 61.2443 and the LoRA reference is 68.0328, both from the same
25,010-row COCO test split.

Three facts hold across all four runs. First, extra recurrent passes change the result
by less than half a percentage point in either direction, and the sign depends on K.
Second, the best of the four, K=16 at 61.3779, exceeds the frozen backbone by only
0.1336 points while the LoRA baseline exceeds it by 6.7885 points. Third, the four K
values span only 0.4563 points, which is too narrow to name a best K from one seed per
setting.

### Recorded training-time diagnostics

| ID | K | Final late-fusion gate | Final fusion attention entropy (nats) | Step where `pooling_collapse` first became true | Final slot pairwise cosine | Final Pass 1→4 relative slot update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EXP-004A | 8 | +0.00537 | 0.0758 | 650 | 0.6315 | 0.9955 → 0.2555 |
| EXP-004B | 12 | −0.00533 | 0.0111 | 250 | 0.6224 | 0.9861 → 0.2538 |
| EXP-004C | 16 | −0.00529 | 0.0128 | 200 | 0.6249 | 0.9846 → 0.2534 |
| EXP-004D | 32 | +0.00535 | 0.0053 | 200 | 0.6132 | 0.9530 → 0.2530 |

These are recorded values from `train_metrics.jsonl`, not inferred quantities. The gate
is `tanh(gamma)` in `fused = eos_hidden_state + gate * delta`. A final magnitude near
0.0053 is the scalar multiplier on `delta`; it is not by itself the norm ratio between
the fused residual and EOS, which was not logged. The gate magnitude grew monotonically
and then flattened at nearly the same value in all four runs, while fusion attention
entropy fell toward zero, meaning the attention selected essentially one slot. These
observations are consistent with the small measured differences against the frozen
baseline, but they do not alone prove the residual's actual magnitude or why the gate
followed this trajectory. Those quantities require the diagnostic ablations that have
not yet run.

## EXP-005 — COCO query-only history recurrent Block, K=8, R=1, fixed exit

- Status/date: passed. Training ran from 2026-08-01T17:25:17Z to
  2026-08-01T18:16:08Z; full-test evaluation finished at 2026-08-01T18:20:02Z.
- Objective: parameter-matched non-recurrent control for the active query-only design.
  It applies the trainable Block once and tests whether the new history readout helps
  before attributing any gain to repeated shared computation.
- Route/node: `8XV100`,
  `pt-cd238bc011a547dfa1a2f106b7bf6b1c-worker-0`, 8 × Tesla V100-SXM2-32GB.
- Code: commit `9792f7021807d1d441618b95345fde61c565822c`.
- Data: 566,747 COCO train rows and 25,010 held-out test caption rows; no validation.
  Train manifest SHA-256
  `555211fd08280e4e9ab72f040d64b002c1e2aa4b72f6cb6b42427adabb381ff8`;
  test manifest SHA-256
  `a36d9000f665bbaf6bc7d7e472f38e069bb201c0cf6770c0784b498e7193ec89`.
- Candidates: immutable FP16 unit-normalized Qwen banks. Training uses COCO train image
  and caption manifest hashes
  `9c8fb538574c0687dd415e4dad9212454bcb8d4ff6d0d9e8cda1e6496aed692c` and
  `9d2f1b0b77ccce7cb9ead1a6f188cb0bfd8f5fb5cdd8ab3d51b794236a02b0c1`;
  test uses COCO test image and caption hashes
  `b23e6fd29ddd472b3c1c95dd2d1d6278a40e45c94862053aa0015c878549110d` and
  `642299c06d1a7003ebd861a3f5d341501fd5c6f4333e3f37fa7497e908027635`.
  Candidate Qwen forward calls were exactly zero.
- Model: `query_only_history_recurrent_no_lora_v1`, K=8, R=1, fixed exit,
  288-dimensional state, two-layer shared Block, frozen Layer 7/14/21/28 histories,
  EOS-conditioned slot attention pooling, and zero-gated residual fusion. It has
  4,878,321 trainable and inference-retained parameters, with no LoRA.
- Optimization: AdamW, betas 0.9/0.95, seed 42, 1 epoch, 2,214 optimizer steps,
  learning rate 1e-4, linear decay, warmup ratio 0.02, weight decay 0.01, gradient clip
  1.0, temperature 0.02, main InfoNCE weight 1.0, auxiliary InfoNCE weight 0.1.
- Runtime: frozen Qwen FP16 plus recurrent FP16 autocast, scaled dot-product attention,
  per-device batch 32, global contrastive/optimizer batch 256, 4 data workers, and visual
  length bucketing. Training took 3,050.82 seconds; steady median throughput was 188.41
  samples/second; peak allocated memory was 5.40 GiB per rank. Test took 175.24 seconds
  with 5.27 GiB rank-zero peak allocated memory.
- Checkpoints: one rolling checkpoint only, `step002214.pt`; final recurrent model
  SHA-256 `fd27ef1bb8ad97d69bac937d235920ddebd8acdc9386ca8b89002f96e6886ffc`.
  The backbone checksum stayed
  `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`.
- Evidence: tmux `query_recurrent_v1_9792f70_20260802`; log
  `/home/mnt/liyiwei/loopedTransformer/outputs/query_recurrent_v1_9792f70_20260802.log`;
  output root
  `/home/mnt/liyiwei/loopedTransformer/outputs/query_recurrent_v1_9792f70_20260802/coco_k8_r1_fixed`.

The in-run Pass 0 is the correct reference for the measured gain because it uses the
same immutable FP16 candidate banks. It differs by 0.0046 mAP points from EXP-000A,
whose candidates were cached through the older evaluation path. Pass 1 improved
equal-direction mean mAP by 0.5286 points over this in-run Pass 0. Because R=1, Pass 1,
fixed output, dynamic-hard alias, and dynamic-soft alias are numerically identical; this
run does not demonstrate a benefit from recurrence.

### Pass 0 and Pass 1 test metrics (%)

| Pass | Direction | mAP | P@1 | P@5 | P@10 | P@20 | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Text→image | 64.5988 | 53.5946 | 15.5378 | 8.5598 | 4.5850 | 53.5946 | 77.6889 | 85.5978 | 91.6993 | 64.5988 | 69.1608 |
| 0 | Image→text | 57.8989 | 74.4600 | 52.3240 | 32.6860 | 18.8970 | 14.8867 | 52.3060 | 65.3467 | 75.5587 | 81.5428 | 64.7786 |
| 0 | Equal-direction mean | 61.2489 | 64.0273 | 33.9309 | 20.6229 | 11.7410 | 34.2406 | 64.9975 | 75.4722 | 83.6290 | 73.0708 | 66.9697 |
| 1 | Text→image | 65.2048 | 54.3023 | 15.6457 | 8.6014 | 4.5962 | 54.3023 | 78.2287 | 86.0136 | 91.9232 | 65.2048 | 69.7338 |
| 1 | Image→text | 58.3502 | 74.4600 | 52.7120 | 32.8920 | 18.9970 | 14.8867 | 52.6927 | 65.7580 | 75.9567 | 81.6651 | 65.1256 |
| 1 | Equal-direction mean | 61.7775 | 64.3811 | 34.1789 | 20.7467 | 11.7966 | 34.5945 | 65.4607 | 75.8858 | 83.9399 | 73.4350 | 67.4297 |

Coverage was 100%. The final logged loss was 0.9861, the final gradient norm was 0.3951,
and all recorded losses and gradients were finite. Final slot pairwise absolute cosine
was 0.9772, so slot collapse remains a diagnostic concern for the R=4 comparisons.

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

This is the single primary comparison table. Rows distinguish experiments. The first
header level groups results by dataset; the second level lists the compared metrics.
COCO uses the equal-direction mean of text-to-image and image-to-text. GQA Balanced and
CLEVR use answer retrieval. Compare metrics only within the same dataset.

The active query-only recurrent counts are 4,876,305 at K=1, 4,877,169 at K=4, and
4,878,321 at K=8; every parameter is retained for inference. EXP-005 reports Pass 1
because it is the fixed R=1 control. The four damped recurrent rows are superseded
historical EXP-004 Pass-4 results with their older parameter accounting. `N/A` means the
corresponding full held-out test result does not exist.

<table>
<thead>
<tr>
<th rowspan="2">Experiment</th>
<th rowspan="2">K</th>
<th rowspan="2">R</th>
<th rowspan="2">Added trainable parameters</th>
<th colspan="12">COCO</th>
<th colspan="12">GQA Balanced</th>
<th colspan="12">CLEVR</th>
</tr>
<tr>
<th>Status</th><th>mAP</th><th>P@1</th><th>P@5</th><th>P@10</th><th>P@20</th>
<th>R@1</th><th>R@5</th><th>R@10</th><th>R@20</th><th>MRR</th><th>nDCG@10</th>
<th>Status</th><th>mAP</th><th>P@1</th><th>P@5</th><th>P@10</th><th>P@20</th>
<th>R@1</th><th>R@5</th><th>R@10</th><th>R@20</th><th>MRR</th><th>nDCG@10</th>
<th>Status</th><th>mAP</th><th>P@1</th><th>P@5</th><th>P@10</th><th>P@20</th>
<th>R@1</th><th>R@5</th><th>R@10</th><th>R@20</th><th>MRR</th><th>nDCG@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>Frozen backbone</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>Passed</td>
<td>61.2443</td><td>64.0153</td><td>33.9117</td><td>20.6115</td><td>11.7428</td>
<td>34.2366</td><td>64.9735</td><td>75.4542</td><td>83.6250</td><td>73.0671</td>
<td>66.9576</td>
<td>Passed</td>
<td>52.1141</td><td>36.2697</td><td>14.4077</td><td>8.5292</td><td>4.6554</td>
<td>36.2697</td><td>72.0385</td><td>85.2918</td><td>93.1070</td><td>52.1141</td>
<td>59.5010</td>
<td>Passed</td>
<td>84.9359</td><td>73.1707</td><td>19.8499</td><td>9.9993</td><td>5.0000</td>
<td>73.1707</td><td>99.2493</td><td>99.9933</td><td>100.0000</td><td>84.9359</td>
<td>88.7750</td>
</tr>
<tr>
<td>Backbone + LoRA</td>
<td>0</td>
<td>1</td>
<td>31,195,136</td>
<td>Passed</td>
<td>68.0328</td><td>70.2716</td><td>37.3290</td><td>22.4826</td><td>12.6300</td>
<td>38.8849</td><td>71.0345</td><td>81.2199</td><td>88.4540</td><td>78.5132</td>
<td>73.3522</td>
<td>Passed</td>
<td>74.9712</td><td>62.9035</td><td>17.9043</td><td>9.3473</td><td>4.8056</td>
<td>62.9035</td><td>89.5214</td><td>93.4727</td><td>96.1123</td><td>74.9712</td>
<td>79.3467</td>
<td>Passed</td>
<td>99.2076</td><td>98.4640</td><td>19.9997</td><td>10.0000</td><td>5.0000</td>
<td>98.4640</td><td>99.9987</td><td>100.0000</td><td>100.0000</td><td>99.2076</td>
<td>99.4138</td>
</tr>
<tr>
<td>Backbone + LoRA, decoder layers 24–27 only</td>
<td>0</td>
<td>1</td>
<td>4,456,448</td>
<td>Passed</td>
<td>64.8443</td><td>66.7622</td><td>35.3295</td><td>21.4518</td><td>12.1616</td>
<td>37.0155</td><td>68.1486</td><td>78.5195</td><td>86.1819</td><td>75.8752</td>
<td>70.3234</td>
<td>Running</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
<tr>
<td>Query-only history recurrent Block (no LoRA), fixed Pass 1</td>
<td>8</td>
<td>1</td>
<td>4,878,321</td>
<td>Passed</td>
<td>61.7775</td><td>64.3811</td><td>34.1789</td><td>20.7467</td><td>11.7966</td>
<td>34.5945</td><td>65.4607</td><td>75.8858</td><td>83.9399</td><td>73.4350</td>
<td>67.4297</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
<tr>
<td>Superseded damped recurrent latent slots (no LoRA), Pass 4</td>
<td>8</td>
<td>4</td>
<td>2,641,921</td>
<td>Passed</td>
<td>61.0328</td><td>63.0428</td><td>33.0560</td><td>20.2317</td><td>11.5925</td>
<td>34.9362</td><td>64.7595</td><td>75.1130</td><td>83.2989</td><td>72.4682</td><td>66.6219</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
<tr>
<td>Superseded damped recurrent latent slots (no LoRA), Pass 4</td>
<td>12</td>
<td>4</td>
<td>2,650,113</td>
<td>Passed</td>
<td>60.9216</td><td>63.0370</td><td>33.1513</td><td>20.2729</td><td>11.6177</td>
<td>34.6980</td><td>64.6189</td><td>74.9158</td><td>83.2030</td><td>72.3803</td><td>66.4778</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
<tr>
<td>Superseded damped recurrent latent slots (no LoRA), Pass 4</td>
<td>16</td>
<td>4</td>
<td>2,658,305</td>
<td>Passed</td>
<td>61.3779</td><td>63.4129</td><td>33.4504</td><td>20.4321</td><td>11.7182</td>
<td>34.9299</td><td>65.1075</td><td>75.3321</td><td>83.6843</td><td>72.7805</td><td>66.9165</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
<tr>
<td>Superseded damped recurrent latent slots (no LoRA), Pass 4</td>
<td>32</td>
<td>4</td>
<td>2,691,073</td>
<td>Passed</td>
<td>61.3595</td><td>63.3088</td><td>33.2960</td><td>20.2955</td><td>11.6780</td>
<td>35.0658</td><td>65.0624</td><td>75.3096</td><td>83.6089</td><td>72.6758</td><td>66.8573</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
</tbody>
</table>
