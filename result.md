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
| Frozen backbone + damped recurrent latent slots (no LoRA) | Pending | Pending | Pending | No |

The recurrent definition was locked again on 2026-07-31. It contains no LoRA, no
recurrent connector, 8 active latent slots, 4 total passes, slots-only extra passes,
and parameter-free damping with step size 1/4. Its training-only auxiliary head uses
the fixed layer-20 EOS from Pass 1 to softly weight the current pass slots; mean pooling
is only an invalidated ablation. Earlier recurrent trials used unintended LoRA, the
removed learned connector, or mean auxiliary pooling and are invalid for selecting the
current formal configuration. No full recurrent training or test result exists under
the current definition.

Every new recurrent `run_manifest.json`, `training_result.json`, checkpoint metadata,
and `report.json` must identify the architecture as
`damped_mid_decoder_latent_slot_recurrence_no_lora_v3`, the training protocol as
`pure_recurrent_single_stage_eos_weighted_aux_v4`, `backbone_frozen` as true,
`lora_enabled` as false, and `formal_training_stages` as 1. A recurrent checkpoint
missing this identity, or containing LoRA or connector parameters, is not eligible.

The v4 recurrent optimizer has one stage and visits the full train split exactly once.
At every optimizer step, final InfoNCE has weight 1.0, the mean of the four shared-head
per-round auxiliary InfoNCE losses has weight 0.1, and final-slot diversity has weight
0.05. All 2,641,921 trainable parameters follow the same learning-rate schedule from the
first step. Inference retains 2,115,585 learned parameters; the 526,336-parameter
auxiliary head is training-only.

For every formal recurrent test, record all required metrics for Pass 1 through Pass 4.
The concise comparison row must label these as 0, 1, 2, and 3 completed recurrent
updates and report mAP change from the previous pass and from Pass 1. COCO uses its
equal-direction mean in this concise row while retaining both direction-specific metric
tables.

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
| Frozen backbone + damped recurrent latent slots (no LoRA) | Pending | Text→image | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Frozen backbone + damped recurrent latent slots (no LoRA) | Pending | Image→text | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Frozen backbone + damped recurrent latent slots (no LoRA) | Pending | Equal-direction mean | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

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

The current pure recurrent parameter count is 2,641,921 during training. This consists
of 2,115,585 inference parameters plus a 526,336-parameter training-only
EOS-conditioned weighted-slot auxiliary head that is discarded for inference. The
frozen backbone has no trainable parameters in this experiment. `N/A` means the
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
<td>Frozen backbone + damped recurrent latent slots (no LoRA)</td>
<td>8</td>
<td>4</td>
<td>2,641,921</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>Pending</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>
</tr>
</tbody>
</table>
