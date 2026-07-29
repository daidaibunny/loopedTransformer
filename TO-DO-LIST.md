# TO-DO

- [x] Add failing tests for Parquet indexing, image resolution, batching, and frozen smoke guards.
- [x] Implement the mixture dataset and GQA materialized-image resolver.
- [x] Implement source-balanced smoke sample selection and batch collation.
- [x] Materialize and validate GQA Balanced train and validation images.
- [x] Run unit tests and linting locally or in the configured remote environment.
- [x] Run frozen Qwen3-VL-Embedding-2B smoke on GPU 1.
- [x] Verify embeddings and confirm the checkpoint hash is unchanged.
- [x] Benchmark 4,000 mixed-resolution samples with official Qwen preprocessing limits.
- [x] Audit raw and processed resolutions for the exact throughput sample window.
- [x] Create and validate the 100,000-row train subset with the exact 50:35:15 ratio.
- [x] Split 50,000 held-out rows into disjoint 25,000-row validation and test sets.
- [x] Freeze and enforce the required Mix and per-dataset evaluation metric contract.
