# toss_v6 — CTR Prediction Pipeline (DARE + QNN-α)

This repository is an end-to-end pipeline for large-scale click-through rate (CTR) prediction. From Parquet raw data → NPY shard caching → sequence encoder (DARE) + feature interaction (QNN-α) training → ensembling/calibration → submission file generation

- Data preprocessing: Parquet streaming with Polars/PyArrow, numeric imputation, categorical hashing, sequence tokenization, large NPY shard caching (managed with manifest.json)
- Model training: DARE sequence encoder + QNN-α head, AMP/compile, EMA, K-Fold, optional balanced sampler
- Evaluation/Logging: AP/WLL/Score (0.5*AP + 0.5*WLL), TensorBoard, CSV logs
- Inference/Submission: fold checkpoint ensembling (e.g., logit mean), save submission CSV (ID, clicked)
- Utilities/Tools: co-visit (covis) feature building, submission blending, calibration, diagnostics

---

## 1) Quick Start

### Environment Setup
- Prepare Python and CUDA (NVIDIA GPU recommended).
- Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> If you need CUDA, follow the official PyTorch guide to install the version matching your CUDA.

### Configure Data Paths
- Use `cfgs/dare_qnn_next.yaml` (or a variant like `cfgs/v3_base.yaml`).
- Update these fields to your actual data locations:
  - `data.train_path`, `data.test_path` (Parquet)
  - `data.cache_dir` (directory to store shard cache)

### Build Cache (optional: first run)
- Build the cache if it is missing or if `data.use_cache: false`.
- Use the provided script or run a one-liner:

```bash
# Example: equivalent to what shard.sh does
python -c "import yaml,json; from src.data.build_cache_v1 import build_train_and_test; cfg=yaml.safe_load(open('cfgs/dare_qnn_next.yaml')); mp_tr, mp_te = build_train_and_test(cfg); print(mp_tr, mp_te)"
```

### Training
- Default training scripts:

```bash
bash train.sh
# or
python -m src.train --cfg cfgs/dare_qnn_next.yaml
```

- Outputs (checkpoints/logs) are saved under `runs/<exp_name>/`:
  - Checkpoints: `ckpt_folds_*.pt`
  - Logs: `train_log.csv`, TensorBoard directory `runs/<exp_name>/`

### Inference / Submission Generation
- After training, run inference to produce the submission file.

```bash
bash infer.sh
# or
python -m src.infer --cfg cfgs/dare_qnn_next.yaml
```

- Output: `runs/<exp_name>/submission.csv` (header: `ID,clicked`)

### Submission Blending (optional)
- Blend two submissions with various methods.

```bash
python src/tools/blend_submissions.py \
  --sub1 runs/v3_k132_s1/submission.csv \
  --sub2 runs/dare_qnn_next_k100_s1/submission.csv \
  --out runs/blend_logit50_50.csv \
  --method logit_mean --w 0.5
```

---

## 2) Configurations
This summarizes common keys in `cfgs/*.yaml`. Refer to each YAML for details.

- `exp_name`: experiment name (output directory)
- `seed`, `device`, `deterministic`, `amp`, `use_compile`
- `data`
  - `train_path`, `test_path`: Parquet paths
  - `cache_dir`, `use_cache`, `chunked_build`, `shard_rows`
  - `add_isna_mask`, `impute_strategy`
  - `cat_cols`, `num_cols_explicit` or `num_patterns`
  - `hash_buckets` (+`hash_buckets_margin`), `manifest_train`, `manifest_test`
- `sequence`
  - `col`, `max_len`, `pad_id`
  - `top_k`, `recency_tau`
  - `query_mode` (S1|S2|concat), `query_key`
  - `transformer_block`, `tfm.{n_layers,n_heads,dropout,ffn_hidden,...}`
- `model`
  - `emb_dim`, `dare_dropout`, `cat_embedding_dims`
  - `qnn_alpha.{enabled, feature_embed_dim, heads, rank, proj_dim, mlp_hidden, dropout, use_se, se_reduction, use_residual, norm, pair_grouping, aux_head_weight}`
- `train`
  - `batch_size`, `epochs`, `lr`, `weight_decay`, `warmup_epochs`, `grad_clip_norm`, `early_stop_patience`
- `cv`
  - `n_splits`, `group_key`, `stratify_target`, `time_key`, `composite_group`
- `calibration`
  - `enabled`, `method` (temperature), `lr`, `iters`
- `logging`
  - `log_dir`, `tb`, `csv_log`, `verbose_steps`
- `ensemble`
  - `method` (logit_mean|mean|geom_mean|median|trim_mean|rank_avg|weighted)
  - `trim_ratio`, `weights`, `val_weight_temperature`
- `sampler`
  - `type` (for balanced), `pos_fraction`
- `ema`
  - `enabled`, `decay`, `eval_with_ema`, etc.
- `features.covis`
  - Parameters for co-visit (covis) features (optional)

---

## 3) Project Structure
- `cfgs/`: experiment configurations (YAML)
- `src/`
  - `data/`: Parquet → shard cache build, Dataset/Collate (`build_cache_v1.py`, `dataset.py`)
  - `features/`: example co-visit features (`covis.py`)
  - `models/`: DARE, QNN-α, wrapper (`dare.py`, `qnn_alpha.py`, `wrapper.py`)
  - `utils/`: logging/metrics/seed/scheduler/EMA, etc.
  - `tools/`: blending/calibration/diagnostics scripts
  - `train.py`, `infer.py`: training/inference entrypoints
- Top-level scripts: `train.sh`, `infer.sh`, `blend.sh`, `shard.sh`
- Output directory: `runs/<exp_name>/`

---

## 4) Metrics / Logging
- Validation logs include `AP`, `WLL (50:50)`, and `Score = 0.5*AP + 0.5*WLL`.
- Logging: console + CSV (`train_log.csv`) + TensorBoard (`runs/<exp_name>/`)

---

## 5) FAQ / Troubleshooting
- "No checkpoints found" during inference
  - Ensure `runs/<exp_name>/ckpt_folds_*.pt` exist after training and that `exp_name` matches the config.
- OOM (out of memory)
  - Reduce `train.batch_size` or lower `sequence.max_len` / `top_k`. Consider AMP (`amp: bf16|fp16`) if suitable.
- Data path errors
  - Double-check `data.train_path` and `data.test_path` point to real Parquet files.

---
