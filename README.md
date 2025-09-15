# QIN‑style CTR Pipeline

This project is **inspired by QIN** (Quadratic Interest Network). It implements target‑conditioned sparse attention over user sequence + quadratic interactions (QNN‑style), with:

- **Leakage‑free splits**: stratified / group by user / group by session / time‑based.
- **Class imbalance handling**: negative downsampling + unbiased reweighting, or pos_weight.
- **Hard Negative Mining (OHEM)**: keep top‑loss negatives in batch with unbiased weights.
- **Calibration**: temperature scaling on the validation set; applied at prediction.
- **Metrics**: AP, WLL (Weighted LogLoss = 0.5·E[-log p|y=1] + 0.5·E[-log(1−p)|y=0]).

## 1) Setup
```bash
pip install torch pandas numpy pyarrow scikit-learn pyyaml tqdm wandb
```

## 2) Data
```
./data/
  train.parquet
  test.parquet
  sample_submission.csv
```
Drop constants `l_feat_20`, `l_feat_23` by default.

## 3) Configure (`configs/config.yaml`)
Key options:
- `split.method`: `stratified` | `group_user` | `group_session` | `time`
  - When group methods are used, set `split.group_key` or `split.session_key` to the column name.
  - When `time`, set `split.time_col` (datetime) and `split.time_val_ratio`.
- Imbalance: `use_neg_downsampling`, `neg_downsample_ratio`, or `pos_weight`.
- HNM: `hnm.enable`, `hnm.top_neg_frac`, `hnm.min_neg`.
- Calibration: `calibration.temperature_scaling: true`.

## 4) Train
```bash
bash scripts/run_train.sh
```
Look for `STAGE` logs. After training, calibration learns temperature `T` and saves to `artifacts/temperature.json`. Logs print AP/WLL pre/post calibration.

## 5) Predict
```bash
bash scripts/run_predict.sh
```
Creates `outputs/submission.csv` (`ID,clicked`). If calibration exists, it applies `logits / T`.

## Notes
- Validation metrics are computed on the **original distribution** (no downsampling).
- For RAM limits, consider row‑group streaming in training as done in prediction.
```
```bash
pip install torch pandas numpy pyarrow scikit-learn pyyaml
```

## 2) Put data
```
./data/
  train.parquet
  test.parquet
  sample_submission.csv
```

## 3) Configure
Edit `configs/config.yaml` (paths, model dims, imbalance settings). By default we:
- Drop constant columns: `l_feat_20`, `l_feat_23`.
- Hash‑embed all categorical fields (including `inventory_id`, `l_feat_*`, etc.).
- Parse `seq` as comma‑separated ints → hashed token ids, cap length 100.
- Downsample negatives in‑batch (keep 20%). Also set per‑example weights to stay unbiased.

## 4) Train
```bash
bash scripts/run_train.sh
```
Logs include `AP` and `WLL` on the **original distribution** validation set.

## 5) Predict
```bash
bash scripts/run_predict.sh
```
Generates `outputs/submission.csv` with `ID,clicked`.

## Notes
- **AP (Average Precision)** via `sklearn.metrics.average_precision_score`.
- **WLL (Weighted LogLoss)** = 0.5 * mean(-log p | y=1) + 0.5 * mean(-log(1-p) | y=0).
- Evaluation always uses the **original (not downsampled)** validation split.
- For RAM‑limited environments, reduce columns/embeddings or increase `num_workers` and rely on row‑group streaming in `predict.py` (and consider adapting the train loader similarly).