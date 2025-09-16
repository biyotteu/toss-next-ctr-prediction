import os
import json
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .config import Cfg
from .logger import stage
from .dataset_polars import make_dataloaders_polars, infer_feature_types_pl
from .split_polars import leakage_free_split_pl
from .models.qin_like import QINLike
from .models.qin_v9ish import QINV9ish
from .metrics import average_precision, weighted_logloss
from .calibration import TemperatureScaler


def _bool_from_str(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    return x in ("1", "true", "t", "yes", "y")


def calibrate_main(cfg_path: str, ckpt_path: str = None, use_wll_override: bool = None, save_as: str = None):
    cfg = Cfg.load(cfg_path)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    # 1) Load full train (Polars) and split
    stage("Load train parquet (Polars)", 1, 4)
    scan = pl.scan_parquet(cfg.paths.train_parquet)
    cols = [c for c in scan.collect_schema().names() if c not in set(cfg.force_drop_cols)]
    full_pl = pl.read_parquet(cfg.paths.train_parquet, columns=cols)

    stage("Leakage-free split (Polars)", 2, 4)
    train_pl, val_pl = leakage_free_split_pl(full_pl, cfg)
    if val_pl.height == 0:
        raise RuntimeError("Validation split is empty. Adjust split config (val_ratio/time_val_ratio).")

    # 2) Ensure numeric stats are available on cfg
    stats_path = os.path.join(cfg.artifacts_dir, 'num_stats.json')
    num_stats = None
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                num_stats = json.load(f)
            print(f"[NUM-STATS] loaded from {stats_path} (cols={len(num_stats)})")
        except Exception as e:
            print(f"[NUM-STATS] failed to load cached stats, will recompute: {e}")
            num_stats = None
    if num_stats is None:
        cats_all, nums_all = infer_feature_types_pl(train_pl, cfg.label_col, cfg.seq_col)
        cats_all = [c for c in cats_all if c not in set(cfg.force_drop_cols)]
        nums_all = [c for c in nums_all if c not in set(cfg.force_drop_cols)]
        num_stats = {}
        if len(nums_all) > 0:
            stat_df = train_pl.select([pl.col(c).cast(pl.Float64).alias(c) for c in nums_all]).describe()
            stat_col = "stat" if "stat" in stat_df.columns else ("statistic" if "statistic" in stat_df.columns else None)
            if stat_col is None:
                mean_row = train_pl.select([pl.col(c).cast(pl.Float64).mean().alias(c) for c in nums_all]).to_dicts()[0]
                std_row = train_pl.select([pl.col(c).cast(pl.Float64).std().alias(c) for c in nums_all]).to_dicts()[0]
            else:
                rows = stat_df.to_dicts()
                mean_row = next((d for d in rows if d.get(stat_col) in ("mean", "avg", "average")), {})
                std_row  = next((d for d in rows if d.get(stat_col) in ("std", "std_dev", "stddev", "std dev")), {})
                if stat_col in mean_row:
                    mean_row = {k: v for k, v in mean_row.items() if k != stat_col}
                if stat_col in std_row:
                    std_row = {k: v for k, v in std_row.items() if k != stat_col}
            for c in nums_all:
                mu = float(mean_row.get(c, 0.0))
                sig = float(std_row.get(c, 1.0))
                if sig == 0.0 or np.isnan(sig):
                    sig = 1.0
                num_stats[c] = [mu, sig]
        os.makedirs(cfg.artifacts_dir, exist_ok=True)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(num_stats, f)
        print(f"[NUM-STATS] computed via Polars and saved to {stats_path} (cols={len(num_stats)})")
    cfg.d['num_stats'] = num_stats

    # 2) Dataloaders (fit train stats; val has NO downsampling)
    train_loader, val_loader, input_dims = make_dataloaders_polars(train_pl, val_pl, cfg)

    # 3) Build model & load BEST checkpoint
    stage("Build model & load best checkpoint", 3, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = cfg.model.get('name', 'qin_like')
    if model_name == 'qin_v9ish':
        model = QINV9ish(
            n_cat=input_dims['n_cat'], n_num=input_dims['n_num'],
            cat_buckets=input_dims['cat_buckets'], seq_vocab_size=input_dims['seq_vocab_size'],
            emb_dim=cfg.embedding_dim,
            dropout=cfg.model.dropout,
            attn_dim=cfg.model.attn_dim,
            qnn_num_layers=cfg.model.qnn.num_layers,
            qnn_num_row=cfg.model.qnn.num_row,
            qnn_net_dropout=cfg.model.qnn.net_dropout,
            qnn_batch_norm=cfg.model.qnn.batch_norm,
            attn_use_scale=cfg.model.attn_use_scale,
        ).to(device)
    else:
        model = QINLike(
            n_cat=input_dims['n_cat'], n_num=input_dims['n_num'],
            cat_buckets=input_dims['cat_buckets'], seq_vocab_size=input_dims['seq_vocab_size'],
            emb_dim=cfg.embedding_dim,
            hidden_dim=cfg.model.hidden_dim,
            qnn_hidden=cfg.model.qnn_hidden,
            qnn_heads=cfg.model.qnn_heads,
            dropout=cfg.model.dropout,
            attn_dim=cfg.model.attn_dim,
            attn_topk=cfg.model.attn_topk,
        ).to(device)

    ckpt_path = ckpt_path or os.path.join(cfg.output_dir, 'model_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    # 4) Collect validation logits/labels via memmap to avoid OOM
    stage("Infer validation logits", 4, 4)
    n_val = len(val_loader.dataset)
    tmp_dir = os.path.join(cfg.artifacts_dir, "_cal_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    mm_logits_path = os.path.join(tmp_dir, "val_logits.f32")
    mm_labels_path = os.path.join(tmp_dir, "val_labels.i8")
    mm_logits = np.memmap(mm_logits_path, dtype=np.float32, mode='w+', shape=(n_val,))
    mm_labels = np.memmap(mm_labels_path, dtype=np.int8, mode='w+', shape=(n_val,))

    write_ptr = 0
    with torch.no_grad():
        tqdm_val_loader = tqdm(val_loader, desc="Infer validation logits", leave=True)
        for cats, nums, seqs, tgt_ids, labels in tqdm_val_loader:
            bsz = labels.shape[0]
            cats = cats.to(device, non_blocking=True)
            nums = nums.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            logits = model(cats, nums, seqs, tgt_ids)
            mm_logits[write_ptr:write_ptr+bsz] = logits.float().cpu().numpy().reshape(-1)
            mm_labels[write_ptr:write_ptr+bsz] = labels.numpy().astype(np.int8, copy=False)
            write_ptr += bsz
            del logits, cats, nums, seqs, tgt_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    val_logits = np.asarray(mm_logits)
    val_labels = np.asarray(mm_labels).astype(int)

    # Pre-calibration metrics
    probs = 1.0 / (1.0 + np.exp(-val_logits))
    ap = average_precision(val_labels, probs)
    wll = weighted_logloss(val_labels, probs)
    print(f"[VAL][PRE]  AP={ap:.6f}  WLL={wll:.6f}")

    # Fit temperature
    use_wll = cfg.calibration.use_wll_weights if use_wll_override is None else use_wll_override
    scaler = TemperatureScaler(lr=cfg.calibration.lr, max_iter=cfg.calibration.max_iter, use_wll_weights=use_wll)
    T = scaler.fit_from_arrays(val_logits, val_labels)

    # Post-calibration metrics
    probs_c = 1.0 / (1.0 + np.exp(-(val_logits / T)))
    ap_c = average_precision(val_labels, probs_c)
    wll_c = weighted_logloss(val_labels, probs_c)
    score = 0.5 * ap_c + 0.5 * (1.0 / (1.0 + wll_c))
    print(f"[VAL][POST] Score={score:.6f} AP={ap_c:.6f}  WLL={wll_c:.6f}")

    # Save temperature
    fname = save_as or 'temperature.json'
    out_path = os.path.join(cfg.artifacts_dir, fname)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'T': float(T), 'use_wll_weights': bool(use_wll)}, f)
    print(f"[CAL] saved temperature to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='./configs/config.yaml')
    p.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint (default: outputs/model_best.pt)')
    p.add_argument('--use-wll-weights', type=str, default=None, help='Override calibration.use_wll_weights (true/false)')
    p.add_argument('--save-as', type=str, default=None, help='Filename for artifacts (default: temperature.json)')
    args = p.parse_args()

    use_wll_override = _bool_from_str(args.use_wll_weights)
    calibrate_main(args.cfg, args.ckpt, use_wll_override, args.save_as)


