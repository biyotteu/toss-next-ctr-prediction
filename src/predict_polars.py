import os
import json
import numpy as np
import polars as pl
import pandas as pd
import torch
from tqdm import tqdm

from .config import Cfg
from .dataset_polars import CTRDatasetPolars, collate_fn, infer_feature_types_pl
from .models.qin_like import QINLike
from .models.qin_v9ish import QINV9ish


@torch.no_grad()
def predict_main(cfg_path: str, ckpt_path: str = None):
    cfg = Cfg.load(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    ckpt = ckpt_path or os.path.join(cfg.output_dir, 'model_best.pt')
    ck = torch.load(ckpt, map_location=device)
    cfg_loaded = Cfg(ck.get('cfg', cfg.d))

    # numeric stats: load if exists else compute from train parquet (Polars)
    stats_path = os.path.join(cfg.artifacts_dir, 'num_stats.json')
    num_stats = None
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                num_stats = json.load(f)
            print(f"[NUM-STATS] loaded from {stats_path} (cols={len(num_stats)})")
        except Exception as e:
            print(f"[NUM-STATS] failed to load cached stats: {e}")
    if num_stats is None:
        print("[NUM-STATS] not found; computing from train parquet via Polars...")
        scan = pl.scan_parquet(cfg.paths.train_parquet)
        cols = [c for c in scan.collect_schema().names() if c not in set(cfg.force_drop_cols)]
        train_pl_head = pl.read_parquet(cfg.paths.train_parquet, columns=cols)
        cats_all, nums_all = infer_feature_types_pl(train_pl_head, cfg.label_col, cfg.seq_col)
        cats_all = [c for c in cats_all if c not in set(cfg.force_drop_cols)]
        nums_all = [c for c in nums_all if c not in set(cfg.force_drop_cols)]
        num_stats = {}
        if len(nums_all) > 0:
            stat_df = train_pl_head.select([pl.col(c).cast(pl.Float64).alias(c) for c in nums_all]).describe()
            stat_col = "stat" if "stat" in stat_df.columns else ("statistic" if "statistic" in stat_df.columns else None)
            if stat_col is None:
                mean_row = train_pl_head.select([pl.col(c).cast(pl.Float64).mean().alias(c) for c in nums_all]).to_dicts()[0]
                std_row = train_pl_head.select([pl.col(c).cast(pl.Float64).std().alias(c) for c in nums_all]).to_dicts()[0]
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
        print(f"[NUM-STATS] computed and saved to {stats_path} (cols={len(num_stats)})")
    # ensure dict on cfg_loaded
    ns = num_stats or cfg_loaded.d.get('num_stats', {}) or {}
    if not isinstance(ns, dict):
        ns = getattr(ns, 'd', ns)
    if not isinstance(ns, dict):
        ns = {}
    cfg_loaded.d['num_stats'] = ns

    # optional temperature
    T = 1.0
    cal_path = os.path.join(cfg.artifacts_dir, 'temperature.json')
    if os.path.exists(cal_path):
        with open(cal_path, 'r', encoding='utf-8') as f:
            T = float(json.load(f).get('T', 1.0))
        print(f"[CAL] apply temperature T={T:.4f}")

    # Infer features from train header (Polars)
    scan = pl.scan_parquet(cfg.paths.train_parquet)
    cols_train = [c for c in scan.collect_schema().names() if c not in set(cfg.force_drop_cols)]
    train_pl_head = pl.read_parquet(cfg.paths.train_parquet, columns=cols_train)
    cats, nums = infer_feature_types_pl(train_pl_head, cfg.label_col, cfg.seq_col)
    cats = [c for c in cats if c not in set(cfg.force_drop_cols)]
    nums = [c for c in nums if c not in set(cfg.force_drop_cols)]
    tgt_feat = getattr(cfg, 'target_feature', None)
    if tgt_feat in cats:
        cats = [c for c in cats if c != tgt_feat]
    if tgt_feat in nums:
        nums = [c for c in nums if c != tgt_feat]

    # load test parquet (Polars)
    scan_t = pl.scan_parquet(cfg.paths.test_parquet)
    cols_test = [c for c in scan_t.collect_schema().names() if c not in set(cfg.force_drop_cols)]
    test_pl = pl.read_parquet(cfg.paths.test_parquet, columns=cols_test)
    id_col = cfg.id_col

    # build model
    model_name = cfg.model.get('name', 'qin_like')
    if model_name == 'qin_v9ish':
        model = QINV9ish(
            n_cat=len(cats), n_num=len(nums),
            cat_buckets=cfg.category_hash_buckets, seq_vocab_size=cfg.seq_vocab_size,
            emb_dim=cfg.embedding_dim, dropout=cfg.model.dropout, attn_dim=cfg.model.attn_dim,
            qnn_num_layers=cfg.model.qnn.num_layers, qnn_num_row=cfg.model.qnn.num_row,
            qnn_net_dropout=cfg.model.qnn.net_dropout, qnn_batch_norm=cfg.model.qnn.batch_norm,
            attn_use_scale=cfg.model.attn_use_scale,
        ).to(device)
    else:
        model = QINLike(
            n_cat=len(cats), n_num=len(nums),
            cat_buckets=cfg.category_hash_buckets,
            seq_vocab_size=cfg.seq_vocab_size,
            emb_dim=cfg.embedding_dim,
            hidden_dim=cfg.model.hidden_dim,
            qnn_hidden=cfg.model.qnn_hidden,
            qnn_heads=cfg.model.qnn_heads,
            dropout=cfg.model.dropout,
            attn_dim=cfg.model.attn_dim,
            attn_topk=cfg.model.attn_topk,
        ).to(device)

    model.load_state_dict(ck['model_state'])
    model.eval()

    # dataset/cache (Polars)
    cfg_loaded.d.setdefault('dataset_cache', True)
    cfg_loaded.d.setdefault('dataset_cache_backend', 'memmap')
    ds = CTRDatasetPolars(test_pl, cfg_loaded, cats, nums, is_train=False, partition="test")

    bs = cfg.batch_size
    ids_all, probs_all = [], []
    id_vals = test_pl.select(pl.col(id_col)).to_series().to_numpy()

    num_batches = (len(ds) + bs - 1) // bs
    pbar = tqdm(range(0, len(ds), bs), desc="Predict (Polars)", total=num_batches)
    for i in pbar:
        batch = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        cats_b, nums_b, seq_b, tgt_b, _ = collate_fn(batch, cfg_loaded)
        cats_b = cats_b.to(device)
        nums_b = nums_b.to(device)
        seq_b = seq_b.to(device)
        tgt_b = tgt_b.to(device)
        logits = model(cats_b, nums_b, seq_b, tgt_b) / T
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)

    probs_all = np.concatenate(probs_all)
    sub = pd.DataFrame({cfg.id_col: id_vals, 'clicked': probs_all})
    sub = sub[[cfg.id_col, 'clicked']].sort_values(cfg.id_col)
    os.makedirs(os.path.dirname(cfg.pred_output_csv), exist_ok=True)
    sub.to_csv(cfg.pred_output_csv, index=False)
    print(f"[SUBMISSION] saved to {cfg.pred_output_csv}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='./configs/config.yaml')
    p.add_argument('--ckpt', type=str, default='./outputs/model_best.pt')
    args = p.parse_args()
    predict_main(args.cfg, args.ckpt)


