import os
import json
import numpy as np
import torch
from tqdm import tqdm

from .config import Cfg
from .logger import stage
from .dataset import CTRFrame, make_dataloaders
from .split import leakage_free_split
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

    # 1) Load full train and rebuild validation split (no downsampling in val)
    stage("Load train parquet (selected columns)", 1, 4)
    train_frame = CTRFrame(cfg.paths.train_parquet, cfg.force_drop_cols)
    all_cols = [c for c in train_frame.all_cols if c not in set(cfg.force_drop_cols)]
    full_df = train_frame.read_all(columns=all_cols)

    stage("Leakage-free split (for calibration)", 2, 4)
    train_df, val_df = leakage_free_split(full_df, cfg)
    if len(val_df) == 0:
        raise RuntimeError("Validation split is empty. Adjust split config (val_ratio/time_val_ratio).")

    # 2) Dataloaders (fit train stats; val has NO downsampling)
    train_loader, val_loader, input_dims = make_dataloaders(train_df, val_df, cfg)

    # 3) Build model and load BEST checkpoint
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

    # default: outputs/model_best.pt (요청 사항)
    ckpt_path = ckpt_path or os.path.join(cfg.output_dir, 'model_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    # 4) Collect validation logits/labels
    stage("Infer validation logits", 4, 4)
    val_logits, val_labels = [], []
    with torch.no_grad():
        tqdm_val_loader = tqdm(val_loader, desc="Infer validation logits", leave=True)
        for cats, nums, seqs, tgt_ids, labels in tqdm_val_loader:
            cats = cats.to(device, non_blocking=True)
            nums = nums.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            logits = model(cats, nums, seqs, tgt_ids)
            val_logits.append(logits.cpu().numpy())
            val_labels.append(labels.numpy())
    val_logits = np.concatenate(val_logits).astype(np.float32).reshape(-1)
    val_labels = np.concatenate(val_labels).astype(int).reshape(-1)

    # Pre-calibration metrics
    probs = 1.0 / (1.0 + np.exp(-val_logits))
    ap = average_precision(val_labels, probs)
    wll = weighted_logloss(val_labels, probs)
    print(f"[VAL][PRE]  AP={ap:.6f}  WLL={wll:.6f}")

    # Fit temperature (WLL-weighted optional)
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
