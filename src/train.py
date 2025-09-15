import os
import json
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from tqdm import tqdm

from .config import Cfg
from .logger import stage, Timer
from .utils import seed_all
from .dataset import CTRFrame, make_dataloaders
from .models.qin_like import QINLike
from .models.qin_v9ish import QINV9ish
from .losses import weighted_bce_with_logits
from .metrics import average_precision, weighted_logloss
from .split import leakage_free_split
from .calibration import TemperatureScaler

def train_main(cfg_path: str):
    cfg = Cfg.load(cfg_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    timer = Timer()
    seed_all(cfg.seed)

    # 1) Load train parquet
    stage("Load train parquet (selected columns)", 1, 8)
    train_frame = CTRFrame(cfg.paths.train_parquet, cfg.force_drop_cols)
    all_cols = [c for c in train_frame.all_cols if c not in set(cfg.force_drop_cols)]
    train_df = train_frame.read_all(columns=all_cols)

    # 2) Leakage-free split
    stage("Leakage-free split", 2, 8)
    train_part, val_part = leakage_free_split(train_df, cfg)

    # class stats (on train_part)
    pos_rate = float(train_part[cfg.label_col].mean())
    neg_pos_ratio = (1 - pos_rate) / max(pos_rate, 1e-8)
    print(f"[INFO] Train={len(train_part)} Val={len(val_part)} pos_rate={pos_rate:.5f} neg/pos={neg_pos_ratio:.2f}")

    # 3) Dataloaders
    stage("Build DataLoaders", 3, 8)
    train_loader, val_loader, input_dims = make_dataloaders(train_part, val_part, cfg)

    # 4) Model
    stage("Build Model", 4, 8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    # 모델 생성 부분 교체
    model_name = cfg.model.get('name', 'qin_like')
    if model_name == 'qin_v9ish':
        model = QINV9ish(
            n_cat=input_dims['n_cat'], n_num=input_dims['n_num'],
            cat_buckets=input_dims['cat_buckets'], seq_vocab_size=input_dims['seq_vocab_size'],
            emb_dim=cfg.embedding_dim, dropout=cfg.model.dropout, attn_dim=cfg.model.attn_dim,
            qnn_num_layers=cfg.model.qnn.num_layers, qnn_num_row=cfg.model.qnn.num_row,
            qnn_net_dropout=cfg.model.qnn.net_dropout, qnn_batch_norm=cfg.model.qnn.batch_norm,
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

    # 5) Optimizer & pos_weight
    stage("Setup optimizer", 5, 8)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_weight = cfg.pos_weight
    if pos_weight is None:
        pos_weight = max(1.0, neg_pos_ratio)
    pos_weight_t = torch.tensor([pos_weight], device=device)
    print(f"[INFO] Using pos_weight={float(pos_weight):.3f}")

    # 6) Train loop with optional Hard Negative Mining
    stage("Train", 6, 8)
    best_wll = float('inf')
    best_model = None
    best_epoch = None
    best_val_logits = None
    best_val_labels = None
    best_val_probs = None
    best_ckpt = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        
        # tqdm으로 학습 진행상황 표시
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.epochs}', leave=True)
        for i, (cats, nums, seqs, tgt_ids, labels) in enumerate(train_pbar, start=1):
            cats = cats.to(device, non_blocking=True)
            nums = nums.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(cats, nums, seqs, tgt_ids)

            # ---- Hard Negative Mining (within kept batch) ----
            if cfg.hnm.enable:
                with torch.no_grad():
                    per_ex_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                    neg_mask = (labels == 0)
                    pos_mask = (labels == 1)
                    neg_losses = per_ex_loss[neg_mask]
                    if neg_losses.numel() > 0:
                        k = max(int(neg_losses.numel() * cfg.hnm.top_neg_frac), min(cfg.hnm.min_neg, neg_losses.numel()))
                        topk_vals, topk_idx = torch.topk(neg_losses, k)
                        # build weights: keep positives, keep only top-k negatives
                        weights = torch.zeros_like(labels)
                        weights[pos_mask] = 1.0
                        # map back indices
                        neg_indices = torch.nonzero(neg_mask).squeeze(1)
                        keep_neg_indices = neg_indices[topk_idx]
                        weights[keep_neg_indices] = 1.0
                        # unbiased reweighting for negatives
                        neg_keep_frac = max(k / max(neg_losses.numel(), 1), 1e-6)
                        eff_down = cfg.neg_downsample_ratio if cfg.use_neg_downsampling else 1.0
                        w_neg = 1.0 / max(eff_down * neg_keep_frac, 1e-6)
                        weights[keep_neg_indices] = w_neg
                    else:
                        # no negatives -> fall back to ones
                        weights = torch.ones_like(labels)
            else:
                # Original weighting strategy
                if cfg.use_neg_downsampling:
                    weights = torch.ones_like(labels)
                    neg_mask = (labels == 0)
                    if neg_mask.any():
                        w_neg = 1.0 / max(cfg.neg_downsample_ratio, 1e-6)
                        weights[neg_mask] = w_neg
                else:
                    weights = None

            # Loss
            if weights is not None:
                loss = weighted_bce_with_logits(logits, labels, weight=weights)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_t)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()

            losses.append(loss.item())
            
            # tqdm에 실시간 loss 정보 업데이트
            current_loss = np.mean(losses[-min(len(losses), cfg.log_interval):])
            train_pbar.set_postfix({
                'loss': f'{current_loss:.5f}',
                'step': f'{i}/{len(train_loader)}'
            })
            
            if i % cfg.log_interval == 0:
                print(f"\n[E{epoch}] step={i}, loss={current_loss:.5f}")

        # ---- Validation ----
        model.eval()
        val_logits, val_probs, val_labels = [], [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}', leave=False)
            for cats, nums, seqs, tgt_ids, labels in val_pbar:
                cats = cats.to(device)
                nums = nums.to(device)
                seqs = seqs.to(device)
                tgt_ids = tgt_ids.to(device)
                logits = model(cats, nums, seqs, tgt_ids)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
                val_logits.append(logits.cpu().numpy())
                val_labels.append(labels.numpy())
        val_probs = np.concatenate(val_probs)
        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels).astype(int)
        ap = average_precision(val_labels, val_probs)
        wll = weighted_logloss(val_labels, val_probs)
        print(f"[VAL][E{epoch}] AP={ap:.6f}  WLL={wll:.6f}")

        # save checkpoint
        if cfg.save_every_epoch:
            ckpt = os.path.join(cfg.output_dir, f"model_epoch{epoch}_wll{wll:.6f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'cfg': cfg.d,
            }, ckpt)
            print(f"[CKPT] saved to {ckpt}")
        
        # save best model
        if wll < best_wll:
            best_wll = wll
            best_model = model.state_dict()
            best_epoch = epoch
            best_val_logits = val_logits
            best_val_labels = val_labels
            best_val_probs = val_probs
            best_ckpt = os.path.join(cfg.output_dir, f"model_best.pt")
            torch.save({'model_state': best_model, 'cfg': cfg.d}, best_ckpt)
            print(f"[CKPT] saved best to {best_ckpt}")

    # 7) Temperature scaling calibration on final val set
    if cfg.calibration.use_wll_weights:
        stage("Calibrate temperature (val)", 7, 8)
        scaler = TemperatureScaler(
            lr=cfg.calibration.lr,
            max_iter=cfg.calibration.max_iter,
            use_wll_weights=cfg.calibration.use_wll_weights 
        )
        T = scaler.fit_from_arrays(best_val_logits, best_val_labels)
        cal_path = os.path.join(cfg.artifacts_dir, 'temperature.json')
        with open(cal_path, 'w', encoding='utf-8') as f:
            json.dump({'T': float(T)}, f)
        print(f"[CAL] learned temperature T={T:.4f} -> saved {cal_path}")

    # report calibrated metrics
    val_probs_cal = 1.0 / (1.0 + np.exp(-best_val_logits / T))
    ap_c = average_precision(best_val_labels, val_probs_cal)
    wll_c = weighted_logloss(best_val_labels, val_probs_cal)
    print(f"[VAL][CAL] AP={ap_c:.6f}  WLL={wll_c:.6f}")

    # 8) Save final model
    # stage("Save final", 8, 8)
    # final = os.path.join(cfg.output_dir, "model_final.pt")
    # torch.save({'model_state': model.state_dict(), 'cfg': cfg.d}, final)
    # print(f"[DONE] saved final to {final}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='./configs/config.yaml')
    args = p.parse_args()
    train_main(args.cfg)