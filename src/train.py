import os
import json
import numpy as np
import pandas as pd
import gc
import torch
import shutil
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from datetime import datetime
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import Cfg
from .logger import stage, Timer
from .utils import seed_all
from .dataset import CTRFrame, make_dataloaders, infer_feature_types
from .models.qin_like import QINLike
from .models.qin_v9ish import QINV9ish
from .losses import weighted_bce_with_logits
from .metrics import average_precision, weighted_logloss
from .split import leakage_free_split
from .calibration import TemperatureScaler
from .utils import _pick_resume_path

def train_main(cfg_path: str):
    cfg = Cfg.load(cfg_path)
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(cfg.output_dir, 'config.yaml'))
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    # Initialize wandb if enabled and available
    use_wandb = False
    if hasattr(cfg, 'wandb') and cfg.wandb.get('enabled', False) and WANDB_AVAILABLE:
        use_wandb = True
        wandb_config = cfg.wandb
        
        # wandb 실행 이름 생성 (설정되지 않은 경우)
        run_name = wandb_config.get('name', None)
        if run_name is None:
            model_name = cfg.model.get('name', 'qin_like')
            run_name = f"{cfg.exp_name}_{model_name}_lr{cfg.lr}_bs{cfg.batch_size}"
        
        wandb.init(
            project=wandb_config.get('project', 'ctr-prediction'),
            entity=wandb_config.get('entity', None),
            name=run_name,
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config=cfg.d,  # 전체 설정을 wandb에 로깅
        )
        print(f"[INFO] Initialized wandb project: {wandb_config.get('project', 'ctr-prediction')}")
    elif hasattr(cfg, 'wandb') and cfg.wandb.get('enabled', False) and not WANDB_AVAILABLE:
        print("[WARNING] wandb is enabled in config but not installed. Install with: pip install wandb")

    timer = Timer()
    seed_all(cfg.seed)

    # 1) Load train parquet
    stage("Load train parquet (selected columns)", 1, 8)
    train_frame = CTRFrame(cfg.paths.train_parquet, cfg.force_drop_cols)
    all_cols = [c for c in train_frame.all_cols if c not in set(cfg.force_drop_cols)]
    train_df = train_frame.read_all(columns=all_cols)

    # 1.5) Numeric standardization stats (global, cached in artifacts)
    stats_path = os.path.join(cfg.artifacts_dir, 'num_stats.json')
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                num_stats = json.load(f)
            print(f"[NUM-STATS] loaded from {stats_path} (cols={len(num_stats)})")
        except Exception as e:
            print(f"[NUM-STATS] failed to load cached stats, will recompute: {e}")
            num_stats = None
    else:
        num_stats = None

    if num_stats is None:
        cats_all, nums_all = infer_feature_types(train_df, cfg.label_col, cfg.seq_col)
        cats_all = [c for c in cats_all if c not in set(cfg.force_drop_cols)]
        nums_all = [c for c in nums_all if c not in set(cfg.force_drop_cols)]
        num_stats = {}
        for c in nums_all:
            v = pd.to_numeric(train_df[c], errors='coerce')
            mu = float(v.mean())
            sig = float(v.std(ddof=0))
            if sig == 0.0 or np.isnan(sig):
                sig = 1.0
            num_stats[c] = [mu, sig]
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(num_stats, f)
        print(f"[NUM-STATS] computed from full train df and saved to {stats_path} (cols={len(num_stats)})")

    # expose to cfg for downstream Dataset usage and checkpointing
    cfg.d['num_stats'] = num_stats

    # 2) Leakage-free split
    stage("Leakage-free split", 2, 8)
    train_part, val_part = leakage_free_split(train_df, cfg)
    del train_df
    gc.collect()
    # class stats (on train_part)
    pos_rate = float(train_part[cfg.label_col].mean())
    neg_pos_ratio = (1 - pos_rate) / max(pos_rate, 1e-8)
    print(f"[INFO] Train={len(train_part)} Val={len(val_part)} pos_rate={pos_rate:.5f} neg/pos={neg_pos_ratio:.2f}")

    # 3) Dataloaders
    stage("Build DataLoaders", 3, 8)
    train_loader, val_loader, input_dims = make_dataloaders(train_part, val_part, cfg)
    del train_part
    gc.collect()
    del val_part
    gc.collect()
    
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
    # Cast potentially string-typed config values to float safely
    try:
        lr_val = float(cfg.lr)
    except Exception:
        lr_val = 0.001
    try:
        wd_val = 0.0 if cfg.weight_decay is None else float(cfg.weight_decay)
    except Exception:
        wd_val = 0.0
    optim = AdamW(model.parameters(), lr=lr_val, weight_decay=wd_val)

    # Scheduler (OneCycleLR) optional
    scheduler = None
    if hasattr(cfg, 'scheduler') and cfg.scheduler.get('enabled', False) and cfg.scheduler.get('name', 'onecycle') == 'onecycle':
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = OneCycleLR(
            optim,
            max_lr=cfg.scheduler.get('max_lr', cfg.lr * 3.0),
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.scheduler.get('pct_start', 0.1),
            div_factor=cfg.scheduler.get('div_factor', 25.0),
            final_div_factor=cfg.scheduler.get('final_div_factor', 10000.0),
            three_phase=cfg.scheduler.get('three_phase', False),
            anneal_strategy=cfg.scheduler.get('anneal_strategy', 'cos')
        )
        print(f"[INFO] OneCycleLR enabled: total_steps={total_steps}")

    pos_weight = cfg.pos_weight
    if pos_weight is None:
        pos_weight = max(1.0, neg_pos_ratio)
    pos_weight_t = torch.tensor([pos_weight], device=device)
    print(f"[INFO] Using pos_weight={float(pos_weight):.3f}")


    # ADD (resume setup)
    start_epoch = 1
    best_score = float('-inf')
    best_epoch = None
    best_ckpt = None

    # config에 아래 키가 있으면 사용합니다. 없으면 auto_resume=True로 동작
    resume_from = getattr(cfg, 'resume_from', None)         # 명시 경로 (옵션)
    auto_resume = bool(getattr(cfg, 'auto_resume', True))   # 기본 True

    resume_path = _pick_resume_path(cfg.output_dir, resume_from) if (auto_resume or resume_from) else None
    if resume_path:
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model_state'])
        if 'optim_state' in ck:
            try:
                optim.load_state_dict(ck['optim_state'])
            except Exception as e:
                print(f"[RESUME] optimizer state load failed: {e}")
        # restore scheduler state if available
        if 'sched_state' in ck and scheduler is not None:
            try:
                scheduler.load_state_dict(ck['sched_state'])
            except Exception as e:
                print(f"[RESUME] scheduler state load failed: {e}")
        start_epoch = ck.get('epoch', 0) + 1
        best_score = ck.get('best_score', float('-inf'))
        best_epoch = ck.get('best_epoch', None)
        best_ckpt  = os.path.join(cfg.output_dir, "model_best.pt") if os.path.exists(os.path.join(cfg.output_dir, "model_best.pt")) else None
        print(f"[RESUME] from {resume_path} → start_epoch={start_epoch}, best_score={best_score}")
    else:
        print("[RESUME] fresh training (no checkpoint found)")
    # 6) Train loop with optional Hard Negative Mining
    stage("Train", 6, 8)
    best_model = None
    best_val_logits = None
    best_val_labels = None
    best_val_probs = None

    for epoch in range(start_epoch, cfg.epochs + 1):
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

            # ---- Hard Negative Mining (within kept batch) + WLL-aligned class balancing ----
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            if cfg.hnm.enable:
                with torch.no_grad():
                    per_ex_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                    neg_losses = per_ex_loss[neg_mask]
                    if neg_losses.numel() > 0:
                        k = max(int(neg_losses.numel() * cfg.hnm.top_neg_frac), min(cfg.hnm.min_neg, neg_losses.numel()))
                        topk_vals, topk_idx = torch.topk(neg_losses, k)
                        # Build correction weights (selection-prob inverse)
                        corr = torch.zeros_like(labels)
                        corr[pos_mask] = 1.0  # positives kept fully
                        # map back indices
                        neg_indices = torch.nonzero(neg_mask).squeeze(1)
                        keep_neg_indices = neg_indices[topk_idx]
                        # selection prob: eff_down (from collate) * neg_keep_frac (from HNM)
                        neg_keep_frac = max(k / max(neg_losses.numel(), 1), 1e-6)
                        eff_down = cfg.neg_downsample_ratio if cfg.use_neg_downsampling else 1.0
                        keep_prob = max(eff_down * neg_keep_frac, 1e-6)
                        corr[keep_neg_indices] = 1.0 / keep_prob
                    else:
                        # no negatives -> fall back to ones
                        corr = torch.ones_like(labels)
            else:
                # Build correction weights for downsampling-only or none
                corr = torch.ones_like(labels)
                if cfg.use_neg_downsampling:
                    if neg_mask.any():
                        corr[neg_mask] = 1.0 / max(cfg.neg_downsample_ratio, 1e-6)

            # Convert correction weights to WLL-balanced per-example weights
            N = labels.numel()
            with torch.no_grad():
                sum_pos_corr = float((corr[pos_mask]).sum().item())
                sum_neg_corr = float((corr[neg_mask]).sum().item())
                if sum_pos_corr > 0.0 and sum_neg_corr > 0.0:
                    weights = torch.zeros_like(labels)
                    weights[pos_mask] = (N * 0.5) * (corr[pos_mask] / sum_pos_corr)
                    weights[neg_mask] = (N * 0.5) * (corr[neg_mask] / sum_neg_corr)
                else:
                    # degenerate batch: use uniform weights
                    weights = torch.ones_like(labels)

            # Loss (WLL-aligned weighted BCE)
            loss = weighted_bce_with_logits(logits, labels, weight=weights)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            
            # tqdm에 실시간 loss 정보 업데이트
            current_loss = np.mean(losses[-min(len(losses), cfg.log_interval):])
            train_pbar.set_postfix({
                'loss': f'{current_loss:.5f}',
                'step': f'{i}/{len(train_loader)}'
            })
            
            if i % cfg.log_interval == 0:
                print(f"\n[E{epoch}] step={i}, loss={current_loss:.5f}")
                
                # wandb 로깅
                if use_wandb:
                    wandb.log({
                        'train/loss': current_loss,
                        'train/epoch': epoch,
                        'train/step': (epoch - 1) * len(train_loader) + i,
                    })

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
        score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
        print(f"[VAL][E{epoch}] Score={score:.6f} AP={ap:.6f}  WLL={wll:.6f}")

        # wandb 검증 메트릭 로깅
        if use_wandb:
            epoch_train_loss = np.mean(losses) if losses else 0.0
            wandb.log({
                'val/score': score,
                'val/average_precision': ap,
                'val/weighted_logloss': wll,
                'train/epoch_loss': epoch_train_loss,
                'epoch': epoch,
            })

        # save checkpoint (latest)
        if cfg.save_every_epoch:
            ckpt = os.path.join(cfg.output_dir, f"model_epoch{epoch}.pt")
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'cfg': cfg.d,
                'optim_state': optim.state_dict(),
                'sched_state': scheduler.state_dict() if scheduler is not None else None,
                'best_score': best_score,
                'best_epoch': best_epoch,
            }
            torch.save(state, ckpt)
            torch.save(state, os.path.join(cfg.output_dir, "model_latest.pt"))
            print(f"[CKPT] saved to {ckpt} (and updated model_latest.pt)")
                    
        # save best model (by score)
        if score > best_score:
            best_score = score
            best_model = model.state_dict()
            best_epoch = epoch
            best_val_logits = val_logits
            best_val_labels = val_labels
            best_val_probs = val_probs
            best_ckpt = os.path.join(cfg.output_dir, "model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state': best_model,
                'cfg': cfg.d,
                'optim_state': optim.state_dict(),
                'sched_state': scheduler.state_dict() if scheduler is not None else None,
                'best_score': best_score,
                'best_epoch': best_epoch,
            }, best_ckpt)
            print(f"[CKPT] saved best to {best_ckpt}")
            
            # wandb에 최고 성능 메트릭 로깅
            if use_wandb:
                wandb.log({
                    'best/score': best_score,
                    'best/ap': ap,
                    'best/wll': wll,
                    'best/epoch': best_epoch,
                })

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
        score = 0.5 * ap_c + 0.5 * (1.0 / (1.0 + wll_c))
        print(f"[VAL][CAL] Score={score:.6f} AP={ap_c:.6f}  WLL={wll_c:.6f}")
        
        # wandb에 calibration 결과 로깅
        if use_wandb:
            wandb.log({
                'calibrated/temperature': float(T),
                'calibrated/ap': ap_c,
                'calibrated/wll': wll_c,
            })
    
    # wandb 세션 종료
    if use_wandb:
        wandb.finish()
        print("[INFO] wandb session finished")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='./configs/config.yaml')
    args = p.parse_args()
    train_main(args.cfg)