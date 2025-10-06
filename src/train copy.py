import os, yaml, math, json
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm
import torch.nn.functional as F
import gc

from src.data.sampler import BalancedBatchSampler
from src.utils.seed import set_seed 
from src.utils.log import Logger
from src.utils.metrics import final_score
from src.utils.calibration import Calibrator
from src.utils.sched import cosine_warmup_lr
from src.data.dataset import ShardedDataset, load_labels_groups_for_split, collate_sharded
from src.models.wrapper import CTRModel
from src.utils.ema import build_ema

def logit_l2(logits): 
    return (logits.float().pow(2).mean())

def bce_wll_style(logits, labels):
    """
    WLL(50:50) = 0.5 * mean_pos(-log σ(z)) + 0.5 * mean_neg(-log(1-σ(z)))
               = 0.5 * mean_pos(softplus(-z)) + 0.5 * mean_neg(softplus(z))
    """
    y = labels.to(dtype=logits.dtype)
    pos = (y > 0.5)
    neg = ~pos

    if pos.any():
        pos_loss = F.softplus(-logits[pos]).mean()
    else:
        pos_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    if neg.any():
        neg_loss = F.softplus( logits[neg]).mean()
    else:
        neg_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    return 0.5 * (pos_loss + neg_loss)

def train_one_fold(cfg, fold, idx_tr, idx_va, manifest_path, logger):
    device = cfg["device"]
    bs = cfg["train"]["batch_size"]; epochs = cfg["train"]["epochs"]; warmup = cfg["train"]["warmup_epochs"]
    use_balanced = (cfg.get("sampler", {}) or {}).get("type", "") == "balanced"
    pos_frac = float(cfg.get("sampler", {}).get("pos_fraction", 0.5))

    cat_cols = cfg["data"]["cat_cols"]

    train_ds = ShardedDataset(manifest_path, idx_tr, train=True,  cat_cols=cat_cols)
    val_ds   = ShardedDataset(manifest_path, idx_va, train=True,  cat_cols=cat_cols)

    if use_balanced:
        y_np = train_ds.arrs["y"]
        bsampler = BalancedBatchSampler(y_np, batch_size=bs, pos_fraction=pos_frac, replacement=True, seed=int(cfg.get("seed", 777)), drop_last=True)
        tr_loader = DataLoader(train_ds, batch_size=bs, sampler=bsampler, num_workers=8,  pin_memory=True, persistent_workers=False, collate_fn=collate_sharded)
    else:
        tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8,  pin_memory=True, persistent_workers=False, collate_fn=collate_sharded)
    
    va_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=False, collate_fn=collate_sharded)

    # 차원정보
    with open(manifest_path, "r") as f:
        man = json.load(f)
    # 임의 vocab 상한
    seq_vocab = int(10_000_000)

    # 카디널리티: 해시 버킷과 동일
    cardinals = {c: cfg["data"]["hash_buckets"].get(c, 1000003) + cfg["data"].get("hash_buckets_margin", 0) for c in cat_cols}

    # num/mask 차원은 샤드 파일 0개를 열어 shape를 확인하는 대신 manifest의 첫 샤드를 로드
    first = man["shards"][0]
    X_num_dim = np.load(first["X_num"]["path"], mmap_mode="r").shape[1]
    X_mask_dim= np.load(first["X_mask"]["path"],mmap_mode="r").shape[1]

    model = CTRModel(cfg, seq_vocab, X_num_dim, X_mask_dim, cardinals, cat_cols).to(device)
    ema = None
    if cfg.get("ema", {}).get("enabled", False):
        ema = build_ema(model, cfg)


    # AMP/compile
    use_amp = cfg["amp"] in ["bf16","fp16"]
    amp_dtype = torch.bfloat16 if cfg["amp"]=="bf16" else torch.float16
    if cfg.get("use_compile", False):
        model = torch.compile(model, mode="reduce-overhead")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and cfg["amp"]=="fp16")

    global_step = 0
    best_score, best_state, wait = -1e9, None, 0
    steps_per_epoch = math.ceil(len(tr_loader))

    aux_w = float(cfg["model"]["qnn_alpha"].get("aux_head_weight", 0.0))

    for epoch in range(1, epochs+1):
        model.train()
        tr_losses = []
        
        tqdm_tr = tqdm(tr_loader, desc=f"train[{fold}/{epoch}]", total=len(tr_loader), unit="batch")
        for step, batch in enumerate(tqdm_tr):
            global_step += 1
            y = batch.pop("y").to(device)
            opt.param_groups[0]["lr"] = cosine_warmup_lr(epoch-1, step, steps_per_epoch,
                                                         cfg["train"]["lr"], warmup, epochs)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            else:
                from contextlib import nullcontext; ctx = nullcontext()

            with ctx:
                logits, prob, aux_logit = model(batch)
                loss = bce_wll_style(logits, y)
                if aux_w > 0:
                    aux_loss = bce_wll_style(aux_logit, y)
                    loss = loss + aux_w * aux_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg["train"]["grad_clip_norm"]>0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if cfg["train"]["grad_clip_norm"]>0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                opt.step()

            # ===== EMA UPDATE (after optimizer.step) =====
            if ema is not None:
                ema.update(model, global_step)
            # ============================================

            tr_losses.append(float(loss.detach().cpu()))
            tqdm_tr.set_postfix(loss=np.mean(tr_losses))

        # --- Validation with EMA weights ---
        use_ema_eval = (ema is not None) and cfg["ema"].get("eval_with_ema", True)

        if use_ema_eval:
            ema.store(model)       # 현재(학습) 가중치 저장
            ema.copy_to(model)     # 모델을 EMA 가중치로 덮어쓰기

        # Validation
        model.eval()
        all_p, all_y, all_z = [], [], []
        with torch.no_grad():
            tqdm_va = tqdm(va_loader, desc=f"val[{fold}/{epoch}]", total=len(va_loader), unit="batch")
            for batch in tqdm_va:
                y = batch.pop("y").numpy()
                z, p, aux = model(batch)
                all_p.append(p.detach().cpu().numpy())
                all_y.append(y)
                all_z.append(z.detach().cpu().numpy())
        y_true = np.concatenate(all_y)
        p_raw = np.concatenate(all_p)
        z_raw = np.concatenate(all_z)
        ap, wll, score = final_score(y_true, p_raw)

        # Calibration(Optional)
        cal, ap_cal, wll_cal, score_cal = None, None, None, None
        if cfg["calibration"]["enabled"]:
            # T, p_cal, wll_cal = fit_temperature(z_raw.astype(np.float32), y_true.astype(np.int32),
            #                                     lr=cfg["calibration"]["lr"], iters=cfg["calibration"]["iters"])
            # ap_cal = ap
            # score_cal = 0.5*ap_cal + 0.5*wll_cal
            cal = Calibrator(method=cfg["calibration"].get("method", "temperature"),
                            lr=float(cfg["calibration"].get("lr", 0.05)),
                            iters=int(cfg["calibration"].get("iters", 200)))
            cal.fit(z_raw, y_true)
            p_cal = cal.predict_proba(z_raw)
            ap_cal, wll_cal, score_cal = final_score(y_true, p_cal)

        logger.row(fold=fold, epoch=epoch, split="val",
                   loss=np.mean(tr_losses), AP=round(ap,6), WLL=round(wll,6),
                   Score=round(score,6), lr=opt.param_groups[0]["lr"],
                   bs=cfg["train"]["batch_size"], K=cfg["sequence"]["top_k"], tau=cfg["sequence"]["recency_tau"])
        logger.csv(fold=fold, epoch=epoch, split="val",
                   loss=np.mean(tr_losses), AP=ap, WLL=wll, Score=score,
                   lr=opt.param_groups[0]["lr"], bs=cfg["train"]["batch_size"],
                   K=cfg["sequence"]["top_k"], tau=cfg["sequence"]["recency_tau"])
        logger.scalars(f"fold{fold}", epoch, train_loss=np.mean(tr_losses), val_AP=ap, val_WLL=wll, val_Score=score)

        if cfg["calibration"]["enabled"]:
            logger.row(fold=fold, epoch=epoch, split="val_cal",
                       loss="--", AP=round(ap_cal,6), WLL=round(wll_cal,6),
                       Score=round(score_cal,6), lr=opt.param_groups[0]["lr"],
                       bs=cfg["train"]["batch_size"], K=cfg["sequence"]["top_k"], tau=cfg["sequence"]["recency_tau"])
            logger.scalars(f"fold{fold}", epoch, val_WLL_cal=wll_cal, val_Score_cal=score_cal)

        if use_ema_eval:
            ema.restore(model)     # 원래 가중치로 복구
        
        cur = score_cal if (cfg.get("calibration", {}).get("enabled", False) and score_cal is not None) else score
        if cur > best_score:
            best_score = cur
            best_state = {
                "model": model.state_dict(), 
                "cfg": cfg, 
                "best_score": best_score,
                "epoch": epoch, 
                "calibrator": cal,
                "ema": (ema.state_dict() if ema is not None else None),
                "global_step": global_step,
                } # cal은 None일 수도 있음
            wait = 0
        else:
            wait += 1
            if wait >= cfg["train"]["early_stop_patience"]:
                break

    def _free_fold():
        try:
            # tqdm 핸들러 닫기 (남아있으면 참조 유지될 수 있음)
            try: tqdm_tr.close()
            except: pass
            try: tqdm_va.close()
            except: pass
        except: pass

        # 로더/데이터셋
        for obj in [tr_loader, va_loader, train_ds, val_ds]:
            try: del obj
            except: pass

        # 모델/옵티마/스케일러/EMA
        for obj in [model, opt, scaler, ema]:
            try:
                if obj is None: 
                    continue
                # VRAM 먼저 비우고 싶으면 CPU로 이동
                if hasattr(obj, "to"):
                    try: obj.to("cpu")
                    except: pass
                del obj
            except: pass

        # 큰 임시 배열/버퍼
        for arr in ["y_true", "p_raw", "z_raw", "tr_losses"]:
            if arr in locals():
                try: exec(f"del {arr}")
                except: pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _free_fold()
    return best_state, best_score

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    set_seed(cfg["seed"], deterministic=cfg.get("deterministic", True))
    device = "cuda" if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu"
    cfg["device"] = device

    logger = Logger(os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"]),
                    tb=cfg["logging"]["tb"], csv_log=cfg["logging"]["csv_log"])

    manifest_path = cfg["data"]["manifest_train"]
    # y, groups 로드
    y, groups = load_labels_groups_for_split(manifest_path)
    n_splits = int(cfg["cv"]["n_splits"])
    effective_splits = max(5, n_splits)
    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=cfg["seed"])

    # bests = []
    out_dir = os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"])
    for fold, (tr, va) in enumerate(sgkf.split(np.zeros_like(y), y, groups)):
        if n_splits == 1 and fold > 0:
            break
        ckpt_path = os.path.join(out_dir, f"ckpt_folds_{fold}.pt")
        if os.path.exists(ckpt_path):
            continue
        state, score = train_one_fold(cfg, fold, tr, va, manifest_path, logger)
        torch.save({"state": state, "score": score}, os.path.join(out_dir, f"ckpt_folds_{fold}.pt"))
        # fold별 학습 후 VRAM 점유를 줄이기 위해 불필요한 객체/캐시 삭제
        del state, score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # bests.append((state, score))

    # os.makedirs(out_dir, exist_ok=True)
    # torch.save({"folds": bests}, os.path.join(out_dir, "ckpt_folds.pt"))
    # logger.row(fold="all", epoch="-", split="done", loss="-", AP="-", WLL="-", Score=max(s for _,s in bests),
    #            lr="-", bs="-", K="-", tau="-")

if __name__ == "__main__":
    import argparse 
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
