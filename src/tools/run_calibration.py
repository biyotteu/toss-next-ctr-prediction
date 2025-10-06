# -*- coding: utf-8 -*-
"""
OOF 로짓/라벨로 캘리브레이션만 별도로 학습/저장하는 스크립트.
- OOF 파일(oof_z.npy, oof_y.npy)이 있으면 그대로 사용
- 없으면 ckpt_folds_*.pt 로부터 OOF 로짓을 재계산(검증 셋) 후 사용
- 학습된 보정은 JSON 메타(cal_meta.json)와 선택적으로 pickle(calibrator.pkl)로 저장

사용 예:
  # 1) OOF 파일이 있을 때 (보정만)
  python -m src.tools.run_calibration --cfg config/dare_qnn.yaml \
      --oof-z outputs/oof_z.npy --oof-y outputs/oof_y.npy \
      --method temperature --save-dir runs/exp123/calibration

  # 2) OOF 파일이 없을 때 (체크포인트로 OOF 산출 + 보정)
  python -m src.tools.run_calibration --cfg config/dare_qnn.yaml \
      --compute-oof --save-oof-dir outputs/oof_exp123 \
      --method temperature+isotonic --save-dir runs/exp123/calibration
"""
from __future__ import annotations
import os, json, math, argparse, numpy as np, torch
from glob import glob
from tqdm.auto import tqdm

# 프로젝트 모듈
from src.data.dataset import ShardedDataset, collate_sharded, load_labels_groups_for_split
from src.models.wrapper import CTRModel
from src.utils.metrics import final_score
from src.utils.ema import ModelEMA
from src.utils.seed import set_seed

# 안전 캘리브레이터 (가드/가중치 포함 버전)
from src.utils.calibration import Calibrator

def _sorted_ckpts(run_dir: str) -> list[str]:
    paths = sorted(glob(os.path.join(run_dir, "ckpt_folds_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under: {run_dir}")
    return paths

@torch.no_grad()
def _compute_oof_from_ckpts(cfg: dict,
                            manifest_train: str,
                            save_oof_dir: str,
                            device: str = "cuda") -> tuple[str,str]:
    """
    ckpt_folds_*.pt들과 동일한 SGKF 설정으로 각 fold의 val에 대해 로짓 산출 → OOF 생성.
    메모리 절약을 위해 memmap로 (N,) 배열에 인덱스 위치대로 기록.
    """
    os.makedirs(save_oof_dir, exist_ok=True)

    # 라벨/그룹과 총 행수
    y_all, groups = load_labels_groups_for_split(manifest_train)
    N = int(y_all.shape[0])

    # OOF memmap 생성 (float32/uint8)
    oof_z_path = os.path.join(save_oof_dir, "oof_z.npy")
    oof_y_path = os.path.join(save_oof_dir, "oof_y.npy")
    oof_z = np.memmap(oof_z_path, mode="w+", dtype=np.float32, shape=(N,))
    oof_y = np.memmap(oof_y_path, mode="w+", dtype=np.uint8,  shape=(N,))
    oof_y[:] = 255  # sentinel(아직 미기록)

    # 차원 정보
    with open(manifest_train, "r") as f:
        man = json.load(f)
    first = man["shards"][0]
    X_num_dim = np.load(first["X_num"]["path"], mmap_mode="r").shape[1]
    X_mask_dim= np.load(first["X_mask"]["path"],mmap_mode="r").shape[1]

    # 모델 공통 준비
    seq_vocab = 10_000_000
    cat_cols  = cfg["data"]["cat_cols"]
    cardinals = {c: cfg["data"]["hash_buckets"].get(c, 1000003) + cfg["data"].get("hash_buckets_margin", 0)
                 for c in cat_cols}

    # SGKF를 훈련과 동일하게 재현
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = int(cfg["cv"]["n_splits"])
    effective_splits = max(5, n_splits)  # train.py와 동일
    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=cfg["seed"])

    # 체크포인트 로딩
    run_dir = os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"])
    ckpts = _sorted_ckpts(run_dir)
    print(f"[OOF] using {len(ckpts)} checkpoints.")

    # fold별로 val 인덱스에 예측 기록
    fold_id = 0
    for (tr_idx, va_idx), ckpt_path in tqdm(zip(sgkf.split(np.zeros_like(y_all), y_all, groups),
                                                ckpts),
                                            total=effective_splits, desc="build OOF"):
        if n_splits == 1 and fold_id > 0:
            break
        state, score = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 모델 + EMA 적용
        m = CTRModel(cfg, seq_vocab, X_num_dim, X_mask_dim, cardinals, cat_cols).to(device)
        m.load_state_dict(state["model"], strict=True)
        ema_state = state.get("ema", None)
        if ema_state is not None:
            ema = ModelEMA(m, base_decay=ema_state.get("base_decay", 0.999))
            ema.load_state_dict(ema_state)
            ema.copy_to(m)
            del ema
        m.eval()

        # 해당 fold의 val 서브셋
        ds_va = ShardedDataset(manifest_train, va_idx, train=True, cat_cols=cat_cols)
        bs = int(cfg["train"]["batch_size"])
        dl_va = torch.utils.data.DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=8,
                                            pin_memory=True, persistent_workers=False,
                                            collate_fn=collate_sharded)

        # va_idx에 대한 순차 기록
        offset = 0
        for batch in dl_va:
            y = batch["y"].numpy()      # (B,)
            z, p, aux = m({k:v for k,v in batch.items() if k!="y"})
            z_np = z.detach().float().view(-1).cpu().numpy()
            L = len(z_np)
            sl = slice(offset, offset+L)
            tgt_rows = va_idx[sl]  # 원본 인덱스 위치
            oof_z[tgt_rows] = z_np
            oof_y[tgt_rows] = y
            offset += L

        fold_id += 1
        # 메모리 회수
        del m, state, z, p, aux
        torch.cuda.empty_cache()

    # 기록 안된 sentinel이 없어야 함
    if int((oof_y == 255).sum()) != 0:
        raise RuntimeError("OOF write incomplete: some rows were not filled.")
    # flush
    del oof_z, oof_y

    return oof_z_path, oof_y_path

def _eval_and_log(y: np.ndarray, z: np.ndarray, cal: Calibrator | None):
    from src.utils.metrics import final_score
    # before
    p0 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    ap0, wll0, s0 = final_score(y, p0)
    # after
    if cal is not None:
        p1 = cal.predict_proba(z)
        ap1, wll1, s1 = final_score(y, p1)
    else:
        ap1, wll1, s1 = ap0, wll0, s0
    print(f"[OOF] Before  AP={ap0:.6f}  WLL={wll0:.6f}  Score={s0:.6f}")
    print(f"[OOF] After   AP={ap1:.6f}  WLL={wll1:.6f}  Score={s1:.6f}")
    return {"before": {"AP": ap0, "WLL": wll0, "Score": s0},
            "after":  {"AP": ap1, "WLL": wll1, "Score": s1}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--oof-z", type=str, default=None, help="precomputed OOF logits .npy")
    ap.add_argument("--oof-y", type=str, default=None, help="precomputed OOF labels .npy")
    ap.add_argument("--compute-oof", action="store_true", help="build OOF from ckpts if OOF npy not provided")
    ap.add_argument("--save-oof-dir", type=str, default="outputs/oof", help="where to dump computed OOF npy")
    ap.add_argument("--method", type=str, default="temperature",
                    choices=["temperature","isotonic","temperature+isotonic","platt"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--clamp-T", type=float, nargs=2, default=[0.2, 5.0])
    ap.add_argument("--l2-reg", type=float, default=1e-3)
    ap.add_argument("--min-iso-nodes", type=int, default=8)
    ap.add_argument("--save-dir", type=str, default=None, help="where to save cal_meta.json")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    # cfg
    cfg = json.loads(json.dumps(__import__("yaml").safe_load(open(args.cfg))))
    set_seed(int(cfg.get("seed", args.seed)), deterministic=cfg.get("deterministic", True))
    device = "cuda" if (torch.cuda.is_available() and cfg.get("device","cuda")=="cuda") else "cpu"

    # OOF 준비
    if args.oof_z and args.oof_y:
        oof_z_path, oof_y_path = args.oof_z, args.oof_y
    elif args.compute_oof:
        man_train = cfg["data"]["manifest_train"]
        run_dir   = os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"])
        # ckpt 존재 확인
        _ = _sorted_ckpts(run_dir)
        oof_z_path, oof_y_path = _compute_oof_from_ckpts(cfg, man_train, args.save_oof_dir, device=device)
    else:
        raise ValueError("Provide --oof-z & --oof-y, or set --compute-oof")

    z = np.load(oof_z_path, mmap_mode="r").astype(np.float64)
    y = np.load(oof_y_path, mmap_mode="r").astype(np.int32)
    assert z.shape == y.shape, "OOF shapes mismatch"

    # 보정 학습
    cal = Calibrator(method=args.method, lr=args.lr, iters=args.iters,
                     clamp_T=tuple(args.clamp_T), l2_reg=args.l2_reg,
                     min_iso_nodes=args.min_iso_nodes)
    cal.fit(z, y)

    # 평가 로그
    metrics = _eval_and_log(y, z, cal)

    # 저장 폴더
    save_dir = args.save_dir or os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"], "calibration")
    os.makedirs(save_dir, exist_ok=True)

    # 메타 저장 (JSON)
    meta = {"method": args.method, "clamp_T": list(args.clamp_T),
            "l2_reg": args.l2_reg, "min_iso_nodes": args.min_iso_nodes}
    if cal.temp_scaler is not None:
        T = float(torch.exp(cal.temp_scaler.log_temp).item())
        # 안전 클램프 반영
        T = float(np.clip(T, args.clamp_T[0], args.clamp_T[1]))
        meta["T"] = T
    if cal.iso is not None:
        # piecewise 를 재현할 수 있도록 threshold/value 저장
        try:
            meta["iso_x"] = cal.iso.X_thresholds_.tolist()
            meta["iso_y"] = cal.iso.y_thresholds_.tolist()
        except Exception:
            pass

    with open(os.path.join(save_dir, "cal_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(save_dir, "oof_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 원하면 객체도 저장(선택): 피클 의존성을 싫어하면 생략 가능
    try:
        import pickle
        with open(os.path.join(save_dir, "calibrator.pkl"), "wb") as f:
            pickle.dump(cal, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

    print(f"[✓] Saved cal_meta.json to {save_dir}")
    print(f"[✓] OOF before/after metrics saved to {save_dir}/oof_metrics.json")

if __name__ == "__main__":
    main()
