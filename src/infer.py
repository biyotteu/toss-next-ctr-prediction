import os, yaml, json, numpy as np, torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from glob import glob
from src.data.dataset import ShardedDataset, collate_sharded
from src.models.wrapper import CTRModel
from src.utils.metrics import ensemble_probs
from src.utils.ema import ModelEMA

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu"

    man_path = cfg["data"]["manifest_test"]
    with open(man_path, "r") as f:
        man = json.load(f)

    print("ensemble method: ", cfg["ensemble"]["method"])

    N = man["rows"]
    idx = np.arange(N, dtype=np.int64)

    ds = ShardedDataset(man_path, idx, train=False, cat_cols=cfg["data"]["cat_cols"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=8,
                    pin_memory=True, persistent_workers=False, collate_fn=collate_sharded)

    ckpt_paths = sorted(glob(os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"], "ckpt_folds_*.pt")))
    assert ckpt_paths, "No checkpoints found"

    entries = []
    for ckpt_path in tqdm(ckpt_paths, desc="load checkpoints"):
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # weights_only 인자 빼도 무방
        # 1) (state, score) 튜플 포맷
        if isinstance(obj, tuple) and len(obj) == 2:
            state, score = obj
            entries.append((state, float(score) if score is not None else -1.0))
            continue

        # 2) 단일 dict 포맷 (state 자체가 dict이거나, {"state":..., "score":...})
        if isinstance(obj, dict) and "folds" not in obj:
            # 예: {"model":..., "best_score":...}  또는 {"state":..., "score":...}
            state = obj.get("state", obj)  # "state" 키가 없으면 obj 자체가 state
            score = obj.get("best_score", obj.get("score", -1.0))
            # 최소 키 체크
            if "model" not in state:
                raise KeyError(f"Checkpoint {ckpt_path} has no 'model' key in state")
            entries.append((state, float(score)))
            continue

        # 3) 합본 포맷 {"folds": [...]}
        if isinstance(obj, dict) and "folds" in obj:
            for item in obj["folds"]:
                if isinstance(item, tuple) and len(item) == 2:
                    s, sc = item
                    entries.append((s, float(sc) if sc is not None else -1.0))
                elif isinstance(item, dict):
                    s = item.get("state", item)
                    sc = item.get("best_score", item.get("score", -1.0))
                    if "model" not in s:
                        raise KeyError(f"Combined checkpoint entry has no 'model' key")
                    entries.append((s, float(sc)))
                else:
                    raise TypeError(f"Unknown entry type inside 'folds' for {ckpt_path}: {type(item)}")
            continue

        raise TypeError(f"Unknown checkpoint format for {ckpt_path}: {type(obj)}")
    # for ckpt_path in tqdm(ckpt_paths, desc="load checkpoints"):
    #     state, score = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    #     entries.append((state, score))
    # folds = torch.load(os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"], "ckpt_folds.pt"), map_location="cpu", weights_only=False)
    # entries = folds["folds"]
    # 차원
    first = man["shards"][0]
    X_num_dim = np.load(first["X_num"]["path"], mmap_mode="r").shape[1]
    X_mask_dim= np.load(first["X_mask"]["path"],mmap_mode="r").shape[1]
    seq_vocab = 10_000_000
    cat_cols = cfg["data"]["cat_cols"]
    cardinals = {c: cfg["data"]["hash_buckets"].get(c, 1000003) + cfg["data"].get("hash_buckets_margin", 0) for c in cat_cols}

    models, calibrators  = [], [] # emas =  []
    for state, _ in entries:
        ema = None
        m = CTRModel(cfg, seq_vocab, X_num_dim, X_mask_dim, cardinals, cat_cols).to(device)
        m.load_state_dict(state["model"], strict=True)
        m.eval()

        ema_state = state.get("ema", None)
        if ema_state is not None:
            ema = ModelEMA(m, base_decay=ema_state.get("base_decay", 0.999))
            ema.load_state_dict(ema_state)
            ema.copy_to(m)     # 모델 가중치를 EMA로 덮기
            del ema
        
        models.append(m)
        calibrators.append(state.get("calibrator", None))
        # if ema is not None:
        #     emas.append(ema)
        # Ts.append(state.get("T", None))

    preds, id_list = [], []
    with torch.no_grad():
        tqdm_dl = tqdm(dl, desc="infer", total=len(dl), unit="batch")
        for batch in tqdm_dl:
            ids = batch.pop("ids")
            z_list, p_list = [], []
            for i, m in enumerate(models):
                z, p, aux = m(batch)
                cal = calibrators[i]

                # cal = None
                if cfg["calibration"]["enabled"] and cal is not None:
                    # Calibrator는 CPU/NumPy 기반 → 배치 단위로 logits만 CPU로 보냄
                    z_np = z.detach().float().view(-1).cpu().numpy()
                    p_np = cal.predict_proba(z_np)                # shape: (B,)
                    p = torch.from_numpy(p_np).to(z.device).view_as(z).float()

                # if cfg["calibration"]["lambda_mix"] > 0.0:
                #     p = cfg["calibration"]["lambda_mix"] * p + (1 - cfg["calibration"]["lambda_mix"]) * p_np
                
                # 안전 클리핑
                p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
                z_list.append(z)
                p_list.append(p)
            # 단일 모델일 때는 곧바로 해당 확률 사용(모든 앙상블 설정 무시)
            if len(p_list) == 1:
                p_ens = p_list[0].view(-1)
            else:
                # ==== 여기부터: 다양한 앙상블 ====
                ens_cfg = cfg.get("ensemble", {}) or {}
                method   = ens_cfg.get("method", "logit_mean")
                trim_r   = float(ens_cfg.get("trim_ratio", 0.0))
                weights  = None

                if method == "val_weighted":
                    # ckpt의 fold별 점수로 가중치 생성(softmax)
                    scores = []
                    for it in entries:
                        if isinstance(it, dict) and "score" in it:
                            scores.append(float(it["score"]))
                        else:
                            # 구포맷(tuple)일 때 두 번째 원소가 score
                            scores.append(float(it[1]))
                    temp = float(ens_cfg.get("val_weight_temperature", 10.0))
                    s = torch.tensor(scores, dtype=torch.float32, device=p_list[0].device)
                    w = torch.softmax(s / max(1e-6, temp), dim=0)
                    weights = w
                    # val_weighted은 확률 가중 평균을 기본으로, 원하면 logit_mean으로 바꿔도 됨
                    method_use = "weighted"
                elif method == "weighted":
                    w_cfg = ens_cfg.get("weights", [])
                    assert len(w_cfg) == len(p_list), "weights length must match #folds/models"
                    weights = torch.tensor(w_cfg, dtype=torch.float32, device=p_list[0].device)
                    method_use = "weighted"
                else:
                    method_use = method

                p_ens = ensemble_probs(method_use, p_list, z_list=z_list, weights=weights, trim_ratio=trim_r)
                # ==== 여기까지 ====

            preds.append(p_ens.cpu().numpy())
            id_list.append(ids)

    p = np.concatenate(preds)
    ids = np.concatenate(id_list)
    out_dir = os.path.join(cfg["logging"]["log_dir"], cfg["exp_name"])
    os.makedirs(out_dir, exist_ok=True)
    # 혼합 dtype 안전 저장(문자열 ID + float 확률)
    rows = np.empty((ids.shape[0], 2), dtype=object)
    rows[:, 0] = ids.astype(str)
    rows[:, 1] = p.astype(np.float64)
    np.savetxt(os.path.join(out_dir, "submission.csv"),
               rows, delimiter=",", header="ID,clicked", comments="", fmt=["%s","%.8f"])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
