import numpy as np
import torch
from sklearn.metrics import average_precision_score

def weighted_logloss_50_50(y_true: np.ndarray, y_prob: np.ndarray, eps=1e-12):
    y_true = y_true.astype(np.float64, copy=False)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob.astype(np.float64, copy=False), eps, 1 - eps)
    pos = y_true == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        # 극단 방어
        return float("nan")
    pos_loss = -np.log(y_prob[pos]).mean()
    neg_loss = -np.log(1.0 - y_prob[neg]).mean()
    return 0.5 * (pos_loss + neg_loss)

def ap_score(y_true: np.ndarray, y_prob: np.ndarray):
    if y_true.mean() in [0.0, 1.0] or len(np.unique(y_true)) < 2:
        return 0.0

    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 1e-12, 1-1e-12)
    return float(average_precision_score(y_true, y_prob))

def final_score(y_true, y_prob):
    ap = ap_score(y_true, y_prob)
    wll = weighted_logloss_50_50(y_true, y_prob)
    return ap, wll, 0.5 * ap + 0.5 * wll


def _safe_prob_to_logit(p, eps=1e-7):
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

def _rank_avg_stack(p_list):
    # p_list: list[tensor (B,)]
    # 각 모델별로 배치 차원에서 순위화 → [0,1] 정규화 → 평균 → 그 자체를 확률로 사용(단순)
    ranks = []
    for p in p_list:
        # argsort 두 번으로 rank (작을수록 낮은 랭크)
        order = torch.argsort(p)
        r = torch.zeros_like(p).scatter_(0, order, torch.arange(p.numel(), device=p.device))
        r = (r + 1).float() / (p.numel() + 1.0)
        ranks.append(r)
    return torch.stack(ranks, 0).mean(0)

def ensemble_probs(method, p_list, z_list=None, weights=None, trim_ratio=0.0):
    """
    method: mean | logit_mean | geom_mean | median | trim_mean | rank_avg | weighted
    p_list: list[tensor(B,)]
    z_list: list[tensor(B,)]  # logits (optional, stacked/미래 확장용)
    weights: tensor(M,) or None
    """
    M = len(p_list)
    P = torch.stack(p_list, 0)    # (M,B)
    if weights is not None:
        w = weights / weights.sum()
    else:
        w = None

    if method == "mean":
        return (P.mean(0) if w is None else (P * w.view(-1,1)).sum(0))
    if method == "geom_mean":
        Pc = torch.clamp(P, 1e-7, 1-1e-7)
        logP = torch.log(Pc)
        return torch.exp(logP.mean(0) if w is None else (logP * w.view(-1,1)).sum(0))
    if method == "logit_mean":
        L = _safe_prob_to_logit(P)
        Lm = (L.mean(0) if w is None else (L * w.view(-1,1)).sum(0))
        return torch.sigmoid(Lm)
    if method == "median":
        return torch.median(P, 0).values
    if method == "trim_mean":
        k = int(max(0, min(P.shape[0]//2, round(P.shape[0]*trim_ratio))))
        if k == 0:
            return P.mean(0)
        # 모델 축 정렬 후 가장 낮은 k, 높은 k 잘라 평균
        Ps, _ = torch.sort(P, dim=0)
        return Ps[k: M-k].mean(0)
    if method == "rank_avg":
        return _rank_avg_stack([p for p in p_list])
    if method == "weighted":
        assert w is not None, "weights required for method='weighted'"
        return (P * w.view(-1,1)).sum(0)
    raise ValueError(f"Unknown ensemble method: {method}")