import numpy as np
from sklearn.metrics import average_precision_score

EPS = 1e-15

def average_precision(y_true, y_prob):
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, y_prob))

# Weighted LogLoss: 50% positives + 50% negatives
# WLL = 0.5*mean(-log p | y=1) + 0.5*mean(-log (1-p) | y=0)

def weighted_logloss(y_true, y_prob):
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    # 더 강하게 클리핑
    p = np.clip(p, EPS, 1.0 - EPS)

    pos = (y_true == 1)
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        # 수치안정 버전
        return float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log1p(-p)))

    l_pos = -np.log(p[pos]).mean()
    l_neg = -np.log1p(-p[neg]).mean()  # log(1-p) 대신 log1p(-p)
    return float(0.5*l_pos + 0.5*l_neg)

# Combined leaderboard metric (optional):
# 0.5*AP + 0.5*(1 - normalized WLL). We keep AP/WLL separate in logs.