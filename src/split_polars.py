import numpy as np
import polars as pl


def _stratified_split_pl(df: pl.DataFrame, label_col: str, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    # compute per-class indices
    labels = df.select(pl.col(label_col).cast(pl.Int32)).to_series().to_numpy()
    idx = np.arange(df.height)
    tr_idx, va_idx = [], []
    for y in np.unique(labels):
        cls_idx = idx[labels == y]
        rng.shuffle(cls_idx)
        cut = int(len(cls_idx) * (1 - val_ratio))
        tr_idx.append(cls_idx[:cut])
        va_idx.append(cls_idx[cut:])
    tr_idx = np.concatenate(tr_idx) if tr_idx else np.array([], dtype=int)
    va_idx = np.concatenate(va_idx) if va_idx else np.array([], dtype=int)
    return df.gather(tr_idx), df.gather(va_idx)


def _group_split_pl(df: pl.DataFrame, label_col: str, group_col: str, val_ratio: float, seed: int):
    if group_col not in df.columns:
        return _stratified_split_pl(df, label_col, val_ratio, seed)
    rng = np.random.default_rng(seed)
    groups = df.select(pl.col(group_col)).to_series().to_numpy()
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    cut = int(len(uniq) * (1 - val_ratio))
    tr_groups = set(uniq[:cut])
    tr_mask = np.isin(groups, list(tr_groups))
    va_mask = ~tr_mask
    tr_idx = np.nonzero(tr_mask)[0]
    va_idx = np.nonzero(va_mask)[0]
    return df.gather(tr_idx), df.gather(va_idx)


def _time_split_pl(df: pl.DataFrame, label_col: str, time_col: str, val_ratio: float):
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in DataFrame for time-based split")
    ts = df.select(pl.col(time_col)).to_series().to_numpy()
    order = np.argsort(ts)
    cut = int(df.height * (1 - val_ratio))
    tr_idx = order[:cut]
    va_idx = order[cut:]
    return df.gather(tr_idx), df.gather(va_idx)


def leakage_free_split_pl(df: pl.DataFrame, cfg):
    method = cfg.split.method
    val_ratio = cfg.split.time_val_ratio if method == 'time' else cfg.val_ratio
    if method == 'group_user':
        return _group_split_pl(df, cfg.label_col, cfg.split.group_key, val_ratio, cfg.seed)
    elif method == 'group_session':
        return _group_split_pl(df, cfg.label_col, cfg.split.session_key, val_ratio, cfg.seed)
    elif method == 'time':
        return _time_split_pl(df, cfg.label_col, cfg.split.time_col, val_ratio)
    else:
        return _stratified_split_pl(df, cfg.label_col, val_ratio, cfg.seed)


