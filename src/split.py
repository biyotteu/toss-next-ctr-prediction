import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


def _stratified_split(df, label_col: str, val_ratio: float, seed: int):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_tr, idx_va = next(sss.split(np.zeros(len(df)), df[label_col].astype(int)))
    return df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_va].reset_index(drop=True)


def _group_split(df, label_col: str, group_col: str, val_ratio: float, seed: int):
    if group_col not in df.columns:
        # fallback to stratified
        return _stratified_split(df, label_col, val_ratio, seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    groups = df[group_col]
    idx_tr, idx_va = next(gss.split(df, df[label_col].astype(int), groups=groups))
    return df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_va].reset_index(drop=True)


def _time_split(df, label_col: str, time_col: str, val_ratio: float):
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in DataFrame for time-based split")
    ts = pd.to_datetime(df[time_col])
    order = np.argsort(ts.values)
    cut = int(len(df) * (1 - val_ratio))
    idx_tr = order[:cut]
    idx_va = order[cut:]
    return df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_va].reset_index(drop=True)


def leakage_free_split(df: pd.DataFrame, cfg):
    method = cfg.split.method
    val_ratio = cfg.split.time_val_ratio if method == 'time' else cfg.val_ratio
    if method == 'group_user':
        return _group_split(df, cfg.label_col, cfg.split.group_key, val_ratio, cfg.seed)
    elif method == 'group_session':
        return _group_split(df, cfg.label_col, cfg.split.session_key, val_ratio, cfg.seed)
    elif method == 'time':
        return _time_split(df, cfg.label_col, cfg.split.time_col, val_ratio)
    else:
        return _stratified_split(df, cfg.label_col, val_ratio, cfg.seed)