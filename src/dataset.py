import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .utils import hash64
from .data_schema import CAT_PATTERNS, EXCLUDE_COLS, match_any


# =========================
# Parquet lightweight frame
# =========================
class CTRFrame:
    """Lightweight loader around parquet with column pruning."""
    def __init__(self, parquet_path: str, drop_cols: List[str]):
        self.pf = pq.ParquetFile(parquet_path)
        self.drop_cols = set(drop_cols)
        schema = self.pf.schema_arrow
        self.all_cols = [n for n in schema.names if n not in self.drop_cols]

    def read_all(self, columns: List[str] = None) -> pd.DataFrame:
        cols = columns or self.all_cols
        table = self.pf.read(columns=cols)
        return table.to_pandas(types_mapper=pd.ArrowDtype)

    def read_row_group(self, rg_idx: int, columns: List[str] = None) -> pd.DataFrame:
        cols = columns or self.all_cols
        table = self.pf.read_row_group(rg_idx, columns=cols)
        return table.to_pandas(types_mapper=pd.ArrowDtype)

    @property
    def num_row_groups(self):
        return self.pf.num_row_groups


# ======================
# Feature type inference
# ======================
def infer_feature_types(df: pd.DataFrame, label_col: str, seq_col: str):
    cats, nums = [], []
    for c in df.columns:
        if c in EXCLUDE_COLS or c == label_col or c == seq_col:
            continue
        if df[c].dtype == 'object' or any(match_any(c, CAT_PATTERNS)):
            cats.append(c)
        else:
            nums.append(c)
    return cats, nums


# ======================
# Sequence parsing utils
# ======================
def parse_seq_col(s: str, max_len: int) -> List[str]:
    if not isinstance(s, str):
        return []
    toks = [t.strip() for t in s.split(',') if t.strip() != ""]
    if max_len > 0:
        toks = toks[-max_len:]
    return toks


def _seq_worker(args):
    """MP worker: parse+hash a chunk of sequence strings."""
    arr, maxL, vocab = args  # arr: list[str|NaN]
    out = []
    for s in arr:
        if not isinstance(s, str):
            out.append([0]); continue
        toks = [t.strip() for t in s.split(',') if t.strip() != ""]
        if maxL > 0:
            toks = toks[-maxL:]
        if not toks:
            out.append([0]); continue
        ids = [1 + hash64(t, vocab - 1) for t in toks]
        out.append(ids)
    return out  # list[list[int]]


# ======================
# Categorical hash utils
# ======================
def _hash_series_fnv_parallel(s: pd.Series, buckets: int, workers: int, progress: bool, chunk_rows: int) -> np.ndarray:
    """Keep EXACT same mapping as hash64(FNV). Parallelized over chunks (Python-level)."""
    s = s.astype("string").fillna("")
    chunks = [s.iloc[i:i+chunk_rows] for i in range(0, len(s), chunk_rows)]
    out = [None] * len(chunks)

    if len(chunks) == 1 or workers <= 1:
        for i, ch in enumerate(tqdm(chunks, desc="cache: cats(fnv)", disable=not progress)):
            out[i] = ch.apply(lambda x: 0 if x == "" else 1 + hash64(x, buckets - 1)).to_numpy(np.int64)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(lambda ser: ser.apply(lambda x: 0 if x == "" else 1 + hash64(x, buckets - 1)).to_numpy(np.int64), ch): i
                    for i, ch in enumerate(chunks)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: cats(fnv)", disable=not progress):
                out[futs[fut]] = fut.result()
    return np.concatenate(out) if len(out) > 1 else out[0]


def _hash_series_pandas(s: pd.Series, buckets: int) -> np.ndarray:
    """Very fast vectorized hash via pandas (SipHash64). Mapping differs from FNV."""
    from pandas.util import hash_pandas_object
    if buckets <= 1 or len(s) == 0:
        return np.zeros(len(s), dtype=np.int64)
    hv = hash_pandas_object(s.astype("string").fillna(""), index=False).values  # uint64
    return (hv % (buckets - 1)).astype(np.int64) + 1


# ===========
# CTR Dataset
# ===========
class CTRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, cats: List[str], nums: List[str], is_train: bool):
        self.df = df
        self.cfg = cfg
        self.cats = cats
        self.nums = nums
        self.is_train = is_train

        # numeric stats
        stats = {}
        if is_train:
            for c in nums:
                v = pd.to_numeric(df[c], errors='coerce')
                mu, sig = float(v.mean()), float(v.std(ddof=0))
                if sig == 0 or np.isnan(sig): sig = 1.0
                stats[c] = (mu, sig)
            self.stats = stats
        else:
            self.stats = getattr(cfg, 'num_stats', {})

        # cache toggles
        self.cache = bool(getattr(cfg, "dataset_cache", False))
        if self.cache:
            progress = bool(getattr(cfg, "dataset_cache_progress", True))
            workers = int(getattr(cfg, "dataset_cache_workers", max(1, (os.cpu_count() or 2)//2)))
            chunk_rows = int(getattr(cfg, "dataset_cache_chunk_rows", 200_000))
            hash_mode = str(getattr(cfg, "dataset_cache_hash", "fnv")).lower()  # "fnv" | "pandas"

            # ---------- Categorical (parallel or vectorized) ----------
            B = self.cfg.category_hash_buckets
            cat_blocks = []
            for c in tqdm(self.cats, desc=f"cache: cats[{hash_mode}]", disable=not progress):
                if hash_mode == "pandas":
                    ids = _hash_series_pandas(df[c], B)
                else:
                    ids = _hash_series_fnv_parallel(df[c], B, workers, progress=False, chunk_rows=chunk_rows)
                cat_blocks.append(ids)
            cat_mat = np.stack(cat_blocks, axis=1) if cat_blocks else np.zeros((len(df), 0), dtype=np.int64)
            self._cats = torch.tensor(cat_mat, dtype=torch.long)

            # ---------- Numeric (vectorized) ----------
            xs = []
            for c in tqdm(self.nums, desc="cache: nums", disable=not progress):
                v = pd.to_numeric(df[c], errors='coerce').fillna(self.cfg.numeric_fillna).to_numpy(np.float32)
                mu, sig = self.stats.get(c, (0.0, 1.0))
                xs.append((v - mu) / (sig if sig else 1.0))
            num_mat = np.stack(xs, axis=1) if xs else np.zeros((len(df), 0), dtype=np.float32)
            self._nums = torch.tensor(num_mat, dtype=torch.float32)

            # ---------- Sequence (multiprocessing over chunks) ----------
            self._seqs = []
            vocab = self.cfg.seq_vocab_size
            maxL = self.cfg.seq_max_len
            seq_ser = df[self.cfg.seq_col].astype("string") if self.cfg.seq_col in df.columns else pd.Series([""] * len(df))
            chunks = [seq_ser.iloc[i:i+chunk_rows].tolist() for i in range(0, len(seq_ser), chunk_rows)]
            out_lists = [None] * len(chunks)

            if len(chunks) == 1 or workers == 1:
                for ci, chunk in enumerate(tqdm(chunks, desc="cache: seq", disable=not progress)):
                    out_lists[ci] = _seq_worker((chunk, maxL, vocab))
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_seq_worker, (chunks[i], maxL, vocab)): i for i in range(len(chunks))}
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq", disable=not progress):
                        out_lists[futs[fut]] = fut.result()
            for ids in (x for chunk in out_lists for x in chunk):
                self._seqs.append(torch.tensor(ids if ids else [0], dtype=torch.long))

            # ---------- Target id (same hash choice as cats) ----------
            tgt_col = self.cfg.target_feature if self.cfg.target_feature in df.columns else 'inventory_id'
            if tgt_col in df.columns:
                if hash_mode == "pandas":
                    t_ids = _hash_series_pandas(df[tgt_col], B)
                else:
                    t_ids = _hash_series_fnv_parallel(df[tgt_col], B, workers, progress=False, chunk_rows=chunk_rows)
            else:
                t_ids = np.zeros(len(df), dtype=np.int64)
            self._tgt = torch.tensor(t_ids, dtype=torch.long)

            # ---------- Label ----------
            if self.cfg.label_col in df.columns:
                self._labels = torch.tensor(df[self.cfg.label_col].to_numpy(np.float32), dtype=torch.float32)
            else:
                self._labels = torch.tensor(np.full(len(df), -1.0, np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    # Fallback encoders (no-cache path)
    def _encode_cats(self, row) -> torch.Tensor:
        vals = []
        B = self.cfg.category_hash_buckets
        for c in self.cats:
            v = row[c]
            if pd.isna(v):
                vals.append(0)
            else:
                vals.append(1 + hash64(str(v), B-1))
        return torch.tensor(vals, dtype=torch.long)

    def _encode_nums(self, row) -> torch.Tensor:
        xs = []
        for c in self.nums:
            v = pd.to_numeric(row[c], errors='coerce')
            mu, sig = self.stats.get(c, (0.0, 1.0))
            if pd.isna(v):
                v = self.cfg.numeric_fillna
            x = (float(v) - mu) / (sig if sig else 1.0)
            xs.append(x)
        return torch.tensor(xs, dtype=torch.float32)

    def _encode_seq(self, s) -> torch.Tensor:
        toks = parse_seq_col(s, self.cfg.seq_max_len)
        ids = [1 + hash64(tok, self.cfg.seq_vocab_size-1) for tok in toks]
        if len(ids) == 0:
            return torch.zeros(1, dtype=torch.long)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        if self.cache:
            return self._cats[idx], self._nums[idx], self._seqs[idx], self._tgt[idx], self._labels[idx]
        # fallback (no-cache)
        row = self.df.iloc[idx]
        cats = self._encode_cats(row)
        nums = self._encode_nums(row)
        seq = self._encode_seq(row.get(self.cfg.seq_col) if self.cfg.seq_col in self.df.columns else None)
        tgt_val = row[self.cfg.target_feature] if self.cfg.target_feature in self.df.columns else row.get('inventory_id')
        tgt_id = 1 + hash64(str(tgt_val), self.cfg.category_hash_buckets-1) if pd.notna(tgt_val) else 0
        label = float(row[self.cfg.label_col]) if self.cfg.label_col in self.df.columns else -1.0
        return cats, nums, seq, int(tgt_id), label


# ============================
# Collate & Dataloader builders
# ============================
def _pad_sequences(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(s.shape[0] for s in seqs)
    out = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, -len(s):] = s
    return out

def collate_fn_train(batch, cfg):
    cats, nums, seqs, tgt_ids, labels = zip(*batch)
    cats = torch.stack(cats)
    nums = torch.stack(nums) if len(nums[0]) > 0 else torch.zeros(len(batch), 0, dtype=torch.float32)
    seq_pad = _pad_sequences(seqs)
    tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    # in-batch negative downsampling (kept unbiased via weights in train loop)
    if cfg.use_neg_downsampling and (labels >= 0).all():
        pos_idx = (labels == 1)
        neg_idx = (labels == 0)
        keep = pos_idx.clone()
        neg = torch.nonzero(neg_idx).flatten()
        if len(neg) > 0:
            k = max(int(len(neg) * cfg.neg_downsample_ratio), 1)
            perm = torch.randperm(len(neg))[:k]
            keep[neg[perm]] = True
        if pos_idx.sum() < cfg.min_pos_per_batch and pos_idx.sum() > 0:
            keep |= pos_idx
        cats, nums, seq_pad, tgt_ids, labels = cats[keep], nums[keep], seq_pad[keep], tgt_ids[keep], labels[keep]
    return cats, nums, seq_pad, tgt_ids, labels

def collate_fn_eval(batch, cfg):
    cats, nums, seqs, tgt_ids, labels = zip(*batch)
    cats = torch.stack(cats)
    nums = torch.stack(nums) if len(nums[0]) > 0 else torch.zeros(len(batch), 0, dtype=torch.float32)
    seq_pad = _pad_sequences(seqs)
    tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return cats, nums, seq_pad, tgt_ids, labels

def make_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg) -> Tuple[DataLoader, DataLoader, dict]:
    cats, nums = infer_feature_types(train_df, cfg.label_col, cfg.seq_col)
    cats = [c for c in cats if c not in set(cfg.force_drop_cols)]
    nums = [c for c in nums if c not in set(cfg.force_drop_cols)]

    train_ds = CTRDataset(train_df, cfg, cats, nums, is_train=True)
    cfg.d['num_stats'] = train_ds.stats  # pass stats to val/test
    val_ds = CTRDataset(val_df, cfg, cats, nums, is_train=False)

    # DataLoader perf flags from config (Linux OK)
    pw = bool(getattr(cfg, "persistent_workers", True)) and getattr(cfg, "num_workers", 0) > 0
    pf = getattr(cfg, "prefetch_factor", 8) if getattr(cfg, "num_workers", 0) > 0 else None
    pin = bool(getattr(cfg, "pin_memory", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        collate_fn=lambda b: collate_fn_train(b, cfg),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        collate_fn=lambda b: collate_fn_eval(b, cfg),
        drop_last=False,
    )

    input_dims = {
        'n_cat': len(cats),
        'n_num': len(nums),
        'seq_vocab_size': cfg.seq_vocab_size,
        'cat_buckets': cfg.category_hash_buckets,
    }
    return train_loader, val_loader, input_dims
