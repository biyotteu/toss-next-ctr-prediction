# src/dataset.py
import os
import gc
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from numpy.lib.format import open_memmap

from .utils import hash64
from .data_schema import CAT_PATTERNS, EXCLUDE_COLS, match_any


# ----------------------------
# Parquet lightweight wrapper
# ----------------------------
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
        return table.to_pandas()

    def read_row_group(self, rg_idx: int, columns: List[str] = None) -> pd.DataFrame:
        cols = columns or self.all_cols
        table = self.pf.read_row_group(rg_idx, columns=cols)
        return table.to_pandas()

    @property
    def num_row_groups(self):
        return self.pf.num_row_groups


# ----------------------------
# Feature inference
# ----------------------------
def infer_feature_types(df: pd.DataFrame, label_col: str, seq_col: str):
    cats, nums = [], []
    for c in df.columns:
        if c in EXCLUDE_COLS or c == label_col or c == seq_col:
            continue
        if df[c].dtype == 'object' or match_any(c, CAT_PATTERNS):
            cats.append(c)
        else:
            nums.append(c)
    return cats, nums


# ----------------------------
# Hash helpers
# ----------------------------
def _hash_list_fnv_chunk(lst, buckets: int) -> np.ndarray:
    """Pickle-safe worker: list[str] -> np.int64 hashed ids (FNV mapping)."""
    out = np.empty(len(lst), dtype=np.int64)
    Bm1 = max(buckets - 1, 1)
    for i, x in enumerate(lst):
        s = "" if x is None else str(x)
        out[i] = 0 if s == "" else (1 + hash64(s, Bm1))
    return out

def _hash_series_pandas(s: pd.Series, buckets: int) -> np.ndarray:
    """Very fast vectorized hash (SipHash via pandas). Mapping differs from FNV."""
    from pandas.util import hash_pandas_object
    if buckets <= 1 or len(s) == 0:
        return np.zeros(len(s), dtype=np.int64)
    hv = hash_pandas_object(s.astype("string").fillna(""), index=False).values
    return (hv % (buckets - 1)).astype(np.int64) + 1


def _hash_series_fnv_parallel(
    s: pd.Series, buckets: int, workers: int, chunk_rows: int, progress: bool
) -> np.ndarray:
    """Keep EXACT same mapping as hash64(FNV). Parallelized over chunks."""
    s = s.astype("string").fillna("")
    # 워커에 pandas 객체 대신 list[str]만 전달 (pickle-friendly)
    chunks = [s.iloc[i:i + chunk_rows].astype("string").fillna("").tolist()
              for i in range(0, len(s), chunk_rows)]
    out = [None] * len(chunks)

    if len(chunks) == 1 or workers <= 1:
        for i, ch in enumerate(tqdm(chunks, desc="cache: cats(fnv)", disable=not progress)):
            out[i] = _hash_list_fnv_chunk(ch, buckets)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_hash_list_fnv_chunk, ch, buckets): i
                    for i, ch in enumerate(chunks)}
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="cache: cats(fnv)", disable=not progress):
                out[futs[fut]] = fut.result()

    return np.concatenate(out) if len(out) > 1 else out[0]



# ----------------------------
# Sequence workers (per-chunk)
# ----------------------------
def _seq_parse_chunk(arr, maxL: int, vocab: int):
    """
    Returns:
      lengths: np.int32 [n_rows]
      flat    : np.int32 [sum(lengths)]
    """
    lengths = []
    flat = []
    for s in arr:
        if not isinstance(s, str):
            lengths.append(1)
            flat.append(0)
            continue
        toks = [t.strip() for t in s.split(',') if t.strip() != ""]
        if maxL > 0:
            toks = toks[-maxL:]
        if not toks:
            lengths.append(1)
            flat.append(0)
            continue
        ids = [1 + hash64(t, vocab - 1) for t in toks]
        lengths.append(len(ids))
        flat.extend(ids)
    return (np.asarray(lengths, dtype=np.int32), np.asarray(flat, dtype=np.int32))


# ----------------------------
# CTR Dataset (with cache)
# ----------------------------
class CTRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, cats: List[str], nums: List[str], is_train: bool, partition: Optional[str] = None):
        self.df = df
        self.cfg = cfg
        self.cats = cats
        self.nums = nums
        self.is_train = is_train
        # partition tag: explicit > inferred
        if partition is not None:
            self.partition = partition
        else:
            if is_train:
                self.partition = "train"
            else:
                self.partition = "val" if cfg.label_col in df.columns else "test"

        # numeric z-scoring stats: use precomputed stats from cfg
        # (stats must be provided via cfg.num_stats; no computation here)
        raw_stats = getattr(cfg, 'num_stats', {})
        # normalize format: accept {col: (mu, sig)} or {col: [mu, sig]} or {col: {"mu":, "sig":}}
        stats = {}
        for c, v in raw_stats.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                mu, sig = float(v[0]), float(v[1])
            elif isinstance(v, dict) and 'mu' in v and 'sig' in v:
                mu, sig = float(v['mu']), float(v['sig'])
            else:
                continue
            if sig == 0.0 or np.isnan(sig):
                sig = 1.0
            stats[c] = (mu, sig)
        self.stats = stats

        # caching toggles
        self.cache = bool(getattr(cfg, "dataset_cache", False))
        self.cache_backend = str(getattr(cfg, "dataset_cache_backend", "memmap")).lower()  # "memmap" | "ram"
        self.cache_dir = str(getattr(cfg, "dataset_cache_dir", "./artifacts/cache_seq"))
        self.progress = bool(getattr(cfg, "dataset_cache_progress", True))
        self.workers = int(getattr(cfg, "dataset_cache_workers", max(1, (os.cpu_count() or 2) // 2)))
        print(f"self.workers: {self.workers}")
        self.chunk_rows = int(getattr(cfg, "dataset_cache_chunk_rows", 200_000))
        self.hash_mode = str(getattr(cfg, "dataset_cache_hash", "fnv")).lower()  # "fnv" | "pandas"

        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._build_cache(df)

    def __len__(self):
        return len(self.df)

    # --------- cache builders ----------
    def _build_cache(self, df: pd.DataFrame):
        N = len(df)

        # 0) Common cache base paths
        B = self.cfg.category_hash_buckets
        C = len(self.cats)
        F = len(self.nums)
        part = self.partition
        base_ds = os.path.join(self.cache_dir, f"ds_{part}_{N}_C{C}_F{F}_B{B}_{self.hash_mode}.")
        path_cats = base_ds + "cats.int64.npy"
        path_nums = base_ds + "nums.f32.npy"
        path_tgt  = base_ds + "tgt.int64.npy"
        path_lab  = base_ds + "labels.f32.npy"

        # 1) Categorical (load if exists; else build and persist)
        B = self.cfg.category_hash_buckets
        if C == 0:
            self._cats = torch.zeros((N, 0), dtype=torch.long)
        elif os.path.exists(path_cats):
            arr = np.load(path_cats, mmap_mode='r')
            self._cats = torch.from_numpy(np.array(arr, copy=True)).long()
        else:
            cat_mat = np.empty((N, C), dtype=np.int64)
            for j, c in enumerate(tqdm(self.cats, desc=f"cache: cats[{self.hash_mode}]", disable=not self.progress)):
                if self.hash_mode == "pandas":
                    col = _hash_series_pandas(df[c], B)
                else:
                    col = _hash_series_fnv_parallel(df[c], B, self.workers, self.chunk_rows, progress=False)
                cat_mat[:, j] = col
            np.save(path_cats, cat_mat, allow_pickle=False)
            self._cats = torch.from_numpy(cat_mat.copy()).long()
            del cat_mat
            gc.collect()

        # 2) Numeric (pre-allocate, fill per column, free immediately)
        if F == 0:
            self._nums = torch.zeros((N, 0), dtype=torch.float32)
        elif os.path.exists(path_nums):
            arr = np.load(path_nums, mmap_mode='r')
            self._nums = torch.from_numpy(np.array(arr, copy=True)).float()
        else:
            num_mat = np.empty((N, F), dtype=np.float32)
            for j, c in enumerate(tqdm(self.nums, desc="cache: nums", disable=not self.progress)):
                v = pd.to_numeric(df[c], errors='coerce').fillna(self.cfg.numeric_fillna).to_numpy(np.float32)
                mu, sig = self.stats.get(c, (0.0, 1.0))
                num_mat[:, j] = (v - mu) / (sig if sig else 1.0)
                del v
                gc.collect()
            np.save(path_nums, num_mat, allow_pickle=False)
            self._nums = torch.from_numpy(num_mat.copy()).float()
            del num_mat
            gc.collect()

        # 3) Target id
        if os.path.exists(path_tgt):
            arr = np.load(path_tgt, mmap_mode='r')
            self._tgt = torch.from_numpy(np.array(arr, copy=True)).long()
        else:
            tgt_col = self.cfg.target_feature if self.cfg.target_feature in df.columns else 'inventory_id'
            if tgt_col in df.columns:
                if self.hash_mode == "pandas":
                    t_ids = _hash_series_pandas(df[tgt_col], B)
                else:
                    t_ids = _hash_series_fnv_parallel(df[tgt_col], B, self.workers, self.chunk_rows, progress=False)
            else:
                t_ids = np.zeros(N, dtype=np.int64)
            np.save(path_tgt, t_ids, allow_pickle=False)
            self._tgt = torch.tensor(t_ids, dtype=torch.long)
            del t_ids
            gc.collect()

        # 4) Labels
        if os.path.exists(path_lab):
            arr = np.load(path_lab, mmap_mode='r')
            self._labels = torch.from_numpy(np.array(arr, copy=True)).float()
        else:
            if self.cfg.label_col in df.columns:
                labs = df[self.cfg.label_col].to_numpy(np.float32)
            else:
                labs = np.full(N, -1.0, np.float32)
            np.save(path_lab, labs, allow_pickle=False)
            self._labels = torch.tensor(labs, dtype=torch.float32)

        # 5) Sequence → RAM list OR memmap ragged
        if self.cache_backend == "ram":
            self._build_seq_ram(df)
        else:
            self._build_seq_memmap(df)

    def _build_seq_ram(self, df: pd.DataFrame):
        # 소규모 데이터에만 권장(메모리 사용 큼)
        self._seqs = []
        vocab = self.cfg.seq_vocab_size
        maxL = self.cfg.seq_max_len
        seq_ser = df[self.cfg.seq_col].astype("string") if self.cfg.seq_col in df.columns else pd.Series([""] * len(df))
        chunks = [seq_ser.iloc[i:i + self.chunk_rows].tolist() for i in range(0, len(seq_ser), self.chunk_rows)]
        for chunk in tqdm(chunks, desc="cache: seq[ram]", disable=not self.progress):
            lens, flat = _seq_parse_chunk(chunk, maxL, vocab)
            off = 0
            for L in lens:
                ids = flat[off:off + L]
                self._seqs.append(torch.tensor(ids if L > 0 else [0], dtype=torch.long))
                off += L
            del lens, flat
            gc.collect()

    def _build_seq_memmap(self, df: pd.DataFrame):
        vocab = self.cfg.seq_vocab_size
        maxL = self.cfg.seq_max_len
        N = len(df)
        seq_ser = df[self.cfg.seq_col].astype("string") if self.cfg.seq_col in df.columns else pd.Series([""] * N)

        base = os.path.join(self.cache_dir, f"seq_{self.partition}_{N}_L{maxL}_V{vocab}.")
        path_len = base + "len.npy"
        path_off = base + "off.npy"
        path_dat = base + "dat.int32"

        # 이미 생성된 경우: 바로 mmap
        if os.path.exists(path_len) and os.path.exists(path_off) and os.path.exists(path_dat):
            self._seq_len = np.load(path_len, mmap_mode='r')
            self._seq_off = np.load(path_off, mmap_mode='r')
            self._seq_dat = np.memmap(path_dat, dtype=np.int32, mode='r')
            return

        chunks = [(i, min(i + self.chunk_rows, N)) for i in range(0, N, self.chunk_rows)]

        # -------- Pass 1: 길이만 스트리밍으로 기록 --------
        seq_len_mm = open_memmap(path_len, mode='w+', dtype=np.int32, shape=(N,))
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {}
            for (s, e) in chunks:
                arr = seq_ser.iloc[s:e].tolist()
                futs[ex.submit(_seq_parse_chunk, arr, maxL, vocab)] = (s, e)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq(pass1 len)", disable=not self.progress):
                (s, e) = futs[fut]
                lens, _ = fut.result()
                seq_len_mm[s:e] = lens
                del lens
                gc.collect()

        # 오프셋 memmap 생성 및 cumsum (in-place)
        seq_off_mm = open_memmap(path_off, mode='w+', dtype=np.int64, shape=(N + 1,))
        seq_off_mm[0] = 0
        np.cumsum(seq_len_mm, out=seq_off_mm[1:])
        n_tokens = int(seq_off_mm[-1])

        # Pass1 임시 메모리 해제
        del seq_len_mm
        gc.collect()

        # -------- Pass 2: 토큰 flat을 해당 구간에 바로 쓰기 --------
        seq_dat = np.memmap(path_dat, dtype=np.int32, mode='w+', shape=(n_tokens,))
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {}
            for (s, e) in chunks:
                arr = seq_ser.iloc[s:e].tolist()
                futs[ex.submit(_seq_parse_chunk, arr, maxL, vocab)] = (s, e)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq(pass2 write)", disable=not self.progress):
                (s, e) = futs[fut]
                lens, flat = fut.result()
                a = int(seq_off_mm[s])  # 작은 슬라이스만 참조
                b = int(seq_off_mm[e])
                if flat.size:
                    seq_dat[a:b] = flat
                del lens, flat
                gc.collect()
        seq_dat.flush()

        # 최종 read-only memmap 오픈
        # (seq_off_mm은 이미 파일 기반이므로 그대로 읽기 전용으로 다시 로드)
        del seq_off_mm
        gc.collect()
        self._seq_len = np.load(path_len, mmap_mode='r')
        self._seq_off = np.load(path_off, mmap_mode='r')
        self._seq_dat = np.memmap(path_dat, dtype=np.int32, mode='r')

    # --------- encoders (no-cache fallback) ----------
    def _encode_cats(self, row) -> torch.Tensor:
        vals = []
        B = self.cfg.category_hash_buckets
        for c in self.cats:
            v = row[c]
            vals.append(0 if pd.isna(v) else 1 + hash64(str(v), B - 1))
        return torch.tensor(vals, dtype=torch.long)

    def _encode_nums(self, row) -> torch.Tensor:
        xs = []
        for c in self.nums:
            v = pd.to_numeric(row[c], errors='coerce')
            mu, sig = self.stats.get(c, (0.0, 1.0))
            if pd.isna(v):
                v = self.cfg.numeric_fillna
            xs.append((float(v) - mu) / (sig if sig else 1.0))
        return torch.tensor(xs, dtype=torch.float32)

    def _encode_seq(self, s) -> torch.Tensor:
        if not isinstance(s, str):
            return torch.zeros(1, dtype=torch.long)
        toks = [t.strip() for t in s.split(',') if t.strip() != ""]
        if self.cfg.seq_max_len > 0:
            toks = toks[-self.cfg.seq_max_len:]
        ids = [1 + hash64(tok, self.cfg.seq_vocab_size - 1) for tok in toks] or [0]
        return torch.tensor(ids, dtype=torch.long)

    # --------- dataset API ----------
    def __getitem__(self, idx):
        if self.cache:
            cats = self._cats[idx]
            nums = self._nums[idx]
            tgt = self._tgt[idx]
            lab = self._labels[idx]
            if self.cache_backend == "ram":
                seq = self._seqs[idx]
            else:
                a, b = int(self._seq_off[idx]), int(self._seq_off[idx + 1])
                seq_np = self._seq_dat[a:b]  # view into memmap
                seq = torch.from_numpy(np.array(seq_np, copy=True))  # copy to avoid holding file handle
            return cats, nums, seq, tgt, lab

        # fallback (no cache)
        row = self.df.iloc[idx]
        cats = self._encode_cats(row)
        nums = self._encode_nums(row)
        seq = self._encode_seq(row.get(self.cfg.seq_col))
        tgtv = row[self.cfg.target_feature] if self.cfg.target_feature in self.df.columns else row.get('inventory_id')
        tgt = 1 + hash64(str(tgtv), self.cfg.category_hash_buckets - 1) if pd.notna(tgtv) else 0
        lab = float(row[self.cfg.label_col]) if self.cfg.label_col in self.df.columns else -1.0
        return cats, nums, seq, int(tgt), lab


# ----------------------------
# Collate & Dataloaders
# ----------------------------
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

    # in-batch negative downsampling (train only)
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


# Backward-compatibility: some scripts import `collate_fn` for eval/inference
collate_fn = collate_fn_eval


def make_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg) -> Tuple[DataLoader, DataLoader, dict]:
    cats, nums = infer_feature_types(train_df, cfg.label_col, cfg.seq_col)
    cats = [c for c in cats if c not in set(cfg.force_drop_cols)]
    nums = [c for c in nums if c not in set(cfg.force_drop_cols)]

    # Remove target_feature from categorical features to avoid duplicating the same signal
    tgt_feat = getattr(cfg, "target_feature", None)
    if tgt_feat in cats:
        cats = [c for c in cats if c != tgt_feat]

    train_ds = CTRDataset(train_df, cfg, cats, nums, is_train=True, partition="train")
    cfg.d['num_stats'] = train_ds.stats  # pass stats to val/test
    val_ds = CTRDataset(val_df, cfg, cats, nums, is_train=False, partition="val")

    # DataLoader perf flags
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
