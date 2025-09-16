import os
import gc
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import polars as pl

from numpy.lib.format import open_memmap

from .utils import hash64
from .data_schema import CAT_PATTERNS, EXCLUDE_COLS, match_any
from .dataset import collate_fn_train, collate_fn_eval
from tqdm import tqdm


def infer_feature_types_pl(df: pl.DataFrame, label_col: str, seq_col: str):
    cats, nums = [], []
    for c, dt in zip(df.columns, df.dtypes):
        if c in EXCLUDE_COLS or c == label_col or c == seq_col:
            continue
        if dt == pl.Utf8 or match_any(c, CAT_PATTERNS):
            cats.append(c)
        else:
            nums.append(c)
    return cats, nums


def _seq_parse_chunk(arr, maxL: int, vocab: int):
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


class CTRDatasetPolars(Dataset):
    def __init__(self, df: pl.DataFrame, cfg, cats: List[str], nums: List[str], is_train: bool, partition: Optional[str] = None):
        self.df = df
        self.cfg = cfg
        self.cats = cats
        self.nums = nums
        self.is_train = is_train
        if partition is not None:
            self.partition = partition
        else:
            if is_train:
                self.partition = "train"
            else:
                self.partition = "val" if cfg.label_col in df.columns else "test"

        raw_stats = getattr(cfg, 'num_stats', {}) or {}
        if not isinstance(raw_stats, dict):
            raw_stats = getattr(raw_stats, 'd', raw_stats)
        if not isinstance(raw_stats, dict):
            raw_stats = {}
        if not raw_stats:
            raise ValueError("cfg.num_stats is required and must be a non-empty dict. Compute stats in training (artifacts/num_stats.json) and pass via cfg.d['num_stats'].")
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

        self.cache = bool(getattr(cfg, "dataset_cache", False))
        self.cache_backend = str(getattr(cfg, "dataset_cache_backend", "memmap")).lower()
        self.cache_dir = str(getattr(cfg, "dataset_cache_dir", "./artifacts/cache_seq"))
        self.progress = bool(getattr(cfg, "dataset_cache_progress", True))
        self.chunk_rows = int(getattr(cfg, "dataset_cache_chunk_rows", 1_000_000))
        self.workers = int(getattr(cfg, "dataset_cache_workers", max(1, (os.cpu_count() or 2) // 2)))

        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._build_cache(self.df)
        else:
            self.N = self.df.height

    def __len__(self):
        if self.df is not None:
            return self.df.height
        return getattr(self, 'N', 0)

    def _build_cache(self, df: pl.DataFrame):
        N = df.height
        self.N = N

        B = self.cfg.category_hash_buckets
        C = len(self.cats)
        F = len(self.nums)
        part = self.partition
        base_ds = os.path.join(self.cache_dir, f"ds_{part}_{N}_C{C}_F{F}_B{B}_polars.")
        path_cats32 = base_ds + "cats.int32.npy"
        path_cats64 = base_ds + "cats.int64.npy"
        path_nums   = base_ds + "nums.f32.npy"
        path_tgt32  = base_ds + "tgt.int32.npy"
        path_tgt64  = base_ds + "tgt.int64.npy"
        path_lab    = base_ds + "labels.f32.npy"

        # 1) categorical
        if C == 0:
            self._cats = torch.zeros((N, 0), dtype=torch.long)
        elif os.path.exists(path_cats32) or os.path.exists(path_cats64):
            print(f"Loading cats from {path_cats32} or {path_cats64}")
            pth = path_cats32 if os.path.exists(path_cats32) else path_cats64
            arr = np.load(pth, mmap_mode='r')
            # keep compact in RAM as int32; convert to long on access
            arr_np = np.array(arr, copy=True)
            if arr_np.dtype != np.int32:
                arr_np = arr_np.astype(np.int32, copy=False)
            self._cats = torch.from_numpy(arr_np).int()
        else:
            # build per-column into memmap to minimize peak RAM
            cat_mm = open_memmap(path_cats32, mode='w+', dtype=np.int32, shape=(N, C))
            for j, c in enumerate(tqdm(self.cats, desc="cache: cats[polars]", disable=not self.progress)):
                col_np = df.select((pl.col(c).cast(pl.Utf8).fill_null("").hash().mod(B-1).cast(pl.Int64) + 1)).to_numpy().reshape(-1)
                cat_mm[:, j] = col_np.astype(np.int32, copy=False)
                del col_np
                gc.collect()
            del cat_mm
            gc.collect()
            arr = np.load(path_cats32, mmap_mode='r')
            self._cats = torch.from_numpy(np.array(arr, copy=True)).int()

        # 2) numeric
        if F == 0:
            self._nums = torch.zeros((N, 0), dtype=torch.float32)
        elif os.path.exists(path_nums):
            print(f"Loading nums from {path_nums}")
            arr = np.load(path_nums, mmap_mode='r')
            self._nums = torch.from_numpy(np.array(arr, copy=True)).float()
        else:
            # stream into memmap col-by-col
            num_mm = open_memmap(path_nums, mode='w+', dtype=np.float32, shape=(N, F))
            for j, c in enumerate(tqdm(self.nums, desc="cache: nums", disable=not self.progress)):
                mu, sig = self.stats.get(c, (0.0, 1.0))
                expr = (pl.col(c).cast(pl.Float64).fill_nan(self.cfg.numeric_fillna).fill_null(self.cfg.numeric_fillna) - mu) / (sig if sig else 1.0)
                col_np = df.select(expr.alias(c)).to_numpy().reshape(-1).astype(np.float32, copy=False)
                num_mm[:, j] = col_np
                del col_np
                gc.collect()
            del num_mm
            gc.collect()
            arr = np.load(path_nums, mmap_mode='r')
            self._nums = torch.from_numpy(np.array(arr, copy=True)).float()

        # 3) target id
        if os.path.exists(path_tgt32) or os.path.exists(path_tgt64):
            print(f"Loading tgt from {path_tgt32} or {path_tgt64}")
            pth = path_tgt32 if os.path.exists(path_tgt32) else path_tgt64
            arr = np.load(pth, mmap_mode='r')
            arr_np = np.array(arr, copy=True)
            if arr_np.dtype != np.int32:
                arr_np = arr_np.astype(np.int32, copy=False)
            self._tgt = torch.from_numpy(arr_np).int()
        else:
            tgt_col = self.cfg.target_feature if self.cfg.target_feature in df.columns else 'inventory_id'
            if tgt_col in df.columns:
                t_ids = (df.select(pl.col(tgt_col).cast(pl.Utf8).fill_null("").hash().mod(B-1).cast(pl.Int64) + 1).to_numpy().reshape(-1)).astype(np.int32, copy=False)
            else:
                t_ids = np.zeros(N, dtype=np.int32)
            np.save(path_tgt32, t_ids, allow_pickle=False)
            self._tgt = torch.from_numpy(t_ids.copy()).int()
            del t_ids

        # 4) labels
        if os.path.exists(path_lab):
            print(f"Loading labels from {path_lab}")
            arr = np.load(path_lab, mmap_mode='r')
            self._labels = torch.from_numpy(np.array(arr, copy=True)).float()
        else:
            if self.cfg.label_col in df.columns:
                labs = df.select(pl.col(self.cfg.label_col).cast(pl.Float32)).to_numpy().reshape(-1)
            else:
                labs = np.full(N, -1.0, np.float32)
            np.save(path_lab, labs, allow_pickle=False)
            self._labels = torch.tensor(labs, dtype=torch.float32)

        # 5) sequence: memmap two-pass (reuse python helper)
        if self.cache_backend == "ram":
            self._build_seq_ram(df)
        else:
            self._build_seq_memmap(df)
        # release original DataFrame memory as caches are ready
        self.df = None
        gc.collect()

    def _build_seq_ram(self, df: pl.DataFrame):
        vocab = self.cfg.seq_vocab_size
        maxL = self.cfg.seq_max_len
        N = df.height
        seq_col = self.cfg.seq_col
        if seq_col in df.columns:
            seq_ser = df.select(pl.col(seq_col).cast(pl.Utf8).fill_null("")).to_series().to_list()
        else:
            seq_ser = [""] * N

        base = os.path.join(self.cache_dir, f"seq_{self.partition}_{N}_L{maxL}_V{vocab}.")
        path_len = base + "len.npy"
        path_off = base + "off.npy"
        path_dat = base + "dat.int32"

        # If existing memmap cache is present, reuse directly to avoid rebuild
        if os.path.exists(path_len) and os.path.exists(path_off) and os.path.exists(path_dat):
            print(f"Loading seq len from {path_len}")
            print(f"Loading seq off from {path_off}")
            print(f"Loading seq dat from {path_dat}")
            self._seq_len = np.load(path_len, mmap_mode='r')
            self._seq_off = np.load(path_off, mmap_mode='r')
            self._seq_dat = np.memmap(path_dat, dtype=np.int32, mode='r')
            return

        chunks = [(i, min(i + self.chunk_rows, N)) for i in range(0, N, self.chunk_rows)]

        # Pass 1: compute lengths in parallel (pure RAM)
        seq_len = np.empty(N, dtype=np.int32)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {}
            for (s, e) in chunks:
                arr = seq_ser[s:e]
                futs[ex.submit(_seq_parse_chunk, arr, maxL, vocab)] = (s, e)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq(pass1 len)[ram]", disable=not self.progress):
                (s, e) = futs[fut]
                lens, _ = fut.result()
                seq_len[s:e] = lens
        seq_off = np.empty(N + 1, dtype=np.int64)
        seq_off[0] = 0
        np.cumsum(seq_len, out=seq_off[1:])
        n_tokens = int(seq_off[-1])

        # Pass 2: fill tokens in parallel (pure RAM)
        seq_dat = np.empty(n_tokens, dtype=np.int32)
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {}
            for (s, e) in chunks:
                arr = seq_ser[s:e]
                futs[ex.submit(_seq_parse_chunk, arr, maxL, vocab)] = (s, e)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq(pass2 write)[ram]", disable=not self.progress):
                (s, e) = futs[fut]
                lens, flat = fut.result()
                a = int(seq_off[s])
                b = int(seq_off[e])
                if flat.size:
                    seq_dat[a:b] = flat
        # assign RAM ragged arrays
        self._seq_len = seq_len
        self._seq_off = seq_off
        self._seq_dat = seq_dat

        # persist to cache files for reuse on next run
        try:
            np.save(path_len, seq_len, allow_pickle=False)
            np.save(path_off, seq_off, allow_pickle=False)
            dat_mm = np.memmap(path_dat, dtype=np.int32, mode='w+', shape=(n_tokens,))
            dat_mm[:] = seq_dat[:]
            dat_mm.flush()
            del dat_mm
        except Exception:
            pass

    def _build_seq_memmap(self, df: pl.DataFrame):
        vocab = self.cfg.seq_vocab_size
        maxL = self.cfg.seq_max_len
        N = df.height
        seq_col = self.cfg.seq_col
        if seq_col in df.columns:
            seq_ser = df.select(pl.col(seq_col).cast(pl.Utf8).fill_null("")).to_series().to_list()
        else:
            seq_ser = [""] * N

        base = os.path.join(self.cache_dir, f"seq_{self.partition}_{N}_L{maxL}_V{vocab}.")
        path_len = base + "len.npy"
        path_off = base + "off.npy"
        path_dat = base + "dat.int32"

        if os.path.exists(path_len) and os.path.exists(path_off) and os.path.exists(path_dat):
            print(f"Loading seq len from {path_len}")
            print(f"Loading seq off from {path_off}")
            print(f"Loading seq dat from {path_dat}")
            self._seq_len = np.load(path_len, mmap_mode='r')
            self._seq_off = np.load(path_off, mmap_mode='r')
            self._seq_dat = np.memmap(path_dat, dtype=np.int32, mode='r')
            return

        chunks = [(i, min(i + self.chunk_rows, N)) for i in range(0, N, self.chunk_rows)]

        seq_len_mm = open_memmap(path_len, mode='w+', dtype=np.int32, shape=(N,))
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {}
            for (s, e) in chunks:
                arr = seq_ser[s:e]
                futs[ex.submit(_seq_parse_chunk, arr, maxL, vocab)] = (s, e)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq(pass1 len)", disable=not self.progress):
                (s, e) = futs[fut]
                lens, _ = fut.result()
                seq_len_mm[s:e] = lens
        seq_off_mm = open_memmap(path_off, mode='w+', dtype=np.int64, shape=(N + 1,))
        seq_off_mm[0] = 0
        np.cumsum(seq_len_mm, out=seq_off_mm[1:])
        n_tokens = int(seq_off_mm[-1])

        del seq_len_mm
        gc.collect()

        seq_dat = np.memmap(path_dat, dtype=np.int32, mode='w+', shape=(n_tokens,))
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {}
            for (s, e) in chunks:
                arr = seq_ser[s:e]
                futs[ex.submit(_seq_parse_chunk, arr, maxL, vocab)] = (s, e)
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cache: seq(pass2 write)", disable=not self.progress):
                (s, e) = futs[fut]
                lens, flat = fut.result()
                a = int(seq_off_mm[s])
                b = int(seq_off_mm[e])
                if flat.size:
                    seq_dat[a:b] = flat
        seq_dat.flush()

        del seq_off_mm
        gc.collect()
        self._seq_len = np.load(path_len, mmap_mode='r')
        self._seq_off = np.load(path_off, mmap_mode='r')
        self._seq_dat = np.memmap(path_dat, dtype=np.int32, mode='r')

    # expose tensor items like original dataset
    def __getitem__(self, idx):
        cats = self._cats[idx].long()
        nums = self._nums[idx]
        tgt = self._tgt[idx].long()
        lab = self._labels[idx]
        if hasattr(self, "_seqs"):
            seq = self._seqs[idx]
        else:
            a, b = int(self._seq_off[idx]), int(self._seq_off[idx + 1])
            seq_np = self._seq_dat[a:b]
            seq = torch.from_numpy(np.array(seq_np, copy=True))
        return cats, nums, seq, tgt, lab


# Collate alias for compatibility
collate_fn = collate_fn_eval


def make_dataloaders_polars(train_df: pl.DataFrame, val_df: pl.DataFrame, cfg) -> Tuple[DataLoader, DataLoader, dict]:
    cats, nums = infer_feature_types_pl(train_df, cfg.label_col, cfg.seq_col)
    cats = [c for c in cats if c not in set(cfg.force_drop_cols)]
    nums = [c for c in nums if c not in set(cfg.force_drop_cols)]

    tgt_feat = getattr(cfg, "target_feature", None)
    if tgt_feat in cats:
        cats = [c for c in cats if c != tgt_feat]

    train_ds = CTRDatasetPolars(train_df, cfg, cats, nums, is_train=True, partition="train")
    cfg.d['num_stats'] = train_ds.stats
    val_ds = CTRDatasetPolars(val_df, cfg, cats, nums, is_train=False, partition="val")

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


