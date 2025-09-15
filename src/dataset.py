import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import hash64
from .data_schema import CAT_PATTERNS, EXCLUDE_COLS, match_any
from types import SimpleNamespace
from tqdm import tqdm

class CollateWithCfg:
    """Windows spawn에서 picklable한 collate_fn 래퍼"""
    def __init__(self, use_neg_downsampling: bool, neg_downsample_ratio: float, min_pos_per_batch: int):
        self.ns = SimpleNamespace(
            use_neg_downsampling=use_neg_downsampling,
            neg_downsample_ratio=neg_downsample_ratio,
            min_pos_per_batch=min_pos_per_batch,
        )

    def __call__(self, batch):
        # 기존 collate_fn을 그대로 재사용하되 cfg 대신 namesapce 주입
        return collate_fn(batch, self.ns)

class CTRFrame:
    """Lightweight loader around parquet with column pruning and type optimization."""
    def __init__(self, parquet_path: str, drop_cols: List[str]):
        self.pf = pq.ParquetFile(parquet_path)
        self.drop_cols = set(drop_cols)
        schema = self.pf.schema_arrow
        self.all_cols = [n for n in schema.names if n not in self.drop_cols]

    def _convert_string_to_large_string(self, table):
        """Convert string columns to large_string to handle large data"""
        schema = table.schema
        new_fields = []
        for field in schema:
            if field.type == pa.string():
                new_fields.append(pa.field(field.name, pa.large_string()))
            else:
                new_fields.append(field)
        
        new_schema = pa.schema(new_fields)
        return table.cast(new_schema)

    def read_all(self, columns: List[str] = None) -> pd.DataFrame:
        cols = columns or self.all_cols
        table = self.pf.read(columns=cols)
        table = self._convert_string_to_large_string(table)
        return table.to_pandas(types_mapper=pd.ArrowDtype)

    def read_row_group(self, rg_idx: int, columns: List[str] = None) -> pd.DataFrame:
        cols = columns or self.all_cols
        table = self.pf.read_row_group(rg_idx, columns=cols)
        table = self._convert_string_to_large_string(table)
        return table.to_pandas(types_mapper=pd.ArrowDtype)

    @property
    def num_row_groups(self):
        return self.pf.num_row_groups


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


def parse_seq_col(s: str, max_len: int) -> List[str]:
    if not isinstance(s, str):
        return []
    toks = s.split(',')
    if max_len > 0:
        toks = toks[-max_len:]
    return toks

class CTRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, cats: List[str], nums: List[str], is_train: bool):
        self.df = df
        self.cfg = cfg
        self.cats = cats
        self.nums = nums
        self.is_train = is_train
        self.cache = bool(getattr(cfg, "dataset_cache", False))

        # stats for numeric standardization (fit on train, reuse for val/test)
        stats = {}
        if is_train:
            for c in nums:
                v = pd.to_numeric(df[c], errors='coerce')
                mu, sig = float(v.mean()), float(v.std(ddof=0))
                if sig == 0 or np.isnan(sig):
                    sig = 1.0
                stats[c] = (mu, sig)
            self.stats = stats
        else:
            self.stats = getattr(cfg, 'num_stats', {})

        # 3) 캐시 경로: 한 번만 벡터화/전처리하여 텐서로 보관
        if self.cache:
            # --- Categorical: hash bucket ---
            cat_ids = []
            B = self.cfg.category_hash_buckets
            cat_bar = tqdm(self.cats, desc="Encoding categories", leave=False)
            for c in cat_bar:
                v = df[c].astype("string").fillna("")        # NaN -> ""
                # "" → 0 (pad), 그 외는 1..B 범위
                ids = v.apply(lambda x: 0 if x == "" else 1 + hash64(x, B - 1)).to_numpy(np.int64)
                cat_ids.append(ids)
            self._cats = torch.tensor(np.stack(cat_ids, axis=1) if cat_ids else np.zeros((len(df), 0), np.int64),
                                      dtype=torch.long)

            # --- Numeric: z-score ---
            xs = []
            num_bar = tqdm(self.nums, desc="Encoding numerics", leave=False)
            for c in num_bar:
                v = pd.to_numeric(df[c], errors='coerce').fillna(self.cfg.numeric_fillna).to_numpy(np.float32)
                mu, sig = self.stats.get(c, (0.0, 1.0))
                xs.append((v - mu) / (sig if sig else 1.0))
            self._nums = torch.tensor(np.stack(xs, axis=1) if xs else np.zeros((len(df), 0), np.float32),
                                      dtype=torch.float32)

            # --- Sequence: 각 행을 텐서로 저장(가변 길이) ---
            self._seqs = []
            maxL = self.cfg.seq_max_len
            vocab = self.cfg.seq_vocab_size
            seq_series = df[self.cfg.seq_col].astype("string") if self.cfg.seq_col in df.columns else pd.Series([""] * len(df))
            seq_bar = tqdm(seq_series, desc="Encoding sequences", leave=False)
            for s in seq_bar:
                toks = [] if s is None or s != s else s.split(',')
                if maxL > 0:
                    toks = toks[-maxL:]
                # pad 토큰=0, 실제 토큰=1..vocab 범위
                ids = [1 + hash64(tok, vocab - 1) for tok in toks] if len(toks) > 0 else [0]
                self._seqs.append(torch.tensor(ids, dtype=torch.long))

            # --- Target id (target_feature 있으면 사용, 없으면 inventory_id로 폴백) ---
            tgt_col = self.cfg.target_feature if self.cfg.target_feature in df.columns else 'inventory_id'
            t_ser = df[tgt_col].astype("string").fillna("") if tgt_col in df.columns else pd.Series([""] * len(df))
            t = t_ser.apply(lambda x: 1 + hash64(x, self.cfg.category_hash_buckets - 1) if x != "" else 0).to_numpy(np.int64)
            self._tgt = torch.tensor(t, dtype=torch.long)

            # --- Label (테스트셋이면 -1로 채움) ---
            if self.cfg.label_col in df.columns:
                self._labels = torch.tensor(df[self.cfg.label_col].to_numpy(np.float32), dtype=torch.float32)
            else:
                self._labels = torch.tensor(np.full(len(df), -1.0, np.float32), dtype=torch.float32)

        # pack tensors lazily in __getitem__ to avoid memory blow-up

    def __len__(self):
        return len(self.df)

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
            return torch.zeros(1, dtype=torch.long)  # at least 1 pad
        return torch.tensor(ids, dtype=torch.long)

    # ====== __getitem__ ======
    def __getitem__(self, idx):
        if self.cache:
            # 캐시된 텐서들을 그대로 반환 (seq는 가변 길이 텐서 리스트)
            return self._cats[idx], self._nums[idx], self._seqs[idx], self._tgt[idx], self._labels[idx]

        # fallback: 행 단위 인코딩(기존 방식)
        row = self.df.iloc[idx]
        cats = self._encode_cats(row)
        nums = self._encode_nums(row)
        seq = self._encode_seq(row.get(self.cfg.seq_col) if self.cfg.seq_col in self.df.columns else None)

        if self.cfg.target_feature in self.df.columns:
            tgt_val = row[self.cfg.target_feature]
        else:
            tgt_val = row.get('inventory_id')
        tgt_id = 1 + hash64(str(tgt_val), self.cfg.category_hash_buckets - 1) if pd.notna(tgt_val) else 0

        label = float(row[self.cfg.label_col]) if self.cfg.label_col in self.df.columns else -1.0
        return cats, nums, seq, int(tgt_id), label


def collate_fn(batch, cfg):
    cats, nums, seqs, tgt_ids, labels = zip(*batch)
    cats = torch.stack(cats)
    nums = torch.stack(nums)
    # pad sequences
    max_len = max(s.shape[0] for s in seqs)
    seq_pad = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq_pad[i, -len(s):] = s  # right align
    tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    # optional negative downsampling inside batch (keep all positives)
    if cfg.use_neg_downsampling and (labels >= 0).all():
        pos_idx = (labels == 1)
        neg_idx = (labels == 0)
        keep = pos_idx.clone()
        neg = torch.nonzero(neg_idx).flatten()
        if len(neg) > 0:
            k = int(len(neg) * cfg.neg_downsample_ratio)
            k = max(k, 1)
            perm = torch.randperm(len(neg))[:k]
            keep[neg[perm]] = True
        # ensure min positives per batch
        if pos_idx.sum() < cfg.min_pos_per_batch and pos_idx.sum() > 0:
            keep |= pos_idx
        cats, nums, seq_pad, tgt_ids, labels = cats[keep], nums[keep], seq_pad[keep], tgt_ids[keep], labels[keep]

    return cats, nums, seq_pad, tgt_ids, labels


def make_dataloaders(train_df, val_df, cfg):
    # infer feature types from train
    cats, nums = infer_feature_types(train_df, cfg.label_col, cfg.seq_col)
    # force drop columns
    cats = [c for c in cats if c not in set(cfg.force_drop_cols)]
    nums = [c for c in nums if c not in set(cfg.force_drop_cols)]

    train_ds = CTRDataset(train_df, cfg, cats, nums, is_train=True)
    # stash numeric stats for val/test
    cfg.d['num_stats'] = train_ds.stats

    val_ds = CTRDataset(val_df, cfg, cats, nums, is_train=False)

    # train_collate = CollateWithCfg(
    #     use_neg_downsampling=cfg.use_neg_downsampling,
    #     neg_downsample_ratio=cfg.neg_downsample_ratio,
    #     min_pos_per_batch=cfg.min_pos_per_batch,
    # )
    # val_collate = CollateWithCfg(   # 검증은 다운샘플 X
    #     use_neg_downsampling=False,
    #     neg_downsample_ratio=cfg.neg_downsample_ratio,
    #     min_pos_per_batch=cfg.min_pos_per_batch,
    # )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=lambda b: collate_fn(b, cfg),
        # collate_fn=train_collate,
        drop_last=False,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=lambda b: collate_fn(b, cfg),
        # collate_fn=val_collate,
        drop_last=False,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor
    )

    input_dims = {
        'n_cat': len(cats),
        'n_num': len(nums),
        'seq_vocab_size': cfg.seq_vocab_size,
        'cat_buckets': cfg.category_hash_buckets,
    }
    return train_loader, val_loader, input_dims