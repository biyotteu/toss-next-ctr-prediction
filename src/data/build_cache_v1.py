from __future__ import annotations
import os, json, re, hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import polars as pl
import pyarrow.dataset as ds
from tqdm.auto import tqdm

# ========== 유틸 ==========

def _match_patterns(cols: List[str], patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        regex = re.compile("^" + p.replace("*", ".*") + "$")
        out += [c for c in cols if regex.match(c)]
    return sorted(list(dict.fromkeys(out)))

def _hash_int_stable(x: int, buckets: int) -> int:
    # 세션 독립, 완전 재현: md5 기반
    return int(hashlib.md5(str(int(x)).encode()).hexdigest()[:8], 16) % buckets

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _np_save(path: str, arr: np.ndarray):
    # np.save를 쓰면 나중에 np.load(..., mmap_mode="r") 가능(메모리 절약)
    np.save(path, arr)

# ========== 스키마/통계 ==========

def analyze_schema_and_stats(parquet_path: str,
                             target_col: Optional[str],
                             seq_col: str,
                             cat_cols: List[str],
                             num_patterns: List[str],
                             group_key: str,
                             impute_strategy: str,
                             num_cols_explicit: List[str] | None = None,
                             remove_cols: List[str] | None = None) -> Dict:

    """
    - 전체 컬럼 목록
    - 수치 컬럼 리스트(패턴 매칭)
    - 수치 컬럼 전역 median(결측 대치용; streaming으로 계산)
    - 행수
    """
    scan = pl.scan_parquet(parquet_path)
    cols = scan.collect_schema().names()
    if num_cols_explicit:
        num_cols = [c for c in num_cols_explicit if c in cols]
    else:
        num_cols = [c for c in _match_patterns(cols, num_patterns)
                    if c not in cat_cols and c not in [target_col, seq_col, group_key, "ID"] and c in cols]
    if remove_cols:
        num_cols = [c for c in num_cols if c not in remove_cols]
    # 전역 메디안(정확): streaming True로 전체 파일에서 계산
    if impute_strategy == "median" and len(num_cols) > 0:
        med_df = scan.select([pl.col(c).median().alias(c) for c in num_cols]).collect(streaming=True)
        # median이 NaN인 컬럼(전체 결측 등)은 0.0으로 안전 대체
        med_map = {}
        for c in num_cols:
            v = med_df[c][0]
            v = 0.0 if (v is None or (isinstance(v, float) and (v != v))) else float(v)
            med_map[c] = v
    else:
        med_map = {c: 0.0 for c in num_cols}

    n_rows = scan.select(pl.len()).collect(streaming=True).item()

    return {
        "all_cols": cols,
        "num_cols": num_cols,
        "med_map": med_map,
        "n_rows": int(n_rows)
    }

# ========== 샤드 저장 ==========

def _process_arrow_batch(
    df: pl.DataFrame,
    *,
    is_train: bool,
    target_col: Optional[str],
    seq_col: str,
    cat_cols: List[str],
    hash_buckets: Dict[str,int],
    hash_buckets_margin: int,
    num_cols: List[str],
    med_map: Dict[str,float],
    max_len: int,
    pad_id: int,
    group_key: str,
    time_key: Optional[str] = None,
    composite_group: bool = False
) -> Dict[str, np.ndarray]:
    cols = df.columns

    # y / groups / ids
    y = df[target_col].cast(pl.Int8).to_numpy() if (is_train and target_col in cols) else None

    # group_key도 문자열/float 혼재 가능 → 안정 해시로 int64 그룹 생성
    # groups (composite: group_key × time_key)
    if composite_group and (group_key in cols) and (time_key is not None) and (time_key in cols):
        g_expr = pl.struct([
            pl.col(group_key).cast(pl.Utf8).fill_null("NA"),
            pl.col(time_key).cast(pl.Utf8).fill_null("NA")
        ]).hash(seed=2025, seed_1=0).alias("g")
        g_series = df.select(g_expr).get_column("g")     # Expr 평가 → Series
    elif group_key in cols:
        g_expr = pl.col(group_key).cast(pl.Utf8).fill_null("NA").hash(seed=2025, seed_1=0).alias("g")
        g_series = df.select(g_expr).get_column("g")     # Expr 평가 → Series
    else:
        g_series = pl.Series("g", np.zeros(df.height, dtype=np.int64))

    # numpy로 변환 후 모듈러 & int64 캐스팅
    groups = (g_series.to_numpy() % (2**31 - 1)).astype(np.int64)

    # ✅ ID는 문자열로 보관(제출 포맷 안전)
    if "ID" in cols:
        ids_np = df["ID"].cast(pl.Utf8).fill_null("").to_numpy()  # polars→numpy: 종종 object dtype
    else:
        ids_np = np.arange(df.height, dtype=np.int64).astype(str)
    ids = np.asarray(ids_np, dtype="U64")
    # ✅ 카테고리 해시 (열별 버킷 적용, 타입 혼재 안전/재현성 보장)
    X_cat_list = []
    for c in cat_cols:
        hb = hash_buckets.get(c, 1000003) + hash_buckets_margin
        if c in cols:
            hashed = df[c].cast(pl.Utf8).fill_null("NA").hash(seed=2025, seed_1=0) % hb
            X_cat_list.append(hashed.cast(pl.Int32).to_numpy())
        else:
            X_cat_list.append(np.zeros((df.height,), np.int32))
    X_cat = np.stack(X_cat_list, axis=1).astype(np.int32) if X_cat_list else np.zeros((df.height,0), np.int32)

    # 수치 + 결측마스크
    if num_cols:
        X_num = df.select(num_cols).to_numpy().astype(np.float32)
        mask = np.isnan(X_num).astype(np.uint8)
        for j, c in enumerate(num_cols):
            if mask[:, j].any():
                X_num[mask[:, j], j] = med_map.get(c, 0.0)
        # 혹시 남아있는 NaN/Inf를 최종적으로 제거
        np.nan_to_num(X_num, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        X_num = np.zeros((df.height, 0), np.float32)
        mask  = np.zeros((df.height, 0), np.uint8)

    # seq -> (B, L) int32
    s = df[seq_col].fill_null("").to_list() if seq_col in cols else ["" for _ in range(df.height)]
    seq = np.full((df.height, max_len), pad_id, dtype=np.int32)
    for i, st in enumerate(s):
        if not st: continue
        toks = [int(x) for x in str(st).split(",") if x]
        toks = toks[-max_len:]
        if toks:
            seq[i, -len(toks):] = np.asarray(toks, dtype=np.int32)

    return {
        "X_num": X_num,
        "X_mask": mask,
        "X_cat": X_cat,
        "seq": seq,
        "y": (y if y is not None else np.zeros((df.height,), np.int8)),
        "groups": groups,
        "ids": ids
    }


def _save_shard(shard_dir: str, arrays: Dict[str,np.ndarray]) -> Dict:
    _ensure_dir(shard_dir)
    meta = {}
    for k, v in arrays.items():
        path = os.path.join(shard_dir, f"{k}.npy")
        _np_save(path, v)
        meta[k] = {"path": path, "shape": list(v.shape), "dtype": str(v.dtype)}
    meta["rows"] = arrays["seq"].shape[0]
    return meta

def build_sharded_cache(
    parquet_path: str,
    out_dir: str,
    *,
    is_train: bool,
    target_col: Optional[str],
    seq_col: str,
    cat_cols: List[str],
    hash_buckets: Dict[str,int],
    hash_buckets_margin: int,
    num_patterns: List[str],
    max_len: int,
    pad_id: int,
    group_key: str,
    time_key: Optional[str] = None,
    composite_group: bool = False,
    shard_rows: int = 2_000_000,
    impute_strategy: str = "median",
    num_cols_explicit: List[str] | None = None,
    remove_cols: List[str] | None = None
) -> str:
    """
    Parquet → 다수 샤드 저장(.npy) + manifest.json 반환경로 리턴
    """
    _ensure_dir(out_dir)
    schema = analyze_schema_and_stats(
        parquet_path, target_col, seq_col, cat_cols, num_patterns, group_key, impute_strategy, num_cols_explicit, remove_cols
    )
    num_cols = schema["num_cols"]; med_map = schema["med_map"]; total_n = schema["n_rows"]


    # tqdm bars
    est_shards = max(1, (total_n + shard_rows - 1) // shard_rows)
    p_rows   = tqdm(total=total_n, desc=f"[{'train' if is_train else 'test'}] rows processed", unit="rows")
    p_shards = tqdm(total=est_shards, desc=f"[{'train' if is_train else 'test'}] shards saved", unit="shard")

    # Arrow dataset streaming
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(batch_size=200_000)
    it = scanner.to_batches()  # iterable of RecordBatch

    shard_idx = 0
    row_buf = 0
    acc = {k: [] for k in ["X_num","X_mask","X_cat","seq","y","groups","ids"]}
    manifest = {"parquet": parquet_path, "is_train": is_train, "rows": 0, "shards": [],
                "num_cols": num_cols, "cat_cols": cat_cols, "group_key": group_key, "seq_col": seq_col}

    def flush_shard():
        nonlocal shard_idx, acc, row_buf, manifest
        if row_buf == 0: return
        # concat
        arrays = {k: np.concatenate(acc[k], axis=0) if len(acc[k]) else np.zeros((0,0), dtype=np.float32)
                  for k in ["X_num","X_mask","X_cat","seq","y","groups","ids"]}
        sdir = os.path.join(out_dir, f"shard_{shard_idx:03d}")
        meta = _save_shard(sdir, arrays)
        meta["index"] = shard_idx
        meta["start"] = manifest["rows"]
        meta["end"]   = manifest["rows"] + meta["rows"]   # exclusive
        manifest["shards"].append(meta)
        manifest["rows"] += meta["rows"]

        shard_idx += 1
        row_buf = 0
        acc = {k: [] for k in ["X_num","X_mask","X_cat","seq","y","groups","ids"]}
        return True

    for rb in it:
        df = pl.from_arrow(rb)
        df = df.drop(remove_cols) if remove_cols else df
        batch = _process_arrow_batch(
            df,
            is_train=is_train,
            target_col=target_col,
            seq_col=seq_col,
            cat_cols=cat_cols,
            hash_buckets=hash_buckets,
            hash_buckets_margin=hash_buckets_margin,
            num_cols=num_cols,
            med_map=med_map,
            max_len=max_len,
            pad_id=pad_id,
            group_key=group_key,
            time_key=time_key,
            composite_group=composite_group
        )

        # 누적
        for k in acc.keys():
            acc[k].append(batch[k])
        row_buf += df.height

        p_rows.update(df.height)

        # 샤드 컷
        while row_buf >= shard_rows:
            # 자르고 넘치면 스플릿
            cut = shard_rows
            arrays_cut = {k: np.concatenate(acc[k], axis=0) for k in acc.keys()}
            arrays_head = {k: arrays_cut[k][:cut] for k in acc.keys()}
            arrays_tail = {k: arrays_cut[k][cut:] for k in acc.keys()}
            sdir = os.path.join(out_dir, f"shard_{shard_idx:03d}")
            meta = _save_shard(sdir, arrays_head)
            meta["index"] = shard_idx
            meta["start"] = manifest["rows"]
            meta["end"]   = manifest["rows"] + meta["rows"]
            manifest["shards"].append(meta)
            manifest["rows"] += meta["rows"]
            shard_idx += 1
            
            # 샤드 진행바
            p_shards.update(1)

            # tail을 다시 acc로
            acc = {k: [arrays_tail[k]] for k in acc.keys()}
            row_buf = arrays_tail["seq"].shape[0]

    # 마지막 flush
    if flush_shard():
        p_shards.update(1)

    # tqdm 닫기
    p_rows.close()
    p_shards.close()

    # manifest 저장
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return man_path

# 진입점 헬퍼(학습/테스트)
def build_train_and_test(cfg: dict) -> Tuple[str,str]:
    mp_train = build_sharded_cache(
        parquet_path=cfg["data"]["train_path"],
        out_dir=os.path.join(cfg["data"]["cache_dir"], "train"),
        is_train=True,
        target_col="clicked",
        seq_col=cfg["sequence"]["col"],
        cat_cols=cfg["data"]["cat_cols"],
        hash_buckets=cfg["data"]["hash_buckets"],
        hash_buckets_margin=cfg["data"].get("hash_buckets_margin", 0),
        num_patterns=cfg["data"]["num_patterns"],
        num_cols_explicit=cfg["data"].get("num_cols_explicit"),
        max_len=cfg["sequence"]["max_len"],
        pad_id=cfg["sequence"]["pad_id"],
        group_key=cfg["cv"]["group_key"],
        time_key=cfg["cv"].get("time_key"),
        composite_group=bool(cfg["cv"].get("composite_group", False)),
        shard_rows=cfg["data"].get("shard_rows", 2_000_000),
        impute_strategy=cfg["data"]["impute_strategy"],
        remove_cols=cfg["data"].get("remove_cols")
    )
    mp_test = build_sharded_cache(
        parquet_path=cfg["data"]["test_path"],
        out_dir=os.path.join(cfg["data"]["cache_dir"], "test"),
        is_train=False,
        target_col=None,
        seq_col=cfg["sequence"]["col"],
        cat_cols=cfg["data"]["cat_cols"],
        hash_buckets=cfg["data"]["hash_buckets"],
        hash_buckets_margin=cfg["data"].get("hash_buckets_margin", 0),
        num_patterns=cfg["data"]["num_patterns"],
        num_cols_explicit=cfg["data"].get("num_cols_explicit"),
        max_len=cfg["sequence"]["max_len"],
        pad_id=cfg["sequence"]["pad_id"],
        group_key=cfg["cv"]["group_key"],
        time_key=cfg["cv"].get("time_key"),
        composite_group=bool(cfg["cv"].get("composite_group", False)),
        shard_rows=cfg["data"].get("shard_rows", 2_000_000),
        impute_strategy=cfg["data"]["impute_strategy"],
        remove_cols=cfg["data"].get("remove_cols")
    )
    return mp_train, mp_test

# import yaml,json; 
# from src.data.build_cache import build_train_and_test; 
# cfg=yaml.safe_load(open('cfgs/dare_qnn.yaml')); 
# mp_tr, mp_te = build_train_and_test(cfg); print(mp_tr, mp_te)