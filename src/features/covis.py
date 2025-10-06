from __future__ import annotations
import os, math, itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import polars as pl
from tqdm.auto import tqdm

# ---------- 설정 ----------
@dataclass
class CoVisCfg:
    train_path: str
    test_path: str
    seq_col: str = "seq"
    id_col_test: str = "ID"

    target_keys: List[str] = None
    use_time_bin: bool = True
    time_bin: str = "day_of_week"  # or "hour" or "day_of_week_hour"

    seq_top_k: int = 120
    recency_tau: int = 512

    min_impr: int = 10
    prior_strength: int = 50
    ctr_clip: Tuple[float,float] = (1e-3, 0.999)
    backoff: List[str] = None

    agg_topn: int = 3
    agg_outputs: List[str] = None

    n_folds: int = 5
    group_key: str = "inventory_id"
    time_key: Optional[str] = "day_of_week"
    composite_group: bool = True

    work_dir: str = "./cache/covis"

    def __post_init__(self):
        if self.target_keys is None:
            self.target_keys = ["inventory_id"]
        if self.backoff is None:
            self.backoff = ["pair","token","target","global"]
        if self.agg_outputs is None:
            self.agg_outputs = ["sum_ctr","mean_ctr","max_ctr","top3_mean_ctr","wmean_ctr","sum_impr","max_impr","pnorm_ctr"]


# ---------- 유틸 ----------
def _seq_to_topk_expr(seq_col: str, k: int) -> pl.Expr:
    """
    '8,72,86,...' -> [int]
    뒤에서부터 k개를 취하는 구현 (가장 최신 로그를 더 반영하는 형태)
    """
    return (
        pl.when(pl.col(seq_col).is_null())
          .then(pl.lit([]))
          .otherwise(pl.col(seq_col).cast(pl.Utf8)
                   .str.split_exact(",", strict=False, include_remaining=True)
                   .arr.eval(pl.element().cast(pl.Int32), parallel=True)
                   .arr.explode()  # 하나씩 풀고
                   .over(pl.int_ranges(0, pl.len()))  # dummy trick
          )
    )

def _parse_seq_topk(df: pl.LazyFrame, seq_col: str, top_k: int) -> pl.LazyFrame:
    """
    '8,72,86,...' → 리스트 → 뒤에서 top_k만 취해서 다시 리스트로.
    list 네임스페이스 또는 explode+cum_count 방식으로 구현.
    """
    return (
        df.with_columns([
            pl.when(pl.col(seq_col).is_null())
              .then(pl.lit([], dtype=pl.List(pl.Utf8)))
              .otherwise(pl.col(seq_col).cast(pl.Utf8).str.split(","))
              .alias("_tok_str")
        ])
        .with_columns([
            pl.col("_tok_str").list.eval(pl.element().cast(pl.Int32, strict=False)).alias("_tok_int")
        ])
        .with_columns([
            pl.col("_tok_int").list.reverse().list.head(top_k).list.reverse().alias("seq_topk")
        ])
        .drop(["_tok_str","_tok_int"])
    )
    
def _recency_weights_expr(topk_col: str, tau: int) -> pl.Expr:
    """
    seq_topk의 길이가 L일 때, 뒤에서부터의 위치 pos=0..L-1에 대해 w=exp(-pos/tau)
    """
    return (
        pl.when(pl.col(topk_col).list.len() == 0)
          .then(pl.lit([], dtype=pl.List(pl.Float32)))
          .otherwise(
              pl.arange(0, pl.col(topk_col).list.len()).cast(pl.Float32)
                .arr.eval(pl.element().cast(pl.Float32))
                .map_elements(lambda a: float(np.exp(-a / float(tau))), return_dtype=pl.List(pl.Float32))
          )
    )


def _make_timebin_expr(cfg: CoVisCfg) -> Optional[pl.Expr]:
    if not cfg.use_time_bin:
        return None
    if cfg.time_bin == "day_of_week_hour":
        return (pl.col("day_of_week").cast(pl.Int32)*24 + pl.col("hour").cast(pl.Int32)).alias("time_bin")
    else:
        return pl.col(cfg.time_bin).cast(pl.Int32).alias("time_bin")


def _beta_smooth_ctr(clicks: pl.Expr, impr: pl.Expr, p0: float, S: int) -> pl.Expr:
    alpha = p0 * S
    beta = (1.0 - p0) * S
    return ((clicks + alpha) / (impr + alpha + beta)).clip_min(1e-9).clip_max(1-1e-9)


# ---------- 1) Fold 생성 (캐시의 그룹 전략과 일치) ----------
def make_folds(cfg: CoVisCfg) -> pl.DataFrame:
    lf = pl.scan_parquet(cfg.train_path)

    # 그룹키 & 시간키를 결합
    if cfg.composite_group and cfg.time_key is not None:
        g_expr = pl.struct([
            pl.col(cfg.group_key).cast(pl.Utf8).fill_null("NA"),
            pl.col(cfg.time_key).cast(pl.Utf8).fill_null("NA")
        ]).hash(seed=2025, seed_1=0)
    else:
        g_expr = pl.col(cfg.group_key).cast(pl.Utf8).fill_null("NA").hash(seed=2025, seed_1=0)

    lf = lf.select([
        pl.len().alias("_n"),
        g_expr.alias("g"),
    ]).with_row_index("rid")
    df = lf.collect()
    # 해시 그룹별 라운드로빈으로 fold 배정 (근사 GroupKFold)
    ng = df["g"].n_unique()
    # 안정적 분배: 그룹 해시값을 정렬해 round-robin
    g2fold = (
        df.select(["g"])
          .unique()
          .sort("g")
          .with_row_index("ord")
          .with_columns((pl.col("ord") % cfg.n_folds).alias("fold"))
          .drop("ord")
    )
    out = (
        df.join(g2fold, on="g", how="left")
          .select(["rid","fold"])
          .sort("rid")
    )
    return out


# ---------- 2) Pair 통계 구축 (train 전체 / OOF) ----------
def _pair_stats_from_scan(
    scan: pl.LazyFrame,
    cfg: CoVisCfg,
    targets: List[str],
    use_oof_mask: Optional[pl.Series] = None,   # rid 기준 True=사용
) -> pl.DataFrame:
    """
    (token, target[, time_bin]) → (impr, clicks, ctr_smoothed, last_seen_time 등)
    """
    # rid 추가(OOF 마스킹용)
    scan = scan.with_row_index("rid")

    # OOF 마스크가 있으면 필터링
    if use_oof_mask is not None:
        mask_lf = pl.LazyFrame(pl.DataFrame({"rid": use_oof_mask.index, "_keep": use_oof_mask.values}))
        scan = scan.join(mask_lf, on="rid", how="left").filter(pl.col("_keep") == True).drop("_keep")

    timebin_expr = _make_timebin_expr(cfg)

    lf = _parse_seq_topk(scan, cfg.seq_col, cfg.seq_top_k)
    lf = lf.with_row_index("rid") \
           .select(["rid","clicked"] + targets + (["day_of_week","hour"] if cfg.use_time_bin else []) + ["seq_topk"])
    # explode → pos = cum_count() over rid (0,1,2,...)
    lf = lf.explode("seq_topk").rename({"seq_topk":"token"})
    lf = lf.with_columns([
        (pl.col("token").cum_count().over("rid") - 1).alias("pos")
    ])
    lf = lf.with_columns([
        (-pl.col("pos") / float(cfg.recency_tau)).exp().alias("w_rec")
    ])
    # token은 최종적으로 int
    lf = lf.with_columns(pl.col("token").cast(pl.Int32, strict=False))

    if timebin_expr is not None:
        lf = lf.with_columns([ timebin_expr ])

    # group by key
    keys = ["token"] + targets + (["time_bin"] if cfg.use_time_bin else [])
    grp = lf.group_by(keys).agg([
        pl.len().alias("impr"),
        pl.col("clicked").sum().alias("clicks"),
        pl.col("w_rec").sum().alias("w_rec_sum"),
        pl.max("pos").alias("max_pos"),
    ])

    # 글로벌 p0
    # NOTE: 전체 스캔에서 계산(OOF면 그 subset 내 p0)
    smry = lf.select(pl.col("clicked").mean().alias("p0")).collect()
    p0 = float(smry["p0"][0]) if smry.height > 0 else 0.019

    # smoothing + clip + min_impr guard
    out = grp.with_columns([
        _beta_smooth_ctr(pl.col("clicks"), pl.col("impr"), p0=p0, S=cfg.prior_strength).alias("ctr"),
        pl.when(pl.col("impr") < cfg.min_impr).then(pl.lit(True)).otherwise(pl.lit(False)).alias("is_lowcount"),
    ]).with_columns([
        pl.col("ctr").clip(cfg.ctr_clip[0], cfg.ctr_clip[1]).alias("ctr")
    ])
    return out.collect()


def build_pair_stats_all(cfg: CoVisCfg, folds: pl.DataFrame):
    os.makedirs(cfg.work_dir, exist_ok=True)
    tr = pl.scan_parquet(cfg.train_path)

    # Full-train (test용)
    for tgt in tqdm(cfg.target_keys, desc="Building pair stats for full-train"):
        tbl = _pair_stats_from_scan(tr, cfg, [tgt], use_oof_mask=None)
        tbl.write_parquet(os.path.join(cfg.work_dir, f"pair_full_{tgt}.parquet"))

    # OOF per fold (val용)
    # fold별로 "학습 파트"만 남기는 마스크
    rid2fold = folds.sort("rid")
    N = rid2fold.height
    idx = rid2fold["rid"].to_numpy()
    fld = rid2fold["fold"].to_numpy()
    for f in tqdm(range(cfg.n_folds), desc="Building pair stats for OOF"):
        keep = (fld != f)  # 해당 fold를 제외한 나머지로 통계
        mask = pl.Series(name="rid", values=idx)
        mkeep = pl.Series(name="_keep", values=keep)
        mdf = pl.DataFrame({"rid": mask, "_keep": mkeep})
        for tgt in tqdm(cfg.target_keys, desc=f"Building pair stats for OOF fold {f}"):
            tbl = _pair_stats_from_scan(tr, cfg, [tgt], use_oof_mask=mdf.set_index("rid")["_keep"])
            tbl.write_parquet(os.path.join(cfg.work_dir, f"pair_oof_f{f}_{tgt}.parquet"))


# ---------- 3) 행 단위 피처 생성(OOF & Test) ----------
def _row_features_from_pair_tbl(
    scan: pl.LazyFrame,
    pair_tbls: Dict[str, pl.LazyFrame],
    cfg: CoVisCfg,
    is_test: bool,
) -> pl.DataFrame:
    """
    각 행에 대해 seq_topk를 폭발 → (token × target)와 조인 → 집계 피처 산출
    """
    scan = scan.with_row_index("rid")
    timebin_expr = _make_timebin_expr(cfg)
    # seq → tokens + pos + recency weight
    lf = _parse_seq_topk(scan, cfg.seq_col, cfg.seq_top_k).with_columns([
        pl.col("seq_topk").alias("_tok_list"),
        pl.when(pl.col("seq_topk").list.len()==0).then(pl.lit([], dtype=pl.List(pl.Float32)))
         .otherwise(pl.arange(0, pl.col("seq_topk").list.len()).cast(pl.Float32)).alias("_pos_idx"),
    ]).select(["rid"] + cfg.target_keys + ["clicked"]*(0 if is_test else 1) + (["day_of_week","hour"] if cfg.use_time_bin else []) + ["_tok_list","_pos_idx"])

    lf = lf.with_row_index("rid") \
           .select(["rid"] + cfg.target_keys + (["clicked"] if not is_test else []) \
                   + (["day_of_week","hour"] if cfg.use_time_bin else []) + ["seq_topk"])
    lf = lf.explode("seq_topk").rename({"seq_topk":"token"})
    lf = lf.with_columns([
        (pl.col("token").cum_count().over("rid") - 1).alias("pos")
    ])
    lf = lf.with_columns([
        (-pl.col("pos") / float(cfg.recency_tau)).exp().alias("w_rec")
    ])
    lf = lf.with_columns(pl.col("token").cast(pl.Int32, strict=False))
    if timebin_expr is not None:
        lf = lf.with_columns([ timebin_expr ])

    outs: List[pl.LazyFrame] = []
    for tgt in tqdm(cfg.target_keys, desc="Building row features for OOF"):
        keys = ["token", tgt] + (["time_bin"] if cfg.use_time_bin else [])
        join_tbl = pair_tbls[tgt]  # LazyFrame

        j = lf.join(join_tbl, on=keys, how="left") \
             .with_columns([
                 pl.col("impr").fill_null(0).alias(f"impr_{tgt}"),
                 pl.col("ctr").fill_null(None).alias(f"ctr_{tgt}"),
             ]) \
             .select(["rid","w_rec", f"impr_{tgt}", f"ctr_{tgt}"])

        # 집계
        aggs = []
        if "sum_ctr" in cfg.agg_outputs:
            aggs.append(pl.col(f"ctr_{tgt}").sum().alias(f"{tgt}_sum_ctr"))
        if "mean_ctr" in cfg.agg_outputs:
            aggs.append(pl.col(f"ctr_{tgt}").mean().alias(f"{tgt}_mean_ctr"))
        if "max_ctr" in cfg.agg_outputs:
            aggs.append(pl.col(f"ctr_{tgt}").max().alias(f"{tgt}_max_ctr"))
        if "top3_mean_ctr" in cfg.agg_outputs:
            aggs.append(pl.col(f"ctr_{tgt}").sort(descending=True).head(cfg.agg_topn).mean().alias(f"{tgt}_top{cfg.agg_topn}_mean_ctr"))
        if "wmean_ctr" in cfg.agg_outputs:
            aggs.append((pl.col(f"ctr_{tgt}") * pl.col("w_rec")).sum() / pl.col("w_rec").sum().alias(f"{tgt}_wmean_ctr"))
        if "sum_impr" in cfg.agg_outputs:
            aggs.append(pl.col(f"impr_{tgt}").sum().alias(f"{tgt}_sum_impr"))
        if "max_impr" in cfg.agg_outputs:
            aggs.append(pl.col(f"impr_{tgt}").max().alias(f"{tgt}_max_impr"))
        if "pnorm_ctr" in cfg.agg_outputs:
            aggs.append(pl.col(f"ctr_{tgt}").pow(2.0).mean().pow(0.5).alias(f"{tgt}_pnorm_ctr"))

        jf = j.group_by("rid").agg(aggs)
        outs.append(jf)

    # target별 피처를 행 기준으로 합치기
    feat = outs[0]
    for k in range(1, len(outs)):
        feat = feat.join(outs[k], on="rid", how="outer")
    # NaN → 0/중립치
    feat = feat.fill_null(0.0).with_columns([pl.col("rid")]).sort("rid")
    return feat.collect()


def build_row_features_oof_and_test(cfg: CoVisCfg, folds: pl.DataFrame):
    os.makedirs(cfg.work_dir, exist_ok=True)
    # pair 테이블을 LazyFrame으로 준비
    for f in tqdm(range(cfg.n_folds), desc="Building row features for OOF"):
        pair_tbls = {}
        for tgt in cfg.target_keys:
            path = os.path.join(cfg.work_dir, f"pair_oof_f{f}_{tgt}.parquet")
            pair_tbls[tgt] = pl.scan_parquet(path)
        # 해당 fold의 VAL 행만 추출하여 조인 → OOF 피처
        tr = pl.scan_parquet(cfg.train_path)
        rid_mask = folds.filter(pl.col("fold")==f)["rid"].to_numpy()
        mask_df = pl.DataFrame({"rid": rid_mask, "_keep": np.ones(len(rid_mask), dtype=bool)})
        tr_f = tr.with_row_index("rid").join(mask_df.lazy(), on="rid", how="inner").drop("_keep")
        feat = _row_features_from_pair_tbl(tr_f, pair_tbls, cfg, is_test=False)
        feat.write_parquet(os.path.join(cfg.work_dir, f"rowfeat_oof_f{f}.parquet"))

    # TEST: full pair로 조인
    pair_tbls_full = {}
    for tgt in tqdm(cfg.target_keys, desc="Building row features for full-train"):
        pair_tbls_full[tgt] = pl.scan_parquet(os.path.join(cfg.work_dir, f"pair_full_{tgt}.parquet"))
    te = pl.scan_parquet(cfg.test_path)
    feat_te = _row_features_from_pair_tbl(te, pair_tbls_full, cfg, is_test=True)
    # test는 ID로 조인할 예정이라 rid→ID로 치환(동일 순서라고 가정하지 않음)
    ids = pl.scan_parquet(cfg.test_path).select([pl.col(cfg.id_col_test)]).with_row_index("rid").collect()
    feat_te = feat_te.join(ids, on="rid", how="left").drop("rid").rename({cfg.id_col_test: "ID"})
    feat_te.write_parquet(os.path.join(cfg.work_dir, f"rowfeat_test.parquet"))
