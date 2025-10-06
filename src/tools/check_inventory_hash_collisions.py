#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inventory_id 해시 버킷 충돌률 점검 스크립트 (Polars 1.25+ 호환)

- 여러 parquet에서 inventory_id를 읽어 합친 뒤,
  안정적인 64-bit 해시(UInt64) → 모듈러 m으로 버킷 할당.
- 원래 고유치 n_unique와 버킷 고유치 unique_buckets를 비교해
  collision_ratio = 1 - (unique_buckets / n_unique) 계산.
- m 후보들을 한 번에 점검하고, 목표 충돌률에 맞춘 권장 m(근사)도 제시.

예:
  python tools/check_inventory_hash_collisions.py \
    --paths data/train.parquet data/test.parquet \
    --col inventory_id \
    --m 1000000 2000000 3000000 \
    --sample-frac 1.0 \
    --collision-thr 0.005 \
    --config config/config.yaml
"""

from __future__ import annotations
import argparse, math, os, sys, yaml
import polars as pl

# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=str, nargs="+", required=True, help="parquet 파일 경로들 (train/test 등)")
    ap.add_argument("--col", type=str, default="inventory_id", help="체크할 컬럼명")
    ap.add_argument("--m", type=int, nargs="*", default=[], help="점검할 버킷 사이즈 리스트 (예: 1000000 2000000)")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="0<frac<=1.0, 성능 위해 샘플링할 비율")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--collision-thr", type=float, default=0.005, help="권장 임계 충돌률 (예: 0.005 = 0.5%)")
    ap.add_argument("--config", type=str, default=None, help="YAML config에서 data.hash_buckets.inventory_id 기본값을 읽기")
    return ap.parse_args()

def read_config_default_m(cfg_path: str | None) -> int | None:
    if not cfg_path or not os.path.exists(cfg_path):
        return None
    try:
        cfg = yaml.safe_load(open(cfg_path))
        return (cfg or {}).get("data", {}).get("hash_buckets", {}).get("inventory_id", None)
    except Exception:
        return None

# -------------------------- Core logic --------------------------
def collect_unique_and_hashed(paths, col, sample_frac, seed):
    """
    여러 parquet에서 col만 뽑아 하나로 합치고,
    - lf_nuniq: 원래 고유치(문자열 기준)를 담은 LazyFrame (단일 스칼라열)
    - lf_hid  : 안정 해시(UInt64) 열 'hid' 를 담은 LazyFrame
    을 반환.
    """
    scans = []
    for p in paths:
        lf = pl.scan_parquet(p)
        schema = lf.collect_schema()
        if col not in schema.names():
            print(f"[warn] {p}: '{col}' 컬럼 없음 → 건너뜀", file=sys.stderr)
            continue
        # 문자열/정수 혼재 대비: 문자열로 통일 후 해시
        lf = lf.select(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
        scans.append(lf)
    if not scans:
        raise RuntimeError("입력 경로에 대상 컬럼이 없습니다.")

    lf_all = pl.concat(scans, how="vertical_relaxed")
    if sample_frac < 1.0:
        lf_all = lf_all.sample(fraction=sample_frac, with_replacement=False, seed=seed)

    # 원래 고유치 (스칼라)
    lf_nuniq = lf_all.select(pl.col(col).n_unique().alias("n_unique"))

    # 안정 해시(UInt64) 그대로 유지
    lf_hid = lf_all.select(pl.col(col).hash(seed=2025, seed_1=0).cast(pl.UInt64).alias("hid"))

    return lf_nuniq, lf_hid

def unique_buckets_for_m(lf_hid: pl.LazyFrame, m: int) -> int:
    """hid % m의 고유 버킷 수 (UInt64에서 직접 mod)"""
    mod_lit = pl.lit(m, dtype=pl.UInt64)
    res = (
        lf_hid
        .select((pl.col("hid") % mod_lit).alias("bucket"))
        .select(pl.col("bucket").n_unique().alias("unique_buckets"))
        .collect()   # Polars 1.25+: streaming 인자 사용 안 함
    )
    return int(res["unique_buckets"][0])

def n_unique_total(lf_nuniq: pl.LazyFrame) -> int:
    res = lf_nuniq.select(pl.col("n_unique").max()).collect()
    return int(res["n_unique"][0])

def print_report(n_unique: int, m_list: list[int], ub_dict: dict[int,int], thr: float):
    print("\n=== inventory_id 해시 버킷 충돌 점검 ===")
    print(f"- 관측 고유치 (n_unique): {n_unique:,}")
    print(f"- 임계 충돌률(threshold): {thr:.4%}")
    print("- 후보 m 및 충돌률:")
    for m in sorted(m_list):
        ub = ub_dict[m]
        ratio = 1.0 - (ub / max(1, n_unique))
        print(f"  m={m:,}  unique_buckets={ub:,}  collision_ratio={ratio:.4%}")

def recommend_m(n_unique: int, target_ratio: float) -> int:
    """
    목표 충돌률 r (ex: 0.005)
    충돌률 r = 1 - U/n,  U ≈ m*(1 - exp(-n/m)) (balls-in-bins 근사)
    → m을 이 식으로 수치적으로 찾는다.
    """
    if n_unique <= 1:
        return 1
    n = float(n_unique)
    desired_U = n * (1.0 - target_ratio)
    lo = max(1.0, n * 0.05)
    hi = n * 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        U_est = mid * (1.0 - math.exp(-n / mid))
        if U_est >= desired_U:
            hi = mid
        else:
            lo = mid
    m_est = int(math.ceil(hi))
    unit = 500_000
    m_round = int(math.ceil(m_est / unit) * unit)
    return max(1, m_round)

# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    # config에서 기본 m 읽기(있으면)
    m_from_cfg = read_config_default_m(args.config)
    m_list = list(dict.fromkeys([*(args.m or []), *( [m_from_cfg] if m_from_cfg else [] )]))
    if not m_list:
        m_list = [1020, 1020, 500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]

    # 두 개의 LazyFrame으로 받으세요 (중요!)
    lf_nuniq, lf_hid = collect_unique_and_hashed(args.paths, args.col, args.sample_frac, args.seed)

    n_u = n_unique_total(lf_nuniq)

    ub = {}
    for m in m_list:
        ub[m] = unique_buckets_for_m(lf_hid, m)

    print_report(n_u, m_list, ub, args.collision_thr)

    rec = recommend_m(n_u, args.collision_thr)
    print(f"\n=> 목표 충돌률 {args.collision_thr:.2%}를 만족하도록 한 권장 m (근사): {rec:,}")
    ok = [m for m in sorted(m_list) if (1 - (ub[m] / max(1, n_u))) <= args.collision_thr]
    if ok:
        print(f"=> 후보 중 기준 충족 최소 m: {ok[0]:,}")
    else:
        print("=> 현재 후보 중 기준 충족 값이 없습니다. 위 권장 m을 고려하세요.")

if __name__ == "__main__":
    main()
