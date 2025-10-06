# tools/guess_numeric_types.py
"""
Train 데이터에서 각 컬럼이
  - 연속형(float)인지,
  - 실수로 저장된 정수(=카테고리)인지
판별하고,
카테고리로 판정되면 권장 vocab size(여유분 포함)과 임베딩 차원을 추정.

출력:
  - outputs/column_type_report.csv
  - outputs/column_suggestions.yaml
"""

from __future__ import annotations
import argparse, os, math, yaml
from typing import Dict, List
import polars as pl

# ---- 임베딩 차원 규칙 (config의 model.emb_dim_rule과 동일 로직) ----
def dim_rule(card: int, rule_cfg: Dict) -> int:
    t = (rule_cfg or {}).get("type", "sqrt")
    min_d = int((rule_cfg or {}).get("min_d", 8))
    max_d = int((rule_cfg or {}).get("max_d", 64))
    if t == "sqrt":
        d = int(math.ceil(math.sqrt(max(2, card))))
    elif t == "log":
        d = int(math.ceil(math.log2(max(2, card)) + 1))
    else:
        d = int((rule_cfg or {}).get("default", 32))
    return max(min_d, min(max_d, d))

def is_float_dtype(dt) -> bool:
    return dt in (pl.Float32, pl.Float64)

def is_int_dtype(dt) -> bool:
    return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

def analyze(train_path: str,
            label_col: str,
            seq_col: str,
            eps: float = 1e-6,
            int_like_thr: float = 0.999,
            vocab_margin_frac: float = 0.02,
            vocab_min_extra: int = 1000,
            emb_dim_rule_cfg: Dict = None,
            drop_cols: List[str] | None = None,
            drop_prefixes: List[str] | None = None):

    os.makedirs("outputs", exist_ok=True)

    # 스키마 수집
    schema = pl.scan_parquet(train_path).collect_schema()
    all_cols = schema.names()
    dtypes = schema.dtypes()

    # 제외할 컬럼(라벨/시퀀스/명시 드롭)
    drop_cols = set(drop_cols or [])
    for pfx in (drop_prefixes or []):
        for c in all_cols:
            if c.startswith(pfx):
                drop_cols.add(c)

    targets = []
    for c, dt in zip(all_cols, dtypes):
        if c in (label_col, seq_col):  # 라벨/시퀀스 제외
            continue
        if c in drop_cols:
            continue
        # 수치/문자 모두 후보(문자는 나중에 수치 캐스팅 시도)
        targets.append((c, dt))

    reports = []
    for c, dt in targets:
        lf = pl.scan_parquet(train_path)

        # 숫자 캐스팅 컬럼(문자열도 수치형으로 바꿔보기; 캐스팅 실패값은 null)
        num_col = pl.col(c).cast(pl.Float64, strict=False)

        # 기본 통계 (count/nulls/n_unique + 숫자형 min/max)
        exprs = [
            pl.len().alias("count"),
            pl.col(c).null_count().alias("null_count"),
            pl.col(c).n_unique().alias("n_unique"),
            # 문자열일 수 있으므로 원래 min/max는 쓰지 않고, 숫자 캐스팅한 min/max만 사용
            num_col.min().alias("num_min"),
            num_col.max().alias("num_max"),
        ]

        # 정수 유사도: |x - round(x)| <= eps (가능한 경우에만)
        if is_int_dtype(dt):
            int_like_expr = pl.lit(True)
        else:
            int_like_expr = (num_col.round(0) - num_col).abs() <= eps
        exprs.append(int_like_expr.cast(pl.Float64).mean().alias("frac_int_like"))

        # 최신 Polars: streaming 인자 deprecated → collect() 만 사용
        stats = lf.select(exprs).collect().to_dicts()[0]

        n = int(stats["count"])
        nulls = int(stats["null_count"])
        non_null = max(0, n - nulls)
        n_unique = int(stats["n_unique"])
        vmin_num = stats["num_min"]
        vmax_num = stats["num_max"]
        frac_int = float(stats["frac_int_like"]) if stats["frac_int_like"] is not None else 0.0
        unique_ratio = (n_unique / non_null) if non_null > 0 else 0.0

        # 판정 로직
        inferred = "continuous"
        rec_vocab = None
        rec_emb_dim = None
        est_params = None
        notes = []

        if non_null == 0:
            inferred = "empty"
            notes.append("all_null")
        else:
            # 실수로 저장된 정수? (또는 원래 정수형?)
            if (frac_int >= int_like_thr) or is_int_dtype(dt):
                inferred = "int_like_categorical"
                # 권장 vocab (pad=0, OOV=1 가정, 여유 마진 적용)
                margin_extra = max(int(n_unique * vocab_margin_frac), vocab_min_extra)
                rec_vocab = int(n_unique + 2 + margin_extra)

                # 임베딩 차원 제안
                rec_emb_dim = dim_rule(n_unique, emb_dim_rule_cfg or {"type": "sqrt", "min_d": 8, "max_d": 64})
                est_params = int(rec_vocab * rec_emb_dim)

                if unique_ratio > 0.5:
                    notes.append("very_high_unique_ratio(>0.5) — ID 가능성 큼; 메모리/학습시간 주의")
                if n_unique <= 2:
                    notes.append("binary_like")
            else:
                inferred = "continuous"
                # 숫자 범위가 있고 고유값 비율이 매우 낮으면 binning+embedding 제안
                if (vmin_num is not None) and (vmax_num is not None):
                    span = vmax_num - vmin_num
                    if span != 0 and unique_ratio < 0.001:
                        notes.append("low_unique_ratio_continuous — quantile binning + embedding 고려")

        reports.append({
            "col": c,
            "dtype": str(dt),
            "count": n,
            "null_count": nulls,
            "non_null": non_null,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "num_min": vmin_num,
            "num_max": vmax_num,
            "frac_int_like": frac_int,
            "inferred": inferred,
            "rec_vocab": rec_vocab,
            "rec_emb_dim": rec_emb_dim,
            "est_emb_params": est_params,
            "notes": ";".join(notes),
        })

    # CSV 저장
    df_out = pl.DataFrame(reports)
    out_csv = "outputs/column_type_report.csv"
    df_out.write_csv(out_csv)

    # YAML 제안 (config에 바로 붙여넣기 용)
    cat_include = [r["col"] for r in reports if r["inferred"] == "int_like_categorical"]
    dense_include = [r["col"] for r in reports if r["inferred"] == "continuous"]
    suggestions = {
        "cat_include": cat_include,
        "dense_include": dense_include,
        "embedding_recommendations": {
                r["col"]: {
                    "recommended_vocab": int(r["rec_vocab"]) if r["rec_vocab"] is not None else None,
                    "recommended_emb_dim": int(r["rec_emb_dim"]) if r["rec_emb_dim"] is not None else None,
                    "estimated_params": int(r["est_emb_params"]) if r["est_emb_params"] is not None else None,
                    "n_unique_observed": int(r["n_unique"]),
                    "notes": r["notes"] or None,
                }
                for r in reports if r["inferred"] == "int_like_categorical"
        }
    }
    out_yaml = "outputs/column_suggestions.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(suggestions, f, allow_unicode=True, sort_keys=False)

    print(f"[✓] Saved: {out_csv}")
    print(f"[✓] Saved: {out_yaml}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", type=str, default="data/train.parquet")
    ap.add_argument("--label", type=str, default="clicked")
    ap.add_argument("--seq", type=str, default="seq")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--int-like-thr", type=float, default=0.999)
    ap.add_argument("--vocab-margin-frac", type=float, default=0.02)
    ap.add_argument("--vocab-min-extra", type=int, default=1000)
    ap.add_argument("--config", type=str, default="config/config.yaml",
                    help="임베딩 차원 규칙(model.emb_dim_rule) 읽을 때 사용(없으면 기본 sqrt)")
    ap.add_argument("--drop-cols", type=str, nargs="*", default=None)
    ap.add_argument("--drop-prefixes", type=str, nargs="*", default=None)
    args = ap.parse_args()

    emb_dim_rule_cfg = None
    if os.path.exists(args.config):
        try:
            cfg = yaml.safe_load(open(args.config))
            emb_dim_rule_cfg = (cfg or {}).get("model", {}).get("emb_dim_rule", None)
        except Exception:
            emb_dim_rule_cfg = None

    analyze(
        train_path=args.train_path,
        label_col=args.label,
        seq_col=args.seq,
        eps=args.eps,
        int_like_thr=args.int_like_thr,
        vocab_margin_frac=args.vocab_margin_frac,
        vocab_min_extra=args.vocab_min_extra,
        emb_dim_rule_cfg=emb_dim_rule_cfg,
        drop_cols=args.drop_cols,
        drop_prefixes=args.drop_prefixes,
    )

if __name__ == "__main__":
    main()
