from __future__ import annotations
import argparse, os, yaml
from src.features.covis import CoVisCfg, make_folds, build_pair_stats_all, build_row_features_oof_and_test
import polars as pl

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    d = cfg["data"]; s = cfg["sequence"]; fcfg = cfg["features"]["covis"]

    c = CoVisCfg(
        train_path=d["train_path"],
        test_path=d["test_path"],
        seq_col=s["col"],
        id_col_test="ID",
        target_keys=fcfg["target_keys"],
        use_time_bin=fcfg["use_time_bin"],
        time_bin=fcfg["time_bin"],
        seq_top_k=fcfg["seq_top_k"],
        recency_tau=fcfg["recency_tau"],
        min_impr=fcfg["min_impr"],
        prior_strength=fcfg["prior_strength"],
        ctr_clip=tuple(fcfg["ctr_clip"]),
        backoff=fcfg["backoff"],
        agg_topn=int(fcfg["agg"]["topn"]),
        agg_outputs=fcfg["agg"]["outputs"],
        n_folds=cfg["cv"]["n_splits"],
        group_key=cfg["cv"]["group_key"],
        time_key=cfg["cv"].get("time_key"),
        composite_group=bool(cfg["cv"].get("composite_group", False)),
        work_dir=fcfg["work_dir"],
    )

    folds = make_folds(c)
    os.makedirs(c.work_dir, exist_ok=True)
    folds.write_parquet(os.path.join(c.work_dir, "folds.parquet"))

    build_pair_stats_all(c, folds)
    build_row_features_oof_and_test(c, folds)
    print("[✓] CoVis features built:", c.work_dir)

    paths = [os.path.join(c.work_dir, f"rowfeat_oof_f{f}.parquet") for f in range(c.n_folds)]
    feat_all = pl.concat([pl.scan_parquet(p) for p in paths]).collect()
    feat_all.write_parquet(os.path.join(c.work_dir, "rowfeat_oof_all.parquet"))
    print("[✓] Wrote:", os.path.join(c.work_dir, "rowfeat_oof_all.parquet"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()
    main(args.cfg)
