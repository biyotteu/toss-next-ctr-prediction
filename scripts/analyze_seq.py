import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze seq column statistics from a parquet file.")
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(Path("data") / "train.parquet"),
        help="Path to parquet file (default: data/train.parquet)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="seq",
        help="Name of the column containing comma-separated integer sequences (default: seq)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path("artifacts")),
        help="Directory to write outputs (default: artifacts)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Top-K tokens to include in a separate summary CSV (default: 100)",
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=None,
        help="Optionally limit to the first N rows for a quick analysis (default: all)",
    )
    return parser.parse_args()


def analyze_sequences(series: pd.Series) -> Tuple[List[int], Counter, int, int, int]:
    """Analyze sequences represented as comma-separated integers in a pandas Series.

    Returns:
        lengths: list of sequence lengths per row
        token_counter: Counter of token -> frequency across all rows
        null_count: number of NaN/None values
        empty_string_count: number of empty-string entries
        invalid_token_count: number of tokens that failed to parse as integers
    """
    lengths: List[int] = []
    token_counter: Counter = Counter()
    null_count = 0
    empty_string_count = 0
    invalid_token_count = 0

    # Iterate row-wise for robust parsing and frequency counting
    for value in series:
        if pd.isna(value):
            null_count += 1
            lengths.append(0)
            continue

        text = str(value).strip()
        if text == "":
            empty_string_count += 1
            lengths.append(0)
            continue

        tokens = text.split(",")
        valid_token_count = 0
        for raw in tokens:
            token_text = raw.strip()
            if token_text == "":
                # Skip empty fragments like trailing commas
                continue
            try:
                token_int = int(token_text)
                token_counter[token_int] += 1
                valid_token_count += 1
            except Exception:
                invalid_token_count += 1
        lengths.append(valid_token_count)

    return lengths, token_counter, null_count, empty_string_count, invalid_token_count


def compute_length_summary(lengths: List[int]) -> dict:
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        return {}
    summary = {
        "count": int(arr.size),
        "min": int(arr.min(initial=0)),
        "max": int(arr.max(initial=0)),
        "mean": float(arr.mean()) if arr.size > 0 else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)) if arr.size > 0 else 0.0,
        "p90": float(np.percentile(arr, 90)) if arr.size > 0 else 0.0,
        "p95": float(np.percentile(arr, 95)) if arr.size > 0 else 0.0,
        "p99": float(np.percentile(arr, 99)) if arr.size > 0 else 0.0,
    }
    return summary


def save_length_distribution(lengths: List[int], outpath: Path) -> None:
    length_counter = Counter(lengths)
    dist_df = (
        pd.DataFrame({"length": list(length_counter.keys()), "count": list(length_counter.values())})
        .sort_values(["length"])  # ascending by length
        .reset_index(drop=True)
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(outpath, index=False)


def save_token_frequencies(token_counter: Counter, out_csv: Path, topk_csv: Path, topk: int) -> None:
    items = sorted(token_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    df_all = pd.DataFrame(items, columns=["token", "count"])  # full frequency table
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)

    if topk is not None and topk > 0:
        df_topk = df_all.head(topk)
        df_topk.to_csv(topk_csv, index=False)


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Read only the target column to minimize memory usage
    use_cols = [args.column]
    df = pd.read_parquet(parquet_path, columns=use_cols)
    if args.sample_rows is not None and args.sample_rows > 0:
        df = df.head(args.sample_rows)

    lengths, token_counter, null_count, empty_count, invalid_token_count = analyze_sequences(df[args.column])

    # Compute and print summary
    length_summary = compute_length_summary(lengths)

    print("=== Seq Length Summary ===")
    if length_summary:
        for k in ["count", "min", "max", "mean", "std", "p50", "p90", "p95", "p99"]:
            print(f"{k:>5}: {length_summary.get(k)}")
    else:
        print("No data.")

    print("\n=== Data Quality ===")
    total_rows = len(df)
    print(f"total_rows           : {total_rows}")
    print(f"null_count           : {null_count}")
    print(f"empty_string_count   : {empty_count}")
    print(f"invalid_token_count  : {invalid_token_count}")

    # Save distributions
    save_length_distribution(lengths, outdir / "seq_length_distribution.csv")
    save_token_frequencies(
        token_counter,
        out_csv=outdir / "token_frequency.csv",
        topk_csv=outdir / f"token_frequency_top{args.topk}.csv",
        topk=args.topk,
    )

    # Save a compact JSON summary for easy reuse
    summary_payload = {
        "parquet": str(parquet_path),
        "column": args.column,
        "total_rows": total_rows,
        "null_count": null_count,
        "empty_string_count": empty_count,
        "invalid_token_count": invalid_token_count,
        "length_summary": length_summary,
        "num_unique_tokens": int(len(token_counter)),
    }
    (outdir / "seq_analysis_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2))

    # Show top-10 tokens inline for quick inspection
    top10 = token_counter.most_common(10)
    print("\n=== Top 10 Tokens ===")
    if top10:
        for token, cnt in top10:
            print(f"{token:>8} : {cnt}")
    else:
        print("No tokens found.")


if __name__ == "__main__":
    main()
