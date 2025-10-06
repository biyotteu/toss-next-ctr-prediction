# tools/blend_submissions.py
import argparse, numpy as np, pandas as pd

EPS = 1e-7

def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p/(1.0 - p))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def blend(df1, df2, method="logit_mean", w=0.5):
    # 안전 병합/검증
    df = pd.merge(df1, df2, on="ID", how="inner", validate="one_to_one", suffixes=("_1", "_2"))
    p1 = df["clicked_1"].astype(float).to_numpy()
    p2 = df["clicked_2"].astype(float).to_numpy()

    if method == "mean":
        p = w*p1 + (1.0 - w)*p2

    elif method == "logit_mean":
        z = w*logit(p1) + (1.0 - w)*logit(p2)
        p = sigmoid(z)

    elif method == "rank_mean":
        # 0~1로 정규화된 순위 평균
        r1 = pd.Series(p1).rank(method="average").to_numpy()
        r2 = pd.Series(p2).rank(method="average").to_numpy()
        r1 = (r1 - 1) / (len(r1) - 1 + 1e-12)
        r2 = (r2 - 1) / (len(r2) - 1 + 1e-12)
        p = w*r1 + (1.0 - w)*r2  # 순위 평균 자체를 확률로 씀(후속 캘리브레이션 가능)

    else:
        raise ValueError(f"Unknown method: {method}")

    p = np.clip(p, EPS, 1.0 - EPS)
    out = pd.DataFrame({"ID": df["ID"].astype(str), "clicked": p.astype(float)})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub1", required=True, help="첫 번째 제출 CSV 경로 (ID,clicked)")
    ap.add_argument("--sub2", required=True, help="두 번째 제출 CSV 경로 (ID,clicked)")
    ap.add_argument("--out",  required=True, help="출력 CSV 경로")
    ap.add_argument("--method", default="logit_mean", choices=["logit_mean","mean","rank_mean"])
    ap.add_argument("--w", type=float, default=0.5, help="첫 번째 제출물 가중치 (0~1)")
    args = ap.parse_args()

    # dtype 안전: ID는 문자열로 고정
    usecols = ["ID","clicked"]
    df1 = pd.read_csv(args.sub1, usecols=usecols, dtype={"ID":str, "clicked":float})
    df2 = pd.read_csv(args.sub2, usecols=usecols, dtype={"ID":str, "clicked":float})
    df1 = df1.rename(columns={"clicked":"clicked_1"})
    df2 = df2.rename(columns={"clicked":"clicked_2"})

    out = blend(df1, df2, method=args.method, w=args.w)

    # 간단한 진단(상관/분포)
    p1 = df1["clicked_1"].to_numpy(); p2 = df2["clicked_2"].to_numpy(); p = out["clicked"].to_numpy()
    corr = np.corrcoef(p1, p2)[0,1]
    print(f"[info] method={args.method}, w={args.w}")
    print(f"[info] corr(sub1, sub2)={corr:.4f}")
    print(f"[info] mean p1={p1.mean():.4f}, p2={p2.mean():.4f}, ens={p.mean():.4f}")

    out.to_csv(args.out, index=False)
    print(f"[✓] saved: {args.out}")

if __name__ == "__main__":
    main()
