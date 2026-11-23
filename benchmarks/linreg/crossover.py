"""Crossover grid for patched dispatch: worst data kind per shape, mlxlearn Layer 1 vs stock.

Each shape is timed on data that exercises every solver path (Gaussian -> Gram,
one-hot -> Ritz, mixed scales -> Ritz/exact, cond10 -> exact), gapped and
alternating. Output: JSONL, one record per (shape, data).
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_linreg import make_data  # noqa: E402


def timeit(fn, rounds, pause):
    ts = []
    for r in range(rounds + 1):
        gc.collect()
        time.sleep(pause)
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if r:
            ts.append(dt)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", choices=["tall", "wide", "smoke"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pause", type=float, default=0.2)
    args = ap.parse_args()
    import sklearn.linear_model as skl

    import mlxlearn.linear_model as mll

    if args.grid == "tall":
        shapes = [(n, d) for n in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
                  for d in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n >= 2 * d]
    elif args.grid == "wide":
        shapes = [(n, m * n) for n in (32, 64, 128, 256, 512, 1024, 2048) for m in (2, 4, 8)]
    else:
        shapes = [(4096, 32), (512, 2048)]
    kinds = ["gauss", "onehot", "mixed", "cond10"]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["n"], r["d"], r["data"], r.get("intercept", True)))
    for n, d in shapes:
        for kind in kinds:
            for intercept in ((True, False) if args.grid == "wide" else (True,)):
                if (n, d, kind, intercept) in done:
                    continue
                X, y, _ = make_data({"data": kind, "n": n, "d": d})

                def fit_stock(X=X, y=y, intercept=intercept):
                    return skl.LinearRegression(fit_intercept=intercept).fit(X, y)

                def fit_mlx(X=X, y=y, intercept=intercept):
                    return mll.LinearRegression(fit_intercept=intercept).fit(X, y)

                sk = timeit(fit_stock, args.rounds, args.pause)
                ml = timeit(fit_mlx, args.rounds, args.pause)
                # a second pass in the opposite order, so neither variant always runs first
                ml += timeit(fit_mlx, args.rounds, args.pause)
                sk += timeit(fit_stock, args.rounds, args.pause)
                m = mll.LinearRegression(fit_intercept=intercept).fit(X, y)
                r = skl.LinearRegression(fit_intercept=intercept).fit(X, y)
                rec = {"n": n, "d": d, "data": kind, "intercept": intercept,
                       "sk_ms": float(np.median(sk)) * 1e3, "mlx_ms": float(np.median(ml)) * 1e3,
                       "rank": int(m.rank_), "rank_ref": int(r.rank_)}
                rec["speedup"] = rec["sk_ms"] / rec["mlx_ms"]
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{n}x{d} {kind} icpt={intercept}: sk {rec['sk_ms']:.2f} mlx {rec['mlx_ms']:.2f} "
                      f"{rec['speedup']:.2f}x rank {rec['rank']}/{rec['rank_ref']}", flush=True)
    print("CROSSOVER_DONE", flush=True)


if __name__ == "__main__":
    main()
