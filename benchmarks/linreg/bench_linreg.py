#!/usr/bin/env python
"""LinearRegression benchmark: mlxlearn (direct, GPU, patched) against scikit-learn.

Each config builds one dataset and times ``fit`` for every variant in rotated
order with a pause before each call (scikit-learn's BLAS threads and the GPU go
cold, as in interleaved real use). Correctness is checked against scikit-learn
(objective, coefficients, rank_, singular_). Results append to a JSONL file and
resume by key.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def make_data(cfg, seed=0):
    rng = np.random.default_rng(seed)
    n, d, k = cfg["n"], cfg["d"], cfg.get("k", 1)
    kind = cfg["data"]
    if kind == "gauss":
        X = rng.standard_normal((n, d))
    elif kind == "repeated_rows":
        rows = rng.standard_normal((max(1, n // 2), d))
        X = rows[rng.integers(len(rows), size=n)]
    elif kind == "scaled_columns":
        X = rng.standard_normal((n, d)) * np.geomspace(1.0, 1e-9, d)
    elif kind == "offset":
        X = rng.standard_normal((n, d)) + 1e3
    elif kind.startswith("cond"):
        # X = U diag(s) V^T with log-spaced singular values (condition 10**e)
        e = float(kind[4:])
        r = min(n, d)
        U, _ = np.linalg.qr(rng.standard_normal((n, r)))
        V, _ = np.linalg.qr(rng.standard_normal((d, r)))
        s = np.logspace(0, -e, r)
        X = (U * s) @ V.T * np.sqrt(n)
    elif kind == "rankdef":
        # rank d // 2: second half are linear combinations of the first half
        h = max(1, d // 2)
        A = rng.standard_normal((n, h))
        B = A @ rng.standard_normal((h, d - h))
        X = np.hstack([A, B])
    elif kind == "onehot":
        # one categorical feature with d levels, every level kept (collinear
        # with the intercept), plus nothing else
        codes = rng.integers(0, d, n)
        X = np.zeros((n, d))
        X[np.arange(n), codes] = 1.0
    elif kind == "mixed":
        # numeric columns plus a one-hot block (collinear with the intercept)
        h = max(1, d // 2)
        codes = rng.integers(0, d - h, n)
        oh = np.zeros((n, d - h))
        oh[np.arange(n), codes] = 1.0
        X = np.hstack([rng.standard_normal((n, h)) * rng.uniform(0.1, 100, h), oh])
    elif kind == "integer":
        X = rng.poisson(3.0, (n, d)).astype(np.float64)
    else:
        raise ValueError(kind)
    W = rng.standard_normal((d, k))
    y = X @ W + 0.1 * rng.standard_normal((n, k))
    if k == 1:
        y = y[:, 0]
    if cfg.get("dtype") == "float32":
        X = X.astype(np.float32)
        y = y.astype(np.float32)
    sw = None
    if cfg.get("sw"):
        sw = rng.uniform(0.0, 2.0, n)
        sw[rng.random(n) < 0.05] = 0.0
    return X, y, sw


def variants(names):
    import sklearn.linear_model as skl

    import mlxlearn.linear_model as mll
    import mlxlearn.patching._estimators as mlp
    from mlxlearn import config_context

    out = {}
    if "sk" in names:
        out["sk"] = lambda kw: (skl.LinearRegression(**kw), None)
    if "mlx" in names:
        out["mlx"] = lambda kw: (mll.LinearRegression(**kw), None)
    if "mlx_gpu" in names:
        out["mlx_gpu"] = lambda kw: (mll.LinearRegression(**kw), lambda: config_context(device="gpu"))
    if "mlx_patched" in names:
        out["mlx_patched"] = lambda kw: (mlp.LinearRegression(**kw), None)
    return out


def objective(est, X, y, sw):
    pred = est.predict(X)
    r = (np.asarray(pred, np.float64) - np.asarray(y, np.float64))
    sq = r * r if r.ndim == 1 else (r * r).sum(axis=1)
    if sw is None:
        return 0.5 * float(sq.mean())
    return 0.5 * float(np.dot(sw, sq)) / float(sw.sum())


def check(est, ref, X, y, sw):
    out = {}
    o, oref = objective(est, X, y, sw), objective(ref, X, y, sw)
    scale = max(oref, 1e-30 + 1e-12 * 0.5 * float(np.mean(np.asarray(y, np.float64) ** 2)))
    out["obj_rel"] = (o - oref) / scale
    c, cref = np.asarray(est.coef_, np.float64), np.asarray(ref.coef_, np.float64)
    out["coef_rel"] = float(np.linalg.norm(c - cref) / max(np.linalg.norm(cref), 1e-300))
    out["rank"] = int(getattr(est, "rank_", -1))
    out["rank_ref"] = int(getattr(ref, "rank_", -1))
    s, sref = getattr(est, "singular_", None), getattr(ref, "singular_", None)
    if s is not None and sref is not None and np.size(s) == np.size(sref) and np.size(sref):
        out["sing_rel"] = float(np.max(np.abs(np.asarray(s) - np.asarray(sref)))
                                / max(float(np.max(sref)), 1e-300))
    out["backend"] = getattr(est, "execution_backend_", None)
    return out


def run_config(cfg, args):
    X, y, sw = make_data(cfg, seed=cfg.get("seed", 0))
    kw = {}
    if not cfg.get("intercept", True):
        kw["fit_intercept"] = False
    specs = variants(args.variants)
    names = list(specs)
    rec = {"label": args.label, "cfg": cfg, "host": platform.node(), "fit": {}, "check": {},
           "errors": {}}
    ref = None
    fitted = {}
    rounds = args.rounds
    for r in range(rounds + 1):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:
            order = order[::-1]
        for name in order:
            if name in rec["errors"]:
                continue
            est, ctx = specs[name](kw)
            gc.collect()
            time.sleep(args.pause)
            try:
                if ctx is not None:
                    with ctx():
                        t0 = time.perf_counter()
                        est.fit(X, y, sample_weight=sw) if sw is not None else est.fit(X, y)
                        dt = time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    est.fit(X, y, sample_weight=sw) if sw is not None else est.fit(X, y)
                    dt = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                rec["errors"][name] = repr(exc)[:500]
                continue
            if r > 0:  # round 0 is an untimed warm-up
                rec["fit"].setdefault(name, []).append(dt)
            fitted[name] = est
            if name == "sk":
                ref = est
            if dt > args.slow and r > 0:
                rec["errors"].setdefault(name + ":slow", f"{dt:.1f}s")
    if ref is not None:
        for name, est in fitted.items():
            if name != "sk":
                try:
                    rec["check"][name] = check(est, ref, X, y, sw)
                except Exception as exc:  # noqa: BLE001
                    rec["check"][name] = {"error": repr(exc)[:300]}
    med = {k: float(np.median(v)) for k, v in rec["fit"].items() if v}
    rec["median"] = med
    if "sk" in med:
        rec["speedup"] = {k: med["sk"] / v for k, v in med.items() if k != "sk"}
    return rec


def key_of(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in sorted(cfg))


def grid_core():
    cfgs = []
    for n in (1_000, 10_000, 100_000, 1_000_000):
        for d in (4, 16, 64, 256, 1024, 4096):
            if n * d > 2e8 or d > 4 * n:
                continue
            cfgs.append({"data": "gauss", "n": n, "d": d})
    return cfgs


def grid_hard():
    cfgs = []
    for data in ("offset", "cond6", "cond10", "cond14", "rankdef", "onehot", "mixed", "integer"):
        for n, d in ((10_000, 64), (100_000, 64), (100_000, 256), (20_000, 1024), (1_000_000, 32)):
            if data == "onehot" and d > 256:
                continue
            cfgs.append({"data": data, "n": n, "d": d})
    return cfgs


def grid_options():
    cfgs = []
    for n, d in ((10_000, 64), (100_000, 256), (1_000_000, 32)):
        cfgs.append({"data": "gauss", "n": n, "d": d, "sw": True})
        cfgs.append({"data": "gauss", "n": n, "d": d, "intercept": False})
        cfgs.append({"data": "gauss", "n": n, "d": d, "k": 10})
        cfgs.append({"data": "gauss", "n": n, "d": d, "dtype": "float32"})
    for n, d in ((256, 2048), (1_000, 4096), (1_000, 1_000), (2_000, 8_192)):
        cfgs.append({"data": "gauss", "n": n, "d": d})
        cfgs.append({"data": "gauss", "n": n, "d": d, "intercept": False})
    for n, d in ((100, 4), (500, 16), (2_000, 64)):
        cfgs.append({"data": "gauss", "n": n, "d": d})
    return cfgs


def grid_specialized():
    return [
        {"data": "gauss", "n": 100, "d": 4},
        {"data": "gauss", "n": 256_000, "d": 1},
        {"data": "gauss", "n": 128_000, "d": 1, "sw": True, "k": 4},
        {"data": "repeated_rows", "n": 512, "d": 2048},
        {"data": "scaled_columns", "n": 8_000, "d": 64},
        {"data": "scaled_columns", "n": 16_000, "d": 512},
    ]


GRIDS = {"specialized": grid_specialized, "core": grid_core, "hard": grid_hard, "options": grid_options,
         "smoke": lambda: [{"data": "gauss", "n": 20_000, "d": 64}, {"data": "cond10", "n": 10_000, "d": 64}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--slow", type=float, default=120.0)
    ap.add_argument("--variants", nargs="*", default=["sk", "mlx", "mlx_gpu", "mlx_patched"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    import mlx.core as mx
    import scipy
    import sklearn

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["label"], key_of(r["cfg"])))
            except Exception:  # noqa: BLE001
                pass
    cfgs = GRIDS[args.grid]()
    if args.only:
        cfgs = [c for c in cfgs if args.only in key_of(c)]
    meta = {"python": sys.version.split()[0], "mlx": mx.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "device": str(mx.device_info().get("architecture")), "cpu_count": os.cpu_count()}
    print(json.dumps({"meta": meta, "configs": len(cfgs)}), flush=True)
    for i, cfg in enumerate(cfgs):
        if (args.label, key_of(cfg)) in done:
            continue
        t0 = time.time()
        try:
            rec = run_config(dict(cfg), args)
            rec["meta"] = meta
        except Exception as exc:  # noqa: BLE001
            rec = {"label": args.label, "cfg": cfg, "error": repr(exc), "trace": traceback.format_exc()[-1500:]}
        rec["wall_s"] = time.time() - t0
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        med = rec.get("median", {})
        sp = rec.get("speedup", {})
        chk = rec.get("check", {})
        bad = {k: (round(v.get("obj_rel", 0), 4), v.get("rank"), v.get("rank_ref"))
               for k, v in chk.items() if "error" in v or abs(v.get("obj_rel", 0)) > 1e-6
               or v.get("rank") != v.get("rank_ref")}
        print(f"[{i+1}/{len(cfgs)}] {key_of(cfg)} | " + " ".join(f"{k}={v*1e3:.1f}ms" for k, v in med.items())
              + " | " + " ".join(f"{k}:{v:.2f}x" for k, v in sp.items())
              + (f" | BAD {bad}" if bad else "") + (f" | ERR {rec.get('errors')}" if rec.get('errors') else "")
              + f" ({rec['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
