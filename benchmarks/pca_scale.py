"""Benchmark PCA fit and transform on synthetic matrices up to one million rows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA as SkPCA

from mlxlearn._solvers.pca import pca_transform_uses_gpu
from mlxlearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "1859e09"


def run_case(X, query, k, repeats, warmups, before):
    classes = {"sklearn": SkPCA, "before": before, "mlxlearn": PCA}
    names = tuple(classes)
    timings = {name: {"fit": [], "transform_train": [], "transform_test": []} for name in names}
    models = {}
    for iteration in range(warmups + repeats):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            model = classes[name](n_components=k, random_state=0)
            for op, call in (
                ("fit", lambda model=model: model.fit(X)),
                ("transform_train", lambda model=model: model.transform(X)),
                ("transform_test", lambda model=model: model.transform(query)),
            ):
                start = time.perf_counter()
                call()
                if iteration >= warmups:
                    timings[name][op].append((time.perf_counter() - start) * 1000)
            models[name] = model

    results = {}
    for name, model in models.items():
        actual = model.transform(query[:1024]).astype(np.float64)
        centered = query[:1024].astype(np.float64) - model.mean_
        expected = centered @ model.components_.T.astype(np.float64)
        error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        if not np.isfinite(error):
            raise ValueError(f"{name} produced a nonfinite projection")
        backend = model.execution_backend_ if name != "sklearn" else "cpu"
        if name == "mlxlearn":
            transform_backend = (
                "mlx"
                if backend == "mlx"
                and pca_transform_uses_gpu(X, model.mean_, model.explained_variance_)
                else "cpu"
            )
        else:
            transform_backend = "see estimator dispatch" if name == "before" else "cpu"
        results[name] = {
            "fit_backend": backend,
            "transform_train_backend": transform_backend,
            "solver": model._fit_svd_solver,
            "samples_ms": timings[name],
            "median_ms": {op: statistics.median(values) for op, values in timings[name].items()},
            "explained_variance_ratio": float(
                np.sum(model.explained_variance_ratio_, dtype=np.float64)
            ),
            "projection_relative_error": error,
        }
    return results


def main(before):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    parser.add_argument("--features", type=int, nargs="+", default=[256, 1024])
    args = parser.parse_args()
    diff = subprocess.check_output(
        ["git", "diff", "--", "src/mlxlearn/_solvers/pca.py", "src/mlxlearn/decomposition/_pca.py"],
        cwd=ROOT,
    )
    report = {
        "baseline_commit": BASELINE,
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "machine": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "mlx": mx.__version__,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "data": "Independent standard normal training and 10,000-row query arrays; seed 42.",
        "components": 32,
        "cases": [],
    }
    for rows in args.rows:
        for features in args.features:
            for dtype in (np.float32, np.float64):
                rng = np.random.default_rng(42)
                X = rng.standard_normal((rows, features), dtype=dtype)
                query = rng.standard_normal((10_000, features), dtype=dtype)
                result = run_case(X, query, 32, args.repeats, args.warmups, before)
                case = {
                    "rows": rows,
                    "features": features,
                    "dtype": np.dtype(dtype).name,
                    "results": result,
                }
                report["cases"].append(case)
                args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                speed = {
                    op: round(result["sklearn"]["median_ms"][op] / value, 2)
                    for op, value in result["mlxlearn"]["median_ms"].items()
                }
                print(rows, features, np.dtype(dtype).name, speed, flush=True)
                del X, query, result
                mx.clear_cache()
                gc.collect()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pca-scale-baseline-") as temporary:
        archive = subprocess.check_output(["git", "archive", BASELINE, "src/mlxlearn"], cwd=ROOT)
        subprocess.run(["tar", "-x", "-C", temporary], input=archive, check=True)
        source = Path(temporary) / "src"
        (source / "mlxlearn").rename(source / "mlxlearn_before")
        sys.path.insert(0, str(source))
        from mlxlearn_before.decomposition import PCA as BeforePCA

        main(BeforePCA)
