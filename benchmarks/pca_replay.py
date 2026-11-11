"""Replay the PCA benchmark panels used in the numerical review."""

import argparse
import gc
import json
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA

from mlxlearn._common.config import tuning_context
from mlxlearn.decomposition import PCA

# Load the committed pre-transform-fix source independently of the working tree.
BASELINE = "1859e09"
baseline_dir = tempfile.TemporaryDirectory(prefix="pca-replay-baseline-")
archive = subprocess.check_output(["git", "archive", BASELINE, "src/mlxlearn"])
subprocess.run(["tar", "-x", "-C", baseline_dir.name], input=archive, check=True)
source = Path(baseline_dir.name) / "src"
(source / "mlxlearn").rename(source / "mlxlearn_before")
sys.path.insert(0, str(source))
from mlxlearn_before.decomposition import PCA as OldPCA  # noqa: E402

warnings.simplefilter("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("panel", choices=("original", "gram", "stress"))
args = parser.parse_args()
original = [
    (256, 64, 32, np.float64, "auto", "fit"),
    (10_000, 64, 32, np.float32, "auto", "fit"),
    (20_000, 512, 32, np.float32, "auto", "fit_transform"),
    (40_000, 256, 32, np.float64, "auto", "fit"),
    (100_000, 256, 32, np.float64, "auto", "fit"),
    (1_000, 1_024, 32, np.float32, "auto", "fit"),
    (8_000, 1_024, 32, np.float32, "auto", "fit"),
    (20_000, 1_024, 32, np.float64, "auto", "fit"),
    (20_000, 1_024, 32, np.float32, "auto", "fit_transform"),
    (100_000, 1_024, 32, np.float64, "auto", "fit"),
    (1_000, 1_000, 32, np.float32, "full", "fit"),
    (128, 512, 32, np.float32, "covariance_eigh", "fit"),
    (512, 512, 32, np.float32, "covariance_eigh", "fit"),
    (2_000, 32_768, 32, np.float32, "randomized", "fit"),
]
stress = [
    (32, 512, 8, 10, "covariance_eigh", "iso", "fit"),
    (128, 512, 32, 10, "covariance_eigh", "iso", "fit"),
    (64, 2_048, 32, 10, "covariance_eigh", "iso", "fit"),
    (64, 2_048, 32, 10, "covariance_eigh", "lowrank", "fit_transform"),
    (8_000, 1_024, 512, 10, "randomized", "iso", "fit"),
    (8_000, 1_024, 900, 10, "randomized", "iso", "fit"),
    (8_000, 1_024, 900, 10, "randomized", "lowrank", "fit"),
    (8_000, 1_024, 32, 990, "randomized", "iso", "fit"),
    (8_000, 1_024, 32, 2_000, "randomized", "iso", "fit"),
    (2_000, 4_096, 1_500, 10, "randomized", "iso", "fit_transform"),
    (2_000, 4_096, 1_500, 10, "randomized", "lowrank", "fit_transform"),
]
gram = [
    (8_000, 2_048, 128, 10, "randomized", "iso", "fit"),
    (8_000, 2_048, 128, 10, "randomized", "lowrank", "fit"),
]


def measure(X, k, oversamples, solver, op, kind, repeats, warmups, with_cpu=False):
    kwargs = {
        "n_components": k,
        "n_oversamples": oversamples,
        "svd_solver": solver,
        "random_state": 0,
    }
    classes = {"sklearn": SkPCA, "before": OldPCA, "after": PCA}
    if with_cpu:
        classes["after_cpu"] = PCA
    routes = list(classes)

    def run(route):
        model = classes[route](**kwargs)
        if route == "after_cpu":
            with tuning_context(force_backend="cpu"):
                getattr(model, op)(X)
        else:
            getattr(model, op)(X)
        return model

    routing = {}
    for route in routes:
        for _ in range(warmups):
            model = run(route)
        routing[route] = (model._fit_svd_solver, getattr(model, "execution_backend_", "sklearn"))
    values = {route: [] for route in routes}
    for r in range(repeats):
        order = routes[r % len(routes) :] + routes[: r % len(routes)]
        for route in order:
            t = time.perf_counter()
            run(route)
            values[route].append(1e3 * (time.perf_counter() - t))
    med = {route: round(statistics.median(v), 3) for route, v in values.items()}
    spreads = {route: [round(min(v), 3), round(max(v), 3)] for route, v in values.items()}
    record = {
        "shape": X.shape,
        "k": k,
        "oversamples": oversamples,
        "dtype": X.dtype.name,
        "solver": solver,
        "kind": kind,
        "op": op,
        "routes": routing,
        "ms": med,
        "spread": spreads,
        "samples_ms": values,
    }
    print(json.dumps(record), flush=True)


if args.panel == "original":
    rng = np.random.default_rng(0)
    for n, d, k, dtype, solver, op in original:
        X = rng.standard_normal((n, d), dtype=dtype)
        measure(X, k, 10, solver, op, "iso", 9, 2)
        del X
        gc.collect()
else:
    rng = np.random.default_rng(34)
    for j, (n, d, k, over, solver, kind, op) in enumerate(
        stress if args.panel == "stress" else gram
    ):
        if kind == "iso":
            X = rng.standard_normal((n, d), dtype=np.float32)
        else:
            r = min(n, d) // 8
            X = rng.standard_normal((n, r), dtype=np.float32) @ rng.standard_normal(
                (r, d), dtype=np.float32
            )
            X += 0.001 * rng.standard_normal((n, d), dtype=np.float32) + 10
        if args.panel == "gram" or j >= len(stress) - 2:
            measure(
                X,
                k,
                over,
                solver,
                op,
                kind,
                5 if args.panel == "stress" else 9,
                1 if args.panel == "stress" else 2,
                with_cpu=args.panel == "stress",
            )
        del X
        gc.collect()
