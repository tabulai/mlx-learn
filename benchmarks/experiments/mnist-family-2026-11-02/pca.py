"""Benchmark PCA on MNIST-family training and held-out images."""

import argparse
import gc
import gzip
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np
import scipy
import sklearn
from data import SPLITS, load
from sklearn.decomposition import PCA as SkPCA

from mlxlearn._solvers.pca import _projection_needs_centering, pca_transform_uses_gpu
from mlxlearn.decomposition import PCA as MlxPCA

REPO = Path(__file__).resolve().parents[3]

BASELINE = "1859e09"


def read_dataset(name, root):
    X_train, _, X_held_out, _ = load(name)
    splits = {"train": X_train}
    train_file, held_out_file = SPLITS[name]
    files = {"train": train_file}
    if name == "Kannada_MNIST":
        test_file = "test.csv.gz"
        with gzip.open(root / name / "input" / test_file, "rt") as source:
            assert source.readline().split(",", 1)[0] == "id"
            table = np.loadtxt(source, delimiter=",", dtype=np.uint16)
        assert np.unique(table[:, 0]).size == table.shape[0]
        splits.update(test=table[:, 1:], dig_holdout=X_held_out)
        files.update(test=test_file, dig_holdout=held_out_file)
        sizes = {"train": 60_000, "test": 5_000, "dig_holdout": 10_240}
    else:
        splits["test"] = X_held_out
        files["test"] = held_out_file
        sizes = {"train": 60_000, "test": 10_000}
    provenance = {}
    for label, matrix in splits.items():
        assert matrix.shape == (sizes[label], 784)
        assert matrix.min() >= 0 and matrix.max() <= 255
        path = root / name / "input" / files[label]
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
        provenance[label] = {
            "file": str(path.relative_to(root)),
            "sha256": digest.hexdigest(),
            "shape": list(matrix.shape),
        }
    return splits, provenance


def timed_call(operation, samples, key, timed):
    start = time.perf_counter_ns()
    result = operation()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if timed:
        samples[key].append(elapsed_ms)
    return result


def measure(inputs, solver, k, warmups, repeats):
    samples = {
        name: {"fit_train": [], **{f"transform_{split}": [] for split in inputs}}
        for name in CLASSES
    }
    models = {}
    names = tuple(CLASSES)
    split_names = tuple(inputs)
    for round_no in range(warmups + repeats):
        first = round_no % len(names)
        split_first = round_no % len(split_names)
        timed = round_no >= warmups
        for name in names[first:] + names[:first]:
            cls = CLASSES[name]
            model = cls(n_components=k, svd_solver=solver, random_state=0)
            timed_call(
                lambda model=model: model.fit(inputs["train"]), samples[name], "fit_train", timed
            )
            for split in split_names[split_first:] + split_names[:split_first]:
                timed_call(
                    lambda model=model, split=split: model.transform(inputs[split]),
                    samples[name],
                    f"transform_{split}",
                    timed,
                )
            models[name] = model
    return models, samples


def sum_squares(A):
    return float(np.einsum("ij,ij->", A, A))


def check_quality(model, inputs, reference, basis):
    mean = model.mean_.astype(np.float64)
    components = model.components_.astype(np.float64)
    k = components.shape[0]
    component_gram = components @ components.T
    overlap = basis[:, :k].T @ np.linalg.qr(components.T, mode="reduced")[0]
    cosines = np.clip(np.linalg.svd(overlap, compute_uv=False), 0, 1)
    variance = float(model.explained_variance_ratio_.sum(dtype=np.float64))
    reference_variance = float(reference.explained_variance_ratio_[:k].sum())
    quality = {
        "training_explained_variance_ratio": variance,
        "reference_explained_variance_ratio": reference_variance,
        "explained_variance_gap_pp": (variance - reference_variance) * 100,
        "orthogonality_max_abs": float(np.max(np.abs(component_gram - np.eye(k)))),
        "max_principal_angle_deg": float(np.degrees(np.arccos(cosines[-1]))),
        "relative_projector_distance": float(np.sqrt(np.mean(1 - cosines**2))),
        "splits": {},
    }
    for split, matrix in inputs.items():
        centered = np.subtract(matrix, mean, dtype=np.float64)
        exact = centered @ components.T
        Z = model.transform(matrix).astype(np.float64)
        assert Z.shape == (matrix.shape[0], k) and np.isfinite(Z).all()
        projection_error = np.sqrt(sum_squares(Z - exact) / sum_squares(exact))
        # Evaluate the actual transform and the learned basis with float64 arithmetic.
        residual_ss = sum_squares(centered) - 2 * float(np.einsum("ij,ij->", Z, exact))
        residual_ss += float(np.einsum("ij,ij->", Z @ component_gram, Z))
        quality["splits"][split] = {
            "projection_relative_frobenius_error": float(projection_error),
            "reconstruction_rmse": float(np.sqrt(residual_ss / matrix.size)),
        }
    return quality


def backend_info(name, model, inputs):
    fit = model.execution_backend_ if name != "sklearn" else "cpu"
    routes = {"fit_train": fit}
    for split, matrix in inputs.items():
        if name == "mlxlearn":
            use_gpu = fit == "mlx" and pca_transform_uses_gpu(
                matrix, model.mean_, model.explained_variance_
            )
            routes[f"transform_{split}"] = "mlx" if use_gpu else "cpu"
            continue
        host = (
            fit != "mlx"
            or matrix.size < 65_536
            or (
                np.result_type(matrix.dtype, model.mean_.dtype) == np.float64
                and _projection_needs_centering(
                    model.mean_, model.explained_variance_, np.dtype(np.float32)
                )
            )
        )
        routes[f"transform_{split}"] = "cpu" if host else "mlx"
    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["MNIST_FAMILY_ROOT"])
    evidence = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--",
                    "src/mlxlearn/_solvers/pca.py",
                    "src/mlxlearn/decomposition/_pca.py",
                ],
                cwd=REPO,
            )
        ).hexdigest(),
        "baseline_commit": BASELINE,
        "system": platform.platform(),
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "mlx": mx.__version__,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "random_state": 0,
        "preprocessing": "Contiguous arrays, pixel values divided by 255 in input dtype.",
        "reference": "scikit-learn covariance_eigh on float64 training pixels.",
        "datasets": {},
        "cases": [],
    }
    for name in SPLITS:
        raw, provenance = read_dataset(name, root)
        evidence["datasets"][name] = provenance
        X64 = {
            split: np.ascontiguousarray(matrix, dtype=np.float64) / 255
            for split, matrix in raw.items()
        }
        reference = SkPCA(n_components=128, svd_solver="covariance_eigh").fit(X64["train"])
        reference_basis = np.linalg.qr(reference.components_.T, mode="reduced")[0]
        for dtype in (np.float32, np.float64):
            inputs = (
                X64
                if dtype == np.float64
                else {
                    split: np.ascontiguousarray(matrix, dtype=dtype) / dtype(255)
                    for split, matrix in raw.items()
                }
            )
            for solver, k in product(("auto", "randomized"), (32, 128)):
                models, samples = measure(inputs, solver, k, args.warmups, args.repeats)
                case = {
                    "dataset": name,
                    "dtype": np.dtype(dtype).name,
                    "solver": solver,
                    "components": k,
                }
                for route, model in models.items():
                    case[route] = {
                        "solver": model._fit_svd_solver,
                        "backend": backend_info(route, model, inputs),
                        "ms": samples[route],
                        "median_ms": {
                            key: statistics.median(values) for key, values in samples[route].items()
                        },
                        "quality": check_quality(model, inputs, reference, reference_basis),
                    }
                evidence["cases"].append(case)
                args.out.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
                speedups = {
                    op: round(case["sklearn"]["median_ms"][op] / value, 2)
                    for op, value in case["mlxlearn"]["median_ms"].items()
                }
                print(name, np.dtype(dtype).name, solver, k, speedups, flush=True)
            del inputs
        del raw, X64
        gc.collect()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pca-transform-baseline-") as temporary:
        archive = subprocess.check_output(["git", "archive", BASELINE, "src/mlxlearn"], cwd=REPO)
        subprocess.run(["tar", "-x", "-C", temporary], input=archive, check=True)
        source = Path(temporary) / "src"
        (source / "mlxlearn").rename(source / "mlxlearn_before")
        sys.path.insert(0, str(source))
        from mlxlearn_before.decomposition import PCA as BeforePCA

        CLASSES = {"sklearn": SkPCA, "before": BeforePCA, "mlxlearn": MlxPCA}
        main()
