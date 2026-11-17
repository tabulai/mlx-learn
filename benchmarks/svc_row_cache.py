"""Interleave end-to-end SVC fits from two separately built source checkouts.

Build both trees with the same interpreter via setup.py build_ext --inplace --force.
Run python -m benchmarks.svc_row_cache --baseline /path/to/main --output results.json.
The Fashion-MNIST cases use the MNIST_FAMILY_CACHE from the MNIST-family benchmarks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

from benchmarks.svc_working_set import (
    ROOT,
    check_import_roots,
    controlled_environment,
    sha256,
    source_hashes,
    timed,
)

# name, training rows, dimensions, seed, kernel backend, cache MiB
CASES = (
    ("tiny", 1600, 32, 41, "cpu", 0.05),
    ("cpu4", 4000, 32, 41, "cpu", 4),
    ("cpu24a", 4000, 32, 41, "cpu", 24),
    ("cpu24b", 4000, 32, 42, "cpu", 24),
    ("cpu64", 4000, 32, 41, "cpu", 64),
    ("gpu24", 4000, 256, 41, "mlx", 24),
    ("fashion200", 60000, 784, None, "mlx", 200),
    ("fashion2000", 60000, 784, None, "mlx", 2000),
)


def fixture(case, folder):
    import numpy as np
    from sklearn.datasets import make_classification

    _, n, d, seed, _, _ = case
    if seed is None:
        cache = Path(
            os.environ.get("MNIST_FAMILY_CACHE", Path.home() / ".cache/mlxlearn/mnist-family")
        )
        with np.load(cache / "fashion_mnist.npz") as data:
            train = data["Xtr"].astype(np.float64) / 255
            query = data["Xte"].astype(np.float64) / 255
            labels = data["ytr"]
        if train.shape != (n, d) or query.shape != (10000, d):
            raise ValueError("Expected the full Fashion-MNIST training and test images")
    else:
        x, y = make_classification(
            n_samples=n + 257, n_features=d, n_informative=20, random_state=seed
        )
        x = x.astype(np.float32)
        train, labels, query = x[:n], y[:n], x[n:]
    hashes = {}
    for name, array in (("train", train), ("labels", labels), ("query", query)):
        path = folder / f"{name}.npy"
        np.save(path, array)
        hashes[name] = sha256(path)
    return hashes


def counter(solve, counts):
    def instrument(y, C, diag, fetch_rows, config, **kwargs):
        def fetch(ids):
            counts["calls"] += 1
            counts["rows"] += len(ids)
            return fetch_rows(ids)

        return solve(y, C, diag, fetch, config, **kwargs)

    return instrument


def binary_kkt_gap(model, train, labels):
    """Rebuild the dual gradient from all training points using float64 RBF distances."""
    import numpy as np
    from scipy.spatial.distance import cdist

    sign = np.where(labels == model.classes_[1], 1.0, -1.0)
    alpha = np.zeros(len(train))
    alpha[model.support_] = np.abs(model.dual_coef_[0])
    # sign - K @ signed_alpha is the negative signed dual gradient. A
    # feasible pair with maximum positive gap is SMO's stopping diagnostic.
    score = sign.copy()
    for start in range(0, len(train), 128):
        block = slice(start, start + 128)
        distances = cdist(train[block], model.support_vectors_, metric="sqeuclidean")
        score[block] -= np.exp(-model._gamma * distances) @ model.dual_coef_[0]
    up = ((sign > 0) & (alpha < model.C)) | ((sign < 0) & (alpha > 0))
    low = ((sign < 0) & (alpha < model.C)) | ((sign > 0) & (alpha > 0))
    return float(score[up].max() - score[low].min())


def worker(args):
    import mlx.core as mx
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    from mlxlearn._common.config import tuning_context
    from mlxlearn._solvers import _smo_block_native, block_smo
    from mlxlearn.svm import SVC

    case = next(c for c in CASES if c[0] == args.worker)
    name, _, _, _, backend, cache = case
    folder = args.folder
    train, labels, query = (
        np.load(folder / f"{key}.npy", mmap_mode="r") for key in ("train", "labels", "query")
    )
    check_import_roots(args.source)
    supports_row_cache = bool(getattr(_smo_block_native, "SUPPORTS_ROW_CACHE", False))
    if args.label == "current" and not supports_row_cache:
        raise RuntimeError(
            "Rebuild the candidate native extension with support for spare cached rows"
        )
    identity = {
        "supports_row_cache": supports_row_cache,
        "extension_sha256": sha256(_smo_block_native.__file__),
        "source_sha256": hashlib.sha256(
            json.dumps(source_hashes(args.source), sort_keys=True).encode()
        ).hexdigest(),
        "version": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    print(json.dumps(identity), flush=True)

    with tuning_context(force_backend=backend):
        for line in sys.stdin:
            probe = line.strip() == "trace"
            counts = {"rows": 0, "calls": 0}

            monitor = (
                patch.object(block_smo, "solve_c_svc", counter(block_smo.solve_c_svc, counts))
                if probe
                else contextlib.nullcontext()
            )
            with warnings.catch_warnings(record=True) as caught, monitor:
                warnings.simplefilter("always", ConvergenceWarning)
                seconds, model = timed(
                    lambda: SVC(cache_size=cache, decision_function_shape="ovo").fit(train, labels),
                    mx,
                )
            if probe:
                counts["independent_float64_kkt"] = binary_kkt_gap(model, train, labels)
                print(json.dumps(counts), flush=True)
                continue
            pred, margin = model.predict(query), model.decision_function(query)
            check_import_roots(args.source)
            kkt = np.asarray(model._pair_kkt_violation_)
            solvers = set(model._pair_solver_)
            result = {
                "seconds": seconds,
                "max_kkt": float(kkt.max()),
                "fit_status": int(model.fit_status_),
                "tol": model.tol,
                "n_iter": model.n_iter_.tolist(),
                "solver": sorted(solvers),
                "support_sha256": hashlib.sha256(model.support_.tobytes()).hexdigest(),
                "backend": model.execution_backend_,
                "convergence_warnings": [
                    str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
                ],
                "finite": bool(np.isfinite(kkt).all() and np.isfinite(margin).all()),
            }
            if solvers != {"native_working_set"} or model.execution_backend_ != backend:
                raise RuntimeError(f"Unexpected solver or backend for {name}: {result}")
            np.savez(folder / f"{args.label}.npz", prediction=pred, margin=margin)
            print(json.dumps(result), flush=True)
            del model


def read(proc):
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"Benchmark worker stopped with code {proc.poll()}")
    return json.loads(line)


def start(source, name, label, folder):
    command = [
        sys.executable,
        "-m",
        "benchmarks.svc_row_cache",
        "--worker",
        name,
        "--source",
        str(source),
        "--folder",
        str(folder),
        "--label",
        label,
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=folder,
        env=controlled_environment(source),
    )


def request(proc, action):
    proc.stdin.write(action + "\n")
    proc.stdin.flush()
    return read(proc)


def measure(case, sources, repeats, folder):
    import numpy as np

    name, n, d, seed, backend, cache = case
    hashes = fixture(case, folder)
    order = list(sources)
    random.Random(41 + CASES.index(case)).shuffle(order)
    procs = {}
    rounds = []
    result = {
        "name": name,
        "n": n,
        "d": d,
        "seed": seed,
        "backend": backend,
        "cache_mib": cache,
        "fixture_sha256": hashes,
        "rounds": rounds,
        "decision_atol": 0.02,
        "decision_rtol": 0.005,
    }
    try:
        for label in order:
            procs[label] = start(sources[label], name, label, folder)
        result["versions"] = {label: read(proc) for label, proc in procs.items()}
        for i in range(repeats + 1):
            order.reverse()
            records = {label: request(procs[label], "fit") for label in order}
            main, current = records["main"], records["current"]
            with np.load(folder / "main.npz") as a, np.load(folder / "current.npz") as b:
                q, pairs = (257, 1) if seed is not None else (10000, 45)
                margin_shape = (q,) if pairs == 1 else (q, pairs)
                for output in (a, b):
                    if output["prediction"].shape != (q,) or output["margin"].shape != margin_shape:
                        raise ValueError(f"Unexpected held-out output shape for {name}")
                main_margin, candidate_margin = a["margin"], b["margin"]
                delta = np.abs(main_margin - candidate_margin)
                margins_close = np.allclose(
                    candidate_margin,
                    main_margin,
                    atol=result["decision_atol"],
                    rtol=result["decision_rtol"],
                )
                prediction_equal = np.array_equal(a["prediction"], b["prediction"])
            passed = all(
                r["fit_status"] == 0
                and r["finite"]
                and r["max_kkt"] < r["tol"]
                and not r["convergence_warnings"]
                for r in records.values()
            )
            same_solver = (
                main["n_iter"] == current["n_iter"]
                and main["support_sha256"] == current["support_sha256"]
            )
            rounds.append(
                {
                    "warmup": i == 0,
                    "order": order.copy(),
                    "records": records,
                    "margins_compared": delta.size,
                    "max_abs_margin_delta": float(delta.max()),
                    "prediction_equal": prediction_equal,
                    "strict_replay_passed": bool(
                        passed and same_solver and prediction_equal and delta.max() <= 1e-6
                    ),
                    "iterations_equal": main["n_iter"] == current["n_iter"],
                    "support_equal": main["support_sha256"] == current["support_sha256"],
                    "margins_equal": bool(delta.max() == 0),
                    "passed": bool(passed and prediction_equal and margins_close),
                }
            )
        if seed is not None:
            result["untimed_row_fetches"] = {
                label: request(procs[label], "trace") for label in order
            }
        medians = {
            label: statistics.median(r["records"][label]["seconds"] for r in rounds[1:])
            for label in order
        }
        result["median_s"] = medians
        result["passed"] = all(r["passed"] for r in rounds) and all(
            v["independent_float64_kkt"] < 0.001
            for v in result.get("untimed_row_fetches", {}).values()
        )
        result["speedup"] = medians["main"] / medians["current"] if result["passed"] else None
    finally:
        for proc in procs.values():
            proc.stdin.close()
            proc.wait(timeout=30)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--image-repeats", type=int, default=3)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--folder", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if not args.baseline or not args.output or min(args.repeats, args.image_repeats) <= 0:
        parser.error("Provide --baseline, --output, and positive repeat counts.")
    import importlib.metadata

    import mlx.core as mx

    sources = {"main": args.baseline.resolve(), "current": ROOT}
    if sources["main"] == ROOT:
        parser.error("Use a separately built baseline checkout.")
    results = []
    report = {
        "schema": 2,
        "python": sys.version,
        "platform": platform.platform(),
        "device": mx.device_info(),
        "versions": {
            k: importlib.metadata.version(k) for k in ("scikit-learn", "numpy", "mlx", "scipy")
        },
        "harness_sha256": sha256(__file__),
        "cases": results,
        "threads": {
            k: v
            for k, v in controlled_environment(ROOT).items()
            if k
            in (
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        if args.cases and case[0] not in args.cases:
            continue
        with tempfile.TemporaryDirectory(prefix="svc-row-cache-") as temp:
            result = measure(
                case, sources, args.image_repeats if case[3] is None else args.repeats, Path(temp)
            )
        results.append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(case[0], result["median_s"], result["speedup"], result["passed"], flush=True)
    if not all(r["passed"] for r in results):
        raise SystemExit("One or more benchmark audits failed; see output for affected cases.")


if __name__ == "__main__":
    main()
