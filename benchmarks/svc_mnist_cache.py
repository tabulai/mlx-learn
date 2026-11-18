"""Time RBF SVC fit and held-out inference on MNIST in isolated worker processes.

Run python -m benchmarks.svc_mnist_cache --baseline /path/to/built/main --output results.json.
Requires the mnist.npz file used by the repository's MNIST-family benchmarks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path

from benchmarks.svc_row_cache import read, request
from benchmarks.svc_working_set import (
    ROOT,
    THREAD_VARIABLES,
    check_import_roots,
    controlled_environment,
    sha256,
    timed,
)

VARIANTS = ("sklearn", "main", "current")


def cpu_time(call):
    start = time.perf_counter()
    result = call()
    return time.perf_counter() - start, result


def worker(args):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    with np.load(args.data) as data:
        train, query = (data[key].astype(np.float64) / 255 for key in ("Xtr", "Xte"))
        labels, truth = data["ytr"], data["yte"]
    if (train.shape, query.shape, labels.shape, truth.shape) != (
        (60000, 784), (10000, 784), (60000,), (10000,)
    ):
        raise ValueError("Expected standard 60k/10k MNIST splits")

    clock, context, identity = cpu_time, contextlib.nullcontext(), {}
    if args.worker == "sklearn":
        from sklearn.svm import SVC

        identity["sklearn"] = importlib.metadata.version("scikit-learn")
        identity["estimator_module"] = SVC.__module__
    else:
        import mlx.core as mx

        from mlxlearn._common.config import tuning_context
        from mlxlearn._solvers import _smo_block_native
        from mlxlearn.svm import SVC

        def clock(call):
            return timed(call, mx)

        context = tuning_context(force_backend="mlx")
        check_import_roots(args.source)
        has_cache = bool(getattr(_smo_block_native, "SUPPORTS_ROW_CACHE", False))
        if has_cache != (args.worker == "current"):
            raise RuntimeError("Unexpected native row-cache capability; check baseline and rebuild")
        identity = {
            "commit": subprocess.check_output(
                ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "native_sha256": sha256(_smo_block_native.__file__),
            "supports_row_cache": has_cache,
            "device": mx.device_info(),
        }
    print(json.dumps(identity), flush=True)
    with context:
        model = None
        predictions = None
        for action in sys.stdin:
            if action.strip() == "predict" and model is not None:
                elapsed, predicted = clock(partial(model.predict, query))
                if not np.array_equal(predicted, predictions):
                    raise ValueError("Prediction changed in the interleaved inference timing")
                print(json.dumps(elapsed), flush=True)
                continue
            model = None
            if action.strip() != "fit":
                raise ValueError("Unknown worker request")
            repeats = 2 if args.worker == "sklearn" else 3
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fit, model = clock(
                    lambda: SVC(cache_size=args.cache, decision_function_shape="ovo").fit(train, labels)
                )
            first, predictions = clock(partial(model.predict, query))
            warm = []
            for _ in range(repeats):
                elapsed, repeated = clock(partial(model.predict, query))
                if not np.array_equal(repeated, predictions):
                    raise ValueError("Prediction changed between calls")
                warm.append(elapsed)
            decision, margin = clock(partial(model.decision_function, query))
            if margin.shape != (10000, 45) or predictions.shape != (10000,) or not np.isfinite(margin).all():
                raise ValueError("Invalid held-out margins or predictions")
            if model.fit_status_ or any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise ValueError("Solver did not converge")
            kkt = None
            if args.worker != "sklearn":
                check_import_roots(args.source)
                kkt = float(np.max(model._pair_kkt_violation_))
                if not kkt < model.tol or set(model._pair_solver_) != {"native_working_set"}:
                    raise ValueError("Incorrect MLX solver or convergence residual")
            np.savez(args.folder / f"{args.worker}.npz", margin=margin, prediction=predictions)
            record = {
                "fit_s": fit, "first_predict_s": first, "warm_predict_s": warm,
                "decision_s": decision, "accuracy": float(np.mean(predictions == truth)),
                "support_count": len(model.support_), "max_kkt": kkt, "fit_status": int(model.fit_status_),
                "n_iter": model.n_iter_.tolist(),
                "margin_sha256": hashlib.sha256(margin.tobytes()).hexdigest(),
                "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
            }
            print(json.dumps(record), flush=True)



def compare(folder, reference):
    import numpy as np

    with np.load(folder / "current.npz") as cur, np.load(folder / f"{reference}.npz") as ref:
        delta = np.abs(cur["margin"] - ref["margin"])
        return {
            "predictions_different": int(np.count_nonzero(cur["prediction"] != ref["prediction"])),
            "ovo_margin_max_abs_delta": float(delta.max()),
            "ovo_margin_median_abs_delta": float(np.median(delta)),
            "ovo_margins_close": bool(np.allclose(cur["margin"], ref["margin"], atol=0.02, rtol=0.005)),
        }


def campaign(args, cache, report):
    def save():
        pending = args.output.with_suffix(".partial")
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(args.output)

    case = {"identities": {}, "rounds": {v: [] for v in VARIANTS}, "order": []}
    report["cases"][str(cache)] = case
    with tempfile.TemporaryDirectory(prefix="svc-mnist-") as tmp:
        folder = Path(tmp)
        with contextlib.ExitStack() as stack:
            workers = {}
            for variant in VARIANTS:
                source = args.baseline if variant == "main" else ROOT
                cmd = [sys.executable, "-m", "benchmarks.svc_mnist_cache", "--worker", variant,
                       "--cache", str(cache), "--data", str(args.data), "--source", str(source),
                       "--folder", str(folder)]
                proc = stack.enter_context(subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                    cwd=folder, env=controlled_environment(source)
                ))
                stack.callback(proc.stdin.close)
                workers[variant] = proc
                case["identities"][variant] = read(proc)
            order = ["main", "current"] if cache == 200 else ["current", "main"]
            for i in range(args.repeats + 1):
                for v in (["sklearn"] + order if i == 0 else order):
                    r = request(workers[v], "fit")
                    r["warmup"] = i == 0 and v != "sklearn"
                    case["rounds"][v].append(r)
                    case["order"].append(v)
                    save()
                    print(cache, v, f"fit {r['fit_s']:.3f}s, first predict {r['first_predict_s']:.3f}s, "
                          f"warm predict {statistics.median(r['warm_predict_s']):.3f}s", flush=True)
                order.reverse()
            case["paired_warm_predictions"] = []
            for _ in range(11):
                case["paired_warm_predictions"].append({
                    "order": list(order), "seconds": {v: request(workers[v], "predict") for v in order}
                })
                order.reverse()
        case["comparison"] = {ref: compare(folder, ref) for ref in ("main", "sklearn")}
        for records in case["rounds"].values():
            for field in ("margin_sha256", "prediction_sha256"):
                if len({r[field] for r in records}) != 1:
                    raise ValueError("Output changed between fits")
        save()
        if not all(c["ovo_margins_close"] for c in case["comparison"].values()):
            raise ValueError("OVO margin comparison outside published tolerance")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--repeats", type=int, default=3, help="timed fresh fits per MLX variant/cache")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get(
        "MNIST_FAMILY_CACHE", Path.home() / ".cache/mlxlearn/mnist-family"
    )) / "mnist.npz")
    ap.add_argument("--worker", choices=VARIANTS, help=argparse.SUPPRESS)
    ap.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--cache", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--folder", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)
    if not args.baseline or not args.output or args.repeats < 1:
        ap.error("Supply --baseline, --output and positive --repeats")
    args.baseline = args.baseline.resolve()
    args.data = args.data.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "MNIST", "data_sha256": sha256(args.data), "driver_sha256": sha256(__file__),
        "platform": platform.platform(), "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "scikit-learn", "numpy")},
        "threads": {name: controlled_environment(ROOT)[name] for name in THREAD_VARIABLES},
        "cases": {},
    }
    for cache in (200, 2000):
        campaign(args, cache, report)


if __name__ == "__main__":
    main()
