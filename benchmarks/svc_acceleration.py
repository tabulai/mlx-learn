"""Compare complete fresh SVC fits with the frozen pre-native implementation.

Run alone (not alongside other benchmarks)::

    VECLIB_MAXIMUM_THREADS=1 python -m benchmarks.svc_acceleration --repeats 3 \
        --output benchmarks/results/svc-native-2026-10-31.json

The baseline archive contains the original estimator, kernel evaluator, and SMO
solver. It is loaded into separate modules without mutating production globals.
First calls are reported separately; subsequent fresh fits alternate order.
A separate same-Gram probe measures only optimizer work and independently checks
dual feasibility, KKT residuals and the objective on the exact supplied matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.svm import SVC as SkSVC
from threadpoolctl import threadpool_info, threadpool_limits

from mlxlearn._common.config import tuning_context
from mlxlearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks/reference/svc-pre-native-2026-10-31.zip"
CASES = (
    ("rbf_narrow", "rbf", 2000, 20, 2, "cpu", 200.0),
    ("rbf_wide", "rbf", 4000, 256, 2, "mlx", 200.0),
    ("linear", "linear", 1200, 20, 2, "cpu", 200.0),
    ("poly", "poly", 2000, 32, 2, "cpu", 200.0),
    ("sigmoid", "sigmoid", 2000, 32, 2, "cpu", 200.0),
    ("rbf_multiclass", "rbf", 1600, 32, 4, "mlx", 200.0),
    ("rbf_bounded_cache", "rbf", 1600, 32, 2, "mlx", 0.05),
)


def load_baseline(directory):
    manifest = json.loads(REFERENCE.with_suffix(".manifest.json").read_text())
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("The baseline archive hash does not match its manifest")
    names = {
        "src/mlxlearn/_kernels/kernels.py": "mlxlearn._kernels._svc_baseline_kernels",
        "src/mlxlearn/_solvers/smo.py": "mlxlearn._solvers._svc_baseline_smo",
        "src/mlxlearn/svm/_classes.py": "mlxlearn.svm._svc_baseline_classes",
    }
    modules = []
    with zipfile.ZipFile(REFERENCE) as archive:
        for path, name in names.items():
            source = archive.read(path)
            if hashlib.sha256(source).hexdigest() != manifest["files"][path]:
                raise ValueError(f"Baseline source hash mismatch: {path}")
            file = Path(directory) / Path(path).name
            file.write_bytes(source)
            spec = importlib.util.spec_from_file_location(name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
    kernels, smo, estimator = modules
    for name in (
        "KernelSpec", "resolve_gamma", "kernel_diagonal", "kernel_matrix",
        "build_kernel_rows", "kernel_combine",
    ):
        setattr(estimator, name, getattr(kernels, name))
    estimator.solve_c_svc = smo.solve_c_svc
    estimator.SMOConfig = smo.SMOConfig
    return estimator.SVC, smo


def legacy_estimator(directory):
    """Load the archived estimator while leaving production bindings untouched."""
    return load_baseline(directory)[0]


def timed(call):
    mx.synchronize()
    start = time.perf_counter()
    result = call()
    mx.synchronize()
    return time.perf_counter() - start, result


def summary(samples):
    return {"median_ms": statistics.median(samples) * 1000, "samples_s": samples}


def run_case(case, legacy, repeats, seed):
    name, kernel, n, d, classes, backend, cache = case
    X, y = make_classification(
        n_samples=n + 257, n_features=d, n_informative=min(12, d - 4),
        n_redundant=4, n_classes=classes, random_state=seed,
    )
    X = X.astype(np.float32)
    query, target = X[n:], y[n:]
    train, labels = X[:n], y[:n]
    variants = {"legacy": legacy, "native": SVC, "sklearn": SkSVC}
    fit_times = {v: [] for v in variants}
    predict_times = {v: [] for v in variants}
    first = {}
    models = {}
    fit_status_history = {v: [] for v in variants}

    def fit(cls):
        with tuning_context(force_backend=backend):
            return cls(kernel=kernel, cache_size=cache).fit(train, labels)

    for variant, cls in variants.items():
        elapsed, model = timed(lambda cls=cls: fit(cls))
        prediction_elapsed, _ = timed(lambda model=model: model.predict(query))
        first[variant] = {"fit_s": elapsed, "predict_s": prediction_elapsed}
        fit_status_history[variant].append(int(model.fit_status_))
    for repeat in range(repeats):
        order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
        for variant in order:
            elapsed, model = timed(lambda variant=variant: fit(variants[variant]))
            fit_times[variant].append(elapsed)
            elapsed, _ = timed(lambda model=model: model.predict(query))
            predict_times[variant].append(elapsed)
            models[variant] = model
            fit_status_history[variant].append(int(model.fit_status_))

    quality, passed = estimator_quality(kernel, models, query, target, fit_status_history)
    return {
        "name": name, "shape": [n, d], "classes": classes, "kernel": kernel,
        "backend": backend, "cache_size_mb": cache, "seed": seed,
        "first_call": first, "fit": {v: summary(s) for v, s in fit_times.items()},
        "predict": {v: summary(s) for v, s in predict_times.items()},
        "quality": quality, "quality_passed": passed,
        "fit_speedup_vs_legacy": statistics.median(fit_times["legacy"]) / statistics.median(fit_times["native"]),
        "predict_speedup_vs_legacy": statistics.median(predict_times["legacy"]) / statistics.median(predict_times["native"]),
    }


def estimator_quality(kernel, models, query, target, fit_status_history=None):
    """Compare final models, and reject any failed fit from the measured series."""
    decisions = {v: m.decision_function(query) for v, m in models.items()}
    predictions = {v: m.predict(query) for v, m in models.items()}
    quality = {}
    for variant, model in models.items():
        reported_gap = getattr(model, "_pair_kkt_violation_", None)
        quality[variant] = {
            "accuracy": float(np.mean(predictions[variant] == target)),
            "agreement_with_legacy": float(np.mean(predictions[variant] == predictions["legacy"])),
            "agreement_with_sklearn": float(np.mean(predictions[variant] == predictions["sklearn"])),
            "max_decision_difference_from_legacy": float(np.max(np.abs(decisions[variant] - decisions["legacy"]))),
            "max_decision_difference_from_sklearn": float(np.max(np.abs(decisions[variant] - decisions["sklearn"]))),
            "reported_pair_kkt_violation": None if reported_gap is None else reported_gap.tolist(),
            "n_iter": model.n_iter_.tolist(),
            "n_support": model.n_support_.tolist(),
            "fit_status": int(model.fit_status_),
            "fit_status_samples": (
                fit_status_history[variant] if fit_status_history is not None
                else [int(model.fit_status_)]
            ),
            "backend": getattr(model, "execution_backend_", "sklearn"),
        }
    native_quality = quality["native"]
    passed = all(q["fit_status"] == 0 and all(status == 0 for status in q["fit_status_samples"])
                 for q in quality.values())
    passed &= all(bool(np.isfinite(values).all()) for values in decisions.values())
    passed &= native_quality["agreement_with_legacy"] >= 0.99
    passed &= native_quality["agreement_with_sklearn"] >= (0.95 if kernel == "sigmoid" else 0.99)
    if kernel != "sigmoid":
        passed &= bool(np.allclose(decisions["native"], decisions["legacy"], atol=0.02, rtol=0.005))
    return quality, bool(passed)


def solver_problem(kernel, seed, *, n=1000, d=24):
    """One deterministic matrix shared by both optimizers; preparation is untimed."""
    if kernel not in ("linear", "rbf"):
        raise ValueError("The same-Gram probe supports linear and rbf kernels")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32).astype(np.float64)
    y = np.where(X[:, 0] + 0.4 * rng.normal(size=n) > 0, 1.0, -1.0)
    gram64 = X @ X.T
    if kernel == "rbf":
        squared = np.sum(X * X, axis=1)
        gram64 = np.exp(-0.03 * np.maximum(squared[:, None] + squared[None, :] - 2 * gram64, 0))
    return y, np.full(n, 0.5), np.diag(gram64).copy(), gram64.astype(np.float32)


def solver_diagnostics(state, y, bounds, gram):
    """Recompute the objective and constraints independently of solver state."""
    alpha = np.asarray(state.alpha, dtype=np.float64)
    signed_alpha = alpha * y
    product = gram.astype(np.float64) @ signed_alpha
    yg = product - y
    upper = alpha >= bounds
    lower = ~upper & (alpha <= 0)
    up = np.where(y > 0, ~upper, ~lower)
    low = np.where(y > 0, ~lower, ~upper)
    gap = float(np.max(-yg[up]) + np.max(yg[low])) if up.any() and low.any() else 0.0
    return {
        "n_iter": int(state.n_iter), "converged": bool(state.converged),
        "rho": float(state.rho), "reported_kkt_violation": float(state.max_kkt_violation),
        "independent_kkt_violation": gap,
        "equality_residual": float(abs(alpha @ y)),
        "lower_box_violation": float(max(0, np.max(-alpha))),
        "upper_box_violation": float(max(0, np.max(alpha - bounds))),
        "dual_objective": float(0.5 * signed_alpha @ product - np.sum(alpha)),
    }


def run_solver_probe(kernel, frozen_smo, repeats, seed, *, n=1000, d=24):
    """Compare archived and native SMO on identical kernel bytes, with no timing gate."""
    from mlxlearn._solvers.native_smo import solve_c_svc

    y, bounds, diagonal, gram = solver_problem(kernel, seed, n=n, d=d)
    config = frozen_smo.SMOConfig(tol=1e-3, max_iter=100000, cache_rows=n)
    variants = {
        "legacy": lambda: frozen_smo.solve_c_svc(y, bounds, diagonal, gram.__getitem__, config),
        "native": lambda: solve_c_svc(y, bounds, diagonal, None, config, gram=gram),
    }
    samples = {name: [] for name in variants}
    states = {}
    first = {}
    for name, call in variants.items():
        elapsed, state = timed(call)
        if state is None:
            raise RuntimeError("The native SMO extension is required for this benchmark")
        first[name] = elapsed
    for repeat in range(repeats):
        order = list(variants) if repeat % 2 == 0 else list(reversed(variants))
        for name in order:
            elapsed, states[name] = timed(variants[name])
            samples[name].append(elapsed)
    quality = {name: solver_diagnostics(state, y, bounds, gram) for name, state in states.items()}
    alpha_error = float(np.max(np.abs(states["native"].alpha - states["legacy"].alpha)))
    objective_error = abs(quality["native"]["dual_objective"] - quality["legacy"]["dual_objective"])
    passed = all(
        q["converged"] and q["independent_kkt_violation"] < config.tol + 1e-9
        and q["equality_residual"] <= 1e-9
        and q["lower_box_violation"] <= 1e-12 and q["upper_box_violation"] <= 1e-12
        and all(np.isfinite(value) for value in q.values())
        for q in quality.values()
    )
    passed &= alpha_error <= 1e-10 and objective_error <= 1e-10 * max(1, abs(quality["legacy"]["dual_objective"]))
    passed &= quality["native"]["n_iter"] == quality["legacy"]["n_iter"]
    return {
        "kernel": kernel, "shape": [n, d], "seed": seed,
        "tol": config.tol, "C": 0.5, "max_iter": config.max_iter,
        "kernel_row_dtype": "float32", "state_dtype": "float64",
        "gram_sha256": hashlib.sha256(gram.tobytes()).hexdigest(),
        "first_call_s": first, "solve": {name: summary(values) for name, values in samples.items()},
        "quality": quality, "max_alpha_difference": alpha_error,
        "alpha_exact_equal": bool(np.array_equal(states["native"].alpha, states["legacy"].alpha)),
        "objective_difference": objective_error, "quality_passed": bool(passed),
        "speedup_vs_legacy": statistics.median(samples["legacy"]) / statistics.median(samples["native"]),
        "timing_scope": "same precomputed Gram; full solver call including native buffer validation, excluding matrix preparation and diagnostics",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from mlxlearn._kernels import _svc_cpu_native
    from mlxlearn._solvers import _smo_native

    sources = [
        "src/mlxlearn/svm/_classes.py", "src/mlxlearn/_kernels/kernels.py",
        "src/mlxlearn/_kernels/svc_cpu.py", "src/mlxlearn/_kernels/svc_fused.py",
        "src/mlxlearn/_kernels/_svc_cpu_native.cpp", "src/mlxlearn/_solvers/native_smo.py",
        "src/mlxlearn/_solvers/_smo_native.cpp", "setup.py", "benchmarks/svc_acceleration.py",
    ]
    output = {
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "mlx": mx.__version__,
            "device": mx.device_info(), "threads": threadpool_info(),
            "threadpoolctl_limit": 1,
            "accelerate_thread_limit": os.environ.get("VECLIB_MAXIMUM_THREADS", "OS default"),
            "repeats": args.repeats,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            "extension_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (_svc_cpu_native, _smo_native)},
            "baseline_manifest": json.loads(REFERENCE.with_suffix(".manifest.json").read_text()),
            "timing_scope": "first call separately; alternating subsequent complete fresh fits and 257-row predict calls; synchronized MLX",
        }, "cases": [], "solver_probes": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="svc-baseline-") as directory, threadpool_limits(limits=1):
        legacy, frozen_smo = load_baseline(directory)
        for case in CASES[:2] if args.smoke else CASES:
            result = run_case(case, legacy, args.repeats, args.seed)
            output["cases"].append(result)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            print(f"{result['name']}: fit {result['fit_speedup_vs_legacy']:.2f}x, predict {result['predict_speedup_vs_legacy']:.2f}x; quality={result['quality_passed']}", flush=True)
        for kernel in ("rbf", "linear"):
            probe = run_solver_probe(kernel, frozen_smo, args.repeats, args.seed)
            output["solver_probes"].append(probe)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            print(f"{kernel} same-Gram solver: {probe['speedup_vs_legacy']:.2f}x; quality={probe['quality_passed']}", flush=True)
    if not all(c["quality_passed"] for c in output["cases"] + output["solver_probes"]):
        raise SystemExit("SVC benchmark quality gate failed")


if __name__ == "__main__":
    main()
