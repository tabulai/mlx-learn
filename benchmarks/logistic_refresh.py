"""Current logistic fit and inference versus stock sklearn, without old solvers.

Run alone: solver instrumentation observes calls in this process and is restored
after each fit. Host input conversion, validation and optimization are inside
the fit timer. Inference reuses a fitted estimator; independent scoring is never
timed. The first warmup call is recorded separately, not called process-cold.
Resume skips completed rows, including recorded failures; retry a failed case
with a new output path so the original evidence remains available.

    python -m benchmarks.logistic_refresh --output bench/logistic-refresh.json
    python -m benchmarks.logistic_refresh --output bench/logistic-refresh.json --resume
    python -m benchmarks.logistic_refresh --case wide_20k_4096 --tol 1e-5 \
        --output bench/logistic-refresh-strict.json

Default thread settings are preserved. --threads 1 requests a threadpoolctl
limit, which may not control Apple Accelerate. Set VECLIB_MAXIMUM_THREADS before
starting Python if wanted; this module records, and never sets, that variable.
"""

from __future__ import annotations

# ruff: noqa: E402 -- thread environment is captured before numerical imports.
import os

# Capture these before importing NumPy, sklearn, MLX, or their benchmark helpers.
THREAD_ENV_AT_IMPORT = {
    name: os.environ.get(name)
    for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
}

import argparse
import gc
import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

from mlxlearn import config_context
from mlxlearn._common.config import tuning_context
from mlxlearn._solvers import logistic, native_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module
from mlxlearn.patching._estimators import LogisticRegression as PatchedLogisticRegression

from .logistic_gpu import Case, make_data, record_environment, summarize
from .logistic_native import PROFILES as NATIVE_PROFILES
from .logistic_wide import PROFILES as WIDE_PROFILES
from .logistic_wide import loaded_native_extension, observed_implementation
from .run import PROFILES as GENERIC_PROFILES

VARIANTS = ("direct_auto", "patched_auto", "forced_gpu", "sklearn")
OPERATIONS = ("predict", "predict_proba")
SCHEMA_VERSION = 2
QUALITY_CERTIFICATE = "float64-original-data-v2"
# The repository's public probability parity tolerance (docs/fp32_policy.md).
PROBA_ATOL, PROBA_RTOL = 2e-3, 1e-7


def case_signature(case):
    return tuple(value for name, value in asdict(case).items() if name != "name")


def full_cases():
    """Deduplicate fixture semantics, preserving names from established suites."""
    candidates = [*NATIVE_PROFILES["full"], *WIDE_PROFILES["full"]]
    candidates += [Case(f"grid_{c.n_samples}_{c.n_features}", c.n_samples, c.n_features)
                   for c in GENERIC_PROFILES["full"] if c.algorithm == "logreg"]
    candidates.append(Case("multiclass_weighted_wide", 20_000, 1_024,
                           n_classes=5, weighted=True))
    cases, seen, aliases = [], {}, {}
    for case in candidates:
        signature = case_signature(case)
        if signature not in seen:
            seen[signature] = case.name
            cases.append(case)
        aliases[case.name] = seen[signature]
    return tuple(cases), aliases


CASES, CASE_ALIASES = full_cases()


def selected_cases(names):
    if not names:
        return CASES
    unknown = set(names) - CASE_ALIASES.keys()
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    selected = {CASE_ALIASES[name] for name in names}
    return tuple(case for case in CASES if case.name in selected)


def rotated(variants, repetition):
    offset = repetition % len(variants)
    return variants[offset:] + variants[:offset]


def source_hashes():
    """Hash current source, native binaries, build inputs and reused harnesses."""
    root = Path(__file__).resolve().parents[1]
    if Path(logistic.__file__).resolve() != root / "src/mlxlearn/_solvers/logistic.py":
        raise RuntimeError("install this working tree before benchmarking its current source")
    suffixes = {".py", ".pyi", ".c", ".cc", ".cpp", ".h", ".hpp", ".mm", ".m",
                ".metal", ".so", ".dylib"}
    paths = {root / name for name in ("setup.py", "pyproject.toml", "MANIFEST.in")}
    for package in ("mlxlearn", "mlxlearn_device"):
        paths.update(path for path in (root / "src" / package).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and (path.suffix in suffixes or path.name == "py.typed"))
    paths.update(root / "benchmarks" / name for name in (
        "logistic_refresh.py", "logistic_gpu.py", "logistic_native.py", "logistic_wide.py",
        "run.py", "_harness.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def timed_call(function):
    """Retain a failing call and its elapsed time instead of discarding samples."""
    value = None
    error = None
    elapsed = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        was_enabled = gc.isenabled()
        start = None
        try:
            mx.synchronize()
            gc.disable()
            start = time.perf_counter()
            value = function()
            mx.synchronize()
            elapsed = time.perf_counter() - start
        except Exception as exc:
            elapsed = None if start is None else time.perf_counter() - start
            error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            if was_enabled:
                gc.enable()
    return elapsed, value, {
        "warnings": [{"category": item.category.__name__, "message": str(item.message)}
                     for item in caught],
        "error": error,
    }


def fit_once(variant, X, y, weights, *, tol, max_iter):
    cls = {"sklearn": SkLogisticRegression, "patched_auto": PatchedLogisticRegression,
           "direct_auto": LogisticRegression, "forced_gpu": LogisticRegression}[variant]
    with (config_context(device="auto"),
          tuning_context(force_backend="mlx" if variant == "forced_gpu" else ""),
          observed_implementation(estimator_module, logistic, native_logistic) as record):
        elapsed, model, detail = timed_call(
            lambda: cls(C=1.0, tol=tol, max_iter=max_iter).fit(X, y, sample_weight=weights))
    detail.update(record)
    detail["backend"] = getattr(model, "_execution_backend_", "sklearn" if model else None)
    detail["n_iter"] = None if model is None else model.n_iter_.tolist()
    detail["implementation"] = (record["solver_calls"][-1]["implementation"]
                                if record["solver_calls"] else detail["backend"])
    detail["seconds"] = elapsed
    return elapsed, model, detail


def quality(model, reference, X, y, weights):
    """Certify this C=1 fit in float64; check public predictions on original X.

    Promoting residuals after a float32 matrix product is insufficient: the
    rounded logits have already changed the gradient. Cast every numerical
    input before the first calculation, including stock float32 coefficients.
    Keep the original input for prediction agreement so the public boundary is
    assessed at the dtype actually benchmarked.
    """
    X64 = np.asarray(X, dtype=np.float64)
    W64 = np.asarray(model.coef_, dtype=np.float64)
    b64 = np.asarray(model.intercept_, dtype=np.float64)
    reference_W64 = np.asarray(reference.coef_, dtype=np.float64)
    reference_b64 = np.asarray(reference.intercept_, dtype=np.float64)
    weights64 = None if weights is None else np.asarray(weights, dtype=np.float64)
    weight_sum = np.float64(len(X64)) if weights64 is None else weights64.sum(dtype=np.float64)
    codes = np.searchsorted(model.classes_, y)
    n_classes = len(model.classes_)
    logits64 = X64 @ W64.T + b64
    if n_classes == 2:
        residual64 = (expit(logits64[:, 0]) - codes.astype(np.float64))[:, None]
    else:
        residual64 = softmax(logits64, axis=1)
        residual64[np.arange(len(X64)), codes] -= np.float64(1.0)
    if weights64 is not None:
        residual64 *= weights64[:, None]
    gradient_W64 = residual64.T @ X64 / weight_sum + W64 / weight_sum
    gradient_b64 = residual64.sum(axis=0, dtype=np.float64) / weight_sum
    objective = logistic.reference_objective(
        W64, b64, X64, codes, n_classes=n_classes, C=1.0, sample_weight=weights64)
    reference_objective = logistic.reference_objective(
        reference_W64, reference_b64, X64, codes,
        n_classes=n_classes, C=1.0, sample_weight=weights64)
    return {
        "objective_float64": objective,
        "objective_relative_gap_to_sklearn": (objective - reference_objective)
        / max(abs(reference_objective), np.finfo(np.float64).tiny),
        "gradient_max_abs_float64": float(max(np.abs(gradient_W64).max(),
                                               np.abs(gradient_b64).max())),
        "prediction_agreement_sklearn": float(np.mean(model.predict(X) == reference.predict(X))),
        "max_coefficient_difference_sklearn": float(np.abs(W64 - reference_W64).max()),
    }


def fit_checks(variant, detail, score, tol):
    """Own accuracy checks remain distinct from the stock reference's quality."""
    finite = bool(score) and all(math.isfinite(score.get(key, math.nan)) for key in (
        "objective_float64", "gradient_max_abs_float64", "objective_relative_gap_to_sklearn",
        "prediction_agreement_sklearn"))
    checks = {
        "completed": detail.get("error") is None and score is not None,
        "finite": finite,
        "gradient": finite and score["gradient_max_abs_float64"] <= tol,
        "objective": finite and score["objective_relative_gap_to_sklearn"] <= 1e-5,
        "predictions": finite and score["prediction_agreement_sklearn"] >= .995,
        "convergence": not any(w["category"] in ("ConvergenceWarning", "DeviceConvergenceWarning")
                               for w in detail.get("warnings", []))
        and all(state.get("converged", False) for state in detail.get("solutions", [])),
        "backend": detail.get("backend") in {
            "sklearn": {"sklearn"}, "forced_gpu": {"mlx"},
            "direct_auto": {"cpu", "mlx"}, "patched_auto": {"cpu", "mlx", "sklearn"},
        }[variant],
    }
    if detail.get("backend") in ("cpu", "mlx"):
        checks["solver_observed"] = len(detail.get("solutions", [])) == 1 and bool(
            detail.get("solver_calls"))
    return checks


def audit_models(models, details, X, y, weights, tol):
    scores = {}
    reference = models.get("sklearn")
    for variant in VARIANTS:
        score, error = None, None
        if models.get(variant) is not None and reference is not None:
            try:
                score = quality(models[variant], reference, X, y, weights)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
        checks = fit_checks(variant, details[variant], score, tol)
        scores[variant] = {"score": score, "scoring_error": error, "checks": checks,
                           "own_checks_pass": all(checks.values())}
    reference_valid = scores["sklearn"]["own_checks_pass"]
    for item in scores.values():
        item["reference_valid"] = reference_valid
        item["passed"] = item["own_checks_pass"] and reference_valid
    return scores


def timing_summary(samples, expected):
    valid = [sample for sample in samples if sample is not None and math.isfinite(sample)]
    summary = summarize(valid) if valid else None
    return {"samples_s": samples, "summary": summary,
            "complete": len(valid) == expected and all(sample >= 0 for sample in valid)}


def speedups(timings, passed):
    result = {}
    reference = timings["sklearn"]
    for variant in VARIANTS:
        current = timings[variant]
        ratio = None
        if reference["summary"] and current["summary"] and current["summary"]["median_ms"] > 0:
            ratio = reference["summary"]["median_ms"] / current["summary"]["median_ms"]
        qualified = bool(ratio is not None and current["complete"] and reference["complete"]
                         and passed[variant] and passed["sklearn"])
        result[variant] = {"timing_ratio": ratio, "quality_qualified": qualified,
                           "speedup_over_sklearn": ratio if qualified else None}
    return result


def inference_check(operation, output, reference, rows, classes):
    if output is None or reference is None:
        return {"passed": False, "reason": "missing measured output or stock reference"}
    output, reference = np.asarray(output), np.asarray(reference)
    expected = (rows,) if operation == "predict" else (rows, classes)
    shape_ok = output.shape == expected and reference.shape == expected
    finite = shape_ok and bool(np.all(np.isfinite(output)) and np.all(np.isfinite(reference)))
    if not finite:
        return {"passed": False, "reason": "nonfinite output or unexpected output shape"}
    if operation == "predict":
        agreement = float(np.mean(output == reference))
        return {"passed": agreement >= .995, "prediction_agreement_sklearn": agreement}
    agreement = float(np.mean(np.argmax(output, axis=1) == np.argmax(reference, axis=1)))
    # sklearn may return float32 multiclass probabilities. Float64 summation
    # avoids adding another reduction error, but cannot remove their rounding.
    normalization_tolerances = [float(max(1e-8, 8 * np.finfo(array.dtype).eps))
                                if array.dtype.kind == "f" else 1e-8
                                for array in (output, reference)]
    normalization_errors = [float(np.max(np.abs(array.sum(axis=1, dtype=np.float64) - 1)))
                            for array in (output, reference)]
    probabilities_valid = bool(np.all((output >= 0) & (output <= 1))
                               and np.all((reference >= 0) & (reference <= 1))
                               and all(error <= limit for error, limit in zip(
                                   normalization_errors, normalization_tolerances, strict=True)))
    probabilities_close = bool(np.allclose(output, reference, rtol=PROBA_RTOL, atol=PROBA_ATOL))
    return {"passed": probabilities_valid and probabilities_close and agreement >= .995,
            "probabilities_valid": probabilities_valid,
            "probabilities_close": probabilities_close,
            "probability_atol": PROBA_ATOL, "probability_rtol": PROBA_RTOL,
            "normalization_max_error": normalization_errors,
            "normalization_tolerance": normalization_tolerances,
            "prediction_agreement_sklearn": agreement,
            "max_probability_difference_sklearn": float(np.max(np.abs(output - reference)))}


def benchmark_inference(models, Q, *, repeats, warmup, fit_passed, classes):
    result = {}
    for operation in OPERATIONS:
        samples, details, first, audits, orders = ({v: [] for v in VARIANTS},
            {v: [] for v in VARIANTS}, {}, {v: [] for v in VARIANTS}, [])
        for repetition in range(warmup + repeats):
            order = rotated(VARIANTS, repetition)
            outputs = {}
            if repetition >= warmup:
                orders.append(order)
            for variant in order:
                model = models.get(variant)
                if model is None:
                    elapsed, value, detail = None, None, {
                        "warnings": [], "error": {"type": "MissingModel", "message": "fit failed"}}
                else:
                    method = getattr(model, operation)
                    elapsed, value, detail = timed_call(lambda method=method: method(Q))
                detail["seconds"] = elapsed
                if repetition == 0:
                    first[variant] = detail
                if repetition >= warmup:
                    samples[variant].append(elapsed)
                    details[variant].append(detail)
                    outputs[variant] = value
            if repetition >= warmup:
                for variant in VARIANTS:
                    audit = inference_check(operation, outputs[variant], outputs["sklearn"],
                                            len(Q), classes)
                    audit["passed"] = audit["passed"] and details[variant][-1]["error"] is None
                    audits[variant].append(audit)
        timings = {v: timing_summary(samples[v], repeats) for v in VARIANTS}
        passed = {v: fit_passed[v] and all(a["passed"] for a in audits[v]) for v in VARIANTS}
        result[operation] = {
            "boundary": "reused fitted model, host NumPy queries and outputs; fit excluded",
            "first_case_calls_not_process_cold": first, "timed_orders": orders,
            "timings": timings, "call_details": details, "quality_each_call": audits,
            "quality_passed": passed, "speedups": speedups(timings, passed),
        }
    return result


def run_case(case, *, seed, dtype, repeats, warmup, tol, max_iter, queries):
    gc.collect()
    X, y, weights = make_data(case, seed, dtype=dtype)
    samples, details, audits = ({v: [] for v in VARIANTS} for _ in range(3))
    first, orders, last_models = {}, [], {}
    for repetition in range(warmup + repeats):
        order = rotated(VARIANTS, repetition)
        models, current_details = {}, {}
        if repetition >= warmup:
            orders.append(order)
        for variant in order:
            elapsed, model, detail = fit_once(variant, X, y, weights, tol=tol, max_iter=max_iter)
            if repetition == 0:
                first[variant] = detail
            if repetition >= warmup:
                samples[variant].append(elapsed)
                details[variant].append(detail)
                models[variant], current_details[variant] = model, detail
        if repetition >= warmup:
            scores = audit_models(models, current_details, X, y, weights, tol)
            for variant in VARIANTS:
                audits[variant].append(scores[variant])
            # Only the final repetition's models survive for inference. Earlier
            # models, arrays and solver objects never accumulate in the payload.
            last_models = models
    timings = {v: timing_summary(samples[v], repeats) for v in VARIANTS}
    passed = {v: all(item["passed"] for item in audits[v]) for v in VARIANTS}
    row = {
        "case": asdict(case), "seed": seed, "dtype": dtype,
        "aliases": [name for name, canonical in CASE_ALIASES.items() if canonical == case.name],
        "parameters": {"C": 1.0, "tol": tol, "max_iter": max_iter, "fit_intercept": True},
        "input_sha256": {"X": hashlib.sha256(memoryview(X)).hexdigest(),
                         "y": hashlib.sha256(memoryview(y)).hexdigest(),
                         "sample_weight": None if weights is None else hashlib.sha256(memoryview(weights)).hexdigest()},
        "fit": {"boundary": "fresh estimator construction, validation, conversion and fit",
                "first_case_calls_not_process_cold": first, "timed_orders": orders,
                "timings": timings, "fit_details": details, "quality_each_fit": audits,
                "quality_passed": passed, "speedups": speedups(timings, passed)},
        "inference": {},
    }
    query_case = replace(case, name="held_out_queries", n_samples=max(queries), weighted=False)
    Q, _, _ = make_data(query_case, seed + 10_000, dtype=dtype)
    del X, y, weights
    for count in queries:
        row["inference"][str(count)] = benchmark_inference(
            last_models, Q[:count], repeats=repeats, warmup=warmup,
            fit_passed=passed, classes=case.n_classes)
    row["quality_audit_pass"] = all(passed.values()) and all(
        all(operation["quality_passed"].values())
        for query in row["inference"].values() for operation in query.values())
    return row


def checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    os.replace(temporary, path)


def validate_resume(payload, parameters, hashes):
    if (payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("quality_certificate") != QUALITY_CERTIFICATE):
        raise ValueError("resume requires the current float64 quality certificate schema")
    if payload.get("parameters") != parameters:
        raise ValueError("resume parameters differ; use a separate output for another panel")
    if payload.get("source_integrity_before") != hashes:
        raise ValueError("source or native binary changed since the checkpoint")
    if payload.get("thread_environment_before_import") != THREAD_ENV_AT_IMPORT:
        raise ValueError("thread environment changed since the checkpoint")


def runtime_signature(environment):
    return {key: environment.get(key) for key in (
        "python", "platform", "machine", "processor", "cpu", "memory_bytes", "cores",
        "device", "mlx", "numpy", "scipy", "sklearn", "mlxlearn", "tuning",
        "native_extension", "threadpools_inside_requested_limit")}


def validate_runtime(payload, environment):
    if payload.get("sessions") and runtime_signature(payload["sessions"][0]["environment"]) != runtime_signature(environment):
        raise ValueError("runtime, package versions, tuning or detected thread pools changed since the checkpoint")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="canonical fixture name or established alias")
    parser.add_argument("--seed", action="append", type=int, help="repeat for multiple seeds; default 41")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument("--queries", action="append", type=int, help="repeat for query counts; default 1,256,4096")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--threads", type=int, help="requested threadpoolctl limit; default leaves pools unchanged")
    parser.add_argument("--output", type=Path, default=Path("bench/logistic-refresh.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args(argv)
    try:
        cases = selected_cases(args.case)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list_cases:
        print(json.dumps([asdict(case) for case in cases], indent=2))
        return 0
    seeds = list(dict.fromkeys(args.seed or [41]))
    queries = sorted(set(args.queries or [1, 256, 4096]))
    if (min(args.repeats, args.warmup, args.max_iter, *queries) < 1
        or min(seeds) < 0 or not math.isfinite(args.tol) or args.tol <= 0
        or (args.threads is not None and args.threads < 1)):
        parser.error("budgets and query sizes must be positive, seeds nonnegative, and tolerance finite positive")
    dtypes = ["float32", "float64"] if args.dtype == "both" else [args.dtype]
    parameters = {"cases": [asdict(case) for case in cases], "seeds": seeds, "dtypes": dtypes,
                  "queries": queries, "repeats": args.repeats, "warmup": args.warmup,
                  "tol": args.tol, "max_iter": args.max_iter, "requested_thread_limit": args.threads,
                  "panel": "default_tolerance" if args.tol == 1e-4 else f"tol_{args.tol:g}"}
    hashes = source_hashes()
    if args.resume:
        if not args.output.is_file():
            parser.error("--resume requires an existing checkpoint")
        payload = json.loads(args.output.read_text())
        try:
            validate_resume(payload, parameters, hashes)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.output.exists():
            parser.error("output already exists; use --resume or a new output path")
        payload = {
            "schema_version": SCHEMA_VERSION, "quality_certificate": QUALITY_CERTIFICATE,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": __doc__, "parameters": parameters, "variants": list(VARIANTS),
            "thread_environment_before_import": THREAD_ENV_AT_IMPORT,
            "thread_policy": "default pools unchanged" if args.threads is None else
                "threadpoolctl limit requested; Apple Accelerate may be uncontrolled",
            "source_integrity_before": hashes, "results": [], "sessions": [],
        }
    with threadpool_limits(limits=args.threads):
        environment = record_environment()
        environment["native_extension"] = loaded_native_extension()
        environment["threadpools_inside_requested_limit"] = threadpool_info()
        try:
            validate_runtime(payload, environment)
        except ValueError as exc:
            parser.error(str(exc))
        payload["sessions"].append({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "environment": environment})
        checkpoint(args.output, payload)
        completed = {(row["case"]["name"], row["dtype"], row["seed"]) for row in payload["results"]}
        for seed in seeds:
            for dtype in dtypes:
                for case in cases:
                    key = (case.name, dtype, seed)
                    if key in completed:
                        continue
                    print(f"Running {case.name} {dtype} seed={seed} ({case.n_samples}x{case.n_features})", flush=True)
                    try:
                        row = run_case(case, seed=seed, dtype=dtype, repeats=args.repeats,
                                       warmup=args.warmup, tol=args.tol,
                                       max_iter=args.max_iter, queries=queries)
                    except Exception as exc:
                        row = {"case": asdict(case), "dtype": dtype, "seed": seed,
                               "error": {"type": type(exc).__name__, "message": str(exc)},
                               "quality_audit_pass": False}
                    payload["results"].append(row)
                    payload["source_integrity_after"] = source_hashes()
                    payload["source_integrity_pass"] = hashes == payload["source_integrity_after"]
                    checkpoint(args.output, payload)
                    print(f"Completed {case.name}: quality={row['quality_audit_pass']}", flush=True)
                    if not payload["source_integrity_pass"]:
                        print("Source changed during measurement; checkpoint retained, stopping.", flush=True)
                        return 1
    payload["source_integrity_after"] = source_hashes()
    payload["source_integrity_pass"] = hashes == payload["source_integrity_after"]
    payload["quality_audit_pass"] = all(row["quality_audit_pass"] for row in payload["results"])
    payload["complete"] = len(payload["results"]) == len(cases) * len(dtypes) * len(seeds)
    checkpoint(args.output, payload)
    return 0 if payload["quality_audit_pass"] and payload["source_integrity_pass"] and payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
