"""Quality-qualified current KNN/SVC grid, run one case per fresh process.

The shape grid comes from benchmarks.run. KNN fits are measured once per matrix
shape and retain the three original query-count labels in their provenance.
Caller inputs retain the original grid's float64 dtype. Data generation, reference searches and quality checks
are outside timing. This is a current comparison, not a historical reproduction.

Example (run without concurrent benchmarks)::

    python -m benchmarks.grid_refresh --thread-policy 1 --repeats 7 --seed 41 \
        --output benchmarks/results/grid-refresh.json

Resume an interrupted run by repeating the command with --resume. Source,
extension, environment-policy and case identities must still match.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
MODULE = "benchmarks.grid_refresh"
SCHEMA_VERSION = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = [ROOT / "setup.py", ROOT / "pyproject.toml", ROOT / "MANIFEST.in",
             ROOT / "benchmarks/run.py", Path(__file__).resolve()]
    paths += [p for p in (ROOT / "src").rglob("*")
              if p.is_file() and p.suffix in {".py", ".pyi", ".c", ".cc", ".cpp", ".mm", ".h", ".hpp", ".metal", ".so"}]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def controlled_environment(policy):
    env = os.environ.copy()
    for key in THREAD_VARIABLES:
        if policy == "1":
            env[key] = "1"
        else:
            env.pop(key, None)
    # A previous development shell must not silently force a different route.
    for key in tuple(env):
        if key.startswith("MLXLEARN_"):
            env.pop(key)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["PYTHONNOUSERSITE"] = "1"
    return env


def all_cases():
    # Called after the command's environment policy has been applied.
    from .run import PROFILES

    cases = {}
    for original in PROFILES["full"]:
        if original.algorithm not in {"knn", "svc"}:
            continue
        item = asdict(original)
        if original.algorithm == "knn" and original.operation == "fit":
            item["n_queries"] = 0
            key = f"knn/fit/{original.n_samples}x{original.n_features}"
        else:
            key = original.label
        if key not in cases:
            cases[key] = {"id": key, **item, "original_grid_labels": []}
        cases[key]["original_grid_labels"].append(original.label)
    return list(cases.values())


def select_cases(algorithms=None, patterns=None):
    cases = [case for case in all_cases()
             if not algorithms or case["algorithm"] in algorithms]
    if patterns:
        matched = {pattern: any(fnmatch.fnmatchcase(c["id"], pattern) for c in cases)
                   for pattern in patterns}
        if not all(matched.values()):
            raise ValueError(f"Case patterns matched nothing: {[p for p, ok in matched.items() if not ok]}")
        cases = [case for case in cases
                 if any(fnmatch.fnmatchcase(case["id"], pattern) for pattern in patterns)]
    return cases


def make_data(case, seed):
    import numpy as np

    # Matrix data are shared across fit/query counts and independent of run order.
    identity = f"{seed}/{case['algorithm']}/{case['n_samples']}/{case['n_features']}"
    stable_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    rng = np.random.default_rng(stable_seed)
    n, d = case["n_samples"], case["n_features"]
    train = rng.standard_normal((n, d))
    if case["algorithm"] == "knn":
        query = rng.standard_normal((case["n_queries"] or 17, d))
        return train, None, query, None
    labels = (train[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    query_rows = min(1000, n) if case.get("operation") == "predict" else 257
    query = rng.standard_normal((query_rows, d))
    target = (query[:, 0] + rng.normal(scale=0.5, size=len(query)) > 0).astype(np.int32)
    return train, labels, query, target


def independent_neighbors(train, query, k=10, count=9, *, canonical_float32=False):
    """Direct float64 distances, bounded storage, on evenly spaced query rows."""
    import numpy as np

    rows = np.unique(np.linspace(0, len(query) - 1, min(count, len(query)), dtype=int))
    expected_indices, expected_squared, unique_rank = [], [], []
    for row in rows:
        distances = np.empty(len(train), dtype=np.float64)
        point = query[row].astype(np.float32).astype(np.float64) if canonical_float32 else query[row].astype(np.float64)
        for start in range(0, len(train), 4096):
            block = train[start:start + 4096]
            if canonical_float32:
                block = block.astype(np.float32)
            delta = block.astype(np.float64) - point
            distances[start:start + len(delta)] = np.sum(delta * delta, axis=1)
        selected = np.argpartition(distances, k - 1)[:k]
        # Include all exact boundary ties before applying deterministic index order.
        selected = np.flatnonzero(distances <= np.max(distances[selected]))
        boundary_unique = len(selected) == k
        selected = selected[np.lexsort((selected, distances[selected]))][:k]
        expected_indices.append(selected)
        expected_squared.append(distances[selected])
        unique_rank.append(boundary_unique and bool(np.all(np.diff(distances[selected]) > 0)))
    return {"rows": rows, "indices": np.asarray(expected_indices),
            "squared": np.asarray(expected_squared), "canonical_float32": canonical_float32,
            "unique_rank": np.asarray(unique_rank)}


def audit_neighbors(value, train, query, reference):
    import numpy as np

    distances, indices = (np.asarray(v) for v in value)
    shape = (len(query), 10)
    valid = (indices.shape == shape and distances.shape == shape
             and indices.dtype.kind in "iu" and np.all(np.isfinite(distances))
             and np.all(distances >= 0) and np.all((indices >= 0) & (indices < len(train))))
    if valid:
        valid = bool(np.all(np.diff(np.sort(indices, axis=1), axis=1) != 0))
    if not valid:
        return {"passed": False, "valid_shapes_ids_distances": False}
    rows = reference["rows"]
    chosen = indices[rows]
    points, queries = train[chosen], query[rows, None]
    if reference["canonical_float32"]:
        points, queries = points.astype(np.float32), queries.astype(np.float32)
    delta = points.astype(np.float64) - queries.astype(np.float64)
    squared = np.sum(delta * delta, axis=2)
    membership = bool(np.array_equal(np.sort(squared, axis=1), reference["squared"]))
    ordered = bool(np.all(np.diff(squared, axis=1) >= 0))
    distance_ok = bool(np.allclose(distances[rows], np.sqrt(squared), atol=1e-6, rtol=1e-6))
    strict = bool(np.array_equal(chosen, reference["indices"]))
    return {
        "passed": membership and ordered and distance_ok,
        "valid_shapes_ids_distances": True, "audited_query_rows": rows.tolist(),
        "reference_geometry": "canonical_float32" if reference["canonical_float32"] else "caller_float64",
        "audited_indices": chosen.tolist(),
        "indices_match_direct_float64": strict, "exact_membership": membership,
        "exact_distance_order": ordered, "reported_distances_close": distance_ok,
        "valid_exact_tie_alternative": not strict and membership and ordered,
        "max_reported_distance_error": float(np.max(np.abs(distances[rows] - np.sqrt(squared)))),
    }


def metadata(model, caught=()):
    def serial(value):
        return value.tolist() if hasattr(value, "tolist") else value

    return {
        "backend": getattr(model, "execution_backend_", "sklearn"),
        "query_backend": getattr(model, "last_query_backend_", None),
        "execution_policy": getattr(model, "execution_policy_", None),
        "sklearn_fit_method": getattr(model, "_fit_method", None),
        "n_iter": serial(getattr(model, "n_iter_", None)),
        "fit_status": serial(getattr(model, "fit_status_", None)),
        "n_support": serial(getattr(model, "n_support_", None)),
        "pair_solver": serial(getattr(model, "_pair_solver_", None)),
        "pair_kkt_violation": serial(getattr(model, "_pair_kkt_violation_", None)),
        "warnings": [{"category": warning.category.__name__, "message": str(warning.message)}
                     for warning in caught],
    }


def audit_svc(model, query, target, timed_prediction=None):
    import numpy as np

    prediction = np.asarray(model.predict(query))
    margin = np.asarray(model.decision_function(query), dtype=np.float64)
    gaps = getattr(model, "_pair_kkt_violation_", None)
    gap_ok = gaps is None or bool(np.all(np.isfinite(gaps)) and np.all(np.asarray(gaps) < model.tol + 1e-10))
    return {
        "prediction": prediction.tolist(), "margin": margin.tolist(),
        "accuracy": float(np.mean(prediction == target)),
        "passed": bool(model.fit_status_ == 0 and prediction.shape == target.shape
                       and margin.shape == target.shape and np.all(np.isfinite(margin)) and gap_ok
                       and (timed_prediction is None or np.array_equal(prediction, timed_prediction))),
        "reported_kkt_passed": gap_ok,
    }


def compare_svc_trials(trials):
    import numpy as np

    reference = trials["sklearn"]["audit"]
    reference_prediction = np.asarray(reference["prediction"])
    reference_margin = np.asarray(reference["margin"])
    passed = True
    for trial in trials.values():
        audit = trial["audit"]
        agreement = float(np.mean(np.asarray(audit["prediction"]) == reference_prediction))
        actual, expected = np.asarray(audit["margin"]), reference_margin
        close = bool(np.allclose(actual, expected, atol=0.02, rtol=0.005))
        audit.update(agreement_with_sklearn=agreement, margins_close_to_sklearn=close,
                     max_margin_difference_from_sklearn=float(np.max(np.abs(actual - expected))))
        audit["passed"] &= agreement >= 0.99 and close
        passed &= audit["passed"]
        del audit["prediction"], audit["margin"]
    return bool(passed)


def compare_knn_trials(trials, references):
    """Require strict cross-library IDs wherever precision leaves rank unchanged."""
    import numpy as np

    canonical, original = references["canonical"], references["original"]
    same_indices = np.all(canonical["indices"] == original["indices"], axis=1)
    unique = canonical["unique_rank"] & original["unique_rank"]
    stable = same_indices & unique
    ambiguous_rows = canonical["rows"][~same_indices].tolist()
    tie_rows = canonical["rows"][~unique].tolist()
    baseline = trials["sklearn_brute"]["audit"].get("audited_indices")
    for trial in trials.values():
        audit = trial["audit"]
        actual = audit.pop("audited_indices", None)
        match = (actual is not None and baseline is not None
                 and np.array_equal(np.asarray(actual)[stable], np.asarray(baseline)[stable]))
        audit.update(stable_rank_indices_match_sklearn=bool(match),
                     precision_ambiguous_query_rows=ambiguous_rows,
                     exact_distance_tie_query_rows=tie_rows)
        audit["passed"] &= match


def timed(call, mx):
    mx.synchronize()
    start = time.perf_counter()
    result = call()
    mx.synchronize()
    return time.perf_counter() - start, result


def summarize(samples):
    return {"samples_s": samples, "median_s": statistics.median(samples),
            "min_s": min(samples), "max_s": max(samples)}


def worker(case, args):
    import mlx.core as mx
    import numpy as np
    import scipy
    import sklearn
    from sklearn.neighbors import NearestNeighbors as SkNeighbors
    from sklearn.svm import SVC as SkSVC
    from threadpoolctl import threadpool_info, threadpool_limits

    import mlxlearn
    from mlxlearn.neighbors import NearestNeighbors
    from mlxlearn.patching import _estimators as patched
    from mlxlearn.svm import SVC

    if not Path(mlxlearn.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("Benchmark imported mlxlearn from outside the requested checkout")
    train, labels, query, target = make_data(case, args.seed)
    fixture_hashes = {key: hashlib.sha256(memoryview(value)).hexdigest()
                      for key, value in (("train", train), ("labels", labels), ("query", query), ("target", target))
                      if value is not None}
    is_knn = case["algorithm"] == "knn"
    factories = ({
        "direct_auto": lambda: NearestNeighbors(n_neighbors=10),
        "patched_auto": lambda: patched.NearestNeighbors(n_neighbors=10),
        "sklearn_brute": lambda: SkNeighbors(n_neighbors=10, algorithm="brute"),
        "sklearn_auto": lambda: SkNeighbors(n_neighbors=10, algorithm="auto"),
    } if is_knn else {
        "direct_auto": lambda: SVC(kernel="rbf", decision_function_shape="ovo"),
        "patched_auto": lambda: patched.SVC(kernel="rbf", decision_function_shape="ovo"),
        "sklearn": lambda: SkSVC(kernel="rbf", decision_function_shape="ovo"),
    })
    def fit(model):
        return model.fit(train) if is_knn else model.fit(train, labels)
    references = ({"canonical": independent_neighbors(train, query, canonical_float32=True),
                   "original": independent_neighbors(train, query)} if is_knn else None)
    samples = {variant: [] for variant in factories}
    audits, orders, setup = [], [], {}
    models = {}
    limits = threadpool_limits(limits=1) if args.thread_policy == "1" else contextlib.nullcontext()
    with limits:
        if case["operation"] != "fit":
            for variant, factory in factories.items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    models[variant] = fit(factory())
                setup[variant] = metadata(models[variant], caught)
        for repeat in range(args.repeats + 1):
            variants = list(factories)
            if repeat % 2:
                variants.reverse()
            offset = (repeat // 2) % len(variants)
            variants = variants[offset:] + variants[:offset]
            orders.append(variants)
            trial = {}
            for variant in variants:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if case["operation"] == "fit":
                        elapsed, model = timed(lambda variant=variant: fit(factories[variant]()), mx)
                        value = model.kneighbors(query) if is_knn else None
                    else:
                        model = models[variant]
                        call = (lambda model=model: model.kneighbors(query)) if is_knn else (lambda model=model: model.predict(query))
                        elapsed, value = timed(call, mx)
                native_backend = getattr(model, "execution_backend_", "sklearn") != "sklearn"
                audit = (audit_neighbors(value, train, query, references["canonical" if native_backend else "original"])
                         if is_knn else audit_svc(model, query, target, timed_prediction=value))
                details = metadata(model, caught)
                audit["passed"] &= not any("ConvergenceWarning" in warning["category"] for warning in details["warnings"])
                trial[variant] = {"seconds": elapsed, "model": details, "audit": audit}
                if repeat:
                    samples[variant].append(elapsed)
                if case["operation"] == "fit":
                    del model
            if is_knn:
                compare_knn_trials(trial, references)
            else:
                compare_svc_trials(trial)
            audits.append(trial)
        pools = threadpool_info()
    passed = all(t["audit"]["passed"] for trial in audits for t in trial.values())
    passed &= not any("ConvergenceWarning" in warning["category"]
                      for details in setup.values() for warning in details["warnings"])
    timings = {variant: summarize(values) for variant, values in samples.items()}
    baselines = ("sklearn_brute", "sklearn_auto") if is_knn else ("sklearn",)
    ratios = {variant: {baseline: timings[baseline]["median_s"] / timings[variant]["median_s"]
                       for baseline in baselines} for variant in ("direct_auto", "patched_auto")}
    return {
        "case": case, "input_dtype": "float64", "query_rows": len(query),
        "query_source": "independent heldout Gaussian rows", "fixture_sha256": fixture_hashes,
        "timings": timings, "warmup_trials": audits[:1], "measured_trials": audits[1:],
        "setup_models": setup, "orders_including_warmup": orders,
        "quality_passed": bool(passed), "qualified_speedup": ratios if passed else None,
        "quality_policy": ({"reference": "direct float64 distances on up to nine evenly spaced queries; canonical float32 coordinates for mlxlearn, original float64 coordinates for sklearn",
                            "membership": "exact; true distance ties accepted", "distance_atol": 1e-6, "distance_rtol": 1e-6}
                           if is_knn else {"fit_status": 0, "prediction_agreement": 0.99,
                                           "margin_atol": 0.02, "margin_rtol": 0.005}),
        "environment": {"python": sys.version, "platform": platform.platform(), "pid": os.getpid(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                        "mlx": mx.__version__, "device": mx.device_info(), "thread_policy": args.thread_policy,
                        "thread_environment": {key: os.environ.get(key) for key in THREAD_VARIABLES},
                        "threadpools_inside_policy": pools},
        "source_identity": source_identity(),
    }


def load_checkpoint(path, identity, resume):
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "identity": identity, "results": {}, "errors": {}, "attempts": []}
    if not resume:
        raise ValueError("Output already exists; choose a new output or request --resume")
    report = json.loads(path.read_text())
    if report.get("schema_version") != SCHEMA_VERSION or report.get("identity") != identity:
        raise ValueError("Resume identity differs: source, extension, parameters or selected cases changed")
    return report


def checkpoint(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(path)


def driver(args):
    cases = select_cases(args.algorithm, args.cases)
    if args.list_cases:
        for case in cases:
            print(case["id"])
        return 0
    if not args.output:
        raise ValueError("--output is required")
    identity = {"sources": source_identity(), "seed": args.seed, "repeats": args.repeats,
                "warmup": 1, "thread_policy": args.thread_policy, "cases": cases,
                "python_executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name)
                             for name in ("numpy", "scipy", "scikit-learn", "mlx", "threadpoolctl")}}
    report = load_checkpoint(args.output, identity, args.resume)
    checkpoint(args.output, report)
    for index, case in enumerate(cases, 1):
        if case["id"] in report["results"]:
            continue
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="mlxlearn-grid-") as temporary:
            destination = Path(temporary) / "result.json"
            command = [sys.executable, "-m", MODULE, "--worker-case", case["id"],
                       "--thread-policy", args.thread_policy, "--repeats", str(args.repeats),
                       "--seed", str(args.seed), "--output", str(destination)]
            started_at = datetime.now(timezone.utc).isoformat()
            completed = subprocess.run(command, cwd=ROOT, env=controlled_environment(args.thread_policy),
                                       capture_output=True, text=True, check=False)
            report["attempts"].append({"case": case["id"], "started_at": started_at,
                                       "finished_at": datetime.now(timezone.utc).isoformat(),
                                       "command": command, "returncode": completed.returncode,
                                       "stdout": completed.stdout, "stderr": completed.stderr})
            if completed.returncode:
                report["errors"][case["id"]] = {"returncode": completed.returncode,
                                                "stdout": completed.stdout, "stderr": completed.stderr}
            else:
                result = json.loads(destination.read_text())
                if result["case"] != case or result["source_identity"] != identity["sources"]:
                    raise ValueError("Worker case/source identity changed during benchmark")
                report["results"][case["id"]] = result
                report["errors"].pop(case["id"], None)
                print(f"    quality={result['quality_passed']}; speedup={result['qualified_speedup']}", flush=True)
        checkpoint(args.output, report)
    report["complete"] = len(report["results"]) == len(cases)
    report["quality_passed"] = report["complete"] and all(result["quality_passed"] for result in report["results"].values())
    checkpoint(args.output, report)
    return 0 if report["quality_passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-policy", choices=("1", "default"), required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--algorithm", nargs="+", choices=("knn", "svc"))
    parser.add_argument("--cases", nargs="+", help="Quoted case IDs or shell-style patterns")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    controlled = controlled_environment(args.thread_policy)
    for key in tuple(os.environ):
        if key.startswith("MLXLEARN_") or key in THREAD_VARIABLES:
            os.environ.pop(key)
    os.environ.update(controlled)
    if args.worker_case:
        case = next(case for case in all_cases() if case["id"] == args.worker_case)
        checkpoint(args.output, worker(case, args))
        return 0
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
