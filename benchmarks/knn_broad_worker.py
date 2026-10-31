"""Isolated public-API workers for the broad sklearn kNN audit.

Strict canonical output checks are retained alongside a separate, exact
float64 check for valid alternative ordering/membership at genuine ties.
This module changes no production code or estimator implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmarks.knn_pipeline import run_block, sha256, source_hashes  # noqa: E402

VARIANTS = ("current", "sklearn_default", "sklearn_1", "sklearn_auto")


def native_cpu_metadata(source_root):
    """Identify optional native code without importing it before cold timing."""
    import importlib.machinery
    import importlib.util

    root = Path(source_root).resolve()
    relative = "mlxlearn/_kernels/_neighbors_cpu_native.cpp"
    source = root / relative
    result = {
        "module": "mlxlearn._kernels._neighbors_cpu_native",
        "available": False, "source": None, "binary": None,
        "discovery": "find_spec performs no additional extension import; normal estimator dependencies may already load it",
        "already_loaded_in_process": "mlxlearn._kernels._neighbors_cpu_native" in sys.modules,
    }
    # Frozen earlier GPU sources do not contain this optional implementation.
    # Never search the live installation on behalf of such a reference.
    if not source.is_file():
        return result
    result["source"] = {"path": str(source), "relative_path": relative,
                        "sha256": sha256(source.read_bytes())}
    spec = importlib.util.find_spec(result["module"])
    if spec is None or spec.origin is None:
        return result
    binary = Path(spec.origin).resolve()
    if not any(str(binary).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise RuntimeError(f"Native CPU module resolved to a non-extension file: {binary}")
    if not binary.is_relative_to(root):
        raise RuntimeError(f"Native CPU extension is outside the selected source root: {binary}")
    result["binary"] = {"path": str(binary), "sha256": sha256(binary.read_bytes())}
    result["available"] = True
    return result


def backend_metadata(model, *, current_mode, is_sklearn, operation):
    """Describe actual query execution, independently of fitted storage/policy.

    Frozen GPU sources predate per-query telemetry. Only explicitly forced GPU
    runs may use their established fitted-backend property as the fallback.
    Fit-only's query is an untimed accuracy check, never part of its fit timing.
    """
    if is_sklearn:
        policy, fitted, queried = "fixed", "sklearn", "sklearn"
    else:
        fitted = model.execution_backend_
        policy = getattr(model, "execution_policy_", "fixed")
        queried = getattr(model, "last_query_backend_", None)
        if current_mode == "auto":
            if policy != "auto" or queried not in {"cpu", "mlx"}:
                raise RuntimeError("Automatic benchmark requires actual policy/query backend telemetry")
        elif current_mode == "gpu":
            queried = fitted if queried is None else queried
            if policy != "fixed" or fitted != "mlx" or queried != "mlx":
                raise RuntimeError("Benchmark silently left the requested GPU backend")
        else:
            raise ValueError(f"Unknown current mode: {current_mode}")
    return {
        "backend": queried, "fit_backend": fitted, "execution_policy": policy,
        "query_backend": queried,
        "query_backend_phase": "untimed_fit_audit" if operation == "fit" else "timed_query",
    }


def _maximum_error(actual, expected):
    import numpy as np

    if actual.shape != expected.shape:
        return None
    difference = np.abs(actual - expected)
    return float(np.max(difference)) if difference.size and np.all(np.isfinite(difference)) else None


def audit_search(actual_indices, data, actual_distances=None):
    """Keep strict results and diagnose genuine ties without changing the gate."""
    import numpy as np

    indices = np.asarray(actual_indices)
    expected_indices = data["indices"]
    expected_squared = data["expected_squared"]
    index_ok = np.array_equal(indices, expected_indices)
    shape_ok = indices.shape == expected_indices.shape
    integer_ids = indices.dtype.kind in "iu"
    valid = shape_ok and integer_ids and bool(np.all((indices >= 0) & (indices < len(data["X"]))))
    unique = valid and bool(np.all(np.diff(np.sort(indices, axis=1), axis=1) != 0))
    actual_squared = None
    if valid and unique:
        # Convert only returned coordinates, not the entire training matrix.
        query64 = data.get("_queries64")
        if query64 is None:
            query64 = data["Q"].astype(np.float64)
        delta = data["X"][indices].astype(np.float64) - query64[:, None, :]
        actual_squared = np.sum(delta * delta, axis=2, dtype=np.float64)
    membership = actual_squared is not None and np.array_equal(
        np.sort(actual_squared, axis=1), expected_squared,
    )
    ordered = actual_squared is not None and bool(np.all(np.diff(actual_squared, axis=1) >= 0))
    distance_ok = True
    maximum_error = None
    selected_error = None
    if actual_distances is not None:
        distances = np.asarray(actual_distances)
        expected_distances = data["distances"]
        distance_ok = distances.shape == expected_distances.shape and bool(np.allclose(
            distances, expected_distances, atol=1e-6, rtol=1e-6,
        ))
        maximum_error = _maximum_error(distances, expected_distances)
        if actual_squared is not None:
            selected_error = _maximum_error(distances, np.sqrt(actual_squared))
    strict = bool(index_ok and distance_ok)
    alternative = bool(not index_ok and membership and ordered)
    if strict:
        failure_kind = None
    elif not valid or not unique:
        failure_kind = "invalid_or_duplicate_neighbor_ids"
    elif not membership:
        failure_kind = "incorrect_neighbor_membership"
    elif not ordered:
        failure_kind = "incorrect_distance_order"
    elif not distance_ok:
        failure_kind = "reported_distance_error"
    else:
        failure_kind = "valid_exact_tie_alternative"
    result = {
        "pass": strict,
        "indices_match": bool(index_ok),
        "valid_unique_ids": bool(valid and unique),
        "ties_allowed_membership_pass": bool(membership),
        "distance_order_pass": bool(ordered),
        "ties_allowed_search_pass": bool(membership and ordered and distance_ok),
        "valid_tie_alternative": alternative,
        "failure_kind": failure_kind,
    }
    if actual_distances is not None:
        result.update(distance_match=distance_ok, distance_max_abs=maximum_error,
                      distance_max_abs_from_returned_points=selected_error)
    return result


def audit_prediction(actual, data, operation):
    """Flag boundary-tie ambiguity without asserting an alternative is valid."""
    import numpy as np

    actual = np.asarray(actual)
    expected = data[operation]
    shape_ok = actual.shape == expected.shape
    if shape_ok:
        matched = (actual == expected if operation.startswith("predict")
                   else np.isclose(actual, expected, atol=2e-6, rtol=2e-5))
        per_row = matched if matched.ndim == 1 else np.all(matched, axis=tuple(range(1, matched.ndim)))
        mismatch = ~per_row
        equal = bool(np.all(matched))
    else:
        mismatch = np.ones(len(data["Q"]), bool)
        equal = False
    boundary_ties = np.asarray(data.get("boundary_tie_mask", np.zeros(len(data["Q"]), bool)))
    count = int(np.count_nonzero(mismatch))
    tie_count = int(np.count_nonzero(mismatch & boundary_ties))
    return {
        "pass": equal,
        "max_abs": _maximum_error(actual, expected),
        "shape_match": shape_ok,
        "mismatch_query_count": count,
        "mismatch_queries_first16": np.flatnonzero(mismatch)[:16].tolist(),
        "mismatch_boundary_tie_query_count": tie_count,
        "mismatch_without_boundary_tie_query_count": count - tie_count,
        "all_mismatches_on_boundary_ties": bool(count > 0 and tie_count == count),
        "failure_kind": None if equal else (
            "canonical_prediction_mismatch_on_boundary_ties" if count and tie_count == count
            else "canonical_prediction_mismatch"
        ),
        "tie_alternative_prediction_validated": False,
    }


def worker_main(args):
    import mlx.core as mx
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    variant = args.worker
    is_sklearn = variant.startswith("sklearn")
    algorithm = "auto" if variant == "sklearn_auto" else "brute"
    if is_sklearn:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
        tuning = contextlib.nullcontext
    else:
        import mlxlearn
        from mlxlearn._common.config import config_context, tuning_context
        from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

        if not Path(mlxlearn.__file__).resolve().is_relative_to(Path(args.source_root).resolve()):
            raise RuntimeError(f"Wrong package imported: {mlxlearn.__file__}")

        @contextlib.contextmanager
        def tuning():
            # Explicitly override ambient tuning/device settings in both modes.
            with config_context(device="auto"), tuning_context(
                force_backend="" if args.current_mode == "auto" else "mlx",
            ):
                yield

    policy = threadpool_limits(limits=1) if variant == "sklearn_1" else contextlib.nullcontext()
    cache = {}

    def timed(call):
        mx.synchronize()
        start = time.perf_counter()
        result = call()
        mx.synchronize()
        return result, (time.perf_counter() - start) * 1000

    def fixture(case):
        name = case["name"]
        if name not in cache:
            with np.load(Path(args.fixtures) / f"{name}.npz") as loaded:
                data = {key: loaded[key] for key in loaded.files}
            data["_queries64"] = data["Q"].astype(np.float64)
            cache[name] = data
        return cache[name]

    def fit(data, k):
        return NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(data["X"])

    def query(model, data, distance=True):
        return model.kneighbors(data["Q"], return_distance=distance)

    def fit_query(data, k):
        model = fit(data, k)
        data["last_fit_query_model"] = model
        return query(model, data)

    def audit(value, data, operation):
        if operation in {"cold", "fit", "fit_query", "kneighbors"}:
            distances, indices = value
            return audit_search(indices, data, distances)
        if operation == "indices":
            return audit_search(value, data)
        return audit_prediction(value, data, operation)

    def execute(operation, case):
        data = fixture(case)
        k = case["k"]
        with tuning():
            if operation == "cold":
                model, fit_ms = timed(partial(fit, data, k))
                data["model"] = model
                value, query_ms = timed(partial(query, model, data))
                result = {"fit_ms": fit_ms, "query_ms": query_ms, "audit": audit(value, data, operation)}
            elif operation == "fit":
                model, elapsed = timed(partial(fit, data, k))
                value = query(model, data)
                mx.synchronize()
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            elif operation == "fit_query":
                value, elapsed = timed(partial(fit_query, data, k))
                model = data["last_fit_query_model"]
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            elif operation in {"kneighbors", "indices"}:
                model = data["model"]
                value, elapsed = timed(partial(query, model, data, distance=operation != "indices"))
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            elif operation in {"predict_uniform", "proba_uniform", "regress_uniform",
                               "predict_distance", "proba_distance", "regress_distance"}:
                kind, weighting = operation.split("_")
                model_key = ("regressor" if kind == "regress" else "classifier") + weighting
                if model_key not in data:
                    cls = KNeighborsRegressor if kind == "regress" else KNeighborsClassifier
                    target = data["targets"] if kind == "regress" else data["y"]
                    data[model_key] = cls(n_neighbors=k, weights=weighting, algorithm=algorithm).fit(
                        data["X"], target,
                    )
                model = data[model_key]
                method = model.predict_proba if kind == "proba" else model.predict
                value, elapsed = timed(partial(method, data["Q"]))
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            else:
                raise ValueError(f"Unknown operation: {operation}")
            result.update(backend_metadata(model, current_mode=args.current_mode,
                                           is_sklearn=is_sklearn, operation=operation))
            result.update(algorithm=algorithm,
                          effective_fit_method=getattr(model, "_fit_method", None))
        return result

    with policy:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request["op"]
                if operation == "info":
                    import scipy
                    import sklearn

                    result = {
                        "variant": variant, "algorithm": algorithm,
                        "backend": "sklearn" if is_sklearn else (
                            "auto" if args.current_mode == "auto" else "mlx"),
                        "current_mode": args.current_mode,
                        "python": sys.version, "mlx": mx.__version__, "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "threadpools": threadpool_info(),
                        "scipy": scipy.__version__,
                        "hardware": dict(mx.device_info()), "input_dtype": "float32",
                        "source_root": str(Path(args.source_root).resolve()),
                        "source_hashes": source_hashes(Path(args.source_root)),
                        "worker_sha256": sha256(Path(__file__).read_bytes()),
                        "native_cpu_extension": None if is_sklearn else native_cpu_metadata(args.source_root),
                        "audit_policy": "Strict canonical checks remain the primary pass flag. "
                        "Search tie-permitted results additionally require exact equality of direct "
                        "float64 squared distances, valid unique IDs and distance ordering. "
                        "Prediction boundary-tie ambiguity is flagged, not accepted as validated.",
                    }
                elif operation == "integrity":
                    result = source_hashes(Path(args.source_root))
                elif operation == "block":
                    observed = []

                    def recorded_execute(operation, case, observed=observed):
                        sample = execute(operation, case)
                        observed.append({key: sample[key] for key in (
                            "backend", "fit_backend", "execution_policy", "query_backend",
                            "query_backend_phase",
                        )})
                        return sample

                    result = run_block(recorded_execute, request)
                    result["warmup_backend_records"] = observed[:result["warmup_count"]]
                else:
                    result = execute(operation, request["case"])
                print(json.dumps(result), flush=True)
            except Exception:
                print(json.dumps({"error": traceback.format_exc()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=VARIANTS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--current-mode", choices=("gpu", "auto"), default="gpu")
    worker_main(parser.parse_args())


if __name__ == "__main__":
    main()
