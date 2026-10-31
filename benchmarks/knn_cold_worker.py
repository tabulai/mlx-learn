"""Exactly one first fit/query in a new process; metadata/audits follow timing."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

ENTRY_NS = time.perf_counter_ns()


def loaded_modules():
    return {name: name in sys.modules for name in (
        "numpy", "scipy", "sklearn", "sklearn.neighbors", "mlx.core", "mlxlearn._kernels._neighbors_cpu_native",
    )}


def run(args):
    before_input = loaded_modules()
    started = time.perf_counter_ns()
    import numpy as np

    numpy_done = time.perf_counter_ns()
    with np.load(args.fixture) as fixture:
        X, Q = fixture["X"], fixture["Q"]
    input_done = time.perf_counter_ns()
    after_input = loaded_modules()
    is_sklearn = args.variant.startswith("sklearn")
    algorithm = "auto" if args.variant == "sklearn_auto" else "brute"
    estimator_start = time.perf_counter_ns()
    with contextlib.ExitStack() as contexts:
        if is_sklearn:
            from sklearn.neighbors import NearestNeighbors

            if args.variant == "sklearn_1":
                from threadpoolctl import threadpool_limits

                contexts.enter_context(threadpool_limits(limits=1))
        else:
            from mlxlearn._common.config import config_context, tuning_context
            from mlxlearn.neighbors import NearestNeighbors

            contexts.enter_context(config_context(device="auto"))
            contexts.enter_context(tuning_context(force_backend=""))
        before_fit = loaded_modules()
        fit_start = time.perf_counter_ns()
        model = NearestNeighbors(n_neighbors=args.k, algorithm=algorithm).fit(X)
        fit_done = time.perf_counter_ns()
        distances, indices = model.kneighbors(Q)
        ready = time.perf_counter_ns()
        # No external MLX import/synchronization occurs before first use. The
        # public query returns NumPy arrays, establishing query completion.
        # Any work deferred by fit is included in first-query/combined latency.
        timings = {
            "numpy_import_ms": (numpy_done - started) / 1e6,
            "fixture_load_ms": (input_done - numpy_done) / 1e6,
            "estimator_setup_ms": (fit_start - estimator_start) / 1e6,
            "fit_ms": (fit_done - fit_start) / 1e6,
            "query_ms": (ready - fit_done) / 1e6,
            "fit_query_ms": (ready - fit_start) / 1e6,
            "estimator_to_result_ms": (ready - estimator_start) / 1e6,
            "worker_entry_to_result_ms": (ready - ENTRY_NS) / 1e6,
        }
        print(json.dumps({"kind": "result_ready", "timings": timings,
                          "entry_ns": ENTRY_NS, "result_ready_ns": ready,
                          "loaded_modules": {"before_input": before_input,
                                             "after_input": after_input, "before_fit": before_fit}}), flush=True)

        # Everything below, including numerical audit imports, native hashes,
        # threadpool inventory and fixture hashing, follows the result marker.
        from benchmarks.knn_broad_worker import audit_search, backend_metadata, native_cpu_metadata
        from benchmarks.knn_pipeline import sha256, source_hashes

        with np.load(args.fixture) as fixture:
            data = {name: fixture[name] for name in fixture.files}
        audit = audit_search(indices, data, distances)
        metadata = backend_metadata(model, current_mode="auto", is_sklearn=is_sklearn, operation="cold")
        if not is_sklearn:
            import mlxlearn

            if not Path(mlxlearn.__file__).resolve().is_relative_to(args.source_root.resolve()):
                raise RuntimeError("Fresh-process worker imported the wrong mlxlearn source")
        import scipy
        import sklearn
        from threadpoolctl import threadpool_info

        worker = {
            "variant": args.variant, "current_mode": "auto", "algorithm": algorithm,
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "source_root": str(args.source_root.resolve()),
            "source_hashes": source_hashes(args.source_root), "threadpools": threadpool_info(),
            "native_cpu_extension": None if is_sklearn else native_cpu_metadata(args.source_root),
            "worker_sha256": sha256(Path(__file__).read_bytes()),
        }
        print(json.dumps({"kind": "audited_result", "audit": audit, "worker": worker,
                          "fixture_sha256": sha256(args.fixture.read_bytes()),
                          "effective_fit_method": getattr(model, "_fit_method", None), **metadata}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("current", "previous_auto", "sklearn_default", "sklearn_1", "sklearn_auto"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        print(json.dumps({"kind": "error", "error": traceback.format_exc(), "audit": {"pass": False}}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
