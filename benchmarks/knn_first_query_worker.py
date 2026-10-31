"""Initialized process, fresh model each request, first and second public queries."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ENTRY_NS = time.perf_counter_ns()


def modules():
    return {name: name in sys.modules for name in (
        "numpy", "scipy", "sklearn", "sklearn.neighbors", "mlx.core", "mlxlearn._kernels._neighbors_cpu_native",
    )}


def run(args):
    before_imports = modules()
    started = time.perf_counter_ns()
    import numpy as np

    numpy_done = time.perf_counter_ns()
    is_sklearn = args.variant.startswith("sklearn")
    algorithm = "auto" if args.variant == "sklearn_auto" else "brute"
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
        initialized = time.perf_counter_ns()
        setup = {"numpy_import_ms": (numpy_done - started) / 1e6,
                 "estimator_import_setup_ms": (initialized - numpy_done) / 1e6,
                 "worker_entry_to_initialized_ms": (initialized - ENTRY_NS) / 1e6,
                 "entry_ns": ENTRY_NS, "initialized_ns": initialized,
                 "loaded_modules_before_imports": before_imports,
                 "loaded_modules_after_imports": modules(), "pid": os.getpid(),
                 "model_prewarm_count": 0, "query_prewarm_count": 0}
        print(json.dumps({"kind": "initialized", "setup": setup}), flush=True)
        # Metadata discovery is deferred until the end; in particular find_spec
        # must not import an otherwise-unused kernel parent before first use.
        from benchmarks.knn_broad_worker import audit_search, backend_metadata, native_cpu_metadata
        from benchmarks.knn_pipeline import sha256, source_hashes

        cache, rounds = {}, {}
        model_number = 0
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["op"] == "info":
                    if not is_sklearn:
                        import mlxlearn

                        if not Path(mlxlearn.__file__).resolve().is_relative_to(args.source_root.resolve()):
                            raise RuntimeError("Wrong source imported by first-query worker")
                    from threadpoolctl import threadpool_info

                    result = {"variant": args.variant, "current_mode": "auto", "algorithm": algorithm,
                              "source_root": str(args.source_root.resolve()), "source_hashes": source_hashes(args.source_root),
                              "native_cpu_extension": None if is_sklearn else native_cpu_metadata(args.source_root),
                              "worker_sha256": sha256(Path(__file__).read_bytes()), "setup": setup,
                              "model_count": model_number, "fixture_round_counts": rounds,
                              "threadpools": threadpool_info(), "python": sys.version, "numpy": np.__version__}
                elif request["op"] == "fresh_model":
                    case = request["case"]
                    name = case["name"]
                    repeat = rounds.get(name, 0)
                    if request["round"] != repeat:
                        raise ValueError("Missing/repeated fresh-model round")
                    rounds[name] = repeat + 1
                    load_start = time.perf_counter_ns()
                    if name not in cache:
                        path = args.fixtures / (name + ".npz")
                        with np.load(path) as loaded:
                            cache[name] = {key: loaded[key] for key in loaded.files}
                    data = cache[name]
                    load_done = time.perf_counter_ns()
                    before_fit = modules()
                    model_number += 1
                    fit_start = time.perf_counter_ns()
                    model = NearestNeighbors(n_neighbors=case["k"], algorithm=algorithm).fit(data["X"])
                    fit_done = time.perf_counter_ns()
                    first_distances, first_indices = model.kneighbors(data["Q"])
                    first_done = time.perf_counter_ns()
                    first_backend = backend_metadata(model, current_mode="auto", is_sklearn=is_sklearn, operation="kneighbors")
                    second_start = time.perf_counter_ns()
                    second_distances, second_indices = model.kneighbors(data["Q"])
                    second_done = time.perf_counter_ns()
                    second_backend = backend_metadata(model, current_mode="auto", is_sklearn=is_sklearn, operation="kneighbors")
                    # Public calls return NumPy arrays, so no harness MLX import
                    # or pre-query synchronization initializes the device early.
                    result = {"round": repeat, "model_number": model_number,
                              "first_model_in_worker": model_number == 1, "first_model_for_fixture": repeat == 0,
                              "fixture_load_ms": (load_done - load_start) / 1e6,
                              "loaded_modules_before_fit": before_fit,
                              "timings": {"fit_ms": (fit_done - fit_start) / 1e6,
                                          "first_query_ms": (first_done - fit_done) / 1e6,
                                          "fit_first_query_ms": (first_done - fit_start) / 1e6,
                                          "second_query_ms": (second_done - second_start) / 1e6},
                              "first": {"audit": audit_search(first_indices, data, first_distances), **first_backend},
                              "second": {"audit": audit_search(second_indices, data, second_distances), **second_backend},
                              "effective_fit_method": getattr(model, "_fit_method", None)}
                    # Drop each model outside the timers. No model or query
                    # result is reused by a later fresh-model request.
                    del model, first_distances, first_indices, second_distances, second_indices
                else:
                    raise ValueError("Unknown first-query worker request")
                print(json.dumps(result), flush=True)
            except Exception:
                print(json.dumps({"error": traceback.format_exc(), "audit": {"pass": False}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("current", "previous_auto", "sklearn_default", "sklearn_1", "sklearn_auto"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
