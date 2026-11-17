"""Direct Metal versus upstream Faiss e-means with fit-window process memory.

Each measurement is a fresh process and one first (unwarmed) fit. The parent
samples macOS process memory while the child is stopped at explicit fit
boundaries. Loading, initialization and shared quality scoring are untimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402


def sources():
    result = base.source_hashes()
    for name in ("compare.py", "memory.py"):
        path = HERE / name
        result[str(path.relative_to(base.REPO))] = base.file_sha256(path)
    return result


def event(phase, **fields):
    print(json.dumps({"phase": phase, **fields}), flush=True)


def worker(config):
    captured = sources()
    if captured != config["expected_source_sha256"] or base.package_versions() != config["expected_versions"]:
        raise RuntimeError("Comparison source or environment changed before the run")
    import faiss

    faiss.omp_set_num_threads(config["threads"])
    data, _, reference = base.reference_modules(config["faiss_source"], config["cache"])
    x, test, metadata = base.load_dataset(config["dataset"], data, config["cache"], config["rows"])
    n, d = x.shape
    k = config["k"] or (int(2 ** round(np.log2(16 * np.sqrt(n)))) if config["dataset"] in base.VIBE
                        else 16384 if n <= 2_000_000 else 65536)
    if k > n:
        raise ValueError("More clusters than training rows")
    initial = x[np.random.RandomState(config["seed"]).permutation(n)[:k]].copy()
    initial_hash = base.array_sha256(initial)
    gpu = None
    if config["method"] == "metal":
        import mlx.core as mx

        from mlxlearn.cluster import EMeans

        mx.synchronize()
        mx.reset_peak_memory()
        gpu = mx.metal.device_info()

        def fit():
            model = EMeans(n_clusters=k, max_iter=config["epochs"], random_state=config["seed"],
                           init=initial, compute_labels=False, precision="float32", recipe="faiss",
                           finalization="last_epoch", assignment="auto").fit(x)
            centroids = np.array(model.cluster_centers_, dtype=np.float32, copy=True)
            mx.synchronize()
            return model, centroids
    else:
        def fit():
            model = reference.NumpyEmeans(d, k, nepochs=config["epochs"], seed=config["seed"],
                                          capture_assignments=False)
            model.train(x, init_centroids=initial)
            return model, np.array(model.centroids, dtype=np.float32, copy=True)

    event("FIT_READY", n=n, d=d, k=k)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("Missing parent start handshake")
    start = time.perf_counter()
    model, centroids = fit()
    seconds = time.perf_counter() - start
    gpu_memory = None
    if config["method"] == "metal":
        gpu_memory = {"peak_active_bytes": mx.get_peak_memory(), "active_after_fit_bytes": mx.get_active_memory(),
                      "cached_after_fit_bytes": mx.get_cache_memory()}
    event("FIT_DONE", train_seconds=seconds)
    if sys.stdin.readline().strip() != "STOP_ACK":
        raise RuntimeError("Missing parent end handshake")
    test_mse = base.mse(centroids, test)
    ids = np.linspace(0, n - 1, min(n, config["train_score_rows"])).astype(int)
    train_mse = base.mse(centroids, x[ids]) if len(ids) else None
    if sources() != captured or base.package_versions() != config["expected_versions"]:
        raise RuntimeError("Comparison source or environment changed during the run")
    result = {
        "status": "completed", "kind": "faiss_direct_comparison", "config": config,
        "dataset": metadata, "n": n, "d": d, "k": k, "train_seconds": seconds,
        "test_mse": test_mse, "train_sample_mse": train_mse, "train_score_rows": len(ids),
        "initial_centroids_sha256": initial_hash, "final_centroids_sha256": base.array_sha256(centroids),
        "recipe_resolved": getattr(model, "recipe_", getattr(model, "recipe", None)),
        "n_splits": getattr(model, "n_splits_", getattr(model, "n_splits", None)),
        "source_sha256": captured, "versions": config["expected_versions"], "hardware": base.host_identity(),
        "metal_device": gpu, "mlx_memory": gpu_memory, "faiss_compile_options": faiss.get_compile_options(),
        "effective_omp_threads": faiss.omp_get_max_threads(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")},
        "faiss_reference_revision": base.FAISS_REVISION,
        "reference_backend": "Unmodified upstream NumpyEmeans; faiss.knn uses CPU IndexFlat/Accelerate/OpenMP",
        "timing_scope": "one synchronized first fit in fresh process; includes validation, upload, shuffling, finalization, centroid copy; excludes load, initialization, scoring",
        "scoring": "shared Faiss IndexFlatL2 float32 nearest assignment followed by float64 direct residuals",
        "memory_scope": "parent samples current process physical footprint and RSS only between fit handshakes; MLX allocator separately",
    }
    base.atomic_json(config["worker_output"], result)
    event("SCORED", test_mse=test_mse)


def measured_run(config, timeout):
    from memory import MemorySampler

    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # A crashed previous attempt must not supply the next worker's result.
    worker_output = Path(config["worker_output"])
    worker_output.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(config["threads"])
    if config["veclib_threads"] is None:
        environment.pop("VECLIB_MAXIMUM_THREADS", None)
    else:
        environment["VECLIB_MAXIMUM_THREADS"] = str(config["veclib_threads"])
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = output.with_suffix(".log")
    started = time.monotonic()
    sample = None
    memory = None
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                                   cwd=base.REPO, env=environment, bufsize=1)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    print(f"TIMEOUT {config['dataset']} {config['method']} after {elapsed:.0f}s", flush=True)
                    process.kill()
                    raise TimeoutError(f"Worker exceeded {timeout}s")
                ready = selector.select(min(30, timeout - elapsed))
                if not ready:
                    print(f"ALIVE {config['dataset']} {config['method']} trial={config['trial']} elapsed={elapsed:.0f}s", flush=True)
                    continue
                line = process.stdout.readline()
                if not line:
                    break
                log.write(line)
                log.flush()
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("phase") == "FIT_READY":
                    # Assign only after successful startup. An initial rusage
                    # failure must not leave an unstarted sampler to stop.
                    sample = MemorySampler(process.pid, interval_seconds=config["memory_interval_seconds"]).start()
                    process.stdin.write("GO\n")
                    process.stdin.flush()
                    print(f"FIT {config['dataset']} {config['method']} n={message['n']} k={message['k']}", flush=True)
                elif message.get("phase") == "FIT_DONE":
                    memory = sample.stop()
                    sample = None
                    if not memory.get("complete", False):
                        raise RuntimeError("Incomplete fit-window memory sampling; refusing to publish a partial peak")
                    process.stdin.write("STOP_ACK\n")
                    process.stdin.flush()
                    print(f"FIT_DONE {config['dataset']} {config['method']} {message['train_seconds']:.3f}s", flush=True)
            code = process.wait(timeout=30)
            if code:
                raise RuntimeError(f"Worker exited {code}; inspect {log_path}")
            if memory is None:
                raise RuntimeError("Worker did not provide both fit boundaries")
        finally:
            selector.close()
            try:
                if sample is not None:
                    sample.stop()
            finally:
                # Sampler cleanup is fallible; it must never strand a child
                # waiting for GO/STOP_ACK or skip closing the pipe handles.
                try:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                finally:
                    process.stdin.close()
                    process.stdout.close()
    result = json.loads(worker_output.read_text())
    if (result.get("status") != "completed"
        or result.get("kind") != "faiss_direct_comparison"
        or result.get("config") != config
        or result.get("source_sha256") != config["expected_source_sha256"]
        or result.get("versions") != config["expected_versions"]
        or result.get("hardware") != config["expected_host"]):
        raise RuntimeError("Worker result provenance does not match the requested measurement")
    result["memory"] = memory
    result["wall_seconds_including_load_and_score"] = time.monotonic() - started
    base.atomic_json(output, result)
    worker_output.unlink()
    print(json.dumps({"completed": str(output), "seconds": result["train_seconds"],
                      "test_mse": result["test_mse"],
                      "peak_footprint_gib": memory["peak_physical_footprint_bytes"] / 2**30}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=["core1m", "core10m", "vibe", "all"], default="all")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=["metal", "faiss-emeans"], default=["metal", "faiss-emeans"])
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--veclib-threads", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--memory-interval", type=float, default=.01)
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--train-score-rows", type=int, default=10000)
    parser.add_argument("--cache", default=str(base.DEFAULT_CACHE))
    parser.add_argument("--faiss-source", default=str(base.REPO / "tmp/e_means/faiss"))
    parser.add_argument("--out", default=str(HERE / "results/faiss_comparison"))
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return
    if not np.isfinite(args.epochs) or args.epochs <= 0 or args.repeats < 1 or args.threads < 1 or args.memory_interval <= 0:
        parser.error("Epochs, repeats, threads and memory interval must be positive")
    if (not np.isfinite(args.memory_interval) or not np.isfinite(args.timeout) or args.timeout <= 0
        or args.train_score_rows < 0 or (args.rows is not None and args.rows < 1)
        or (args.k is not None and args.k < 1) or (args.veclib_threads is not None and args.veclib_threads < 1)):
        parser.error("Invalid memory interval, timeout, row limit, cluster count or BLAS thread count")
    expected_sources, versions, hardware = sources(), base.package_versions(), base.host_identity()
    names = args.datasets or base.dataset_names(args.suite)
    for dataset_index, name in enumerate(names):
        for trial in range(args.repeats):
            # Alternate order to reduce a fixed method-order advantage.
            methods = args.methods if (dataset_index + trial) % 2 == 0 else args.methods[::-1]
            for method in methods:
                config = {"dataset": name, "method": method, "epochs": args.epochs, "seed": args.seed,
                          "trial": trial, "threads": args.threads, "veclib_threads": args.veclib_threads,
                          "rows": args.rows, "k": args.k, "precision": "float32", "recipe": "faiss",
                          "finalization": "last_epoch", "assignment": "auto", "warmup": False,
                          "train_score_rows": args.train_score_rows, "memory_interval_seconds": args.memory_interval,
                          "cache": str(Path(args.cache).resolve()), "faiss_source": str(Path(args.faiss_source).resolve()),
                          "expected_source_sha256": expected_sources, "expected_versions": versions,
                          "expected_host": hardware}
                digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
                path = Path(args.out) / f"{name}-{method}-s{args.seed}-t{trial}-{digest}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get("status") == "completed" and saved.get("source_sha256") == expected_sources
                        and saved.get("versions") == versions and saved.get("hardware") == hardware):
                        print(f"SKIP {path}", flush=True)
                        continue
                config["output"] = str(path.resolve())
                config["worker_output"] = str(path.resolve()) + ".worker.tmp"
                measured_run(config, args.timeout)
    print("COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
