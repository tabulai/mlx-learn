"""Resident-data BF16 e-means benchmarks using the audited H100 timing protocol.

All fits run on Apple Silicon. H100 values are published context, not new GPU
measurements. Historical benchmark results and drivers are never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402

KIND = "h100_protocol_metal"
ASSIGNMENT = {"baseline": "gemm", "optimized": "auto"}


def sources(protocol):
    result = base.source_hashes()
    paths = [HERE / "h100.py", HERE / "memory.py", Path(protocol),
             *sorted((base.REPO / "src/mlxlearn/_kernels").glob("*.py"))]
    for path in paths:
        key = str(path.relative_to(base.REPO)) if path.is_relative_to(base.REPO) else str(path)
        result[key] = base.file_sha256(path)
    return result


def event(phase, **values):
    print(json.dumps({"phase": phase, **values}), flush=True)


def expected_calls(first_seconds):
    return 3 if first_seconds < 2.0 else 1


def check_environment(config):
    if (sources(config["protocol_path"]) != config["expected_source_sha256"]
        or base.package_versions() != config["expected_versions"]
        or base.host_identity() != config["expected_host"]):
        raise RuntimeError("Benchmark source, software, or host changed")


def validate_dataset(metadata, n, d, config):
    if config["rows"] is not None:
        return
    entry = config["protocol_config"]
    if (n, d) != (entry["n"], entry["d"]):
        raise RuntimeError("Loaded full dataset shape differs from protocol manifest")
    for key in ("train_sample_sha256", "test_sha256", "train_rows_used", "test_rows_used", "dimension"):
        if key not in entry["dataset_identity"] or metadata.get(key) != entry["dataset_identity"][key]:
            raise RuntimeError(f"Loaded dataset identity {key} differs from protocol manifest")


def worker(config):
    check_environment(config)
    import faiss
    import mlx.core as mx

    from mlxlearn.cluster import EMeans

    faiss.omp_set_num_threads(config["threads"])
    data, _, _ = base.reference_modules(config["faiss_source"], config["cache"])
    x, test, metadata = base.load_dataset(config["dataset"], data, config["cache"], config["rows"])
    n, d = x.shape
    k = config["k_override"] or config["protocol_config"]["k"]
    if k > n:
        raise ValueError("More clusters than input rows")
    validate_dataset(metadata, n, d, config)
    initial = x[np.random.RandomState(config["seed"]).permutation(n)[:k]].copy()
    initial_hash = base.array_sha256(initial)
    kwargs = {"n_clusters": k, "max_iter": config["epochs"], "random_state": config["seed"], "init": initial,
              "compute_labels": False, "recipe": "faiss", "batch_law": "throughput", "precision": "bfloat16",
              "finalization": "last_epoch", "assignment": ASSIGNMENT[config["mode"]]}
    preparation_start = time.perf_counter()
    preparer = EMeans(**kwargs)
    prepared = preparer.prepare_device(x)
    mx.synchronize()
    preparation_seconds = time.perf_counter() - preparation_start
    del preparer

    def fit():
        # The exact same init object preserves the preparation signature.
        model = EMeans(**kwargs)
        output = model.train_device(prepared, capture_assignments=True)
        mx.synchronize()
        return model, output

    warmup_start = time.perf_counter()
    model, output = fit()
    warmup_seconds = time.perf_counter() - warmup_start
    del model, output
    mx.synchronize()
    event("WARMUP_DONE", seconds=warmup_seconds)
    samples, allocator_samples = [], []
    first_centers = first_recipe = first_history = first_assignments = first_splits = None
    for call_index in range(3):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        event("FIT_READY", call=call_index, n=n, d=d, k=k)
        if sys.stdin.readline().strip() != "GO":
            raise RuntimeError("Missing parent GO handshake")
        start = time.perf_counter()
        model, output = fit()
        seconds = time.perf_counter() - start
        samples.append(seconds)
        allocator_samples.append({"peak_active_bytes": mx.get_peak_memory(),
                                  "active_after_fit_bytes": mx.get_active_memory(),
                                  "cached_after_fit_bytes": mx.get_cache_memory()})
        event("FIT_DONE", call=call_index, train_seconds=seconds)
        if sys.stdin.readline().strip() != "STOP_ACK":
            raise RuntimeError("Missing parent STOP_ACK handshake")
        if call_index == 0:
            first_centers = np.array(output.centers, dtype=np.float32, copy=True)
            first_recipe = dict(model.recipe_)
            first_history = list(output.history)
            first_splits = int(output.split_count.item())
            if output.assignments is None:
                raise RuntimeError("Device fit omitted requested last-pass labels")
            # Record metadata only; the reference does not export its labels.
            first_assignments = {"shape": list(output.assignments.shape), "dtype": str(output.assignments.dtype),
                                 "semantics": "online final-pass labels, no extra nearest-centroid pass"}
        del model, output
        if len(samples) >= expected_calls(samples[0]):
            break
    test_mse = base.mse(first_centers, test)
    score_ids = np.linspace(0, n - 1, min(n, config["train_score_rows"])).astype(int)
    train_mse = base.mse(first_centers, x[score_ids]) if len(score_ids) else None
    check_environment(config)
    result = {
        "status": "completed", "kind": KIND, "config": config, "dataset": metadata,
        "n": n, "d": d, "k": k, "train_seconds": median(samples), "train_seconds_samples": samples,
        "test_mse": test_mse, "train_sample_mse": train_mse, "train_score_rows": len(score_ids),
        "initial_centroids_sha256": initial_hash, "final_centroids_sha256": base.array_sha256(first_centers),
        "recipe_resolved": first_recipe, "iteration_history": first_history, "n_splits": first_splits,
        "captured_assignments": first_assignments, "mlx_memory_samples": allocator_samples,
        "preparation_seconds": preparation_seconds, "warmup_seconds": warmup_seconds,
        "source_sha256": config["expected_source_sha256"], "versions": base.package_versions(),
        "hardware": base.host_identity(), "metal_device": mx.metal.device_info(),
        "faiss_reference_revision": base.FAISS_REVISION,
        "timing_scope": "resident FP32 input; identical full warmup; new model plus device fit and sync; shuffling, BF16 casts, online history and final-pass labels included; preparation/upload and host output/scoring excluded",
        "quality_fit_index": 0,
        "scoring": "shared Faiss IndexFlatL2 float32 nearest assignment then direct float64 squared residuals",
        "memory_scope": "sampled per timed-fit process footprint/RSS including resident-input baseline; warmup, cache clearing and output export excluded; MLX allocator recorded separately",
        "differences_from_h100": ["Apple Metal hardware and kernels, no local H100",
                                   "NumPy permutations/donor RNG instead of CUDA Torch RNG",
                                   "full input finite/range validation in untimed preparation; per-fit signature/state checks inside timing",
                                   "online objective/history accumulation FP32 on Metal versus upstream FP64"],
    }
    base.atomic_json(config["worker_output"], result)
    event("SCORED", test_mse=test_mse)


def validate_result(result, config, memories):
    if (result.get("status") != "completed" or result.get("kind") != KIND
        or result.get("config") != config or result.get("source_sha256") != config["expected_source_sha256"]
        or result.get("versions") != config["expected_versions"] or result.get("hardware") != config["expected_host"]):
        raise RuntimeError("Worker result provenance mismatch")
    samples = result.get("train_seconds_samples", [])
    if (not samples or any(not math.isfinite(s) or s <= 0 for s in samples)
        or len(samples) != expected_calls(samples[0]) or len(memories) != len(samples)
        or not math.isclose(result["train_seconds"], median(samples), rel_tol=1e-12)
        or not math.isfinite(result.get("test_mse", math.nan)) or result["test_mse"] < 0):
        raise RuntimeError("Invalid timed samples, adaptive call count, or quality")
    if config["memory"] and any(not m or not m.get("complete") for m in memories):
        raise RuntimeError("Incomplete fit-window memory sampling")


def measured_run(config, timeout):
    from memory import MemorySampler

    output, temporary = Path(config["output"]), Path(config["worker_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(config["threads"]), PYTHONUNBUFFERED="1")
    environment.pop("VECLIB_MAXIMUM_THREADS", None)
    started, sample, active_call, memories = time.monotonic(), None, None, []
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                                   cwd=base.REPO, env=environment, bufsize=1)
        selector = None
        try:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    raise TimeoutError(f"Worker exceeded {timeout}s")
                if not selector.select(min(30, timeout - elapsed)):
                    print(f"ALIVE {config['dataset']} {config['mode']} seed={config['seed']} {elapsed:.0f}s", flush=True)
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
                    if active_call is not None or message.get("call") != len(memories):
                        raise RuntimeError("Out-of-order FIT_READY")
                    if config["memory"]:
                        sample = MemorySampler(process.pid, interval_seconds=config["memory_interval_seconds"]).start()
                    active_call = message["call"]
                    process.stdin.write("GO\n")
                    process.stdin.flush()
                elif message.get("phase") == "FIT_DONE":
                    if active_call is None or message.get("call") != active_call:
                        raise RuntimeError("Out-of-order FIT_DONE")
                    memory = sample.stop() if sample is not None else None
                    sample = None
                    if config["memory"] and not memory.get("complete"):
                        raise RuntimeError("Incomplete fit-window memory sampling")
                    memories.append(memory)
                    active_call = None
                    process.stdin.write("STOP_ACK\n")
                    process.stdin.flush()
                    print(f"FIT_DONE {config['dataset']} {config['mode']} s{config['seed']} call{message['call']} {message['train_seconds']:.3f}s", flush=True)
            code = process.wait(timeout=30)
            if code:
                raise RuntimeError(f"Worker exited {code}; inspect {output.with_suffix('.log')}")
            if active_call is not None or not memories:
                raise RuntimeError("Incomplete fit handshakes")
        finally:
            try:
                if selector is not None:
                    selector.close()
            finally:
                try:
                    if sample is not None:
                        sample.stop()
                finally:
                    try:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                    finally:
                        process.stdin.close()
                        process.stdout.close()
    result = json.loads(temporary.read_text())
    validate_result(result, config, memories)
    result["memory_samples"] = memories
    result["wall_seconds_including_load_warmup_score"] = time.monotonic() - started
    base.atomic_json(output, result)
    temporary.unlink()
    print(json.dumps({"completed": str(output), "seconds": result["train_seconds"], "test_mse": result["test_mse"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=["core1m", "vibe", "all"], default="all")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=list(ASSIGNMENT), default=list(ASSIGNMENT))
    parser.add_argument("--seeds", nargs="+", type=int, help="Override per-dataset manifest seeds, e.g. --seeds 10 for pilot")
    parser.add_argument("--protocol", type=Path, default=HERE / "h100_protocol.json")
    parser.add_argument("--rows", type=int, help="Explicit subset for smoke tests; excluded from full-scale report")
    parser.add_argument("--k", type=int, help="Explicit smoke-test cluster count")
    parser.add_argument("--epochs", type=float, help="Explicit diagnostic budget override")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--train-score-rows", type=int, default=10000)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--memory-interval", type=float, default=.01)
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--cache", default=str(base.DEFAULT_CACHE))
    parser.add_argument("--faiss-source", default=str(base.REPO / "tmp/e_means/faiss"))
    parser.add_argument("--out", type=Path, default=HERE / "results/h100_protocol")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return
    if (args.threads < 1 or args.train_score_rows < 0
        or any(v is not None and (not math.isfinite(v) or v <= 0) for v in (args.rows, args.k, args.epochs, args.memory_interval, args.timeout))
        or args.seeds is not None and any(not 0 <= s < 2**32 for s in args.seeds)):
        parser.error("Invalid positive size/timing/thread parameter or seed")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    configurations = protocol["configurations"]
    known = {c["dataset"] for c in configurations}
    if args.datasets and not set(args.datasets) <= known:
        parser.error("Unknown dataset; only configurations in the audited manifest are supported")
    expected_sources, versions, hardware = sources(protocol_path), base.package_versions(), base.host_identity()
    for index, entry in enumerate(configurations):
        if args.datasets and entry["dataset"] not in args.datasets or args.suite != "all" and entry["suite"] != args.suite:
            continue
        for seed in args.seeds if args.seeds is not None else entry["seed_ids"]:
            modes = args.modes if (index + seed) % 2 == 0 else args.modes[::-1]
            for mode in modes:
                config = {"dataset": entry["dataset"], "mode": mode, "seed": seed,
                          "epochs": args.epochs if args.epochs is not None else entry["epochs"],
                          "epochs_override": args.epochs, "rows": args.rows, "k_override": args.k,
                          "precision": "bfloat16", "recipe": "faiss", "finalization": "last_epoch",
                          "assignment": ASSIGNMENT[mode], "warmup": True, "capture_assignments": True,
                          "threads": args.threads, "train_score_rows": args.train_score_rows,
                          "memory": not args.no_memory, "memory_interval_seconds": args.memory_interval,
                          "protocol_path": str(protocol_path), "protocol_config": entry,
                          "cache": str(Path(args.cache).resolve()), "faiss_source": str(Path(args.faiss_source).resolve()),
                          "expected_source_sha256": expected_sources, "expected_versions": versions, "expected_host": hardware}
                digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
                path = args.out / f"{entry['dataset']}-{mode}-s{seed}-{digest}.json"
                config["output"], config["worker_output"] = str(path.resolve()), str(path.resolve()) + ".worker.tmp"
                if path.exists() and not args.force:
                    saved = json.loads(path.read_text())
                    validate_result(saved, config, saved.get("memory_samples", []))
                    print(f"SKIP {path}", flush=True)
                    continue
                measured_run(config, args.timeout)


if __name__ == "__main__":
    main()
