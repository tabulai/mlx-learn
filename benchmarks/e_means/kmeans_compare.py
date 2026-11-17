"""Warmed resident FP32 Lloyd/e-means comparison with final labels and inertia.

The four methods use identical data, initial centroids, integer training-pass
budgets and test scoring. e-means retains its moving mini-batch assignments;
Lloyd retains fixed centroids during each pass. No algorithm identity is claimed.
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
import large_compare as previous  # noqa: E402

import run as base  # noqa: E402

KIND = "integer_pass_clustering_final_labels_v1"
MODES = ("metal-kmeans", "sklearn-kmeans", "metal-emeans", "faiss-emeans")


def sources(protocol):
    paths = [Path(__file__), Path(protocol), HERE / "large_compare.py", HERE / "run.py",
             HERE / "memory.py", HERE / "vibe_manifest.json",
             *sorted((base.REPO / "src/mlxlearn").rglob("*.py"))]
    return {(str(p.resolve().relative_to(base.REPO)) if p.resolve().is_relative_to(base.REPO)
             else str(p.resolve())): base.file_sha256(p) for p in paths}


def sklearn_sources():
    import sklearn.cluster

    folder = Path(sklearn.cluster.__file__).parent
    paths = [*folder.glob("_kmeans.py"), *folder.glob("_k_means*.so")]
    return {str(p): base.file_sha256(p) for p in sorted(paths)}


def check_environment(config):
    if (sources(config["protocol_path"]) != config["expected_source_sha256"]
            or previous.reference_sources(config["faiss_source"]) != config["expected_reference_source_sha256"]
            or sklearn_sources() != config["expected_sklearn_sha256"]
            or base.package_versions() != config["expected_versions"]
            or base.host_identity() != config["expected_host"]):
        raise RuntimeError("Measured source, reference, host or software changed")


def event(phase, **values):
    print(json.dumps({"phase": phase, **values}), flush=True)


def final_cpu_assign(x, centers):
    """Bound assignment scratch; final FP32 labels/distances for original Faiss."""
    import faiss

    index = faiss.IndexFlatL2(centers.shape[1])
    index.add(centers)
    labels = np.empty(len(x), dtype=np.int32)
    inertia = 0.0
    for start in range(0, len(x), 65536):
        end = min(start + 65536, len(x))
        distances, ids = index.search(x[start:end], 1)
        labels[start:end] = ids[:, 0]
        # Direct residuals avoid cancellation in Gram distance objective.
        delta = x[start:end] - centers[ids[:, 0]]
        inertia += float(np.sum(delta * delta, dtype=np.float64))
    return labels, inertia


def final_metal_assign(data, centers, offset):
    """Center only bounded tiles, never a second full 38-GiB DINO matrix."""
    import mlx.core as mx

    from mlxlearn._kernels.e_means import assign

    labels, losses = [], []
    centered_centers = centers - offset
    for start in range(0, len(data), 65536):
        ids, distances = assign(data[start:start + 65536] - offset, centered_centers)
        ids, loss = ids.astype(mx.int32), mx.sum(distances)
        mx.eval(ids, loss)
        labels.append(ids)
        losses.append(loss)
    labels, inertia = mx.concatenate(labels), mx.sum(mx.stack(losses))
    mx.eval(labels, inertia)
    return labels, inertia


def worker(config):
    check_environment(config)
    import faiss
    from memory import snapshot
    from threadpoolctl import threadpool_info

    faiss.omp_set_num_threads(config["threads"])
    data, _, reference = base.reference_modules(config["faiss_source"], config["cache"])
    x, test, metadata = base.load_dataset(config["dataset"], data, config["cache"], config["rows"])
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    previous.validate_dataset(metadata, n, d, config)
    k = config["k_override"] or config["protocol_config"]["k"]
    if k > n:
        raise ValueError("More clusters than rows")
    initial = x[np.random.RandomState(config["seed"]).permutation(n)[:k]].copy()
    initial_hash = base.array_sha256(initial)
    score_ids = np.linspace(0, n - 1, min(n, config["train_score_rows"])).astype(int)
    train_sample = x[score_ids].copy()
    train_sample_hash = base.array_sha256(train_sample)
    mode, mx, metal_device = config["mode"], None, None
    start_preparation = time.perf_counter()
    is_metal = mode.startswith("metal-")
    if is_metal:
        import mlx.core as mx

        from mlxlearn.cluster import EMeans, KMeans

        if mode == "metal-kmeans":
            estimator = KMeans
            kwargs = {"n_clusters": k, "init": initial, "n_init": 1, "random_state": config["seed"],
                          "max_iter": config["epochs"], "tol": 0.0, "algorithm": "lloyd"}
        else:
            estimator = EMeans
            kwargs = {"n_clusters": k, "init": initial, "random_state": config["seed"], "max_iter": config["epochs"],
                          "recipe": "faiss", "batch_law": "quality", "precision": "float32", "finalization": "last_epoch",
                          "compute_labels": False, "assignment": "auto"}
        preparer = estimator(**kwargs)
        prepared = preparer.prepare_device(x)
        mx.synchronize()
        metal_device = mx.metal.device_info()
        del preparer, x
        gc.collect()

        def fit():
            model = estimator(**kwargs)
            output = model.train_device(prepared)
            if mode == "metal-emeans":
                labels, inertia = final_metal_assign(prepared.data, output.centers, prepared.offset)
                iterations, history = output.epochs, list(output.history)
            else:
                labels, inertia = output.labels, output.inertia
                iterations, history = output.n_iter, list(output.history)
            mx.synchronize()
            return model, output.centers, labels, inertia, iterations, history
    elif mode == "sklearn-kmeans":
        from sklearn.cluster import KMeans

        def fit():
            # copy_x=False avoids a full additional input copy; exact source
            # coordinates are reloaded outside timing after every prior call.
            model = KMeans(n_clusters=k, init=initial, n_init=1, max_iter=config["epochs"],
                           tol=0.0, random_state=config["seed"], algorithm="lloyd", copy_x=False)
            model.fit(x)
            return model, model.cluster_centers_, model.labels_, model.inertia_, model.n_iter_, []
    else:
        def fit():
            model = reference.NumpyEmeans(d, k, nepochs=config["epochs"], batch_law="quality",
                                          seed=config["seed"], capture_assignments=False,
                                          max_points_per_centroid=None)
            model.train(x, init_centroids=initial)
            labels, inertia = final_cpu_assign(x, model.centroids)
            return model, model.centroids, labels, inertia, len(model.iteration_stats), list(model.iteration_stats)
    preparation_seconds = time.perf_counter() - start_preparation
    preparation_memory = snapshot(os.getpid())
    event("WARMUP_START", n=n, d=d, k=k)
    warmup_start = time.perf_counter()
    output = fit()
    warmup_seconds = time.perf_counter() - warmup_start
    del output
    gc.collect()
    if mx is not None:
        mx.synchronize()
    event("WARMUP_DONE", seconds=warmup_seconds)
    samples, allocator_samples = [], []
    for call in range(3):
        if mode == "sklearn-kmeans":
            # sklearn recenters/restores its input in place; restore exact source
            # bits rather than feed its rounded warmup input into the timed fit.
            del x
            gc.collect()
            x, reloaded_test, reloaded_metadata = base.load_dataset(
                config["dataset"], data, config["cache"], config["rows"])
            x = np.ascontiguousarray(x, dtype=np.float32)
            if (reloaded_metadata != metadata
                    or base.array_sha256(x[score_ids]) != train_sample_hash):
                raise RuntimeError("Reloaded sklearn input differs from original source")
            del reloaded_test
            # Force the fresh input resident and check finiteness before timing.
            if not math.isfinite(float(x.min())) or not math.isfinite(float(x.max())):
                raise RuntimeError("Reloaded input is nonfinite")
        gc.collect()
        if mx is not None:
            mx.synchronize()
            mx.clear_cache()
            mx.synchronize()
            mx.reset_peak_memory()
        event("FIT_READY", call=call, n=n, d=d, k=k)
        if sys.stdin.readline().strip() != "GO":
            raise RuntimeError("Missing GO")
        started = time.perf_counter()
        output = fit()
        elapsed = time.perf_counter() - started
        samples.append(elapsed)
        allocator_samples.append(None if mx is None else {
            "peak_active_bytes": mx.get_peak_memory(), "active_after_fit_bytes": mx.get_active_memory(),
            "cached_after_fit_bytes": mx.get_cache_memory()})
        event("FIT_DONE", call=call, train_seconds=elapsed)
        if sys.stdin.readline().strip() != "STOP_ACK":
            raise RuntimeError("Missing STOP_ACK")
        model, centers, labels, inertia, iterations, history = output
        if call == 0:
            first_centers = np.array(centers, dtype=np.float32, copy=True)
            first_labels = np.array(labels, copy=True)
            first_inertia = float(inertia.item() if hasattr(inertia, "item") else inertia)
            first_iterations, first_history = int(iterations), history
            recipe = dict(model.recipe_ if mode == "metal-emeans" else model.recipe) if mode.endswith("emeans") else None
            # Small direct label audit against returned final centers; all train
            # rows are labelled, but scoring/reference work stays outside timing.
            label_sample = first_labels[score_ids].copy()
            first_labels_hash = base.array_sha256(first_labels)
            first_labels_metadata = {"shape": list(first_labels.shape), "dtype": str(first_labels.dtype),
                                     "semantics": "nearest final centroids; no update after final assignment"}
            del first_labels
            threads_observed = getattr(model, "_n_threads", None)
        del output, model, centers, labels, inertia
        if len(samples) >= previous.expected_calls(samples[0]):
            break
    test_mse = base.mse(first_centers, test)
    train_mse = base.mse(first_centers, train_sample)
    audit_index = faiss.IndexFlatL2(d)
    audit_index.add(first_centers)
    _, audit_ids = audit_index.search(train_sample, 1)
    audit_delta = train_sample.astype(np.float64) - first_centers[label_sample].astype(np.float64)
    selected_mse = float(np.mean(np.sum(audit_delta * audit_delta, axis=1)))
    check_environment(config)
    result = {"status": "completed", "kind": KIND, "config": config, "dataset": metadata, "n": n, "d": d, "k": k,
                  "train_seconds": median(samples), "train_seconds_samples": samples, "quality_fit_index": 0,
                  "test_mse": test_mse, "train_sample_mse": train_mse, "final_inertia": first_inertia,
                  "label_sample_mse": selected_mse, "label_sample_agreement": float(np.mean(label_sample == audit_ids[:, 0])),
                  "final_labels": first_labels_metadata, "final_labels_sha256": first_labels_hash,
                  "actual_iterations": first_iterations, "iteration_history": first_history, "recipe_resolved": recipe,
                  "train_score_sample_sha256": train_sample_hash, "train_score_rows": len(train_sample),
                  "initial_centroids_sha256": initial_hash, "final_centroids_sha256": base.array_sha256(first_centers),
                  "source_sha256": config["expected_source_sha256"], "reference_source_sha256": config["expected_reference_source_sha256"],
                  "sklearn_source_sha256": config["expected_sklearn_sha256"], "versions": base.package_versions(), "hardware": base.host_identity(),
                  "mlx_memory_samples": allocator_samples, "preparation_seconds": preparation_seconds,
                  "preparation_process_memory": preparation_memory, "warmup_seconds": warmup_seconds,
                  "metal_device": metal_device, "effective_omp_threads": faiss.omp_get_max_threads(),
                  "sklearn_effective_threads": threads_observed, "threadpools": threadpool_info(),
                  "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                  "full_host_input_released": is_metal,
                  "timing_scope": "resident FP32 input and explicit initialization; fresh model plus full training, final nearest-centroid labels/inertia and synchronization; excludes load, initial device preparation, warmup, cache clearing, host export and quality scoring",
                  "memory_scope": "sampled timed-fit process physical footprint/RSS including resident-input baseline; preparation/lifetime and MLX allocator separate",
                  "numerical_notes": "Lloyd fixed-centroid pass versus e-means moving mini-batches; FP32 training with backend-specific reductions and empty-cluster handling. sklearn copy_x=False centering, validation and sample-weight allocation inside fit; Metal resident preparation outside timer."}
    base.atomic_json(config["worker_output"], result)
    event("SCORED", test_mse=test_mse)


def validate_result(result, config, memories):
    if (result.get("status") != "completed" or result.get("kind") != KIND or result.get("config") != config
            or result.get("source_sha256") != config["expected_source_sha256"]
            or result.get("reference_source_sha256") != config["expected_reference_source_sha256"]
            or result.get("sklearn_source_sha256") != config["expected_sklearn_sha256"]
            or result.get("versions") != config["expected_versions"] or result.get("hardware") != config["expected_host"]):
        raise RuntimeError("Worker provenance differs")
    if config["mode"] not in MODES or config["precision"] != "float32" or config["epochs"] != int(config["epochs"]):
        raise RuntimeError("Invalid mode, precision or integer budget")
    if (isinstance(config["epochs"], bool) or config["epochs"] < 1
            or any(config.get(key) != expected for key, expected in
                   {"n_init": 1, "tol": 0.0, "final_labels": True, "warmup": True, "memory": True}.items())):
        raise RuntimeError("Configuration differs from the complete warmed final-label protocol")
    samples = result.get("train_seconds_samples", [])
    if (not samples or any(not math.isfinite(v) or v <= 0 for v in samples)
            or len(samples) != previous.expected_calls(samples[0]) or len(memories) != len(samples)
            or not math.isclose(result["train_seconds"], median(samples), rel_tol=1e-12)
            or any(not m or not m.get("complete") for m in memories)):
        raise RuntimeError("Invalid adaptive timing or memory samples")
    previous.validate_dataset(result["dataset"], result["n"], result["d"], config)
    if result["k"] != (config["k_override"] or config["protocol_config"]["k"]):
        raise RuntimeError("Wrong cluster count")
    if not config["diagnostic_subset"] and (
            config["epochs"] != config["protocol_config"]["epochs"]
            or config["seed"] != 10 or config["seed"] not in config["protocol_config"]["seed_ids"]
            or any(config.get(key) is not None for key in ("rows", "k_override", "epochs_override"))
            or config["threads"] != 8 or config["train_score_rows"] != 10000
            or config["memory_interval_seconds"] != .01):
        raise RuntimeError("Changed full campaign budget, seed, override or measurement condition")
    if not 1 <= result["actual_iterations"] <= config["epochs"]:
        raise RuntimeError("Unexpected iteration count")
    labels = result.get("final_labels", {})
    if labels.get("shape") != [result["n"]] or labels.get("dtype") not in {"int32", "int64", "uint32"}:
        raise RuntimeError("Missing final labels")
    if (labels.get("semantics") != "nearest final centroids; no update after final assignment"
            or result.get("quality_fit_index") != 0
            or result.get("train_score_rows") != min(result["n"], config["train_score_rows"])):
        raise RuntimeError("Final-label or first-fit scoring semantics differ")
    for field in ("initial_centroids_sha256", "final_centroids_sha256", "final_labels_sha256", "train_score_sample_sha256"):
        digest = result.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Missing or invalid {field}")
    agreement = result.get("label_sample_agreement", math.nan)
    if not math.isfinite(agreement) or not 0 <= agreement <= 1:
        raise RuntimeError("Invalid final-label agreement diagnostic")
    if result.get("full_host_input_released") is not config["mode"].startswith("metal-"):
        raise RuntimeError("Full host input ownership policy differs")
    for field in ("test_mse", "train_sample_mse", "final_inertia", "label_sample_mse"):
        if not math.isfinite(result.get(field, math.nan)) or result[field] < 0:
            raise RuntimeError(f"Invalid {field}")
    if result["label_sample_mse"] > result["train_sample_mse"] * 1.0001 + 1e-7:
        raise RuntimeError("Final labels differ materially from nearest final centers")
    if (result.get("effective_omp_threads") != config["threads"]
            or result.get("thread_environment", {}).get("OMP_NUM_THREADS") != str(config["threads"])
            or result.get("thread_environment", {}).get("VECLIB_MAXIMUM_THREADS") is not None
            or config["mode"] == "sklearn-kmeans" and result["sklearn_effective_threads"] != config["threads"]):
        raise RuntimeError("Thread configuration differs")
    if config["mode"].endswith("emeans"):
        recipe = result["recipe_resolved"]
        if (recipe["nepochs"] != config["epochs"] or recipe["batch_law"] != "quality"
                or recipe["epoch_rows"] != [result["n"]] * config["epochs"]
                or recipe["seed"] != config["seed"] or result["actual_iterations"] != config["epochs"]
                or len(result["iteration_history"]) != config["epochs"]):
            raise RuntimeError("e-means pass budget differs")
        if config["mode"] == "metal-emeans" and (
                recipe.get("precision") != "float32" or recipe.get("finalization") != "last_epoch"
                or recipe.get("assignment") != "auto"):
            raise RuntimeError("Metal e-means precision/finalization differs")


# The parent handshake/controller below is retained from large_compare.py so
# that timing and memory phase boundaries match the completed e-means campaign.
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
                    print(f"FIT_START {config['dataset']} {config['mode']} s{config['seed']} call{active_call}", flush=True)
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
                elif message.get("phase") in {"WARMUP_START", "WARMUP_DONE"}:
                    print(f"{message['phase']} {config['dataset']} {config['mode']} s{config['seed']} {message.get('seconds', '')}", flush=True)
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
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rows", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--protocol", type=Path, default=HERE / "kmeans_protocol.json")
    parser.add_argument("--cache", default=str(base.DEFAULT_CACHE))
    parser.add_argument("--faiss-source", default=str(base.REPO / "tmp/e_means/faiss"))
    parser.add_argument("--out", type=Path, default=HERE / "results/kmeans_comparison")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return
    if (args.threads < 1 or any(v is not None and (not math.isfinite(v) or v <= 0) for v in (args.rows, args.k, args.epochs, args.timeout))
            or len(set(args.modes)) != len(args.modes)):
        parser.error("Invalid sizes, timeout, threads or duplicate modes")
    protocol_path = args.protocol.resolve()
    entries = json.loads(protocol_path.read_text())["configurations"]
    if args.datasets and (len(set(args.datasets)) != len(args.datasets)
                          or not set(args.datasets) <= {entry["dataset"] for entry in entries}):
        parser.error("Unknown or duplicate datasets")
    captured_sources = sources(protocol_path)
    captured_reference = previous.reference_sources(args.faiss_source)
    captured_sklearn = sklearn_sources()
    versions, hardware = base.package_versions(), base.host_identity()
    for index, entry in enumerate(entries):
        if args.datasets and entry["dataset"] not in args.datasets:
            continue
        modes = args.modes[index % len(args.modes):] + args.modes[:index % len(args.modes)]
        for mode in modes:
            config = {"dataset": entry["dataset"], "mode": mode, "seed": 10,
                          "epochs": args.epochs if args.epochs is not None else entry["epochs"],
                          "rows": args.rows, "k_override": args.k, "epochs_override": args.epochs,
                          "diagnostic_subset": any(v is not None for v in (args.rows, args.k, args.epochs)) or args.threads != 8,
                          "precision": "float32", "n_init": 1, "tol": 0.0, "final_labels": True, "warmup": True,
                          "threads": args.threads, "train_score_rows": 10000, "memory": True,
                          "memory_interval_seconds": .01, "protocol_path": str(protocol_path),
                          "protocol_config": entry, "cache": str(Path(args.cache).resolve()),
                          "faiss_source": str(Path(args.faiss_source).resolve()),
                          "expected_source_sha256": captured_sources, "expected_reference_source_sha256": captured_reference,
                          "expected_sklearn_sha256": captured_sklearn, "expected_versions": versions, "expected_host": hardware}
            digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
            path = args.out / f"{entry['dataset']}-{mode}-s10-{digest}.json"
            config.update(output=str(path.resolve()), worker_output=str(path.resolve()) + ".worker.tmp")
            if path.exists():
                saved = json.loads(path.read_text())
                validate_result(saved, config, saved.get("memory_samples", []))
                print(f"SKIP {path}", flush=True)
                continue
            measured_run(config, args.timeout)


if __name__ == "__main__":
    main()
