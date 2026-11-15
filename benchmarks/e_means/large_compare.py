"""Resident, warmed 10M e-means: unmodified Faiss CPU and Metal.

The primary comparison uses FP32 operands/state in both implementations.
Native Metal BF16 is a separately labelled supplement, never an FP32 pair.
The fit-window controller follows h100.py; historical drivers are unchanged.
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

KIND = "large_faiss_resident"
MODES = ("metal-fp32", "faiss-fp32", "metal-bf16")
PRIMARY_MODES = ("metal-fp32", "faiss-fp32")
IDENTITY_FIELDS = ("train_sample_sha256", "test_sha256", "train_rows_used", "test_rows_used", "dimension")
RECIPE_FIELDS = ("batch_law", "batch_size", "alpha0", "nepochs", "epoch_rows", "epoch_steps", "total_steps",
                 "alpha_min_factor", "init_usage", "split_epsilon", "unused_threshold", "check_unused_every", "tail_policy")


def sources(protocol):
    result = base.source_hashes()
    for path in [Path(__file__), HERE / "memory.py", Path(protocol),
                 *sorted((base.REPO / "src/mlxlearn").rglob("*.py"))]:
        resolved = path.resolve()
        key = str(resolved.relative_to(base.REPO)) if resolved.is_relative_to(base.REPO) else str(resolved)
        result[key] = base.file_sha256(path)
    return result


def reference_sources(root):
    """Require the pinned, unmodified checkout and hash its actual input files."""
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != base.FAISS_REVISION:
        raise RuntimeError("Reference checkout is not the pinned Faiss revision")
    clean = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--",
                            "contrib/e_means", "benchs/e_means"], check=False)
    if clean.returncode:
        raise RuntimeError("Reference e-means checkout has modifications")
    paths = [*sorted((root / "contrib/e_means").glob("*.py")),
             root / "benchs/e_means/paper_data.py", root / "benchs/e_means/paper_download.py",
             *sorted((root / "benchs/e_means/paper_splits").glob("*"))]
    return {str(path.relative_to(root)): base.file_sha256(path) for path in paths if path.is_file()}


def check_environment(config):
    if (sources(config["protocol_path"]) != config["expected_source_sha256"]
        or reference_sources(config["faiss_source"]) != config["expected_reference_source_sha256"]
        or base.package_versions() != config["expected_versions"]
        or base.host_identity() != config["expected_host"]):
        raise RuntimeError("Benchmark/reference source, software, or host changed")


def expected_calls(seconds):
    return 3 if seconds < 2.0 else 1


def event(phase, **values):
    print(json.dumps({"phase": phase, **values}), flush=True)


def validate_dataset(metadata, n, d, config):
    entry = config["protocol_config"]
    if config["dataset"] != entry["dataset"]:
        raise RuntimeError("Dataset name differs from protocol entry")
    if d != entry["d"] or config["rows"] is None and n != entry["n"]:
        raise RuntimeError("Dataset shape differs from the protocol manifest")
    identity = entry.get("dataset_identity")
    if config["rows"] is None:
        if not identity or any(key not in identity or metadata.get(key) != identity[key] for key in IDENTITY_FIELDS):
            raise RuntimeError("Dataset fingerprint or evaluation shape differs from the manifest")


def algorithm_comparison(recipe):
    return {key: recipe[key] for key in RECIPE_FIELDS}


def semantic_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(semantic_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(semantic_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and (isinstance(left, float) or isinstance(right, float)):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)
    return left == right


def worker(config):
    check_environment(config)
    import faiss

    faiss.omp_set_num_threads(config["threads"])
    data, _, reference = base.reference_modules(config["faiss_source"], config["cache"])
    x, test, metadata = base.load_dataset(config["dataset"], data, config["cache"], config["rows"])
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    validate_dataset(metadata, n, d, config)
    k = config["k_override"] or config["protocol_config"]["k"]
    if k > n:
        raise ValueError("More clusters than input rows")
    initial = x[np.random.RandomState(config["seed"]).permutation(n)[:k]].copy()
    initial_hash = base.array_sha256(initial)
    score_ids = np.linspace(0, n - 1, min(n, config["train_score_rows"])).astype(int)
    train_sample = np.array(x[score_ids], dtype=np.float32, copy=True)
    train_sample_hash = base.array_sha256(train_sample)
    is_metal = config["mode"] != "faiss-fp32"
    preparation_start = time.perf_counter()
    mx = None
    metal_device = None
    if is_metal:
        import mlx.core as mx

        from mlxlearn.cluster import EMeans

        kwargs = {"n_clusters": k, "max_iter": config["epochs"], "random_state": config["seed"], "init": initial,
                  "compute_labels": False, "recipe": "faiss", "batch_law": "quality", "precision": config["precision"],
                  "finalization": "last_epoch", "assignment": "auto"}
        preparer = EMeans(**kwargs)
        prepared = preparer.prepare_device(x)
        mx.synchronize()
        metal_device = mx.metal.device_info()
        # Retain the small, explicitly copied quality sample and initialization,
        # but no Python reference to the complete NumPy training matrix.
        del preparer, x
        gc.collect()

        def fit():
            model = EMeans(**kwargs)
            output = model.train_device(prepared, capture_assignments=True)
            mx.synchronize()
            return model, output
    else:
        def fit():
            model = reference.NumpyEmeans(d, k, nepochs=config["epochs"], batch_law="quality",
                                          seed=config["seed"], capture_assignments=True,
                                          max_points_per_centroid=None)
            model.train(x, init_centroids=initial)
            return model, None
    preparation_seconds = time.perf_counter() - preparation_start
    from memory import snapshot

    preparation_process_memory = snapshot(os.getpid())
    event("WARMUP_START", n=n, d=d, k=k)
    warmup_start = time.perf_counter()
    model, output = fit()
    warmup_seconds = time.perf_counter() - warmup_start
    del model, output
    gc.collect()
    if mx is not None:
        mx.synchronize()
    event("WARMUP_DONE", seconds=warmup_seconds)
    samples, allocator_samples = [], []
    first_centers = first_recipe = first_history = first_labels = first_splits = first_steps = None
    for call_index in range(3):
        gc.collect()
        if mx is not None:
            mx.synchronize()
            mx.clear_cache()
            mx.synchronize()
            mx.reset_peak_memory()
        event("FIT_READY", call=call_index, n=n, d=d, k=k)
        if sys.stdin.readline().strip() != "GO":
            raise RuntimeError("Missing parent GO handshake")
        started = time.perf_counter()
        model, output = fit()
        seconds = time.perf_counter() - started
        samples.append(seconds)
        allocator_samples.append(None if mx is None else {
            "peak_active_bytes": mx.get_peak_memory(), "active_after_fit_bytes": mx.get_active_memory(),
            "cached_after_fit_bytes": mx.get_cache_memory()})
        event("FIT_DONE", call=call_index, train_seconds=seconds)
        if sys.stdin.readline().strip() != "STOP_ACK":
            raise RuntimeError("Missing parent STOP_ACK handshake")
        if call_index == 0:
            first_centers = np.array(output.centers if is_metal else model.centroids, dtype=np.float32, copy=True)
            first_recipe = dict(model.recipe_ if is_metal else model.recipe)
            first_history = list(output.history if is_metal else model.iteration_stats)
            first_splits = int(output.split_count.item()) if is_metal else int(model.n_splits)
            first_steps = int(output.steps) if is_metal else int(model._step)
            labels = output.assignments if is_metal else model.assignments
            if labels is None:
                raise RuntimeError("Fit omitted requested final-pass assignments")
            first_labels = {"shape": list(labels.shape), "dtype": str(labels.dtype),
                            "semantics": "online final-pass labels; no extra reassignment"}
            del labels
        del model, output
        if len(samples) >= expected_calls(samples[0]):
            break
    test_mse = base.mse(first_centers, test)
    train_mse = base.mse(first_centers, train_sample) if len(train_sample) else None
    check_environment(config)
    result = {
        "status": "completed", "kind": KIND, "config": config, "dataset": metadata,
        "n": n, "d": d, "k": k, "train_seconds": median(samples), "train_seconds_samples": samples,
        "quality_fit_index": 0, "test_mse": test_mse, "train_sample_mse": train_mse,
        "train_score_rows": len(train_sample), "train_score_sample_sha256": train_sample_hash,
        "initial_centroids_sha256": initial_hash, "final_centroids_sha256": base.array_sha256(first_centers),
        "recipe_resolved": first_recipe, "algorithm_comparison": algorithm_comparison(first_recipe),
        "iteration_history": first_history, "captured_assignments": first_labels, "n_splits": first_splits,
        "n_steps": first_steps,
        "mlx_memory_samples": allocator_samples, "preparation_seconds": preparation_seconds,
        "warmup_seconds": warmup_seconds, "source_sha256": config["expected_source_sha256"],
        "preparation_process_memory": preparation_process_memory,
        "reference_source_sha256": config["expected_reference_source_sha256"],
        "faiss_reference_revision": base.FAISS_REVISION, "versions": base.package_versions(),
        "hardware": base.host_identity(), "metal_device": metal_device,
        "faiss_compile_options": faiss.get_compile_options(), "effective_omp_threads": faiss.omp_get_max_threads(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")},
        "reference_backend": "unmodified pinned NumpyEmeans; Faiss CPU IndexFlat/Accelerate/OpenMP",
        "full_host_input_released": is_metal,
        "memory_policy": "Metal releases the full NumPy input reference after device preparation; CPU retains its contiguous input; both retain explicit initialization and copied scoring sample; underlying storage ownership is MLX/NumPy managed",
        "timing_scope": "resident FP32 input and explicit initialization; full untimed warmup; timed new model plus complete training/finalization and synchronization, including permutations/history/online int64 labels; preparation, cache clearing, output export and scoring excluded",
        "backend_timing_notes": "Metal full validation/upload/recipe preparation outside timer, per-fit signature checks inside; unmodified CPU train performs its no-copy contiguous conversion and recipe/state initialization inside timer",
        "scoring": "shared Faiss IndexFlatL2 float32 nearest assignment followed by direct float64 squared residuals",
        "memory_scope": "parent-sampled process physical footprint/RSS during each timed fit including resident-input baseline; MLX active allocator separately; excludes warmup/cache clearing/export/scoring",
        "numerical_differences": ["Metal FP32 assignment uses global-mean centering; original Faiss and supplemental BF16 use original coordinates",
                                  "parallel statistics/reduction order and algebraically equivalent EMA arithmetic differ",
                                  "Metal online objective uses FP32; original CPU objective and epoch counts use FP64",
                                  "donor sampling arithmetic/consumption may differ; identical seed is not bitwise trajectory parity"],
    }
    base.atomic_json(config["worker_output"], result)
    event("SCORED", test_mse=test_mse)


def validate_result(result, config, memories):
    if (result.get("status") != "completed" or result.get("kind") != KIND or result.get("config") != config
        or result.get("source_sha256") != config["expected_source_sha256"]
        or result.get("reference_source_sha256") != config["expected_reference_source_sha256"]
        or result.get("faiss_reference_revision") != base.FAISS_REVISION
        or result.get("versions") != config["expected_versions"] or result.get("hardware") != config["expected_host"]):
        raise RuntimeError("Worker provenance differs from the requested measurement")
    mode = config["mode"]
    if mode not in MODES or config["precision"] != ("bfloat16" if mode == "metal-bf16" else "float32"):
        raise RuntimeError("Invalid mode/precision association")
    required = {"recipe": "faiss", "batch_law": "quality", "finalization": "last_epoch",
                "warmup": True, "capture_assignments": True}
    if any(config.get(key) != value for key, value in required.items()):
        raise RuntimeError("Configuration does not use the complete warmed quality-law protocol")
    entry = config["protocol_config"]
    if not config.get("diagnostic_subset") and (config["epochs"] != entry["epochs"]
                                               or config["seed"] not in entry["seed_ids"]
                                               or any(config.get(key) is not None for key in ("rows", "k_override", "epochs_override"))):
        raise RuntimeError("Full campaign budget, seed or override differs from manifest")
    samples = result.get("train_seconds_samples", [])
    if (not samples or any(not math.isfinite(s) or s <= 0 for s in samples)
        or len(samples) != expected_calls(samples[0]) or len(memories) != len(samples)
        or not math.isclose(result.get("train_seconds", math.nan), median(samples), rel_tol=1e-12)
        or not math.isfinite(result.get("test_mse", math.nan)) or result["test_mse"] < 0
        or result.get("quality_fit_index") != 0):
        raise RuntimeError("Invalid adaptive timed samples or first-fit quality")
    if config["memory"] and any(not m or not m.get("complete") for m in memories):
        raise RuntimeError("Incomplete fit-window memory sampling")
    validate_dataset(result.get("dataset", {}), result["n"], result["d"], config)
    if result["k"] != (config["k_override"] or config["protocol_config"]["k"]):
        raise RuntimeError("Cluster count differs from requested workload")
    recipe = result["recipe_resolved"]
    if (not semantic_equal(result.get("algorithm_comparison"), algorithm_comparison(recipe)) or recipe["nepochs"] != config["epochs"]
        or recipe["batch_law"] != "quality" or recipe["seed"] != config["seed"]):
        raise RuntimeError("Applied recipe differs from requested workload")
    if not config.get("diagnostic_subset"):
        for key, expected in config["protocol_config"].get("recipe_expected", {}).items():
            if not semantic_equal(recipe.get(key), expected):
                raise RuntimeError(f"Applied recipe {key} differs from frozen protocol")
    if mode != "faiss-fp32" and (recipe.get("precision") != config["precision"]
                                 or recipe.get("finalization") != "last_epoch"):
        raise RuntimeError("Metal applied precision/finalization differs")
    history = result.get("iteration_history", [])
    if len(history) != len(recipe["epoch_rows"]) or result.get("n_steps") != recipe["total_steps"]:
        raise RuntimeError("Observed pass/update count differs from recipe")
    if mode != "faiss-fp32" and ([stat.get("rows") for stat in history] != recipe["epoch_rows"]
                                 or history[-1].get("steps") != recipe["total_steps"]):
        raise RuntimeError("Observed Metal history differs from recipe")
    labels = result.get("captured_assignments", {})
    if (labels.get("shape") != [result["n"]] or "int64" not in labels.get("dtype", "")
        or not result.get("iteration_history") or not result.get("initial_centroids_sha256")
        or not result.get("final_centroids_sha256") or not result.get("train_score_sample_sha256")):
        raise RuntimeError("Missing final-pass labels, history or input/output hashes")
    if result.get("full_host_input_released") is not (mode != "faiss-fp32"):
        raise RuntimeError("Input memory policy was not applied")
    if (result.get("effective_omp_threads") != config["threads"]
        or result.get("thread_environment", {}).get("OMP_NUM_THREADS") != str(config["threads"])
        or result.get("thread_environment", {}).get("VECLIB_MAXIMUM_THREADS") is not None):
        raise RuntimeError("Thread configuration differs from requested unrestricted-Accelerate setting")


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


def pair_results(records):
    """Only FP32 Metal/CPU records enter the primary, precision-matched ratio."""
    groups, supplements = {}, []
    for row in records:
        config = row["config"]
        validate_result(row, config, row.get("memory_samples", []))
        if config["mode"] == "metal-bf16":
            supplements.append(row)
            continue
        common = {key: value for key, value in config.items() if key not in {"mode", "output", "worker_output"}}
        identity = {"config": common, **{key: row.get(key) for key in
                    ("n", "d", "k", "dataset", "initial_centroids_sha256", "train_score_sample_sha256", "versions",
                     "hardware", "source_sha256", "reference_source_sha256",
                     "faiss_compile_options", "effective_omp_threads", "thread_environment", "timing_scope", "scoring", "memory_scope", "memory_policy")}}
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(key, {})
        if config["mode"] in group:
            raise RuntimeError("Duplicate primary mode for the same comparison")
        group[config["mode"]] = row
    pairs, unpaired = [], []
    for group in groups.values():
        if set(group) != set(PRIMARY_MODES):
            unpaired.extend(group.values())
            continue
        metal, faiss = group["metal-fp32"], group["faiss-fp32"]
        if not semantic_equal(metal["algorithm_comparison"], faiss["algorithm_comparison"]):
            unpaired.extend((metal, faiss))
            continue
        pairs.append({"metal": metal, "faiss": faiss, "speedup": faiss["train_seconds"] / metal["train_seconds"],
                      "quality_delta_pct": 100 * (metal["test_mse"] / faiss["test_mse"] - 1) if faiss["test_mse"] else None})
    return pairs, unpaired + supplements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rows", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--train-score-rows", type=int, default=10000)
    parser.add_argument("--memory-interval", type=float, default=.01)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--timeout", type=float, default=8 * 3600)
    parser.add_argument("--protocol", type=Path, default=HERE / "large_protocol.json")
    parser.add_argument("--cache", default=str(base.DEFAULT_CACHE))
    parser.add_argument("--faiss-source", default=str(base.REPO / "tmp/e_means/faiss"))
    parser.add_argument("--out", type=Path, default=HERE / "results/large_comparison")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker))
        return
    if (args.threads < 1 or args.train_score_rows < 0
        or any(v is not None and (not math.isfinite(v) or v <= 0) for v in (args.rows, args.k, args.epochs, args.memory_interval, args.timeout))
        or len(set(args.modes)) != len(args.modes)
        or args.seeds is not None and (any(not 0 <= seed < 2**32 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds))):
        parser.error("Invalid sizes, timing, threads, seeds or duplicate modes")
    protocol_path = args.protocol.resolve()
    entries = json.loads(protocol_path.read_text())["configurations"]
    if args.datasets and (len(set(args.datasets)) != len(args.datasets)
                          or not set(args.datasets) <= {entry["dataset"] for entry in entries}):
        parser.error("Unknown or duplicate dataset")
    captured_sources = sources(protocol_path)
    captured_reference = reference_sources(args.faiss_source)
    versions, hardware = base.package_versions(), base.host_identity()
    for index, entry in enumerate(entries):
        if args.datasets and entry["dataset"] not in args.datasets:
            continue
        for seed in args.seeds if args.seeds is not None else entry["seed_ids"]:
            modes = args.modes[index % len(args.modes):] + args.modes[:index % len(args.modes)]
            if seed % 2:
                modes = modes[::-1]
            for mode in modes:
                config = {"dataset": entry["dataset"], "mode": mode, "seed": seed,
                          "epochs": args.epochs if args.epochs is not None else entry["epochs"],
                          "epochs_override": args.epochs, "rows": args.rows, "k_override": args.k,
                          "diagnostic_subset": any(v is not None for v in (args.rows, args.k, args.epochs)) or seed not in entry["seed_ids"],
                          "precision": "bfloat16" if mode == "metal-bf16" else "float32",
                          "recipe": "faiss", "batch_law": "quality", "finalization": "last_epoch",
                          "assignment": "auto",
                          "capture_assignments": True, "warmup": True, "threads": args.threads,
                          "train_score_rows": args.train_score_rows, "memory": not args.no_memory,
                          "memory_interval_seconds": args.memory_interval, "protocol_path": str(protocol_path),
                          "protocol_config": entry, "cache": str(Path(args.cache).resolve()),
                          "faiss_source": str(Path(args.faiss_source).resolve()), "expected_source_sha256": captured_sources,
                          "expected_reference_source_sha256": captured_reference,
                          "expected_versions": versions, "expected_host": hardware}
                digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
                path = args.out / f"{entry['dataset']}-{mode}-s{seed}-{digest}.json"
                config.update(output=str(path.resolve()), worker_output=str(path.resolve()) + ".worker.tmp")
                if path.exists() and not args.force:
                    saved = json.loads(path.read_text())
                    validate_result(saved, config, saved.get("memory_samples", []))
                    print(f"SKIP {path}", flush=True)
                    continue
                measured_run(config, args.timeout)


if __name__ == "__main__":
    main()
