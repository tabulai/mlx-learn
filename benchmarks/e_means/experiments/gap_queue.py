"""Ablate periodic synchronization during resident BF16 e-means training.

This diagnostic tests queue-lifetime effects without editing production code.
N=0 passes every batch to async_eval; N>0 replaces every Nth batch submission
with eval on the same pending arrays. Counters reset for every complete fit.
Run GPU measurements only with exclusive benchmark ownership.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
sys.path.insert(0, str(BENCHMARKS))
import h100  # noqa: E402

base = h100.base


@contextlib.contextmanager
def async_eval_policy(mx, period):
    """Replace every period-th async submission, restoring the API on exit."""
    if isinstance(period, bool) or not isinstance(period, int) or period < 0:
        raise ValueError("period must be a nonnegative integer")
    original = mx.async_eval
    counters = {"batch_submissions": 0, "synchronous_replacements": 0}

    def submit(*pending, **kwargs):
        counters["batch_submissions"] += 1
        if period and counters["batch_submissions"] % period == 0:
            counters["synchronous_replacements"] += 1
            return mx.eval(*pending, **kwargs)
        return original(*pending, **kwargs)

    mx.async_eval = submit
    try:
        yield counters
    finally:
        mx.async_eval = original


def history_without_times(history):
    """Retain every history diagnostic except elapsed wall-clock fields."""
    return [
        {key: value for key, value in epoch.items()
         if key not in {"seconds", "cumulative_seconds"}}
        for epoch in history
    ]


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def self_test():
    """CPU-only checks of queue scheduling, per-fit reset, and restoration."""
    class FakeMLX:
        def __init__(self):
            self.calls = []

        def async_eval(self, *args, **kwargs):
            self.calls.append(("async", args, kwargs))

        def eval(self, *args, **kwargs):
            self.calls.append(("sync", args, kwargs))

    fake = FakeMLX()
    original = fake.async_eval
    for period in (0, 2, 4, 8):
        for _fit in range(2):
            fake.calls.clear()
            with async_eval_policy(fake, period) as counters:
                for batch in range(9):
                    fake.async_eval(batch, marker="retained")
            assert fake.async_eval == original
            assert counters["batch_submissions"] == 9
            assert counters["synchronous_replacements"] == (9 // period if period else 0)
            assert [kind for kind, _, _ in fake.calls] == [
                "sync" if period and (batch + 1) % period == 0 else "async"
                for batch in range(9)
            ]
            assert all(kwargs == {"marker": "retained"} for _, _, kwargs in fake.calls)
    try:
        with async_eval_policy(fake, 2):
            raise RuntimeError("intentional restoration probe")
    except RuntimeError:
        pass
    assert fake.async_eval == original
    assert history_without_times([{"epoch": 1, "seconds": 9, "cumulative_seconds": 9,
                                  "moving_mse": 0.5, "splits": 4}]) == [
                                      {"epoch": 1, "moving_mse": 0.5, "splits": 4}]
    print("CPU policy checks passed: 8 fit resets, periodic replacement, exception restoration, history filtering.")


def run(args):
    import faiss
    import mlx.core as mx
    import numpy as np

    from mlxlearn.cluster import EMeans

    faiss.omp_set_num_threads(args.threads)
    protocol_path = BENCHMARKS / "h100_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    entries = {entry["dataset"]: entry for entry in protocol["configurations"]}
    if any(name not in entries for name in args.datasets):
        raise ValueError("Dataset is absent from the audited protocol")
    data, _, _ = base.reference_modules(args.faiss_source, args.cache)
    source_hashes = h100.sources(protocol_path)
    source_hashes[str(Path(__file__).relative_to(base.REPO))] = base.file_sha256(Path(__file__))
    script = Path(__file__).read_text()
    method_source = inspect.getsource(async_eval_policy)
    train_source = inspect.getsource(EMeans.train_device)
    result = {
        "kind": "metal_async_queue_ablation",
        "status": "running",
        "script": script,
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "method_source": method_source,
        "method_source_sha256": hashlib.sha256(method_source.encode()).hexdigest(),
        "train_device_source": train_source,
        "train_device_source_sha256": hashlib.sha256(train_source.encode()).hexdigest(),
        "source_sha256": source_hashes,
        "versions": base.package_versions(),
        "hardware": base.host_identity(),
        "metal_device": mx.device_info(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "protocol": {
            "input": "resident FP32, same explicit initial centers, Faiss throughput recipe, BF16 assignment/statistics, last_epoch, online final-pass labels",
            "seed": args.seed,
            "periods": args.periods,
            "warmup": "one full fit per period; discarded outputs; all methods warm before timed interleaved rounds",
            "samples_per_period": args.repeats,
            "timing": "new estimator, wrapper installation/restoration, train_device including periodic eval, and final synchronize; excludes cache clearing, preparation and NumPy output export",
            "memory": "MLX active allocation peak reset after gc/sync/cache clear before each fit; includes resident prepared input baseline; does not measure process physical footprint",
            "intervention": "replace each Nth mx.async_eval call with mx.eval on identical pending arrays; counter starts at zero for every fit; N=0 uses a counting passthrough wrapper",
            "history": "every history field retained except seconds and cumulative_seconds; hashes are diagnostics, not an assumption of bitwise determinism",
        },
        "records": [],
    }

    def save():
        base.atomic_json(args.output, result)

    def check_sources():
        current = h100.sources(protocol_path)
        current[str(Path(__file__).relative_to(base.REPO))] = base.file_sha256(Path(__file__))
        if current != source_hashes:
            raise RuntimeError("Measured runtime or experiment source changed during the ablation")

    save()
    for name in args.datasets:
        entry = entries[name]
        x, test, metadata = base.load_dataset(name, data, args.cache)
        n, d = x.shape
        h100.validate_dataset(metadata, n, d, {"rows": None, "protocol_config": entry})
        k = entry["k"]
        initial = x[np.random.RandomState(args.seed).permutation(n)[:k]].copy()
        kwargs = {"n_clusters": k, "max_iter": entry["epochs"], "random_state": args.seed,
                  "init": initial, "compute_labels": False, "recipe": "faiss",
                  "batch_law": "throughput", "precision": "bfloat16",
                  "assignment": "auto", "finalization": "last_epoch"}
        prepared = EMeans(**kwargs).prepare_device(x)
        mx.synchronize()
        dataset_record = {
            "dataset": name, "n": n, "d": d, "k": k,
            "epochs": entry["epochs"], "dataset_metadata": metadata,
            "initial_centroids_sha256": base.array_sha256(initial),
            "recipe_resolved": prepared.recipe,
            "warmups": [], "samples": [], "execution_order": [],
        }
        result["records"].append(dataset_record)

        def fit(period, kwargs=kwargs, prepared=prepared):
            check_sources()
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            mx.synchronize()
            mx.reset_peak_memory()
            baseline = {"active_bytes": mx.get_active_memory(), "cached_bytes": mx.get_cache_memory()}
            started = time.perf_counter()
            with async_eval_policy(mx, period) as counters:
                model = EMeans(**kwargs)
                output = model.train_device(prepared, capture_assignments=True)
                mx.synchronize()
            elapsed = time.perf_counter() - started
            allocation = {"peak_active_bytes": mx.get_peak_memory(),
                          "active_after_fit_bytes": mx.get_active_memory(),
                          "cached_after_fit_bytes": mx.get_cache_memory(),
                          "baseline_active_bytes": baseline["active_bytes"],
                          "baseline_cached_bytes": baseline["cached_bytes"]}
            allocation["peak_active_increment_bytes"] = allocation["peak_active_bytes"] - baseline["active_bytes"]
            if counters["batch_submissions"] != output.steps:
                raise RuntimeError("async_eval submissions no longer correspond one-to-one with statistical batches")
            if output.assignments is None:
                raise RuntimeError("Fit omitted requested online assignments")
            centers = np.array(output.centers, dtype=np.float32, copy=True)
            history = history_without_times(output.history)
            history_json = canonical_json(history)
            sample = {
                "period": period, "seconds": elapsed, "mlx_memory": allocation,
                "batch_submissions": counters["batch_submissions"],
                "synchronous_replacements": counters["synchronous_replacements"],
                "centroids_sha256": base.array_sha256(centers),
                "history_without_times": history,
                "history_without_times_rendered": json.dumps(history, sort_keys=True, indent=2, allow_nan=False),
                "history_without_times_sha256": hashlib.sha256(history_json.encode()).hexdigest(),
                "splits": int(output.split_count.item()),
                "steps": output.steps, "epochs": output.epochs,
                "captured_assignments_shape": list(output.assignments.shape),
            }
            del centers, model, output
            check_sources()
            return sample

        for period in args.periods:
            sample = fit(period)
            dataset_record["warmups"].append(sample)
            print(json.dumps({"phase": "warmup", "dataset": name, "period": period,
                              "seconds": sample["seconds"], "mlx_memory": sample["mlx_memory"],
                              "centroids_sha256": sample["centroids_sha256"]}), flush=True)
            save()
        for repetition in range(args.repeats):
            # Alternate direction and rotate the first method to reduce fixed
            # ordering effects without changing the data/algorithm seed.
            offset = repetition % len(args.periods)
            order = args.periods[offset:] + args.periods[:offset]
            if repetition % 2:
                order = list(reversed(order))
            dataset_record["execution_order"].append(order)
            for period in order:
                sample = fit(period)
                sample["repetition"] = repetition
                dataset_record["samples"].append(sample)
                print(json.dumps({"phase": "timed_fit", "dataset": name, "period": period,
                                  "repetition": repetition, "seconds": sample["seconds"],
                                  "mlx_memory": sample["mlx_memory"],
                                  "centroids_sha256": sample["centroids_sha256"]}), flush=True)
                save()
        dataset_record["summary"] = []
        for period in args.periods:
            samples = [sample for sample in dataset_record["samples"] if sample["period"] == period]
            dataset_record["summary"].append({
                "period": period,
                "median_seconds": median(sample["seconds"] for sample in samples),
                "median_peak_active_bytes": median(sample["mlx_memory"]["peak_active_bytes"] for sample in samples),
                "median_peak_increment_bytes": median(sample["mlx_memory"]["peak_active_increment_bytes"] for sample in samples),
                "distinct_centroid_hashes": sorted({sample["centroids_sha256"] for sample in samples}),
                "distinct_history_hashes": sorted({sample["history_without_times_sha256"] for sample in samples}),
            })
        save()
        del fit, prepared, x, test, initial, kwargs
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    check_sources()
    result["status"] = "completed"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="CPU-only scheduling checks; no MLX import")
    parser.add_argument("--datasets", nargs="+", default=["spacev-1M", "vibe-agnews"])
    parser.add_argument("--periods", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--faiss-source", type=Path, default=base.REPO / "tmp/e_means/faiss")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error("--output is required for GPU experiments")
    if args.output.exists():
        parser.error("--output already exists; preserve prior evidence and choose a new path")
    if args.repeats < 1 or args.threads < 1 or not 0 <= args.seed < 2**32:
        parser.error("positive repeats/threads and a uint32 seed are required")
    if any(period < 0 for period in args.periods) or len(set(args.periods)) != len(args.periods):
        parser.error("periods must be distinct nonnegative integers")
    run(args)


if __name__ == "__main__":
    main()
