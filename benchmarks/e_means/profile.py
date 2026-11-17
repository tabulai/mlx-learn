"""Stage-level synchronized GPU profiling on real e-means datasets.

Instrumented calls force synchronization and therefore measure isolated stage
cost, not an additive hardware trace of the normal asynchronous execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402


def evaluate(value):
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            evaluate(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["deep1b-1M", "dino10b-1M-div4096"])
    parser.add_argument("--out", type=Path, default=HERE / "results/optimization/profile-matched.json")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--assignment", choices=["gemm", "auto", "fused"], default="auto")
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()
    module = importlib.import_module("mlxlearn.cluster._e_means")
    data, _, _ = base.reference_modules(base.REPO / "tmp/e_means/faiss", base.DEFAULT_CACHE)
    records = []
    source = base.source_hashes()
    kernel = base.REPO / "src/mlxlearn/_kernels/e_means_bf16.py"
    if kernel.exists():
        source[str(kernel.relative_to(base.REPO))] = base.file_sha256(kernel)
    for name in args.datasets:
        x, _, metadata = base.load_dataset(name, data, base.DEFAULT_CACHE)
        n, d = x.shape
        k = 16384
        initial = x[np.random.RandomState(10).permutation(n)[:k]].copy()
        device, centers = mx.array(x), mx.array(initial)
        mx.eval(device, centers)

        parameters = {"n_clusters": k, "max_iter": args.epochs, "recipe": "faiss", "init": centers,
                      "compute_labels": False, "random_state": 10, "precision": args.precision,
                      "assignment": args.assignment, "batch_law": "throughput"}
        prepared = module.EMeans(**parameters).prepare_device(device)

        def fit(parameters=parameters, prepared=prepared):
            return module.EMeans(**parameters).train_device(prepared, capture_assignments=True)

        fit()
        mx.synchronize()
        started = time.perf_counter()
        result = fit()
        mx.synchronize()
        normal = time.perf_counter() - started
        totals = defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        originals = {}

        def wrap(label, function, totals=totals):
            def measured(*a, **kw):
                mx.synchronize()
                before = time.perf_counter()
                result = function(*a, **kw)
                evaluate(result)
                mx.synchronize()
                totals[label]["seconds"] += time.perf_counter() - before
                totals[label]["calls"] += 1
                return result
            return measured

        for label in ("assign", "batch_stats", "ema_update", "revive"):
            originals[label] = getattr(module, label)
            setattr(module, label, wrap(label, originals[label]))
        started = time.perf_counter()
        try:
            fit()
            mx.synchronize()
        finally:
            for label, function in originals.items():
                setattr(module, label, function)
        instrumented = time.perf_counter() - started
        record = {"dataset": name, "n": n, "d": d, "k": k,
                  "normal_warmed_resident_fit_seconds": normal,
                  "instrumented_fit_seconds": instrumented, "stage_totals": dict(totals),
                  "other_seconds": instrumented - sum(v["seconds"] for v in totals.values()),
                  "recipe": prepared.recipe, "steps": result.steps, "data": metadata}
        records.append(record)
        print(json.dumps(record), flush=True)
    after = base.source_hashes()
    if kernel.exists():
        after[str(kernel.relative_to(base.REPO))] = base.file_sha256(kernel)
    if after != source:
        raise RuntimeError("Implementation changed during profiling")
    payload = {"kind": "stage_profile", "status": "completed", "source_sha256": source,
               "profile_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "hardware": base.host_identity(), "versions": base.package_versions(),
               "protocol": "GPU-resident float32 input, Faiss throughput recipe, last_epoch, seed10, online labels; normal one full warmup and one timed fit; isolated instrumented stages serialize execution; diagnostic, not the full adaptive-call benchmark",
               "precision": args.precision, "assignment": args.assignment, "epochs": args.epochs,
               "records": records}
    base.atomic_json(args.out, payload)


if __name__ == "__main__":
    main()
