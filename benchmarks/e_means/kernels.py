"""Synchronized Metal assignment tuning; synthetic inputs, not dataset results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxlearn._kernels.e_means import assign, batch_stats, ema_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results/kernels.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    records = []
    mx.random.seed(123)
    order = random.Random(123)
    for d in [96, 128, 256, 512, 1024]:
        n, k = 16384, 16384
        x = mx.random.normal((n, d))
        centers = mx.random.normal((k, d))
        mx.eval(x, centers)
        options = [("gemm", "float32", b) for b in [1024, 2048, 4096, 8192, 16384]]
        options += [("fused", "float32", 8192), ("fused", "float16", 8192)]
        order.shuffle(options)
        oracle, _ = assign(x, centers, method="gemm", batch_size=8192)
        oracle = np.array(oracle)
        for method, precision, block in options:
            for _ in range(2):
                labels, distances = assign(x, centers, method=method, precision=precision, batch_size=block)
                mx.eval(labels, distances)
            samples = []
            for _ in range(args.repeats):
                mx.synchronize()
                start = time.perf_counter()
                labels, distances = assign(x, centers, method=method, precision=precision, batch_size=block)
                mx.eval(labels, distances)
                samples.append(time.perf_counter() - start)
            record = {
                "n": n, "k": k, "d": d, "method": method, "precision": precision,
                "rows_per_tile": block, "seconds": statistics.median(samples),
                "samples": samples, "agreement_with_float32_gemm": float(np.mean(oracle == np.array(labels))),
            }
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "samples"}), flush=True)
        samples = []
        counts = mx.ones((k,), dtype=mx.float32)
        for i in range(args.repeats + 2):
            start = time.perf_counter()
            sums, hits = batch_stats(x, labels, k)
            updated, mass = ema_update(centers, counts, sums, hits, 0.1)
            mx.eval(updated, mass)
            if i >= 2:
                samples.append(time.perf_counter() - start)
        records.append({"n": n, "k": k, "d": d, "method": "statistics_and_ema",
                        "seconds": statistics.median(samples), "samples": samples})
    import mlxlearn._kernels.e_means as kernels

    payload = {
        "workload": "synthetic standard-normal float32; kernel tuning only",
        "timing": "2 warmups then synchronized median; randomized configuration order",
        "hardware": mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info(),
        "platform": platform.platform(), "mlx": mx.__version__,
        "kernel_sha256": hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        "results": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
