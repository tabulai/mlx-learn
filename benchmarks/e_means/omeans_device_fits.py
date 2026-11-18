"""o-means versus e-means fits on a machine without the benchmark dataset cache.

Reads ``<prefix>-train.npy`` and ``<prefix>-test.npy`` (float32 matrices
exported from the cached datasets) and needs only MLX and mlxlearn, so it can
check another Apple GPU before ``assignment="auto"`` is enabled for it. Each
fit is timed as the median of three synchronized ``train_device`` calls after
a full warm-up fit, from resident inputs. Held-out error uses mlxlearn's own
full search, not the Faiss scorer of the main campaign.

    python omeans_device_fits.py /path/to/spacev-1M 1,2,3
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

from mlxlearn.cluster import EMeans, OMeans


def timed_fit(model, x, repeats=3):
    prepared = model.prepare_device(x)
    model.train_device(prepared)
    times, result = [], None
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        result = model.train_device(prepared)
        mx.eval(result.centers)
        mx.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def held_out_mse(model, result, test):
    _, distances = model._assign(mx.array(test), result.centers)
    return float(np.asarray(distances).sum(dtype=np.float64) / len(test))


def main():
    prefix, budgets = sys.argv[1], [float(value) for value in sys.argv[2].split(",")]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    x, test = np.load(prefix + "-train.npy"), np.load(prefix + "-test.npy")
    n, d = x.shape
    print(f"== {prefix.split('/')[-1]}: n={n} d={d} k={k} on {mx.device_info()['device_name']}", flush=True)
    for precision in ("float32", "bfloat16"):
        for epochs in budgets:
            reference = EMeans(k, max_iter=epochs, recipe="faiss", batch_law="throughput",
                               random_state=10, precision=precision, compute_labels=False)
            seconds, fit = timed_fit(reference, x)
            error = held_out_mse(reference, fit, test)
            line = f"{precision:8s} T={epochs}: emeans {seconds:.3f}s"
            for mode in ("dense", "two_level"):
                model = OMeans(k, max_iter=epochs, assignment=mode, random_state=10,
                               precision=precision, compute_labels=False)
                elapsed, result = timed_fit(model, x)
                fraction = (sum(record["evaluated_distances"] for record in result.history)
                            / (sum(record["rows"] for record in result.history) * k))
                line += (f" | {mode} {elapsed:.3f}s ({seconds / elapsed:.2f}x) "
                         f"mse {100 * (held_out_mse(model, result, test) / error - 1):+.2f}%"
                         f" frac {fraction:.3f}")
            print(line, flush=True)


if __name__ == "__main__":
    main()
