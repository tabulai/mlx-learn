"""o-means versus e-means fits on a machine without the benchmark dataset cache.

Reads ``<prefix>-train.npy`` and ``<prefix>-test.npy`` (float32 matrices
exported from the cached datasets) and needs only MLX and mlxlearn, so it can
check another Apple GPU before ``assignment="auto"`` is enabled for it. For
each precision and epoch budget, ``EMeans(recipe="faiss")`` and ``OMeans``
with the full and the two-level search are warmed up with a complete fit and
then timed in turn, round robin, from resident inputs; the median of three
synchronized ``train_device`` calls is reported. Taking turns keeps the three
in the same machine state, which matters on a loaded or throttling machine.
Each call waits for nominal macOS thermal pressure, and the worst level seen
is printed. Held-out error uses mlxlearn's own full search, not the Faiss
scorer of the main campaign.

    python omeans_device_fits.py /path/to/spacev-1M 1,2,3
"""

from __future__ import annotations

import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

from mlxlearn.cluster import EMeans, OMeans


def thermal_pressure():
    """macOS thermal pressure level, 0 for nominal; ``None`` if it cannot be read."""
    try:
        answer = subprocess.run(
            ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
            capture_output=True, text=True, timeout=5, check=False).stdout.split()
        return int(answer[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def timed_fits(models, x, repeats=3, settle=300.0):
    """Median seconds and last result per model, timed round robin."""
    prepared = {name: model.prepare_device(x) for name, model in models.items()}
    for name, model in models.items():
        model.train_device(prepared[name])
    times, results, worst = {name: [] for name in models}, {}, 0
    for _ in range(repeats):
        for name, model in models.items():
            waited = 0.0
            while thermal_pressure() and waited < settle:
                time.sleep(1.0)
                waited += 1.0
            mx.clear_cache()
            mx.synchronize()
            start = time.perf_counter()
            results[name] = model.train_device(prepared[name])
            mx.synchronize()
            times[name].append(time.perf_counter() - start)
            worst = max(worst, thermal_pressure() or 0)
    return {name: float(np.median(values)) for name, values in times.items()}, results, worst


def held_out_mse(model, result, test):
    _, distances = model._assign(mx.array(test), result.centers)
    return float(np.asarray(distances).sum(dtype=np.float64) / len(test))


def main():
    prefix, budgets = sys.argv[1], [float(value) for value in sys.argv[2].split(",")]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    x, test = mx.array(np.load(prefix + "-train.npy")), np.load(prefix + "-test.npy")
    mx.eval(x)
    n, d = x.shape
    print(f"== {prefix.split('/')[-1]}: n={n} d={d} k={k} on {mx.device_info()['device_name']}", flush=True)
    for precision in ("float32", "bfloat16"):
        for epochs in budgets:
            shared = {"max_iter": epochs, "random_state": 10, "precision": precision,
                      "compute_labels": False}
            models = {"emeans": EMeans(k, recipe="faiss", batch_law="throughput", **shared),
                      "dense": OMeans(k, assignment="dense", **shared),
                      "two_level": OMeans(k, assignment="two_level", **shared)}
            seconds, results, pressure = timed_fits(models, x)
            error = held_out_mse(models["emeans"], results["emeans"], test)
            line = f"{precision:8s} T={epochs:g}: emeans {seconds['emeans']:.3f}s"
            for mode in ("dense", "two_level"):
                result = results[mode]
                fraction = (sum(record["evaluated_distances"] for record in result.history)
                            / (sum(record["rows"] for record in result.history) * k))
                line += (f" | {mode} {seconds[mode]:.3f}s ({seconds['emeans'] / seconds[mode]:.2f}x) "
                         f"mse {100 * (held_out_mse(models[mode], result, test) / error - 1):+.2f}%"
                         f" frac {fraction:.3f}")
            print(line + f" | thermal {pressure}", flush=True)
            del models, results
            mx.clear_cache()


if __name__ == "__main__":
    main()
