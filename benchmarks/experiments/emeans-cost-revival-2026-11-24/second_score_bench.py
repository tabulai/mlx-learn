"""Cost of the second-nearest score in the dense assignment kernels.

Times ``assign`` with and without ``second=True`` on synthetic mixtures, the
two calls taking turns so that both see the same machine state, and checks the
margins against a GEMM + ``select_topk`` reference on a slice. Also times the
two statistics kernels of cost-based revival next to ``batch_stats`` and
``ema_update``.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/second_score_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.e_means.omeans_compare import settle, thermal_pressure, worst_pressure  # noqa: E402

from mlxlearn._kernels.e_means import (  # noqa: E402
    assign,
    batch_stats,
    ema_update,
    removal_stats,
    removal_update,
)
from mlxlearn_device._binary32 import matmul  # noqa: E402
from mlxlearn_device._neighbors_fused import select_topk  # noqa: E402

SHAPES = [(32768, 16384, 100), (32768, 16384, 96), (32768, 16384, 256), (32768, 16384, 512),
          (65536, 65536, 100), (32768, 1024, 100)]


def once(function):
    start = time.perf_counter()
    outputs = function()
    mx.eval(*outputs)
    mx.synchronize()
    return (time.perf_counter() - start) * 1e3


PRESSURES = []


def paired(first, second, rounds=6, repeats=5):
    """Median milliseconds of two functions measured in alternating blocks.

    Every block starts at nominal thermal pressure where the machine reaches
    it within a minute; the worst level seen is reported at the end.
    """
    a, b = [], []
    for function in (first, second):
        once(function)
        once(function)
    for index in range(rounds):
        pair = ((first, a), (second, b)) if index % 2 == 0 else ((second, b), (first, a))
        for function, times in pair:
            _, before = settle(60.0)
            times += [once(function) for _ in range(repeats)]
            PRESSURES.append(worst_pressure(before, thermal_pressure()))
    return float(np.median(a)), float(np.median(b))


def measure(rows, k, d, rng):
    modes = max(64, k // 4)
    means = rng.standard_normal((modes, d), dtype=np.float32)
    x = mx.array(means[rng.integers(0, modes, rows)]
                 + 0.5 * rng.standard_normal((rows, d), dtype=np.float32))
    centers = mx.array(means[rng.integers(0, modes, k)]
                       + 0.5 * rng.standard_normal((k, d), dtype=np.float32))
    mx.eval(x, centers)
    for precision in ("float32", "bfloat16"):
        check = min(rows, 4096)
        labels, _, margins = assign(x[:check], centers, precision=precision, second=True)
        dtype = mx.bfloat16 if precision == "bfloat16" else mx.float32
        xr = x[:check].astype(dtype).astype(mx.float32)
        cr = centers.astype(dtype).astype(mx.float32)
        scores = mx.sum(cr * cr, axis=1) - 2.0 * matmul(xr, cr, transpose_b=True)
        values, ids = select_topk(scores, 2)
        agree = float(mx.mean((labels == ids[:, 0].astype(mx.uint32)).astype(mx.float32)).item())
        scale = float(mx.max(mx.abs(values)).item())
        error = float(mx.max(mx.abs(margins - (values[:, 1] - values[:, 0]))).item()) / scale
        plain, both = paired(
            lambda precision=precision: assign(x, centers, precision=precision),
            lambda precision=precision: assign(x, centers, precision=precision, second=True))
        print(f"rows={rows} k={k} d={d} {precision}: assign {plain:.3f} ms, with margin "
              f"{both:.3f} ms ({100 * (both / plain - 1):+.1f}%) | labels equal {agree:.4f}, "
              f"relative margin error {error:.1e}", flush=True)
    labels, _, margins = assign(x, centers, second=True)
    axes = centers / mx.sqrt(mx.sum(centers * centers, axis=1, keepdims=True))
    long_axis = mx.array(rng.standard_normal((k, d), dtype=np.float32))
    removal = mx.zeros((k,), mx.float32)
    sums, counts = batch_stats(x, labels, k)
    axis_sums, costs = removal_stats(x, labels, margins, centers, axes)
    mx.eval(labels, margins, axes, long_axis, sums, counts, axis_sums, costs)
    stats, extra = paired(lambda: batch_stats(x, labels, k),
                          lambda: removal_stats(x, labels, margins, centers, axes))
    ema, update = paired(lambda: ema_update(centers, counts + 1, sums, counts, 0.07),
                         lambda: removal_update(long_axis, removal, axis_sums, costs, axes, 0.07))
    print(f"rows={rows} k={k} d={d}: batch_stats {stats:.3f} ms, removal_stats {extra:.3f} ms, "
          f"ema_update {ema:.3f} ms, removal_update {update:.3f} ms", flush=True)


def main():
    shapes = [tuple(int(v) for v in arg.split(",")) for arg in sys.argv[1:]] or SHAPES
    rng = np.random.default_rng(0)
    print(f"device {mx.device_info()['device_name']}, MLX {mx.__version__}")
    with mx.stream(mx.gpu):
        for rows, k, d in shapes:
            measure(rows, k, d, rng)
    print(f"worst thermal pressure level during timed blocks: {worst_pressure(*PRESSURES)}")


if __name__ == "__main__":
    main()
