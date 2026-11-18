"""Kernel-level comparison of o-means' two-level assignment with dense assignment.

One batch of training rows is assigned against centroids taken from a short
e-means fit, so the cells have the size distribution of a real fit. Each
stage is timed with its own synchronization, which adds one dispatch per
stage; the whole assignment is also timed as one graph.

Example::

    .venv/bin/python -m benchmarks.e_means.two_level_bench --dataset spacev-1M \
        --precision float32 bfloat16 --output two-level.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

try:
    from . import run as base
except ImportError:  # executed as a script
    import run as base

from mlxlearn._kernels.e_means import assign
from mlxlearn._kernels.o_means import (
    bucket_pairs,
    build_cells,
    probe_cells,
    reduce_links,
    scalar_scan,
    sorted_mirror,
    tile_shape,
    tiled_scan,
    two_level_assign,
)
from mlxlearn.cluster import EMeans
from mlxlearn.cluster._o_schedule import num_cells, num_probes


def timed(function, repeats):
    """Median and minimum milliseconds of synchronized evaluations."""
    def once():
        start = time.perf_counter()
        outputs = function()
        mx.eval(*outputs)
        mx.synchronize()
        return time.perf_counter() - start

    once()
    times = [once() for _ in range(repeats)]
    return float(np.median(times)) * 1e3, float(np.min(times)) * 1e3


def synthetic(rows, d, modes, seed):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((modes, d), dtype=np.float32)
    x = means[rng.integers(0, modes, rows)] + 0.3 * rng.standard_normal((rows, d), dtype=np.float32)
    return x.astype(np.float32)


def measure(x, args, precision, tile):
    n, d = x.shape
    k = args.k
    model = EMeans(k, max_iter=args.fit_epochs, recipe="faiss", batch_law="throughput",
                   random_state=args.seed, compute_labels=False).fit(x)
    centers, offset = mx.array(model.cluster_centers_), mx.array(model.offset_)
    bf16 = precision == "bfloat16"
    cell_count = args.cells or num_cells(k)
    n_probe = args.probes or num_probes(k, d, cell_count)
    search_centers = centers if bf16 else centers - offset
    cells = build_cells(search_centers, cell_count, 4, rng=np.random.RandomState(args.seed))
    order = np.random.RandomState(args.seed).permutation(n)[:args.rows]
    rows = mx.array(x[np.sort(order)])
    batch = rows.astype(mx.bfloat16) if bf16 else rows - offset
    child = mx.full((k,), -1, dtype=mx.int32)
    mirror, norms = sorted_mirror(centers, cells, offset=None if bf16 else offset, precision=precision)
    probes = probe_cells(batch, cells, n_probe)
    bm = (tile or tile_shape(batch.dtype, d))[0]
    pairs, plan = bucket_pairs(probes, cells, bm)
    scores, positions = tiled_scan(batch, mirror, norms, probes, cells, tile=tile)
    mx.eval(centers, offset, batch, mirror, norms, probes, pairs, plan, scores, positions,
            *cells.arrays())

    def dense():
        return assign(batch, search_centers, precision=precision)[:1]

    stages = {
        "dense": dense,
        "mirror": lambda: sorted_mirror(centers, cells, offset=None if bf16 else offset,
                                        precision=precision),
        "probe": lambda: (probe_cells(batch, cells, n_probe),),
        "bucket": lambda: bucket_pairs(probes, cells, bm),
        "scan": lambda: tiled_scan(batch, mirror, norms, probes, cells, tile=tile),
        "reduce": lambda: reduce_links(batch, mirror, norms, scores, positions, cells, child),
        "two_level": lambda: two_level_assign(
            batch, *sorted_mirror(centers, cells, offset=None if bf16 else offset,
                                  precision=precision),
            cells, n_probe, child, tile=tile)[:1],
    }
    if args.scalar:
        stages["scalar_scan"] = lambda: scalar_scan(batch, mirror, norms, probes, cells)
    record = {name: dict(zip(("median_ms", "min_ms"), timed(function, args.repeats), strict=True))
              for name, function in stages.items()}

    sizes = np.asarray(cells.sizes).astype(np.int64)
    probed = np.asarray(probes)
    pair_counts = np.bincount(probed.reshape(-1), minlength=cell_count)
    useful = int((pair_counts * sizes).sum())
    bn = (tile or tile_shape(batch.dtype, d))[3]
    padded = int(((-(-pair_counts // bm) * bm) * (-(-sizes // bn) * bn)).sum())
    labels = np.asarray(two_level_assign(batch, mirror, norms, cells, n_probe, child, tile=tile)[0])
    exact = np.asarray(dense()[0])
    flops = 2.0 * d
    record.update(
        precision=precision, tile=list(tile or tile_shape(batch.dtype, d)), n=n, d=d, k=k,
        rows=int(batch.shape[0]), n_cells=cell_count, n_probe=n_probe,
        cell_size_min=int(sizes.min()), cell_size_median=float(np.median(sizes)),
        cell_size_max=int(sizes.max()), empty_cells=int((sizes == 0).sum()),
        useful_distances=useful, padded_distances=padded,
        evaluated_fraction=(useful + batch.shape[0] * cell_count) / (batch.shape[0] * k),
        miss_rate=float(np.mean(labels != exact)),
        dense_tflops=flops * batch.shape[0] * k / record["dense"]["median_ms"] / 1e9,
        scan_useful_tflops=flops * useful / record["scan"]["median_ms"] / 1e9,
        scan_padded_tflops=flops * padded / record["scan"]["median_ms"] / 1e9,
        speedup=record["dense"]["median_ms"] / record["two_level"]["median_ms"],
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", help="cached e-means benchmark dataset; omit for synthetic data")
    parser.add_argument("--data", type=Path, help="float32 training matrix stored as .npy")
    parser.add_argument("--synthetic", nargs=2, type=int, metavar=("N", "D"), default=(1_000_000, 96))
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--rows", type=int, default=32768, help="rows in the assigned batch")
    parser.add_argument("--cells", type=int)
    parser.add_argument("--probes", type=int)
    parser.add_argument("--precision", nargs="+", default=["float32"], choices=["float32", "bfloat16"])
    parser.add_argument("--tile", nargs="+", default=["default"],
                        help="BM,BK,PAD,BN tile shapes, or 'default' for the built-in shape")
    parser.add_argument("--fit-epochs", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--scalar", action="store_true", help="also time the scalar reference scan")
    parser.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--faiss-source", type=Path, default=base.REPO / "tmp/e_means/faiss")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data:
        x = np.ascontiguousarray(np.load(args.data, mmap_mode="r"), dtype=np.float32)
        name = args.data.stem
    elif args.dataset:
        data, _, _ = base.reference_modules(args.faiss_source, args.cache)
        x, _, _ = base.load_dataset(args.dataset, data, args.cache, None)
        name = args.dataset
    else:
        x = synthetic(args.synthetic[0], args.synthetic[1], 4 * args.k, args.seed)
        name = f"synthetic-{args.synthetic[0]}x{args.synthetic[1]}"
    info = mx.device_info()
    records = []
    for precision in args.precision:
        for text in args.tile:
            tile = None if text == "default" else tuple(int(v) for v in text.split(","))
            record = measure(x, args, precision, tile)
            record.update(dataset=name, device=info["device_name"], mlx=mx.__version__,
                          macos=platform.mac_ver()[0])
            records.append(record)
            stage = {key: record[key]["median_ms"] for key in
                     ("dense", "two_level", "mirror", "probe", "bucket", "scan", "reduce")}
            print(f"{name} d={record['d']} k={record['k']} {precision} tile={record['tile']}: "
                  + " ".join(f"{key}={value:.2f}ms" for key, value in stage.items())
                  + f" | speedup {record['speedup']:.2f}x fraction {record['evaluated_fraction']:.3f}"
                  f" miss {100 * record['miss_rate']:.2f}% | TFLOP/s dense {record['dense_tflops']:.2f}"
                  f" scan useful {record['scan_useful_tflops']:.2f} padded {record['scan_padded_tflops']:.2f}",
                  flush=True)
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
