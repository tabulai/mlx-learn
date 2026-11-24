"""Scratch measurements for assessing FAISS PRs 5915 / 5965 against mlxlearn.

Nothing in the repository is modified. Three sub-commands:

  stages   time the existing two-level assignment stage by stage, with the
           probe stage split into its coarse GEMM and its selection, and time
           a top-2 dense assignment built from existing kernels (upper bound)
  proto    prototype of cost-based revival (PR 5915, closing rule of
           capture_assignments=False) against the library's mass-based rule
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarks.e_means import run as base  # noqa: E402

from mlxlearn._kernels.e_means import assign, batch_stats, ema_update  # noqa: E402
from mlxlearn._kernels.o_means import (  # noqa: E402
    bucket_pairs,
    build_cells,
    probe_cells,
    reduce_links,
    sorted_mirror,
    tile_shape,
    tiled_scan,
    two_level_assign,
)
from mlxlearn.cluster import EMeans  # noqa: E402
from mlxlearn.cluster._e_means import revive, split_plan  # noqa: E402
from mlxlearn.cluster._o_schedule import num_cells, num_probes  # noqa: E402
from mlxlearn.cluster._recipe import resolve_recipe, scheduled_alpha  # noqa: E402
from mlxlearn_device._binary32 import matmul  # noqa: E402
from mlxlearn_device._neighbors_fused import select_topk  # noqa: E402

REMOVAL_THRESHOLD = 0.3


def timed(function, repeats=9):
    def once():
        start = time.perf_counter()
        outputs = function()
        mx.eval(*outputs)
        mx.synchronize()
        return time.perf_counter() - start

    once()
    once()
    times = [once() for _ in range(repeats)]
    return round(float(np.median(times)) * 1e3, 3), round(float(np.min(times)) * 1e3, 3)


def top2(xc, cc, norms, tile_rows=16384):
    """Two smallest scores ||c||^2 - 2 x.c per row and their centroid ids."""
    values, ids = [], []
    for start in range(0, xc.shape[0], tile_rows):
        scores = norms - 2.0 * matmul(xc[start:start + tile_rows], cc, transpose_b=True)
        v, i = select_topk(scores, 2)
        values.append(v)
        ids.append(i)
    if len(values) == 1:
        return values[0], ids[0]
    return mx.concatenate(values), mx.concatenate(ids)


def removal_batch(batch, centers, axes, labels, margin, k):
    """Batch sums [sum p y | sum d2 - d1] of PR 5915 from existing kernels."""
    index = labels.astype(mx.int32)
    y = batch - centers[index]
    p = mx.sum(y * axes[index], axis=1, keepdims=True)
    sums, _ = batch_stats(p * y, labels, k)
    cost = mx.zeros((k,), mx.float32).at[index].add(margin)
    return sums, cost


FAISS_SOURCE = REPO / "tmp/e_means/faiss"
CACHE = base.DEFAULT_CACHE


def load(name):
    """Training and test rows of a cached benchmark dataset, or of
    ``<name>-train.npy`` with ``<name>-test.npy`` beside it, for a machine
    without the dataset cache."""
    if str(name).endswith("-train.npy"):
        test = np.load(str(name).replace("-train.npy", "-test.npy"))
        return (np.ascontiguousarray(np.load(name), dtype=np.float32),
                np.ascontiguousarray(test, dtype=np.float32))
    data, _, _ = base.reference_modules(FAISS_SOURCE, CACHE)
    x, test, _ = base.load_dataset(name, data, CACHE, None)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(test, dtype=np.float32)


def stages(args):
    x, _ = load(args.dataset)
    n, d = x.shape
    k, seed = args.k, args.seed
    model = EMeans(k, max_iter=1.0, recipe="faiss", batch_law="throughput",
                   random_state=seed, compute_labels=False).fit(x)
    centers, offset = mx.array(model.cluster_centers_), mx.array(model.offset_)
    cell_count = args.cells or num_cells(k)
    n_probe = args.probes or num_probes(k, d, cell_count)
    search_centers = centers - offset
    search_norms = mx.sum(search_centers * search_centers, axis=1)
    cells = build_cells(search_centers, cell_count, 4, rng=np.random.RandomState(seed))
    order = np.random.RandomState(seed).permutation(n)[:args.rows]
    raw = mx.array(x[np.sort(order)])
    batch = raw - offset
    child = mx.full((k,), -1, dtype=mx.int32)
    mirror, norms = sorted_mirror(centers, cells, offset=offset, precision="float32")
    probes = probe_cells(batch, cells, n_probe)
    bm = tile_shape(batch.dtype, d)[0]
    pairs, plan = bucket_pairs(probes, cells, bm)
    scores, positions = tiled_scan(batch, mirror, norms, probes, cells)
    coarse = cells.norms - 2.0 * matmul(batch, cells.centres, transpose_b=True)
    labels = assign(batch, search_centers)[0]
    values2, ids2 = top2(batch, search_centers, search_norms)
    margin = values2[:, 1] - values2[:, 0]
    rng = np.random.RandomState(seed)
    axes = rng.standard_normal((k, d)).astype(np.float32)
    axes = mx.array(axes / np.linalg.norm(axes, axis=1, keepdims=True))
    mx.eval(centers, offset, raw, batch, mirror, norms, probes, pairs, plan, scores, positions,
            coarse, labels, values2, ids2, margin, axes, search_centers, search_norms, *cells.arrays())
    agree = float(mx.mean((ids2[:, 0].astype(mx.uint32) == labels).astype(mx.float32)).item())
    time.sleep(args.rest)
    table = {
        "dense (library auto)": lambda: assign(batch, search_centers)[:1],
        "dense gemm + argmin": lambda: assign(batch, search_centers, method="gemm")[:1],
        "dense gemm + select_topk(2)": lambda: top2(batch, search_centers, search_norms),
        "two_level (whole)": lambda: two_level_assign(
            batch, *sorted_mirror(centers, cells, offset=offset, precision="float32"),
            cells, n_probe, child)[:1],
        "  mirror": lambda: sorted_mirror(centers, cells, offset=offset, precision="float32"),
        "  probe (gemm + select)": lambda: (probe_cells(batch, cells, n_probe),),
        "    probe gemm only": lambda: (cells.norms - 2.0 * matmul(batch, cells.centres, transpose_b=True),),
        "    probe select only": lambda: select_topk(coarse, n_probe),
        "  bucket": lambda: bucket_pairs(probes, cells, bm),
        "  scan": lambda: tiled_scan(batch, mirror, norms, probes, cells),
        "  reduce": lambda: reduce_links(batch, mirror, norms, scores, positions, cells, child),
        "stats (sums, counts)": lambda: batch_stats(raw, labels, k),
        "removal stats (U, R)": lambda: removal_batch(raw, centers, axes, labels, margin, k),
    }
    record = {"dataset": args.dataset, "n": n, "d": d, "k": k, "rows": args.rows,
              "cells": cell_count, "probes": n_probe, "top2_first_matches_dense": agree,
              "device": mx.device_info()["device_name"], "mlx": mx.__version__}
    print(json.dumps(record))
    for name, function in table.items():
        median, minimum = timed(function, args.repeats)
        record[name.strip()] = median
        print(f"{name:34s} median {median:8.3f} ms   min {minimum:8.3f} ms", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(record, indent=1) + "\n")


def separate(donor, child):
    collapsed = (donor == child).all(axis=1)
    value = child[:, 0]
    with np.errstate(over="ignore"):
        neighbor = np.nextafter(value, np.float32(np.inf))
    neighbor = np.where(np.isfinite(neighbor), neighbor, np.nextafter(value, np.float32(-np.inf)))
    child[:, 0] = np.where(collapsed, neighbor, value)


def cost_check(centers, mass, long_axis, removal, axes):
    """PR 5915's _choose_removal_splits and _apply_splits, on the host in FP64."""
    r = np.array(removal).astype(np.float64)
    usage = np.array(mass).astype(np.float64)
    u = np.array(long_axis)
    norm = np.linalg.norm(u.astype(np.float64), axis=1)
    candidates = np.flatnonzero((r <= 0) | (r < REMOVAL_THRESHOLD * np.median(r)))
    info = {"candidates": int(len(candidates)), "zero_cost": int((r <= 0).sum()), "moves": 0}
    if len(candidates) == 0:
        return None, info
    kept = candidates[np.argsort(r[candidates], kind="stable")].astype(np.int64)
    donors = usage > 0
    donors[kept] = False
    donors = np.flatnonzero(donors)
    donors = donors[np.argsort(-norm[donors].astype(np.float32), kind="stable")]
    moves = min(len(kept), len(donors))
    if moves == 0:
        return None, info
    source, target = donors[:moves], kept[:moves]
    c = np.array(centers)
    a = np.array(axes)
    m = np.array(mass)
    r32 = np.array(removal)
    position = c[source].astype(np.float64)
    nv, mv = norm[source], usage[source]
    valid = (nv > 0) & (mv > 0) & np.isfinite(nv) & np.isfinite(mv)
    spread = np.sqrt(2.0 * np.where(valid, nv / np.where(valid, mv, 1.0), 0.0) / np.pi)
    offset = spread[:, None] * a[source].astype(np.float64)
    donor = (position - offset).astype(np.float32)
    child = (position + offset).astype(np.float32)
    separate(donor, child)
    c[source], c[target] = donor, child
    m[source] /= 2.0
    m[target] = m[source]
    u[source] /= 2.0
    u[target] = u[source]
    r32[source] /= 2.0
    r32[target] = r32[source]
    a[target] = a[source]
    touched = np.zeros(len(m), dtype=bool)
    touched[source] = touched[target] = True
    info["moves"] = int(moves)
    info["source"], info["target"] = source, target
    return (mx.array(c), mx.array(m), mx.array(u), mx.array(r32), mx.array(a), mx.array(touched)), info


def axis_split(centers, mass, long_axis, axes, plan):
    """Mass-selected splits (library plan) placed at the Gaussian half-means along the donor's axis."""
    is_child, _, source_of, _, _ = (np.array(v) for v in plan)
    target = np.flatnonzero(is_child)
    info = {"candidates": int(len(target)), "zero_cost": 0, "moves": int(len(target))}
    if len(target) == 0:
        return None, info
    source = source_of[target].astype(np.int64)
    c, a, m, u = np.array(centers), np.array(axes), np.array(mass), np.array(long_axis)
    norm = np.linalg.norm(u[source].astype(np.float64), axis=1)
    usage = m[source].astype(np.float64)
    valid = (norm > 0) & (usage > 0) & np.isfinite(norm) & np.isfinite(usage)
    spread = np.sqrt(2.0 * np.where(valid, norm / np.where(valid, usage, 1.0), 0.0) / np.pi)
    position = c[source].astype(np.float64)
    offset = spread[:, None] * a[source].astype(np.float64)
    donor = (position - offset).astype(np.float32)
    child = (position + offset).astype(np.float32)
    separate(donor, child)
    c[source], c[target] = donor, child
    m[source] /= 2.0
    m[target] = m[source]
    u[source] /= 2.0
    u[target] = u[source]
    a[target] = a[source]
    return (mx.array(c), mx.array(m), mx.array(u), mx.array(a)), info


def proto_fit(x, k, epochs, seed, mode, *, check_from=0.0):
    """One fit with the Faiss recipe.

    ``mode``: 'mass' (library rule), 'cost' (PR 5915 default) or 'massaxis'
    (mass-selected donors and dead clusters, split along the Oja axis: PR
    5915 with a numeric ``unused_threshold``, which needs no second-nearest).
    """
    n, d = x.shape
    recipe = resolve_recipe(n, k, d, epochs, None, None, "auto", "faiss", None)
    size = recipe["batch_size"]
    data = mx.array(x)
    offset = mx.mean(data, axis=0)
    centers = data[mx.array(np.random.RandomState(seed).permutation(n)[:k], dtype=mx.int32)]
    mass = mx.full((k,), recipe["init_usage"], mx.float32)
    rng = np.random.RandomState(seed)
    pass_sums, pass_counts = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
    long_axis = removal = axes = None
    if mode in ("cost", "massaxis"):
        a = rng.standard_normal((k, d))
        axes = mx.array((a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32))
        long_axis, removal = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
    mx.eval(data, offset, centers)
    splits = mx.array(0, dtype=mx.int32)
    moves, checks, step = [], [], 0
    last = len(recipe["epoch_rows"]) - 1
    started = time.perf_counter()
    for epoch, rows in enumerate(recipe["epoch_rows"]):
        order = mx.array(np.random.RandomState((seed * 1000003 + epoch) % 2**32).permutation(n),
                         dtype=mx.int32)
        final = epoch == last
        for start in range(0, rows, size):
            batch = data[order[start:min(start + size, rows)]]
            if mode == "cost":
                cc = centers - offset
                values, ids = top2(batch - offset, cc, mx.sum(cc * cc, axis=1))
                labels = ids[:, 0].astype(mx.uint32)
                margin = values[:, 1] - values[:, 0]
                batch_axis, batch_cost = removal_batch(batch, centers, axes, labels, margin, k)
            else:
                labels, _ = assign(batch - offset, centers - offset)
                if mode == "massaxis":
                    batch_axis, _ = removal_batch(batch, centers, axes, labels,
                                                  mx.zeros((batch.shape[0],), mx.float32), k)
            sums, counts = batch_stats(batch, labels, k)
            if final:
                pass_sums, pass_counts = pass_sums + sums, pass_counts + counts
            alpha = scheduled_alpha(recipe["alpha0"], step, recipe["total_steps"])
            previous = mass
            centers, mass = ema_update(centers, mass, sums, counts, alpha)
            if alpha == 1.0:
                mass = mx.where(counts > 0, mass, previous)
            if mode in ("cost", "massaxis"):
                long_axis = (1.0 - alpha) * long_axis + alpha * batch_axis
                if mode == "cost":
                    removal = (1.0 - alpha) * removal + alpha * batch_cost
                norm = mx.sqrt(mx.sum(long_axis * long_axis, axis=1, keepdims=True))
                axes = mx.where(norm > 0, long_axis / mx.where(norm > 0, norm, 1.0), axes)
            if (step + 1) % recipe["check_unused_every"] == 0:
                if mode == "mass" and recipe["unused_threshold"] > 0:
                    uniform = mx.array(rng.random_sample(k), dtype=mx.float32)
                    centers, mass, number, _ = revive(centers, mass, uniform, recipe["unused_threshold"],
                                                      recipe["split_epsilon"])
                    splits = splits + number
                elif mode == "massaxis" and recipe["unused_threshold"] > 0:
                    uniform = mx.array(rng.random_sample(k), dtype=mx.float32)
                    result, info = axis_split(centers, mass, long_axis, axes,
                                              split_plan(mass, uniform, recipe["unused_threshold"]))
                    info["step"] = step + 1
                    checks.append(info)
                    if result is not None:
                        centers, mass, long_axis, axes = result
                        moves.append(info["moves"])
                elif mode == "cost":
                    result, info = cost_check(centers, mass, long_axis, removal, axes)
                    info["step"] = step + 1
                    info.pop("source", None)
                    info.pop("target", None)
                    checks.append(info)
                    if result is not None:
                        centers, mass, long_axis, removal, axes, touched = result
                        moves.append(info["moves"])
                        if final:
                            pass_sums = mx.where(touched[:, None], 0.0, pass_sums)
                            pass_counts = mx.where(touched, 0.0, pass_counts)
            pending = [centers, mass, pass_sums, pass_counts, splits]
            if mode in ("cost", "massaxis"):
                pending.extend((long_axis, axes))
            if mode == "cost":
                pending.append(removal)
            mx.eval(*pending)
            step += 1
    closing = mx.where((pass_counts > 0)[:, None], pass_sums / mx.maximum(pass_counts[:, None], 1), centers)
    mx.eval(closing)
    seconds = time.perf_counter() - started
    total = int(sum(moves)) if mode in ("cost", "massaxis") else int(splits.item())
    return np.array(closing), {
        "seconds": seconds, "steps": step, "batch_size": size, "alpha0": recipe["alpha0"],
        "splits": total, "checks": checks,
        "unclosed": int((np.array(pass_counts) == 0).sum()),
    }


def train_mse(x, centers, rows=200_000, seed=0):
    """Float32 GPU assignment on a fixed training sample (comparison only)."""
    pick = np.sort(np.random.RandomState(seed).permutation(len(x))[:rows])
    sample = mx.array(x[pick])
    c = mx.array(centers)
    offset = mx.mean(sample, axis=0)
    _, distances = assign(sample - offset, c - offset)
    return float(np.asarray(distances).sum(dtype=np.float64) / len(pick))


def proto(args):
    out = Path(args.output)
    for name in args.datasets:
        x, test = load(name)
        n, d = x.shape
        for k in args.k:
            for seed in args.seeds:
                for epochs in args.epochs:
                    for mode in args.modes:
                        if mode in ("lib", "libcost"):
                            # ``libcost`` is the library's own cost-based revival.
                            options = {"revival": "cost"} if mode == "libcost" else {}
                            options.update(args.option or {})
                            model = EMeans(k, max_iter=epochs, recipe="faiss", random_state=seed,
                                           compute_labels=False, **options).fit(x)
                            centers = model.cluster_centers_
                            extra = {"seconds": model.optimization_time_, "splits": model.n_splits_,
                                     "steps": model.n_steps_, "batch_size": model.recipe_["batch_size"]}
                        else:
                            centers, extra = proto_fit(x, k, epochs, seed, mode)
                        record = {"dataset": name, "n": n, "d": d, "k": k, "seed": seed,
                                  "epochs": epochs, "mode": mode,
                                  "test_mse": base.mse(centers, test),
                                  "train_mse": train_mse(x, centers), **extra}
                        with out.open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                        checks = record.pop("checks", None)
                        note = ""
                        if checks:
                            series = [c["moves"] for c in checks]
                            note = f" moves/check first={series[:3]} last={series[-3:]}"
                        print(f"{name} k={k} seed={seed} epochs={epochs} {mode:5s} "
                              f"test={record['test_mse']:.6g} train={record['train_mse']:.6g} "
                              f"t={record['seconds']:.2f}s splits={record['splits']}"
                              f" steps={record['steps']} B={record['batch_size']}{note}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stages")
    s.add_argument("--dataset", default="spacev-1M")
    s.add_argument("--k", type=int, default=16384)
    s.add_argument("--rows", type=int, default=32768)
    s.add_argument("--cells", type=int)
    s.add_argument("--probes", type=int)
    s.add_argument("--seed", type=int, default=10)
    s.add_argument("--repeats", type=int, default=9)
    s.add_argument("--rest", type=float, default=15.0)
    s.add_argument("--output")
    p = sub.add_parser("proto")
    p.add_argument("--datasets", nargs="+", default=["deep1b-1M"])
    p.add_argument("--k", nargs="+", type=int, default=[16384])
    p.add_argument("--seeds", nargs="+", type=int, default=[10])
    p.add_argument("--epochs", nargs="+", type=float, default=[2.0])
    p.add_argument("--modes", nargs="+", default=["lib", "mass", "cost"])
    p.add_argument("--output", required=True)
    p.add_argument("--option", type=json.loads,
                   help="JSON dictionary of further EMeans options for the library modes")
    for command in (s, p):
        command.add_argument("--faiss-source", type=Path, default=FAISS_SOURCE)
        command.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    globals().update(FAISS_SOURCE=args.faiss_source, CACHE=args.cache)
    with mx.stream(mx.gpu):
        {"stages": stages, "proto": proto}[args.command](args)


if __name__ == "__main__":
    main()
