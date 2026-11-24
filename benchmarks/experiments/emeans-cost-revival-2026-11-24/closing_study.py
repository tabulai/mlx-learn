"""Which closing mean should cost-based revival use?

One training run per (dataset, seed, budget, revival rule) captures the
labels of the closing pass; every closing rule is then evaluated from that
same run, so the rules are compared on identical centroid trajectories.

Closing rules (the returned centroid is the mean of its closing-pass rows; a
centroid without rows keeps its EMA position):

  ema       no closing mean: the EMA centroids
  none      every row counts for the centroid it was assigned to (the
            library's rule under mass-based revival)
  post      a split in the closing pass clears the sums of both ends, so they
            close from rows assigned afterwards (upstream without captured
            assignments, and its IVF/sharded classes)
  mover     as ``post``, but only the moved centroid is cleared
  second    rows assigned before their centroid moved go to their
            second-nearest of the same assignment, or are searched again if
            that one moved too; no twin step
  search    those rows are all searched again; no twin step
  twin      only the twin step: a donor's rows from before a split go to the
            nearer of the split's two centroids; stale rows are dropped
  handover  ``second`` and the twin step (upstream's default)
  searchtwin ``search`` and the twin step (what ``EMeans`` implements)
  fresh     every closing-pass row is assigned again to the final EMA
            centroids (one more assignment pass: the reference)

``none/ema-touched`` returns the EMA position of every centroid a
closing-pass move involved and the plain closing mean of the others; it was
the first implementation, replaced after ``surplus.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assess  # noqa: E402
from assess import base, cost_check, removal_batch, top2, train_mse  # noqa: E402

from mlxlearn._kernels.e_means import assign, batch_stats, ema_update  # noqa: E402
from mlxlearn.cluster._e_means import revive  # noqa: E402
from mlxlearn.cluster._recipe import resolve_recipe, scheduled_alpha  # noqa: E402

CHUNK = 65536
SHOWN = ("ema", "post", "mover", "twin", "handover", "searchtwin", "fresh", "none+8", "mover+8",
         "twin+8", "handover+8", "none/ema-moved", "none/ema-touched", "twin/ema-moved")


def fit(x, k, epochs, seed, mode):
    n, d = x.shape
    recipe = resolve_recipe(n, k, d, epochs, None, None, "auto", "faiss", None)
    size = recipe["batch_size"]
    data = mx.array(x)
    offset = mx.mean(data, axis=0)
    centers = data[mx.array(np.random.RandomState(seed).permutation(n)[:k], dtype=mx.int32)]
    mass = mx.full((k,), recipe["init_usage"], mx.float32)
    rng = np.random.RandomState(seed)
    zeros = lambda: (mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32))  # noqa: E731
    acc = {"none": zeros(), "post": zeros(), "mover": zeros()}
    long_axis = removal = axes = None
    if mode == "cost":
        a = rng.standard_normal((k, d))
        axes = mx.array((a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32))
        long_axis, removal = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
    mx.eval(data, offset, centers)
    labels_seen, second_seen = [], []
    moved_at = np.full(k, -1, dtype=np.int64)
    events, done, step, moves_total, last_pass_moves = [], 0, 0, 0, 0
    last = len(recipe["epoch_rows"]) - 1
    order_last = None
    for epoch, rows in enumerate(recipe["epoch_rows"]):
        order_host = np.random.RandomState((seed * 1000003 + epoch) % 2**32).permutation(n)
        order = mx.array(order_host, dtype=mx.int32)
        final = epoch == last
        if final:
            order_last = order_host[:rows]
        for start in range(0, rows, size):
            batch = data[order[start:min(start + size, rows)]]
            if mode == "cost":
                cc = centers - offset
                values, ids = top2(batch - offset, cc, mx.sum(cc * cc, axis=1))
                labels = ids[:, 0].astype(mx.uint32)
                batch_axis, batch_cost = removal_batch(batch, centers, axes, labels,
                                                       values[:, 1] - values[:, 0], k)
            else:
                labels, _ = assign(batch - offset, centers - offset)
            sums, counts = batch_stats(batch, labels, k)
            if final:
                acc = {name: (s + sums, c + counts) for name, (s, c) in acc.items()}
                labels_seen.append(labels)
                if mode == "cost":
                    second_seen.append(ids[:, 1])
                done += batch.shape[0]
            alpha = scheduled_alpha(recipe["alpha0"], step, recipe["total_steps"])
            previous = mass
            centers, mass = ema_update(centers, mass, sums, counts, alpha)
            if alpha == 1.0:
                mass = mx.where(counts > 0, mass, previous)
            if mode == "cost":
                long_axis = (1.0 - alpha) * long_axis + alpha * batch_axis
                removal = (1.0 - alpha) * removal + alpha * batch_cost
                norm = mx.sqrt(mx.sum(long_axis * long_axis, axis=1, keepdims=True))
                axes = mx.where(norm > 0, long_axis / mx.where(norm > 0, norm, 1.0), axes)
            if (step + 1) % recipe["check_unused_every"] == 0:
                if mode == "mass" and recipe["unused_threshold"] > 0:
                    uniform = mx.array(rng.random_sample(k), dtype=mx.float32)
                    centers, mass, number, _ = revive(centers, mass, uniform, recipe["unused_threshold"],
                                                      recipe["split_epsilon"])
                elif mode == "cost":
                    result, info = cost_check(centers, mass, long_axis, removal, axes)
                    if result is not None:
                        centers, mass, long_axis, removal, axes, touched = result
                        moves_total += info["moves"]
                        if final:
                            last_pass_moves += info["moves"]
                            source, target = info["source"], info["target"]
                            moved = np.zeros(k, dtype=bool)
                            moved[target] = True
                            moved = mx.array(moved)
                            acc["post"] = (mx.where(touched[:, None], 0.0, acc["post"][0]),
                                           mx.where(touched, 0.0, acc["post"][1]))
                            acc["mover"] = (mx.where(moved[:, None], 0.0, acc["mover"][0]),
                                            mx.where(moved, 0.0, acc["mover"][1]))
                            moved_at[target] = done
                            events.append((target.copy(), source.copy(), done))
            pending = [centers, mass] + [v for pair in acc.values() for v in pair]
            if mode == "cost":
                pending.extend((long_axis, removal, axes))
            if final:
                pending.append(labels_seen[-1])
                if mode == "cost":
                    pending.append(second_seen[-1])
            mx.eval(*pending)
            step += 1
    rows = recipe["epoch_rows"][-1]
    labels = np.concatenate([np.asarray(v) for v in labels_seen]).astype(np.int64)
    second = (np.concatenate([np.asarray(v) for v in second_seen]).astype(np.int64)
              if mode == "cost" else None)
    ema = np.array(centers)

    def close(sums, counts, weight=0.0, prefer_ema=None):
        """Mean of the closing rows, shrunk toward the EMA position by ``weight`` rows."""
        if weight:
            closing = (sums + weight * centers) / (counts + weight)[:, None]
        else:
            closing = mx.where((counts > 0)[:, None], sums / mx.maximum(counts[:, None], 1), centers)
        if prefer_ema is not None:
            closing = mx.where(mx.array(prefer_ema)[:, None], centers, closing)
        return np.array(closing), int((np.asarray(counts) == 0).sum())

    def rebuild(final_labels, keep=None):
        sums, counts = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
        index = np.arange(rows) if keep is None else np.flatnonzero(keep)
        for start in range(0, len(index), CHUNK):
            part = index[start:start + CHUNK]
            batch = data[mx.array(order_last[part], dtype=mx.int32)]
            s, c = batch_stats(batch, mx.array(final_labels[part], dtype=mx.uint32), k)
            sums, counts = sums + s, counts + c
            mx.eval(sums, counts)
        return sums, counts

    def search(index):
        found = []
        for start in range(0, len(index), CHUNK):
            part = index[start:start + CHUNK]
            batch = data[mx.array(order_last[part], dtype=mx.int32)]
            found.append(np.asarray(assign(batch - offset, centers - offset)[0]).astype(np.int64))
        return np.concatenate(found) if found else np.zeros(0, np.int64)

    detail = {"moves": moves_total, "last_pass_moves": last_pass_moves,
              "steps": step, "batch_size": size, "rows_last": rows}
    stats = {"none": acc["none"]}
    out = {"ema": (ema, 0)}
    rank = np.arange(rows)
    moved = np.zeros(k, dtype=bool)
    split = np.zeros(k, dtype=bool)
    if mode == "cost":
        for target, source, _ in events:
            moved[target] = True
            split[source] = True
        detail.update(moved=int(moved.sum()), touched=int((moved | split).sum()))
        stats["post"], stats["mover"] = acc["post"], acc["mover"]
        stale = rank < moved_at[labels]
        usable = stale & (second >= 0) & (moved_at[np.maximum(second, 0)] <= rank)
        detail.update(stale=int(stale.sum()), usable=int(usable.sum()))
        searched_all = None
        if stale.any():
            searched_all = np.full(rows, -1, dtype=np.int64)
            searched_all[stale] = search(np.flatnonzero(stale))

        def stale_rule(kind):
            result = labels.copy()
            if searched_all is None:  # no closing-pass row is stale
                return result
            if kind == "second":
                result[usable] = second[usable]
                rest = stale & ~usable
                result[rest] = searched_all[rest]
            elif kind == "search" and searched_all is not None:
                result[stale] = searched_all[stale]
            return result

        def twin_rule(result):
            if not events:
                return result, 0
            target = np.concatenate([[0]] + [t for t, _, _ in events]).astype(np.int64)
            source = np.concatenate([[k]] + [s for _, s, _ in events]).astype(np.int64)
            born = np.concatenate([[0]] + [np.full(len(t), b) for t, _, b in events]).astype(np.int64)
            span = rows + 1
            by = np.argsort(source * span + born, kind="stable")
            target, source, born = target[by], source[by], born[by]
            key = source * span + born
            position = np.minimum(np.searchsorted(key, result * span + rank, side="right"), len(key) - 1)
            child = target[position]
            twin = (~stale & (source[position] == result) & (born[position] > rank)
                    & (moved_at[child] == born[position]))
            index = np.flatnonzero(twin)
            own, other = result[index], child[index]
            points = x[order_last[index]].astype(np.float64)
            near_own = np.square(points - ema[own]).sum(axis=1)
            near_other = np.square(points - ema[other]).sum(axis=1)
            result = result.copy()
            result[index] = np.where((near_other < near_own) | ((near_other == near_own) & (other < own)),
                                     other, own)
            return result, len(index)

        stats["second"] = rebuild(stale_rule("second"))
        stats["search"] = rebuild(stale_rule("search"))
        twin_labels, twins = twin_rule(labels.copy())
        detail["twin_rows"] = twins
        stats["twin"] = rebuild(twin_labels, keep=~stale)
        stats["handover"] = rebuild(twin_rule(stale_rule("second"))[0])
        stats["searchtwin"] = rebuild(twin_rule(stale_rule("search"))[0])
    stats["fresh"] = rebuild(search(rank))
    for name, pair in stats.items():
        out[name] = close(*pair)
    # Mean of the closing rows shrunk toward the EMA position by a few rows.
    for name in ("none", "post", "mover", "twin", "handover"):
        if name in stats:
            for weight in (2.0, 8.0, 32.0):
                out[f"{name}+{weight:g}"] = close(*stats[name], weight=weight)
    if mode == "cost":
        # EMA position for the centroids a closing-pass move involved.
        out["none/ema-moved"] = close(*stats["none"], prefer_ema=moved)
        out["none/ema-touched"] = close(*stats["none"], prefer_ema=moved | split)
        out["twin/ema-moved"] = close(*stats["twin"], prefer_ema=moved)
    return out, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["deep1b-1M"])
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10])
    parser.add_argument("--epochs", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--modes", nargs="+", default=["mass", "cost"])
    parser.add_argument("--faiss-source", type=Path, default=assess.FAISS_SOURCE)
    parser.add_argument("--cache", type=Path, default=assess.CACHE)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assess.FAISS_SOURCE, assess.CACHE = args.faiss_source, args.cache
    with mx.stream(mx.gpu):
        for name in args.datasets:
            x, test = assess.load(name)
            for seed in args.seeds:
                for epochs in args.epochs:
                    for mode in args.modes:
                        started = time.perf_counter()
                        closings, detail = fit(x, args.k, epochs, seed, mode)
                        record = {"dataset": name, "n": len(x), "d": x.shape[1], "k": args.k, "seed": seed,
                                  "epochs": epochs, "mode": mode, **detail,
                                  "seconds": time.perf_counter() - started, "rules": {}}
                        for rule, (centers, unclosed) in closings.items():
                            record["rules"][rule] = {"test_mse": base.mse(centers, test),
                                                     "train_mse": train_mse(x, centers),
                                                     "unclosed": unclosed}
                        with open(args.output, "a") as handle:
                            handle.write(json.dumps(record) + "\n")
                        reference = record["rules"]["none"]["test_mse"]
                        print(f"{name} seed={seed} epochs={epochs:g} {mode}: "
                              + " ".join(f"{rule}={100 * (v['test_mse'] / reference - 1):+.2f}%"
                                         for rule, v in record["rules"].items()
                                         if rule in SHOWN)
                              + f" | none={reference:.6g} " + " ".join(f"{key}={value}" for key, value in detail.items()),
                              flush=True)


if __name__ == "__main__":
    main()
