"""What the moves do when centroids outnumber separated clusters.

Runs the training loop of ``EMeans.train_device`` from the library's own
kernels on a mixture of ``surplus.py`` and, at every revival check, counts

- the moved centroids whose nearest other centroid moves at the same check
  (two centroids that share a cluster are both cheap to remove), and
- the modes left without a centroid within the noise radius of their centre,
  before and after the check.

It then compares the modes without a centroid at the end of training with a
fit under the mass rule.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/surplus_moves.py \
        --case 200 273 24 30000 --epochs 3
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

from mlxlearn._kernels.e_means import assign, batch_stats, ema_update, removal_stats, removal_update
from mlxlearn.cluster import EMeans
from mlxlearn.cluster._e_means import _normalize, epoch_seed, removal_plan, revive_by_cost
from mlxlearn.cluster._recipe import scheduled_alpha


def uncovered(centers, means):
    """Modes whose centre has no centroid within the noise radius sqrt(d)."""
    c, m = np.asarray(centers, dtype=np.float64), means.astype(np.float64)
    distance = (m * m).sum(axis=1)[:, None] - 2.0 * m @ c.T + (c * c).sum(axis=1)[None]
    return int((distance.min(axis=1) > means.shape[1]).sum())


def trace(train, means, k, epochs, seed):
    """One fit with the cost rule; per check: moves, moves with a moving neighbour, uncovered modes."""
    model = EMeans(k, max_iter=epochs, recipe="faiss", random_state=seed, revival="cost", compute_labels=False)
    prepared = model.prepare_device(train)
    recipe, data, centers, offset = prepared.recipe, prepared.data, prepared.initial, prepared.offset
    n, d = data.shape
    mass = mx.full((k,), recipe["init_usage"], dtype=mx.float32)
    axes = _normalize(mx.random.normal((k, d), key=mx.random.key(prepared.seed)))
    long_axis, removal = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
    size, step, checks = recipe["batch_size"], 0, []
    for epoch, rows in enumerate(recipe["epoch_rows"]):
        order = mx.array(np.random.RandomState(
            epoch_seed(prepared.seed, epoch, distinct=True)).permutation(n), dtype=mx.int32)
        for start in range(0, rows, size):
            batch = data[order[start:min(start + size, rows)]]
            labels, _, margins = assign(batch - offset, centers - offset, second=True)
            axis_sums, costs = removal_stats(batch, labels, margins, centers, axes)
            sums, counts = batch_stats(batch, labels, k)
            alpha = scheduled_alpha(recipe["alpha0"], step, recipe["total_steps"])
            previous = mass
            centers, mass = ema_update(centers, mass, sums, counts, alpha)
            if alpha == 1.0:
                mass = mx.where(counts > 0, mass, previous)
            long_axis, removal, axes, lengths = removal_update(long_axis, removal, axis_sums, costs, axes, alpha)
            if (step + 1) % recipe["check_unused_every"] == 0:
                plan = removal_plan(removal, mass, lengths)
                before = np.asarray(centers).astype(np.float64)
                moved = np.flatnonzero(np.asarray(plan[0]))
                together = 0
                if len(moved):
                    distance = ((before[moved] ** 2).sum(axis=1)[:, None] - 2.0 * before[moved] @ before.T
                                + (before ** 2).sum(axis=1)[None])
                    distance[np.arange(len(moved)), moved] = np.inf
                    together = int(np.isin(distance.argmin(axis=1), moved).sum())
                centers, mass, long_axis, removal, axes, _, _ = revive_by_cost(
                    centers, mass, long_axis, removal, axes, lengths, plan, prepared.bounds)
                checks.append({"step": step + 1, "moves": len(moved), "together": together,
                               "uncovered_before": uncovered(before, means),
                               "uncovered_after": uncovered(centers, means)})
            mx.eval(centers, mass, long_axis, removal, axes)
            step += 1
    return checks, uncovered(centers, means), step


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", nargs=4, type=int, default=[200, 273, 24, 30_000],
                        metavar=("MODES", "CENTROIDS", "FEATURES", "ROWS"))
    parser.add_argument("--spread", type=float, default=3.0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    modes, k, d, n = args.case
    rng = np.random.default_rng(1)
    means = rng.normal(size=(modes, d)).astype(np.float32) * args.spread
    train = means[rng.integers(0, modes, n)] + rng.normal(size=(n, d)).astype(np.float32)
    print(f"{modes} modes, {k} centroids, d={d}, {n} rows, spread {args.spread:g}, {args.epochs:g} epochs")
    with mx.stream(mx.gpu):
        for seed in args.seeds:
            checks, left, steps = trace(train, means, k, args.epochs, seed)
            mass = EMeans(k, max_iter=args.epochs, recipe="faiss", random_state=seed, compute_labels=False).fit(train)
            moves = sum(check["moves"] for check in checks)
            together = sum(check["together"] for check in checks)
            last = checks[-1]
            print(f"  seed {seed}: {moves} moves in {len(checks)} checks, {together / max(moves, 1):.0%} with the nearest "
                  f"centroid moving at the same check; modes without a centroid, mean over checks: "
                  f"{np.mean([c['uncovered_before'] for c in checks]):.1f} before, "
                  f"{np.mean([c['uncovered_after'] for c in checks]):.1f} after; last check at step {last['step']} of "
                  f"{steps}: {last['moves']} moves, {last['uncovered_before']} -> {last['uncovered_after']}; "
                  f"at the end: cost rule {left}, mass rule {uncovered(mass.ema_centers_, means)}", flush=True)


if __name__ == "__main__":
    main()
