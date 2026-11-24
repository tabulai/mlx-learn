"""Where the time of the closing of ``EMeans(revival="cost")`` goes.

A fit whose centroids moved during the last pass ends in
``EMeans._reroute_closing``. This script runs fits with a copy of that method
that synchronizes after each of its three parts, and prints their times next
to the untimed fit and a fit under the mass rule. The copy follows the method
at the commit that added it; compare the two before trusting it after a change
there.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/closing_profile.py \
        --datasets deep1b-1M spacev-1M --epochs 1 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assess  # noqa: E402

import mlxlearn.cluster._e_means as module  # noqa: E402
from mlxlearn._kernels.e_means import batch_stats, gather_bf16, twin_route  # noqa: E402
from mlxlearn.cluster import EMeans  # noqa: E402

PARTS = ("labels and mask", "stale rows assigned again", "twins routed, sums rebuilt")
measured = {}


def timed_reroute(self, data, order, log, centers, *, keep_labels=False):
    mx.eval(centers, log.moved_at, log.moves)
    mx.synchronize()
    clock = [time.perf_counter()]

    def lap(name):
        mx.synchronize()
        clock.append(time.perf_counter())
        measured[name] = clock[-1] - clock[-2]

    k, batch_size = int(self.n_clusters), self.recipe_["batch_size"]
    rows = log.done
    labels, stale, twins = log.finish()
    readout = module._normalize(centers) if self.spherical else centers

    def gather(batch_order):
        return gather_bf16(data, batch_order) if self.precision == "bfloat16" else data[batch_order]

    positions = np.flatnonzero(np.asarray(stale))
    mx.eval(labels, twins)
    lap(PARTS[0])
    measured.update(rows=rows, stale=int(positions.size), twins=int(mx.sum(twins >= 0).item()), calls=0)
    clock[-1] = time.perf_counter()
    if positions.size:
        found, start = [], 0
        chunks = module._search_chunks(positions.size, batch_size)
        measured["calls"] = len(chunks)
        for taken, size in chunks:
            padded = np.resize(positions[start:start + taken], size).astype(np.int32)
            found.append(self._assign(gather(order[mx.array(padded)]), readout)[0][:taken])
            mx.eval(found[-1])
            start += taken
        labels[mx.array(positions.astype(np.int32))] = found[0] if len(found) == 1 else mx.concatenate(found)
        mx.eval(labels)
    lap(PARTS[1])
    sums, counts = mx.zeros((k, data.shape[1]), mx.float32), mx.zeros((k,), mx.float32)
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        batch = gather(order[start:stop])
        final = twin_route(batch, labels[start:stop], twins[start:stop], readout)
        batch_sums, batch_counts = batch_stats(batch, final, k)
        sums, counts = sums + batch_sums, counts + batch_counts
        mx.async_eval(sums, counts, final)
    mx.eval(sums, counts)
    lap(PARTS[2])
    return sums, counts, None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["deep1b-1M"])
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--epochs", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--precision", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--faiss-source", type=Path, default=assess.FAISS_SOURCE)
    parser.add_argument("--cache", type=Path, default=assess.CACHE)
    args = parser.parse_args()
    assess.FAISS_SOURCE, assess.CACHE = args.faiss_source, args.cache
    original = EMeans._reroute_closing
    print(f"{mx.device_info()['device_name']}, MLX {mx.__version__}, {args.k} centroids, seed {args.seed}")
    with mx.stream(mx.gpu):
        for name in args.datasets:
            x, _ = assess.load(name)
            label = Path(str(name)).name.replace("-train.npy", "")
            for precision in args.precision:
                for epochs in args.epochs:
                    options = {"max_iter": epochs, "recipe": "faiss", "random_state": args.seed,
                               "precision": precision, "compute_labels": False}
                    seconds = {}
                    for revival in ("mass", "cost"):
                        model = EMeans(args.k, revival=revival, **options)
                        prepared = model.prepare_device(x)
                        model.train_device(prepared)
                        seconds[revival] = float(np.median(
                            [model.train_device(prepared).seconds for _ in range(args.repeats)]))
                    EMeans._reroute_closing = timed_reroute
                    try:
                        model.train_device(prepared)
                        model.train_device(prepared)
                    finally:
                        EMeans._reroute_closing = original
                    closing = sum(measured[part] for part in PARTS)
                    print(f"{label} {precision} epochs={epochs:g}: mass {seconds['mass']:.3f} s, cost {seconds['cost']:.3f} s; "
                          f"closing {1e3 * closing:.1f} ms ({closing / seconds['cost']:.1%} of the fit): "
                          + ", ".join(f"{part} {1e3 * measured[part]:.1f} ms" for part in PARTS)
                          + f"; {measured['stale'] / measured['rows']:.2%} of rows stale ({measured['calls']} calls), "
                          f"{measured['twins'] / measured['rows']:.1%} with a twin", flush=True)


if __name__ == "__main__":
    main()
