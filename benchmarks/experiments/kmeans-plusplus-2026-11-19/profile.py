"""Where k-means++ seeding spends its time: scikit-learn's helper and the Metal screen.

``reference`` times a stage-by-stage replica of ``sklearn.cluster.kmeans_plusplus`` for
float32 input and checks that it selects the helper's centers. ``screen`` reports the
stages of mlxlearn's seeding, which selects the same centers.

    MNIST_FAMILY_ROOT=... python profile.py reference --centers 1024
    MNIST_FAMILY_ROOT=... python profile.py screen --centers 1024 4096
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import kmeans_plusplus

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "mnist-family-2026-11-02"))
from data import load  # noqa: E402

CAPTURE_STEPS = (1, 4, 16, 64, 256, 1024, 4096)


def reference_profile(x, k, seed):
    """Run the helper's steps with a timer around each stage."""
    n, d = x.shape
    trials = 2 + int(np.log(k))
    seconds = defaultdict(float)
    clock = time.perf_counter

    def distances(candidates):
        # Float32 input: chunks of rows are upcast to float64 for every call.
        count = len(candidates)
        budget = max(((count + n) * d + count * n) / 10, 10 * 2**17)
        length = max(int((-2 * d + math.sqrt((2 * d) ** 2 + 4 * budget)) / 2), 1)
        assert count <= length
        out = np.empty((count, n), np.float32)
        wide = candidates.astype(np.float64)
        own = np.einsum("ij,ij->i", wide, wide)[:, None]
        for start in range(0, n, length):
            a = clock()
            block = x[start:start + length].astype(np.float64)
            b = clock()
            norms = np.einsum("ij,ij->i", block, block)[None, :]
            c = clock()
            product = -2 * (wide @ block.T)
            e = clock()
            product += own
            product += norms
            out[:, start:start + length] = product
            f = clock()
            seconds["distances: upcast rows to float64"] += b - a
            seconds["distances: row norms"] += c - b
            seconds["distances: GEMM"] += e - c
            seconds["distances: add norms, round to float32"] += f - e
        np.maximum(out, 0, out=out)
        return out

    rng = np.random.RandomState(seed)
    weights = np.ones(n, np.float32)
    indices = np.full(k, -1, dtype=int)
    start = clock()
    indices[0] = rng.choice(n, p=weights / weights.sum())
    closest = distances(x[indices[:1]])
    potential = closest @ weights
    captured = {}
    for step in range(1, k):
        a = clock()
        draws = rng.uniform(size=trials) * potential
        b = clock()
        prefix = np.cumsum(weights * closest)
        c = clock()
        ids = np.searchsorted(prefix, draws)
        np.clip(ids, None, n - 1, out=ids)
        candidates = x[ids]
        e = clock()
        table = distances(candidates)
        f = clock()
        if step in CAPTURE_STEPS:
            captured[step] = float(np.mean(table < closest))
        g = clock()
        np.minimum(closest, table, out=table)
        h = clock()
        potentials = table @ weights.reshape(-1, 1)
        best = np.argmin(potentials)
        potential, closest = potentials[best], table[best]
        indices[step] = ids[best]
        i = clock()
        seconds["random draws"] += b - a
        seconds["prefix sums"] += c - b
        seconds["search, gather candidates"] += e - c
        seconds["minimum with closest"] += h - g
        seconds["potentials, best candidate"] += i - h
        start += g - f
    total = clock() - start
    return indices, {"total": total, "trials": trials, "seconds": dict(seconds), "captured": captured}


def screen_profile(x, k, seed):
    import mlx.core as mx

    from mlxlearn.cluster import _plusplus

    data = mx.array(x)
    host = np.asarray(data)
    _plusplus.plusplus_indices(data, host, max(64, k // 16), np.random.RandomState(seed))
    profile = {}
    start = time.perf_counter()
    indices = _plusplus.plusplus_indices(data, host, k, np.random.RandomState(seed), profile=profile)
    return indices, {"total": time.perf_counter() - start, **profile}


def table(title, stages, total, steps):
    print(f"\n{title}: {total:.2f} s, {1e3 * total / steps:.3f} ms per center\n")
    print("| Stage | Seconds | Share | Microseconds per center |")
    print("|---|---:|---:|---:|")
    for name, value in sorted(stages.items(), key=lambda item: -item[1]):
        print(f"| {name} | {value:.3f} | {100 * value / total:.1f}% | {1e6 * value / steps:.0f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["reference", "screen"])
    parser.add_argument("--centers", type=int, nargs="+", default=[1024])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--verify", action="store_true", help="also run the helper and compare indices")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    x = (load("MNIST")[0] / 255.0).astype(np.float32)
    records = []
    for k in args.centers:
        runner = reference_profile if args.mode == "reference" else screen_profile
        indices, record = runner(x, k, args.seed)
        record.update(mode=args.mode, centers=k, seed=args.seed, n_samples=len(x), n_features=x.shape[1])
        if args.verify:
            record["same_as_helper"] = bool(np.array_equal(indices, kmeans_plusplus(x, k, random_state=args.seed)[1]))
        records.append(record)
        if args.mode == "reference":
            table(f"scikit-learn helper, {k} centers, {record['trials']} candidates per center",
                  record["seconds"], record["total"], k - 1)
            print("\nShare of (candidate, row) pairs nearer than the row's closest center:",
                  ", ".join(f"step {s}: {v:.5f}" for s, v in record["captured"].items()))
        else:
            stages = {name: record[name] for name in ("prepare", "project", "draw", "screen", "collect",
                                                     "evaluate", "select")}
            table(f"Metal screen, {k} centers, {record['trials']} candidates per center",
                  stages, record["total"], k - 1)
            print("\n" + ", ".join(f"{name}: {record[name]}" for name in (
                "certified", "pairs", "rows", "dense_steps", "contests", "reference_chunks",
                "projected_steps", "second_prefix", "unsure_rows")))
        if "same_as_helper" in record:
            print(f"same indices as sklearn.cluster.kmeans_plusplus: {record['same_as_helper']}")
    if args.out:
        args.out.write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
