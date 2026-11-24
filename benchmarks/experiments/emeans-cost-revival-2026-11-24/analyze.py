"""Summarize a revival comparison: one rule against the library's mass rule.

Reads the JSON lines written by ``assess.py proto``. ``--mode`` names the
compared fits: ``libcost`` (``EMeans(revival="cost")``), or the prototype
modes ``cost`` and ``massaxis``. For every budget it prints the change in
error at equal epochs and the epoch budget at which the mass rule reaches the
same error, by linear interpolation of its mean error over the budgets run.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/analyze.py \
        results/library_grid.jsonl --mode libcost
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load(paths):
    records = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                r = json.loads(line)
                records[(r["dataset"], r["k"], r["mode"], r["epochs"], r["seed"])] = r
    return records


def equivalent_epochs(curve, target):
    """Epoch budget at which the mass rule's mean error reaches ``target``."""
    points = sorted(curve.items())
    if target >= points[0][1]:
        return f"<={points[0][0]:g}"
    for (e0, m0), (e1, m1) in zip(points, points[1:], strict=False):
        if m1 <= target <= m0 and m0 != m1:
            return f"{e0 + (e1 - e0) * (m0 - target) / (m0 - m1):.2f}"
    return f">{points[-1][0]:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--mode", default="libcost")
    parser.add_argument("--metric", default="test_mse", choices=["test_mse", "train_mse"])
    args = parser.parse_args()
    records = load(args.paths)
    mode, metric = args.mode, args.metric
    overall = defaultdict(list)
    for dataset, k in sorted({(d, k) for d, k, *_ in records}):
        keys = [key for key in records if key[0] == dataset and key[1] == k]
        seeds = sorted({key[4] for key in keys})
        epochs = sorted({key[3] for key in keys})
        mean = defaultdict(dict)
        for name in ("lib", mode):
            for e in epochs:
                values = [records[(dataset, k, name, e, s)][metric] for s in seeds
                          if (dataset, k, name, e, s) in records]
                if len(values) == len(seeds):
                    mean[name][e] = float(np.mean(values))
        first = records[keys[0]]
        print(f"\n== {dataset}  n={first['n']:,} d={first['d']} k={k}  seeds={seeds}  {metric}")
        print("  mass rule, mean error by epochs: "
              + "  ".join(f"{e:g}: {m:.6g}" for e, m in sorted(mean["lib"].items())))
        for e in sorted(mean[mode]):
            if e not in mean["lib"]:
                continue
            ours = [records[(dataset, k, mode, e, s)] for s in seeds]
            theirs = [records[(dataset, k, "lib", e, s)] for s in seeds]
            paired = [a[metric] / b[metric] - 1 for a, b in zip(ours, theirs, strict=True)]
            overall[e].append(100 * float(np.mean(paired)))
            print(f"  {mode} at {e:g} epochs: {mean[mode][e]:.6g}  {100 * np.mean(paired):+.2f}% "
                  f"(seeds {', '.join(f'{100 * p:+.2f}%' for p in paired)})"
                  f" | mass rule needs {equivalent_epochs(mean['lib'], mean[mode][e])} epochs"
                  f" | moves {np.mean([r['splits'] for r in ours]):.0f}"
                  f" vs splits {np.mean([r['splits'] for r in theirs]):.0f}")
    print("\nmean over datasets: " + "  ".join(
        f"{e:g} epochs {np.mean(values):+.2f}%" for e, values in sorted(overall.items())))


if __name__ == "__main__":
    main()
