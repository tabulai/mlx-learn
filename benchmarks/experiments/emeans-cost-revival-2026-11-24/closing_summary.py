"""Tabulate ``closing_study.py`` results.

Held-out error of cost-based revival under each closing rule, in percent
against the mass rule's own closing (rule ``none`` of the mass-mode fit with
the same dataset, seed and budget), averaged over seeds.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/closing_summary.py \
        results/closing_study.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

RULES = ["ema", "none", "post", "mover", "twin", "handover", "searchtwin", "handover+32",
         "none/ema-touched", "fresh"]


def main(path):
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    mass = {(r["dataset"], r["seed"], r["epochs"]): r["rules"]["none"]["test_mse"]
            for r in rows if r["mode"] == "mass"}
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        reference = mass.get((r["dataset"], r["seed"], r["epochs"]))
        if r["mode"] != "cost" or reference is None:
            continue
        for rule in RULES:
            table[(r["epochs"], r["dataset"])][rule].append(
                100 * (r["rules"][rule]["test_mse"] / reference - 1))
    print(f"{'dataset':30s} {'epochs':>6s} " + " ".join(f"{rule:>16s}" for rule in RULES))
    overall = defaultdict(list)
    for (epochs, dataset), values in sorted(table.items()):
        print(f"{dataset:30s} {epochs:6g} " + " ".join(f"{np.mean(values[rule]):+16.2f}" for rule in RULES))
        for rule in RULES:
            overall[(epochs, rule)].append(np.mean(values[rule]))
    for epochs in sorted({e for e, _ in overall}):
        print(f"{'mean over datasets':30s} {epochs:6g} "
              + " ".join(f"{np.mean(overall[(epochs, rule)]):+16.2f}" for rule in RULES))


if __name__ == "__main__":
    main(sys.argv[1])
