"""Cost-based revival where most centroids move: more centroids than separated clusters.

Rows are drawn around Gaussian modes with unit noise. With more centroids than
well-separated modes the surplus centroids pair up inside modes, a paired
centroid is cheap to remove, and many centroids move at every check, up to the
end of training. The embedding datasets of the closing study do not show that
regime, and the closing rule first implemented (the EMA position of every
centroid a closing-pass move involved) failed in it.

For every mixture, budget and seed the script fits ``EMeans`` with both revival
rules and scores the closing means and the EMA centroids on held-out rows.
With ``--rules`` it also trains the prototype loop of ``closing_study.py`` once
per rule set and scores every closing rule on that one trajectory.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/surplus.py \
        --rules --output results/surplus.jsonl
    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/surplus.py \
        --summarize results/surplus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import closing_study  # noqa: E402

from mlxlearn.cluster import EMeans  # noqa: E402

# Modes, centroids, features, training rows, spread of the mode centres in units of the noise.
CASES = [
    (3000, 4096, 32, 200_000, 3.0),
    (1500, 2048, 32, 100_000, 3.0),
    (300, 410, 16, 40_000, 3.0),
    (200, 273, 24, 30_000, 3.0),
    (150, 205, 16, 20_000, 3.0),
    (3000, 4096, 32, 200_000, 1.0),
    (8192, 4096, 32, 200_000, 3.0),
]
HELD_OUT = 20_000
RULES = ("ema", "none", "post", "twin", "handover", "searchtwin", "fresh", "none/ema-touched")


def mixture(modes, d, n, spread):
    rng = np.random.default_rng(1)
    means = rng.normal(size=(modes, d)).astype(np.float32) * spread
    x = means[rng.integers(0, modes, n + HELD_OUT)] + rng.normal(size=(n + HELD_OUT, d)).astype(np.float32)
    return x[:n], x[n:]


def held_out(centers, test):
    """Mean squared distance to the nearest centroid, in float64."""
    c, x = np.asarray(centers, dtype=np.float64), np.asarray(test, dtype=np.float64)
    norms, best = (c * c).sum(axis=1), np.empty(len(x))
    for start in range(0, len(x), 4096):
        part = x[start:start + 4096]
        best[start:start + 4096] = ((part * part).sum(axis=1)[:, None] - 2.0 * part @ c.T + norms[None]).min(axis=1)
    return float(np.maximum(best, 0.0).mean())


def run(args):
    with mx.stream(mx.gpu):
        for modes, k, d, n, spread in CASES:
            train, test = mixture(modes, d, n, spread)
            for epochs in args.epochs:
                for seed in args.seeds:
                    record = {"modes": modes, "k": k, "d": d, "n": n, "spread": spread, "epochs": epochs,
                              "seed": seed, "device": mx.device_info()["device_name"], "mlx": mx.__version__}
                    for revival in ("mass", "cost"):
                        model = EMeans(k, max_iter=epochs, recipe="faiss", random_state=seed, revival=revival,
                                       compute_labels=False).fit(train)
                        record[revival] = {"closing": held_out(model.last_epoch_centers_, test),
                                           "ema": held_out(model.ema_centers_, test),
                                           "moves": model.n_splits_, "batch_size": model.recipe_["batch_size"],
                                           "steps": model.n_steps_}
                    if args.rules:
                        mass, _ = closing_study.fit(train, k, epochs, seed, "mass")
                        cost, detail = closing_study.fit(train, k, epochs, seed, "cost")
                        record["prototype"] = {
                            "mass": held_out(mass["none"][0], test),
                            "rules": {rule: held_out(cost[rule][0], test) for rule in RULES if rule in cost},
                            **{key: detail[key] for key in ("moves", "last_pass_moves", "rows_last")},
                            **{key: detail.get(key, 0) for key in ("touched", "stale", "twin_rows")}}
                    with open(args.output, "a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(f"{modes} modes, {k} centroids, d={d}, spread {spread:g}, {epochs:g} epochs, seed {seed}: "
                          f"cost against mass {100 * (record['cost']['closing'] / record['mass']['closing'] - 1):+.2f}% "
                          f"({record['cost']['moves']} moves)", flush=True)


def summarize(path):
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["modes"], r["k"], r["d"], r["n"], r["spread"], r["epochs"])].append(r)
    for (modes, k, d, n, spread, epochs), group in groups.items():
        closing = np.array([100 * (r["cost"]["closing"] / r["mass"]["closing"] - 1) for r in group])
        ema = np.array([100 * (r["cost"]["ema"] / r["mass"]["closing"] - 1) for r in group])
        print(f"{modes} modes, {k} centroids, d={d}, n={n}, spread {spread:g}, {epochs:g} epochs, {len(group)} seeds, "
              f"batch {group[0]['cost']['batch_size']}, {np.mean([r['cost']['moves'] for r in group]):.0f} moves")
        print(f"    EMeans closing means: {closing.mean():+.2f}% (best {closing.min():+.2f}%, worst {closing.max():+.2f}%, "
              f"{int((closing < 0).sum())} of {len(closing)} seeds favour the cost rule); "
              f"EMA centroids {ema.mean():+.2f}% (worst {ema.max():+.2f}%)")
        if "prototype" in group[0]:
            involved = np.mean([r["prototype"]["touched"] for r in group]) / k
            stale = np.mean([r["prototype"]["stale"] / r["prototype"]["rows_last"] for r in group])
            parts = []
            for rule in RULES:
                values = [100 * (r["prototype"]["rules"][rule] / r["prototype"]["mass"] - 1) for r in group
                          if rule in r["prototype"]["rules"]]
                if values:
                    parts.append(f"{rule} {np.mean(values):+.2f}")
            print(f"    closing rules on the prototype loop ({involved:.0%} of centroids in a closing-pass move, "
                  f"{stale:.1%} stale rows): " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--epochs", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--rules", action="store_true", help="also score every closing rule on the prototype loop")
    parser.add_argument("--output")
    parser.add_argument("--summarize", metavar="RESULTS", help="tabulate a results file and stop")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif not args.output:
        parser.error("--output is required")
    else:
        run(args)


if __name__ == "__main__":
    main()
