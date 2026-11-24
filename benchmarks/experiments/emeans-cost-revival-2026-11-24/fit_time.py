"""Fit time of cost-based revival against mass-based revival at equal epochs.

Both fits start from resident inputs (``prepare_device``) and are timed with
``train_device``, taking turns so that a laptop's thermal state is shared.
Every timed call waits for nominal thermal pressure first and records the
worst level seen.

    .venv/bin/python benchmarks/experiments/emeans-cost-revival-2026-11-24/fit_time.py \
        --datasets deep1b-1M spacev-1M --output results/fit_time.jsonl
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
from benchmarks.e_means.omeans_compare import settle, thermal_pressure, worst_pressure  # noqa: E402

from mlxlearn.cluster import EMeans  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["deep1b-1M"])
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--epochs", nargs="+", type=float, default=[2.0])
    parser.add_argument("--precision", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--settle", type=float, default=120.0)
    parser.add_argument("--rest", type=float, default=0.0, help="idle seconds between timed calls")
    parser.add_argument("--faiss-source", type=Path, default=assess.FAISS_SOURCE)
    parser.add_argument("--cache", type=Path, default=assess.CACHE)
    parser.add_argument("--output")
    args = parser.parse_args()
    assess.FAISS_SOURCE, assess.CACHE = args.faiss_source, args.cache
    device = mx.device_info()["device_name"]
    with mx.stream(mx.gpu):
        for name in args.datasets:
            x, _ = assess.load(name)
            for precision in args.precision:
                for epochs in args.epochs:
                    models, prepared = {}, {}
                    for revival in ("mass", "cost"):
                        models[revival] = EMeans(args.k, max_iter=epochs, recipe="faiss",
                                                 random_state=args.seed, precision=precision,
                                                 revival=revival, compute_labels=False)
                        prepared[revival] = models[revival].prepare_device(x)
                        models[revival].train_device(prepared[revival])  # compile and warm up
                    times, pressures = {"mass": [], "cost": []}, []
                    for round_index in range(args.rounds):
                        order = ("mass", "cost") if round_index % 2 == 0 else ("cost", "mass")
                        for revival in order:
                            time.sleep(args.rest)
                            _, before = settle(args.settle)
                            result = models[revival].train_device(prepared[revival])
                            times[revival].append(result.seconds)
                            pressures.append(worst_pressure(before, thermal_pressure()))
                    mass, cost = (float(np.median(times[key])) for key in ("mass", "cost"))
                    record = {"dataset": name, "n": len(x), "d": x.shape[1], "k": args.k,
                              "precision": precision, "epochs": epochs, "seed": args.seed,
                              "device": device, "mlx": mx.__version__, "rounds": args.rounds,
                              "mass_seconds": times["mass"], "cost_seconds": times["cost"],
                              "mass_median": mass, "cost_median": cost, "ratio": cost / mass,
                              "thermal_pressure": worst_pressure(*pressures)}
                    print(f"{name} {precision} epochs={epochs:g}: mass {mass:.3f} s, cost {cost:.3f} s, "
                          f"cost/mass {cost / mass:.3f} (thermal pressure {record['thermal_pressure']})",
                          flush=True)
                    if args.output:
                        with open(args.output, "a") as handle:
                            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
