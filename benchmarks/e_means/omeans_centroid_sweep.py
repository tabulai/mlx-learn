"""Where the two-level search starts to pay: o-means against e-means over k.

For each centroid count the same fit is run as ``EMeans(recipe="faiss")``, as
``OMeans(assignment="two_level")`` and as ``OMeans(assignment="auto")``, from
one resident training matrix. This is the measurement behind the
``assignment="auto"`` rule: it shows the smallest k at which the two-level
search is faster on a GPU, and what ``auto`` chooses there.

Each (dataset, k) runs in a fresh process. Within a configuration the three
models are warmed up with a full fit and then timed in turn, round robin, so
that drift during the run affects them equally; the reported time is the
median of the timed calls. Each call waits for nominal macOS thermal pressure
and the worst level seen is recorded with the row (see ``omeans_compare``).

    .venv/bin/python -m benchmarks.e_means.omeans_centroid_sweep run \
        --datasets spacev-1M imagenet-clip-512-normalized --output sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from . import run as base
    from .omeans_compare import settle, thermal_pressure, worst_pressure
except ImportError:  # executed as a script
    from omeans_compare import settle, thermal_pressure, worst_pressure

    import run as base

DEFAULT_K = (2176, 3072, 4096, 4224, 6144, 8192, 12288)
MODELS = ("emeans", "two_level", "auto")


def model_for(name, k, epochs, precision, seed):
    from mlxlearn.cluster import EMeans, OMeans

    shared = {"max_iter": epochs, "random_state": seed, "precision": precision, "compute_labels": False}
    if name == "emeans":
        return EMeans(k, recipe="faiss", batch_law="throughput", **shared)
    return OMeans(k, assignment=name, **shared)


def worker(config):
    import mlx.core as mx
    import numpy as np

    data, _, _ = base.reference_modules(config["faiss_source"], config["cache"])
    x, test, _ = base.load_dataset(config["dataset"], data, config["cache"], None)
    n, d = x.shape
    k, seed = config["k"], config["seed"]
    resident = mx.array(x)
    mx.eval(resident)
    for precision in config["precision"]:
        for epochs in config["budgets"]:
            models = {name: model_for(name, k, epochs, precision, seed) for name in MODELS}
            prepared = {name: model.prepare_device(resident) for name, model in models.items()}
            results, times, pressures = {}, {name: [] for name in MODELS}, []
            for name, model in models.items():
                model.train_device(prepared[name])
            for _ in range(config["repeats"]):
                for name, model in models.items():
                    _, before = settle(config["settle"])
                    mx.clear_cache()
                    mx.synchronize()
                    start = time.perf_counter()
                    results[name] = model.train_device(prepared[name])
                    mx.synchronize()
                    times[name].append(time.perf_counter() - start)
                    pressures.append(worst_pressure(before, thermal_pressure()))
            reference = float(np.median(times["emeans"]))
            error = base.mse(np.asarray(results["emeans"].centers), test)
            row = {"dataset": config["dataset"], "n": n, "d": d, "k": k, "precision": precision,
                   "epochs": epochs, "seed": seed, "device": mx.device_info()["device_name"],
                   "emeans_seconds": reference, "emeans_seconds_all": times["emeans"],
                   "emeans_test_mse": error, "thermal_pressure": worst_pressure(*pressures)}
            for name in ("two_level", "auto"):
                result, recipe = results[name], models[name].recipe_
                rows = sum(record["rows"] for record in result.history)
                seconds = float(np.median(times[name]))
                row[name] = {
                    "seconds": seconds, "seconds_all": times[name], "speedup": reference / seconds,
                    "relative_test_mse": base.mse(np.asarray(result.centers), test) / error - 1,
                    "evaluated_fraction": sum(record["evaluated_distances"]
                                              for record in result.history) / (rows * k),
                    "search": recipe["search"], "search_reason": recipe.get("search_reason"),
                    "expected_fraction": recipe.get("expected_fraction"),
                    "search_fallback_step": recipe.get("search_fallback_step"),
                    "n_cells": recipe["n_cells"], "n_probe": recipe["n_probe"],
                }
            with open(config["output"], "a") as handle:
                handle.write(json.dumps(row) + "\n")
            two, auto = row["two_level"], row["auto"]
            print(f"{config['dataset']} d={d} k={k} {precision} T={epochs:g}: e-means {reference:.3f}s | "
                  f"two-level {two['speedup']:.2f}x, error {two['relative_test_mse']:+.2%}, "
                  f"{two['evaluated_fraction']:.3f} of the distances | auto {auto['speedup']:.2f}x "
                  f"({auto['search']}, fallback {auto['search_fallback_step']}) | "
                  f"thermal {row['thermal_pressure']}", flush=True)
            del models, prepared, results
            mx.clear_cache()


def run(args):
    for dataset in args.datasets:
        for k in args.k:
            config = {"dataset": dataset, "k": k, "seed": args.seed, "precision": args.precision,
                      "budgets": args.budgets, "repeats": args.repeats, "settle": args.settle,
                      "output": str(args.output),
                      "cache": str(args.cache), "faiss_source": str(args.faiss_source)}
            command = [sys.executable, str(Path(__file__).resolve()), "worker", json.dumps(config)]
            if subprocess.run(command, check=False).returncode:
                raise SystemExit(f"{dataset} k={k} failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("run")
    sweep.add_argument("--datasets", nargs="+", required=True)
    sweep.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    sweep.add_argument("--budgets", nargs="+", type=float, default=[1.0, 2.0])
    sweep.add_argument("--precision", nargs="+", choices=["float32", "bfloat16"],
                       default=["float32", "bfloat16"])
    sweep.add_argument("--repeats", type=int, default=3)
    sweep.add_argument("--seed", type=int, default=10)
    sweep.add_argument("--settle", type=float, default=300.0,
                       help="longest wait, in seconds, for nominal thermal pressure before a timed call")
    sweep.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    sweep.add_argument("--faiss-source", type=Path, default=base.REPO / "tmp/e_means/faiss")
    sweep.add_argument("--output", type=Path, required=True)
    measurement = commands.add_parser("worker")
    measurement.add_argument("config")
    args = parser.parse_args()
    if args.command == "worker":
        worker(json.loads(args.config))
    else:
        run(args)


if __name__ == "__main__":
    main()
