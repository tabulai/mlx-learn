"""Benchmark OMeans on MNIST-family images against EMeans, KMeans and scikit-learn.

One dataset and one centroid count per process. Pixels are scaled to [0, 1] in
float32; models train on the 60,000-row training split and are scored on the
held-out split by exact nearest-centroid assignment in float64.

Two things are measured.

Training from resident inputs. ``EMeans(recipe="faiss")`` at 1, 2 and 3 epochs,
``OMeans`` with the two-level search at nine epoch budgets, ``OMeans`` with a
full search (the same loop, as a control), and ``OMeans`` with its default
``assignment="auto"``. The timed call is ``train_device``. All configurations
of a seed are warmed up and then timed in rounds, each round in a freshly
drawn order, so that they share the machine's state; the median over rounds is
reported. The public ``fit`` of ``EMeans`` and ``OMeans`` on a NumPy matrix,
with default epochs and the final label pass, takes turns in the same rounds.

Reference fits, once each with the first seed: mlxlearn's Metal ``KMeans`` with
default settings and with random initialization, and scikit-learn's
``MiniBatchKMeans`` and ``KMeans`` with default settings. They run after the
timed rounds because scikit-learn loads every CPU core.

    MNIST_FAMILY_ROOT=/path/to/datasets python omeans.py --dataset MNIST --k 4096 \\
        --out /tmp/mnist_omeans_k4096.json.gz
"""

import argparse
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import sklearn
from data import SPLITS, load, write_json
from sklearn.cluster import KMeans as SkKMeans
from sklearn.cluster import MiniBatchKMeans as SkMiniBatchKMeans

from mlxlearn.cluster import EMeans, KMeans, OMeans

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks/e_means"))
from omeans_compare import settle, sources, thermal_pressure, worst_pressure  # noqa: E402

EMEANS_BUDGETS = (1, 2, 3)
OMEANS_BUDGETS = (0.8, 1, 1.6, 2, 2.5, 3.2, 4, 5, 8)
DENSE_BUDGETS = (1, 2)
REFERENCES = ("mlx-kmeans", "mlx-kmeans-random", "sklearn-minibatch", "sklearn-kmeans")


def measured_sources():
    """Checksums of the mlxlearn sources this process imported."""
    return {name: digest for name, digest in sources().items() if name.startswith("src/")}


def held_out_mse(centers, rows, chunk=2000):
    """Mean squared distance to the nearest centroid, computed in float64."""
    centers = np.asarray(centers, dtype=np.float64)
    norms = np.einsum("ij,ij->i", centers, centers)
    total = 0.0
    for start in range(0, len(rows), chunk):
        x = rows[start:start + chunk].astype(np.float64)
        distances = np.einsum("ij,ij->i", x, x)[:, None] - 2.0 * (x @ centers.T) + norms
        total += np.maximum(distances.min(axis=1), 0.0).sum()
    return float(total / len(rows))


def configurations(k, seed):
    """Every timed configuration of one seed: (kind, epochs) -> (model, uses the public fit)."""
    shared = {"random_state": seed, "compute_labels": False}
    models = {}
    for epochs in EMEANS_BUDGETS:
        models[("emeans", epochs)] = (
            EMeans(k, max_iter=epochs, recipe="faiss", batch_law="throughput", **shared), False)
    for epochs in OMEANS_BUDGETS:
        models[("omeans", epochs)] = (OMeans(k, max_iter=epochs, assignment="two_level", **shared), False)
    for epochs in DENSE_BUDGETS:
        models[("omeans-dense", epochs)] = (OMeans(k, max_iter=epochs, assignment="dense", **shared), False)
    for epochs in EMEANS_BUDGETS:
        models[("omeans-auto", epochs)] = (OMeans(k, max_iter=epochs, **shared), False)
    # The public estimators as a user calls them: default epochs, labels included.
    models[("fit-emeans", 3)] = (EMeans(k, recipe="faiss", batch_law="throughput", random_state=seed), True)
    models[("fit-omeans", 3)] = (OMeans(k, random_state=seed), True)
    return models


def measure_seed(x, resident, validation, k, seed, warmups, rounds, limit):
    slots = []
    for (kind, epochs), (model, public) in configurations(k, seed).items():
        slot = {"kind": kind, "epochs": epochs, "model": model, "public": public,
                "prepared": None if public else model.prepare_device(resident),
                "seconds": [], "pressures": [], "waited": 0.0, "result": None}
        slots.append(slot)

    def call(slot):
        if slot["public"]:
            slot["model"].fit(x)
        else:
            slot["result"] = slot["model"].train_device(slot["prepared"])
        mx.synchronize()

    for _ in range(warmups):
        for slot in slots:
            call(slot)
    for index in range(rounds):
        order = list(range(len(slots)))
        random.Random(seed * 1_000_003 + index).shuffle(order)
        for position in order:
            slot = slots[position]
            paused, before = settle(limit)
            slot["waited"] += paused
            mx.synchronize()
            start = time.perf_counter()
            call(slot)
            slot["seconds"].append(time.perf_counter() - start)
            slot["pressures"].append(worst_pressure(before, thermal_pressure()))

    records = []
    for slot in slots:
        model, result = slot["model"], slot["result"]
        centers = model.cluster_centers_ if slot["public"] else np.asarray(result.centers)
        recipe = model.recipe_
        record = {
            "kind": slot["kind"], "epochs": slot["epochs"],
            "seconds": slot["seconds"], "median_seconds": statistics.median(slot["seconds"]),
            "thermal_pressure": worst_pressure(*slot["pressures"]), "settle_seconds": slot["waited"],
            "held_out_mse": held_out_mse(centers, validation),
            "batch_size": recipe["batch_size"], "search": recipe.get("search"),
            "search_reason": recipe.get("search_reason"),
            "search_fallback_step": recipe.get("search_fallback_step"),
            "n_cells": recipe.get("n_cells"), "n_probe": recipe.get("n_probe"),
            "expected_fraction": recipe.get("expected_fraction"),
        }
        if not slot["public"]:
            history = list(result.history)
            evaluated = sum(entry.get("evaluated_distances", entry["rows"] * k) for entry in history)
            record.update(steps=result.steps, splits=int(result.split_count.item()),
                          evaluated_fraction=float(evaluated / (sum(e["rows"] for e in history) * k)))
        records.append(record)
    return records


def reference_fits(x, validation, k, seed, names):
    """Default-settings fits of the full-batch and scikit-learn estimators, once each.

    mlxlearn's ``KMeans`` is also fitted with random initialization, because its
    default k-means++ initialization dominates its time at these centroid counts.
    """
    candidates = {
        "mlx-kmeans": lambda: KMeans(k, random_state=seed),
        "mlx-kmeans-random": lambda: KMeans(k, init="random", random_state=seed),
        "sklearn-minibatch": lambda: SkMiniBatchKMeans(k, random_state=seed),
        "sklearn-kmeans": lambda: SkKMeans(k, random_state=seed),
    }
    if any(name.startswith("mlx") for name in names):
        # Compile the Lloyd kernels without paying for a full fit.
        KMeans(k, init="random", random_state=seed, max_iter=2).fit(x)
    out = {}
    for name, build in candidates.items():
        if name not in names:
            continue
        mx.synchronize()
        start = time.perf_counter()
        model = build().fit(x)
        mx.synchronize()
        seconds = time.perf_counter() - start
        out[name] = {"seconds": [seconds], "median_seconds": seconds, "n_iter": int(model.n_iter_),
                     "held_out_mse": held_out_mse(model.cluster_centers_, validation)}
        print(f"  {name}: {seconds:.1f} s, {model.n_iter_} iterations, "
              f"held-out MSE {out[name]['held_out_mse']:.4f}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=list(SPLITS), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--settle", type=float, default=120.0,
                        help="longest wait, in seconds, for nominal thermal pressure before a timed call")
    parser.add_argument("--references", nargs="*", choices=REFERENCES, default=list(REFERENCES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train, _, held_out, _ = load(args.dataset)
    x = np.ascontiguousarray(train / 255.0, dtype=np.float32)
    validation = np.ascontiguousarray(held_out / 255.0, dtype=np.float32)
    resident = mx.array(x)
    mx.eval(resident)
    captured = measured_sources()

    seeds = {}
    for seed in args.seeds:
        seeds[str(seed)] = measure_seed(x, resident, validation, args.k, seed,
                                        args.warmups, args.rounds, args.settle)
        by_kind = {(r["kind"], r["epochs"]): r for r in seeds[str(seed)]}
        for epochs in EMEANS_BUDGETS:
            emeans, auto = by_kind[("emeans", epochs)], by_kind[("omeans-auto", epochs)]
            print(f"{args.dataset} k={args.k} seed {seed} T={epochs}: e-means "
                  f"{1e3 * emeans['median_seconds']:.1f} ms, o-means {1e3 * auto['median_seconds']:.1f} ms "
                  f"({emeans['median_seconds'] / auto['median_seconds']:.2f}x, {auto['search']}, "
                  f"{auto['evaluated_fraction']:.2f} of the distances), error "
                  f"{auto['held_out_mse'] / emeans['held_out_mse'] - 1:+.2%}", flush=True)
    if measured_sources() != captured:
        raise RuntimeError("Measured source changed during the measurement")
    references = reference_fits(x, validation, args.k, args.seeds[0], args.references)

    info = mx.device_info()
    write_json(args.out, {
        "meta": {
            "dataset": args.dataset, "k": args.k, "train_shape": list(x.shape),
            "held_out_shape": list(validation.shape), "seeds": args.seeds,
            "warmups": args.warmups, "rounds": args.rounds, "reference_seed": args.seeds[0],
            "device": {key: info[key] for key in ("device_name", "architecture", "memory_size")},
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "mlx": mx.__version__, "scikit_learn": sklearn.__version__,
            "source_sha256": captured,
            "timing": "interleaved rounds; train_device from resident inputs, public fit from NumPy",
        },
        "seeds": seeds, "references": references,
    })


if __name__ == "__main__":
    main()
