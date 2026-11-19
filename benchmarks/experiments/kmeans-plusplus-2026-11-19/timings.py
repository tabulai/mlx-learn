"""Time k-means++ seeding and whole KMeans fits on MNIST, before and after the Metal screen.

``seeding`` times ``sklearn.cluster.kmeans_plusplus`` against mlxlearn's seeding and checks
that both select the same rows. ``fits`` times ``KMeans(k)`` with default settings, the same
fit with the seeding routed to scikit-learn's helper as before the change (``reference``),
and ``KMeans(k, init="random")``.

Variants run in interleaved rounds. This laptop's GPU throttles under sustained load, so
every timed call first waits for nominal macOS thermal pressure (level 0) and records the
level it started at and the seconds it waited.

    MNIST_FAMILY_ROOT=... python timings.py seeding --rounds 3 --out results/seeding.json
    MNIST_FAMILY_ROOT=... python timings.py fits --rounds 3 --reference-rounds 1 --out results/fits.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from sklearn.cluster import kmeans_plusplus

from mlxlearn.cluster import KMeans, _plusplus

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "mnist-family-2026-11-02"))
from data import load  # noqa: E402


def thermal():
    out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                         capture_output=True, text=True).stdout.split()
    return int(out[-1]) if out else -1


def rested(limit=900.0):
    """Wait for nominal thermal pressure; return the level reached and the seconds waited."""
    start = time.perf_counter()
    while (level := thermal()) > 0 and time.perf_counter() - start < limit:
        time.sleep(5)
    return {"thermal": level, "waited": time.perf_counter() - start}


def seeding(x, centers, rounds, reference_rounds, seed):
    data = mx.array(x)
    host = np.asarray(data)
    _plusplus.plusplus_indices(data, host, 64, np.random.RandomState(seed))
    expected = {}
    for round_index in range(rounds):
        for k in centers:
            if round_index < reference_rounds:
                state = rested()
                start = time.perf_counter()
                _, expected[k] = kmeans_plusplus(x, k, random_state=seed)
                yield {"task": "seeding", "variant": "sklearn", "centers": k, "round": round_index,
                       "seconds": time.perf_counter() - start, **state}
            profile = {}
            state = rested()
            start = time.perf_counter()
            indices = _plusplus.plusplus_indices(data, host, k, np.random.RandomState(seed), profile=profile)
            yield {"task": "seeding", "variant": "mlxlearn", "centers": k, "round": round_index,
                   "seconds": time.perf_counter() - start, **state,
                   "same_indices": bool(np.array_equal(indices, expected[k])), "profile": profile}


def fits(x, centers, rounds, reference_rounds, seed):
    KMeans(64, init="random", random_state=seed, max_iter=2).fit(x)
    KMeans(64, random_state=seed, max_iter=2).fit(x)
    gate = _plusplus._MIN_SCREEN_WORK
    for round_index in range(rounds):
        for k in centers:
            variants = [("default", {}), ("random", {"init": "random"})]
            if round_index < reference_rounds:
                variants.append(("reference", {}))
            for name, kwargs in variants:
                # ``reference`` seeds as before the change: scikit-learn's helper on the host.
                _plusplus._MIN_SCREEN_WORK = float("inf") if name == "reference" else gate
                state = rested()
                start = time.perf_counter()
                try:
                    model = KMeans(k, random_state=seed, **kwargs).fit(x)
                finally:
                    _plusplus._MIN_SCREEN_WORK = gate
                yield {"task": "fit", "variant": name, "centers": k, "round": round_index,
                       "seconds": time.perf_counter() - start, **state,
                       "lloyd_seconds": model.optimization_time_, "n_iter": model.n_iter_,
                       "n_init": model.n_init_, "inertia": model.inertia_}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=["seeding", "fits"])
    parser.add_argument("--centers", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reference-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    x = (load("MNIST")[0] / 255.0).astype(np.float32)
    runner = seeding if args.task == "seeding" else fits
    records = []
    for record in runner(x, args.centers, args.rounds, max(1, args.reference_rounds), args.seed):
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "profile"}), flush=True)
    if args.out:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        args.out.write_text(json.dumps({"device": info.get("device_name"), "records": records}, indent=1))


if __name__ == "__main__":
    main()
