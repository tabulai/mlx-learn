"""o-means versus e-means with the fits timed in turn, for a GPU that throttles.

``omeans_compare`` times one configuration after another. That is sound on a
machine whose fit time stays flat under load, and it is how the reference
campaign was measured. A laptop is different: sustained load slows its full
search more than its two-level search, so configurations timed minutes apart
are timed in different machine states and the comparison favours o-means.

This driver measures the same models, budgets and protocol, and writes the
same records, but interleaves the timed calls. Every configuration of a
(dataset, precision, seed) is prepared and warmed up with one complete fit.
The timed calls are then made in rounds: each round visits the configurations
once, in an order drawn afresh for the round, so no configuration is always
timed early or late. A configuration whose first timed fit took two seconds or
more is timed once, the others three times, and the median is reported, as in
``omeans_compare``. Held-out error is scored after the last timed call, so
scoring never leaves the GPU idle between two of them.

Each timed call waits for nominal thermal pressure, at most ``--settle``
seconds, and the level before and after the call is recorded. Pressure that
rises during a call is kept, not rejected: the interleaving, not the level, is
what makes the times comparable here.

    .venv/bin/python -m benchmarks.e_means.omeans_interleaved run --seeds 10 11 12
    .venv/bin/python -m benchmarks.e_means.omeans_report \\
        --results benchmarks/e_means/results/omeans_interleaved --output report.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import omeans_compare as campaign  # noqa: E402

import run as base  # noqa: E402

TIMING = "interleaved"
ROUNDS = 3
DEFAULT_OUT = HERE / "results/omeans_interleaved"


def sources():
    """The sequential driver's measured sources, and this driver."""
    captured = campaign.sources()
    captured[str(Path(__file__).resolve().relative_to(base.REPO))] = base.file_sha256(Path(__file__))
    return captured


def round_order(count, seed, index):
    """The order in which round ``index`` visits ``count`` configurations."""
    order = list(range(count))
    random.Random(seed * 1_000_003 + index).shuffle(order)
    return order


def timed_rounds(first_seconds):
    """Rounds in which a configuration is timed, given its first timed fit."""
    return campaign.expected_calls(first_seconds)


def worker(config):
    import mlx.core as mx
    import numpy as np

    import mlxlearn

    captured = sources()
    if config["expected_sources"] != captured:
        raise RuntimeError("Measured source changed since the campaign started")
    import faiss

    faiss.omp_set_num_threads(config["threads"])
    data, _, _ = base.reference_modules(config["faiss_source"], config["cache"])
    x, test, data_meta = base.load_dataset(config["dataset"], data, config["cache"], config["rows"])
    n, d = x.shape
    k, seed, precision = config["k"], config["seed"], config["precision"]
    law = "throughput" if n <= 2_000_000 else "quality"
    resident = mx.array(x)
    mx.eval(resident)
    wake_sample = resident[:4096].astype(mx.float32)

    # Preparation holds no copy of the data, so every configuration can stay
    # prepared for the whole measurement.
    slots = []
    for mode in config["modes"]:
        for epochs in config["epochs"][mode]:
            model = campaign.model_for(mode, k, epochs, law, precision, seed)
            prepared = model.prepare_device(resident)
            model.train_device(prepared)
            slots.append({"mode": mode, "epochs": epochs, "model": model, "prepared": prepared,
                          "times": [], "pressures": [], "visits": [], "waited": 0.0, "result": None})

    for index in range(ROUNDS):
        for rank, position in enumerate(round_order(len(slots), seed, index)):
            slot = slots[position]
            if index >= timed_rounds(slot["times"][0] if slot["times"] else 0.0):
                continue
            paused, before = campaign.settle(config["settle"])
            if paused:
                # The GPU went idle while it cooled: bring it back to its
                # working state before the timer starts.
                slot["waited"] += paused
                until = time.perf_counter() + 0.1
                while time.perf_counter() < until:
                    mx.eval(mx.sum(wake_sample @ wake_sample.T))
            mx.clear_cache()
            mx.synchronize()
            start = time.perf_counter()
            slot["result"] = slot["model"].train_device(slot["prepared"])
            mx.synchronize()
            slot["times"].append(time.perf_counter() - start)
            slot["pressures"].append(campaign.worst_pressure(before, campaign.thermal_pressure()))
            slot["visits"].append([index, rank])

    fits = []
    sample = x[np.linspace(0, n - 1, min(n, 10_000)).astype(int)]
    for slot in slots:
        result, times = slot["result"], slot["times"]
        centers = np.asarray(result.centers)
        history = list(result.history)
        evaluated = sum(record.get("evaluated_distances", record["rows"] * k) for record in history)
        fit = {
            "mode": slot["mode"], "epochs": slot["epochs"], "seconds": float(np.median(times)),
            "seconds_all": times, "timed_calls": len(times), "timed_visits": slot["visits"],
            "thermal_pressure": campaign.worst_pressure(*slot["pressures"]),
            "thermal_pressure_all": slot["pressures"], "settle_seconds": slot["waited"],
            "test_mse": base.mse(centers, test),
            "train_sample_mse": base.mse(centers, sample),
            "evaluated_distances": float(evaluated),
            "evaluated_fraction": float(evaluated / (sum(r["rows"] for r in history) * k)),
            "steps": result.steps, "splits": int(result.split_count.item()),
            "recipe": {key: value for key, value in slot["model"].recipe_.items()
                       if isinstance(value, (str, int, float, bool, list, type(None)))},
        }
        fits.append(fit)
        print(f"{config['dataset']} {precision} s{seed} {fit['mode']} T={fit['epochs']}: "
              f"{fit['seconds']:.3f}s mse {fit['test_mse']:.6g} "
              f"fraction {fit['evaluated_fraction']:.3f} "
              f"thermal {fit['thermal_pressure_all']}"
              + (f" (waited {slot['waited']:.0f}s)" if slot["waited"] else ""), flush=True)
    if sources() != captured:
        raise RuntimeError("Measured source changed during the measurement")
    info = mx.device_info()
    base.atomic_json(config["output"], {
        "kind": campaign.KIND, "timing": TIMING, "rounds": ROUNDS,
        "dataset": config["dataset"], "n": n, "d": d, "k": k,
        "precision": precision, "seed": seed, "batch_law": law, "rows_limit": config["rows"],
        "modes": config["modes"], "fits": fits, "data": data_meta,
        "sources": captured, "versions": base.package_versions(), "host": base.host_identity(),
        "device": {key: info[key] for key in ("device_name", "architecture", "memory_size")},
        "mlxlearn_file": str(Path(mlxlearn.__file__).resolve()),
        "protocol_sha256": base.file_sha256(config["protocol"]),
        "timing_scope": "resident train_device: shuffling, training, closing mean",
        "finished_unix": time.time(),
    })


def run(args):
    protocol = campaign.load_protocol(args.protocol)
    entries = {entry["dataset"]: entry for entry in protocol["configurations"]}
    names = args.datasets or list(entries)
    unknown = [name for name in names if name not in entries]
    if unknown:
        raise SystemExit(f"Not in the protocol: {unknown}")
    captured = sources()
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    environment.pop("VECLIB_MAXIMUM_THREADS", None)
    modes = [mode for mode in campaign.MODES if mode in args.modes]
    for name in names:
        for precision in args.precision:
            for seed in args.seeds:
                output = args.out / campaign.record_name(name, precision, seed)
                if output.exists() and not args.force:
                    existing = json.loads(output.read_text())
                    if existing.get("sources") == captured:
                        print(f"skip {output.name}: already measured from these sources", flush=True)
                        continue
                config = {
                    "dataset": name, "k": args.k or entries[name]["k"], "rows": args.rows,
                    "precision": precision, "seed": seed, "modes": modes,
                    "epochs": {mode: protocol["epochs"][mode] for mode in modes},
                    "threads": args.threads, "settle": args.settle, "cache": str(args.cache),
                    "faiss_source": str(args.faiss_source), "protocol": str(args.protocol),
                    "expected_sources": captured, "output": str(output),
                }
                done = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "worker", json.dumps(config)],
                    env=environment, timeout=args.timeout, check=False)
                if done.returncode:
                    raise SystemExit(f"Worker failed for {output.name}; no record was written")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("run")
    sweep.add_argument("--datasets", nargs="+", help="default: every protocol dataset, in order")
    sweep.add_argument("--modes", nargs="+", choices=campaign.MODES, default=list(campaign.MODES))
    sweep.add_argument("--precision", nargs="+", choices=["float32", "bfloat16"],
                       default=["float32", "bfloat16"])
    sweep.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    sweep.add_argument("--rows", type=int, help="diagnostic subset of the training rows")
    sweep.add_argument("--k", type=int, help="diagnostic override of the protocol's k")
    sweep.add_argument("--threads", type=int, default=8)
    sweep.add_argument("--timeout", type=float, default=4 * 3600.0)
    sweep.add_argument("--settle", type=float, default=30.0,
                       help="longest wait, in seconds, for nominal thermal pressure before a timed call")
    sweep.add_argument("--protocol", type=Path, default=campaign.DEFAULT_PROTOCOL)
    sweep.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    sweep.add_argument("--faiss-source", type=Path, default=base.REPO / "tmp/e_means/faiss")
    sweep.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sweep.add_argument("--force", action="store_true")
    measurement = commands.add_parser("worker")
    measurement.add_argument("config")
    args = parser.parse_args()
    if args.command == "worker":
        worker(json.loads(args.config))
    else:
        run(args)


if __name__ == "__main__":
    main()
