"""o-means versus e-means on one Apple GPU: fit time, held-out error, distances.

Each (dataset, precision, seed) runs in a fresh process with the training
matrix resident on the GPU. Every (mode, epoch budget) gets one complete
untimed warm-up fit and then one or three timed ``train_device`` calls; the
median is reported. The timer covers shuffling, training and the closing
mean. It excludes loading, validation, upload, warm-up and scoring; no final
label pass is timed for either method. Held-out error uses the shared scorer:
exact FP32 nearest-centroid assignment and a float64 residual.

Modes:

``metal-emeans``        ``EMeans(recipe="faiss")`` with o-means' batch law.
``metal-omeans``        ``OMeans(assignment="two_level")``.
``metal-omeans-dense``  ``OMeans(assignment="dense")``: the same loop and
                        closing mean with a full search, which separates the
                        effect of the closing mean from that of the search.

Run from a checkout nobody else edits: the campaign records the hash of every
measured source file and each worker refuses to start or finish on a change.

    .venv/bin/python -m benchmarks.e_means.omeans_compare run --seeds 10 11 12
    .venv/bin/python -m benchmarks.e_means.omeans_report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402

KIND = "omeans_comparison_v1"
MODES = ("metal-emeans", "metal-omeans", "metal-omeans-dense")
DEFAULT_PROTOCOL = HERE / "omeans_protocol.json"
DEFAULT_OUT = HERE / "results/omeans_comparison"


def sources():
    """Hashes of the measured implementation, this driver and its protocol."""
    paths = [Path(__file__), HERE / "run.py", DEFAULT_PROTOCOL,
             *sorted((base.REPO / "src/mlxlearn").rglob("*.py")),
             *sorted((base.REPO / "src/mlxlearn_device").rglob("*.py"))]
    return {str(path.resolve().relative_to(base.REPO)): base.file_sha256(path) for path in paths}


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("kind") != KIND:
        raise ValueError("Not an o-means comparison protocol")
    for mode, budgets in protocol["epochs"].items():
        if mode not in MODES or not budgets or any(b <= 0 for b in budgets):
            raise ValueError(f"Invalid epoch budgets for {mode}")
    return protocol


def expected_calls(seconds):
    """Three timed calls for short fits, one for long ones."""
    return 3 if seconds < 2.0 else 1


def record_name(dataset, precision, seed):
    return f"{dataset}-{precision}-s{seed}.json"


def model_for(mode, k, epochs, law, precision, seed):
    from mlxlearn.cluster import EMeans, OMeans

    shared = {"max_iter": epochs, "batch_law": law, "random_state": seed,
              "precision": precision, "compute_labels": False}
    if mode == "metal-emeans":
        return EMeans(k, recipe="faiss", **shared)
    return OMeans(k, assignment="two_level" if mode == "metal-omeans" else "dense", **shared)


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
    # One resident copy serves every fit; each fit prepares its own recipe,
    # initial centroids and assignment origin from it.
    resident = mx.array(x)
    mx.eval(resident)
    fits = []
    for mode in config["modes"]:
        for epochs in config["epochs"][mode]:
            model = model_for(mode, k, epochs, law, precision, seed)
            current = model.prepare_device(resident)
            model.train_device(current)
            times, result = [], None
            while len(times) < expected_calls(times[0] if times else 0.0):
                mx.clear_cache()
                mx.synchronize()
                start = time.perf_counter()
                result = model.train_device(current)
                mx.synchronize()
                times.append(time.perf_counter() - start)
            centers = np.asarray(result.centers)
            history = list(result.history)
            evaluated = sum(record.get("evaluated_distances", record["rows"] * k) for record in history)
            fit = {
                "mode": mode, "epochs": epochs, "seconds": float(np.median(times)),
                "seconds_all": times, "timed_calls": len(times),
                "test_mse": base.mse(centers, test),
                "train_sample_mse": base.mse(
                    centers, x[np.linspace(0, n - 1, min(n, 10_000)).astype(int)]),
                "evaluated_distances": float(evaluated),
                "evaluated_fraction": float(evaluated / (sum(r["rows"] for r in history) * k)),
                "steps": result.steps, "splits": int(result.split_count.item()),
                "recipe": {key: value for key, value in model.recipe_.items()
                           if isinstance(value, (str, int, float, bool, list, type(None)))},
            }
            fits.append(fit)
            print(f"{config['dataset']} {precision} s{seed} {mode} T={epochs}: "
                  f"{fit['seconds']:.3f}s mse {fit['test_mse']:.6g} "
                  f"fraction {fit['evaluated_fraction']:.3f}", flush=True)
            del model, current, result
    if sources() != captured:
        raise RuntimeError("Measured source changed during the measurement")
    info = mx.device_info()
    base.atomic_json(config["output"], {
        "kind": KIND, "dataset": config["dataset"], "n": n, "d": d, "k": k,
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
    protocol = load_protocol(args.protocol)
    entries = {entry["dataset"]: entry for entry in protocol["configurations"]}
    names = args.datasets or list(entries)
    unknown = [name for name in names if name not in entries]
    if unknown:
        raise SystemExit(f"Not in the protocol: {unknown}")
    captured = sources()
    args.out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    environment.pop("VECLIB_MAXIMUM_THREADS", None)
    for index, name in enumerate(names):
        # Alternate which method runs first, so neither always pays start-up.
        modes = [mode for mode in MODES if mode in args.modes]
        if index % 2:
            modes.reverse()
        for precision in args.precision:
            for seed in args.seeds:
                output = args.out / record_name(name, precision, seed)
                if output.exists() and not args.force:
                    existing = json.loads(output.read_text())
                    if existing.get("sources") == captured:
                        print(f"skip {output.name}: already measured from these sources", flush=True)
                        continue
                config = {
                    "dataset": name, "k": args.k or entries[name]["k"], "rows": args.rows,
                    "precision": precision, "seed": seed, "modes": modes,
                    "epochs": {mode: protocol["epochs"][mode] for mode in modes},
                    "threads": args.threads, "cache": str(args.cache),
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
    campaign = commands.add_parser("run")
    campaign.add_argument("--datasets", nargs="+", help="default: every protocol dataset, in order")
    campaign.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    campaign.add_argument("--precision", nargs="+", choices=["float32", "bfloat16"],
                          default=["float32", "bfloat16"])
    campaign.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    campaign.add_argument("--rows", type=int, help="diagnostic subset of the training rows")
    campaign.add_argument("--k", type=int, help="diagnostic override of the protocol's k")
    campaign.add_argument("--threads", type=int, default=8)
    campaign.add_argument("--timeout", type=float, default=4 * 3600.0)
    campaign.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    campaign.add_argument("--cache", type=Path, default=base.DEFAULT_CACHE)
    campaign.add_argument("--faiss-source", type=Path, default=base.REPO / "tmp/e_means/faiss")
    campaign.add_argument("--out", type=Path, default=DEFAULT_OUT)
    campaign.add_argument("--force", action="store_true")
    measurement = commands.add_parser("worker")
    measurement.add_argument("config")
    args = parser.parse_args()
    if args.command == "worker":
        worker(json.loads(args.config))
    else:
        run(args)


if __name__ == "__main__":
    main()
