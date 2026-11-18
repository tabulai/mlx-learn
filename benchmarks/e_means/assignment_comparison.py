"""Benchmark fixed-center assignment and public clustering fits from two git revisions.

Run python -m benchmarks.e_means.assignment_comparison --help for dataset and size
options. Real fixtures use the same cached, scaled vectors as the repository's
10M comparison. All timings and paired numerical comparisons are saved to JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Workers run with -I, which omits both the checkout and the current directory.
sys.path.insert(0, str(ROOT))
from benchmarks import knn_integration as sources  # noqa: E402
from benchmarks.e_means import run as protocol  # noqa: E402


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def fixture(args, folder):
    rng = np.random.RandomState(args.seed)
    if args.dataset in protocol.CORE_10M:
        data, _, _ = protocol.reference_modules(args.faiss_source, args.cache)
        x, test, metadata = protocol.load_dataset(args.dataset, data, args.cache, args.rows)
        if len(x) != args.rows:
            raise ValueError(f"Requested {args.rows} rows, got {len(x)}")
        init = x[rng.permutation(len(x))[:args.centers]].copy()
    else:
        metadata = {"distribution": args.dataset}
        shape = (args.rows, args.features)
        if args.dataset == "separated":
            init = rng.normal(0, 4, (args.centers, args.features)).astype(np.float32)
            x = init[np.arange(args.rows) % args.centers] + rng.normal(0, 0.25, shape).astype(np.float32)
            test = init[rng.randint(args.centers, size=512)] + rng.normal(0, 0.25, (512, args.features))
        elif args.dataset == "ties":
            x = rng.randint(-2, 3, shape).astype(np.float32)
            init = x[rng.permutation(len(x))[:args.centers]].copy()
            init[1::2] = init[:args.centers - 1:2]
            test = rng.randint(-2, 3, (512, args.features))
        else:
            x = rng.normal(size=shape).astype(np.float32)
            init = x[rng.permutation(len(x))[:args.centers]].copy()
            test = rng.normal(size=(512, args.features))
    test = np.asarray(test[:512], dtype=np.float32)
    metadata.update({"shape": list(x.shape), "centers": len(init), "holdout_rows": len(test)})
    for name, arr in (("data", x), ("initial", init), ("holdout", test)):
        np.save(folder / f"{name}.npy", arr)
        metadata[f"{name}_sha256"] = digest(arr)
    return metadata


def array_comparison(current, baseline):
    """Compare all labels, centers and direct squared distances, preserving bits."""
    result = {}
    for key, value in current.items():
        other = baseline[key]
        if value.dtype != other.dtype or value.shape != other.shape:
            raise ValueError(f"Different shape/dtype for {key}")
        different = np.count_nonzero(value != other)
        record = {"bit_exact": digest(value) == digest(other), "different": int(different),
                  "elements": int(value.size)}
        if np.issubdtype(value.dtype, np.floating):
            # Bound CPU scratch even for full-scale fit centers.
            worst = 0.0
            for start in range(0, len(value), 8192):
                a = value[start:start + 8192].astype(np.float64)
                b = other[start:start + 8192].astype(np.float64)
                worst = max(worst, float(np.max(np.abs(a - b))))
            record["max_abs_difference"] = worst
        result[key] = record
    return result


def oracle(x, centers, labels):
    """Independently check a spread of rows using direct float64 squared L2."""
    rows = np.linspace(0, len(x) - 1, min(8, len(x)), dtype=np.int64)
    expected = []
    for row in rows:
        point = x[row].astype(np.float64)
        squared = np.empty(len(centers), dtype=np.float64)
        for j in range(0, len(centers), 2048):
            delta = centers[j:j + 2048].astype(np.float64) - point
            squared[j:j + len(delta)] = np.einsum("ij,ij->i", delta, delta)
        expected.append(int(np.argmin(squared)))
    return {"rows": rows.tolist(), "expected": expected, "observed": labels[rows].tolist()}


def worker(request):
    sys.path.insert(0, str(Path(request["snapshot"]) / "src"))
    import mlx.core as mx

    from mlxlearn._kernels.e_means import assign
    from mlxlearn.cluster import EMeans, KMeans

    folder = Path(request["folder"])
    x, init, test = (np.load(folder / f"{v}.npy", mmap_mode="r") for v in ("data", "initial", "holdout"))
    op, params = request["operation"], request["config"]
    if op == "assign":
        xd, cd = mx.array(x), mx.array(init)
        mx.eval(xd, cd)
    # Warm the same entry point and shape before each measured repetition.
    for trial in range(request["warmups"] + 1):
        model = None
        if op != "assign":
            kwargs = {"n_clusters": len(init), "init": init,
                      "random_state": params["seed"], "max_iter": params["iterations"]}
            model = (KMeans(**kwargs, n_init=1, tol=0) if op == "kmeans"
                     else EMeans(**kwargs, recipe="faiss", batch_law="quality"))
        mx.synchronize()
        mx.reset_peak_memory()
        active = mx.get_active_memory()
        start = time.perf_counter()
        if op == "assign":
            labels, distances = assign(xd, cd)
            mx.eval(labels, distances)
        else:
            model.fit(x)
        elapsed = time.perf_counter() - start
        peak = mx.get_peak_memory()
        if trial < request["warmups"]:
            if op == "assign":
                del labels, distances
            del model
    device = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    result = {"seconds": elapsed, "mlx_active_before": active, "mlx_peak_active": peak,
              "device": {k: device.get(k) for k in ("device_name", "architecture", "memory_size")},
              "mlx": mx.__version__}
    if op == "assign":
        arrays = {"labels": np.asarray(labels), "squared_distances": np.asarray(distances)}
        result["direct_float64"] = oracle(x, init, arrays["labels"])
    else:
        arrays = {"labels": model.labels_, "centers": model.cluster_centers_}
        predictions = model.predict(test)
        residuals = test.astype(np.float64) - arrays["centers"][predictions].astype(np.float64)
        arrays["holdout_labels"] = predictions
        result["direct_float64_holdout"] = oracle(test, arrays["centers"], predictions)
        result.update({"inertia": model.inertia_, "iterations": model.n_iter_,
                       "history": model.iteration_stats_, "holdout_mse": float(np.square(residuals).sum(1).mean())})
        if op == "kmeans":
            result["convergence_reason"] = model.convergence_reason_
    result["hashes"] = {k: digest(v) for k, v in arrays.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite output")
    result["loaded_sources"] = sources.loaded_sources(Path(request["snapshot"]))
    np.savez(request["arrays"], **arrays)
    Path(request["record"]).write_text(json.dumps(result, allow_nan=False))


def compare(args):
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__},
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sources": {}, "operations": {}, "complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        target = args.output.with_suffix(args.output.suffix + ".tmp")
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        target.replace(args.output)

    with tempfile.TemporaryDirectory(prefix="cluster-assignment-") as tmp:
        folder = Path(tmp)
        report["fixture"] = fixture(args, folder)
        for label, revision in (("main", args.baseline), ("candidate", args.candidate)):
            frozen = sources.snapshot(ROOT, revision, folder / label)
            report["sources"][label] = {k: frozen[k] for k in ("commit", "sha256")}
        for op in args.operations:
            results = report["operations"][op] = {"pairs": []}
            first = np.random.RandomState(args.seed).randint(2)
            for rep in range(args.repeats):
                order = ["main", "candidate"] if (rep + first) % 2 == 0 else ["candidate", "main"]
                pair = {"order": order, "samples": {}}
                for label in order:
                    sample = folder / f"{op}-{rep}-{label}.npz"
                    params = {"folder": str(folder), "snapshot": str(folder / label), "operation": op,
                              "record": str(folder / "record.json"), "arrays": str(sample),
                              "warmups": args.warmups, "config": report["config"]}
                    subprocess.run([sys.executable, "-I", "-B", __file__, "--worker", json.dumps(params)],
                                   check=True, timeout=args.timeout)
                    raw = json.loads((folder / "record.json").read_text())
                    report["sources"][label].setdefault("loaded", {}).update(raw.pop("loaded_sources"))
                    pair["samples"][label] = raw
                with np.load(folder / f"{op}-{rep}-main.npz") as main, np.load(
                    folder / f"{op}-{rep}-candidate.npz") as candidate:
                    pair["comparison"] = array_comparison(candidate, main)
                if rep:
                    with np.load(folder / f"{op}-0-main.npz") as first_main, np.load(
                        folder / f"{op}-{rep}-main.npz") as main:
                        pair["main_repeat_comparison"] = array_comparison(main, first_main)
                results["pairs"].append(pair)
                save()
            for label in ("main", "candidate"):
                results[f"{label}_median_seconds"] = statistics.median(
                    p["samples"][label]["seconds"] for p in results["pairs"])
            results["speedup"] = results["main_median_seconds"] / results["candidate_median_seconds"]
            print(f"{op}: {results['speedup']:.3f}x", flush=True)
        report["complete"] = True
        save()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="main")
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--dataset", choices=[*protocol.CORE_10M, "gaussian", "ties", "separated"],
                        default="deep1b-10M")
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--centers", type=int, default=8192)
    parser.add_argument("--features", type=int, default=96, help="Synthetic fixtures only")
    parser.add_argument("--operations", nargs="+", choices=["assign", "kmeans", "emeans"],
                        default=["assign", "kmeans", "emeans"])
    parser.add_argument("--iterations", type=int, default=5, help="Lloyd cap or e-means epochs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per worker")
    parser.add_argument("--cache", type=Path, default=protocol.DEFAULT_CACHE)
    parser.add_argument("--faiss-source", type=Path, default=ROOT / "tmp/e_means/faiss")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < args.centers or min(args.rows, args.centers, args.features, args.iterations,
                                       args.repeats, args.timeout) < 1 or args.warmups < 0:
        parser.error("Invalid dataset dimensions or repeat/iteration counts")
    if args.output.exists():
        parser.error("Choose a new output file")
    compare(args)


if __name__ == "__main__":
    main()
