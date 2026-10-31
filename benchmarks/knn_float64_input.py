"""Measure checked float64 input conversion and ownership transfer for KNN.

The fixture's float32 values are widened exactly to float64. This isolates the
input-conversion boundary while retaining the same neighbor accuracy reference.
Current and frozen implementations use float32 device arithmetic; sklearn gets
the public float64 inputs and retains its unmodified default thread count.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks/reference/knn-pre-v2-2026-10-28.zip"
OPERATIONS = (
    "fit", "fit_query", "kneighbors", "indices", "predict_uniform", "proba_uniform",
    "regress_uniform", "predict_distance", "proba_distance", "regress_distance",
)


def memory_worker(args):
    """Measure host allocations separately, never inside a latency sample."""
    import tracemalloc

    import mlx.core as mx
    import numpy as np

    import mlxlearn
    from mlxlearn._common.config import tuning_context
    from mlxlearn.neighbors import NearestNeighbors

    if not Path(mlxlearn.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError("Memory worker imported the wrong package")
    data = dict(np.load(args.fixtures / "batch_50000_32_k5.npz"))
    samples = []
    with tuning_context(force_backend="mlx"):
        for _ in range(5):
            model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(data["X"])
        mx.synchronize()
        for _ in range(5):
            del model
            gc.collect()
            tracemalloc.start()
            try:
                model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(data["X"])
                mx.synchronize()
                live, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            distances, indices = model.kneighbors(data["Q"])
            samples.append({
                "live_bytes": live, "peak_bytes": peak,
                "quality_pass": bool(np.array_equal(indices, data["indices"])
                                     and np.allclose(distances, data["distances"],
                                                     atol=1e-6, rtol=1e-6)),
                "backend": model.execution_backend_,
            })
    return {
        "policy": "Five warmed fresh fits in a separate process. Input allocation precedes "
                  "tracing. tracemalloc measures Python/NumPy allocations, not all MLX/Metal "
                  "allocations, GPU memory or process RSS. Query correctness checked after tracing.",
        "samples": samples,
        "median_peak_bytes": statistics.median(s["peak_bytes"] for s in samples),
        "median_live_bytes": statistics.median(s["live_bytes"] for s in samples),
        "quality_pass": all(s["quality_pass"] and s["backend"] == "mlx" for s in samples),
    }


def run(args):
    import numpy as np

    from .knn_pipeline import CASES, Worker, extract_reference, make_fixture, sha256, source_hashes

    case = next(case for case in CASES if case.name == "batch_50000_32_k5")
    names = ["current", "reference", "sklearn_default"]
    report = {
        "schema_version": 1, "case": asdict(case), "dataset_seed": args.seed,
        "input_dtype": "float64", "repeats": args.repeats, "warmup": args.warmup,
        "warmup_min_seconds": args.warmup_seconds,
        "timing_policy": "Audited consecutive calls inside isolated workers; no IPC between "
                         "samples. Each operation/variant block uses the declared count and "
                         "duration warmup. Full public input conversion and output boundaries "
                         "are timed. Variant block order alternates by operation. Every call "
                         "is synchronized; correctness is checked outside timing. Default "
                         "sklearn threads and waiting policy are unchanged.",
        "input_policy": "Generate the existing float32 fixture, then widen only X and Q to "
                        "float64 exactly, preserving values, targets and reference outputs. "
                        "Both implementations cast back to float32; sklearn computes on float64.",
        "fit_policy": "fit_query constructs and queries a fresh model on every call, including "
                      "conversion, owned storage and all deferred preparation.",
        "variants": {name: {} for name in names},
    }
    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-f64-") as folder:
        folder = Path(folder)
        reference, fixtures = folder / "reference", folder / "fixtures"
        fixtures.mkdir()
        report["reference"] = extract_reference(args.reference_archive, reference)
        report["original_fixture"] = make_fixture(case, args.seed, fixtures)
        path = fixtures / f"{case.name}.npz"
        data = dict(np.load(path))
        data["X"] = data["X"].astype(np.float64)
        data["Q"] = data["Q"].astype(np.float64)
        np.savez(path, **data)
        report["float64_fixture_sha256"] = sha256(path.read_bytes())
        workers = {}
        try:
            for name in names:
                workers[name] = Worker(name, reference if name == "reference" else ROOT / "src", fixtures)
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            for name in names:
                report["variants"][name]["cold"] = workers[name].request({
                    "op": "cold", "case": asdict(case),
                })
            for index, operation in enumerate(OPERATIONS):
                order = names if index % 2 == 0 else list(reversed(names))
                for name in order:
                    report["variants"][name][operation] = workers[name].request({
                        "op": "block", "operation": operation, "case": asdict(case),
                        "repeats": args.repeats, "warmup": args.warmup,
                        "warmup_seconds": args.warmup_seconds,
                    })
                print(operation, {name: round(report["variants"][name][operation]["median_ms"], 3)
                                  for name in names}, flush=True)
            report["source_integrity_pass"] = all(
                worker.request({"op": "integrity"}) == worker.info["source_hashes"]
                for worker in workers.values()
            )
        finally:
            for worker in workers.values():
                worker.close()
        report["quality_by_variant"] = {
            name: values["cold"]["audit"]["pass"] and all(
                values[operation]["quality_pass"]
                and values[operation]["warmup_quality_pass"] for operation in OPERATIONS
            ) for name, values in report["variants"].items()
        }
        if args.memory:
            report["host_memory"] = {}
            for name, source in (("current", ROOT / "src"), ("reference", reference)):
                env = os.environ.copy()
                env["PYTHONPATH"] = str(source)
                worker = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--memory-worker", name,
                     "--fixtures", str(fixtures), "--source-root", str(source)],
                    env=env, capture_output=True, text=True, check=True,
                )
                report["host_memory"][name] = json.loads(worker.stdout)
        report["source_integrity_pass"] = report["source_integrity_pass"] and all(
            source_hashes(reference if name == "reference" else ROOT / "src")
            == workers[name].info["source_hashes"] for name in names
        )
    report["quality_pass"] = all(report["quality_by_variant"].values())
    report["summary"] = {}
    for operation in OPERATIONS:
        times = {name: value[operation]["median_ms"] for name, value in report["variants"].items()}
        report["summary"][operation] = {
            "median_ms": times,
            "speedup_vs_previous": times["reference"] / times["current"],
            "speedup_vs_sklearn_default": times["sklearn_default"] / times["current"],
        }
    report["harness_sha256"] = {name: sha256(Path(__file__).with_name(name).read_bytes())
                                 for name in ("knn_float64_input.py", "knn_pipeline.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    memory_ok = all(value["quality_pass"] for value in report.get("host_memory", {}).values())
    print(f"quality_pass={report['quality_pass']} source_integrity_pass="
          f"{report['source_integrity_pass']} memory_quality_pass={memory_ok}: {args.output}")
    return 0 if report["quality_pass"] and report["source_integrity_pass"] and memory_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-float64-input-2026-10-28.json")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-worker", choices=["current", "reference"])
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.memory_worker:
        print(json.dumps(memory_worker(args)))
        return 0
    if (args.repeats < 1 or args.warmup < 0 or not math.isfinite(args.warmup_seconds)
            or args.warmup_seconds < 0):
        parser.error("require positive repeats and nonnegative warmups")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
