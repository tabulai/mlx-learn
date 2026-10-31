"""Isolate the Gaussian cost of the scale guard in one public-API process.

    python3 -m benchmarks.knn_scale_guard_control \
        --output benchmarks/results/knn-scale-guard-control-2026-10-29.json

Both variants use integer search. Only the guard helper is replaced by True for
the control, leaving ownership, lazy preparation and the cached decision intact.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from benchmarks.knn_pipeline import CASES, ROOT, make_fixture, sha256, source_hashes, summarize

NAMES = ("current", "guard_disabled")
OPERATIONS = ("fit_query", "kneighbors")


@contextlib.contextmanager
def guard_policy(neighbors, *, disabled):
    original = neighbors._prefer_integer_training
    if disabled:
        neighbors._prefer_integer_training = lambda train: True
    try:
        yield
    finally:
        neighbors._prefer_integer_training = original


def harness_hashes():
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes())
        for path in (Path(__file__).resolve(), ROOT / "benchmarks/knn_pipeline.py")
    }


def run(args):
    if (args.pairs < 1 or not math.isfinite(args.warmup_seconds)
            or args.warmup_seconds <= 0 or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("Require positive pairs/warmup_seconds and distinct seeds")
    import numpy as np

    case = next(case for case in CASES if case.name == "batch_50000_32_k5")
    report = {
        "schema_version": 1,
        "status": "running",
        "diagnostic_only": True,
        "case": asdict(case),
        "dataset_seeds": args.seeds,
        "pairs": args.pairs,
        "warmup_seconds_per_variant": args.warmup_seconds,
        "platform": platform.platform(),
        "protocol": (
            "One process; synchronized public NumPy estimator calls; fresh fit_query includes "
            "all preparation. Each variant receives duration warmup, then 31 alternating AB/BA "
            "pairs by default. Starting order rotates across seeds. Accuracy, backend, prepared "
            "filter and cached guard decision are checked outside every timer."
        ),
        "control": (
            "Only _prefer_integer_training is temporarily replaced by a helper returning True. "
            "Both variants retain the same cached decision branch and integer search. This "
            "isolates guard evaluation cost, not all differences from an archived implementation."
        ),
        "ratio_definition": "guard_disabled milliseconds / current milliseconds; above 1 favors current",
        "query_tolerances": {"atol": 1e-6, "rtol": 1e-6, "indices": "exact equality"},
        "accuracy_reference": (
            "Existing Gaussian fixture oracle: sklearn brute float64 on float32 inputs, "
            "selected distances recomputed by direct float64 subtraction."
        ),
        "harness_hashes_before": harness_hashes(),
        "cases": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-guard-control-") as temporary:
        fixtures = []
        for seed in args.seeds:
            destination = Path(temporary) / str(seed)
            destination.mkdir()
            fixtures.append((seed, destination, make_fixture(case, seed, destination)))

        import mlx.core as mx

        sys.path.insert(0, str(ROOT / "src"))
        import mlxlearn
        from mlxlearn._common.config import tuning_context
        from mlxlearn.neighbors import NearestNeighbors
        from mlxlearn_device import _neighbors as neighbors
        from mlxlearn_device import _neighbors_integer as integer

        for module in (mlxlearn, neighbors, integer):
            if not Path(module.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
                raise RuntimeError(f"Wrong package imported: {module.__file__}")
        report["runtime"] = {
            "python": sys.version, "numpy": np.__version__, "mlx": mx.__version__,
            "hardware": dict(mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()),
            "integer_supported": bool(integer.is_supported()),
        }
        report["source_hashes_before"] = source_hashes(ROOT / "src")
        if not report["runtime"]["integer_supported"]:
            report["status"] = "unsupported"
            save()
            return 2
        original_guard = neighbors._prefer_integer_training
        try:
            with tuning_context(force_backend="mlx"):
                for seed_index, (seed, destination, fixture) in enumerate(fixtures):
                    with np.load(destination / f"{case.name}.npz") as loaded:
                        data = {name: loaded[name] for name in ("X", "Q", "distances", "indices")}
                    entry = {"dataset_seed": seed, "fixture": fixture, "operations": {}}
                    order = list(NAMES if seed_index % 2 == 0 else reversed(NAMES))
                    for operation in OPERATIONS:
                        models = {
                            name: NearestNeighbors(n_neighbors=case.k, algorithm="brute").fit(data["X"])
                            for name in NAMES
                        } if operation == "kneighbors" else {}
                        last_fresh = {}

                        def call(name, operation=operation, data=data, models=models, last_fresh=last_fresh):
                            with guard_policy(neighbors, disabled=name == "guard_disabled"):
                                mx.synchronize()
                                started = time.perf_counter()
                                if operation == "fit_query":
                                    model = NearestNeighbors(n_neighbors=case.k, algorithm="brute").fit(data["X"])
                                    last_fresh[name] = model
                                else:
                                    model = models[name]
                                distances, indices = model.kneighbors(data["Q"])
                                mx.synchronize()
                                elapsed = (time.perf_counter() - started) * 1000
                                index = model._index_.device_index
                                checks = {
                                    "indices_match": bool(np.array_equal(indices, data["indices"])),
                                    "distances_match": bool(np.allclose(distances, data["distances"], atol=1e-6, rtol=1e-6)),
                                    "backend_match": model.execution_backend_ == "mlx",
                                    "cached_guard_true": index._integer_preferred is True,
                                    "integer_prepared": index._integer is not None,
                                    "compact_unprepared": index._compact is None,
                                }
                                return {
                                    "ms": elapsed,
                                    "audit": {"pass": all(checks.values()), **checks},
                                    "distance_max_abs": float(np.max(np.abs(distances - data["distances"]))),
                                    "guard_decision": index._integer_preferred,
                                    "backend": model.execution_backend_,
                                }

                        warmups, elapsed_warmups = {}, {}
                        for name in order:
                            values = []
                            started = time.perf_counter()
                            while not values or time.perf_counter() - started < args.warmup_seconds:
                                values.append(call(name))
                            elapsed_warmups[name] = time.perf_counter() - started
                            warmups[name] = summarize(values)
                        samples = {name: [] for name in NAMES}
                        pairs = []
                        for pair in range(args.pairs):
                            pair_order = order if pair % 2 == 0 else list(reversed(order))
                            values = {}
                            for name in pair_order:
                                values[name] = call(name)
                                samples[name].append(values[name])
                            pairs.append({
                                "pair": pair, "order": pair_order,
                                "guard_disabled_over_current": values["guard_disabled"]["ms"] / values["current"]["ms"],
                                "current_minus_guard_disabled_ms": values["current"]["ms"] - values["guard_disabled"]["ms"],
                            })
                        ratios = [pair["guard_disabled_over_current"] for pair in pairs]
                        entry["operations"][operation] = {
                            "variants": {name: {
                                "retained": summarize(samples[name]), "warmup": warmups[name],
                                "warmup_elapsed_seconds": elapsed_warmups[name],
                            } for name in NAMES},
                            "pairs": pairs,
                            "median_paired_ratio": statistics.median(ratios),
                            "median_paired_overhead_ms": statistics.median(pair["current_minus_guard_disabled_ms"] for pair in pairs),
                        }
                    report["cases"].append(entry)
                    save()
                    print(f"seed {seed}: " + json.dumps({
                        name: {"paired_ratio": value["median_paired_ratio"],
                               "overhead_ms": value["median_paired_overhead_ms"]}
                        for name, value in entry["operations"].items()
                    }), flush=True)
            report["quality_pass"] = all(
                variant[phase]["quality_pass"]
                for entry in report["cases"] for operation in entry["operations"].values()
                for variant in operation["variants"].values() for phase in ("warmup", "retained")
            )
            report["source_hashes_after"] = source_hashes(ROOT / "src")
            report["source_integrity_pass"] = report["source_hashes_before"] == report["source_hashes_after"]
            report["harness_hashes_after"] = harness_hashes()
            report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
            report["guard_restored"] = neighbors._prefer_integer_training is original_guard
            report["status"] = "complete"
        except Exception:
            report["status"] = "error"
            report["error"] = traceback.format_exc()
            raise
        finally:
            neighbors._prefer_integer_training = original_guard
            save()
    return 0 if all(report[key] for key in (
        "quality_pass", "source_integrity_pass", "harness_integrity_pass", "guard_restored",
    )) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[151, 152, 153])
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-scale-guard-control-2026-10-29.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
