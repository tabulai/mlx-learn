"""Same-process dispatch diagnostic; supplements the isolated public benchmarks.

    python -m benchmarks.knn_review_dispatch \
        --output benchmarks/results/knn-review-dispatch-2026-10-29.json

Force only integer-search eligibility off for the compact comparison, retaining
the public NumPy estimator API. An integer/integer A/A query control estimates
same-process timing noise. Nothing here changes the archived benchmark protocol.
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

from benchmarks.knn_pipeline import ROOT, extract_reference, sha256, source_hashes
from benchmarks.knn_review_distributions import REFERENCE, make_fixture


def harness_hashes():
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes())
        for path in (Path(__file__).resolve(), ROOT / "benchmarks/knn_pipeline.py",
                     ROOT / "benchmarks/knn_review_distributions.py")
    }


@contextlib.contextmanager
def dispatch_route(neighbors, *, compact):
    """Temporarily disable only integer eligibility; always restore the hook."""
    original = neighbors._use_integer_search
    if compact:
        neighbors._use_integer_search = lambda *args, **kwargs: False
    try:
        yield
    finally:
        neighbors._use_integer_search = original


def summary(samples):
    values = [sample["ms"] for sample in samples if "ms" in sample]
    return {
        "median_ms": statistics.median(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "completed_samples": len(values),
        "quality_pass": all(sample.get("audit", {}).get("pass", False) for sample in samples),
        "samples": samples,
    }


def run(args):
    if args.pairs < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        raise ValueError("pairs and warmup_seconds must be positive")
    import numpy as np

    report = {
        "schema_version": 1,
        "status": "running",
        "diagnostic_only": True,
        "platform": platform.platform(),
        "seeds": args.seed or [151, 152, 153],
        "distributions": args.distribution or ["gaussian", "anisotropic"],
        "pairs": args.pairs,
        "warmup_seconds_per_variant": args.warmup_seconds,
        "protocol": (
            "One process, sequential public NumPy estimator calls; GPU completion synchronized. "
            "Compact calls disable only _use_integer_search within a restoring context manager. "
            "Separate fitted query models; fresh fit_query includes deferred preparation. "
            "Each variant receives duration warmup, then paired AB/BA rounds alternate order. "
            "Starting order rotates by seed. Every call is audited outside timing."
        ),
        "purpose": (
            "Diagnose multi-process GPU-context state bias. This is an additional warmed "
            "same-process experiment, not a replacement for the archived isolated benchmark."
        ),
        "query_tolerances": {"atol": 1e-6, "rtol": 1e-6, "indices": "exact equality"},
        "accuracy_reference": "Exhaustive direct float64 distance on rounded float32 input; distance/index tie order.",
        "harness_hashes_before": harness_hashes(),
        "cases": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-dispatch-") as temporary:
        temporary = Path(temporary)
        report["reference"] = extract_reference(args.reference_archive, temporary / "reference")
        fixtures = temporary / "fixtures"
        fixtures.mkdir()
        prepared_cases = []
        for seed_index, seed in enumerate(report["seeds"]):
            for distribution in report["distributions"]:
                destination = fixtures / str(seed) / distribution
                destination.mkdir(parents=True)
                case, fixture = make_fixture(distribution, seed, destination)
                prepared_cases.append((seed_index, case, fixture, destination))

        import mlx.core as mx
        from threadpoolctl import threadpool_info

        sys.path.insert(0, str(ROOT / "src"))
        import mlxlearn
        import mlxlearn_device
        from mlxlearn._common.config import tuning_context
        from mlxlearn.neighbors import NearestNeighbors
        from mlxlearn_device import _neighbors as neighbors
        from mlxlearn_device import _neighbors_integer as integer

        for module in (mlxlearn, mlxlearn_device, neighbors, integer):
            if not Path(module.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
                raise RuntimeError(f"Wrong diagnostic package imported: {module.__file__}")
        report["source_hashes_before"] = source_hashes(ROOT / "src")
        before, frozen = report["source_hashes_before"], report["reference"]["files"]
        report["differences_from_frozen"] = {
            "modified": [name for name in sorted(before.keys() & frozen.keys()) if before[name] != frozen[name]],
            "added": sorted(before.keys() - frozen.keys()),
            "removed": sorted(frozen.keys() - before.keys()),
        }
        compact_key = "mlxlearn_device/_neighbors_compact.py"
        report["compact_matches_frozen_reference"] = before.get(compact_key) == frozen.get(compact_key)
        report["runtime"] = {
            "python": sys.version,
            "numpy": np.__version__,
            "mlx": mx.__version__,
            "threadpools": threadpool_info(),
            "hardware": dict(mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()),
            "integer_supported": bool(integer.is_supported()),
        }
        if not report["runtime"]["integer_supported"]:
            report["status"] = "unsupported"
            save()
            return 2
        original_dispatch = neighbors._use_integer_search
        try:
            with tuning_context(force_backend="mlx"):
                for seed_index, case, fixture, destination in prepared_cases:
                    with np.load(destination / f"{case.name}.npz") as loaded:
                        data = {name: loaded[name] for name in ("X", "Q", "distances", "indices")}
                    entry = {"case": asdict(case), "fixture": fixture, "experiments": {}}

                    def audit(output, data=data):
                        distances, indices = output
                        index_ok = np.array_equal(indices, data["indices"])
                        distance_ok = np.allclose(distances, data["distances"], atol=1e-6, rtol=1e-6)
                        return {"pass": bool(index_ok and distance_ok), "indices_match": bool(index_ok),
                                "distances_match": bool(distance_ok),
                                "distance_max_abs": float(np.max(np.abs(distances - data["distances"])))}

                    for experiment, operation, routes in (
                        ("fit_query_integer_vs_compact", "fit_query", {"integer": False, "compact": True}),
                        ("query_integer_vs_compact", "kneighbors", {"integer": False, "compact": True}),
                        ("query_integer_aa", "kneighbors", {"integer_a": False, "integer_b": False}),
                    ):
                        names = list(routes)
                        models = {
                            name: NearestNeighbors(n_neighbors=case.k, algorithm="brute").fit(data["X"])
                            for name in names
                        } if operation == "kneighbors" else {}
                        last_fresh = {}

                        def call(name, routes=routes, operation=operation, case=case,
                                 data=data, last_fresh=last_fresh, models=models, audit=audit):
                            try:
                                with dispatch_route(neighbors, compact=routes[name]):
                                    mx.synchronize()
                                    started = time.perf_counter()
                                    if operation == "fit_query":
                                        model = NearestNeighbors(n_neighbors=case.k, algorithm="brute").fit(data["X"])
                                        last_fresh[name] = model
                                    else:
                                        model = models[name]
                                    result = model.kneighbors(data["Q"])
                                    mx.synchronize()
                                    elapsed = (time.perf_counter() - started) * 1000
                                    check = audit(result)
                                    index = model._index_.device_index
                                    route = "integer" if index._integer is not None else (
                                        "compact" if index._compact is not None else "direct_or_general"
                                    )
                                    expected_route = "compact" if routes[name] else "integer"
                                    check["route_match"] = route == expected_route
                                    check["pass"] = check["pass"] and check["route_match"]
                                    check["backend_match"] = model.execution_backend_ == "mlx"
                                    check["pass"] = check["pass"] and check["backend_match"]
                                    return {"ms": elapsed, "audit": check, "route": route,
                                            "backend": model.execution_backend_}
                            except Exception:
                                return {"error": traceback.format_exc(), "audit": {"pass": False}}

                        starting_order = names if seed_index % 2 == 0 else list(reversed(names))
                        warmups = {name: [] for name in names}
                        for name in starting_order:
                            started = time.perf_counter()
                            while not warmups[name] or time.perf_counter() - started < args.warmup_seconds:
                                warmups[name].append(call(name))
                        samples, pairs = {name: [] for name in names}, []
                        for pair in range(args.pairs):
                            order = starting_order if pair % 2 == 0 else list(reversed(starting_order))
                            pair_samples = {}
                            for name in order:
                                sample = call(name)
                                samples[name].append(sample)
                                pair_samples[name] = sample
                            numerator, denominator = pair_samples[names[1]], pair_samples[names[0]]
                            pairs.append({
                                "pair": pair, "order": order,
                                "ratio_second_variant_over_first": (
                                    numerator["ms"] / denominator["ms"]
                                    if "ms" in numerator and "ms" in denominator else None
                                ),
                            })
                        ratios = [pair["ratio_second_variant_over_first"] for pair in pairs
                                  if pair["ratio_second_variant_over_first"] is not None]
                        entry["experiments"][experiment] = {
                            "operation": operation,
                            "ratio_definition": f"{names[1]} milliseconds / {names[0]} milliseconds",
                            "paired_ratio_median": statistics.median(ratios) if ratios else None,
                            "paired_ratio_min": min(ratios) if ratios else None,
                            "paired_ratio_max": max(ratios) if ratios else None,
                            "pairs": pairs,
                            "variants": {name: {"retained": summary(samples[name]),
                                                "warmup": summary(warmups[name])} for name in names},
                        }
                    report["cases"].append(entry)
                    save()
                    print(f"{fixture['distribution']} seed {fixture['seed']}: " + json.dumps({
                        name: value["paired_ratio_median"] for name, value in entry["experiments"].items()
                    }), flush=True)
            report["quality_pass"] = all(
                variant[phase]["quality_pass"]
                for case in report["cases"] for experiment in case["experiments"].values()
                for variant in experiment["variants"].values() for phase in ("warmup", "retained")
            )
            report["source_hashes_after"] = source_hashes(ROOT / "src")
            report["source_integrity_pass"] = report["source_hashes_before"] == report["source_hashes_after"]
            report["harness_hashes_after"] = harness_hashes()
            report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
            report["dispatch_restored"] = neighbors._use_integer_search is original_dispatch
            report["status"] = "complete"
        except Exception:
            report["status"] = "error"
            report["error"] = traceback.format_exc()
            raise
        finally:
            neighbors._use_integer_search = original_dispatch
            save()
    return 0 if all(report[key] for key in (
        "quality_pass", "source_integrity_pass", "harness_integrity_pass", "dispatch_restored",
        "compact_matches_frozen_reference",
    )) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--distribution", choices=("gaussian", "anisotropic"), action="append")
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-review-dispatch-2026-10-29.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
