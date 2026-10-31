"""Audited 128-feature kNN comparisons with frozen v3 and precision baselines.

    python3 -m benchmarks.knn_128_control --output /tmp/knn-128-control.json

Four persistent workers isolate current, a current-code A/A control, frozen v3,
and the preceding precision implementation. All GPU calls run sequentially.
Continuous per-worker warmup precedes interleaved, individually audited samples.
"""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import tempfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .knn_pipeline import ROOT, Case, Worker, extract_reference, make_fixture, sha256, source_hashes

VARIANTS = ("current", "current_b", "v3", "precision")
OPERATIONS = ("kneighbors", "fit_query")
DEFAULT_SEEDS = (45, 46, 47, 145, 245, 345)
V3 = ROOT / "benchmarks/reference/knn-pre-v4-2026-10-28.zip"
PRECISION = ROOT / "benchmarks/reference/knn-precision-verified-2026-10-30.zip"


def _hash_files(paths):
    return {str(path): sha256(path.read_bytes()) for path in paths}


def _save(report, path):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _request(worker, request):
    """Retain worker exceptions as failed observations instead of losing a run."""
    try:
        return worker.request(request)
    except Exception:
        return {"error": traceback.format_exc(), "request": request, "audit": {"pass": False}}


def _valid_time(sample):
    value = sample.get("ms")
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _sample_pass(sample):
    return bool(sample.get("audit", {}).get("pass")) and sample.get("backend") == "mlx"


def _summarize(samples, expected_count):
    values = [sample["ms"] for sample in samples if _valid_time(sample)]
    return {
        "median_ms": statistics.median(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "completed_samples": len(values),
        "expected_samples": expected_count,
        "quality_pass": len(samples) == expected_count
        and len(values) == expected_count and all(_sample_pass(sample) for sample in samples),
        "samples": samples,
    }


def _warmup_pass(block, duration):
    audits = block.get("warmup_audits", [])
    return (
        block.get("warmup_min_seconds") == duration
        and block.get("warmup_elapsed_seconds", -1) >= duration
        and block.get("warmup_count", 0) == len(audits)
        and len(audits) > 0
        and all(audit.get("pass", False) for audit in audits)
        and bool(block.get("warmup_quality_pass"))
        and bool(block.get("quality_pass"))
        and _summarize(block.get("samples", []), 1)["quality_pass"]
    )


def _paired_ratios(samples, comparator, expected_count):
    values = []
    pairs = []
    for pair, (current, other) in enumerate(zip(samples["current"], samples[comparator], strict=True)):
        valid = _valid_time(current) and _valid_time(other)
        ratio = other["ms"] / current["ms"] if valid else None
        pairs.append({"pair": pair, "ratio": ratio,
                      "quality_pass": valid and _sample_pass(current) and _sample_pass(other)})
        if ratio is not None:
            values.append(ratio)
    return {
        "ratio_definition": f"{comparator} milliseconds / current milliseconds",
        "paired_ratio_median": statistics.median(values) if values else None,
        "paired_ratio_min": min(values) if values else None,
        "paired_ratio_max": max(values) if values else None,
        "completed_pairs": len(values),
        "expected_pairs": expected_count,
        "quality_pass": len(pairs) == expected_count and all(pair["quality_pass"] for pair in pairs),
        "pairs": pairs,
    }


def _rotation(names, offset):
    offset %= len(names)
    return list(names[offset:]) + list(names[:offset])


def run(args):
    seeds = args.seed or list(DEFAULT_SEEDS)
    harness_paths = [Path(__file__).resolve(), Path(__file__).with_name("knn_pipeline.py").resolve()]
    archive_paths = [args.v3_archive, args.v3_archive.with_suffix(".manifest.json"),
                     args.precision_archive, args.precision_archive.with_suffix(".manifest.json")]
    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "seeds": seeds,
        "case": {"n_train": 4096, "n_query": 128, "n_features": 128, "k": 5},
        "pairs": args.pairs,
        "warmup_seconds_per_variant_operation": args.warmup_seconds,
        "operations": list(OPERATIONS),
        "variants": list(VARIANTS),
        "timing_policy": "Four persistent isolated Python workers; GPU calls run sequentially. "
        "Each variant/operation receives continuous worker-local warmup, followed by alternating "
        "interleaved rounds with one synchronized timed call per variant. Starting order rotates "
        "across datasets and operations; every other round reverses it. No result caching or "
        "production hooks. Public validation, NumPy boundaries, and GPU synchronization are timed; "
        "every output is audited outside timing. Ratios are descriptive, not confidence bounds.",
        "warmup_policy": "Worker block warmup lasts at least the requested duration. Its one "
        "post-warmup sample is retained as a separately audited probe, excluded from paired ratios. "
        "All warmup audits and probe samples are preserved.",
        "fit_query_policy": "Every fit_query call constructs a new estimator and includes its "
        "first query and deferred preparation. kneighbors reuses the model established by cold.",
        "cold_policy": "First fit/query for each dataset in each persistent worker is reported "
        "separately and audited. Earlier datasets can populate Metal compilation caches.",
        "query_tolerances": {"indices": "exact equality", "atol": 1e-6, "rtol": 1e-6},
        "accuracy_reference": "Unchanged knn_pipeline fixture: sklearn brute float64 selects "
        "neighbors on serialized float32 inputs; selected distances are recomputed by direct "
        "float64 subtraction and sorted by distance/index.",
        "acceptance_policy": "Any cold, warmup, probe or retained accuracy failure, missing timed "
        "sample, worker error, or source/fixture/harness integrity failure produces exit status 1. "
        "Timing ratios and slowdowns never change the numerical oracle or hide failed samples.",
        "harness_hashes_before": _hash_files(harness_paths),
        "archive_hashes_before": _hash_files(archive_paths),
        "source_hashes_before": source_hashes(ROOT / "src"),
        "cases": [],
    }
    _save(report, args.output)
    workers = {}
    try:
        with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-128-control-") as temp:
            temp = Path(temp)
            roots = {"current": ROOT / "src", "current_b": ROOT / "src",
                     "v3": temp / "v3", "precision": temp / "precision"}
            manifests = {
                "v3": extract_reference(args.v3_archive, roots["v3"]),
                "precision": extract_reference(args.precision_archive, roots["precision"]),
            }
            report["references"] = manifests
            fixtures = temp / "fixtures"
            fixtures.mkdir()
            cases = [Case(f"batch_128_k5_seed{seed}", 4096, 128, 128, 5) for seed in seeds]
            report["fixtures"] = [dict(make_fixture(case, seed, fixtures), seed=seed)
                                  for seed, case in zip(seeds, cases, strict=True)]
            fixture_paths = [fixtures / f"{case.name}.npz" for case in cases]
            report["fixture_hashes_before"] = {
                path.name: sha256(path.read_bytes()) for path in fixture_paths
            }
            for name in VARIANTS:
                workers[name] = Worker(name, roots[name], fixtures)
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            expected_hashes = {"current": report["source_hashes_before"],
                               "current_b": report["source_hashes_before"],
                               **{name: manifest["files"] for name, manifest in manifests.items()}}
            report["worker_initial_source_match"] = {
                name: worker.info["source_hashes"] == expected_hashes[name]
                for name, worker in workers.items()
            }
            for seed_index, (seed, case) in enumerate(zip(seeds, cases, strict=True)):
                entry = {"seed": seed, "case": asdict(case), "cold": {}, "experiments": {}}
                report["cases"].append(entry)
                for name in _rotation(VARIANTS, seed_index):
                    entry["cold"][name] = _request(workers[name], {"op": "cold", "case": asdict(case)})
                for operation_index, operation in enumerate(OPERATIONS):
                    order = _rotation(VARIANTS, seed_index + operation_index)
                    experiment = {"operation": operation, "starting_order": order,
                                  "warmup": {}, "rounds": [], "variants": {}}
                    entry["experiments"][operation] = experiment
                    for name in order:
                        experiment["warmup"][name] = _request(workers[name], {
                            "op": "block", "operation": operation, "case": asdict(case),
                            "warmup": 0, "warmup_seconds": args.warmup_seconds, "repeats": 1,
                        })
                    samples = {name: [] for name in VARIANTS}
                    for pair in range(args.pairs):
                        pair_order = order if pair % 2 == 0 else list(reversed(order))
                        experiment["rounds"].append({"pair": pair, "order": pair_order})
                        for name in pair_order:
                            samples[name].append(_request(workers[name], {
                                "op": operation, "case": asdict(case),
                            }))
                    experiment["variants"] = {
                        name: _summarize(samples[name], args.pairs) for name in VARIANTS
                    }
                    experiment["comparisons"] = {
                        f"current_vs_{other}": _paired_ratios(samples, other, args.pairs)
                        for other in ("v3", "precision", "current_b")
                    }
                    experiment["quality_by_variant"] = {
                        name: _warmup_pass(experiment["warmup"][name], args.warmup_seconds)
                        and experiment["variants"][name]["quality_pass"] for name in VARIANTS
                    }
                    _save(report, args.output)
                print(f"seed {seed}: " + str({
                    op: {name: value["paired_ratio_median"]
                         for name, value in entry["experiments"][op]["comparisons"].items()}
                    for op in OPERATIONS
                }), flush=True)
            report["worker_source_hashes_after"] = {
                name: _request(worker, {"op": "integrity"}) for name, worker in workers.items()
            }
            report["source_integrity_pass"] = all(report["worker_initial_source_match"].values()) and all(
                report["worker_source_hashes_after"][name] == expected_hashes[name] for name in VARIANTS
            )
            report["fixture_hashes_after"] = {
                path.name: sha256(path.read_bytes()) for path in fixture_paths
            }
            report["fixture_integrity_pass"] = report["fixture_hashes_before"] == report["fixture_hashes_after"]
            report["quality_by_variant"] = {
                name: all(_sample_pass(entry["cold"][name]) and all(
                    experiment["quality_by_variant"][name] for experiment in entry["experiments"].values()
                ) for entry in report["cases"]) for name in VARIANTS
            }
            report["quality_pass"] = all(report["quality_by_variant"].values())
            report["complete_case_count_pass"] = len(report["cases"]) == len(seeds) and all(
                set(entry["experiments"]) == set(OPERATIONS) for entry in report["cases"]
            )
            report["status"] = "complete"
    except Exception:
        report["error"] = traceback.format_exc()
        report["status"] = "error"
    finally:
        for worker in workers.values():
            worker.close()
        report["source_hashes_after"] = source_hashes(ROOT / "src")
        report["current_source_integrity_pass"] = report["source_hashes_before"] == report["source_hashes_after"]
        report["harness_hashes_after"] = _hash_files(harness_paths)
        report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
        report["archive_hashes_after"] = _hash_files(archive_paths)
        report["archive_integrity_pass"] = report["archive_hashes_before"] == report["archive_hashes_after"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["acceptance_pass"] = report["status"] == "complete" and all(report.get(key, False) for key in (
            "quality_pass", "complete_case_count_pass", "source_integrity_pass",
            "current_source_integrity_pass", "fixture_integrity_pass", "harness_integrity_pass",
            "archive_integrity_pass",
        ))
        _save(report, args.output)
    print(f"acceptance_pass={report['acceptance_pass']}: {args.output}", flush=True)
    return 0 if report["acceptance_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--v3-archive", type=Path, default=V3)
    parser.add_argument("--precision-archive", type=Path, default=PRECISION)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-128-control-2026-10-30.json")
    args = parser.parse_args()
    if args.pairs < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        parser.error("require positive pairs and finite positive warmup seconds")
    if args.seed and len(set(args.seed)) != len(args.seed):
        parser.error("seeds must be distinct")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
