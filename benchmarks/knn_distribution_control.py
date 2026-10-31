"""Audited current/A-A/frozen-source comparison on exact broad-panel fixtures.

All timed work is serial. No estimator hooks, output caches, oracle changes or
fixture transforms are used. Freeze current source before running this harness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .knn_128_control import (
    _hash_files,
    _paired_ratios,
    _sample_pass,
    _summarize,
    _warmup_pass,
)
from .knn_broad import BroadWorker, request, rotate, save
from .knn_broad_fixtures import make_fixture, workload_specs
from .knn_pipeline import ROOT, extract_reference, sha256, source_hashes

VARIANTS = ("current", "current_b", "reference")
OPERATIONS = ("kneighbors", "fit_query")
REFERENCE = ROOT / "benchmarks/reference/knn-128-final-2026-10-30.zip"


def select_specs(names=None):
    all_specs = workload_specs()
    if names:
        missing = set(names) - {s["case"]["name"] for s in all_specs}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
        return [s for s in all_specs if s["case"]["name"] in names]
    return [s for s in all_specs if s["suite"] == "distribution_panel"]


def family_summary(cases):
    families = defaultdict(list)
    for case in cases:
        families[case["spec"]["family"]].append(case)
    result = []
    for name, members in families.items():
        operations = {}
        for operation in OPERATIONS:
            entries = [c["experiments"][operation] for c in members]
            variants = {}
            for variant in VARIANTS:
                values = [e["variants"][variant]["median_ms"] for e in entries]
                variants[variant] = {"min": min(values), "max": max(values)} if all(
                    v is not None for v in values) else {"min": None, "max": None}
            comparisons = {}
            for other in VARIANTS[1:]:
                values = [e["comparisons"][f"current_vs_{other}"]["paired_ratio_median"]
                          for e in entries]
                comparisons[other] = {"min": min(values), "max": max(values)} if all(
                    v is not None for v in values) else {"min": None, "max": None}
            operations[operation] = {
                "median_ms_ranges": variants,
                "median_paired_ratio_ranges": comparisons,
            }
        result.append({"family": name, "seeds": [c["spec"]["seed"] for c in members],
                       "operations": operations})
    return result


def run(args):
    specs = select_specs(args.case)
    manifest = json.loads(args.source_manifest.read_text())
    before = source_hashes(ROOT / "src")
    if before != manifest["files"]:
        raise RuntimeError("Current source differs from --source-manifest")
    harness_paths = [ROOT / "benchmarks" / name for name in (
        "knn_distribution_control.py", "knn_128_control.py", "knn_broad.py",
        "knn_broad_worker.py", "knn_broad_fixtures.py", "knn_pipeline.py",
        "knn_review_distributions.py", "knn_acceleration.py",
    )]
    archive_paths = [args.reference_archive, args.reference_archive.with_suffix(".manifest.json"),
                     args.source_manifest]
    report = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "planned_cases": len(specs), "pairs": args.pairs,
        "warmup_seconds_per_variant_operation": args.warmup_seconds,
        "operations": list(OPERATIONS), "variants": list(VARIANTS),
        "source_manifest": str(args.source_manifest.resolve()), "source_snapshot": manifest["archive"],
        "reference_archive": str(args.reference_archive.resolve()),
        "source_hashes_before": before, "harness_hashes_before": _hash_files(harness_paths),
        "archive_hashes_before": _hash_files(archive_paths),
        "timing_policy": "Three persistent isolated public-API workers: current, an identical-code "
        "A/A control, and frozen reference. All calls serial; continuous per-worker warmup, then "
        "alternating interleaved paired rounds. Starting order rotates by dataset and operation. "
        "Every call includes validation, NumPy conversion and synchronization. IPC and audits "
        "are outside timing. Ratios are descriptive; A/A exposes process/timing variation.",
        "warmup_policy": "Each worker/operation warms continuously for at least the requested "
        "duration. Its one post-warmup probe is audited and retained separately, excluded from "
        "paired ratios. All warmup audits and probe samples remain in the report.",
        "fit_query_policy": "Fresh fit plus first query each call includes all deferred preparation. "
        "Repeated kneighbors reuses the estimator established by cold. No result caching.",
        "cold_policy": "First fit/query per fixture in a persistent worker, not fresh-process startup. "
        "Earlier fixtures may populate compiler caches. Cold times remain separate.",
        "accuracy_reference": "Identical knn_broad_fixtures: exhaustive direct float64 differences "
        "on serialized float32 coordinates, canonical distance/index ordering. Exact indices; "
        "distance atol/rtol 1e-6. No centering, rescaling or jitter of benchmark inputs.",
        "acceptance_policy": "Every current, A/A and frozen-reference cold, warmup, probe and retained "
        "audit must pass, with complete valid timing counts and source/fixture/harness/archive "
        "integrity. Speedups do not alter accuracy gates or filter failed records.",
        "cases": [],
    }
    save(report, args.output)
    workers = {}
    try:
        with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-distribution-control-") as tmp:
            temp = Path(tmp)
            roots = {"current": ROOT / "src", "current_b": ROOT / "src", "reference": temp / "reference"}
            reference = extract_reference(args.reference_archive, roots["reference"])
            report["reference"] = reference
            fixtures = temp / "fixtures"
            fixtures.mkdir()
            report["fixtures"] = [make_fixture(spec, fixtures) for spec in specs]
            fixture_paths = [fixtures / f"{s['case']['name']}.npz" for s in specs]
            report["fixture_hashes_before"] = {p.name: sha256(p.read_bytes()) for p in fixture_paths}
            for name in VARIANTS:
                # The worker's current mode means public MLX API; process source
                # roots, never monkeypatches, distinguish the implementations.
                workers[name] = BroadWorker("current", fixtures, source_root=roots[name])
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            expected_hashes = {"current": before, "current_b": before, "reference": reference["files"]}
            report["worker_initial_source_match"] = {
                name: worker.info["source_hashes"] == expected_hashes[name]
                for name, worker in workers.items()
            }
            if not all(report["worker_initial_source_match"].values()):
                raise RuntimeError("A worker loaded source that does not match its manifest")
            for case_index, spec in enumerate(specs):
                case = spec["case"]
                entry = {"spec": spec, "case": case, "cold": {}, "experiments": {}}
                report["cases"].append(entry)
                for name in rotate(VARIANTS, case_index):
                    entry["cold"][name] = request(workers[name], {"op": "cold", "case": case})
                for operation_index, operation in enumerate(OPERATIONS):
                    order = rotate(VARIANTS, case_index + operation_index)
                    experiment = {"operation": operation, "starting_order": order,
                                  "warmup": {}, "rounds": [], "variants": {}}
                    entry["experiments"][operation] = experiment
                    for name in order:
                        experiment["warmup"][name] = request(workers[name], {
                            "op": "block", "operation": operation, "case": case,
                            "warmup": 0, "warmup_seconds": args.warmup_seconds, "repeats": 1,
                        })
                    samples = {name: [] for name in VARIANTS}
                    for pair in range(args.pairs):
                        pair_order = order if pair % 2 == 0 else list(reversed(order))
                        experiment["rounds"].append({"pair": pair, "order": pair_order})
                        for name in pair_order:
                            samples[name].append(request(workers[name], {"op": operation, "case": case}))
                    experiment["variants"] = {
                        name: _summarize(samples[name], args.pairs) for name in VARIANTS
                    }
                    experiment["comparisons"] = {
                        f"current_vs_{other}": _paired_ratios(samples, other, args.pairs)
                        for other in VARIANTS[1:]
                    }
                    for other in VARIANTS[1:]:
                        current_ms = experiment["variants"]["current"]["median_ms"]
                        other_ms = experiment["variants"][other]["median_ms"]
                        experiment["comparisons"][f"current_vs_{other}"]["ratio_of_medians"] = (
                            other_ms / current_ms if current_ms is not None and other_ms is not None else None
                        )
                    experiment["quality_by_variant"] = {
                        name: _warmup_pass(experiment["warmup"][name], args.warmup_seconds)
                        and experiment["variants"][name]["quality_pass"] for name in VARIANTS
                    }
                    save(report, args.output)
                print(f"case {case_index+1}/{len(specs)} {case['name']}: " + str({
                    op: {name: value["paired_ratio_median"]
                         for name, value in entry["experiments"][op]["comparisons"].items()}
                    for op in OPERATIONS
                }), flush=True)
            report["worker_source_hashes_after"] = {
                name: request(worker, {"op": "integrity"}) for name, worker in workers.items()
            }
            report["source_integrity_pass"] = all(report["worker_initial_source_match"].values()) and all(
                report["worker_source_hashes_after"][name] == expected_hashes[name] for name in VARIANTS
            )
            report["fixture_hashes_after"] = {p.name: sha256(p.read_bytes()) for p in fixture_paths}
            report["fixture_integrity_pass"] = report["fixture_hashes_before"] == report["fixture_hashes_after"]
            report["quality_by_variant"] = {
                name: all(_sample_pass(entry["cold"][name]) and all(
                    experiment["quality_by_variant"][name] for experiment in entry["experiments"].values()
                ) for entry in report["cases"]) for name in VARIANTS
            }
            report["quality_pass"] = all(report["quality_by_variant"].values())
            report["complete_case_count_pass"] = len(report["cases"]) == len(specs) and all(
                set(entry["experiments"]) == set(OPERATIONS) for entry in report["cases"]
            )
            report["families"] = family_summary(report["cases"])
            report["status"] = "complete"
    except Exception:
        report["error"] = traceback.format_exc()
        report["status"] = "error"
    finally:
        for worker in workers.values():
            worker.close()
        report["source_hashes_after"] = source_hashes(ROOT / "src")
        report["current_source_integrity_pass"] = before == report["source_hashes_after"]
        report["harness_hashes_after"] = _hash_files(harness_paths)
        report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
        report["archive_hashes_after"] = _hash_files(archive_paths)
        report["archive_integrity_pass"] = report["archive_hashes_before"] == report["archive_hashes_after"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["acceptance_pass"] = report["status"] == "complete" and all(report.get(key, False) for key in (
            "quality_pass", "complete_case_count_pass", "source_integrity_pass", "current_source_integrity_pass",
            "fixture_integrity_pass", "harness_integrity_pass", "archive_integrity_pass",
        ))
        save(report, args.output)
    print(f"acceptance_pass={report['acceptance_pass']}: {args.output}", flush=True)
    return 0 if report["acceptance_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        parser.error("require positive pairs and finite positive warmup seconds")
    if args.case and len(set(args.case)) != len(args.case):
        parser.error("case names must be distinct")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
