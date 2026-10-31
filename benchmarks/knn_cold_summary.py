"""Validate every fresh-process launch and summarize first-use latency."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .knn_cold import METRICS, validate
from .knn_freeze_runtime import verify_runtime_archive
from .knn_reference_runtime import validate_reference_worker


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_report(report, *, broad=None, expected_cases=18, expected_repeats=5):
    validate(report)
    require(expected_cases is None or report["planned_cases"] == expected_cases, "Unexpected cold case count")
    require(expected_repeats is None or report["repeats"] == expected_repeats, "Unexpected cold repeat count")
    for stem in ("source", "harness", "archive", "fixture", "native"):
        require(report[f"{stem}_hashes_before"] == report[f"{stem}_hashes_after"], f"Cold {stem} changed")
        require(report[f"{stem}_integrity_pass"], f"Cold {stem} integrity gate failed")
    require(report["reference_runtime_integrity_pass"], "Reference cold native runtime changed")
    expected_acceptance = report["timing_count_pass"] and all(
        report["quality_by_variant"][name] for name in report["variants"] if not name.startswith("sklearn")
    )
    require(report["acceptance_pass"] == expected_acceptance, "Cold acceptance gate disagrees with accuracy/timing counts")
    fixtures = {f["case"]["name"]: f for f in report["fixtures"]}
    require(len(fixtures) == report["planned_cases"], "Missing/duplicate cold fixtures")
    if broad is not None:
        require(report["source_hashes_before"] == broad["source_hashes_before"], "Cold/broad source mismatch")
        broad_fixtures = {f["case"]["name"]: f for f in broad["fixtures"]}
        require(all(f == broad_fixtures.get(name) for name, f in fixtures.items()), "Cold/broad fixture or oracle changed")
    runtime = report.get("reference_runtime")
    if runtime is not None:
        require(verify_runtime_archive(report["reference_runtime_archive"]) == runtime, "Cold reference runtime archive changed")
        require(runtime["source_files"] == report["reference"]["files"], "Cold reference source/runtime mismatch")
    counts = {v: Counter() for v in report["variants"]}
    routes = {v: Counter() for v in report["variants"]}
    prefit_native = {v: Counter() for v in report["variants"]}
    prefit_stock_neighbors = {v: Counter() for v in report["variants"]}
    seen_launches = set()
    for case in report["cases"]:
        for variant, stats in case["variants"].items():
            for sample in stats["samples"]:
                counts[variant]["requests"] += 1
                counts[variant]["strict_passes"] += bool(sample.get("audit", {}).get("pass"))
                if "error" in sample:
                    counts[variant]["errors"] += 1
                    continue
                counts[variant]["completed_timings"] += 1
                identity = (sample["pid"], sample["entry_ns"])
                require(identity not in seen_launches, "A supposedly fresh process was reused")
                seen_launches.add(identity)
                require(sample["result_ready_ns"] > sample["entry_ns"], "Cold time marker precedes worker entry")
                require(not any(sample["loaded_modules"]["before_input"].values()), "Heavy modules were loaded before cold input setup")
                expected = report["reference"]["files"] if variant == "previous_auto" else report["source_hashes_before"]
                require(sample["worker"]["source_hashes"] == expected, "Cold source identity changed")
                require(sample["fixture_sha256"] == fixtures[case["case"]["name"]]["fixture_sha256"], "Cold fixture identity changed")
                if variant == "previous_auto":
                    validate_reference_worker(sample["worker"], runtime)
                if not variant.startswith("sklearn"):
                    native = sample["worker"]["native_cpu_extension"]
                    binary_hash = native["binary"]["sha256"] if native["available"] else None
                    require(binary_hash == report["native_hashes_before"][variant], "Cold native binary identity changed")
                routes[variant][sample["query_backend"]] += 1
                prefit_native[variant][str(sample["loaded_modules"]["before_fit"]["mlxlearn._kernels._neighbors_cpu_native"])] += 1
                prefit_stock_neighbors[variant][str(sample["loaded_modules"]["before_fit"].get("sklearn.neighbors", "not_recorded"))] += 1
    require(sum(c["requests"] for c in counts.values()) == report["expected_launches"], "Cold audit count changed")
    comparisons = {}
    family_rows = defaultdict(list)
    for variant in report["variants"]:
        if variant == "current":
            continue
        comparisons[variant] = {}
        for metric in METRICS:
            rows = [{"case": case["case"]["name"], "family": case["spec"]["family"],
                     "current_ms": case["variants"]["current"]["metrics"][metric]["median_ms"],
                     "comparator_ms": case["variants"][variant]["metrics"][metric]["median_ms"],
                     **case["comparisons"][variant][metric]} for case in report["cases"]]
            available = [r["ratio_of_medians"] for r in rows if r["ratio_of_medians"] is not None]
            comparisons[variant][metric] = {
                "cell_count": len(rows), "faster_medians": sum(r > 1 for r in available),
                "two_x_medians": sum(r >= 2 for r in available),
                "minimum_speedup": min(available) if available else None,
                "maximum_speedup": max(available) if available else None,
                "all_cells": rows,
                "slower_or_missing_cells": [r for r in rows if r["ratio_of_medians"] is None or r["ratio_of_medians"] < 1],
            }
    for case in report["cases"]:
        family_rows[case["spec"]["family"]].append({
            "case": case["case"]["name"], "seed": case["spec"]["seed"],
            "metrics_by_variant": {v: s["metrics"] for v, s in case["variants"].items()},
        })
    return {
        "schema_version": 1, "evidence_validation_pass": True, "acceptance_pass": report["acceptance_pass"],
        "case_count": report["planned_cases"], "repeats": report["repeats"], "launch_count": report["expected_launches"],
        "audit_counts": {v: dict(c) for v, c in counts.items()},
        "query_backend_counts": {v: dict(c) for v, c in routes.items()},
        "native_loaded_before_fit_counts": {v: dict(c) for v, c in prefit_native.items()},
        "sklearn_neighbors_loaded_before_fit_counts": {v: dict(c) for v, c in prefit_stock_neighbors.items()},
        "comparisons": comparisons, "families": dict(family_rows), "protocol": report["protocol"],
        "interpretation": "Fit/query reductions may reflect moving native initialization into normal estimator "
        "imports. Estimator setup and estimator/process-to-result measures retain that cost. Parent result-marker "
        "latency includes interpreter/input loading/IPC; worker estimator-to-result excludes input preparation. "
        "No launch or failed audit is removed, and no OS/bytecode/Metal-disk-cache coldness is claimed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--broad", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=18)
    parser.add_argument("--expected-repeats", type=int, default=5)
    args = parser.parse_args()
    data = args.input.read_bytes()
    broad = json.loads(args.broad.read_text()) if args.broad else None
    report = summarize_report(json.loads(data), broad=broad, expected_cases=args.expected_cases or None,
                              expected_repeats=args.expected_repeats or None)
    report["raw_evidence"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "launch_count": report["launch_count"],
                      "acceptance_pass": report["acceptance_pass"]}))
    raise SystemExit(0 if report["acceptance_pass"] else 1)


if __name__ == "__main__":
    main()
