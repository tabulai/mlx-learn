"""Summarize a complete broad kNN report without importing benchmark runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .knn_backend_audit import backend_counts, validate_record_backend

ROOT = Path(__file__).resolve().parents[1]
ERROR_FIELDS = ("distance_max_abs", "distance_max_abs_from_returned_points", "max_abs")
INTEGRITY_FIELDS = (
    "initial_worker_source_integrity_pass", "worker_source_integrity_pass",
    "fixture_integrity_pass", "source_integrity_pass", "harness_integrity_pass",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_time(sample):
    return finite_number(sample.get("ms")) and sample["ms"] > 0


def value_range(values):
    values = [v for v in values if finite_number(v)]
    return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}


def equal_number(actual, expected):
    return actual is None if expected is None else finite_number(actual) and math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=1e-12,
    )


def validate(report, expected_cases=72, expected_repeats=15):
    """Reject partial/malformed reports; completed requests with failures remain valid."""
    require(report.get("schema_version") == 1, "Unsupported raw schema_version")
    require(report.get("status") == "complete", "Raw report status must be complete")
    mode = report.get("current_mode", "gpu")
    require(mode in {"gpu", "auto"}, "Unknown current mode")
    telemetry = report.get("backend_telemetry_version") == 1
    require(mode != "auto" or telemetry, "Automatic evidence requires per-query backend telemetry")
    cases = report.get("cases", [])
    require(len(cases) == report.get("planned_cases") and cases, "Missing planned cases")
    require(expected_cases is None or len(cases) == expected_cases, "Unexpected case count")
    repeats, warmup = report["repeats"], report["warmup"]
    require(isinstance(repeats, int) and repeats > 0, "Invalid repeats")
    require(isinstance(warmup, int) and warmup >= 0, "Invalid warmup")
    require(expected_repeats is None or repeats == expected_repeats, "Unexpected repeat count")
    variants, operations, targets = report["variants"], report["operations"], report["target_operations"]
    require("current" in variants and len(variants) == len(set(variants)), "Invalid variants")
    require(len(operations) == len(set(operations)), "Duplicate operations")
    require(targets and len(targets) == len(set(targets)) and set(targets) <= set(operations),
            "Invalid target operations")
    require("fit" not in targets, "Fit-only must not be in the search/inference gate")
    require({"kneighbors", "fit_query"} <= set(operations), "Missing query/fresh operations")
    target = report["target_speedup"]
    require(finite_number(target) and target > 0, "Invalid target speedup")
    comparators = [v for v in variants if v != "current"]
    require(comparators, "Missing comparators")
    require(set(report["workers"]) == set(variants), "Missing worker metadata")
    if telemetry:
        require(all(w.get("current_mode") == mode for w in report["workers"].values()),
                "Worker mode disagrees with report")
    require(set(report["quality_by_variant"]) == set(variants), "Missing quality gates")
    require(set(report["gate_by_comparator"]) == set(comparators), "Missing speed gates")
    require(all(k in report for k in INTEGRITY_FIELDS), "Missing integrity results")
    names = [c["case"]["name"] for c in cases]
    require(len(names) == len(set(names)), "Duplicate cases")
    require(len(report["fixtures"]) == len(cases), "Missing fixture metadata")
    require({f["case"]["name"] for f in report["fixtures"]} == set(names), "Fixture/case mismatch")
    for ci, case in enumerate(cases):
        prefix = f"/cases/{ci}"
        require(case["spec"]["case"] == case["case"], f"{prefix}: conflicting case specification")
        require(set(case["cold"]) == set(variants), f"{prefix}: missing cold variant")
        if telemetry:
            for variant, sample in case["cold"].items():
                validate_record_backend(sample, mode, is_sklearn=variant != "current")
        require(set(case["operations"]) == set(operations), f"{prefix}: missing operation")
        for operation, cell in case["operations"].items():
            where = f"{prefix}/operations/{operation}"
            require(set(cell["variants"]) == set(variants), f"{where}: missing variant")
            require(set(cell["comparisons"]) == set(comparators), f"{where}: missing comparison")
            require(len(cell["rounds"]) == repeats, f"{where}: missing round")
            for r, round_info in enumerate(cell["rounds"]):
                require(round_info["round"] == r and sorted(round_info["order"]) == sorted(variants),
                        f"{where}: invalid round order")
            for variant, stats in cell["variants"].items():
                label = f"{where}/variants/{variant}"
                samples, warmups = stats["samples"], stats["warmup_samples"]
                if telemetry:
                    for sample in samples + warmups:
                        validate_record_backend(sample, mode, is_sklearn=variant != "current")
                require(len(samples) == repeats and stats["expected_samples"] == repeats,
                        f"{label}: missing retained request")
                require(len(warmups) == warmup, f"{label}: missing warmup request")
                times = [s["ms"] for s in samples if valid_time(s)]
                require(stats["completed_samples"] == len(times), f"{label}: inconsistent sample count")
                for key, reducer in (("median_ms", statistics.median), ("min_ms", min), ("max_ms", max)):
                    require(equal_number(stats[key], reducer(times) if times else None),
                            f"{label}: inconsistent {key}")
                strict = len(times) == repeats and all(s.get("audit", {}).get("pass", False) for s in samples)
                warm_strict = all(s.get("audit", {}).get("pass", False) for s in warmups)
                require(stats["quality_pass"] == strict and stats["warmup_quality_pass"] == warm_strict,
                        f"{label}: inconsistent quality gate")
            for comparator, comparison in cell["comparisons"].items():
                label = f"{where}/comparisons/{comparator}"
                current = cell["variants"]["current"]
                baseline = cell["variants"][comparator]
                pairs = comparison["pairs"]
                require(len(pairs) == repeats and comparison["expected_pairs"] == repeats,
                        f"{label}: missing pair")
                for r, pair in enumerate(pairs):
                    a, b = current["samples"][r], baseline["samples"][r]
                    ratio = b["ms"] / a["ms"] if valid_time(a) and valid_time(b) else None
                    require(pair["round"] == r and equal_number(pair["speedup"], ratio),
                            f"{label}: inconsistent pair ratio")
                    require(pair["meets_target"] == (ratio is not None and ratio >= target),
                            f"{label}: inconsistent pair gate")
                available = [p["speedup"] for p in pairs if p["speedup"] is not None]
                ratio = (baseline["median_ms"] / current["median_ms"]
                         if baseline["median_ms"] is not None and current["median_ms"] is not None else None)
                require(equal_number(comparison["ratio_of_medians"], ratio), f"{label}: inconsistent ratio")
                require(comparison["median_meets_target"] == (ratio is not None and ratio >= target),
                        f"{label}: inconsistent median gate")
                require(comparison["completed_pairs"] == len(available), f"{label}: inconsistent pair count")
                require(comparison["pairs_meeting_target"] == sum(p["meets_target"] for p in pairs),
                        f"{label}: inconsistent target pair count")
                require(comparison["all_pairs_meet_target"] == all(p["meets_target"] for p in pairs),
                        f"{label}: inconsistent all-pairs gate")
    return comparators


def speed_summary(cells, comparator, target):
    """Each cell is (case index, case, operation); failures never filter this list."""
    records = []
    for ci, case, operation in cells:
        comparison = case["operations"][operation]["comparisons"][comparator]
        records.append({"case": case["case"]["name"], "seed": case["spec"]["seed"],
                        "operation": operation, "comparison": comparison,
                        "raw_pointer": f"/cases/{ci}/operations/{operation}/comparisons/{comparator}"})
    comparisons = [r["comparison"] for r in records]
    ratios = [c["ratio_of_medians"] for c in comparisons]
    pair_values = [p["speedup"] for c in comparisons for p in c["pairs"]]
    median_pass = sum(c["median_meets_target"] for c in comparisons)
    pair_pass = sum(c["pairs_meeting_target"] for c in comparisons)
    pair_count = sum(c["expected_pairs"] for c in comparisons)
    missed = []
    for record in records:
        comparison = record["comparison"]
        if not comparison["median_meets_target"] or not comparison["all_pairs_meet_target"]:
            missed.append({k: v for k, v in record.items() if k != "comparison"} | {
                "ratio_of_medians": comparison["ratio_of_medians"],
                "median_meets_target": comparison["median_meets_target"],
                "paired_calls_below_target_or_missing": comparison["expected_pairs"] - comparison["pairs_meeting_target"],
                "missing_pairs": comparison["expected_pairs"] - comparison["completed_pairs"],
            })
    available = [r for r in records if finite_number(r["comparison"]["ratio_of_medians"])]
    worst = min(available, key=lambda r: r["comparison"]["ratio_of_medians"]) if available else None
    return {
        "target_speedup": target, "median_cells_meeting_target": median_pass,
        "median_cell_count": len(comparisons), "median_cells_below_target_or_missing": len(comparisons) - median_pass,
        "missing_median_ratios": sum(r is None for r in ratios),
        "ratio_of_medians_range": value_range(ratios),
        "median_across_cell_ratios": statistics.median([r for r in ratios if r is not None]) if available else None,
        "paired_calls_meeting_target": pair_pass, "paired_call_count": pair_count,
        "paired_calls_below_target_or_missing": pair_count - pair_pass,
        "missing_pairs": sum(c["expected_pairs"] - c["completed_pairs"] for c in comparisons),
        "paired_speedup_range": value_range(pair_values),
        "all_medians_meet_target": median_pass == len(comparisons),
        "every_pair_meets_target": pair_pass == pair_count,
        "worst_median_cell": ({k: v for k, v in worst.items() if k != "comparison"} | {
            "ratio_of_medians": worst["comparison"]["ratio_of_medians"],
        }) if worst else None,
        "misses": missed,
    }


def audit_kind(sample):
    audit = sample.get("audit", {})
    if audit.get("pass", False):
        return "strict_pass"
    return audit.get("failure_kind") or ("worker_error" if "error" in sample else "unclassified_audit_failure")


def error_maxima(samples):
    return {key: value_range(s.get("audit", {}).get(key) for s in samples)["max"] for key in ERROR_FIELDS}


def phase_quality(samples):
    kinds = Counter(audit_kind(s) for s in samples)
    return {
        "audit_count": len(samples), "strict_passes": kinds.get("strict_pass", 0),
        "strict_failures": len(samples) - kinds.get("strict_pass", 0),
        "failure_kind_counts": {k: v for k, v in sorted(kinds.items()) if k != "strict_pass"},
        "maximum_errors": error_maxima(samples),
        "prediction_mismatch_query_evaluations": sum(s.get("audit", {}).get("mismatch_query_count", 0) for s in samples),
        "prediction_mismatch_boundary_tie_query_evaluations": sum(
            s.get("audit", {}).get("mismatch_boundary_tie_query_count", 0) for s in samples),
        "prediction_mismatch_without_boundary_tie_query_evaluations": sum(
            s.get("audit", {}).get("mismatch_without_boundary_tie_query_count", 0) for s in samples),
    }


def quality_summary(report, variant):
    phases = {"cold": [], "warmup": [], "retained": []}
    details = []
    for ci, case in enumerate(report["cases"]):
        cells = [("cold", {"cold": ([case["cold"][variant]], f"/cases/{ci}/cold/{variant}")})]
        for operation, cell in case["operations"].items():
            base = f"/cases/{ci}/operations/{operation}/variants/{variant}"
            stats = cell["variants"][variant]
            cells.append((operation, {
                "warmup": (stats["warmup_samples"], base + "/warmup_samples"),
                "retained": (stats["samples"], base + "/samples"),
            }))
        for operation, entries in cells:
            failures = {}
            for phase, (samples, pointer) in entries.items():
                phases[phase].extend(samples)
                failed_indexes = [i for i, sample in enumerate(samples) if audit_kind(sample) != "strict_pass"]
                if not failed_indexes:
                    continue
                evidence = {}
                for i in failed_indexes:
                    kind = audit_kind(samples[i])
                    evidence.setdefault(kind, pointer + ("" if phase == "cold" else f"/{i}") + "/audit")
                failures[phase] = phase_quality(samples) | {
                    "failed_sample_indexes": failed_indexes, "raw_samples_pointer": pointer,
                    "first_raw_audit_pointer_by_kind": evidence,
                    "mismatch_queries_first16_union": sorted({
                        q for i in failed_indexes for q in samples[i].get("audit", {}).get("mismatch_queries_first16", [])
                    }),
                }
            if failures:
                details.append({"case": case["case"]["name"], "family": case["spec"]["family"],
                                "seed": case["spec"]["seed"], "operation": operation, "phases": failures})
    all_samples = [s for samples in phases.values() for s in samples]
    missing_times = sum(not valid_time(s) for s in phases["retained"])
    strict = all(audit_kind(s) == "strict_pass" for s in all_samples) and missing_times == 0
    require(report["quality_by_variant"][variant] == strict, f"Inconsistent top-level quality for {variant}")
    return {
        "strict_quality_pass": strict, "all_phases": phase_quality(all_samples),
        "phases": {phase: phase_quality(samples) for phase, samples in phases.items()},
        "retained_requests_missing_valid_timing": missing_times,
        "unique_failure_case_operation_count": len(details), "failure_case_operations": details,
    }


def routing_summary(report):
    phases = {"cold": [], "warmup": [], "retained": []}
    families = defaultdict(lambda: defaultdict(list))
    for case in report["cases"]:
        family = case["spec"]["family"]
        phases["cold"].append(case["cold"]["current"])
        families[family]["cold"].append(case["cold"]["current"])
        for operation, entry in case["operations"].items():
            current = entry["variants"]["current"]
            phases["warmup"].extend(current["warmup_samples"])
            phases["retained"].extend(current["samples"])
            families[family][operation].extend(current["warmup_samples"] + current["samples"])
    return {
        "scope": "Actual query backends, including fit-only's untimed accuracy query. "
        "Fit storage and execution policy are counted separately; no routing is inferred from timings.",
        "by_phase": {name: backend_counts(records) for name, records in phases.items()},
        "by_family_operation": {
            family: {operation: backend_counts(records) for operation, records in operations.items()}
            for family, operations in families.items()
        },
    }


def summarize_report(report, raw_path, raw_sha256, expected_cases=72, expected_repeats=15):
    comparators = validate(report, expected_cases, expected_repeats)
    indexed_cases = list(enumerate(report["cases"]))
    target = report["target_speedup"]
    target_ops = report["target_operations"]
    all_target_cells = [(ci, c, op) for ci, c in indexed_cases for op in target_ops]
    overall = {v: speed_summary(all_target_cells, v, target) for v in comparators}
    for comparator, gate in report["gate_by_comparator"].items():
        require(all(overall[comparator][k] == value for k, value in gate.items()),
                f"Inconsistent headline speed gate for {comparator}")
    families = defaultdict(list)
    for ci, case in indexed_cases:
        families[case["spec"]["family"]].append((ci, case))
    family_rows = []
    for family, cases in families.items():
        row = {"family": family, "case_count": len(cases), "seeds": [c["spec"]["seed"] for _, c in cases],
               "cases": [c["case"]["name"] for _, c in cases],
               "shape": {k: cases[0][1]["case"][k] for k in ("n_train", "n_query", "n_features", "k")},
               "source_kind": cases[0][1]["spec"]["source_kind"], "query_and_fresh": {}, "comparators": {}}
        for operation in ("kneighbors", "fit_query"):
            row["query_and_fresh"][operation] = {
                "median_ms_range_by_variant": {
                    v: value_range(c["operations"][operation]["variants"][v]["median_ms"] for _, c in cases)
                    for v in report["variants"]
                },
                "ratio_of_medians_range_by_comparator": {
                    v: value_range(c["operations"][operation]["comparisons"][v]["ratio_of_medians"] for _, c in cases)
                    for v in comparators
                },
            }
        cells = [(ci, c, op) for ci, c in cases for op in target_ops]
        row["comparators"] = {v: speed_summary(cells, v, target) for v in comparators}
        family_rows.append(row)
    quality = {v: quality_summary(report, v) for v in report["variants"]}
    integrity = {key: report[key] for key in INTEGRITY_FIELDS}
    require(report["accuracy_and_integrity_pass"] == (
        all(integrity.values()) and quality["current"]["strict_quality_pass"]),
        "Inconsistent accuracy/integrity gate")
    return {
        "schema_version": 1, "status": "complete",
        "current_mode": report.get("current_mode", "gpu"),
        "routing": routing_summary(report),
        "raw_evidence": {"path": str(Path(raw_path).resolve()), "sha256": raw_sha256,
                         "pointer_format": "RFC 6901 JSON pointer into the raw report"},
        "started_at": report["started_at"], "finished_at": report["finished_at"],
        "case_count": len(indexed_cases), "family_count": len(families),
        "repeats": report["repeats"], "warmup": report["warmup"],
        "target_speedup": target, "operations": report["operations"], "target_operations": target_ops,
        "speed_policy": report["two_x_policy"], "accuracy_reference": report["accuracy_reference"],
        "aggregation_policy": "All cases, variants, operations, and requests are retained. Accuracy failures "
        "are never removed from speed statistics. Missing timing pairs fail the speed target. Raw medians "
        "use finite positive timings even when audits fail; missing timings are counted explicitly. "
        "Median-across-cell ratios are descriptive only; gates never use them. Strict accuracy remains "
        "separate from speed. Exact-valid query tie alternatives remain strict failures; boundary-tie "
        "prediction differences remain ambiguous, not validated alternatives. Error maxima include passing audits. "
        "Query-evaluation counts count repeated calls, not unique queries.",
        "protocol": report["protocol"], "platform": report["platform"], "workers": report["workers"],
        "source_manifest": report["source_manifest"], "source_snapshot": report["source_snapshot"],
        "integrity": integrity, "accuracy_and_integrity_pass": report["accuracy_and_integrity_pass"],
        "gate_by_comparator": report["gate_by_comparator"],
        "overall_target_operations": overall,
        "per_operation": {
            op: {"included_in_headline_speed_gate": op in target_ops,
                 "median_ms_range_by_variant": {
                     v: value_range(c["operations"][op]["variants"][v]["median_ms"] for _, c in indexed_cases)
                     for v in report["variants"]},
                 "comparators": {v: speed_summary([(ci, c, op) for ci, c in indexed_cases], v, target)
                                 for v in comparators}}
            for op in report["operations"]
        },
        "families": family_rows, "quality_by_variant": quality,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks/results/knn-broad-2026-10-30.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/knn-broad-summary-2026-10-30.json")
    parser.add_argument("--expected-cases", type=int, default=72, help="Expected cases; 0 accepts the declared count")
    parser.add_argument("--expected-repeats", type=int, default=15, help="Expected retained repeats; 0 accepts the declared count")
    args = parser.parse_args()
    if args.expected_cases < 0 or args.expected_repeats < 0:
        parser.error("Expected counts must be nonnegative")
    try:
        data = args.input.read_bytes()
        summary = summarize_report(json.loads(data), args.input, hashlib.sha256(data).hexdigest(),
                                   args.expected_cases or None, args.expected_repeats or None)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"Cannot summarize report: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "case_count": summary["case_count"],
                      "family_count": summary["family_count"], "gate_by_comparator": summary["gate_by_comparator"]}))


if __name__ == "__main__":
    main()
