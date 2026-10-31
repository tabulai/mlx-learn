"""Validate and render initialized-process first/second-query evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .knn_broad_report import LABELS, table
from .knn_cold_tables import NAMES, link, number, ratio
from .knn_first_query import METRICS, validate
from .knn_freeze_runtime import verify_runtime_archive


def summarize_report(raw, *, broad=None, expected_cases=18, expected_repeats=30):
    validate(raw, expected_cases=expected_cases, expected_repeats=expected_repeats)
    if broad is not None:
        fixtures = {f["case"]["name"]: f for f in broad["fixtures"]}
        if raw["source_hashes_before"] != broad["source_hashes_before"] or any(
            f != fixtures.get(f["case"]["name"]) for f in raw["fixtures"]
        ):
            raise ValueError("First-query/broad source or fixture mismatch")
    if "previous_auto" in raw["variants"]:
        runtime = verify_runtime_archive(raw["reference_runtime_archive"])
        if runtime != raw["reference_runtime"] or runtime["source_files"] != raw["reference"]["files"]:
            raise ValueError("First-query prior native runtime/source changed")
    audits, routing, model_phases, native_before_fit, stock_before_fit = {}, {}, {}, {}, {}
    for variant in raw["variants"]:
        count, routes, phases, native, stock = Counter(), Counter(), Counter(), Counter(), Counter()
        for case in raw["cases"]:
            for sample in case["variants"][variant]["samples"]:
                count["models_requested"] += 1
                if "error" in sample:
                    count["model_errors"] += 1
                    continue
                phases["first_model_in_worker"] += sample["first_model_in_worker"]
                phases["first_model_for_fixture"] += sample["first_model_for_fixture"]
                phases["subsequent_fresh_models"] += not sample["first_model_for_fixture"]
                native[str(sample["loaded_modules_before_fit"]["mlxlearn._kernels._neighbors_cpu_native"])] += 1
                stock[str(sample["loaded_modules_before_fit"].get("sklearn.neighbors", "not_recorded"))] += 1
                for phase in ("first", "second"):
                    count[phase + "_queries"] += 1
                    count[phase + "_strict_passes"] += bool(sample[phase]["audit"].get("pass"))
                    routes[phase + "/" + sample[phase]["query_backend"]] += 1
        audits[variant], routing[variant] = dict(count), dict(routes)
        model_phases[variant], native_before_fit[variant] = dict(phases), dict(native)
        stock_before_fit[variant] = dict(stock)
    comparisons = {}
    for variant in raw["variants"]:
        if variant == "current":
            continue
        comparisons[variant] = {}
        for metric in METRICS:
            cells = [{"case": c["case"]["name"], "family": c["spec"]["family"],
                      "current_ms": c["variants"]["current"]["metrics"]["all_rounds"][metric]["median_ms"],
                      "comparator_ms": c["variants"][variant]["metrics"]["all_rounds"][metric]["median_ms"],
                      **c["comparisons"][variant][metric]} for c in raw["cases"]]
            available = [c["ratio_of_medians"] for c in cells if c["ratio_of_medians"] is not None]
            comparisons[variant][metric] = {"cell_count": len(cells), "faster_medians": sum(r > 1 for r in available),
                "min_ratio": min(available) if available else None, "max_ratio": max(available) if available else None,
                "all_cells": cells, "slower_or_missing_cells": [c for c in cells if c["ratio_of_medians"] is None or c["ratio_of_medians"] < 1]}
    return {"schema_version": 1, "evidence_validation_pass": True, "acceptance_pass": raw["acceptance_pass"],
            "case_count": raw["planned_cases"], "repeats": raw["repeats"], "audit_counts": audits,
            "routing": routing, "model_phases": model_phases, "native_before_fit": native_before_fit,
            "sklearn_neighbors_before_fit": stock_before_fit,
            "setup": raw["setup"], "comparisons": comparisons,
            "interpretation": "First query includes model preparation deferred by fit. Second query reuses only that "
            "fresh model. All rounds remain in the headline median; first-per-fixture and subsequent-model views are "
            "shown separately, not selected according to speed. Process initialization/import cost is separate, and "
            "must not be confused with the dedicated fresh-process cold benchmark."}


def render(raw, raw_path, output, *, broad=None, expected_cases=18, expected_repeats=30):
    summary = summarize_report(raw, broad=broad, expected_cases=expected_cases, expected_repeats=expected_repeats)
    variants = raw["variants"]
    others = [v for v in variants if v != "current"]
    lines = ["# Fresh-model first queries in initialized processes", raw["protocol"], summary["interpretation"],
             f"Acceptance: **{raw['acceptance_pass']}**. {raw['planned_cases']} fixtures × {raw['repeats']} fresh models "
             f"× {len(variants)} workers; each model has exactly one retained first and one retained second query. "
             "No numerical audit failure removes a valid timing. Ratios are comparator/current; bold values below "
             "1× are slower observations. These descriptive ratios are not confidence intervals.",
             "## Initial process setup (once per worker)",
             table(["Worker", "NumPy import ms", "Estimator import/setup ms", "Worker entry→initialized ms", "Parent process→initialized ms"], [
                 [NAMES[v], *(number(raw["setup"][v][m]) for m in ("numpy_import_ms", "estimator_import_setup_ms",
                     "worker_entry_to_initialized_ms", "parent_process_start_to_initialized_ms"))] for v in variants])]
    for metric in METRICS:
        rows = []
        for c in raw["cases"]:
            rows.append([LABELS.get(c["spec"]["family"], c["spec"]["family"]), c["spec"]["seed"],
                number(c["variants"]["current"]["metrics"]["all_rounds"][metric]["median_ms"]),
                *(number(c["variants"][v]["metrics"]["all_rounds"][metric]["median_ms"]) + " / "
                  + ratio(c["comparisons"][v][metric]["ratio_of_medians"]) + " / "
                  + ratio(c["comparisons"][v][metric]["median_paired_ratio"]) for v in others)])
        lines += ["## " + metric, "All requested rounds. Comparator columns show median ms / ratio of medians / median paired ratio.",
                  table(["Family", "Seed", "Current median ms", *(NAMES[v] + " ms / M / P" for v in others)], rows)]
    misses = [[metric, NAMES[v], c["case"], number(c["current_ms"]), number(c["comparator_ms"]),
               ratio(c["ratio_of_medians"]), ratio(c["median_paired_ratio"])]
              for v in others for metric in METRICS for c in summary["comparisons"][v][metric]["slower_or_missing_cells"]]
    lines += ["## Every slower or missing all-round median",
              table(["Metric", "Comparator", "Case", "Current ms", "Comparator ms", "M", "P"], misses)
              if misses else "No slower or missing all-round medians."]
    for scope in ("first_model_for_fixture", "subsequent_fresh_models"):
        lines += ["## " + scope,
                  "All configurations and metrics are retained in this separate view. Earlier fixtures may already "
                  "have initialized process/runtime/compiler state; this table is not fresh-process cold latency.",
                  table(["Case", "Worker", *(m + " median ms" for m in METRICS)], [
                      [c["case"]["name"], NAMES[v], *(number(c["variants"][v]["metrics"][scope][m]["median_ms"]) for m in METRICS)]
                      for c in raw["cases"] for v in variants])]
    failures = [[c["case"]["name"], NAMES[v], i, phase,
                 f"`/cases/{ci}/variants/{v}/samples/{i}`"]
                for ci, c in enumerate(raw["cases"]) for v in variants
                for i, s in enumerate(c["variants"][v]["samples"])
                for phase in (("worker_error",) if "error" in s else ("first", "second"))
                if "error" in s or not s[phase]["audit"].get("pass", False)]
    lines += ["## Every audit failure", table(["Case", "Worker", "Round", "Phase", "Raw pointer"], failures)
              if failures else "All retained first and second queries pass strict audits.",
              "## Audit counts and actual routing", table(["Worker", "Audits", "Query routing", "Native loaded before fit", "Stock sklearn.neighbors loaded before fit"], [
                  [NAMES[v], json.dumps(summary["audit_counts"][v], sort_keys=True),
                   json.dumps(summary["routing"][v], sort_keys=True), json.dumps(summary["native_before_fit"][v], sort_keys=True),
                   json.dumps(summary["sklearn_neighbors_before_fit"][v], sort_keys=True)] for v in variants]),
              "## Evidence", link(raw_path, output, "Raw requests, setup, source/runtime identities, and all audits")]
    return "\n\n".join(lines) + "\n", summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--broad", type=Path)
    parser.add_argument("--expected-cases", type=int, default=18)
    parser.add_argument("--expected-repeats", type=int, default=30)
    args = parser.parse_args()
    if any(p is not None and p.exists() for p in (args.output, args.summary)):
        raise FileExistsError("Refusing to overwrite an existing derived report")
    data = args.input.read_bytes()
    broad = json.loads(args.broad.read_text()) if args.broad else None
    text, summary = render(json.loads(data), args.input, args.output, broad=broad,
                           expected_cases=args.expected_cases or None, expected_repeats=args.expected_repeats or None)
    summary["raw_evidence"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        text += "\n" + link(args.summary, args.output, "Validated summary") + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
