"""Render and independently validate the complete small-workload control."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .knn_128_control import _paired_ratios, _sample_pass, _summarize, _warmup_pass
from .knn_pipeline import sha256
from .knn_small_control import (
    TARGET_FAMILIES,
    VARIANTS,
    family_summary,
    operations_for,
    select_specs,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report, *, expected_cases=48):
    """Recompute raw summaries; retain valid reports containing failed audits."""
    require(report.get("status") == "complete", "Report must be complete")
    require(report.get("schema_version") == 1, "Unexpected schema")
    require(report["variants"] == list(VARIANTS), "Unexpected variants")
    cases = report["cases"]
    require(len(cases) == report["planned_cases"], "Missing cases")
    require(expected_cases is None or len(cases) == expected_cases, "Unexpected case count")
    names = [case["case"]["name"] for case in cases]
    require(len(names) == len(set(names)), "Duplicate cases")
    specs = select_specs(names)
    require([case["spec"] for case in cases] == specs, "Changed case specifications")
    expected_ops = {spec["case"]["name"]: list(operations_for(spec)) for spec in specs}
    require(report["operations_by_case"] == expected_ops, "Changed operation inventory")
    require(report["planned_case_operation_count"] == sum(map(len, expected_ops.values())),
            "Wrong case-operation count")
    require(len(report["fixtures"]) == len(cases), "Missing fixture metadata")
    pairs, duration = report["pairs"], report["warmup_seconds_per_variant_operation"]
    require(isinstance(pairs, int) and pairs > 0, "Invalid pair count")
    require(math.isfinite(duration) and duration > 0, "Invalid warmup duration")
    quality = dict.fromkeys(VARIANTS, True)
    for case in cases:
        require(case["case"] == case["spec"]["case"], "Conflicting case specification")
        require(set(case["experiments"]) == set(operations_for(case["spec"])), "Missing operation")
        require(set(case["cold"]) == set(VARIANTS), "Missing cold record")
        for variant in VARIANTS:
            quality[variant] &= _sample_pass(case["cold"][variant])
        for experiment in case["experiments"].values():
            require(len(experiment["rounds"]) == pairs, "Missing round")
            for pair, entry in enumerate(experiment["rounds"]):
                require(entry["pair"] == pair and sorted(entry["order"]) == sorted(VARIANTS),
                        "Invalid round order")
            samples = {}
            for variant in VARIANTS:
                raw = experiment["variants"][variant]
                samples[variant] = raw["samples"]
                require(raw == _summarize(samples[variant], pairs), "Inconsistent timing summary")
                require(len(samples[variant]) == pairs, "Missing retained request")
                passed = (raw["quality_pass"]
                          and _warmup_pass(experiment["warmup"][variant], duration))
                require(experiment["quality_by_variant"][variant] == passed,
                        "Inconsistent experiment quality")
                quality[variant] &= passed
            for other in VARIANTS[1:]:
                expected = _paired_ratios(samples, other, pairs)
                current_ms = experiment["variants"]["current"]["median_ms"]
                other_ms = experiment["variants"][other]["median_ms"]
                expected["ratio_of_medians"] = (
                    other_ms / current_ms if current_ms is not None and other_ms is not None else None
                )
                require(experiment["comparisons"][f"current_vs_{other}"] == expected,
                        "Inconsistent paired comparison")
    require(report["quality_by_variant"] == quality, "Inconsistent overall quality")
    require(report["quality_pass"] == all(quality.values()), "Inconsistent quality gate")
    require(report["families"] == family_summary(cases), "Inconsistent family summary")
    gates = ("quality_pass", "complete_case_count_pass", "source_integrity_pass",
             "current_source_integrity_pass", "fixture_integrity_pass",
             "harness_integrity_pass", "archive_integrity_pass")
    require(report["complete_case_count_pass"], "Incomplete case count gate")
    require(report["acceptance_pass"] == all(report[field] for field in gates),
            "Inconsistent acceptance gate")


def span(value, suffix=""):
    lo, hi = value["min"], value["max"]
    return "missing" if lo is None or hi is None else f"{lo:.3f}–{hi:.3f}{suffix}"


def render(report, raw_path, output):
    validate(report)
    raw_link = os.path.relpath(raw_path.resolve(), output.resolve().parent)
    lines = [
        "# Small-workload controlled benchmark",
        "",
        f"Completed `{report['finished_at']}`. Acceptance: **{report['acceptance_pass']}**. "
        f"[Raw results]({raw_link}) (SHA-256 `{sha256(raw_path.read_bytes())}`).",
        "",
        f"Three isolated public-API workers: current, identical-code A/A and frozen "
        "`knn-gaussian-fresh-final-2026-10-30.zip`. All GPU calls are serial. "
        f"Each worker/operation gets {report['warmup_seconds_per_variant_operation']:g} seconds "
        f"of continuous warmup followed by {report['pairs']} interleaved paired rounds. "
        "Every call includes public validation, NumPy boundaries and synchronization. "
        "Cold first-fit/query records come from persistent workers, excluding process startup.",
        "",
        "The 18 small-dataset/single-query fixtures cover all nine search/inference operations. "
        "The 30 preservation fixtures cover query and fresh fit-plus-query, including the six "
        "expanded Gaussian seeds. Inputs, exact canonical neighbor ordering, distance tolerances "
        "and prediction oracles are unchanged. Every cold, warmup, probe and retained audit stays "
        "in the raw file, including failures. No result caching or production hooks are used.",
        "",
        "All speedups are comparator/current time. Ranges span dataset seeds. Ratio of medians "
        "and median paired ratios are distinct descriptive statistics; neither is a confidence "
        "interval. A/A shows the corresponding process/timing variation. Values below 1× are "
        "slower observations and are retained.",
        "",
        f"Per-variant strict quality: `{json.dumps(report['quality_by_variant'], sort_keys=True)}`.",
        "",
    ]
    for target, title in ((True, "Small datasets and single queries"),
                          (False, "Gaussian, clustered, offset and unequal-scale preservation")):
        lines.extend([
            f"## {title}", "",
            "| Family | Operation | Seeds | Current ms | Previous ms | Previous/current medians | "
            "Median paired speedup | A/A paired ratio |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for family in report["families"]:
            if (family["family"] in TARGET_FAMILIES) != target:
                continue
            for operation, stats in family["operations"].items():
                times = stats["median_ms_ranges"]
                paired = stats["median_paired_ratio_ranges"]
                median = stats["ratio_of_medians_ranges"]
                seeds = ", ".join(map(str, family["seeds"]))
                lines.append(f"| `{family['family']}` | {operation} | {seeds} | "
                             f"{span(times['current'])} | {span(times['previous'])} | "
                             f"{span(median['previous'], '×')} | {span(paired['previous'], '×')} | "
                             f"{span(paired['current_b'], '×')} |")
        lines.append("")
    lines.extend([
        "The full sklearn panel is a separate run on its unchanged 72 fixtures and all nine "
        "operations. These controlled comparisons isolate source changes; absolute timings "
        "should not be compared across runs with different warmup protocols.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    text = render(report, args.input, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
