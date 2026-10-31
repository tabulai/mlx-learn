"""Render and independently validate the complete automatic CPU/GPU control."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .knn_auto_control import (
    EXTENDED_VARIANTS,
    EXTENDED_WORKER_MODES,
    TARGET_FAMILIES,
    VARIANTS,
    WORKER_MODES,
    _paired_ratios,
    _sample_pass,
    _summarize,
    _warmup_pass,
    family_summary,
    operations_for,
    routing_summary,
    select_specs,
)
from .knn_backend_audit import validate_record_backend
from .knn_order import phase_diagnostics, placement, round_order, williams_orders
from .knn_pipeline import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report, *, expected_cases=48):
    """Recompute raw summaries; retain valid reports containing failed audits."""
    require(report.get("status") == "complete", "Report must be complete")
    require(report.get("schema_version") == 1, "Unexpected schema")
    variants = tuple(report["variants"])
    require(variants in (VARIANTS, EXTENDED_VARIANTS), "Unexpected variants")
    modes = EXTENDED_WORKER_MODES if variants == EXTENDED_VARIANTS else WORKER_MODES
    order_policy = report.get("order_policy")
    if order_policy is not None:
        require(order_policy in ("legacy", "balanced") and report.get("order_telemetry_version") == 1,
                "Unexpected order policy/telemetry")
        require(report.get("order_block") == (williams_orders(variants) if order_policy == "balanced" else None),
                "Changed order design")
    require(report.get("worker_modes") == modes, "Unexpected dispatch modes")
    require(report.get("backend_telemetry_version") == 1, "Missing backend telemetry")
    require(all(report["workers"][v].get("current_mode") == modes[v] for v in variants),
            "Worker modes disagree with report")
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
    quality = dict.fromkeys(variants, True)
    for case in cases:
        require(case["case"] == case["spec"]["case"], "Conflicting case specification")
        require(set(case["experiments"]) == set(operations_for(case["spec"])), "Missing operation")
        require(set(case["cold"]) == set(variants), "Missing cold record")
        for variant in variants:
            validate_record_backend(case["cold"][variant], modes[variant])
            quality[variant] &= _sample_pass(case["cold"][variant])
        for experiment in case["experiments"].values():
            require(len(experiment["rounds"]) == pairs, "Missing round")
            previous_request = experiment.get("starting_order", [None])[-1]
            previous_completed = 0
            for pair, entry in enumerate(experiment["rounds"]):
                require(entry["pair"] == pair and sorted(entry["order"]) == sorted(variants),
                        "Invalid round order")
                if order_policy is not None:
                    require(entry["order"] == round_order(experiment["starting_order"], pair, order_policy),
                            "Retained order differs from declared design")
                    expected_placement = placement(entry["order"], previous_request)
                    require(entry["placement"] == expected_placement, "Wrong position/predecessor telemetry")
                    for name in entry["order"]:
                        context = experiment["variants"][name]["samples"][pair]["order_context"]
                        require(all(context[key] == value for key, value in expected_placement[name].items())
                                and context["pair"] == pair, "Sample position differs from its round")
                        require(previous_completed <= context["requested_at_ns"] < context["completed_at_ns"],
                                "Nonserial parent timing context")
                        elapsed = (context["requested_at_ns"] - experiment["warmup_probe_completed_at_ns"][name]) / 1e9
                        require(math.isfinite(elapsed) and elapsed >= 0
                                and context["seconds_since_warmup_probe"] == elapsed,
                                "Invalid time since warmup probe")
                        previous_completed = context["completed_at_ns"]
                    previous_request = entry["order"][-1]
            samples = {}
            for variant in variants:
                raw = experiment["variants"][variant]
                samples[variant] = raw["samples"]
                block = experiment["warmup"][variant]
                if "error" not in block:
                    require(len(block["warmup_backend_records"]) == block["warmup_count"],
                            "Missing per-call warmup backend telemetry")
                    for record in block["warmup_backend_records"] + block["samples"]:
                        validate_record_backend(record, modes[variant])
                for record in samples[variant]:
                    validate_record_backend(record, modes[variant])
                require(raw == _summarize(samples[variant], pairs), "Inconsistent timing summary")
                require(len(samples[variant]) == pairs, "Missing retained request")
                passed = (raw["quality_pass"]
                          and _warmup_pass(experiment["warmup"][variant], duration))
                require(experiment["quality_by_variant"][variant] == passed,
                        "Inconsistent experiment quality")
                quality[variant] &= passed
            for other in variants[1:]:
                expected = _paired_ratios(samples, other, pairs)
                current_ms = experiment["variants"]["current"]["median_ms"]
                other_ms = experiment["variants"][other]["median_ms"]
                expected["ratio_of_medians"] = (
                    other_ms / current_ms if current_ms is not None and other_ms is not None else None
                )
                require(experiment["comparisons"][f"current_vs_{other}"] == expected,
                        "Inconsistent paired comparison")
            if order_policy is not None:
                require(experiment["phase_diagnostics"] == phase_diagnostics(experiment, variants),
                        "Phase/position diagnostics differ from retained samples")
    require(report["quality_by_variant"] == quality, "Inconsistent overall quality")
    require(report["quality_pass"] == all(quality.values()), "Inconsistent quality gate")
    require(report["families"] == family_summary(cases, variants), "Inconsistent family summary")
    require(report["routing"] == routing_summary(cases, variants), "Inconsistent routing summary")
    gates = ("quality_pass", "complete_case_count_pass", "source_integrity_pass",
             "current_source_integrity_pass", "fixture_integrity_pass",
             "harness_integrity_pass", "archive_integrity_pass")
    if "reference_runtime_integrity_pass" in report:
        gates += ("reference_runtime_integrity_pass",)
    if "previous_auto" in variants:
        require(report["reference_runtime_integrity_pass"], "Reference runtime changed")
        from .knn_reference_runtime import validate_reference_worker
        validate_reference_worker(report["workers"]["previous_auto"], report["reference_runtimes"]["previous_auto"])
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
        "# Automatic CPU/GPU controlled benchmark",
        "",
        f"Completed `{report['finished_at']}`. Acceptance: **{report['acceptance_pass']}**. "
        f"[Raw results]({raw_link}) (SHA-256 `{sha256(raw_path.read_bytes())}`).",
        "",
        f"{len(report['variants'])} isolated public-API workers: current automatic dispatch, an identical automatic A/A "
        "worker, current forced GPU and frozen forced-GPU `knn-small-final-2026-10-30.zip`. "
        + ("An additional previous-auto worker uses the preceding source and exact ABI-matched archived native binary. "
           if "previous_auto" in report["variants"] else "") +
        "All CPU/GPU calls are serial. "
        f"Each worker/operation gets {report['warmup_seconds_per_variant_operation']:g} seconds "
        f"of continuous warmup followed by {report['pairs']} interleaved paired rounds. "
        "Every call includes public validation, NumPy boundaries and synchronization. "
        "Cold first-fit/query records come from persistent workers, excluding process startup. "
        + report["warmup_policy"],
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
    if "order_policy" in report:
        lines.extend(["## Retained request order and phase diagnostics", "",
                      f"Order policy: **{report['order_policy']}**. " + report["order_design"], "",
                      "The raw file keeps each request's position, within-round predecessor, actual previous "
                      "request, and parent elapsed time since that worker's warmup probe. Each operation also "
                      "contains all/first/middle/last and even/odd-round summaries, plus summaries by position "
                      "and predecessor. These diagnostics do not remove rounds, infer clock state or establish "
                      "statistical significance. Historical results without this field retain their original "
                      "legacy order; absolute timings must not be compared as if the protocols were identical.", ""])
    for target, title in ((True, "Small datasets and single queries"),
                          (False, "Gaussian, clustered, offset and unequal-scale preservation")):
        lines.extend([
            f"## {title}", "",
            "| Family | Operation | Seeds | Auto ms | Current GPU ms | Previous GPU ms | "
            "Current GPU/auto medians | Current GPU/auto paired | Previous GPU/auto medians | "
            "Previous GPU/auto paired | Auto A/A paired |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                             f"{span(times['current'])} | {span(times['current_gpu'])} | {span(times['previous_gpu'])} | "
                             f"{span(median['current_gpu'], '×')} | {span(paired['current_gpu'], '×')} | "
                             f"{span(median['previous_gpu'], '×')} | {span(paired['previous_gpu'], '×')} | "
                             f"{span(paired['current_b'], '×')} |")
        lines.append("")
    if "previous_auto" in report["variants"]:
        lines.extend(["## Preceding automatic implementation", "",
                      "The prior automatic source and its exact native runtime run in an isolated worker. "
                      "Speedups are previous-auto/current-auto time, with every seed and operation retained.", "",
                      "| Family | Operation | Seeds | Current auto ms | Previous auto ms | Median ratio | Paired ratio |",
                      "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
        for family in report["families"]:
            for operation, stats in family["operations"].items():
                lines.append(f"| `{family['family']}` | {operation} | {family['seeds']} | "
                             f"{span(stats['median_ms_ranges']['current'])} | "
                             f"{span(stats['median_ms_ranges']['previous_auto'])} | "
                             f"{span(stats['ratio_of_medians_ranges']['previous_auto'], '×')} | "
                             f"{span(stats['median_paired_ratio_ranges']['previous_auto'], '×')} |")
        lines.append("")
    lines.extend(["## Observed backend routing", "",
                  "Counts include every warmup, probe and retained query, with cold calls separate. "
                  "Fit storage is reported independently of actual query execution.", "",
                  "| Variant | Phase | Requests | Query backends | Fit storage |",
                  "| --- | --- | ---: | --- | --- |"])
    for variant, phases in report["routing"].items():
        for phase, counts in phases.items():
            lines.append(f"| {variant} | {phase} | {counts['request_count']} | "
                         f"`{json.dumps(counts['query_backend_counts'], sort_keys=True)}` | "
                         f"`{json.dumps(counts['fit_backend_counts'], sort_keys=True)}` |")
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
