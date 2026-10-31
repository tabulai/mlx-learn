"""Render validated fresh-process latency without hiding setup costs or misses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .knn_broad_report import LABELS, table
from .knn_cold_summary import summarize_report

NAMES = {
    "current": "Current automatic",
    "sklearn_default": "Default-thread brute",
    "sklearn_1": "Single-thread brute",
    "sklearn_auto": "sklearn auto",
    "previous_auto": "Preceding automatic",
}
SECTIONS = (
    ("query_ms", "First query", "First public query after the newly constructed estimator's first fit."),
    ("fit_query_ms", "First fit + query", "Constructor, first fit and first query. Estimator imports/setup are separate."),
    ("estimator_setup_ms", "Estimator import and setup", "Normal estimator dependencies and configuration setup, including native initialization loaded by those imports."),
    ("estimator_to_result_ms", "Estimator import through first result", "Estimator import/setup plus first fit/query. Moving native initialization into imports does not remove it from this measure."),
    ("process_start_to_result_ms", "Process start through first result", "Parent-observed interpreter startup through the first result marker: includes NumPy import, fixture X/Q loading, estimator setup, fit/query and IPC; excludes the later audit and metadata."),
)


def number(value):
    return "missing" if value is None else f"{value:.3f}"


def ratio(value):
    if value is None:
        return "**missing**"
    text = f"{value:.3f}×"
    return f"**{text}**" if value < 1 else text


def link(path, destination, label):
    return f"[{label}]({os.path.relpath(Path(path).resolve(), Path(destination).resolve().parent)})"


def build_report(raw, *, raw_path, output_path, broad=None, expected_cases=18, expected_repeats=5,
                 title="Fresh-process KNN first-use benchmark"):
    summary = summarize_report(raw, broad=broad, expected_cases=expected_cases, expected_repeats=expected_repeats)
    variants = raw["variants"]
    comparators = [v for v in variants if v != "current"]
    lines = [
        f"# {title}",
        f"Completed `{raw.get('finished_at', 'not recorded')}`. This report retains "
        f"**{summary['launch_count']} separate process launches** over {summary['case_count']} fixtures, "
        f"{len(variants)} configurations and {summary['repeats']} launches per fixture/configuration. "
        f"Accuracy/integrity acceptance: **{summary['acceptance_pass']}**.",
        raw["protocol"], summary["interpretation"],
        "All times are milliseconds. Speedup is comparator time / current automatic time; "
        "a value below 1× means current is slower. Bold ratios mark every slower or missing median. "
        "No failed numerical audit removes a timing from the comparison. These are descriptive "
        "process-cold measurements, not a universal first-use latency guarantee or a machine-cold test.",
    ]
    overview = []
    for metric, title_label, _ in SECTIONS:
        for variant in comparators:
            entry = summary["comparisons"][variant][metric]
            overview.append([title_label, NAMES[variant], f"{entry['faster_medians']}/{entry['cell_count']}",
                             ratio(entry["minimum_speedup"]), ratio(entry["maximum_speedup"])])
    lines += ["## Coverage by separately measured phase",
              table(["Phase", "Comparator", "Current medians faster", "Minimum speedup", "Maximum speedup"], overview)]
    for metric, heading, scope in SECTIONS:
        rows = []
        for case in raw["cases"]:
            spec = case["spec"]
            current = case["variants"]["current"]["metrics"][metric]
            row = [LABELS.get(spec["family"], spec["family"]), str(spec["seed"]),
                   number(current["median_ms"]), f"{number(current['min_ms'])}–{number(current['max_ms'])}"]
            for variant in comparators:
                comparator = case["variants"][variant]["metrics"][metric]["median_ms"]
                speedup = case["comparisons"][variant][metric]["ratio_of_medians"]
                row.append(f"{number(comparator)} / {ratio(speedup)}")
            rows.append(row)
        lines += [f"## {heading}", scope,
                  "Each comparator column shows its median milliseconds, then comparator/current speedup. "
                  "The current range spans all retained process launches for that fixture.",
                  table(["Fixture family", "Seed", "Current median ms", "Current min–max ms",
                         *(NAMES[v] + " ms / ratio" for v in comparators)], rows)]
    missing_rows = []
    for metric, heading, _ in SECTIONS:
        for variant in comparators:
            for entry in summary["comparisons"][variant][metric]["slower_or_missing_cells"]:
                missing_rows.append([heading, NAMES[variant], f"`{entry['case']}`", number(entry["current_ms"]),
                                     number(entry["comparator_ms"]), ratio(entry["ratio_of_medians"]),
                                     ratio(entry["median_paired_ratio"])])
    lines += ["## Every slower or missing median",
              "Every below-1 or missing median from the five phase tables is repeated here. "
              "Paired ratios compare the corresponding serial launch rounds; their median need not "
              "equal the ratio of separate medians and is not a confidence interval.",
              table(["Phase", "Comparator", "Case", "Current ms", "Comparator ms", "Median ratio", "Paired ratio"], missing_rows)
              if missing_rows else "No slower or missing medians in these five measured phases."]
    audit_rows = []
    for variant in variants:
        count = summary["audit_counts"][variant]
        audit_rows.append([NAMES[variant], count["requests"], count.get("completed_timings", 0),
                           count.get("strict_passes", 0), count.get("errors", 0),
                           json.dumps(summary["query_backend_counts"][variant], sort_keys=True),
                           json.dumps(summary["native_loaded_before_fit_counts"][variant], sort_keys=True),
                           json.dumps(summary["sklearn_neighbors_loaded_before_fit_counts"][variant], sort_keys=True)])
    lines += ["## Audits, actual routing and native initialization",
              table(["Configuration", "Launches", "Valid timings", "Strict passes", "Errors",
                     "Query backends", "Native already loaded before fit", "Stock sklearn.neighbors loaded before fit"], audit_rows),
              "A true native-loaded-before-fit flag means normal estimator imports/setup loaded the extension. "
              "The setup and total-latency tables retain this cost. The harness performs no native prewarm. "
              "Each raw observation also records module presence before NumPy input preparation and after it. "
              "Stock sklearn.neighbors presence is recorded separately from the top-level sklearn package; "
              "older evidence lacking that field remains labelled not_recorded."]
    failures = []
    for ci, case in enumerate(raw["cases"]):
        for variant, entry in case["variants"].items():
            for si, sample in enumerate(entry["samples"]):
                audit = sample.get("audit", {})
                if audit.get("pass", False) and "error" not in sample:
                    continue
                kind = "worker_error" if "error" in sample else audit.get("failure_kind", "strict_audit_failure")
                failures.append([NAMES[variant], f"`{case['case']['name']}`", si + 1, kind,
                                 f"`/cases/{ci}/variants/{variant}/samples/{si}`"])
    lines += ["### Every failed observation",
              table(["Configuration", "Case", "Launch", "Failure", "Raw JSON pointer"], failures)
              if failures else "Every retained launch passes its strict numerical audit."]
    input_rows = []
    for case in raw["cases"]:
        for variant in variants:
            metrics = case["variants"][variant]["metrics"]
            input_rows.append([f"`{case['case']['name']}`", NAMES[variant],
                               *(number(metrics[m]["median_ms"]) for m in (
                                   "numpy_import_ms", "fixture_load_ms", "fit_ms", "worker_entry_to_result_ms", "process_elapsed_ms",
                               ))])
    lines += ["## Input preparation, fit return and post-result process lifetime",
              "These columns keep the remaining recorded phases visible. Fit return may precede deferred "
              "preparation charged to first query. Full process elapsed includes audits, metadata and process "
              "exit after the result marker, so it is not an inference-latency measure.",
              table(["Case", "Configuration", "NumPy import ms", "Fixture load ms", "First fit ms",
                     "Worker entry→result ms", "Full process elapsed ms"], input_rows),
              "## Evidence", link(raw_path, output_path, "Every raw launch, audit and runtime identity")]
    return "\n\n".join(lines) + "\n", summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--broad", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Fresh-process KNN first-use benchmark")
    parser.add_argument("--expected-cases", type=int, default=18)
    parser.add_argument("--expected-repeats", type=int, default=5)
    args = parser.parse_args()
    for destination in (args.output, args.summary):
        if destination is not None and destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing evidence: {destination}")
    data = args.input.read_bytes()
    broad = json.loads(args.broad.read_text()) if args.broad else None
    text, summary = build_report(json.loads(data), raw_path=args.input, output_path=args.output, broad=broad,
                                 expected_cases=args.expected_cases or None, expected_repeats=args.expected_repeats or None,
                                 title=args.title)
    summary["raw_evidence"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        text += "\n" + link(args.summary, args.output, "Validated derived summary") + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
