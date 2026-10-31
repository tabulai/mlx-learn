"""Render data-derived tables and a figure for any completed broad kNN run.

Unlike the historical narrative report, this generator contains no fixed timing
claims or result paths. It never runs benchmark or GPU workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .knn_broad_report import LABELS, span, table
from .knn_broad_summary import summarize_report

COMPARATORS = ("sklearn_default", "sklearn_auto", "sklearn_1")
NAMES = ("Brute, default threads", "Auto, default threads", "Brute, one thread")


def link(path, output, label):
    return f"[{label}]({os.path.relpath(Path(path).resolve(), output.parent.resolve())})"


def interval(values, digits=2, suffix="×"):
    return span(values, digits, suffix) if values["min"] is not None else "unavailable"


def label(family):
    return LABELS.get(family, family)


def make_plot(raw, summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    families, operations = summary["families"], summary["target_operations"]
    colors = ["#f4b0ad", "#ffe3a0", "#b4dace"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 1, 2, 1e6], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(families) * 0.45)), sharey=True)
    for ax, comparator, title in zip(axes, COMPARATORS, NAMES, strict=True):
        matrix = np.array([
            [min(c["operations"][op]["comparisons"][comparator]["ratio_of_medians"]
                 for c in raw["cases"] if c["spec"]["family"] == family["family"])
             for op in operations] for family in families
        ])
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=7.5)
        ax.set_title(title, fontsize=11, pad=12)
        names = {"fit_query": "Fit + query", "kneighbors": "Neighbors", "indices": "Indices",
                 "predict_uniform": "Class: uniform", "proba_uniform": "Proba: uniform",
                 "regress_uniform": "Regr: uniform", "predict_distance": "Class: distance",
                 "proba_distance": "Proba: distance", "regress_distance": "Regr: distance"}
        ax.set_xticks(range(len(operations)), [names.get(op, op) for op in operations],
                      rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(families)), [label(f["family"]) for f in families], fontsize=9)
        ax.set_xticks(np.arange(-0.5, len(operations), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("kNN speedup by dataset and operation", fontsize=20, x=0.39, y=0.99)
    automatic = summary.get("current_mode", "gpu") == "auto"
    engine = "mlxlearn" if automatic else "MLX"
    dispatch = "automatic CPU/GPU dispatch" if automatic else "forced MLX GPU"
    fig.text(0.235, 0.95, f"Each cell: minimum sklearn / {engine} median-time ratio across dataset seeds. Higher is better.", fontsize=11)
    fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(colors, [
        f"<1×: {engine} slower", "1–2×: below 2×", "≥2×: all seed medians reach 2×",
    ], strict=True)], loc="lower center", ncol=3, bbox_to_anchor=(0.58, 0.038), frameon=False)
    hardware = raw["workers"]["current"]["hardware"]["device_name"]
    fig.text(0.235, 0.016, f"{hardware} · float32 Euclidean · {dispatch} · "
             f"{raw['repeats']} retained calls after {raw['warmup']} warmups · "
             "raw speed ratios; see report for accuracy audit", fontsize=9)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.915, bottom=0.17, wspace=0.055)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _four_worker_control_sections(control):
    expected = {"current", "current_b", "pre_geometry", "distribution"}
    if set(control["variants"]) != expected:
        raise ValueError("Unsupported four-worker control variants")
    rows = []
    for family in control["families"]:
        for operation in control["operations"]:
            entry = family["operations"][operation]
            medians = entry["median_ms_ranges"]
            ratios = entry["ratio_of_medians_ranges"]
            paired = entry["median_paired_ratio_ranges"]
            rows.append([
                label(family["family"]), operation, ", ".join(map(str, family["seeds"])),
                *(interval(medians[v], 3, "") for v in ("current", "pre_geometry", "distribution")),
                interval(ratios["pre_geometry"], 3), interval(paired["pre_geometry"], 3),
                interval(ratios["distribution"], 3), interval(paired["distribution"], 3),
                interval(paired["current_b"], 3),
            ])
    archives = control["reference_archives"]
    return [
        "## Controlled comparison with both preceding sources",
        f"The pre-geometry baseline is `{Path(archives['pre_geometry']).name}`; it checks removal of the "
        f"Gaussian fresh-fit regression. The distribution-fixed baseline is `{Path(archives['distribution']).name}`; "
        "it checks preservation of the clustered/offset gains. Current, an identical-code A/A worker, and both "
        f"frozen baselines ran serially with {control['pairs']} interleaved paired rounds after "
        f"{control['warmup_seconds_per_variant_operation']:g} seconds of continuous warmup per worker/operation. "
        "These are contemporaneous comparisons on identical fixtures. Displayed ranges span the listed "
        "dataset seeds, including the additional Gaussian seeds recorded by the control harness.",
        "All speedups are baseline/current time. The two ratio columns per baseline distinguish the ratio of "
        "per-worker medians from the median of corresponding paired-call ratios; neither replaces the other. "
        "The A/A column is the median paired ratio of identical-code second-worker/current time. Small changes "
        "should be assessed against this process/timing variation and the underlying samples.",
        f"Control accuracy and integrity acceptance: **{control['acceptance_pass']}**. "
        f"Per-variant strict accuracy: `{json.dumps(control['quality_by_variant'], sort_keys=True)}`.",
        table([
            "Family", "Operation", "Seeds", "Current ms", "Pre-geometry ms", "Distribution-fixed ms",
            "Pre-geometry ratio of medians", "Pre-geometry median paired ratio",
            "Distribution-fixed ratio of medians", "Distribution-fixed median paired ratio", "A/A paired ratio",
        ], rows),
    ]


def control_sections(control, raw):
    if control["status"] != "complete":
        raise ValueError("Control report is incomplete")
    if control["source_hashes_before"] != raw["source_hashes_before"]:
        raise ValueError("Control and broad reports benchmark different production sources")
    if "worker_modes" in control:
        from .knn_auto_control_tables import validate

        validate(control)
        if raw.get("current_mode", "gpu") != "auto":
            raise ValueError("An automatic control must accompany an automatic broad run")
        rows = []
        for family in control["families"]:
            for operation, entry in family["operations"].items():
                medians = entry["median_ms_ranges"]
                ratios = entry["ratio_of_medians_ranges"]
                paired = entry["median_paired_ratio_ranges"]
                rows.append([
                    label(family["family"]), operation, ", ".join(map(str, family["seeds"])),
                    interval(medians["current"], 3, ""),
                    interval(ratios["current_gpu"], 3), interval(paired["current_gpu"], 3),
                    interval(ratios["previous_gpu"], 3), interval(paired["previous_gpu"], 3),
                    interval(paired["current_b"], 3),
                ])
        sections = [
            "## Controlled automatic dispatch and GPU comparison",
            f"{len(control['variants'])} serial workers compare current automatic dispatch, an identical automatic A/A worker, "
            "current forced GPU and the preceding frozen small-workload GPU implementation. "
            + ("A fifth worker uses the preceding automatic source and its exact archived native binary. "
               if "previous_auto" in control["variants"] else "") +
            f"The unchanged control covers {control['planned_cases']} fixtures and "
            f"{control['planned_case_operation_count']} case-operations, with {control['pairs']} paired rounds "
            f"after {control['warmup_seconds_per_variant_operation']:g} seconds of continuous warmup per "
            "worker/operation. " + control["warmup_policy"],
            "Ratios are comparator/automatic time. The ratio of per-worker medians and median paired "
            "ratio are shown separately. A/A exposes process/timing variation; these are descriptive "
            "comparisons, not confidence intervals. Each worker retains its actual routing metadata.",
            f"Control accuracy and integrity acceptance: **{control['acceptance_pass']}**. "
            f"Strict quality: `{json.dumps(control['quality_by_variant'], sort_keys=True)}`.",
            table(["Family", "Operation", "Seeds", "Auto ms", "Current GPU/auto medians",
                   "Current GPU/auto paired", "Previous GPU/auto medians", "Previous GPU/auto paired",
                   "Auto A/A paired"], rows),
        ]
        if "previous_auto" in control["variants"]:
            previous_rows = []
            for family in control["families"]:
                for operation, entry in family["operations"].items():
                    previous_rows.append([
                        label(family["family"]), operation, ", ".join(map(str, family["seeds"])),
                        interval(entry["median_ms_ranges"]["current"], 3, ""),
                        interval(entry["median_ms_ranges"]["previous_auto"], 3, ""),
                        interval(entry["ratio_of_medians_ranges"]["previous_auto"], 3),
                        interval(entry["median_paired_ratio_ranges"]["previous_auto"], 3),
                    ])
            sections += ["### Comparison with preceding automatic dispatch",
                         table(["Family", "Operation", "Seeds", "Current auto ms", "Previous auto ms",
                                "Previous/current medians", "Previous/current paired"], previous_rows)]
        return sections
    if "reference_archives" in control:
        return _four_worker_control_sections(control)
    rows = []
    for family in control["families"]:
        for operation in control["operations"]:
            entry = family["operations"][operation]
            rows.append([label(family["family"]), operation, ", ".join(map(str, family["seeds"])),
                         interval(entry["median_ms_ranges"]["current"], 3, ""),
                         interval(entry["median_ms_ranges"]["reference"], 3, ""),
                         interval(entry["median_paired_ratio_ranges"]["reference"]),
                         interval(entry["median_paired_ratio_ranges"]["current_b"])])
    return [
        "## Controlled comparison with the preceding source",
        f"The frozen reference is `{Path(control['reference_archive']).name}`. Current, an identical-code "
        f"A/A worker, and reference ran serially with {control['pairs']} interleaved paired rounds after "
        f"{control['warmup_seconds_per_variant_operation']:g} seconds of continuous warmup per worker/operation. "
        "These are contemporaneous comparisons on identical fixtures, not ratios of separate historical runs. "
        "Speedup is reference/current time; the A/A column uses identical-code second-worker/current time. "
        "Displayed ranges span dataset seeds. Near-parity differences should be assessed against A/A variation.",
        f"Control accuracy and integrity acceptance: **{control['acceptance_pass']}**. "
        f"Per-variant strict accuracy: `{json.dumps(control['quality_by_variant'], sort_keys=True)}`.",
        table(["Family", "Operation", "Seeds", "Current ms", "Reference ms", "Median paired speedup", "A/A paired ratio"], rows),
    ]


def render(raw, summary, args):
    if summary["target_speedup"] != 2:
        raise ValueError("This report's threshold labels require target_speedup=2")
    cases, family_count, repeats = summary["case_count"], summary["family_count"], summary["repeats"]
    count = cases * len(summary["target_operations"])
    best = sum(min(c["operations"][op]["comparisons"][v]["ratio_of_medians"] for v in COMPARATORS) >= 2
               for c in raw["cases"] for op in raw["target_operations"])
    gates = []
    for comparator, name in zip(COMPARATORS, NAMES, strict=True):
        gate = summary["gate_by_comparator"][comparator]
        medians, total = gate["median_cells_meeting_target"], gate["median_cell_count"]
        pairs, pair_total = gate["paired_calls_meeting_target"], gate["paired_call_count"]
        gates.append([name, f"{medians}/{total} ({medians/total:.1%})",
                      f"{pairs:,}/{pair_total:,} ({pairs/pair_total:.1%})",
                      str(gate["all_medians_meet_target"]), str(gate["every_pair_meets_target"])])
    all_medians = sum(all(f["comparators"][v]["all_medians_meet_target"] for v in COMPARATORS)
                      for f in summary["families"])
    all_pairs = sum(all(f["comparators"][v]["every_pair_meets_target"] for v in COMPARATORS)
                   for f in summary["families"])
    sections = [
        f"# {args.title}",
        f"Completed `{raw['finished_at']}`. This run covers **{cases} dataset instances in {family_count} families**, "
        f"{len(summary['target_operations'])} search/inference operations, and **{repeats} retained calls after "
        f"{summary['warmup']} warmups** per dataset-operation-variant. Fit-only is recorded separately and excluded "
        "from the headline 2× counts. Production source and fixture identities are recorded in the raw results.",
        table(["sklearn baseline", "Medians ≥2×", "Individual paired calls ≥2×", "All medians ≥2×?", "Every pair ≥2×?"], gates),
        f"Against the fastest median among the three measured sklearn configurations in each dataset-operation "
        f"cell, **{best}/{count} ({best/count:.1%})** reach 2×. This is a best-of-measured-configurations comparison, "
        "not one fixed sklearn setting. The panel weights each dataset-operation cell equally; percentages do "
        "not estimate real-world workload frequencies. Raw timing counts retain every baseline accuracy failure.",
        f"**{all_medians}/{family_count} families** reach 2× in every seed-operation median against all three "
        f"baselines; **{all_pairs}/{family_count}** also reach 2× in every retained paired call. A family passes "
        "only when all its constituent cases and operations pass, without averaging away misses.",
    ]
    if args.figure:
        make_plot(raw, summary, args.figure)
        sections.append("!" + link(args.figure, args.output, "Minimum speedup across seeds"))
    if args.control:
        sections.extend(control_sections(json.loads(args.control.read_text()), raw))
    worker = raw["workers"]["current"]
    pools = {v: raw["workers"][v]["threadpools"] for v in COMPARATORS}
    thread_counts = {v: [p["num_threads"] for p in pools[v] if p["user_api"] == "openmp"] for v in COMPARATORS}
    mode_description = (
        "The current path uses mlxlearn automatic CPU/GPU dispatch with the public brute-search API. "
        "Every call records its execution policy, fitted storage backend and actual query backend separately. "
        "Automatic calls must select the native CPU or MLX path; sklearn fallback is rejected. "
        if summary.get("current_mode", "gpu") == "auto" else
        "The current path explicitly forces the MLX GPU backend and brute search. These results do not evaluate "
        "automatic mlxlearn CPU/GPU dispatch. "
    )
    sections += [
        "## Timing and dataset scope",
        f"Hardware: {worker['hardware']['device_name']}, {worker['hardware']['memory_size']/2**30:g} GiB memory; "
        f"platform `{raw['platform']}`. Python `{worker['python'].split()[0]}`, MLX `{worker['mlx']}`, "
        f"NumPy `{worker['numpy']}`, sklearn `{worker['sklearn']}`. Reported OpenMP thread counts: "
        f"`{json.dumps(thread_counts, sort_keys=True)}`.",
        raw["protocol"],
        mode_description + "The 'cold' sample is first fit/query for a fixture in a persistent "
        "worker, not process/compiler startup. Timed work includes public validation, NumPy boundaries and GPU "
        "synchronization; IPC and numerical audits are outside timing. Auto's actual fitted algorithm is "
        "recorded per call. Distinct auto/brute workers may differ from timing variation even when both select brute.",
        "Speedup = sklearn median milliseconds / current median milliseconds. Below 1× means current is slower. "
        "Family ranges span seed-specific ratios, not ratios of independently selected timing endpoints. The "
        "paired-call count uses corresponding retained rounds. These are descriptive results, not confidence intervals.",
        "Fixtures cover the previous eleven Gaussian shapes; Gaussian, clusters, +1e4 offsets and unequal scales "
        "at 50k × 32 and 4096 × 128; 100k × 32 Gaussian; and bundled raw Digits, Breast Cancer, Iris and Wine. "
        "All inputs are float32 Euclidean data with no rescaling or jitter. Real data use stratified holdouts. "
        "Class labels reused numerically for real-data regression provide timing fixtures, not regression-quality "
        "evidence. Large real datasets, sparse input, other metrics, float64 and other hardware are outside this run.",
        "## All query workloads",
        "Ranges span seeds. The final column counts all search/inference operation medians reaching 2×, in the "
        "order default brute / auto / single-thread brute.",
    ]
    query_rows, fresh_rows, pair_rows = [], [], []
    for family in summary["families"]:
        name = label(family["family"])
        shape = family["shape"]
        dims = f"{shape['n_train']:,} × {shape['n_features']}; {shape['n_query']}; {shape['k']}"
        query, fresh = (family["query_and_fresh"][op] for op in ("kneighbors", "fit_query"))
        cells = family["comparators"][COMPARATORS[0]]["median_cell_count"]
        counts = " / ".join(str(family["comparators"][v]["median_cells_meeting_target"]) for v in COMPARATORS)
        query_rows.append([name, dims, interval(query["median_ms_range_by_variant"]["current"], 3, ""),
                           *(interval(query["ratio_of_medians_range_by_comparator"][v]) for v in COMPARATORS),
                           f"{counts} (each /{cells})"])
        fresh_rows.append([name, interval(fresh["median_ms_range_by_variant"]["current"], 3, ""),
                           *(interval(fresh["ratio_of_medians_range_by_comparator"][v]) for v in COMPARATORS)])
        pair_rows.append([name, *(f"{family['comparators'][v]['paired_calls_meeting_target']}/"
                                 f"{family['comparators'][v]['paired_call_count']}" for v in COMPARATORS)])
    sections += [table(["Family", "Train × features; queries; k", "Current query ms", "vs default brute", "vs auto", "vs one thread", "Medians ≥2×"], query_rows),
                 "## Fresh fit + query", table(["Family", "Current fresh ms", "vs default brute", "vs auto", "vs one thread"], fresh_rows),
                 "## Every operation and paired call"]
    if summary.get("current_mode", "gpu") == "auto":
        routes = summary["routing"]
        rows = [[phase, counts["request_count"], json.dumps(counts["query_backend_counts"], sort_keys=True),
                 json.dumps(counts["fit_backend_counts"], sort_keys=True),
                 json.dumps(counts["execution_policy_counts"], sort_keys=True)]
                for phase, counts in routes["by_phase"].items()]
        sections[-1:-1] = ["## Observed automatic routing", routes["scope"],
                          table(["Phase", "Requests", "Query backends", "Fit storage", "Policies"], rows),
                          "Per-family and per-operation routing counts are recorded in the validated summary."]
    operation_rows = []
    for op in summary["operations"]:
        entry = summary["per_operation"][op]
        operation_rows.append([f"`{op}`" + (" (excluded from headline)" if op == "fit" else ""), *(
            f"{entry['comparators'][v]['median_cells_meeting_target']}/{cases}; "
            f"{entry['comparators'][v]['paired_calls_meeting_target']}/{cases*repeats:,}" for v in COMPARATORS)])
    sections += ["Operation cells report median counts, then individual paired-call counts. Fit-only excludes "
                 "first-query preparation and is not a substitute for fresh fit + query.",
                 table(["Operation", *NAMES], operation_rows),
                 "Family-level paired-call counts retain every individual threshold miss:",
                 table(["Family", *NAMES], pair_rows), "## Accuracy and integrity", raw["accuracy_reference"]]
    audit_rows = []
    for variant in raw["variants"]:
        audit = summary["quality_by_variant"][variant]["all_phases"]
        kinds = audit["failure_kind_counts"]
        audit_rows.append([variant, f"{audit['strict_passes']:,}/{audit['audit_count']:,}",
                           json.dumps(kinds, sort_keys=True), json.dumps(audit["maximum_errors"], sort_keys=True)])
    sections += [table(["Variant", "Strict audit passes", "Failure-kind counts", "Maximum absolute errors"], audit_rows),
                 "Audit counts include cold, warmup and retained calls, and fit-only's untimed query audit. "
                 "They count repeated audit requests rather than distinct datasets. Exact-distance-valid "
                 "alternative ties are valid sklearn neighbor outputs; their canonical-index differences are "
                 "kept separate from incorrect membership, ordering and distance-tolerance failures. Prediction "
                 "differences on boundary ties are flagged as ambiguous and not automatically validated. "
                 "The summary contains per-phase counts and all failing case/operation evidence pointers. "
                 "Accuracy qualifications never remove timings from speed denominators.",
                 f"Integrity checks: `{json.dumps(summary['integrity'], sort_keys=True)}`. "
                 f"Current accuracy plus integrity acceptance: **{summary['accuracy_and_integrity_pass']}**.",
                 "## Artifacts", link(args.input, args.output, "Raw measurements and every audit")]
    if args.summary:
        sections.append(link(args.summary, args.output, "Validated derived summary"))
    if args.control:
        sections.append(link(args.control, args.output, "Controlled source comparison"))
    if args.figure:
        sections.append(link(args.figure.with_suffix(".svg"), args.output, "Vector figure"))
    sections += [f"Production manifest: `{raw['source_manifest']}`. "
                 f"Snapshot: `{json.dumps(raw['source_snapshot'], sort_keys=True)}`. "
                 "The raw report records source, loaded-worker, harness and fixture hashes before and after timing.",
                 "Generated from recorded measurements. No source optimization or performance claim is inferred "
                 "from a separate historical run."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Write the independently validated summary here")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--title", default="Broad kNN benchmark against three sklearn configurations")
    parser.add_argument("--expected-cases", type=int, default=72)
    parser.add_argument("--expected-repeats", type=int, default=15)
    args = parser.parse_args()
    raw_bytes = args.input.read_bytes()
    raw = json.loads(raw_bytes)
    summary = summarize_report(raw, args.input, hashlib.sha256(raw_bytes).hexdigest(),
                               args.expected_cases, args.expected_repeats)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    render(raw, summary, args)
    print(args.output)


if __name__ == "__main__":
    main()
