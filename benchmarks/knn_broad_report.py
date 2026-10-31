"""Render the completed broad kNN measurements; never run timed workloads."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-10-30"
COMPARATORS = ("sklearn_default", "sklearn_auto", "sklearn_1")
NAMES = ("Brute, default 16 threads", "Auto, default 16 threads", "Brute, one thread")
LABELS = {
    "shape_single_8_k5": "Gaussian 8D, single query",
    "shape_batch_8_k5": "Gaussian 8D, k5",
    "shape_batch_32_k5": "Gaussian 32D, k5",
    "shape_batch_32_k32": "Gaussian 32D, k32",
    "shape_batch_128_k5": "Gaussian 128D, k5",
    "shape_batch_1024_k5": "Gaussian 1024D, k5",
    "shape_batch_32_k64": "Gaussian 32D, k64",
    "shape_batch_16_k5": "Gaussian 16D, k5",
    "shape_single_32_k5": "Gaussian 32D, single query",
    "shape_batch_8_k32": "Gaussian 8D, k32",
    "shape_batch_50000_32_k5": "Gaussian 50k × 32 (original panel)",
    "distribution_gaussian_50000x32_k5": "Gaussian 50k × 32",
    "distribution_clusters_50000x32_k5": "Clusters 50k × 32",
    "distribution_offset_1e4_50000x32_k5": "Offset +1e4, 50k × 32",
    "distribution_anisotropic_50000x32_k5": "Unequal scales 50k × 32",
    "distribution_gaussian_4096x128_k5": "Gaussian 4096 × 128",
    "distribution_clusters_4096x128_k5": "Clusters 4096 × 128",
    "distribution_offset_1e4_4096x128_k5": "Offset +1e4, 4096 × 128",
    "distribution_anisotropic_4096x128_k5": "Unequal scales 4096 × 128",
    "large_gaussian_100000x32_k5": "Gaussian 100k × 32",
    "real_digits_raw_k5": "Digits (raw)",
    "real_breast_cancer_raw_k5": "Breast Cancer (raw)",
    "real_iris_raw_k5": "Iris (raw)",
    "real_wine_raw_k5": "Wine (raw)",
}


def span(values, digits=2, suffix="×"):
    return f"{values['min']:.{digits}f}–{values['max']:.{digits}f}{suffix}"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def plot(raw, summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    operations = summary["target_operations"]
    families = summary["families"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 11), sharey=True)
    colors = ["#f4b0ad", "#ffe3a0", "#b4dace"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 1, 2, 1e6], cmap.N)
    for ax, comparator, title in zip(axes, COMPARATORS, NAMES, strict=True):
        matrix = np.array([
            [min(c["operations"][op]["comparisons"][comparator]["ratio_of_medians"]
                 for c in raw["cases"] if c["spec"]["family"] == family["family"])
             for op in operations]
            for family in families
        ])
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=7.5)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xticks(range(len(operations)), [
            "Fit + query", "Neighbors", "Indices", "Class: uniform", "Proba: uniform",
            "Regr: uniform", "Class: distance", "Proba: distance", "Regr: distance",
        ], rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(families)), [LABELS[f["family"]] for f in families], fontsize=9)
        ax.set_xticks(np.arange(-0.5, len(operations), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("kNN speedup is workload-dependent", fontsize=20, x=0.34, y=0.99)
    fig.text(0.235, 0.95, "Each cell: minimum sklearn / MLX median-time ratio across 3 seeds. Higher is better.", fontsize=11)
    fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(colors, [
        "<1×: MLX slower", "1–2×: below target", "≥2×: target met for all 3 seed medians",
    ], strict=True)], loc="lower center", ncol=3, bbox_to_anchor=(0.58, 0.038), frameon=False)
    fig.text(0.235, 0.016, "Apple M4 Max · float32 Euclidean · forced MLX GPU · 15 calls after 5 warmups · raw speed ratios; see report for accuracy audit", fontsize=9)
    fig.subplots_adjust(left=0.235, right=0.99, top=0.915, bottom=0.17, wspace=0.055)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    raw = json.loads((ROOT / f"benchmarks/results/knn-broad-{STAMP}.json").read_text())
    summary = json.loads((ROOT / f"benchmarks/results/knn-broad-summary-{STAMP}.json").read_text())
    assert raw["status"] == "complete" and summary["status"] == "complete"
    assert all(summary["integrity"].values())
    rows = []
    for comparator, label in zip(COMPARATORS, NAMES, strict=True):
        gate = summary["gate_by_comparator"][comparator]
        count, total = gate["median_cells_meeting_target"], gate["median_cell_count"]
        pairs, pair_total = gate["paired_calls_meeting_target"], gate["paired_call_count"]
        rows.append([label, f"{count}/{total} ({count/total:.1%})", f"{pairs:,}/{pair_total:,} ({pairs/pair_total:.1%})", "No"])
    best = sum(min(c["operations"][op]["comparisons"][v]["ratio_of_medians"] for v in COMPARATORS) >= 2
               for c in raw["cases"] for op in raw["target_operations"])
    sections = [
        "# Broad kNN benchmark against sklearn — October 30, 2026",
        "**No: the new MLX GPU version is not consistently 2× faster than sklearn across varied workloads.** "
        "The earlier 2× result applied to the 50,000-row, 32-feature Gaussian workload. This broader run "
        "covers 72 dataset instances in 24 families, three seeds each, and nine search/inference operations. "
        "Large Gaussian and unequal-scale batches retain substantial gains; clustered, offset, small-real-data "
        "and single-query workloads expose limits. Production source was unchanged.",
        table(["sklearn baseline", "Dataset-operation medians ≥2×", "Individual paired calls ≥2×", "Consistently ≥2×?"], rows),
        f"If sklearn is allowed the fastest median of these three measured configurations for each dataset and operation, "
        f"only **{best}/648 ({best/648:.1%})** comparisons reach 2×. This is a best-of-measured-configurations comparison, "
        "not a claim that one fixed sklearn configuration achieves every best result. The percentages describe this "
        "equally weighted benchmark panel, not the frequency of workloads in real applications.",
        "All **14,472 current-version audits pass** (72 first-query, 3,600 warmup, 10,800 retained, including fit-only). "
        "Source, loaded-worker, fixture and harness hashes match before and after the run. Sklearn has some numerical "
        "and canonical-tie audit differences detailed below; all its timings remain in the speed tables.",
        "![Minimum speedup across seeds](assets/knn-broad-2026-10-30.png)",
        "## What holds up",
        "- Gaussian 50k × 32 queries: **4.29–4.90×** versus default brute; fresh fit + query: **3.48–4.42×**. "
        "Both independent seed sets pass every retained paired call for all nine operations against all three sklearn configurations.\n"
        "- Gaussian 100k × 32 queries: **4.30–4.97×**; fresh: **3.48–3.70×** versus default brute.\n"
        "- Unequal-scale 50k × 32 queries: **3.46–3.73×**; fresh: **3.05–3.15×** versus default brute.\n"
        "- Established Gaussian 128-feature queries: **3.22–4.16×** versus default brute, but only **1.69–1.78×** versus single-thread brute.\n"
        "- Offset +1e4 at 50k × 32: **0.86–0.88×** query and **0.82–0.90×** fresh versus default brute.\n"
        "- Clusters at 4096 × 128: **0.96–1.02×** query and **0.88–0.93×** fresh versus default brute.\n"
        "- Iris and Wine queries: **0.35–0.56×** and **0.51–0.68×** versus sklearn auto, which selects KD-tree. "
        "Single-thread brute is faster still on these small datasets.",
        "Only **6/24 families** pass all 27 seed-operation medians against all three baselines: Gaussian batches at "
        "16D and 32D, both 50k Gaussian seed sets, 50k unequal scales, and 100k Gaussian. Five also pass every "
        "retained pair; the 8192 × 32 Gaussian family misses one default-brute pair. This is a workload-specific "
        "acceleration, not a universal sklearn replacement speed guarantee.",
        "## Protocol and scope",
        "Apple M4 Max, 128 GiB unified memory, macOS 26.6.2; Python 3.13.3, MLX 0.32.2, NumPy 2.5.2, "
        "sklearn 1.9.0. Four persistent isolated workers run serially. Current explicitly uses the MLX GPU "
        "backend with brute search; sklearn uses brute/default 16 OpenMP threads, brute/one thread, or auto/default "
        "threads. Auto's actual fitted algorithm is retained per call. This measures the forced GPU path, not "
        "mlxlearn's automatic CPU/GPU dispatch.",
        "Each dataset-operation-variant gets five warmups and 15 retained calls. Variant order rotates by case "
        "and operation and reverses each round. Timing surrounds the public NumPy-input/output call with "
        "synchronization; validation, conversion and GPU transfers are included, worker IPC and oracle checks "
        "are excluded. Fresh fit + query constructs a new estimator each time and includes deferred preparation. "
        "Repeated query/prediction uses a fitted estimator; no result cache is used. Fit-only is separate. "
        "The recorded 'cold' call is first fit/query per fixture in a persistent worker, not a fresh-process "
        "compiler startup measurement. Default and auto workers that both choose brute can differ through "
        "ordinary timing variation; that difference is not an algorithmic auto benefit.",
        "Speedup means sklearn median milliseconds / current median milliseconds, so below 1× means MLX is slower. "
        "Each family range spans its three seed-specific ratios; it is not a ratio of independently selected "
        "range endpoints. Individual paired-call gates use corresponding retained rounds. No confidence "
        "intervals or universal statistical claim are inferred from these descriptive timings.",
        "The panel includes the previous eleven Gaussian shapes (1–256 queries, 8–1024 features, k=5/32/64); "
        "Gaussian, 16-cluster, +1e4-offset and unequal-scale distributions at 50k × 32 and 4096 × 128; a 100k × 32 "
        "Gaussian workload; and bundled Digits, Breast Cancer, Iris and Wine. Unequal feature multipliers span "
        "1e-2 to 1e2. Synthetic batched fixtures include two copied training rows as queries to check zero distances. "
        "Real data use raw float32 features and deterministic stratified holdouts, with no scaling or jitter. "
        "Real classification labels are also used numerically for the regression timing exercise; those "
        "regression timings are not evidence of regression-model quality. Large real datasets, sparse input, "
        "other metrics, float64 and other hardware are outside this run. Exact seeds and fixture hashes are in the raw JSON.",
        "## All query workloads",
        "Ranges below span three seeds. The final column counts all nine operation medians reaching 2× out of "
        "27, in the order default brute / auto / single-thread brute. Query includes distances and indices.",
    ]
    query_rows, fresh_rows, pair_rows = [], [], []
    for family in summary["families"]:
        label = LABELS[family["family"]]
        shape = family["shape"]
        dims = f"{shape['n_train']:,} × {shape['n_features']}; {shape['n_query']}; {shape['k']}"
        query = family["query_and_fresh"]["kneighbors"]
        fresh = family["query_and_fresh"]["fit_query"]
        counts = " / ".join(str(family["comparators"][v]["median_cells_meeting_target"]) for v in COMPARATORS)
        query_rows.append([label, dims, span(query["median_ms_range_by_variant"]["current"], 3, ""),
                           *(span(query["ratio_of_medians_range_by_comparator"][v]) for v in COMPARATORS), counts])
        fresh_rows.append([label, span(fresh["median_ms_range_by_variant"]["current"], 3, ""),
                           *(span(fresh["ratio_of_medians_range_by_comparator"][v]) for v in COMPARATORS)])
        pair_rows.append([label, *(f"{family['comparators'][v]['paired_calls_meeting_target']}/405" for v in COMPARATORS)])
    sections += [table(["Family", "Train × features; queries; k", "Current query ms", "vs default brute", "vs auto", "vs one thread", "Medians ≥2× /27"], query_rows),
                 "## Fresh fit + query", table(["Family", "Current fresh ms", "vs default brute", "vs auto", "vs one thread"], fresh_rows),
                 "## Every operation and retained pair"]
    operation_rows = []
    for op in summary["operations"]:
        entry = summary["per_operation"][op]
        operation_rows.append([f"`{op}`" + (" (excluded from headline)" if op == "fit" else ""), *(
            f"{entry['comparators'][v]['median_cells_meeting_target']}/72; "
            f"{entry['comparators'][v]['paired_calls_meeting_target']}/1,080" for v in COMPARATORS)])
    sections += ["Each entry is median cells ≥2× /72; individual paired calls ≥2× /1,080. Uniform and distance "
                 "refer to neighbor weighting. Fit-only excludes first-query preparation and is not a substitute for fresh fit + query.",
                 table(["Operation", *NAMES], operation_rows),
                 "Family-level individual-call counts below include nine operations × three seeds × 15 retained pairs; "
                 "no median or mean hides a slower call.", table(["Family", *NAMES], pair_rows),
                 "## Precision and tie audit",
                 "The independent oracle exhaustively computes direct float64 squared differences on the actual "
                 "stored float32 coordinates, then orders by distance and training index. Neighbor indices must "
                 "match exactly; distance atol/rtol are 1e-6. Class predictions must match exactly; probabilities "
                 "and regression use atol 2e-6 and rtol 2e-5. Current passes every audit. Its maximum absolute "
                 "distance error is 2.296e-5 at larger distances, within the combined absolute/relative tolerance; "
                 "maximum probability/regression absolute error is 3.903e-8."]
    audit_rows = []
    for variant in raw["variants"]:
        a = summary["quality_by_variant"][variant]["all_phases"]
        kinds = a["failure_kind_counts"]
        audit_rows.append([variant, f"{a['strict_passes']:,}/{a['audit_count']:,}",
                           kinds.get("valid_exact_tie_alternative", 0), kinds.get("reported_distance_error", 0),
                           kinds.get("canonical_prediction_mismatch", 0),
                           kinds.get("canonical_prediction_mismatch_on_boundary_ties", 0)])
    sections += [table(["Variant", "Strict passes", "Valid alternative tie choices", "Distance tolerance misses", "Prediction misses away from boundary ties", "Prediction misses on boundary ties"], audit_rows),
                 "Counts are repeated audit calls across cold, warmup and retained phases, including fit-only's "
                 "untimed verification query, not counts of distinct defective datasets. There are **no incorrect "
                 "neighbor-membership or distance-order failures** in any variant. Sklearn's 405 canonical tie "
                 "differences per variant are exact-distance-valid alternatives on Digits/Iris and are not "
                 "correctness bugs. The 80 additional single-thread prediction differences occur on true boundary "
                 "ties in Digits seed 251; they are flagged as ambiguous, not automatically certified equivalent. "
                 "Sklearn distance tolerance misses occur on 13 dataset instances, including copied-row residuals "
                 "in high-dimensional/offset data; maximum absolute error is 0.007307925. Non-boundary prediction "
                 "misses occur on five dataset instances. The summary retains every case/operation and raw JSON "
                 "pointer, with separate phase counts. Baseline strict-quality flags remain false, and speed "
                 "tables are raw timing comparisons rather than equal-accuracy certifications for those cells.",
                 "## Reproduction and artifacts",
                 "- [Raw results, all calls and audits](../benchmarks/results/knn-broad-2026-10-30.json)\n"
                 "- [Compact derived summary and evidence pointers](../benchmarks/results/knn-broad-summary-2026-10-30.json)\n"
                 "- [Execution record](../benchmarks/results/knn-broad-execution-2026-10-30.json) and [run log](../benchmarks/results/knn-broad-2026-10-30.log)\n"
                 "- [Frozen production source](../benchmarks/reference/knn-128-final-2026-10-30.zip) and [manifest](../benchmarks/reference/knn-128-final-2026-10-30.manifest.json)\n"
                 "- [Frozen six-file benchmark harness](../benchmarks/reference/knn-broad-harnesses-2026-10-30.zip) and [manifest](../benchmarks/reference/knn-broad-harnesses-2026-10-30.manifest.json)\n"
                 "- [Summary script](../benchmarks/knn_broad_summary.py), [report/plot generator](../benchmarks/knn_broad_report.py), [SVG figure](assets/knn-broad-2026-10-30.svg)\n"
                 "- [Previous v3 comparison and 128-feature fix](knn-128-feature-fix-2026-10-30.md)",
                 "This task added benchmark and reporting code only. The prior 652-test production validation is "
                 "documented in the preceding report; it was not rerun or counted as new validation here. New "
                 "benchmark validation includes an audited smoke run, full-run audits, summary completeness/counter "
                 "checks, Ruff and source-integrity checks. The execution record reports a 319.7-second full run "
                 "with exit code 0. The final source archive SHA-256 is "
                 "`920753a01375683bcfe776e6bbb79765f6f237471913ce8155a6bdc2806cf7ca`.",
                 "```sh\npython3 -m benchmarks.knn_broad --repeats 15 --warmup 5 \\\n  --output benchmarks/results/knn-broad-2026-10-30.json\n"
                 "python3 -m benchmarks.knn_broad_summary \\\n  --input benchmarks/results/knn-broad-2026-10-30.json \\\n  --output benchmarks/results/knn-broad-summary-2026-10-30.json\n"
                 "python3 -m benchmarks.knn_broad_report\n```"]
    output = ROOT / f"docs/knn-broad-benchmark-{STAMP}.md"
    output.write_text("\n\n".join(sections) + "\n")
    plot(raw, summary, ROOT / f"docs/assets/knn-broad-{STAMP}.png")
    print(output)


if __name__ == "__main__":
    main()
