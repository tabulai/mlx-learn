"""Report only fresh, provenance-matched previous/current Metal comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gap_bench as bench  # noqa: E402
import h100_report as protocol  # noqa: E402

common = protocol.common
KIND = "metal_gap_report_manifest"


def evidence_rows(folder):
    rows = []
    for path in sorted(Path(folder).rglob("*.json")):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and "config" in row and "train_seconds" in row:
            row["_path"] = path.resolve()
            rows.append(row)
    return rows


def aggregate_identity(row):
    """Remove seed and mode, keeping the explicit two-runtime comparison."""
    config = {key: value for key, value in row["config"].items()
              if key not in {"mode", "seed", "expected_source_sha256", "output", "worker_output"}}
    identity = {"config": config, "recipe": {key: value for key, value in row["recipe_resolved"].items() if key != "seed"}}
    identity.update({key: row.get(key) for key in ("n", "d", "k", "dataset", "versions", "hardware", "metal_device",
                                                 "faiss_reference_revision", "timing_scope", "scoring", "memory_scope")})
    return common.stable(identity)


def campaign_identity(row):
    config = row["config"]
    value = {key: config.get(key) for key in ("runtime_comparison", "shared_source_sha256", "expected_versions",
             "expected_host", "archive_sha256", "precision", "recipe", "finalization", "assignment", "warmup",
             "capture_assignments", "threads", "train_score_rows", "memory", "memory_interval_seconds")}
    value.update({key: row.get(key) for key in ("metal_device", "faiss_reference_revision", "timing_scope", "scoring", "memory_scope")})
    return common.stable(value)


def validate_and_pair(rows):
    candidates, excluded = defaultdict(list), []
    for row in rows:
        try:
            if row.get("kind") != bench.KIND:
                raise ValueError("historical or unrelated measurement kind")
            # Both gap variants run native BF16 auto assignment. Reuse the full
            # workload/precision checks without disguising their source hashes.
            view = {**row, "config": {**row["config"], "mode": "optimized"}}
            reason = protocol.problem(view)
            if reason:
                raise ValueError(reason)
            bench.pair_results([row])  # includes actual loaded-runtime proof
            c = row["config"]
            candidates[(c["dataset"], c["seed"], c["mode"])].append(row)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            excluded.append((row, str(error)))
    unique = []
    for batch in candidates.values():
        if len(batch) != 1:
            excluded.extend((row, "ambiguous duplicate dataset/seed/mode") for row in batch)
        else:
            unique.append(batch[0])
    by_dataset = defaultdict(list)
    for row in unique:
        by_dataset[row["config"]["dataset"]].append(row)
    valid = []
    for batch in by_dataset.values():
        if len({aggregate_identity(row) for row in batch}) != 1:
            excluded.extend((row, "conflicting workload/provenance across dataset seeds or modes") for row in batch)
        else:
            valid.extend(batch)
    if len({campaign_identity(row) for row in valid}) > 1:
        raise RuntimeError("Mixed campaign runtime/shared provenance or measurement conditions across datasets; select a coherent results folder")
    pairs, unpaired = bench.pair_results(valid)
    return valid, pairs, unpaired, excluded


def stats(values):
    return {"median": median(values), "min": min(values), "max": max(values),
            "q25": protocol.percentile(values, .25), "q75": protocol.percentile(values, .75),
            "sample_sd": stdev(values) if len(values) > 1 else None}


def complete_median(values):
    return median(values) if values and all(value is not None for value in values) else None


def groups(pairs):
    buckets = defaultdict(list)
    for pair in pairs:
        buckets[aggregate_identity(pair["optimized"])].append(pair)
    output = []
    for identity, batch in buckets.items():
        batch.sort(key=lambda pair: pair["optimized"]["config"]["seed"])
        first = batch[0]["optimized"]
        reference = first["config"]["protocol_config"].get("reported_h100_emeans_seconds")
        summary = {"dataset": first["config"]["dataset"], "n": first["n"], "d": first["d"], "k": first["k"],
                   "epochs": first["config"]["epochs"], "paired_seeds": [p["optimized"]["config"]["seed"] for p in batch],
                   "pair_count": len(batch), "comparison_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                   "speedup": stats([p["speedup"] for p in batch]),
                   "quality_delta_pct": protocol.paired_quality_summary(batch), "h100_seconds": reference,
                   "pairs": batch}
        for mode in bench.MODES:
            records = [pair[mode] for pair in batch]
            summary[mode + "_seconds"] = stats([row["train_seconds"] for row in records])
            summary[mode + "_test_mse"] = stats([row["test_mse"] for row in records])
            for name, getter in (("footprint", protocol.memory_peak),
                                 ("rss", lambda row: protocol.memory_peak(row, "peak_rss_bytes")),
                                 ("mlx_active", protocol.allocator_peak)):
                values = [getter(row) for row in records]
                summary[mode + "_" + name + "_bytes"] = complete_median(values)
                summary[mode + "_" + name + "_complete_seeds"] = sum(value is not None for value in values)
        summary["contextual_apple_h100_ratio"] = summary["optimized_seconds"]["median"] / reference if reference else None
        output.append(summary)
    return sorted(output, key=lambda group: group["dataset"])


def figures(grouped, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suite in ("core1m", "vibe"):
        selected = [g for g in grouped if g["dataset"].startswith("vibe-") == (suite == "vibe")]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, .48 * len(selected) + 1.7)), layout="constrained")
        for i, group in enumerate(selected):
            for mode, offset, color in (("previous", -.12, "#D66A32"), ("optimized", .12, "#137DAB")):
                timing = group[mode + "_seconds"]
                axes[0].errorbar(timing["median"], i + offset,
                                 xerr=[[timing["median"] - timing["q25"]], [timing["q75"] - timing["median"]]],
                                 fmt="o", color=color, capsize=3, label=mode if i == 0 else None)
        for i, group in enumerate(selected):
            speedup = group["speedup"]
            axes[1].errorbar(speedup["median"], i,
                             xerr=[[speedup["median"] - speedup["min"]], [speedup["max"] - speedup["median"]]],
                             fmt="o", color="#137DAB" if speedup["median"] >= 1 else "#D66A32", capsize=3)
        axes[1].axvline(1, color="#666", linewidth=.8)
        axes[1].set_xlabel("Paired speedup: previous / optimized (×)\nmedian; whisker: observed min–max")
        axes[1].locator_params(axis="x", nbins=5)
        axes[0].set_xlabel("Paired-seed median fit seconds [Q25, Q75]")
        axes[0].set_xlim(left=0)
        axes[0].legend(frameon=False)
        for index, axis in enumerate(axes):
            axis.set_yticks(range(len(selected)), [protocol.label(g["dataset"]) for g in selected] if index == 0 else [""] * len(selected))
            axis.set_ylim(len(selected) - .5, -.5)
            axis.grid(axis="x", alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        hosts = sorted({p["optimized"]["hardware"].get("cpu", "unknown") for g in selected for p in g["pairs"]})
        threads = sorted({p["optimized"]["config"]["threads"] for g in selected for p in g["pairs"]})
        counts = sorted({g["pair_count"] for g in selected})
        fig.suptitle(f"{', '.join(hosts)} · {suite} · native BF16, resident warmed fits\n"
                     f"CPU threads: {threads}; paired seeds per dataset: {counts}; identical epoch budgets; no historical timing baseline", fontsize=11)
        path = folder / f"{suite}-gap-optimization.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def make_report(valid, pairs, unpaired, excluded, grouped, output, folder, plot_paths=(), expected_seeds=(10, 11, 12)):
    def link(path, text):
        return common.link(path, text, output)
    fmt = protocol.f
    planned = 17 * len(expected_seeds)
    observed = sum(pair["optimized"]["config"]["seed"] in expected_seeds for pair in pairs)
    coverage = "Complete planned coverage" if observed == planned and len(grouped) == 17 else "Partial snapshot"
    lines = ["# Fresh Metal e-means optimization and H100 gap investigation", "",
             f"**{coverage}: {len(grouped)}/17 datasets and {observed}/{planned} planned seed pairs are measured.** "
             f"Planned seed IDs: {', '.join(map(str,expected_seeds))}; {len(pairs)} valid pairs are available in total. "
             "The previous and optimized implementations both use native BF16 assignment, the same fixed epoch budget, input, "
             "initial centroids, warm-up and timing controller. Only explicitly recorded runtime sources may differ. "
             "Historical campaign timings are investigation context, not the speedup baseline below.", "",
             "The previous implementation comes from " + link(bench.ARCHIVE, "the frozen second-version source archive") + ". "
             "Each result proves which runtime modules were actually loaded. Shared harness, protocol, hardware, software, "
             "resolved recipe and initialization must match within each seed pair. Conflicting or ambiguous evidence is excluded.", "",
             "## Why the H100 gap remains", "",
             "**The evidence points primarily to matrix-throughput differences, with smaller implementation overheads.** "
             "On two wide test shapes (d=1024 and 3072), this M4 Max's fused assignment reaches about 14.1 useful TFLOP/s; "
             "plain MLX BF16-output matrix multiplication reaches 14.7–14.8 TFLOP/s. Assignment is within about 4–5% of "
             "that simpler operation's time. The plain product omits nearest-centroid reduction and has different output "
             "precision, so it is an empirical arithmetic reference, not a valid replacement or a proven hardware ceiling. "
             + link(HERE / "results/gap_investigation/roofline.json", "Measured matrix-throughput evidence") + ".", "",
             "Using the recorded local assignment work with the paper's rounded H100 times gives a **conditional** "
             "246–595 TFLOP/s. This is plausible against H100 SXM's 989.4 dense BF16 TFLOP/s specification with FP32 "
             "accumulation; the roughly doubled structured-sparse figure is inapplicable. The nominal memory-bandwidth "
             "ratio is only about 6.1× (3350 versus 546 GB/s). Neither this ratio nor a peak/measurement ratio precisely "
             "partitions the performance gap. The exact Table 11 H100 SKU and per-seed work remain unverified. "
             + link(HERE / "results/gap_investigation/workload_analysis.md", "Hardware sources, FLOP accounting and limitations") + ".", "",
             "Fresh serialized stage profiles attribute roughly 84–92% of instrumented time to assignment on DEEP, "
             "GloVe and AG-News; the earlier DINO profile attributes about 96%. Shuffling, casting, statistics, updates "
             "and dispatch remain candidates for incremental improvement, but the profiles do not support them as the "
             "dominant wide-dimension cost. These stage percentages come from synchronized instrumentation, not GPU "
             "hardware-counter traces or an exact decomposition of normal asynchronous wall time. "
             + link(HERE / "results/gap_investigation/stages-before.json", "Stage attribution") + ".", "",
             "## Implemented changes", "",
             "The optimized runtime fuses shuffled FP32 row gathering and BF16 conversion using four-value vector loads. "
             "The recorded gather microbenchmark preserves every BF16 output bit and improves its standalone operation "
             "by 1.24–2.25× on the tested shapes. It also stores partial scores and merges them with contiguous row accesses "
             "for dimensions 96 and 128 when k ≥ 1024 and the batch has at least 32 rows. That assignment change measured "
             "about 1.11–1.13× on its tested low-dimension shapes. Higher-dimension assignment kernels are unchanged; "
             "their implemented change is the gather/conversion step. These microbenchmarks do not establish whole-fit gains; "
             "only the fresh paired table below measures those.", "",
             link(HERE / "results/gap_investigation/gather-vector4.json", "Vector-gather evidence") + " · "
             + link(HERE / "results/gap_investigation/production-validation.json", "Assignment correctness and timing evidence") + ".", "",
             "## Fresh matched results", "",
             "| Dataset | Epochs | Previous s | Optimized s | Paired speedup | MSE change median [min, max] % | Footprint GiB P / O | Seed pairs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in grouped:
        q = g["quality_delta_pct"]
        quality = f"{q['median']:+.5g} [{q['min']:+.5g}, {q['max']:+.5g}]" if q else "—"
        lines.append(f"| {protocol.label(g['dataset'])} | {g['epochs']:g} | {g['previous_seconds']['median']:.3f} | "
                     f"{g['optimized_seconds']['median']:.3f} | {g['speedup']['median']:.3f}× | {quality} | "
                     f"{common.gib(g['previous_footprint_bytes'])} / {common.gib(g['optimized_footprint_bytes'])} | {g['pair_count']} |")
    lines += ["", "Speedup is the median of per-seed previous/optimized time ratios; mode times are medians over those same pairs. "
              "Positive MSE change means worse optimized quality. Each seed's quality uses its first timed fit; time is one "
              "synchronized fit or the median of three when its first fit took less than two seconds. Repeated calls are not independent seeds.", ""]
    quality = protocol.paired_quality_summary(pairs)
    if quality:
        lines += [f"Across {quality['count']} defined seed-level comparisons, observed MSE changes range from {quality['min']:+.6g}% "
                  f"to {quality['max']:+.6g}%. These extrema describe individual seed pairs, not dataset medians or confidence intervals.", ""]
    if plot_paths:
        lines += ["In the figures, runtime whiskers show the interquartile range of per-seed fit times; speedup whiskers "
                  "show the observed minimum and maximum of paired ratios. Neither is a confidence interval. "
                  "Clustering-quality changes are retained numerically in the table and CSV.", ""]
    for path in plot_paths:
        lines += ["!" + link(path, path.stem.replace("-", " ")), ""]
    lines += ["## Variation and published H100 context", "",
              "The H100 seconds below are rounded Table 11 values, not fresh runs. Its per-seed stopping budgets, exact VIBE "
              "splits and seed IDs are unavailable, and the epoch row's GPU association is unverified. Apple/H100 is descriptive "
              "context, not an equal-work or equal-quality hardware speedup. The six core cases have no published absolute "
              "H100 e-means time. Local table-level budgets remain fixed across seeds; no crossing search was performed.", "",
              "| Dataset | Previous s [Q25, Q75]; SD | Optimized s [Q25, Q75]; SD | Seed IDs | H100 s | Approx. Apple/H100 |",
              "|---|---:|---:|---|---:|---:|"]
    for g in grouped:
        cells = [f"[{g[mode+'_seconds']['q25']:.3f}, {g[mode+'_seconds']['q75']:.3f}]; {fmt(g[mode+'_seconds']['sample_sd'])}"
                 for mode in bench.MODES]
        lines.append(f"| {protocol.label(g['dataset'])} | {cells[0]} | {cells[1]} | {', '.join(map(str,g['paired_seeds']))} | "
                     f"{fmt(g['h100_seconds'],2)} | {fmt(g['contextual_apple_h100_ratio'],0)} |")
    lines += ["", "Quartiles interpolate between per-seed fit medians; SD is sample standard deviation and is unavailable for "
              "one seed. The raw CSV preserves each fit's repetitions and the observed quality range. Unpaired records do not "
              "enter any aggregate above.", "", "## Memory metrics", "",
              "All memory comparisons below use the same paired seeds. A seed's peak is the maximum across its timed calls; "
              "table cells are medians of those peaks. Incomplete metric groups are withheld rather than using different seed sets. "
              "Process physical footprint and RSS include resident input, runtime and retained cache. MLX active allocation is "
              "a separate allocator metric. These values are not exclusive workspace and must not be added together. Sampling "
              "can miss short transients; raw evidence preserves sample gaps and baseline/increase diagnostics.", "",
              "| Dataset | RSS GiB P / O | MLX active peak GiB P / O | Complete process seeds P / O | Complete allocator seeds P / O |",
              "|---|---:|---:|---:|---:|"]
    for g in grouped:
        lines.append(f"| {protocol.label(g['dataset'])} | {common.gib(g['previous_rss_bytes'])} / {common.gib(g['optimized_rss_bytes'])} | "
                     f"{common.gib(g['previous_mlx_active_bytes'])} / {common.gib(g['optimized_mlx_active_bytes'])} | "
                     f"{g['previous_footprint_complete_seeds']} / {g['optimized_footprint_complete_seeds']} | "
                     f"{g['previous_mlx_active_complete_seeds']} / {g['optimized_mlx_active_complete_seeds']} |")
    tradeoffs = [g for g in grouped if g["previous_mlx_active_bytes"] and g["optimized_mlx_active_bytes"] is not None
                and g["optimized_mlx_active_bytes"] > g["previous_mlx_active_bytes"]]
    if tradeoffs:
        lines += ["", "**Not all memory metrics improve.** Optimized median per-seed MLX active peaks increase on "
                  + "; ".join(f"{protocol.label(g['dataset'])} (+{100*(g['optimized_mlx_active_bytes']/g['previous_mlx_active_bytes']-1):.2f}%)" for g in tradeoffs)
                  + ". These observed allocator changes do not establish a cause and are separate from process-footprint changes.", ""]
    sampler_path = HERE / "results/sampler_control/summary.json"
    queue_path = HERE / "results/gap_investigation/queue-summary.json"
    if sampler_path.exists() or queue_path.exists():
        lines += ["", "## Measurement and queue controls", ""]
    if sampler_path.exists():
        control = json.loads(sampler_path.read_text())
        lines += ["The external memory sampler was toggled in fresh-process runs of the frozen runtime, with balanced "
                  "condition order and three seeds per dataset. All other fit conditions were retained. The ratios below "
                  "are medians of paired sampled/unsampled times, not ratios of the two displayed time medians.", "",
                  "| Dataset | Sampling enabled s | Sampling disabled s | Paired enabled / disabled |",
                  "|---|---:|---:|---:|"]
        for dataset, metrics in control["summary"].items():
            lines += [f"| {protocol.label(dataset)} | {metrics['median_enabled_seconds']:.3f} | "
                      f"{metrics['median_disabled_seconds']:.3f} | {metrics['median_enabled_over_disabled']:.3f}× |"]
        lines += ["", "These small, mixed-direction differences do not establish a precise sampler cost and do not "
                  "support sampling as the explanation for the much larger earlier timing shift. "
                  + link(sampler_path, "Control samples, commands and evidence hashes") + ".", ""]
    if queue_path.exists():
        lines += ["A separate interleaved experiment drained queued GPU work every two, four or eight batches. "
                  "For SPACEV, draining every two batches reduced active peak allocation from 1.897 to 0.760 GiB, "
                  "with a 2.7% higher median time; individual timing directions varied. The four-batch policy was "
                  "consistently slower on SPACEV. AG-News timing effects were inconclusive. Centroid and timing-free "
                  "history hashes matched across all variants. This establishes a queue-depth contribution to memory "
                  "usage, without proving it is the sole cause of the earlier memory regression. No reliable runtime "
                  "win was established, so production scheduling remains unchanged. "
                  + link(HERE / "results/gap_investigation/queue-summary.md", "Queue ablation and tradeoffs") + ".", ""]
    replacement_path = HERE / "results/gap_telemetry_overlap/replacement_manifest.json"
    if replacement_path.exists():
        lines += ["A rootless telemetry trace observed variable active-residency-weighted GPU frequencies while "
                  "thermal state remained Normal. It does not establish thermal throttling as the cause of earlier "
                  "timing differences. The monitor consumed measurable CPU time, so all four dataset/seed pairs "
                  "whose timed windows overlapped collection were rerun with the monitor off. The table uses those "
                  "clean replacements; every original record is preserved. "
                  + link(HERE / "results/gap_investigation/TELEMETRY.md", "Telemetry semantics") + " · "
                  + link(replacement_path, "Original and replacement evidence") + ".", ""]
    lines += ["", "## Investigation evidence and limits", "",
              link(HERE / "results/gap_investigation/workload_analysis.md", "Workload and hardware audit") + " · "
              + link(HERE / "results/gap_investigation/workload_analysis.json", "Inspectable FLOP calculations") + " · "
              + link(HERE / "results/gap_investigation", "Stage profiles, scheduling experiments and arithmetic proxies") + ".", "",
              "The prior campaign's 33–68× H100 context gap was not a proven hardware limit. Fresh unchanged-code diagnostics "
              "were faster under different measurement conditions, so the table above compares newly measured runtimes under "
              "one controller. H100 dense BF16 throughput makes the published times plausible; its sparse peak is inapplicable. "
              "Hardware specifications and isolated matrix products do not assign a causal fraction of the gap to hardware or code.", "",
              "Both local variants use resident FP32 input, BF16-rounded assignment/statistics operands, FP32 state, last-epoch "
              "closing means and online final-pass int64 labels. Upload, full preparation validation, initialization sampling, "
              "warm-up, cache clearing, host export and shared FP32-ranking/FP64-residual scoring are outside fit timing. "
              "New model creation, signature checks, permutations, updates, objective/history and synchronization are timed. "
              "Online history reductions remain FP32 on Metal versus FP64 upstream, and NumPy/CUDA random streams differ. "
              "Core overlap counts and VIBE's newly prepared holdouts are preserved; ten-million and larger cases are excluded.", ""]
    environments = {common.stable({"hardware": r["hardware"], "versions": r["versions"], "threads": r["config"]["threads"]}): r for r in valid}
    for row in environments.values():
        lines += [f"Hardware: {row['hardware'].get('cpu')}, {row['hardware'].get('ram_bytes',0)/2**30:g} GiB; "
                  f"CPU threads: {row['config']['threads']}. Software: " + ", ".join(f"{k} {v}" for k,v in row["versions"].items()) + ".", ""]
    lines += [link(folder / "summary.csv", "Dataset summary CSV") + " · " + link(folder / "pairs.csv", "Paired seed metrics and raw evidence paths")
              + " · " + link(folder / "report_manifest.json", "Source and evidence hashes") + " · "
              + link(HERE / "H100_PROTOCOL.md", "Audited protocol") + ".", ""]
    if (folder / "validation.json").exists():
        lines += [link(folder / "validation.json", "Final coverage, runtime-source and telemetry-window audit") + ".", ""]
    for archive, digest in sorted({(row["config"].get("archive", str(bench.ARCHIVE)), row["config"]["archive_sha256"]) for row in valid}):
        lines += ["Recorded previous-runtime archive: " + link(Path(archive), Path(archive).name) + f"; SHA-256 `{digest}`.", ""]
    if unpaired:
        lines += [f"{len(unpaired)} valid records have no matched fresh counterpart and are excluded from aggregates.", ""]
    if excluded:
        lines += ["Excluded evidence:", ""]
        lines += [f"- {reason}: {link(row['_path'], row['_path'].name)}" for row, reason in excluded]
        lines.append("")
    lines += ["```bash", ".venv/bin/python benchmarks/e_means/gap_report.py",
              "# Add --plots only after timed fits finish.", "```", ""]
    return "\n".join(lines)


def write_csv(path, rows, empty_fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/gap_protocol")
    parser.add_argument("--output", type=Path, default=REPO / "docs/e-means-gap-investigation-2026-11-14.md")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--figures", type=Path, default=HERE / "gap_figures")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[10, 11, 12], help="Planned seed IDs for coverage labeling")
    args = parser.parse_args()
    if len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("Expected seed IDs must be unique")
    raw = evidence_rows(args.results)
    valid, pairs, unpaired, excluded = validate_and_pair(raw)
    grouped = groups(pairs)
    plot_paths = figures(grouped, args.figures) if args.plots else []
    args.results.mkdir(parents=True, exist_ok=True)
    summaries = [{key: value for key, value in g.items() if key != "pairs"} for g in grouped]
    write_csv(args.results / "summary.csv", summaries, ["dataset"])
    seed_rows = []
    for pair in pairs:
        row = {"dataset": pair["optimized"]["config"]["dataset"], "seed": pair["optimized"]["config"]["seed"],
               "speedup": pair["speedup"], "quality_delta_pct": pair["quality_delta_pct"]}
        for mode in bench.MODES:
            record = pair[mode]
            row.update({mode + "_" + key: value for key, value in {
                "seconds": record["train_seconds"], "seconds_samples": record["train_seconds_samples"],
                "test_mse": record["test_mse"], "footprint_bytes": protocol.memory_peak(record),
                "rss_bytes": protocol.memory_peak(record, "peak_rss_bytes"), "mlx_active_bytes": protocol.allocator_peak(record),
                "evidence_path": str(record["_path"]), "evidence_sha256": common.sha256(record["_path"])}.items()})
        seed_rows.append(row)
    write_csv(args.results / "pairs.csv", seed_rows, ["dataset", "seed"])
    manifest = {"kind": KIND, "valid_records": len(valid), "paired_seeds": len(pairs), "paired_datasets": len(grouped),
                "expected_seed_ids": args.expected_seeds, "expected_seed_pairs": 17 * len(args.expected_seeds),
                "source_sha256": {str(path.relative_to(REPO)): common.sha256(path) for path in
                                  (Path(__file__), HERE / "gap_bench.py", HERE / "h100_report.py", HERE / "comparison_report.py")},
                "evidence": [{"path": str(r["_path"]), "sha256": common.sha256(r["_path"])} for r in raw],
                "unpaired": [str(r["_path"]) for r in unpaired],
                "excluded": [{"path": str(r["_path"]), "reason": reason} for r, reason in excluded]}
    (args.results / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_report(valid, pairs, unpaired, excluded, grouped, args.output, args.results, plot_paths, args.expected_seeds))
    print(json.dumps({"report": str(args.output), "paired_datasets": len(grouped), "paired_seeds": len(pairs), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
