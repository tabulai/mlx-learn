"""Build the resident BF16 report from recorded fits only; no numerical work."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import comparison_report as common  # noqa: E402

KIND = "h100_protocol_metal"


def label(dataset):
    return {"imagenet-clip-512-normalized": "ImageNet-CLIP (1M)",
            "vibe-imagenet": "ImageNet-CLIP (VIBE)"}.get(dataset, common.label(dataset))


def evidence_rows(folder):
    rows = []
    for path in sorted(Path(folder).rglob("*.json")):
        r = json.loads(path.read_text())
        if isinstance(r, dict) and r.get("kind") == KIND:
            r["_path"] = path.resolve()
            rows.append(r)
    return rows


def problem(row):
    c = row.get("config", {})
    if row.get("status") != "completed":
        return "fit not completed"
    required = {"precision": "bfloat16", "recipe": "faiss", "finalization": "last_epoch", "warmup": True,
                "capture_assignments": True, "rows": None, "k_override": None, "epochs_override": None}
    if any(c.get(key, "missing") != value for key, value in required.items()):
        return "not the full audited BF16 protocol"
    if c.get("mode") not in ("baseline", "optimized") or c.get("assignment") != {"baseline": "gemm", "optimized": "auto"}[c["mode"]]:
        return "invalid assignment mode"
    p = c.get("protocol_config", {})
    if (c.get("dataset") != p.get("dataset") or c.get("epochs") != p.get("epochs")
        or any(row.get(key) != p.get(key) for key in ("n", "d", "k"))):
        return "shape or budget differs from manifest"
    if c.get("seed") not in p.get("seed_ids", []):
        return "seed is outside the audited manifest seed set"
    identity_fields = ("train_sample_sha256", "test_sha256", "train_rows_used", "test_rows_used", "dimension")
    if any(key not in p.get("dataset_identity", {}) or row.get("dataset", {}).get(key) != p["dataset_identity"][key]
           for key in identity_fields):
        return "dataset fingerprints or evaluation shape differ from manifest"
    for actual, expected in (("source_sha256", "expected_source_sha256"), ("versions", "expected_versions"), ("hardware", "expected_host")):
        if not row.get(actual) or row[actual] != c.get(expected):
            return "incomplete or mismatched provenance"
    needed = ["benchmarks/e_means/h100.py", "benchmarks/e_means/memory.py", "benchmarks/e_means/h100_protocol.json",
              "src/mlxlearn/cluster/_e_means.py", "src/mlxlearn/_kernels/e_means.py", "src/mlxlearn/_kernels/e_means_bf16.py"]
    if any(not row["source_sha256"].get(key) for key in needed):
        return "missing benchmark/kernel/protocol source hash"
    samples = row.get("train_seconds_samples", [])
    if (not samples or any(not common.positive(s) for s in samples)
        or len(samples) != (3 if samples[0] < 2 else 1)
        or not math.isclose(median(samples), row.get("train_seconds", math.nan), rel_tol=1e-12)):
        return "invalid adaptive call count or median"
    if not common.nonnegative(row.get("test_mse")) or row.get("quality_fit_index") != 0:
        return "invalid first-fit quality result"
    if not row.get("initial_centroids_sha256") or any(not row.get("dataset", {}).get(key) for key in ("train_sample_sha256", "test_sha256")):
        return "missing initialization or dataset fingerprints"
    if not row.get("timing_scope") or not row.get("scoring") or not row.get("memory_scope"):
        return "missing measurement scope"
    recipe = row.get("recipe_resolved", {})
    if any(key not in recipe for key in common.RECIPE_FIELDS):
        return "incomplete resolved recipe"
    if (recipe.get("precision") != "bfloat16" or recipe.get("statistics_representation") != "bfloat16"
        or recipe.get("assignment_centering") != "none" or recipe.get("finalization") != "last_epoch"
        or recipe.get("nepochs") != c["epochs"] or recipe.get("seed") != c.get("seed")):
        return "resolved representation/finalizer/budget differs"
    labels = row.get("captured_assignments", {})
    if labels.get("shape") != [row["n"]] or "int64" not in labels.get("dtype", ""):
        return "missing full int64 last-pass labels"
    if not row.get("iteration_history"):
        return "missing online history"
    return None


def identity(row, *, include_seed=False):
    fields = ("n", "d", "k", "dataset", "source_sha256", "versions", "hardware", "metal_device",
              "faiss_reference_revision", "timing_scope", "scoring", "memory_scope")
    value = {key: row.get(key) for key in fields}
    c = row["config"]
    value["config"] = {key: c.get(key) for key in ("dataset", "epochs", "protocol_config", "precision", "recipe",
                     "finalization", "warmup", "capture_assignments", "threads", "train_score_rows", "memory", "memory_interval_seconds")}
    value["recipe"] = {key: row["recipe_resolved"][key] for key in common.RECIPE_FIELDS if key != "seed"}
    if include_seed:
        value.update(seed=c["seed"], initial=row["initial_centroids_sha256"])
    return common.stable(value)


def validate_and_pair(rows):
    valid, excluded = [], []
    candidates = defaultdict(list)
    for row in rows:
        reason = problem(row)
        if reason:
            excluded.append((row, reason))
        else:
            candidates[(row["config"]["dataset"], row["config"]["seed"], row["config"]["mode"])].append(row)
    for batch in candidates.values():
        if len(batch) != 1:
            excluded.extend((r, "ambiguous duplicate dataset/seed/mode") for r in batch)
        else:
            valid.append(batch[0])
    baseline = {identity(r, include_seed=True): r for r in valid if r["config"]["mode"] == "baseline"}
    pairs = []
    for row in valid:
        if row["config"]["mode"] == "optimized" and identity(row, include_seed=True) in baseline:
            b = baseline[identity(row, include_seed=True)]
            pairs.append({"optimized": row, "baseline": b, "speedup": b["train_seconds"] / row["train_seconds"],
                          "quality_delta_pct": 100 * (row["test_mse"] / b["test_mse"] - 1) if b["test_mse"] else None})
    return valid, pairs, excluded


def memory_peak(row, field="peak_physical_footprint_bytes"):
    samples = row.get("memory_samples", [])
    if not row["config"].get("memory") or len(samples) != len(row["train_seconds_samples"]):
        return None
    if any(not isinstance(m, dict) or common.memory_problem({**row, "memory": m}) is not None for m in samples):
        return None
    return max(m[field] for m in samples)


def allocator_peak(row):
    samples = row.get("mlx_memory_samples", [])
    if len(samples) != len(row["train_seconds_samples"]):
        return None
    values = [m.get("peak_active_bytes") if isinstance(m, dict) else None for m in samples]
    return max(values) if values and all(common.nonnegative(v) for v in values) else None


def memory_groups(rows):
    """All available mode/seed records, kept separate across provenance variants."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[(identity(row), row["config"]["mode"])].append(row)
    result = []
    for (_, mode), batch in buckets.items():
        footprint = [memory_peak(r) for r in batch]
        rss = [memory_peak(r, "peak_rss_bytes") for r in batch]
        active = [allocator_peak(r) for r in batch]
        def present(values):
            return [v for v in values if v is not None]
        result.append({"dataset": batch[0]["config"]["dataset"], "mode": mode,
                       "seed_count": len(batch), "seeds": sorted(r["config"]["seed"] for r in batch),
                       "complete_memory_seeds": len(present(footprint)), "complete_allocator_seeds": len(present(active)),
                       "median_peak_footprint_bytes": median(present(footprint)) if present(footprint) else None,
                       "median_peak_rss_bytes": median(present(rss)) if present(rss) else None,
                       "median_peak_mlx_active_bytes": median(present(active)) if present(active) else None,
                       "raw_evidence_paths": [str(r["_path"]) for r in batch]})
    return sorted(result, key=lambda g: (g["dataset"], g["mode"]))


def paired_allocator_increases(grouped):
    """Report allocator tradeoffs using the same complete seed pairs in each mode."""
    increases = []
    for group in grouped:
        complete = [(pair, allocator_peak(pair["baseline"]), allocator_peak(pair["optimized"]))
                    for pair in group["pairs"]]
        complete = [(pair, baseline, optimized) for pair, baseline, optimized in complete
                    if baseline is not None and optimized is not None]
        if not complete:
            continue
        baseline = median(value for _, value, _ in complete)
        optimized = median(value for _, _, value in complete)
        if optimized <= baseline:
            continue
        footprint = [[memory_peak(pair[mode]) for pair, _, _ in complete]
                     for mode in ("baseline", "optimized")]
        baseline_footprint, optimized_footprint = [median(values) if all(v is not None for v in values) else None
                                                for values in footprint]
        increases.append({"dataset": group["dataset"], "paired_seeds": len(complete),
                          "baseline_active_bytes": baseline, "optimized_active_bytes": optimized,
                          "active_increase_pct": 100 * (optimized / baseline - 1) if baseline else None,
                          "baseline_footprint_bytes": baseline_footprint,
                          "optimized_footprint_bytes": optimized_footprint,
                          "footprint_change_pct": 100 * (optimized_footprint / baseline_footprint - 1)
                          if baseline_footprint and optimized_footprint is not None else None})
    return increases


def groups(rows, pairs):
    buckets = defaultdict(list)
    for row in rows:
        buckets[identity(row)].append(row)
    result = []
    for key, batch in buckets.items():
        opts = sorted((r for r in batch if r["config"]["mode"] == "optimized"), key=lambda r: r["config"]["seed"])
        if not opts:
            continue
        paired = [p for p in pairs if identity(p["optimized"]) == key]
        first = opts[0]
        ref = first["config"]["protocol_config"]
        secs = [r["train_seconds"] for r in opts]
        losses = [r["test_mse"] for r in opts]
        peaks = [memory_peak(r) for r in opts]
        result.append({"dataset": first["config"]["dataset"], "n": first["n"], "d": first["d"], "k": first["k"],
                       "epochs": first["config"]["epochs"], "optimized_rows": opts, "pairs": paired,
                       "optimized_seeds": [r["config"]["seed"] for r in opts], "expected_seeds": ref["seed_ids"],
                       "optimized_seconds": median(secs), "optimized_min_seconds": min(secs), "optimized_max_seconds": max(secs),
                       "optimized_q25_seconds": percentile(secs, .25), "optimized_q75_seconds": percentile(secs, .75),
                       "optimized_std_seconds": stdev(secs) if len(secs) > 1 else None,
                       "optimized_mse": median(losses), "optimized_min_mse": min(losses), "optimized_max_mse": max(losses),
                       "optimized_footprint": median(peaks) if all(v is not None for v in peaks) else None,
                       "h100_seconds": ref.get("reported_h100_emeans_seconds"),
                       "h100_context_ratio": median(secs) / ref["reported_h100_emeans_seconds"] if ref.get("reported_h100_emeans_seconds") else None,
                       "paired_baseline_seconds": median(p["baseline"]["train_seconds"] for p in paired) if paired else None,
                       "paired_optimized_seconds": median(p["optimized"]["train_seconds"] for p in paired) if paired else None,
                       "paired_speedup": median(p["speedup"] for p in paired) if paired else None,
                       "paired_quality_delta_pct": median(p["quality_delta_pct"] for p in paired) if paired and all(p["quality_delta_pct"] is not None for p in paired) else None})
    return sorted(result, key=lambda g: g["dataset"])


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def paired_quality_summary(pairs):
    values = [p["quality_delta_pct"] for p in pairs
              if p["quality_delta_pct"] is not None and math.isfinite(p["quality_delta_pct"])]
    return {"median": median(values), "min": min(values), "max": max(values), "count": len(values)} if values else None


def draw_quality(axis, selected):
    """Keep a visible marker for exact-zero medians and show observed seed spread."""
    for position, group in enumerate(selected):
        quality = paired_quality_summary(group["pairs"])
        if quality is None:
            continue
        center = quality["median"]
        axis.errorbar(center, position,
                      xerr=[[center - quality["min"]], [quality["max"] - center]],
                      fmt="o", color="#D66A32" if center > 0 else "#137DAB", capsize=3, zorder=3)
    axis.axvline(0, color="#666", linewidth=.8, zorder=1)
    axis.set_xlabel("Optimized MSE change vs baseline (%)\ndot: median; whisker: observed min–max\npositive = worse")


def figures(grouped, folder):
    """Create static scientific figures only after the timed campaign ends."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suite in ("core1m", "vibe"):
        selected = [g for g in grouped if g["pairs"] and (g["dataset"].startswith("vibe-") == (suite == "vibe"))]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4.5, .48 * len(selected) + 1.8)), layout="constrained")
        positions = list(range(len(selected)))
        for mode, offset, color, title in (("baseline", -.12, "#D66A32", "BF16-rounded FP32 GEMM"),
                                           ("optimized", .12, "#137DAB", "Optimized BF16")):
            for i, g in enumerate(selected):
                times = [p[mode]["train_seconds"] for p in g["pairs"]]
                center = median(times)
                axes[0].errorbar(center, i + offset,
                                 xerr=[[center - percentile(times, .25)], [percentile(times, .75) - center]],
                                 fmt="o", color=color, capsize=3, label=title if i == 0 else None)
                values = [memory_peak(p[mode]) for p in g["pairs"]]
                if all(v is not None for v in values):
                    axes[1].scatter(median(values) / 2**30, i + offset, color=color, s=35)
        draw_quality(axes[2], selected)
        axes[0].set_xlabel("Paired-seed median fit seconds [Q25, Q75]")
        axes[0].set_xlim(left=0)
        axes[0].legend(loc="best", frameon=False, fontsize=9)
        axes[1].set_xlabel("Median per-seed peak process footprint (GiB)")
        axes[1].set_xlim(left=0)
        axes[2].locator_params(axis="x", nbins=4 if suite == "core1m" else 5)
        for index, axis in enumerate(axes):
            axis.set_yticks(positions, [label(g["dataset"]) for g in selected] if index == 0 else [""] * len(selected))
            axis.set_ylim(len(selected) - .5, -.5)
            axis.grid(axis="x", alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        hosts = sorted({p["optimized"]["hardware"].get("cpu", "unknown") for g in selected for p in g["pairs"]})
        counts = sorted({len(g["pairs"]) for g in selected})
        fig.suptitle(f"{', '.join(hosts)} · {suite} · Resident BF16 e-means, full warm-up\n"
                     f"Audited per-dataset epoch budgets; paired seeds: {', '.join(map(str, counts))} · H100 times excluded from this matched comparison", fontsize=11)
        path = folder / f"{suite}-optimization.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def make_report(rows, pairs, grouped, excluded, output, csv_path, manifest_path, plot_paths=()):
    coverage = len({g["dataset"] for g in grouped})
    lines = ["# Resident BF16 Metal e-means and published H100 context", "",
             f"**{coverage}/17 configurations; {len(rows)} mode/seed records and {len(pairs)} matched seed pairs.** "
             "This rerun uses resident FP32 input, BF16-rounded "
             "assignment and statistics values, FP32 centroid state, last-epoch means, captured moving labels, and an identical full warm-up. "
             "Epoch budgets follow the audited manifest: four published aggregate core H100 crossing budgets, eleven backend-unverified Table 11 budgets, "
             "and two supplemental three-epoch core configurations.", "",
             "**Each table-level epoch budget is held fixed across all local seeds.** The reference core experiment instead "
             "searched each seed independently against its own Flash-KMeans@25 quality target and reported the upper median "
             "crossing budget. Its individual crossing budgets are not released. This is a fixed-budget rerun using the published "
             "summaries, not a reproduction of per-seed H100 stopping epochs. Table 11 likewise supplies no per-seed budget sequence "
             "and does not specify the epoch row's aggregation rule.", "",
             "## Implementation and profiling", "",
             "The original FP32 profile attributed about 94–96% of fit time to assignment on DEEP and DINO. "
             "The optimized implementation uses native BF16 matrix fragments, contiguous vector tile loads, "
             "and a register-level nearest-centroid reduction. It writes one FP32 score and uint32 index per 64-centroid tile, "
             "rather than a full batch-by-centroid score matrix: 32× less score/index storage at equal row counts, "
             "before small merge outputs. Row tile sizes and other buffers differ; measured process-memory ratios are reported below. "
             "Fused centroid norms and residual kernels consume BF16 values directly while accumulating in FP32. "
             "The resident fit API separates preparation and host export from training without removing shuffling, casting, "
             "history, or final-pass labels from the fit timer.", "",
             "Raw stage profiles and kernel tuning sweeps are preserved in "
             + common.link(HERE / "results/optimization", "optimization evidence", output) + ". "
             "Stage instrumentation serializes GPU work, so its stage sums are diagnostic rather than an asynchronous hardware trace. "
             "The optimization speedups below compare matched BF16 runs. Earlier cold FP32 results used a different protocol "
             "and are not a causal measure of this kernel optimization. Their original implementation remains archived in "
             + common.link(HERE / "reference/metal-v1-source.zip", "the pre-optimization source snapshot", output) + ".", "",
             "## Matched Metal optimization comparison", "",
             "Baseline and optimized measurements below use the same data, initial centroids, seed, resolved recipe, source, hardware, "
             "and timing scope. Baseline uses BF16-rounded operands widened to FP32 for MLX GEMM; optimized uses the selected native-BF16 "
             "assignment path. Only seeds measured in both modes enter this table. Extra optimized seeds are reported separately.", "",
             "| Dataset | Epochs | Baseline s | Optimized s | Speedup | Optimized MSE change | Peak footprint GiB B / O | Paired seeds |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for g in grouped:
        ps = g["pairs"]
        if not ps:
            continue
        base_mem, opt_mem = [memory_peak(p["baseline"]) for p in ps], [memory_peak(p["optimized"]) for p in ps]
        mems = [median(v) if all(x is not None for x in v) else None for v in (base_mem, opt_mem)]
        a = f(g["paired_baseline_seconds"])
        b = f(g["paired_optimized_seconds"])
        lines.append(f"| {label(g['dataset'])} | {g['epochs']:g} | {a} | {b} | {f(g['paired_speedup'],2)}× | "
                     f"{f(g['paired_quality_delta_pct'])}% | {common.gib(mems[0])} / {common.gib(mems[1])} | {len(ps)} |")
    lines += ["", "Speedup is the median of paired per-seed time ratios; quality change is the median paired percentage change "
              "(positive means optimized has worse MSE). Mode times are medians over those same paired seeds. "
              "Peak footprint is the median per-seed maximum sampled process footprint over that seed's timed calls, including the resident-input baseline.", ""]
    quality = paired_quality_summary(pairs)
    if quality:
        lines += [f"Across {quality['count']} paired dataset/seed comparisons with defined percentage changes, "
                  f"the observed per-seed MSE change ranges from {quality['min']:+.6g}% to {quality['max']:+.6g}%. "
                  "This range describes individual seed pairs; the table reports per-dataset medians. In the quality figures, "
                  "dots show those medians and whiskers show the observed per-dataset minimum and maximum, not confidence intervals.", ""]
    increases = paired_allocator_increases(grouped)
    if increases:
        lines += ["**Memory metrics do not all improve.** The following comparisons use matching seeds with complete MLX "
                  "allocator records in both modes. MLX active peak is the allocator-reported active allocation peak, separate "
                  "from sampled process physical footprint and RSS. Each mode's value is the median of per-seed maxima over "
                  "timed fits; the percentage compares those two medians.", ""]
        for item in increases:
            change = (f"+{item['active_increase_pct']:.1f}%" if item["active_increase_pct"] is not None
                      else "percentage unavailable from a zero baseline")
            detail = (f"- {label(item['dataset'])}: optimized MLX active peak {common.gib(item['optimized_active_bytes'])} GiB "
                      f"versus baseline {common.gib(item['baseline_active_bytes'])} GiB ({change}; {item['paired_seeds']} paired seeds).")
            if item["footprint_change_pct"] is not None:
                detail += (f" On the same seeds, median sampled process footprint is {common.gib(item['optimized_footprint_bytes'])} GiB "
                           f"versus baseline {common.gib(item['baseline_footprint_bytes'])} GiB ({item['footprint_change_pct']:+.1f}%).")
            lines.append(detail)
        lines.append("")
    for path in plot_paths:
        lines += ["!" + common.link(path, path.stem.replace("-", " "), output), ""]
    lines += [
              "## All optimized seeds and H100 context", "",
              "The H100 column transcribes Table 11's rounded times; **Apple/H100 is descriptive context, not a matched hardware speedup**. "
              "VIBE epoch-to-GPU association is unverified, exact paper splits and seed IDs are unavailable, and no H100 run was performed here. "
              "No absolute H100 e-means time is supplied for the six core1M configurations; Flash-KMeans times are not substituted.", "",
              "| Dataset | Rows × dimensions; k | Epochs | Optimized s median [Q25, Q75] | Time SD s | Test MSE median [min, max] | Seeds observed / expected | H100 s | Approx. Apple / H100 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in grouped:
        lines.append(f"| {label(g['dataset'])} | {g['n']:,} × {g['d']}; {g['k']:,} | {g['epochs']:g} | "
                     f"{g['optimized_seconds']:.3f} [{g['optimized_q25_seconds']:.3f}, {g['optimized_q75_seconds']:.3f}] | {f(g['optimized_std_seconds'])} | "
                     f"{g['optimized_mse']:.8g} [{g['optimized_min_mse']:.8g}, {g['optimized_max_mse']:.8g}] | "
                     f"{len(g['optimized_seeds'])} / {len(g['expected_seeds'])} | {f(g['h100_seconds'],2)} | {f(g['h100_context_ratio'],0)}{'×' if g['h100_context_ratio'] is not None else ''} |")
    lines += ["", "Time quartiles use linear interpolation over per-seed median fit times; SD is sample standard deviation "
              "across those seed medians (unavailable for one seed). Quality ranges describe measured seed outcomes, not confidence intervals. "
              "The CSV also preserves minimum and maximum time. Core seed IDs are 10–19; VIBE IDs 10–29 "
              "are an explicit local choice matching the paper's count of twenty, not verified paper IDs. Each seed's quality comes "
              "from its first timed fit; its runtime is one synchronized fit, or the median of three when its first fit was under two seconds.", "",
              "## Timing, precision, and memory", "",
              "The fit timer includes a new model, signature/state checks, epoch shuffling, per-batch BF16 casts, assignment, "
              "FP32 statistics/updates, revival, online objective/history, moving last-pass int64 label capture, closing means and synchronization. "
              "Input upload, full input finite/range validation, initial-center sampling, full warm-up, cache clearing before each call, "
              "host centroid copy and quality scoring are outside timing. Moving full validation to preparation differs from upstream's "
              "lighter checks. Epoch history accumulation is FP32 on Metal versus FP64 upstream; per-epoch host reads remain inside the fit.", "",
              "Assignment norms use BF16-rounded centroid values widened to FP32; statistics use BF16-rounded rows widened to FP32. "
              "Explicit initial state and output centroids remain FP32, with no coordinate centering. CUDA and NumPy random streams, "
              "reduction orders and hardware differ. The shared evaluator uses Faiss FP32 nearest-centroid assignment and direct FP64 residuals.", "",
              "Process physical footprint and RSS are externally sampled only across each timed fit. They include resident input/runtime "
              "baseline and retained cache, are not exclusive workspace, and must not be added to each other or to MLX allocator metrics. "
              "Sampling can miss transients shorter than its observed gaps. Missing/incomplete memory is suppressed. Raw files retain every "
              "fit's sample count, gap, baseline, peak, increase and MLX allocation diagnostics.", "",
              "The memory appendix uses **all available valid seeds for each mode**, including unpaired mode/seed records. "
              "Each cell is the median of per-seed peaks, where each seed's peak is the maximum over its timed fits. "
              "Counts distinguish available seeds from complete process-memory and allocator records. Incomplete records do not "
              "contribute to the corresponding memory median. These aggregates may use more seeds than the paired comparison above.", "",
              "| Dataset | Mode | Median peak footprint GiB | Median peak RSS GiB | Median MLX active peak GiB | Seeds available; process / allocator complete |",
              "|---|---|---:|---:|---:|---:|"]
    for group in memory_groups(rows):
        lines.append(f"| {label(group['dataset'])} | {group['mode']} | {common.gib(group['median_peak_footprint_bytes'])} | "
                     f"{common.gib(group['median_peak_rss_bytes'])} | {common.gib(group['median_peak_mlx_active_bytes'])} | "
                     f"{group['seed_count']}; {group['complete_memory_seeds']} / {group['complete_allocator_seeds']} |")
    lines += ["", common.link(csv_path.with_name("memory_summary.csv"), "Memory aggregate CSV with raw evidence paths", output) + ".", ""]
    lines += ["", "## Evidence and limits", ""]
    environments = {common.stable({"hardware": r["hardware"], "versions": r["versions"]}): r for r in rows}
    for row in environments.values():
        lines += [f"Hardware: {row['hardware'].get('cpu')}, {row['hardware'].get('ram_bytes',0)/2**30:g} GiB. "
                  + "Versions: " + ", ".join(f"{k} {v}" for k,v in row["versions"].items()) + ".", ""]
    lines += [common.link(HERE / "H100_PROTOCOL.md", "Audited protocol and unresolved differences", output) + " · "
              + common.link(HERE / "h100_protocol.json", "Machine-readable workload manifest", output) + " · "
              + common.link(HERE / "h100_protocol_addendum.json", "Frozen-manifest interpretation notes", output) + " · "
              + common.link(csv_path, "Summary CSV", output) + " · " + common.link(manifest_path, "Raw evidence hashes", output) + ".", "",
              "Core published split overlaps are retained: DEEP1M has four overlapping evaluation vectors and SPACEV1M six; "
              "SIFT1M has no released 1M fingerprint. VIBE uses newly prepared duplicate-disjoint holdouts rather than unpublished paper indices. "
              "The two supplemental core T3 results have no matched published H100 budget/time. Ten-million and larger runs are excluded.", ""]
    if excluded:
        lines += ["Excluded evidence:", ""]
        lines += [f"- {reason}: {common.link(r['_path'],r['_path'].name,output)}" for r, reason in excluded]
    unpaired = [r for r in rows if not any(r is p["baseline"] or r is p["optimized"] for p in pairs)]
    if unpaired:
        lines += ["", f"{len(unpaired)} valid mode/seed records have no provenance-matched counterpart; they do not enter optimization ratios.", ""]
    lines += ["```bash", ".venv/bin/python benchmarks/e_means/h100.py --seeds 10",
              ".venv/bin/python benchmarks/e_means/h100.py  # all manifest seeds, both modes; resumes pilot",
              ".venv/bin/python benchmarks/e_means/h100_report.py",
              "uv run --isolated --no-project --with matplotlib --python .venv/bin/python benchmarks/e_means/h100_report.py --plots", "```", ""]
    validation = csv_path.parent / "validation.json"
    if validation.exists():
        lines += [common.link(validation, "Saved coverage/source audit with raw-file hashes", output) + ".", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/h100_protocol")
    parser.add_argument("--output", type=Path, default=REPO / "docs/e-means-h100-protocol-2026-11-14.md")
    parser.add_argument("--plots", action="store_true", help="Generate static plots; run only after timed fits finish")
    parser.add_argument("--figures", type=Path, default=HERE / "h100_figures")
    args = parser.parse_args()
    raw = evidence_rows(args.results)
    valid, pairs, excluded = validate_and_pair(raw)
    grouped = groups(valid, pairs)
    plot_paths = figures(grouped, args.figures) if args.plots else []
    args.results.mkdir(parents=True, exist_ok=True)
    summary_path, manifest_path = args.results / "summary.csv", args.results / "report_manifest.json"
    summaries = [{k: v for k, v in g.items() if k not in ("optimized_rows", "pairs")} for g in grouped]
    for summary, group in zip(summaries, grouped, strict=True):
        summary["raw_evidence_paths"] = json.dumps([str(r["_path"]) for r in valid if identity(r) == identity(group["optimized_rows"][0])])
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(summaries[0]) if summaries else ["dataset"])
        writer.writeheader()
        writer.writerows(summaries)
    memory_summaries = memory_groups(valid)
    with (args.results / "memory_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(memory_summaries[0]) if memory_summaries else ["dataset", "mode"])
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in g.items()} for g in memory_summaries)
    manifest = {"kind": "h100_report_manifest", "valid_records": len(valid), "paired_seeds": len(pairs),
                "reporter_sha256": common.sha256(__file__), "helper_sha256": common.sha256(HERE / "comparison_report.py"),
                "evidence": [{"path": str(r["_path"]), "sha256": common.sha256(r["_path"])} for r in raw],
                "exclusions": [{"path": str(r["_path"]), "reason": reason} for r, reason in excluded]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_report(valid, pairs, grouped, excluded, args.output, summary_path, manifest_path, plot_paths))
    print(json.dumps({"report": str(args.output), "records": len(valid), "paired_seeds": len(pairs), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
