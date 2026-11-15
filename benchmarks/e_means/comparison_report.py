"""Report fresh-process Metal/Faiss comparisons without importing either runtime.

Only new ``faiss_direct_comparison`` records are eligible. Historical Lloyd and
CPU-reference measurements remain separate. Use --plots after timed fits finish.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORE_1M = (
    "deep1b-1M", "spacev-1M", "fb-ssnpp-1M", "dino10b-1M-div4096",
    "imagenet-clip-512-normalized", "sift1b-1M-div512",
)
CORE_10M = (
    "deep1b-10M", "spacev-10M", "fb-ssnpp-10M", "dino10b-10M-div4096", "sift1b-10M-div512",
)
VIBE = (
    "vibe-agnews", "vibe-arxiv", "vibe-ccnews", "vibe-celeba", "vibe-codesearchnet",
    "vibe-glove", "vibe-gooaq", "vibe-imagenet", "vibe-landmark-dino",
    "vibe-landmark-nomic", "vibe-simplewiki",
)
PRIMARY = CORE_1M + VIBE
LABELS = {
    "deep1b": "DEEP", "spacev": "SPACEV", "fb-ssnpp": "SimSearchNet++",
    "dino10b": "DINO10B prefix", "sift1b": "SIFT",
    "imagenet-clip-512-normalized": "ImageNet-CLIP",
    "vibe-agnews": "AG-News", "vibe-arxiv": "arXiv", "vibe-ccnews": "CC-News",
    "vibe-celeba": "CelebA", "vibe-codesearchnet": "CodeSearchNet",
    "vibe-glove": "GloVe-200", "vibe-gooaq": "GooAQ", "vibe-imagenet": "ImageNet-CLIP",
    "vibe-landmark-dino": "Landmark-DINO", "vibe-landmark-nomic": "Landmark-Nomic",
    "vibe-simplewiki": "SimpleWiki",
}
RECIPE_FIELDS = (
    "batch_law", "batch_size", "alpha0", "nepochs", "epoch_rows", "epoch_steps",
    "total_steps", "alpha_min_factor", "init_usage", "split_epsilon",
    "unused_threshold", "check_unused_every", "tail_policy", "seed",
)
CONFIG_FIELDS = (
    "dataset", "epochs", "seed", "trial", "threads", "veclib_threads", "rows", "k",
    "precision", "recipe", "finalization", "assignment", "warmup", "train_score_rows",
    "memory_interval_seconds",
)
PAIR_FIELDS = (
    "n", "d", "k", "dataset", "initial_centroids_sha256", "source_sha256", "versions",
    "hardware", "faiss_reference_revision", "faiss_compile_options", "effective_omp_threads",
    "thread_environment", "timing_scope", "scoring", "memory_scope",
)
REQUIRED_SOURCES = (
    "benchmarks/e_means/compare.py", "benchmarks/e_means/memory.py",
    "benchmarks/e_means/run.py", "benchmarks/e_means/vibe_manifest.json",
    "src/mlxlearn/cluster/_e_means.py", "src/mlxlearn/cluster/_recipe.py",
    "src/mlxlearn/_kernels/e_means.py",
)


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(name):
    if name in LABELS:
        return LABELS[name]
    return next((title for prefix, title in LABELS.items() if name.startswith(prefix + "-")), name)


def link(path, title, report):
    relative = Path(os.path.relpath(path, Path(report).parent)).as_posix()
    return f"[{title}]({relative})"


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def evidence_rows(folder):
    """Ignore old result kinds and sidecar JSON; do not silently discard bad new fits."""
    rows = []
    for path in sorted(Path(folder).rglob("*.json")):
        item = json.loads(path.read_text())
        if not isinstance(item, dict) or item.get("kind") != "faiss_direct_comparison":
            continue
        item["_path"] = path.resolve()
        rows.append(item)
    return rows


def row_problem(row):
    if row.get("status") != "completed":
        return f"status={row.get('status', 'missing')}"
    c = row.get("config", {})
    if c.get("method") not in ("metal", "faiss-emeans"):
        return "unknown method"
    for field in CONFIG_FIELDS:
        if field not in c:
            return f"missing config.{field}"
    for field, required in {"epochs": 3, "recipe": "faiss", "finalization": "last_epoch",
                            "precision": "float32", "warmup": False, "rows": None, "k": None}.items():
        if c[field] != required:
            return f"outside primary protocol: config.{field}={c[field]!r}"
    if not positive(row.get("train_seconds")) or not nonnegative(row.get("test_mse")):
        return "invalid fit time or test MSE"
    if not all(positive(row.get(key)) for key in ("n", "d", "k")):
        return "invalid n/d/k"
    for field in PAIR_FIELDS:
        if field not in row or row[field] is None:
            return f"missing {field}"
    if not row["initial_centroids_sha256"] or not row["versions"] or not row["hardware"]:
        return "empty initialization or environment provenance"
    sources = row["source_sha256"]
    if not isinstance(sources, dict) or any(not sources.get(k) for k in REQUIRED_SOURCES):
        return "incomplete comparison source hashes"
    data = row["dataset"]
    if not isinstance(data, dict) or any(not data.get(key) for key in ("train_sample_sha256", "test_sha256")):
        return "missing train/test fingerprints"
    for actual, expected in ((sources, c.get("expected_source_sha256")),
                             (row["versions"], c.get("expected_versions")),
                             (row["hardware"], c.get("expected_host"))):
        if actual != expected:
            return "actual provenance differs from expected provenance"
    if row["effective_omp_threads"] != c["threads"]:
        return "effective OpenMP thread count differs from requested count"
    recipe = row.get("recipe_resolved")
    if not isinstance(recipe, dict) or any(key not in recipe for key in RECIPE_FIELDS):
        return "incomplete resolved recipe"
    if recipe["nepochs"] != c["epochs"] or recipe["seed"] != c["seed"]:
        return "resolved epochs/seed differ from configuration"
    if c["dataset"] in CORE_1M and (row["n"] != 1_000_000 or row["k"] != 16384):
        return "core1M requires full million rows and k=16384"
    if c["dataset"] in CORE_10M and (row["n"] != 10_000_000 or row["k"] != 65536):
        return "core10M requires full ten million rows and k=65536"
    return None


def semantic_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(semantic_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def pair_problem(metal, faiss):
    for field in CONFIG_FIELDS:
        if metal["config"][field] != faiss["config"][field]:
            return f"config.{field} differs"
    for field in PAIR_FIELDS:
        if metal[field] != faiss[field]:
            return f"{field} differs"
    for field in RECIPE_FIELDS:
        if not semantic_equal(metal["recipe_resolved"][field], faiss["recipe_resolved"][field]):
            return f"resolved recipe.{field} differs"
    return None


def matched_pairs(rows):
    """Fail closed for ambiguous or incompatible same-dataset/seed/trial records."""
    candidates = defaultdict(list)
    rejected, pairs = [], []
    for row in rows:
        problem = row_problem(row)
        if problem:
            rejected.append({"rows": [row], "reason": problem})
            continue
        c = row["config"]
        candidates[(c["dataset"], c["seed"], c["trial"])].append(row)
    for key in sorted(candidates):
        batch = candidates[key]
        metals = [r for r in batch if r["config"]["method"] == "metal"]
        cpus = [r for r in batch if r["config"]["method"] == "faiss-emeans"]
        if len(metals) != 1 or len(cpus) != 1:
            reason = "pending counterpart" if not metals or not cpus else "ambiguous duplicate method records"
            rejected.append({"rows": batch, "reason": reason})
            continue
        a, b = metals[0], cpus[0]
        problem = pair_problem(a, b)
        if problem:
            rejected.append({"rows": batch, "reason": problem})
            continue
        pairs.append({"metal": a, "faiss": b, "speedup": b["train_seconds"] / a["train_seconds"],
                      "mse_change_pct": (100 * (a["test_mse"] / b["test_mse"] - 1)) if b["test_mse"] else None})
    return pairs, rejected


def memory_problem(row):
    m = row.get("memory", {})
    if m.get("complete") is not True or m.get("sampling_errors") != []:
        return "memory sampling incomplete or errored"
    for key in ("peak_physical_footprint_bytes", "peak_rss_bytes", "sample_interval_seconds",
                "maximum_sample_gap_seconds", "duration_seconds", "sample_count"):
        if not positive(m.get(key)):
            return f"invalid memory.{key}"
    if m["sample_count"] < 2:
        return "fewer than two memory samples"
    if not math.isclose(m["sample_interval_seconds"], row["config"]["memory_interval_seconds"], rel_tol=1e-12):
        return "memory sampling interval differs from configuration"
    for peak, increase, field in (
        ("peak_physical_footprint_bytes", "physical_footprint_peak_increase_bytes", "physical_footprint_bytes"),
        ("peak_rss_bytes", "rss_peak_increase_bytes", "rss_bytes"),
    ):
        baseline = m.get("baseline", {}).get(field)
        end = m.get("end", {}).get(field)
        if not nonnegative(baseline) or not nonnegative(end) or not nonnegative(m.get(increase)):
            return "invalid baseline, end, or memory increase"
        if m[peak] < max(baseline, end) or m[increase] != m[peak] - baseline:
            return "inconsistent sampled memory peak/increase"
    return None


def grouped_pairs(pairs):
    """Aggregate only matched fresh-process trials, never reuse a CPU trial."""
    by_group = defaultdict(list)
    for pair in pairs:
        row = pair["metal"]
        identity = {key: row[key] for key in PAIR_FIELDS if key != "initial_centroids_sha256"}
        identity["config"] = {key: row["config"][key] for key in CONFIG_FIELDS if key not in ("seed", "trial")}
        identity["recipe"] = {key: row["recipe_resolved"][key] for key in RECIPE_FIELDS if key != "seed"}
        by_group[stable(identity)].append(pair)
    groups = []
    for batch in by_group.values():
        first = batch[0]["metal"]
        result = {key: first[key] for key in ("n", "d", "k")}
        result.update(dataset=first["config"]["dataset"], pairs=batch,
                      paired_trials=len(batch), seeds=sorted({p["metal"]["config"]["seed"] for p in batch}),
                      speedup=median(p["speedup"] for p in batch))
        changes = [p["mse_change_pct"] for p in batch if p["mse_change_pct"] is not None]
        result["mse_change_pct"] = median(changes) if len(changes) == len(batch) else None
        for method in ("metal", "faiss"):
            result[f"{method}_seconds"] = median(p[method]["train_seconds"] for p in batch)
            result[f"{method}_mse"] = median(p[method]["test_mse"] for p in batch)
            valid_memory = [p[method] for p in batch if memory_problem(p[method]) is None]
            result[f"{method}_memory_trials"] = len(valid_memory)
            for field in ("peak_physical_footprint_bytes", "peak_rss_bytes", "physical_footprint_peak_increase_bytes",
                          "rss_peak_increase_bytes", "maximum_sample_gap_seconds"):
                result[f"{method}_{field}"] = median(r["memory"][field] for r in valid_memory) if len(valid_memory) == len(batch) else None
            for field in ("physical_footprint_bytes", "rss_bytes"):
                result[f"{method}_baseline_{field}"] = median(r["memory"]["baseline"][field] for r in valid_memory) if len(valid_memory) == len(batch) else None
        both_valid = all(memory_problem(p[m]) is None for p in batch for m in ("metal", "faiss"))
        result["footprint_metal_over_faiss"] = median(p["metal"]["memory"]["peak_physical_footprint_bytes"] / p["faiss"]["memory"]["peak_physical_footprint_bytes"] for p in batch) if both_valid else None
        for field in ("peak_active_bytes", "cached_after_fit_bytes"):
            values = [(p["metal"].get("mlx_memory") or {}).get(field) for p in batch]
            result[f"mlx_{field}"] = median(values) if all(nonnegative(v) for v in values) else None
        groups.append(result)
    return sorted(groups, key=lambda g: g["dataset"])


def fmt(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def gib(value):
    return "—" if value is None else f"{value / 2**30:.3f}"


def main_table(groups, names, report):
    selected = [g for name in names for g in groups if g["dataset"] == name]
    if not selected:
        return ["No completed, provenance-matched pairs yet.", ""]
    lines = ["| Dataset | Rows × dimensions; k | Metal s | Faiss CPU s | CPU / Metal | Peak footprint GiB M / F | Test MSE M / F | MSE change | Paired trials; seeds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for g in selected:
        first = g["pairs"][0]
        a = link(first["metal"]["_path"], fmt(g["metal_seconds"]), report)
        b = link(first["faiss"]["_path"], fmt(g["faiss_seconds"]), report)
        change = "—" if g["mse_change_pct"] is None else f"{g['mse_change_pct']:+.3f}%"
        lines.append(f"| {label(g['dataset'])} | {g['n']:,} × {g['d']}; {g['k']:,} | {a} | {b} | "
                     f"{g['speedup']:.2f}× | {gib(g['metal_peak_physical_footprint_bytes'])} / {gib(g['faiss_peak_physical_footprint_bytes'])} | "
                     f"{g['metal_mse']:.8g} / {g['faiss_mse']:.8g} | {change} | {g['paired_trials']}; {', '.join(map(str, g['seeds']))} |")
    return lines + ["", "M / F means Metal / Faiss CPU. Positive MSE change means Metal has higher (worse) loss. "
                    "Peak footprint is the sampled fit-window process total, including its pre-fit baseline. "
                    "Missing or incomplete memory samples appear as —; timing and quality remain separately usable.", ""]


def coverage_table(rows, groups, rejected):
    complete = {g["dataset"] for g in groups}
    lines = ["| Configuration | Scope | Result |", "|---|---|---|"]
    for name in PRIMARY + CORE_10M:
        available = [r for r in rows if r.get("config", {}).get("dataset") == name]
        reasons = {r["reason"] for r in rejected if any(x.get("config", {}).get("dataset") == name for x in r["rows"])}
        if name in complete:
            status = "paired" + ("; excluded records also listed below" if reasons else "")
        elif reasons:
            status = "; ".join(sorted(reasons))
        elif available:
            status = "unpaired"
        else:
            status = "pending" if name in PRIMARY else "deferred by requested scope; not run"
        lines.append(f"| `{name}` | {'primary' if name in PRIMARY else '10M deferred'} | {status} |")
    return lines + [""]


def memory_table(pairs, report):
    lines = ["| Dataset | Method; seed/trial | Baseline footprint GiB | Peak footprint GiB | Increase GiB | Peak RSS GiB | MLX active peak GiB | MLX cache after fit GiB | Samples; max gap ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for pair in pairs:
        for method in ("metal", "faiss"):
            row = pair[method]
            c = row["config"]
            memory = row.get("memory", {})
            valid = memory_problem(row) is None
            name = link(row["_path"], label(c["dataset"]), report)
            baseline = memory.get("baseline", {}).get("physical_footprint_bytes") if valid else None
            peak = memory.get("peak_physical_footprint_bytes") if valid else None
            increase = memory.get("physical_footprint_peak_increase_bytes") if valid else None
            rss = memory.get("peak_rss_bytes") if valid else None
            allocator = (row.get("mlx_memory") or {}).get("peak_active_bytes") if method == "metal" else None
            cache = (row.get("mlx_memory") or {}).get("cached_after_fit_bytes") if method == "metal" else None
            sampling = f"{memory['sample_count']:,}; {memory['maximum_sample_gap_seconds'] * 1000:.2f}" if valid else "incomplete"
            lines.append(f"| {name} | {method}; {c['seed']}/{c['trial']} | {gib(baseline)} | {gib(peak)} | "
                         f"{gib(increase)} | {gib(rss)} | {gib(allocator)} | {gib(cache)} | {sampling} |")
    return lines + [""]


def memory_variability(rows, pairs):
    """Validate supplemental Metal-only DEEP trials against primary evidence."""
    if not rows:
        return None
    anchors = [p["metal"] for p in pairs if p["metal"]["config"]["dataset"] == "deep1b-1M"]
    anchor = anchors[0] if len(anchors) == 1 else None
    accepted, rejected = [], []
    by_trial = defaultdict(list)
    for row in rows:
        problem = row_problem(row)
        c = row.get("config", {})
        if not problem and (c["dataset"] != "deep1b-1M" or c["method"] != "metal"):
            problem = "supplement requires full DEEP1M Metal fits"
        if not problem and anchor is None:
            problem = "supplement requires one unambiguous paired primary DEEP Metal record"
        if not problem:
            # Trial number intentionally differs; all measured-work/provenance
            # fields, seed and resolved recipe must still match the primary fit.
            comparable = {**row, "config": {**c, "trial": anchor["config"]["trial"]}}
            problem = pair_problem(comparable, anchor) or memory_problem(row)
        if not problem:
            allocator = row.get("mlx_memory") or {}
            if any(not nonnegative(allocator.get(field)) for field in ("peak_active_bytes", "cached_after_fit_bytes")):
                problem = "missing or invalid supplemental MLX allocator fields"
        if problem:
            rejected.append({"rows": [row], "reason": problem})
        else:
            by_trial[c["trial"]].append(row)
    for trial in sorted(by_trial):
        batch = by_trial[trial]
        if len(batch) == 1:
            accepted.append(batch[0])
        else:
            rejected.append({"rows": batch, "reason": "ambiguous duplicate supplemental trial"})
    return {"anchor": anchor, "rows": accepted, "rejected": rejected}


def variability_table(supplement, report):
    if supplement is None:
        return []
    lines = ["## Additional DEEP 1M memory trials", "",
             f"This appendix contains **{len(supplement['rows'])} additional measured Metal-only first fits** "
             "on full DEEP 1M, with the same seed, initial centroids, resolved recipe, source, environment, and protocol "
             "as the primary DEEP Metal record. These trials are **excluded from all primary paired tables, figures, "
             "coverage counts, and headline statistics**. The primary record is repeated here only for context. "
             "Observed differences illustrate process/cache variability; this small sample does not establish a memory confidence interval.", ""]
    if supplement["rows"]:
        lines += ["| Record | Seed; trial | Fit seconds | Peak process footprint GiB | MLX active peak GiB | Retained MLX cache after fit GiB |",
                  "|---|---|---:|---:|---:|---:|"]
        records = [("Primary reference", supplement["anchor"])] + [("Additional trial", r) for r in supplement["rows"]]
        for title, row in records:
            c, m, allocator = row["config"], row["memory"], row["mlx_memory"]
            peak = m["peak_physical_footprint_bytes"] if memory_problem(row) is None else None
            lines.append(f"| {link(row['_path'], title, report)} | {c['seed']}; {c['trial']} | {row['train_seconds']:.3f} | "
                         f"{gib(peak)} | {gib(allocator.get('peak_active_bytes'))} | {gib(allocator.get('cached_after_fit_bytes'))} |")
        lines += ["", "Process footprint includes baseline and retained cache. MLX active and cached allocation diagnostics "
                  "are not additive to process footprint, and active peak and end-of-fit cache are observed at different instants.", ""]
    for rejected in supplement["rejected"]:
        paths = ", ".join(link(r["_path"], r["_path"].name, report) for r in rejected["rows"])
        lines.append(f"- Supplemental record excluded ({rejected['reason']}): {paths}")
    return lines + [""]


def figure_context(groups):
    environments = {(p["metal"]["hardware"].get("cpu", "unknown hardware"), p["faiss"]["effective_omp_threads"])
                    for g in groups for p in g["pairs"]}
    return "; ".join(f"{hardware} · Faiss CPU: {threads} OpenMP threads" for hardware, threads in sorted(environments))


def summary_rows(groups):
    return [{key: (stable(value) if key == "seeds" else value) for key, value in group.items() if key != "pairs"} for group in groups]


def write_csv(groups, path):
    rows = summary_rows(groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "metal_seconds", "faiss_seconds", "speedup"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def figures(groups, folder):
    """Static scientific plots; deliberately imported only on explicit request."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suite, names in (("core1m", CORE_1M), ("vibe", VIBE), ("core10m", CORE_10M)):
        selected = [g for name in names for g in groups if g["dataset"] == name]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, max(4.5, len(selected) * .48 + 1.5)), layout="constrained")
        y = list(range(len(selected)))
        for method, color, title in (("metal", "#137DAB", "Metal GPU"), ("faiss", "#D66A32", "Faiss CPU")):
            axes[0].scatter([g[f"{method}_seconds"] for g in selected], y, color=color, label=title, s=40)
            memory = [(i, g[f"{method}_peak_physical_footprint_bytes"]) for i, g in enumerate(selected)
                      if g[f"{method}_peak_physical_footprint_bytes"] is not None]
            axes[1].scatter([v / 2**30 for _, v in memory], [i for i, _ in memory], color=color, s=40)
        for i, g in enumerate(selected):
            axes[0].plot([g["metal_seconds"], g["faiss_seconds"]], [i, i], color="#C7CBD0", zorder=0)
        changes = [(i, g["mse_change_pct"]) for i, g in enumerate(selected) if g["mse_change_pct"] is not None]
        axes[2].barh([i for i, _ in changes], [v for _, v in changes], color=["#D66A32" if v > 0 else "#137DAB" for _, v in changes])
        axes[2].axvline(0, color="#666", linewidth=.8)
        axes[0].set_xscale("log")
        low, high = axes[0].get_xlim()
        ticks = [value for value in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                 if low <= value <= high]
        axes[0].set_xticks(ticks, [f"{value:g}" for value in ticks])
        axes[0].xaxis.set_minor_formatter(NullFormatter())
        axes[1].set_xlim(left=0)
        axes[0].set_xlabel("First-fit seconds (log scale)")
        axes[1].set_xlabel("Sampled peak process footprint (GiB)")
        axes[2].set_xlabel("Metal MSE change vs Faiss (%)\npositive = worse")
        axes[0].legend(loc="best", frameon=False)
        for i, axis in enumerate(axes):
            axis.set_yticks(y, [label(g["dataset"]) for g in selected] if i == 0 else [""] * len(y))
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
        counts = sorted({g["paired_trials"] for g in selected})
        fig.suptitle(f"Three-epoch e-means · {suite} · Faiss recipe, float32\n"
                     f"{figure_context(selected)}\n"
                     f"Matched fresh-process first fits; paired trials per configuration: {', '.join(map(str, counts))}", fontsize=12)
        path = folder / f"{suite}-metal-faiss.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def build_report(rows, pairs, groups, rejected, report, csv_path, manifest_path, plot_paths, supplement=None):
    primary = [g for g in groups if g["dataset"] in PRIMARY]
    coverage = len({g["dataset"] for g in primary})
    lines = ["# Metal versus original Faiss e-means — 14 November 2026", "",
             "This compares **three-epoch Metal GPU e-means with three-epoch unmodified upstream CPU `NumpyEmeans`**, "
             "using the same current Faiss recipe, float32 data, last-epoch finalization, and identical initial centroids. "
             "Each recorded measurement is one first fit in a fresh process, with no in-process warm-up.", "",
             f"**Primary coverage: {coverage}/{len(PRIMARY)} dataset configurations paired** "
             "(six core 1M configurations and eleven VIBE configurations). The five 10M configurations are deferred "
             "by the requested scope; their absence is not a failed benchmark.", ""]
    if primary and len(primary) == coverage:
        speed = [g["speedup"] for g in primary]
        changes = [g["mse_change_pct"] for g in primary if g["mse_change_pct"] is not None]
        lines.append(f"Across completed primary configurations, the observed CPU/Metal runtime ratio ranges from "
                     f"**{min(speed):.2f}× to {max(speed):.2f}×** (median {median(speed):.2f}×).")
        if changes:
            lines.append(f"Metal test-MSE changes range from **{min(changes):+.3f}% to {max(changes):+.3f}%** "
                         f"(median {median(changes):+.3f}%; positive means worse). These are measured differences, not an assertion of numerical identity.")
            lines.append(f"Metal MSE is lower on **{sum(v < 0 for v in changes)}**, equal on **{sum(v == 0 for v in changes)}**, "
                         f"and higher on **{sum(v > 0 for v in changes)}** completed configurations, using the unrounded paired percentage changes.")
        footprint = [g["footprint_metal_over_faiss"] for g in primary if g["footprint_metal_over_faiss"] is not None]
        if footprint:
            lines.append(f"The observed Metal/Faiss peak-process-footprint ratio ranges from **{min(footprint):.2f}× "
                         f"to {max(footprint):.2f}×** (median {median(footprint):.2f}×; {len(footprint)} configurations with complete memory sampling).")
        lines.append("")
    elif primary:
        lines += ["Multiple provenance/protocol variants are present for at least one configuration; "
                  "no overall range or median is pooled across those variants.", ""]
    lines += ["## Core 1M", "", "Both implementations run three epochs. Every configuration below uses 1,000,000 training vectors and 16,384 clusters.", ""]
    lines += main_table(groups, CORE_1M, report)
    lines += ["## VIBE", "", "Both implementations run three epochs on the full prepared training split. "
              "These use the same public VIBE source vectors but a new, recorded duplicate-disjoint split, "
              "because the paper's exact holdout indices were not released. They are not an exact reproduction of paper Table 11.", ""]
    lines += main_table(groups, VIBE, report)
    if any(g["dataset"] in CORE_10M for g in groups):
        lines += ["## Additional 10M evidence", "", "These results are supplemental and excluded from the 17-configuration headline.", ""]
        lines += main_table(groups, CORE_10M, report)
    if plot_paths:
        lines += ["## Figures", ""]
        for path in plot_paths:
            lines += ["!" + link(path, path.stem.replace("-", " "), report), ""]
    lines += ["## Measurement and interpretation", "",
              "A fresh worker loads data and constructs the shared NumPy initial centroids before signaling `FIT_READY`. "
              "The parent starts external process-memory sampling, then permits the fit. `FIT_DONE` ends that window before quality scoring. "
              "The fit timer includes validation, Metal upload where applicable, shuffling, centroid updates/finalization, "
              "synchronization, and a host centroid copy. Loading, initial-centroid sampling, and scoring are excluded. "
              "Any compilation that occurs inside the fit is included. Cold here means no in-process algorithm warm-up; "
              "operating-system file and shader caches are not flushed, so this is not a cold-machine measurement.", "",
              "Time, MSE, and memory columns are medians over paired fresh-process trials when more than one exists. "
              "The speedup column is the median of paired CPU/Metal ratios; MSE change is the median of paired percentage changes. "
              "Every MSE comes from its own timed fit. Repeated trials with the same seed are timing replications, "
              "not independent initialization seeds; the table reports both counts. A single cold fit does not establish timing variance or steady-state throughput. "
              "Headline ranges and medians give each completed configuration one weight and do not pool losses across datasets.", "",
              "The main memory metric is the maximum sampled **process physical footprint during the fit window**, "
              "including the baseline after data loading/imports and any retained MLX cache. It is an observed first-fit peak, "
              "not a deterministic minimum workspace. Retained allocator cache and process footprint can vary between fresh processes "
              "even when the active-allocation peak is unchanged. "
              "RSS is separately reported; it is not interchangeable with physical footprint, particularly for file-backed VIBE inputs. "
              "Baseline-to-peak increase is a phase increase, not total dataset storage or exclusive GPU memory. "
              "MLX's active-allocation peak describes its allocator only. **Do not add footprint, RSS, or MLX allocation:** their accounting overlaps. "
              "The kernel lifetime peak includes loading and is retained only as a raw diagnostic, never used for the fit-window peak. "
              "Polling can miss brief transients; each raw record includes the requested interval, observed maximum gap, sample count, "
              "and errors. Incomplete memory collection is withheld from the memory comparison.", "",
              "Quality uses one shared evaluator: Faiss CPU `IndexFlatL2` float32 nearest-centroid assignment followed by "
              "direct float64 squared residuals averaged per vector (sum over dimensions). Identical initial centroids and matched "
              "recipe constants do not guarantee identical floating-point assignments or revival choices in these iterative algorithms. "
              "The table reports the resulting quality differences directly. Faiss draws donor random variates only on "
              "revival checks requiring a split; Metal draws on every five-batch check, including empty checks, to avoid "
              "CPU synchronization. After an empty check, the same seed therefore follows a different donor-random-number "
              "trajectory. Reduction order, centered assignment, and donor-key precision also differ. Their separate "
              "contributions to the observed MSE differences were not ablated. "
              f"See the {link(REPO / 'docs/e-means.md', 'usage guide and numerical-differences discussion', report)}.", "",
              "Core data and released split fingerprints follow the pinned upstream loader. The released DEEP 1M evaluation "
              "contains four vectors also found in training; SPACEV 1M contains six. These published overlaps are retained, "
              "so the core evaluations are not all strictly disjoint. SIFT 1M uses the same corpus and first-million/next-10K split "
              "but has no released 1M fingerprint. SIFT is divided by 512 and DINO by 4096, as in upstream; "
              "multiply MSE by the divisor squared to recover raw-vector units. VIBE uses source vectors as stored and Euclidean clustering.", "",
              "## Hardware, software, and pairing checks", ""]
    environments = {stable({key: p["metal"][key] for key in ("hardware", "versions", "effective_omp_threads", "thread_environment", "faiss_compile_options")}): p["metal"] for p in pairs}
    if environments:
        for row in environments.values():
            h, c = row["hardware"], row["config"]
            lines.append(f"Host: **{h.get('cpu', 'unknown')}**, {h.get('ram_bytes', 0) / 2**30:g} GiB unified memory; "
                         f"`{h.get('platform', 'unknown')}`. CPU OpenMP threads: **{row['effective_omp_threads']}**. "
                         f"`VECLIB_MAXIMUM_THREADS={c['veclib_threads'] if c['veclib_threads'] is not None else 'unset'}` "
                         "(Accelerate may choose its own worker count). Metal uses the same host's Apple GPU.")
            lines += ["", "Versions: " + "; ".join(f"{key} `{value}`" for key, value in row["versions"].items()) + ".", "",
                      f"Faiss compile options: `{row['faiss_compile_options'].strip()}`.", ""]
    else:
        lines += ["No matched environment has been recorded yet.", ""]
    calibration = HERE / "results/faiss_comparison_calibration/summary.json"
    if calibration.exists():
        cal = json.loads(calibration.read_text())
        full = cal.get("full_deep_seconds", {})
        details = ", ".join(f"{threads} threads: {seconds:.3f} s" for threads, seconds in sorted(full.items(), key=lambda item: int(item[0])))
        lines += [f"CPU selection used {link(calibration, 'bounded thread calibration', report)} "
                  f"({len(cal.get('trials', []))} trials). Full DEEP 1M three-epoch observations: {details}. "
                  f"The chosen setting was {cal['selected_openmp_threads']} OpenMP threads with unrestricted Accelerate. "
                  "One timing per calibration configuration does not establish a globally optimal thread count.", ""]
    revisions = sorted({p["metal"]["faiss_reference_revision"] for p in pairs})
    for revision in revisions:
        lines += [f"Original source: [Faiss contrib/e_means at `{revision[:12]}`](https://github.com/facebookresearch/faiss/tree/{revision}/contrib/e_means). "
                  "The reference is upstream `NumpyEmeans` with CPU Faiss assignment; this is not a CUDA/H100 runtime comparison.", ""]
    lines += ["Pairing requires identical full-data metadata and fingerprints, dimensions, cluster count, initialization hash, "
              "seed/trial, source hashes, versions, host, thread settings, timing/scoring/memory scopes, and requested protocol. "
              "It also checks resolved batch law/size, learning rate, epoch rows/steps, total steps, alpha floor, "
              "initial usage mass, split epsilon, unused-cluster threshold/check cadence, tail policy, and seed. "
              "Numeric recipe comparisons permit only 1e-12 relative / 1e-15 absolute rounding tolerance. "
              "Missing provenance, ambiguous duplicates, or mismatches are excluded with reasons below.", "",
              "## Fit-window memory details", ""]
    lines += memory_table(pairs, report) if pairs else ["No paired measurements yet.", ""]
    lines += ["## Coverage and exclusions", ""]
    lines += coverage_table(rows, groups, rejected)
    for item in rejected:
        paths = ", ".join(link(r["_path"], r["_path"].name, report) for r in item["rows"])
        lines.append(f"- {item['reason']}: {paths}")
    memory_issues = [(p[m], memory_problem(p[m])) for p in pairs for m in ("metal", "faiss") if memory_problem(p[m])]
    for row, problem in memory_issues:
        lines.append(f"- Memory excluded ({problem}): {link(row['_path'], row['_path'].name, report)}")
    lines += [""] + variability_table(supplement, report)
    lines += ["", "## Reproducibility and raw evidence", "",
              f"{link(csv_path, 'Summary CSV', report)} · {link(manifest_path, 'Evidence manifest with all raw record hashes', report)} · "
              f"{link(HERE / 'compare.py', 'fresh-process comparison driver', report)} · "
              f"{link(HERE / 'memory.py', 'external memory sampler', report)} · "
              f"{link(HERE / 'comparison_report.py', 'report generator', report)}.", "",
              f"Validation evidence: {link(HERE / 'results/faiss_comparison_validation/validation.json', 'controller tests and CPU dispatch audit', report)} · "
              f"{link(HERE / 'results/faiss_comparison_validation/memory.json', 'external memory-sampler validation', report)} · "
              f"{link(HERE / 'results/faiss_comparison_validation/report.json', 'report calculations and exclusion tests', report)}.", "",
              f"The {link(REPO / 'docs/e-means-benchmarks-2026-11-14.md', 'earlier paper-recipe/Lloyd report', report)} and "
              "its original measurements are preserved. They are not pooled with these new cold-fit/memory measurements. "
              "The five 10M comparisons remain deferred; 100M/1B/10B, distributed runs, IVF downstream experiments, "
              "and FP8 comparisons are outside the measured scope.", "",
              "```bash", ".venv/bin/python benchmarks/e_means/compare.py --suite core1m --repeats 1",
              ".venv/bin/python benchmarks/e_means/compare.py --suite vibe --repeats 1",
              ".venv/bin/python benchmarks/e_means/comparison_report.py --plots", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/faiss_comparison")
    parser.add_argument("--output", type=Path, default=REPO / "docs/e-means-faiss-comparison-2026-11-14.md")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--figures", type=Path, default=HERE / "faiss_comparison_figures")
    parser.add_argument("--memory-variability", type=Path, help="Optional supplemental results; defaults to the sibling faiss_comparison_memory_variability folder")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    rows = evidence_rows(args.results)
    pairs, rejected = matched_pairs(rows)
    groups = grouped_pairs(pairs)
    supplemental_rows = evidence_rows(args.memory_variability or args.results.parent / "faiss_comparison_memory_variability")
    supplement = memory_variability(supplemental_rows, pairs)
    csv_path = args.csv or args.results / "comparison_summary.csv"
    manifest_path = args.manifest or args.results / "comparison_report_manifest.json"
    write_csv(groups, csv_path)
    plot_paths = figures(groups, args.figures) if args.plots else []
    manifest = {
        "reporter_sha256": sha256(__file__), "primary_configurations": list(PRIMARY),
        "deferred_configurations": list(CORE_10M), "paired_trials": len(pairs),
        "paired_primary_configurations": len({g['dataset'] for g in groups if g['dataset'] in PRIMARY}),
        "evidence": [{"path": str(r["_path"].relative_to(REPO)) if r["_path"].is_relative_to(REPO) else str(r["_path"]),
                      "sha256": sha256(r["_path"]), "status": r.get("status")} for r in rows],
        "exclusions": [{"reason": i["reason"], "paths": [str(r["_path"]) for r in i["rows"]]} for i in rejected],
        "supplemental_memory_variability": {
            "included_trials": len(supplement["rows"]) if supplement else 0,
            "excluded_from_primary": True,
            "evidence": [{"path": str(r["_path"]), "sha256": sha256(r["_path"])} for r in supplemental_rows],
            "exclusions": [{"reason": i["reason"], "paths": [str(r["_path"]) for r in i["rows"]]} for i in supplement["rejected"]] if supplement else [],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(rows, pairs, groups, rejected, args.output, csv_path, manifest_path, plot_paths, supplement))
    print(json.dumps({"report": str(args.output), "paired_trials": len(pairs),
                      "primary_configurations": manifest["paired_primary_configurations"], "excluded_groups": len(rejected)}))


if __name__ == "__main__":
    main()
