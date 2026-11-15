"""Build the e-means benchmark report exclusively from recorded evidence.

This program never imports MLX or runs a fit. Plotting is optional and should
be done after timed campaigns finish. It does not modify the frozen run.py.
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def label(name):
    if name in LABELS:
        return LABELS[name]
    for prefix, text in LABELS.items():
        if name.startswith(prefix + "-"):
            return text
    return name


def link(path, title, report):
    return f"[{title}]({Path(os.path.relpath(path, report.parent)).as_posix()})"


def evidence_rows(folder):
    rows = []
    for path in sorted(Path(folder).rglob("*.json")):
        item = json.loads(path.read_text())
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        if "config" not in item or "train_seconds_samples" not in item:
            continue
        samples = item["train_seconds_samples"]
        if not samples or not all(math.isfinite(x) and x > 0 for x in samples):
            raise ValueError(f"Invalid timer samples in {path}")
        if not math.isclose(median(samples), item["train_seconds"], rel_tol=1e-12):
            raise ValueError(f"Recorded median disagrees with samples in {path}")
        if not math.isfinite(item["test_mse"]) or item["test_mse"] < 0:
            raise ValueError(f"Invalid test MSE in {path}")
        item["_path"] = path.resolve()
        rows.append(item)
    return rows


def pairing_key(row):
    c = row["config"]
    # Match measured work, data, initialization, code and execution environment.
    # The algorithms' recipe/finalizer differ intentionally for a Lloyd baseline.
    return stable({
        "dataset": c["dataset"], "n": row["n"], "d": row["d"], "k": row["k"],
        "precision": c["precision"], "assignment": c["assignment"],
        "seed": c["seed"], "warmup": c["warmup"],
        "initial": row["initial_centroids_sha256"], "data": row["dataset"],
        "code": row.get("source_sha256"), "software": row["versions"],
        "hardware": row["hardware"], "timing_scope": row["timing_scope"],
    })


def matched_pairs(rows):
    baselines = defaultdict(list)
    for row in rows:
        if row["config"]["method"] == "metal-lloyd" and row["config"]["lloyd_iters"] == 25:
            baselines[pairing_key(row)].append(row)
    pairs, unpaired = [], []
    for row in rows:
        if row["config"]["method"] != "metal":
            continue
        candidates = baselines.get(pairing_key(row), []) if row.get("source_sha256") else []
        if not candidates:
            unpaired.append(row)
            continue
        # Multiple campaigns can contain the identical workload with different
        # repetition counts; choose the longest measured baseline deterministically.
        baseline = max(candidates, key=lambda x: (len(x["train_seconds_samples"]), str(x["_path"])))
        pairs.append({"emeans": row, "lloyd": baseline,
                      "speedup": baseline["train_seconds"] / row["train_seconds"],
                      "mse_change_pct": 100 * (row["test_mse"] / baseline["test_mse"] - 1)})
    return pairs, unpaired


def paired_groups(pairs):
    grouped = defaultdict(list)
    for pair in pairs:
        row, c = pair["emeans"], pair["emeans"]["config"]
        group = {k: c[k] for k in ["dataset", "epochs", "precision", "assignment", "recipe", "finalization"]}
        group.update(n=row["n"], d=row["d"], k=row["k"],
                     code=row["source_sha256"], versions=row["versions"], hardware=row["hardware"])
        grouped[stable(group)].append(pair)
    result = []
    for pairs_in_group in grouped.values():
        # Keep one e-means record per seed/config, preferring more repetitions.
        per_seed = {}
        for pair in pairs_in_group:
            row = pair["emeans"]
            seed = row["config"]["seed"]
            old = per_seed.get(seed)
            if old is None or len(row["train_seconds_samples"]) > len(old["emeans"]["train_seconds_samples"]):
                per_seed[seed] = pair
        chosen = [per_seed[s] for s in sorted(per_seed)]
        first = chosen[0]["emeans"]
        result.append({
            "dataset": first["config"]["dataset"], "n": first["n"], "d": first["d"], "k": first["k"],
            "epochs": first["config"]["epochs"], "precision": first["config"]["precision"],
            "recipe": first["config"]["recipe"], "finalization": first["config"]["finalization"],
            "emeans_seconds": median(p["emeans"]["train_seconds"] for p in chosen),
            "lloyd_seconds": median(p["lloyd"]["train_seconds"] for p in chosen),
            "speedup": median(p["speedup"] for p in chosen),
            "mse_change_pct": median(p["mse_change_pct"] for p in chosen),
            "emeans_mse": median(p["emeans"]["test_mse"] for p in chosen),
            "lloyd_mse": median(p["lloyd"]["test_mse"] for p in chosen),
            "seeds": sorted(per_seed), "pairs": chosen,
        })
    return sorted(result, key=lambda x: (x["dataset"], x["precision"], x["epochs"]))


def paired_table(groups, report):
    if not groups:
        return ["No completed, provenance-matched measurements yet.", ""]
    lines = ["| Dataset | Rows × dimensions; k | E-means (s) | Lloyd25 (s) | Speedup | Test MSE change | Seeds; timed fits E/L |",
             "|---|---:|---:|---:|---:|---:|---|", ]
    for g in groups:
        first = g["pairs"][0]
        e = link(first["emeans"]["_path"], f"{g['emeans_seconds']:.3f}", report)
        b = link(first["lloyd"]["_path"], f"{g['lloyd_seconds']:.3f}", report)
        es = sum(len(p["emeans"]["train_seconds_samples"]) for p in g["pairs"])
        bs = sum(len(p["lloyd"]["train_seconds_samples"]) for p in g["pairs"])
        title = label(g["dataset"])
        if g["epochs"] != 3 or g["precision"] != "float32" or g["recipe"] != "paper":
            title += f" ({g['epochs']:g} ep., {g['precision']}, {g['recipe']})"
        lines.append(f"| {title} | {g['n']:,} × {g['d']}; {g['k']:,} | {e} | {b} | "
                     f"{g['speedup']:.2f}× | {g['mse_change_pct']:+.3f}% | "
                     f"{', '.join(map(str, g['seeds']))}; {es}/{bs} |")
    lines.extend(["", "Negative MSE change is better. Timings are medians of synchronized fits; with multiple seeds, "
                  "the table uses medians of per-seed medians and per-seed speedup ratios. "
                  "Each linked timing opens its raw result (additional records are in the evidence index).", ""])
    lines.extend(["| Dataset | E-means test MSE | Lloyd25 test MSE |", "|---|---:|---:|"])
    for g in groups:
        lines.append(f"| {label(g['dataset'])} | {g['emeans_mse']:.9g} | {g['lloyd_mse']:.9g} |")
    return lines + [""]


def independent_table(rows, report):
    if not rows:
        return ["No completed measurements yet.", ""]
    lines = ["| Dataset | Rows × dimensions; k | Epochs | Fit seconds | Test MSE | Paper Flash25 median MSE | Difference | Seed; timed fits |",
             "|---|---:|---:|---:|---:|---:|---:|---|", ]
    for r in sorted(rows, key=lambda r: (r["config"]["dataset"], r["config"]["seed"], r["config"]["epochs"])):
        c, target = r["config"], r.get("paper_quality_reference")
        target_mse = f"{target['median_test_mse']:.9g}" if target else "unavailable"
        change = f"{100 * target['relative_mse']:+.3f}%" if target else "—"
        elapsed_link = link(r["_path"], f"{r['train_seconds']:.3f}", report)
        lines.append(f"| {label(c['dataset'])} | {r['n']:,} × {r['d']}; {r['k']:,} | {c['epochs']:g} | "
                     f"{elapsed_link} | {r['test_mse']:.9g} | "
                     f"{target_mse} | {change} | {c['seed']}; {len(r['train_seconds_samples'])} |")
    return lines + [""]


def stage(row):
    return row["_path"].parent.name


def protocol_description(rows, selected_stage):
    selected = [r for r in rows if stage(r) == selected_stage]
    if not selected:
        return "No completed timing records for this stage yet."
    seeds = sorted({r["config"]["seed"] for r in selected})
    repetitions = sorted({len(r["train_seconds_samples"]) for r in selected})
    warmed = {r["config"]["warmup"] for r in selected}
    prefix = f"Completed records use seeds {', '.join(map(str, seeds))}; timed fits per configuration: {', '.join(map(str, repetitions))}. "
    if warmed == {True}:
        return prefix + "Each has one full untimed warm-up."
    if warmed == {False}:
        return prefix + "Each is a first fit in a fresh process, without a full warm-up; fit-time compilation may be included."
    return prefix + "Warm-up policies differ; consult the linked configuration before comparing times."


def stability_table(rows):
    selected = [r for r in rows if stage(r) in {"core1m", "core1m_seeds"}
                and r["config"]["method"] == "metal" and r["config"]["epochs"] == 3]
    grouped = defaultdict(list)
    for r in selected:
        c = r["config"]
        key = stable({"dataset": c["dataset"], "data": r["dataset"], "sources": r["source_sha256"],
                      "versions": r["versions"], "hardware": r["hardware"], "precision": c["precision"],
                      "recipe": c["recipe"], "finalization": c["finalization"]})
        grouped[key].append(r)
    lines = ["| Dataset | Seeds | Test MSE min | Median | Max | Range / median |",
             "|---|---|---:|---:|---:|---:|"]
    count = 0
    for data in sorted(grouped.values(), key=lambda g: g[0]["config"]["dataset"]):
        per_seed = {}
        for r in data:
            seed = r["config"]["seed"]
            if seed not in per_seed or len(r["train_seconds_samples"]) > len(per_seed[seed]["train_seconds_samples"]):
                per_seed[seed] = r
        if len(per_seed) < 2:
            continue
        mses = [r["test_mse"] for r in per_seed.values()]
        middle = median(mses)
        spread = 100 * (max(mses) - min(mses)) / middle if middle else 0
        lines.append(f"| {label(data[0]['config']['dataset'])} | {', '.join(map(str, sorted(per_seed)))} | "
                     f"{min(mses):.9g} | {middle:.9g} | {max(mses):.9g} | {spread:.3f}% |")
        count += 1
    return lines + [""] if count else ["Fewer than two completed seeds per dataset are available.", ""]


def crossing_table(pairs, report):
    grouped = defaultdict(list)
    for p in pairs:
        if stage(p["emeans"]) in {"core1m", "convergence"}:
            grouped[p["emeans"]["config"]["dataset"]].append(p)
    lines = ["| Dataset | Tested epoch budgets | First observed budget at or below local Lloyd25 MSE | Fit seconds | Matched speedup |",
             "|---|---|---:|---:|---:|"]
    count = 0
    for name, candidates in sorted(grouped.items()):
        if not any(stage(p["emeans"]) == "convergence" for p in candidates):
            continue
        candidates.sort(key=lambda p: p["emeans"]["config"]["epochs"])
        reached = [p for p in candidates if p["mse_change_pct"] <= 0]
        budgets = ", ".join(f"{p['emeans']['config']['epochs']:g}" for p in candidates)
        if reached:
            p = reached[0]
            r = p["emeans"]
            time_link = link(r["_path"], f"{r['train_seconds']:.3f}", report)
            lines.append(f"| {label(name)} | {budgets} | {r['config']['epochs']:g} | {time_link} | {p['speedup']:.2f}× |")
        else:
            lines.append(f"| {label(name)} | {budgets} | not reached | — | — |")
        count += 1
    return lines + [""] if count else ["No completed convergence-budget campaign yet.", ""]


def cpu_table(rows, report):
    selected = [r for r in rows if stage(r) == "cpu_reference"]
    cpu = [r for r in selected if r["config"]["method"] == "faiss-emeans"]
    metal = [r for r in selected if r["config"]["method"] == "metal"]
    lines = ["| Dataset | Metal Faiss recipe (s) | Upstream CPU (s) | CPU / Metal | Metal / CPU test MSE |",
             "|---|---:|---:|---:|---:|"]
    found = 0
    for m in metal:
        for b in cpu:
            mh = {k: v for k, v in m["hardware"].items() if k != "metal"}
            if (m["config"]["dataset"] != b["config"]["dataset"] or m["dataset"] != b["dataset"]
                or m["initial_centroids_sha256"] != b["initial_centroids_sha256"]
                or m["source_sha256"] != b["source_sha256"] or m["versions"] != b["versions"]
                or mh != b["hardware"] or m["config"]["recipe"] != "faiss"
                or m["config"]["precision"] != "float32" or m["config"]["epochs"] != b["config"]["epochs"]
                or m["config"]["warmup"] != b["config"]["warmup"]):
                continue
            mt = link(m["_path"], f"{m['train_seconds']:.3f}", report)
            bt = link(b["_path"], f"{b['train_seconds']:.3f}", report)
            lines.append(f"| {label(m['config']['dataset'])} | {mt} | {bt} | "
                         f"{b['train_seconds']/m['train_seconds']:.2f}× | {m['test_mse']:.9g} / {b['test_mse']:.9g} |")
            found += 1
    return lines + [""] if found else ["No completed, matched upstream CPU reference measurements yet.", ""]


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "n", "d", "k", "epochs", "lloyd_iters", "precision", "seed", "recipe",
              "finalization", "seconds", "samples", "min_seconds", "max_seconds", "test_mse",
              "train_sample_mse", "train_score_rows", "paper_split_exact", "peak_gpu_memory_bytes", "raw_json"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            c = r["config"]
            record = {k: c[k] for k in ["dataset", "method", "epochs", "lloyd_iters", "precision", "seed", "recipe", "finalization"]}
            record.update({k: r.get(k) for k in ["n", "d", "k", "test_mse", "train_sample_mse", "train_score_rows", "peak_gpu_memory_bytes"]})
            record.update(seconds=r["train_seconds"], samples=len(r["train_seconds_samples"]),
                          min_seconds=min(r["train_seconds_samples"]), max_seconds=max(r["train_seconds_samples"]),
                          paper_split_exact=r["dataset"].get("paper_split_exact"),
                          raw_json=str(r["_path"].relative_to(REPO)))
            writer.writerow(record)


def figures(groups, ten_m, folder):
    """Publication-style static plots; call only after timing has finished."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140, "savefig.dpi": 180})
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for section, selected in [("core1m", [g for g in groups if not g["dataset"].startswith("vibe-") and g["n"] == 1_000_000]),
                              ("vibe", [g for g in groups if g["dataset"].startswith("vibe-")])]:
        if not selected:
            continue
        height = max(3.5, 0.44 * len(selected) + 1.6)
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, height), sharey=True,
                                        gridspec_kw={"width_ratios": [1.4, 1]})
        y = np.arange(len(selected))
        names = [label(g["dataset"]) for g in selected]
        left.barh(y - .18, [g["lloyd_seconds"] for g in selected], height=.34,
                  label="Metal Lloyd, 25 epochs", color="#94a3b8")
        budgets = sorted({g["epochs"] for g in selected})
        em_label = f"Metal e-means, {budgets[0]:g} epochs" if len(budgets) == 1 else "Metal e-means (see budgets in report)"
        left.barh(y + .18, [g["emeans_seconds"] for g in selected], height=.34,
                  label=em_label, color="#2563eb")
        left.set_yticks(y, names)
        left.set_xlabel(("Warmed" if section == "core1m" else "First, unwarmed")
                        + " fit time (seconds; lower is better)")
        left.invert_yaxis()
        left.legend(loc="lower right", frameon=False, fontsize=8)
        values = [g["mse_change_pct"] for g in selected]
        right.barh(y, values, height=.55, color=["#0f766e" if v <= 0 else "#c2410c" for v in values])
        right.axvline(0, color="#64748b", linewidth=.9)
        right.set_xlabel("Test MSE change vs. Lloyd25 (%)\nNegative is better")
        for i, g in enumerate(selected):
            left.annotate(f"{g['speedup']:.1f}×", (g["emeans_seconds"], i + .18),
                          xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        for ax in [left, right]:
            ax.grid(axis="x", alpha=.15)
            ax.set_axisbelow(True)
        title = "Core datasets: one million training vectors" if section == "core1m" else "VIBE: full corpora with new, duplicate-disjoint holdouts"
        fig.suptitle(title, x=.02, ha="left", fontsize=13, fontweight="bold")
        first_row = selected[0]["pairs"][0]["emeans"]
        clusters = "16,384 clusters" if section == "core1m" else "8,192 or 16,384 clusters"
        fig.text(.02, .905, f"{first_row['hardware']['cpu']} · float32 · {clusters}", fontsize=9, color="#475569")
        fig.tight_layout(rect=(0, .025, 1, .89))
        path = folder / f"{section}-time-and-quality.png"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        outputs[section] = path
    if ten_m:
        selected = sorted(ten_m, key=lambda r: r["config"]["dataset"])
        fig, ax = plt.subplots(figsize=(9, 4))
        values = [r["train_seconds"] for r in selected]
        bars = ax.bar([label(r["config"]["dataset"]) for r in selected], values, color="#2563eb")
        ax.bar_label(bars, labels=[f"{v:.2f}s" for v in values], padding=3)
        ax.set_ylabel("First, unwarmed e-means fit time (seconds)")
        ax.set_title("Ten million vectors: three-epoch e-means, 65,536 clusters\n"
                     f"{selected[0]['hardware']['cpu']} · float32 · single first fit", loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.16)
        fig.tight_layout()
        path = folder / "core10m-emeans-time.png"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        outputs["core10m"] = path
    return outputs


def build(args):
    folder, report = Path(args.results).resolve(), Path(args.output).resolve()
    rows = evidence_rows(folder)
    pairs, unpaired = matched_pairs(rows)
    core = paired_groups([p for p in pairs if stage(p["emeans"]) == "core1m"
                          and p["emeans"]["n"] == 1_000_000 and p["emeans"]["k"] == 16384
                          and p["emeans"]["config"]["epochs"] == 3
                          and p["emeans"]["config"]["recipe"] == "paper"])
    vibe = paired_groups([p for p in pairs if stage(p["emeans"]) == "vibe"])
    ten_m = [r for r in rows if r["n"] == 10_000_000 and r["config"]["method"] == "metal"]
    plots = figures(core + vibe, ten_m, Path(args.figures).resolve()) if args.plots else {}
    csv_path = folder / "summary.csv"
    write_csv(rows, csv_path)
    lines = ["# e-means on Apple Silicon — benchmark evidence", "", "Recorded campaign: 2026-11-14.", ""]
    if core:
        gains = [g["speedup"] for g in core]
        better = sum(g["mse_change_pct"] <= 0 for g in core)
        lines += [f"On the recorded Apple Silicon machine, **three-epoch e-means** ran "
                  f"**{min(gains):.2f}–{max(gains):.2f}× faster** than **25-epoch Metal Lloyd** "
                  "in the paired core configurations. "
                  f"E-means had equal or lower test MSE in **{better}/{len(core)}** configurations. "
                  "These are measured fixed-budget comparisons; the tables state the budgets, seeds, "
                  "and repetition counts. They are not H100 speedup estimates or an exhaustive search "
                  "for the first equal-quality epoch.", ""]
    else:
        lines += ["No complete paired core results are present yet. This report contains only completed evidence, "
                  "and its missing sections are not benchmark claims.", ""]
    if rows:
        machines = {stable(r["hardware"]): r["hardware"] for r in rows}
        environments = {stable(r["versions"]): r["versions"] for r in rows}
        metal_hosts = {
            stable({k: v for k, v in hardware.items() if k != "metal"})
            for hardware in machines.values() if hardware.get("metal")
        }
        for hardware in machines.values():
            if not hardware.get("metal") and stable(hardware) in metal_hosts:
                continue  # The CPU reference records the same host without GPU metadata.
            lines += [f"Machine: **{hardware['cpu']}**, {hardware['ram_bytes'] / 2**30:.0f} GiB unified RAM; "
                      f"`{hardware['platform']}`. GPU: `{hardware.get('metal', {}).get('device_name', 'not recorded')}`.", ""]
        for versions in environments.values():
            lines += ["Software: " + ", ".join(f"{k} {v}" for k, v in versions.items()) + ".", ""]
        precisions = sorted({r["config"]["precision"] for r in rows if r["config"]["method"].startswith("metal")})
        lines += [f"Recorded Metal assignment precision: **{', '.join(precisions)}**. "
                  "EMA state and accumulated statistics use float32. Reduced precision is never selected implicitly.", ""]
    lines += ["## One-million-row core datasets", "", "These rows use the actual paper corpora at one million training vectors "
              "and 16,384 clusters, comparing **three-epoch e-means with 25-epoch Metal Lloyd**. "
              "Each timing is linked to raw JSON. SIFT1M follows the same prefix/split protocol "
              "but has no released 1M fingerprint; the other core splits are fingerprint-verified against Faiss.", "",
              protocol_description(rows, "core1m"), ""]
    lines += paired_table(core, report)
    if "core1m" in plots:
        lines += [f"![Core data fit time and test quality]({os.path.relpath(plots['core1m'], report.parent)})", ""]
    lines += ["## Seed stability and observed quality crossings", "",
              "The independent seed campaign repeats three-epoch e-means at seeds 11 and 12. "
              "The table combines their completed results with seed 10 and reports the observed MSE range. "
              "These rows measure e-means stability. Additional paired speedups would require their own "
              "seed-matched Lloyd runs.", "", protocol_description(rows, "core1m_seeds"), ""]
    lines += stability_table(rows)
    lines += ["The convergence campaign uses separate fits at one and two epochs; the primary three-epoch "
              "fit supplies the third budget. Each fit has its own cosine schedule. The first successful "
              "tested budget is an observed upper bound on a quality crossing, not a search at the paper's "
              "0.2-epoch resolution. Each point uses its recorded median; the different campaigns' "
              "sample counts are stated separately.", "", protocol_description(rows, "convergence"), ""]
    lines += crossing_table(pairs, report)
    lines += ["## VIBE: separate evaluation splits", "", "These are the eleven public corpora used in the paper's Appendix D.3. "
              "Their full source files were verified by SHA-256. The release does not provide those eleven holdout lists, "
              "so this campaign uses a seeded 20,000-row holdout, removes exact matching vectors from training, "
              "and scores its second 10,000-row half. The full remaining corpus is used; row counts are shown below. "
              "These are new reproducible evaluations, not exact Table 11 reproductions. Cluster count is the power "
              "of two nearest to 16√n. The standard paired comparison is **three-epoch e-means versus "
              "25-epoch Metal Lloyd**; any e-means budget override is stated in its row label. "
              "Embeddings are used as stored and clustered by squared Euclidean distance.", "",
              "**Timing scope:** " + protocol_description(rows, "vibe"), ""]
    lines += paired_table(vibe, report)
    if "vibe" in plots:
        lines += [f"![VIBE fit time and test quality]({os.path.relpath(plots['vibe'], report.parent)})", ""]
    lines += ["## Ten-million-row scaling", "", "These runs use the released exact 10M splits and 65,536 clusters. "
              "Only e-means is timed here. Published Flash-KMeans25 MSE values are quality references on the "
              "same n, k, and evaluation split, but use different initialization RNG and assignment precision. "
              "**No Apple/H100 runtime ratio is calculated.**", "",
              "**Timing scope:** " + protocol_description(rows, "core10m"), ""]
    lines += independent_table(ten_m, report)
    if "core10m" in plots:
        lines += [f"![Ten-million-row e-means times]({os.path.relpath(plots['core10m'], report.parent)})", ""]
    lines += ["## Current upstream CPU reference", "",
              "This separate comparison uses the current-Faiss recipe on both implementations, shared starting "
              "centroids, float32 assignment, and last-epoch means. The CPU arm runs the actual pinned "
              "`NumpyEmeans` source; its thread count is recorded in each raw configuration. "
              "The Metal arm uses `recipe='faiss'`. This is distinct from the primary published-recipe "
              "comparison. The GPU draws donor random variates at every revival check, including checks "
              "that split nothing, to avoid a host synchronization; the CPU skips those draws when no split "
              "is needed. Weighted sampling has the same distribution, but identical seeds therefore need "
              "not produce identical revival trajectories. Centered assignment and different float32 reduction "
              "orders also affect results. These differences were not separately ablated, so the observed "
              "MSE gap is not attributed to any one of them.", ""]
    lines += [protocol_description(rows, "cpu_reference"), ""]
    lines += cpu_table(rows, report)
    extra = [r for r in unpaired if r["n"] != 10_000_000
             and stage(r) not in {"core1m_seeds", "cpu_reference"}]
    if extra:
        lines += ["## Additional unpaired e-means measurements", "", "These rows lack a matching local Lloyd25 measurement "
                  "with the same seed, code, software, precision and dataset identity. They do not support a paired speedup.", ""]
        lines += independent_table(extra, report)
    kernel_path = folder / "kernels.json"
    if kernel_path.exists():
        kernels = json.loads(kernel_path.read_text())
        matching = any(r.get("source_sha256", {}).get("src/mlxlearn/_kernels/e_means.py") == kernels.get("kernel_sha256") for r in rows)
        lines += ["## Assignment tuning", "", "Synthetic standard-normal vectors are used only for kernel tuning, "
                  "not as substitutes for the paper datasets. " + str(kernels.get("timing", "")) + ". "
                  + ("The tuning kernel hash matches the reported implementation. " if matching else "The tuning kernel hash differs from the current recorded implementation; treat these as earlier tuning evidence. ")
                  + link(kernel_path, "Raw kernel measurements", report) + ".", ""]
        lines += ["| n × k × d | Best FP32 GEMM (ms) | Fused FP32 (ms) | GEMM advantage |", "|---|---:|---:|---:|"]
        by_shape = defaultdict(list)
        for record in kernels.get("results", []):
            by_shape[(record["n"], record["k"], record["d"])].append(record)
        for shape, records in sorted(by_shape.items()):
            gemm = [r for r in records if r["method"] == "gemm" and r.get("precision") == "float32"]
            fused = [r for r in records if r["method"] == "fused" and r.get("precision") == "float32"]
            if gemm and fused:
                g, f = min(r["seconds"] for r in gemm), min(r["seconds"] for r in fused)
                lines.append(f"| {shape[0]:,} × {shape[1]:,} × {shape[2]} | {g * 1000:.3f} | {f * 1000:.3f} | {f/g:.2f}× |")
        lines += ["", "Choosing the fastest tested tile is a tuning result on this machine, not a guarantee of the "
                  "fastest possible implementation on every Apple GPU. FP16 assignment changes are separately "
                  "recorded as agreement fractions in the kernel JSON.", ""]
    lines += ["## Measurement protocol and interpretation", "",
              "Each configuration runs in a fresh subprocess. Random initial centroids are shared byte for byte "
              "between paired methods via NumPy RandomState; these starts differ from the paper's CUDA RNG. "
              "A full untimed warm-up precedes the recorded fits unless the raw configuration explicitly says otherwise. "
              "Metal is synchronized on both timer boundaries. Fit time includes validation, NumPy-to-device transfer, "
              "shuffling, optimization, finalization and final centroid copies. Data loading, initial sampling, "
              "final label assignment and scoring are excluded. The timing scope is broader than the paper's resident-GPU timer.", "",
              "Metal Lloyd uses the same assignment and statistics kernels, the same global-mean centering for "
              "assignment, and one centroid update per complete pass. It forces 25 iterations, retains empty centers, "
              "and never subsamples. This is a local full-batch control, not Flash-KMeans or Faiss-GPU. "
              "The e-means recipe and finalizer are recorded explicitly; the paper recipe is distinct from the current Faiss recipe.", "",
              "All methods are scored with the same Faiss IndexFlatL2 float32 nearest-centroid search and direct "
              "float64 residual recomputation. Test MSE is per vector, summed over dimensions. SIFT values use "
              "coordinates divided by 512, and DINO coordinates by 4096; multiply MSE by the divisor squared "
              "to recover raw-coordinate units. Training MSE in the JSON is explicitly a deterministic sample "
              "(normally 10,000 rows), not a full training objective.", "",
              "For a configuration with repeated timed fits, the reported runtime is the median of all timer "
              "samples, while **MSE and the final-centroid hash come from the last timed fit**. The runner "
              "retains only that fit's centroids; it does not report a median MSE across timing repetitions.", "",
              "Repeated fits at the same seed measure timing variation, not independent clustering quality variation. "
              "No confidence intervals or universal performance claims follow from a one-seed campaign. "
              "A fixed-budget runtime advantage should be read together with the MSE change, especially for any "
              "row where e-means has higher error.", ""]
    overlaps = {}
    for r in rows:
        n_overlap = r["dataset"].get("fingerprints", {}).get("n_eval_rows_in_train", 0)
        if n_overlap:
            overlaps[r["config"]["dataset"]] = n_overlap
    if overlaps:
        lines += ["The released core fingerprint manifest records evaluation rows also present in training: "
                  + "; ".join(f"`{name}`: {count}" for name, count in sorted(overlaps.items()))
                  + ". The exact-split campaign preserves those published data definitions; it does not claim "
                  "strict duplicate-disjointness for them. New VIBE splits remove exact train/evaluation duplicates.", ""]
    validation_path = Path(args.validation).resolve()
    lines += ["## Validation", ""]
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        checks = validation.get("checks", validation.get("observations", []))
        totals = validation.get("test_totals")
        if totals:
            lines += [f"Recorded test total: **{totals['combined']} passed** "
                      f"({totals['algorithm_and_benchmark']} algorithm/benchmark tests and "
                      f"{totals['report']} report tests).", ""]
        validated_sources = validation.get("validated_source_sha256", {})
        applies = [r for r in rows if validated_sources and all(
            r.get("source_sha256", {}).get(p) == h for p, h in validated_sources.items())]
        if validated_sources:
            lines += [f"The validation source hashes match **{len(applies)}/{len(rows)}** completed benchmark records.", ""]
        for check in checks:
            count = f" ({check['passed']} passed)" if "passed" in check else ""
            name = check.get("name", check.get("check", "Command check"))
            status = check.get("status", "passed" if check.get("exit_code") == 0 else validation.get("status", "not recorded"))
            text = f"- **{name}**: {status}{count}."
            if check.get("result"):
                text += " " + check["result"] + "."
            if check.get("command"):
                text += f" Command: `{check['command']}`."
            if check.get("details"):
                text += " " + str(check["details"])
            if check.get("scope"):
                text += " " + str(check["scope"])
            log = Path(check["log"]) if check.get("log") else None
            if log is not None:
                log = log if log.is_absolute() else REPO / log
                if log.exists():
                    text += " " + link(log, "Evidence log", report) + "."
            lines.append(text)
        lines += ["", link(validation_path, "Validation record", report) + ".", ""]
    else:
        lines += ["No validation record was supplied to this generator. Benchmark completion alone is not "
                  "a substitute for correctness tests or package validation.", ""]
    lines += ["## Coverage and limits", "",
              "Implemented paths include scheduled EMA, published and current-Faiss recipes, dead-cluster "
              "revival, last-epoch means, primary EMA and slow-EMA finalization, spherical clustering, "
              "float32 assignment and explicitly approximate float16 assignment. Default paper-recipe timing "
              "does not benchmark every optional algorithm variant.", "",
              "This report covers only completed JSON records below. Availability of a downloaded dataset "
              "does not imply its benchmark ran. The 100M, 1B and 10B scales and complete synthetic ablation "
              "grid are **not benchmarked here**. Distributed/sharded training, IVF approximate assignment, "
              "and the paper's BF16/FP8/INT8 CUDA quantization paths are **not implemented in this Metal port**. The full DINO10B "
              "corpus alone is approximately 10.28 TB; its 16,777,216 × 1,024 float32 centroid array is 64 GiB "
              "per state array. Exact single-GPU runs here do not establish the paper's distributed scalability.", "",
              "## Reproduction and raw evidence", "",
              link(HERE / "README.md", "Benchmark commands and dataset protocol", report) + "; "
              + link(HERE / "datasets.json", "dataset inventory", report) + "; "
              + link(HERE / "requirements.txt", "pinned Python packages", report) + "; "
              + link(csv_path, "complete measurement CSV", report) + ".", "",
              "Regenerate this report after measurements finish. Plotting uses Matplotlib; the isolated "
              "environment below leaves the measured Python environment unchanged:", "", "```bash",
              "uv run --isolated --no-project --with matplotlib --python .venv/bin/python \\",
              "  benchmarks/e_means/report.py --plots", "```", "",
              "Pairing requires identical source hashes, software versions, hardware, precision, dataset/train/test "
              "fingerprints and initialization. Workers snapshot provenance before measurement and reject "
              "source or environment changes during a fit. Resume keys include source, software and host identity.", "",
              "| Dataset | Method | Seed | Raw record |", "|---|---|---:|---|"]
    for r in rows:
        c = r["config"]
        lines.append(f"| {c['dataset']} | {c['method']} | {c['seed']} | {link(r['_path'], r['_path'].name, report)} |")
    lines += ["", "Sources: [paper](https://arxiv.org/abs/2611.09501), "
              "[pinned Faiss reproduction code](https://github.com/facebookresearch/faiss/tree/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means), "
              "[VIBE data](https://huggingface.co/datasets/vector-index-bench/vibe).", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    manifest = {"report": str(report.relative_to(REPO)), "report_sha256": sha256(report),
                "generator_sha256": sha256(__file__), "completed_records": len(rows),
                "matched_pairs": len(pairs), "paired_core_groups": len(core), "paired_vibe_groups": len(vibe),
                "ten_m_records": len(ten_m),
                "evidence": {str(r["_path"].relative_to(REPO)): sha256(r["_path"]) for r in rows},
                "figures": {str(p.relative_to(REPO)): sha256(p) for p in plots.values()}}
    (folder / "report-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"report": str(report), "records": len(rows), "paired_core": len(core),
                      "paired_vibe": len(vibe), "ten_m": len(ten_m), "plots": list(plots)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--output", default=str(REPO / "docs/e-means-benchmarks-2026-11-14.md"))
    parser.add_argument("--validation", default=str(HERE / "results/validation.json"))
    parser.add_argument("--figures", default=str(HERE / "figures"))
    parser.add_argument("--plots", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
