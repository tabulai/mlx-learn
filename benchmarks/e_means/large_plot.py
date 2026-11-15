"""Plot validated 10M comparisons after all timed benchmark workers finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = {
    "deep1b-10M": "DEEP", "spacev-10M": "SPACEV", "fb-ssnpp-10M": "SimSearchNet++",
    "dino10b-10M-div4096": "DINO", "sift1b-10M-div512": "SIFT",
}
COLORS = {"metal_fp32": "#176b9a", "faiss_fp32": "#bd6539", "metal_bf16": "#2f8966"}


def main():
    validation = json.loads((HERE / "results/large_comparison/validation.json").read_text())
    if not validation.get("complete"):
        raise ValueError("Plotting requires complete validated benchmark evidence")
    with (HERE / "results/large_comparison/summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 5 and {row["dataset"] for row in rows} == set(NAMES)
    rows = sorted(rows, key=lambda row: list(NAMES).index(row["dataset"]))
    labels = [f"{NAMES[row['dataset']]}  ({float(row['epochs']):g} epochs)" for row in rows]
    destination = HERE / "large_figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    def values(field):
        return np.array([float(row[field]) for row in rows])

    def figure(modes, titles, delta_field, output, heading, subtitle):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1, .9]})
        y = np.arange(len(rows))
        for mode, title, offset in zip(modes, titles, [-.18, .18], strict=True):
            seconds = values(f"{mode}_seconds")
            axes[0].scatter(seconds, y + offset, s=42, label=title, color=COLORS[mode], zorder=3)
            for number, pos in zip(seconds, y + offset, strict=True):
                axes[0].annotate(f"{number:.1f}", (number, pos), xytext=(6, -3),
                                 textcoords="offset points", fontsize=8)
            memory = values(f"{mode}_peak_physical_footprint_bytes") / 2**30
            bars = axes[1].barh(y + offset, memory, height=.32, label=title, color=COLORS[mode])
            axes[1].bar_label(bars, labels=[f"{number:.1f}" for number in memory], padding=3, fontsize=8)
        axes[0].set_xscale("log")
        lower = min(values(f"{mode}_seconds").min() for mode in modes) / 1.4
        upper = max(values(f"{mode}_seconds").max() for mode in modes) * 1.5
        axes[0].set_xlim(lower, upper)
        axes[0].xaxis.set_major_locator(FixedLocator([v for v in (20, 50, 100, 200, 500, 1000, 2000)
                                                     if lower <= v <= upper]))
        axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axes[0].xaxis.set_minor_formatter(NullFormatter())
        axes[0].set_xlabel("Timed fit, seconds (log scale)")
        axes[1].set_xlabel("Fit-window process peak, GiB")
        axes[1].set_xlim(right=max(values(f"{mode}_peak_physical_footprint_bytes").max()
                                   for mode in modes) / 2**30 * 1.25)
        deltas = values(delta_field)
        axes[2].axvline(0, color="#777777", linewidth=1)
        axes[2].scatter(deltas, y, s=45, color=COLORS[modes[0]], zorder=3)
        span = max(np.max(np.abs(deltas)), .01) * 1.5
        axes[2].set_xlim(-span, span)
        for delta, pos in zip(deltas, y, strict=True):
            axes[2].annotate(f"{delta:+.3f}%", (delta, pos), xytext=(5, 7),
                             textcoords="offset points", fontsize=8)
        axes[2].set_xlabel("Test MSE change, %\nPositive means higher loss")
        for ax in axes:
            ax.set_yticks(y, labels if ax is axes[0] else [])
            ax.set_ylim(len(rows) - .5, -.5)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#eeeeee")
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=9)
        fig.suptitle(heading, x=.02, ha="left", fontsize=16)
        fig.text(.02, .89, subtitle, fontsize=10, color="#444444")
        fig.text(.02, .02, "Apple M4 Max · 10,000,000 vectors and 65,536 clusters each · seed 10 · one warmed timed fit per mode\n"
                 "Same initialization, epoch budgets, quality batching and closing means. Sampled process memory includes resident input.",
                 fontsize=9, color="#444444")
        fig.subplots_adjust(left=.18, right=.98, top=.78, bottom=.19, wspace=.25)
        fig.savefig(destination / output, dpi=180, facecolor="white")
        plt.close(fig)

    figure(["metal_fp32", "faiss_fp32"], ["Metal FP32", "Original Faiss CPU FP32"],
           "fp32_test_mse_change_pct", "fp32-comparison.png", "FP32 comparison",
           "Matched precision. MSE change is Metal relative to Faiss; lower runtime and memory are better.")
    figure(["metal_bf16", "metal_fp32"], ["Metal BF16 (supplement)", "Metal FP32"],
           "bf16_vs_fp32_test_mse_change_pct", "bf16-supplement.png", "BF16 supplement",
           "Precision changes in this comparison. MSE change is BF16 relative to FP32; lower runtime and memory are better.")
    print(destination)


if __name__ == "__main__":
    main()
