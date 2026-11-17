"""Render static scientific figures only after all 20 measurements validate.

Example without installing dependencies into the project environment:
  uv run --isolated --no-project --with matplotlib python benchmarks/e_means/kmeans_plot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kmeans_report as report  # noqa: E402

COLORS = {"metal-kmeans": "#0072B2", "sklearn-kmeans": "#666666",
          "metal-emeans": "#009E73", "faiss-emeans": "#E69F00"}
MARKERS = {"metal-kmeans": "o", "sklearn-kmeans": "s", "metal-emeans": "D", "faiss-emeans": "^"}


def validated_rows(folder, protocol_path, archive_path):
    protocol = json.loads(Path(protocol_path).read_text())
    raw, parse_errors = report.load_rows(folder)
    archive_metadata, archive = report.archive_evidence(archive_path)
    groups, validation = report.validate(raw, protocol, report.common.sha256(protocol_path), archive_metadata)
    if parse_errors or not validation["complete"] or validation["expected_records"] != 20:
        raise ValueError("Plots require all 20 compatible full-dataset records, without missing, duplicate or rejected evidence")
    rows, _ = report.summarize(groups)
    return rows, raw, protocol, archive


def render(folder, output, protocol_path, archive_path):
    rows, raw, protocol, archive = validated_rows(folder, protocol_path, archive_path)
    # Importing/initializing the plotting backend is delayed until evidence passes.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    datasets = [entry["dataset"] for entry in protocol["configurations"]]
    by_key = {(row["dataset"], row["mode"]): row for row in rows}
    host = raw[0]["hardware"]
    threads = raw[0]["config"]["threads"]
    host_name = host.get("cpu", "recorded Apple Silicon host")
    calls = sorted({row["timed_calls"] for row in rows})
    calls_text = ("one timed fit per method" if calls == [1]
                  else "per-method median of " + "/".join(map(str, calls)) + " timed fits")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
                         "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 1, figsize=(15, 12.5), sharex=True)
    width = .18
    centers = list(range(len(datasets)))
    handles = []
    for index, mode in enumerate(report.MODES):
        positions = [center + (index - 1.5) * width for center in centers]
        records = [by_key[(name, mode)] for name in datasets]
        axes[0].bar(positions, [row["seconds"] for row in records], width=width * .92,
                    color=COLORS[mode], label=report.NAMES[mode], zorder=3)
        axes[1].bar(positions, [row["peak_physical_footprint_bytes"] / 2**30 for row in records],
                    width=width * .92, color=COLORS[mode], zorder=3)
        relative_error = []
        for name, row in zip(datasets, records, strict=True):
            baseline = by_key[(name, "sklearn-kmeans")]["test_mse"]
            if baseline <= 0:
                raise ValueError("Relative test-MSE panel requires a positive scikit-learn baseline on every dataset")
            relative_error.append(100 * (row["test_mse"] / baseline - 1))
        axes[2].scatter(positions, relative_error, s=54, marker=MARKERS[mode], color=COLORS[mode],
                        edgecolor="white", linewidth=.5, zorder=4)
        handles.append(Line2D([], [], color=COLORS[mode], marker=MARKERS[mode], linestyle="none",
                              markersize=8, label=report.NAMES[mode]))

    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel("Runtime (s; log scale)")
    axes[0].set_title("A. Training plus final labels and inertia — lower is faster", loc="left")
    axes[1].set_ylabel("Process footprint (GiB)")
    axes[1].set_title("B. Sampled peak during the timed fit — includes resident input and allocator cache", loc="left")
    axes[2].axhline(0, color="#555555", linewidth=1, linestyle="--", zorder=2)
    axes[2].set_ylabel("Test MSE change (%)")
    axes[2].set_title("C. Test error relative to scikit-learn Lloyd on each dataset — negative is lower error", loc="left")
    labels = [f"{report.common.label(entry['dataset'])}\nd={entry['d']}; budget={entry['epochs']} passes"
              for entry in protocol["configurations"]]
    axes[2].set_xticks(centers, labels)
    axes[2].set_xlabel("10,000,000 training rows and 65,536 centroids per dataset")
    for axis in axes:
        axis.grid(axis="y", color="#e3e6e8", linewidth=.7, zorder=0)
        axis.set_axisbelow(True)
        axis.set_xlim(-.6, len(datasets) - .4)
    figure.suptitle(f"KMeans and e-means on {host_name}\n"
                   f"FP32 • {threads} OpenMP threads on CPU • unrestricted Accelerate • seed 10",
                   fontsize=15, y=.985)
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .925))
    figure.text(.07, .018, f"Full untimed warmup; {calls_text}. No error bars: one seed. "
                "Test quality uses shared FP32 assignment + FP64 residuals.\n"
                "Preparation/export/scoring excluded from fit time and fit-memory peaks; source report separately records lifetime/RSS/MLX metrics.",
                fontsize=9, color="#444444")
    figure.subplots_adjust(top=.865, bottom=.12, left=.075, right=.985, hspace=.29)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        figure.savefig(path, facecolor="white")
    plt.close(figure)
    manifest = {"kind": "kmeans_static_figure_evidence", "plot_generator_sha256": report.common.sha256(__file__),
                "report_validator_sha256": report.common.sha256(HERE / "kmeans_report.py"),
                "protocol_sha256": report.common.sha256(protocol_path), "source_archive": archive,
                "raw_evidence": [{"path": str(row["_path"]), "sha256": report.common.sha256(row["_path"])} for row in raw],
                "panels": ["runtime_seconds_log_scale", "fit_window_process_physical_footprint_gib",
                           "test_mse_percent_change_relative_to_sklearn_per_dataset"],
                "outputs": [{"path": str(path), "sha256": report.common.sha256(path)} for path in paths]}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--results", type=Path, default=HERE / "results/kmeans_comparison")
    parser.add_argument("--output", type=Path, default=HERE / "kmeans_figures/comparison.png")
    parser.add_argument("--protocol", type=Path, default=HERE / "kmeans_protocol.json")
    parser.add_argument("--archive-metadata", type=Path, default=HERE / "reference/kmeans-comparison-source.json")
    args = parser.parse_args()
    for path in render(args.input, args.output, args.protocol, args.archive_metadata):
        print(path)


if __name__ == "__main__":
    main()
