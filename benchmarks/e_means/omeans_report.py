"""Report for the o-means versus e-means comparison (see omeans_compare.py).

Speed-up at T epochs is e-means' time for a T-epoch fit divided by the time
o-means needs to reach the same held-out error. o-means is measured at a
fixed set of epoch budgets, so that time is reported twice: interpolated
between the two budgets that bracket the target (log time against error), and
as the first measured budget that reaches it, which needs no interpolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402

KIND = "omeans_comparison_v1"
GLOBAL_FIELDS = ("sources", "versions", "host", "device", "protocol_sha256")
DEFAULT_RESULTS = HERE / "results/omeans_comparison"
DEFAULT_OUTPUT = base.REPO / "docs/o-means-benchmarks-2026-11-18.md"


def load(results):
    records = [json.loads(path.read_text()) for path in sorted(Path(results).glob("*.json"))
               if path.name not in {"validation.json"}]
    return [record for record in records if record.get("kind") == KIND]


def validate(records):
    """One campaign: every record measured the same sources on the same machine."""
    if not records:
        raise ValueError("No o-means comparison records")
    for field in GLOBAL_FIELDS:
        if any(record[field] != records[0][field] for record in records):
            raise ValueError(f"mixed source/environment/device campaigns: {field} differs")
    seen = set()
    for record in records:
        key = (record["dataset"], record["precision"], record["seed"])
        if key in seen:
            raise ValueError(f"duplicate record for {key}")
        seen.add(key)
        if record["rows_limit"] is not None:
            raise ValueError(f"diagnostic subset in the campaign: {key}")
        for fit in record["fits"]:
            if not (fit["seconds"] > 0 and math.isfinite(fit["test_mse"])):
                raise ValueError(f"invalid measurement in {key}")
    return {"records": len(records), "datasets": len({r["dataset"] for r in records}),
            "seeds": sorted({r["seed"] for r in records})}


def aggregate(records):
    """Mean time, error and distances per (dataset, precision, mode, epochs) over seeds."""
    groups = defaultdict(list)
    shapes = {}
    for record in records:
        shapes[record["dataset"]] = (record["n"], record["d"], record["k"])
        for fit in record["fits"]:
            groups[(record["dataset"], record["precision"], fit["mode"], fit["epochs"])].append(fit)
    curves = defaultdict(dict)
    for (dataset, precision, mode, epochs), fits in groups.items():
        curves[(dataset, precision)].setdefault(mode, []).append({
            "epochs": epochs, "seeds": len(fits),
            "seconds": sum(f["seconds"] for f in fits) / len(fits),
            "error": sum(f["test_mse"] for f in fits) / len(fits),
            "distances": sum(f["evaluated_distances"] for f in fits) / len(fits),
            "fraction": sum(f["evaluated_fraction"] for f in fits) / len(fits),
        })
    for modes in curves.values():
        for points in modes.values():
            points.sort(key=lambda point: point["epochs"])
    return curves, shapes


def _between(low, high, error, key):
    """Log-linear interpolation of ``key`` at ``error`` between two budgets."""
    if high["error"] == low["error"]:
        return high[key]
    weight = (low["error"] - error) / (low["error"] - high["error"])
    return math.exp(math.log(low[key]) + weight * (math.log(high[key]) - math.log(low[key])))


def crossing(points, target):
    """Cost at which a curve of increasing budgets first reaches ``target`` error.

    Returns ``None`` when no budget reaches it. ``bounded`` marks a target
    already met by the smallest budget, where the true cost may be lower.
    """
    for index, point in enumerate(points):
        if point["error"] <= target:
            if index == 0:
                return {"interpolated": point["seconds"], "measured": point["seconds"],
                        "distances": point["distances"], "epochs": point["epochs"], "bounded": True}
            previous = points[index - 1]
            return {"interpolated": _between(previous, point, target, "seconds"),
                    "measured": point["seconds"],
                    "distances": _between(previous, point, target, "distances"),
                    "epochs": point["epochs"], "bounded": False}
    return None


def error_at(points, seconds):
    """Error of a curve at a time budget, interpolated in log time; None outside it."""
    if seconds < points[0]["seconds"]:
        return None
    for previous, point in zip(points, points[1:], strict=False):
        if point["seconds"] >= seconds:
            weight = ((math.log(seconds) - math.log(previous["seconds"]))
                      / (math.log(point["seconds"]) - math.log(previous["seconds"])))
            return previous["error"] + weight * (point["error"] - previous["error"])
    return points[-1]["error"]


def geometric_mean(values):
    values = [value for value in values if value is not None]
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def summarize(curves, shapes):
    rows = []
    for (dataset, precision), modes in sorted(curves.items()):
        if "metal-emeans" not in modes or "metal-omeans" not in modes:
            continue
        n, d, k = shapes[dataset]
        omeans = modes["metal-omeans"]
        dense = {point["epochs"]: point for point in modes.get("metal-omeans-dense", [])}
        same_budget = {point["epochs"]: point for point in omeans}
        for reference in modes["metal-emeans"]:
            epochs = reference["epochs"]
            reached = crossing(omeans, reference["error"])
            at_time = error_at(omeans, reference["seconds"])
            equal = same_budget.get(epochs)
            rows.append({
                "dataset": dataset, "precision": precision, "n": n, "d": d, "k": k,
                "epochs": epochs, "seeds": reference["seeds"],
                "emeans_seconds": reference["seconds"], "emeans_error": reference["error"],
                "omeans_seconds_interpolated": reached and reached["interpolated"],
                "omeans_seconds_measured": reached and reached["measured"],
                "omeans_epochs_measured": reached and reached["epochs"],
                "bounded": bool(reached and reached["bounded"]),
                "speedup": reached and reference["seconds"] / reached["interpolated"],
                "speedup_measured": reached and reference["seconds"] / reached["measured"],
                "distance_ratio": reached and reached["distances"] / (n * k * epochs),
                "error_at_equal_time": at_time and at_time / reference["error"] - 1,
                "equal_epochs_time_ratio": equal and reference["seconds"] / equal["seconds"],
                "equal_epochs_error": equal and equal["error"] / reference["error"] - 1,
                "equal_epochs_fraction": equal and equal["fraction"],
                "dense_equal_epochs_error": (dense[epochs]["error"] / reference["error"] - 1
                                             if epochs in dense else None),
            })
    return rows


def _number(value, pattern="{:.2f}", missing="n/a"):
    return missing if value is None else pattern.format(value)


def markdown(rows, validation, device, versions):
    lines = [
        "# o-means against e-means on Apple silicon",
        "",
        f"Generated from {validation['records']} records over {validation['datasets']} datasets, "
        f"seeds {validation['seeds']}, on {device['device_name']} "
        f"(MLX {versions.get('mlx')}, Python {versions.get('python')}). "
        "Both methods use the Faiss e-means recipe, the same batch law, seed and starting "
        "centroids, with the training matrix resident on the GPU. Times are medians of "
        "synchronized `train_device` calls after a full warm-up fit; errors are held-out MSE "
        "with exact float32 assignment and a float64 residual.",
        "",
    ]
    for precision in sorted({row["precision"] for row in rows}):
        selected = [row for row in rows if row["precision"] == precision]
        budgets = sorted({row["epochs"] for row in selected})
        lines += [f"## {precision}", "", "### Time to e-means' error", "",
                  "Speed-up is e-means' fit time divided by the time o-means needs to reach the same "
                  "held-out error. `≥` marks a target the smallest o-means budget already met. "
                  "The measured column uses the first o-means budget that reaches the target, "
                  "without interpolation. Distances are those o-means evaluated to reach the "
                  "target, as a fraction of e-means' n × k × T.", "",
                  "| Dataset | n | d | k | " + " | ".join(f"Speed-up at {b:g}" for b in budgets)
                  + " | " + " | ".join(f"Measured at {b:g}" for b in budgets)
                  + " | Distances at " + " / ".join(f"{b:g}" for b in budgets) + " |",
                  "|---|---:|---:|---:|" + "---:|" * (2 * len(budgets) + 1)]
        by_dataset = defaultdict(dict)
        for row in selected:
            by_dataset[row["dataset"]][row["epochs"]] = row
        for dataset, cells in by_dataset.items():
            first = next(iter(cells.values()))
            speed = [("≥ " if cells[b]["bounded"] else "") + _number(cells[b]["speedup"]) if b in cells
                     else "n/a" for b in budgets]
            measured = [_number(cells[b]["speedup_measured"]) if b in cells else "n/a" for b in budgets]
            distances = " / ".join(_number(cells[b]["distance_ratio"]) if b in cells else "n/a"
                                   for b in budgets)
            lines.append(f"| {dataset} | {first['n']:,} | {first['d']} | {first['k']:,} | "
                         + " | ".join(speed) + " | " + " | ".join(measured) + f" | {distances} |")
        means = [_number(geometric_mean([r["speedup"] for r in selected if r["epochs"] == b])) for b in budgets]
        measured_means = [_number(geometric_mean([r["speedup_measured"] for r in selected if r["epochs"] == b]))
                          for b in budgets]
        ratio = _number(geometric_mean([r["distance_ratio"] for r in selected]))
        lines.append("| **Geometric mean** | | | | " + " | ".join(means) + " | "
                     + " | ".join(measured_means) + f" | {ratio} |")
        overall = geometric_mean([row["speedup"] for row in selected])
        overall_measured = geometric_mean([row["speedup_measured"] for row in selected])
        unreached = [f"{row['dataset']} at {row['epochs']:g}" for row in selected if row["speedup"] is None]
        lines += ["", f"Geometric mean over all datasets and targets: {_number(overall)}× interpolated, "
                  f"{_number(overall_measured)}× measured-only."
                  + (f" Not reached within the measured budgets: {', '.join(unreached)}." if unreached else ""),
                  "", "### Same epoch budget", "",
                  "o-means and e-means at the same number of epochs: how much faster o-means' fit is, "
                  "its held-out error relative to e-means' (negative is better), the share of "
                  "distances it evaluated, and the error of o-means with a full search, which "
                  "isolates the closing mean.", "",
                  "| Dataset | " + " | ".join(f"Faster at {b:g}" for b in budgets) + " | "
                  + " | ".join(f"Error at {b:g}" for b in budgets)
                  + " | Distances | Full-search error at " + " / ".join(f"{b:g}" for b in budgets) + " |",
                  "|---|" + "---:|" * (2 * len(budgets) + 2)]
        for dataset, cells in by_dataset.items():
            faster = [_number(cells[b]["equal_epochs_time_ratio"], "{:.2f}×") if b in cells else "n/a"
                      for b in budgets]
            error = [_number(cells[b]["equal_epochs_error"], "{:+.2%}") if b in cells else "n/a"
                     for b in budgets]
            fractions = [cells[b]["equal_epochs_fraction"] for b in budgets
                         if b in cells and cells[b]["equal_epochs_fraction"] is not None]
            dense = " / ".join(_number(cells[b]["dense_equal_epochs_error"], "{:+.2%}") if b in cells
                               else "n/a" for b in budgets)
            lines.append(f"| {dataset} | " + " | ".join(faster) + " | " + " | ".join(error)
                         + f" | {_number(sum(fractions) / len(fractions) if fractions else None, '{:.3f}')} | {dense} |")
        lines += ["", "### Error at e-means' time", "",
                  "o-means' held-out error relative to e-means', when o-means is given the time "
                  "e-means needed. Negative is better.", "",
                  "| Dataset | " + " | ".join(f"At e-means' {b:g}-epoch time" for b in budgets) + " |",
                  "|---|" + "---:|" * len(budgets)]
        for dataset, cells in by_dataset.items():
            lines.append(f"| {dataset} | " + " | ".join(
                _number(cells[b]["error_at_equal_time"], "{:+.2%}") if b in cells else "n/a"
                for b in budgets) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append", type=Path, help="markdown file with notes to place after the tables")
    args = parser.parse_args()
    records = load(args.results)
    validation = validate(records)
    curves, shapes = aggregate(records)
    rows = summarize(curves, shapes)
    base.atomic_json(args.results / "validation.json", validation)
    with open(args.results / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = markdown(rows, validation, records[0]["device"], records[0]["versions"])
    if args.append:
        text += "\n" + args.append.read_text()
    args.output.write_text(text.rstrip("\n") + "\n")
    print(f"{validation['records']} records -> {args.output}")


if __name__ == "__main__":
    main()
