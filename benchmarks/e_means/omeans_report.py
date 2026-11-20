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
import statistics
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


def validate(records, partial=False):
    """One campaign: every record measured the same sources on the same machine.

    Every (dataset, precision) must also have the same seeds, unless ``partial``
    allows a draft from a campaign that is still running. A fit timed under
    thermal pressure is rejected: throttling slows the full search more than
    the two-level search and would overstate o-means. Records of
    ``omeans_interleaved`` are exempt, because their configurations take turns
    and share the machine's state; the levels they saw are counted instead.
    """
    if not records:
        raise ValueError("No o-means comparison records")
    for field in GLOBAL_FIELDS:
        if any(record[field] != records[0][field] for record in records):
            raise ValueError(f"mixed source/environment/device campaigns: {field} differs")
    timing = records[0].get("timing", "sequential")
    if any(record.get("timing", "sequential") != timing for record in records):
        raise ValueError("mixed source/environment/device campaigns: timing differs")
    levels = defaultdict(int)
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
            if "thermal_pressure" not in fit:
                raise ValueError(f"no thermal pressure recorded in {key}")
            if fit["thermal_pressure"] and timing != "interleaved":
                raise ValueError(f"measured under thermal pressure: {key}, {fit['mode']} "
                                 f"at {fit['epochs']:g} epochs, level {fit['thermal_pressure']}")
            for level in fit.get("thermal_pressure_all", [fit["thermal_pressure"]]):
                levels[level] += 1
    seeds = sorted({r["seed"] for r in records})
    groups = defaultdict(set)
    for record in records:
        groups[(record["dataset"], record["precision"])].add(record["seed"])
    missing = sorted(key for key, found in groups.items() if sorted(found) != seeds)
    if missing and not partial:
        raise ValueError(f"incomplete campaign: {missing} lack some of seeds {seeds}")
    return {"records": len(records), "datasets": len({r["dataset"] for r in records}), "seeds": seeds,
            "timing": timing,
            "thermal_calls": {str(level): count for level, count in sorted(
                levels.items(), key=lambda item: (item[0] is None, item[0] or 0))}}


def aggregate(records):
    """Per (dataset, precision, mode, epochs) over seeds: median time, mean error and distances."""
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
            "seconds": statistics.median(f["seconds"] for f in fits),
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


def _extreme(rows, key, pick):
    row = pick((row for row in rows if row[key] is not None), key=lambda row: row[key])
    return f"{row[key]:.2f}× ({row['dataset']}, {row['epochs']:g} epochs)"


def _summary(rows):
    """One row per precision: the means the per-dataset tables below support."""
    lines = ["## Summary", "",
             "Geometric means over datasets and e-means budgets. \"Same epochs\" divides e-means' "
             "fit time by o-means' at the budgets both were measured at.", "",
             "| Precision | Datasets | Time to e-means' error, interpolated | First measured budget "
             "| Slowest | Fastest | Same epochs | Distances at same epochs, median |",
             "|---|---:|---:|---:|---|---|---:|---:|"]
    for precision in sorted({row["precision"] for row in rows}):
        selected = [row for row in rows if row["precision"] == precision]
        fractions = [row["equal_epochs_fraction"] for row in selected
                     if row["equal_epochs_fraction"] is not None]
        lines.append(
            f"| {precision} | {len({row['dataset'] for row in selected})} | "
            f"{_number(geometric_mean([row['speedup'] for row in selected]))}× | "
            f"{_number(geometric_mean([row['speedup_measured'] for row in selected]))}× | "
            f"{_extreme(selected, 'speedup', min)} | {_extreme(selected, 'speedup', max)} | "
            f"{_number(geometric_mean([row['equal_epochs_time_ratio'] for row in selected]))}× | "
            f"{_number(statistics.median(fractions) if fractions else None, '{:.3f}')} |")
    return lines + [""]


THERMAL_LEVELS = {"0": "nominal", "1": "moderate", "2": "heavy", "3": "trapping", "4": "sleeping",
                  "None": "unread"}


def _thermal_sentence(validation):
    """How the campaign was timed and which thermal pressure its calls saw."""
    calls = validation.get("thermal_calls", {})
    total = sum(calls.values())
    if validation.get("timing") != "interleaved":
        return "Every timed call started and ended at nominal macOS thermal pressure."
    shares = ", ".join(f"{THERMAL_LEVELS.get(level, level)} {count / total:.1%}"
                       for level, count in calls.items())
    return ("The timed calls were interleaved: within each dataset, precision and seed, all "
            "configurations took turns in freshly drawn orders, so that they share the machine's "
            "thermal state. Each call waited for nominal macOS thermal pressure, for a bounded "
            f"time; the worst level seen before or after a call was: {shares}.")


TITLE = "o-means against e-means on Apple silicon"


def markdown(rows, validation, device, versions, title=TITLE):
    lines = [
        f"# {title}",
        "",
        f"Generated from {validation['records']} records over {validation['datasets']} datasets, "
        f"seeds {validation['seeds']}, on {device['device_name']} "
        f"(MLX {versions.get('mlx')}, Python {versions.get('python')}). "
        "Both methods use the Faiss e-means recipe, the same batch law, seed and starting "
        "centroids, with the training matrix resident on the GPU. Times are medians of "
        "synchronized `train_device` calls after a full warm-up fit, then the median over "
        "seeds; errors are held-out MSE with exact float32 assignment and a float64 residual, "
        "averaged over seeds. " + _thermal_sentence(validation),
        "",
    ]
    lines += _summary(rows)
    for precision in sorted({row["precision"] for row in rows}):
        selected = [row for row in rows if row["precision"] == precision]
        budgets = sorted({row["epochs"] for row in selected})
        shared = [b for b in budgets if any(row["equal_epochs_time_ratio"] is not None
                                            for row in selected if row["epochs"] == b)]
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
                  "Only the budgets both methods were measured at. "
                  "o-means and e-means at the same number of epochs: how much faster o-means' fit is, "
                  "its held-out error relative to e-means' (negative is better), the share of "
                  "distances it evaluated, and the error of o-means with a full search, which "
                  "isolates the closing mean.", "",
                  "| Dataset | " + " | ".join(f"Faster at {b:g}" for b in shared) + " | "
                  + " | ".join(f"Error at {b:g}" for b in shared)
                  + " | Distances | Full-search error at " + " / ".join(f"{b:g}" for b in shared) + " |",
                  "|---|" + "---:|" * (2 * len(shared) + 2)]
        for dataset, cells in by_dataset.items():
            faster = [_number(cells[b]["equal_epochs_time_ratio"], "{:.2f}×") if b in cells else "n/a"
                      for b in shared]
            error = [_number(cells[b]["equal_epochs_error"], "{:+.2%}") if b in cells else "n/a"
                     for b in shared]
            fractions = [cells[b]["equal_epochs_fraction"] for b in shared
                         if b in cells and cells[b]["equal_epochs_fraction"] is not None]
            dense = " / ".join(_number(cells[b]["dense_equal_epochs_error"], "{:+.2%}") if b in cells
                               else "n/a" for b in shared)
            lines.append(f"| {dataset} | " + " | ".join(faster) + " | " + " | ".join(error)
                         + f" | {_number(sum(fractions) / len(fractions) if fractions else None, '{:.3f}')} | {dense} |")
        lines.append("| **Geometric mean** | " + " | ".join(
            _number(geometric_mean([r["equal_epochs_time_ratio"] for r in selected if r["epochs"] == b]),
                    "{:.2f}×") for b in shared) + " |" + " |" * (len(shared) + 2))
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
    parser.add_argument("--partial", action="store_true",
                        help="draft from a campaign in which some seeds are still missing")
    parser.add_argument("--title", default=TITLE)
    args = parser.parse_args()
    records = load(args.results)
    validation = validate(records, partial=args.partial)
    curves, shapes = aggregate(records)
    rows = summarize(curves, shapes)
    base.atomic_json(args.results / "validation.json", validation)
    with open(args.results / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = markdown(rows, validation, records[0]["device"], records[0]["versions"], args.title)
    if args.append:
        text += "\n" + args.append.read_text()
    args.output.write_text(text.rstrip("\n") + "\n")
    print(f"{validation['records']} records -> {args.output}")


if __name__ == "__main__":
    main()
