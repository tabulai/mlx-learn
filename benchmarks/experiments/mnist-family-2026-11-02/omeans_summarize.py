"""Markdown tables from the ``omeans.py`` results.

Usage: ``python omeans_summarize.py [results-dir]`` (default: the committed ``results/``).
Times are the median over seeds of each seed's median over rounds; errors are means
over seeds.
"""

import statistics
import sys
from pathlib import Path

from data import RESULTS, SPLITS, read_json

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks/e_means"))
from omeans_report import crossing, geometric_mean  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
NAMES = {"MNIST": "MNIST", "Fashion_MNIST": "Fashion-MNIST", "Kannada_MNIST": "Kannada-MNIST"}
BUDGETS = (1, 2, 3)


def load():
    runs = {}
    for path in sorted(DIR.glob("*_omeans_k*.json.gz")):
        run = read_json(path)
        runs[(run["meta"]["dataset"], run["meta"]["k"])] = run
    return runs


def aggregate(run):
    """(kind, epochs) -> median seconds, mean error and mean share of distances over seeds."""
    groups = {}
    for records in run["seeds"].values():
        for record in records:
            groups.setdefault((record["kind"], record["epochs"]), []).append(record)
    out = {}
    for key, records in groups.items():
        fractions = [r["evaluated_fraction"] for r in records if "evaluated_fraction" in r]
        out[key] = {
            "seconds": statistics.median(r["median_seconds"] for r in records),
            "error": statistics.fmean(r["held_out_mse"] for r in records),
            "fraction": statistics.fmean(fractions) if fractions else None,
            "search": sorted({r["search"] for r in records if r["search"]}),
            "fallbacks": sum(r["search_fallback_step"] is not None for r in records),
            "seeds": len(records),
            "pressure": max((r["thermal_pressure"] or 0) for r in records),
        }
    return out


def ordered(runs):
    return [(name, k, runs[(name, k)]) for name in SPLITS for k in sorted({key[1] for key in runs})
            if (name, k) in runs]


def training_table(runs):
    print("\n### Training from resident inputs\n")
    print("| Dataset | Centroids | `auto` uses | Compared | E-means, 1 / 2 / 3 epochs | O-means, 1 / 2 / 3 epochs "
          "| Faster | Error against e-means | Full-search control |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    faster_all = []
    for name, k, run in ordered(runs):
        a = aggregate(run)
        emeans = [a[("emeans", t)] for t in BUDGETS]
        auto = [a[("omeans-auto", t)] for t in BUDGETS]
        faster = [e["seconds"] / o["seconds"] for e, o in zip(emeans, auto, strict=True)]
        control = [a[("emeans", t)]["seconds"] / a[("omeans-dense", t)]["seconds"] for t in (1, 2)]
        searches = ["/".join(o["search"]).replace("_", "-") for o in auto]
        faster_all += [f for f, s in zip(faster, searches, strict=True) if s == "two-level"]
        search = searches[0] if len(set(searches)) == 1 else " / ".join(searches)
        if auto[1]["fallbacks"]:
            search += f", full search in {auto[1]['fallbacks']} of {auto[1]['seeds']} seeds"
        print(f"| {NAMES[name]} | {k:,} | {search} | {auto[1]['fraction']:.0%} | "
              + " / ".join(f"{1e3 * e['seconds']:.0f}" for e in emeans) + " ms | "
              + " / ".join(f"{1e3 * o['seconds']:.0f}" for o in auto) + " ms | "
              + " / ".join(f"{f:.2f}×" for f in faster) + " | "
              + " / ".join(f"{o['error'] / e['error'] - 1:+.2%}" for e, o in zip(emeans, auto, strict=True))
              + f" | {geometric_mean(control):.2f}× |")
    if faster_all:
        print(f"\nGeometric mean where `auto` uses the two-level search: {geometric_mean(faster_all):.2f}×.")


def forced_table(runs):
    print("\n### Two-level search forced, and time to e-means' error\n")
    print("| Dataset | Centroids | Compared | Faster at 1 / 2 epochs | Time to e-means' error at 1 / 2 / 3 epochs |")
    print("|---|---:|---:|---:|---:|")
    for name, k, run in ordered(runs):
        a = aggregate(run)
        n = run["meta"]["train_shape"][0]
        points = sorted(({"epochs": t, "seconds": v["seconds"], "error": v["error"],
                          "distances": v["fraction"] * n * k * t}
                         for (kind, t), v in a.items() if kind == "omeans"), key=lambda p: p["epochs"])
        same = [a[("emeans", t)]["seconds"] / a[("omeans", t)]["seconds"] for t in (1, 2)]
        reached = []
        for t in BUDGETS:
            hit = crossing(points, a[("emeans", t)]["error"])
            reached.append("not reached" if hit is None else
                           ("≥ " if hit["bounded"] else "") + f"{a[('emeans', t)]['seconds'] / hit['interpolated']:.2f}×")
        print(f"| {NAMES[name]} | {k:,} | {a[('omeans', 2)]['fraction']:.0%} | "
              + " / ".join(f"{s:.2f}×" for s in same) + " | " + " / ".join(reached) + " |")


def fit_table(runs):
    print("\n### Public `fit`, and reference fits\n")
    print("`OMeans` and `EMeans` run three epochs and include the final label pass. The others are single "
          "default-settings fits with the first seed; errors are relative to scikit-learn's `KMeans`.\n")
    print("| Dataset | Centroids | `OMeans` | `EMeans` | mlxlearn `KMeans`, random init | mlxlearn `KMeans` "
          "| scikit-learn `MiniBatchKMeans` | scikit-learn `KMeans` |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, k, run in ordered(runs):
        a, references = aggregate(run), run["references"]
        base = references.get("sklearn-kmeans", {}).get("held_out_mse")

        def cell(seconds, error, base=base):
            time = f"{1e3 * seconds:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"
            return time + (f" ({error / base - 1:+.2%})" if base else "")

        cells = [cell(a[("fit-omeans", 3)]["seconds"], a[("fit-omeans", 3)]["error"]),
                 cell(a[("fit-emeans", 3)]["seconds"], a[("fit-emeans", 3)]["error"])]
        for key in ("mlx-kmeans-random", "mlx-kmeans", "sklearn-minibatch"):
            r = references.get(key)
            cells.append(cell(r["median_seconds"], r["held_out_mse"]) if r else "n/a")
        r = references.get("sklearn-kmeans")
        cells.append(f"{r['median_seconds']:.1f} s ({r['n_iter']} iterations)" if r else "n/a")
        print(f"| {NAMES[name]} | {k:,} | " + " | ".join(cells) + " |")


def main():
    runs = load()
    if not runs:
        raise SystemExit(f"No omeans results in {DIR}")
    meta = next(iter(runs.values()))["meta"]
    pressure = max(v["pressure"] for run in runs.values() for v in aggregate(run).values())
    print(f"{len(runs)} runs on {meta['device']['device_name']}, MLX {meta['mlx']}, scikit-learn "
          f"{meta['scikit_learn']}, seeds {meta['seeds']}, {meta['rounds']} rounds after {meta['warmups']} "
          f"warm-ups; worst thermal pressure level seen around a timed call: {pressure}.")
    training_table(runs)
    forced_table(runs)
    fit_table(runs)


if __name__ == "__main__":
    main()
