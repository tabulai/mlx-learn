"""Print the Markdown tables of the README from ``results/``."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

RESULTS = Path(__file__).parent / "results"


def grouped(path):
    groups = defaultdict(list)
    for record in json.loads((RESULTS / path).read_text())["records"]:
        groups[record["variant"], record["centers"]].append(record)
    return groups


def cell(records):
    seconds = [r["seconds"] for r in records]
    if len(seconds) == 1:
        return f"{seconds[0]:.2f} s"
    return f"{median(seconds):.2f} s ({min(seconds):.2f}-{max(seconds):.2f})"


def seeding():
    groups = grouped("seeding.json")
    centers = sorted({k for _, k in groups})
    print("| Centers | scikit-learn helper | Metal screen | Speedup | Same rows |")
    print("|---:|---:|---:|---:|---|")
    for k in centers:
        old, new = groups["sklearn", k], groups["mlxlearn", k]
        ratio = median(r["seconds"] for r in old) / median(r["seconds"] for r in new)
        same = all(r["same_indices"] for r in new)
        print(f"| {k:,} | {cell(old)} | {cell(new)} | {ratio:.0f}x | {'yes' if same else 'NO'} |")


def fits():
    groups = grouped("fits.json")
    centers = sorted({k for _, k in groups})
    print("| Centers | `KMeans(k)` before | `KMeans(k)` | of which Lloyd | `KMeans(k, init=\"random\")` |")
    print("|---:|---:|---:|---:|---:|")
    for k in centers:
        lloyd = median(r["lloyd_seconds"] for r in groups["default", k])
        print(f"| {k:,} | {cell(groups['reference', k])} | {cell(groups['default', k])} | {lloyd:.2f} s "
              f"| {cell(groups['random', k])} |")
    print()
    print("| Centers | Lloyd iterations, k-means++ | Inertia, k-means++ | Inertia, best of ten random |")
    print("|---:|---:|---:|---:|")
    for k in centers:
        new, rand, old = groups["default", k][0], groups["random", k][0], groups["reference", k][0]
        assert new["inertia"] == old["inertia"] and new["n_iter"] == old["n_iter"]
        print(f"| {k:,} | {new['n_iter']} | {new['inertia']:,.0f} | {rand['inertia']:,.0f} |")


def levels():
    records = [r for name in ("seeding.json", "fits.json")
               for r in json.loads((RESULTS / name).read_text())["records"]]
    print(f"Highest thermal pressure level at the start of a timed call: {max(r['thermal'] for r in records)}; "
          f"calls that waited for it: {sum(r['waited'] > 1 for r in records)} of {len(records)}, "
          f"{sum(r['waited'] for r in records):.0f} s in total")


if __name__ == "__main__":
    seeding()
    print()
    fits()
    print()
    levels()
