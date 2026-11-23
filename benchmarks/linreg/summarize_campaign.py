"""Condense a bench_linreg.py campaign: per data kind and option, and every cell below parity."""
import json
import math
import sys
from collections import defaultdict


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def label(c):
    c = dict(c)
    s = f"{c.pop('data')} {c.pop('n')}x{c.pop('d')}"
    return s + "".join(f" {k}={v}" for k, v in sorted(c.items()))


def main(path, pr4=None):
    rows = [json.loads(line) for line in open(path) if '"cfg"' in line]
    old = {}
    if pr4:
        for line in open(pr4):
            r = json.loads(line)
            if "cfg" in r:
                old[tuple(sorted(r["cfg"].items()))] = r.get("speedup", {}).get("mlx")
    groups = defaultdict(list)
    for r in rows:
        c = r["cfg"]
        opt = [k for k in ("sw", "intercept", "k", "dtype") if k in c]
        key = ("option: " + ",".join(f"{k}={c[k]}" for k in opt)) if opt else (
            "wide" if c["d"] > c["n"] else ("tiny" if c["n"] * c["d"] < 50_000 else f"kind: {c['data']}"))
        groups[key].append(r)
    print("| group | cells | rewrite geomean | rewrite min | patched geomean | patched min | PR 4 geomean |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        rs = groups[key]
        new = [r["speedup"].get("mlx") for r in rs]
        pat = [r["speedup"].get("mlx_patched") for r in rs]
        o = [old.get(tuple(sorted(r["cfg"].items()))) for r in rs]
        o = [x for x in o if x]
        print(f"| {key} | {len(rs)} | {geo(new):.2f}x | {min(new):.2f}x | {geo(pat):.2f}x | {min(pat):.2f}x | "
              + (f"{geo(o):.2f}x |" if o else " |"))
    allnew = [r["speedup"]["mlx"] for r in rows]
    allpat = [r["speedup"]["mlx_patched"] for r in rows]
    bad = [(r["speedup"]["mlx"], r["speedup"]["mlx_patched"], label(r["cfg"]), r["median"]["sk"] * 1e3,
            r["check"].get("mlx", {}).get("backend"), r["check"].get("mlx_patched", {}).get("backend")) for r in rows]
    print()
    print(f"all: {len(rows)} cells, rewrite faster in {sum(x > 1 for x in allnew)}, geomean {geo(allnew):.2f}x, "
          f"min {min(allnew):.2f}x; patched geomean {geo(allpat):.2f}x, min {min(allpat):.2f}x")
    print("\ncells below parity (rewrite or patched):")
    for s_new, s_pat, lab, sk, _, b2 in sorted(bad):
        if s_new < 0.99 or s_pat < 0.99:
            print(f"  {lab}: stock {sk:.1f} ms, rewrite {s_new:.2f}x, patched {s_pat:.2f}x (patched backend {b2})")
    mism = [label(r["cfg"]) for r in rows if r["check"].get("mlx", {}).get("rank") != r["check"].get("mlx", {}).get("rank_ref")]
    print("\nrank differences vs stock:", mism)


if __name__ == "__main__":
    main(*sys.argv[1:])
