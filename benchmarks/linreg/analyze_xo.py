"""Worst-case (over data kinds) speedup matrix from crossover.py output."""
import json
import sys
from collections import defaultdict


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    wide = any(not r.get("intercept", True) for r in rows) or any(r["d"] > r["n"] for r in rows)
    worst = defaultdict(lambda: (9.9, ""))
    rank_bad = []
    for r in rows:
        key = (r["n"], r["d"]) if not wide else (r["n"], r["d"], r.get("intercept", True))
        tag = r["data"]
        if r["speedup"] < worst[key][0]:
            worst[key] = (r["speedup"], tag)
        if r["rank"] != r["rank_ref"]:
            rank_bad.append(r)
    ns = sorted({k[0] for k in worst})
    ds = sorted({k[1] for k in worst})
    variants = sorted({k[2] for k in worst}) if wide else [None]
    for v in variants:
        if v is not None:
            print(f"\n=== intercept={v}")
        print("n \\ d " + "".join(f"{d:>9d}" for d in ds))
        for n in ns:
            cells = []
            for d in ds:
                key = (n, d) if v is None else (n, d, v)
                if key in worst:
                    s, tag = worst[key]
                    cells.append(f"{s:6.2f}{tag[0]:>1s}{'*' if s < 1.0 else ' '} ")
                else:
                    cells.append(" " * 9)
            print(f"{n:6d} " + "".join(cells))
    print("\nletters: worst kind (g=gauss o=onehot m=mixed c=cond10); * = slower than stock")
    print(f"rank mismatches: {len(rank_bad)}")
    for r in rank_bad[:10]:
        print("  ", r)


if __name__ == "__main__":
    main()
