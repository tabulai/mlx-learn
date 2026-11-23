"""Score patched-routing floors against crossover.py output from one or more machines.

For every candidate (min_samples, min_work, max_features, min_aspect) it counts
the shapes routed to mlxlearn and the routed shapes where the slowest data kind
is slower than stock on any machine. Usage:

    python calibrate_routing.py [--tolerance=0.995] xo_tall_m5.jsonl xo_tall_m1.jsonl
"""
import itertools
import json
import sys
from collections import defaultdict


def worst(path):
    w = defaultdict(lambda: (9e9, ""))
    for line in open(path):
        r = json.loads(line)
        key = (r["n"], r["d"])
        if r["speedup"] < w[key][0]:
            w[key] = (r["speedup"], r["data"])
    return dict(w)


def routed(n, d, ms, mf, mw, xf, ma):
    return not (n < ms or d < mf or n * d < mw or d > xf or n < ma * d)


def main():
    args = sys.argv[1:]
    tolerance = 0.995  # parity within timing noise
    if args and args[0].startswith("--tolerance="):
        tolerance = float(args.pop(0).split("=", 1)[1])
    machines = {p.split("/")[-1]: worst(p) for p in args}
    cells = sorted(set.intersection(*[set(w) for w in machines.values()]))
    scored = []
    for rule in itertools.product([1024, 2048, 4096, 8192], [1, 8, 16, 32], [0, 1 << 18, 1 << 19, 1 << 20],
                                  [256, 512, 1024], [8, 16, 32, 64]):
        viol, cover, margin = [], 0, 9e9
        for n, d in cells:
            if not routed(n, d, *rule):
                continue
            cover += 1
            for name, w in machines.items():
                s, kind = w[(n, d)]
                margin = min(margin, s)
                if s < tolerance:
                    viol.append((name, n, d, round(s, 2), kind))
        scored.append((len(viol), -cover, rule, margin, viol))
    scored.sort(key=lambda t: (t[0], t[1]))
    for nv, negc, rule, margin, viol in scored[:12]:
        ms, mf, mw, xf, ma = rule
        print(f"min_samples={ms} min_features={mf} min_work={mw} max_features={xf} min_aspect={ma}: "
              f"covers {-negc}/{len(cells)} shapes, below {tolerance} on {nv}, worst routed {margin:.2f}")
        for v in viol[:6]:
            print("    ", v)


if __name__ == "__main__":
    main()
