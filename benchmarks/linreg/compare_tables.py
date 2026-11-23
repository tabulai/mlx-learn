"""Side-by-side table: PR 4 vs the rewrite, from bench_linreg.py JSONL files."""
import json
import math
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "cfg" not in r or "median" not in r:
            continue
        c = r["cfg"]
        out[tuple(sorted(c.items()))] = r
    return out


def label(cfg):
    c = dict(cfg)
    parts = [f"{c.pop('data')} {c.pop('n')}x{c.pop('d')}"]
    for k, v in sorted(c.items()):
        parts.append(f"{k}={v}")
    return " ".join(parts)


def geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    old = load(sys.argv[1]) if sys.argv[1] != "-" else {}
    new = load(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else ""
    print("| config | stock ms | PR 4 | rewrite | patched | path ok |" if old else
          "| config | stock ms | rewrite | patched | path ok |")
    print("|---|---:|---:|---:|---:|---|" if old else "|---|---:|---:|---:|---|")
    sp_new, sp_pat, sp_old, wins, n_cells, bad = [], [], [], 0, 0, []
    for key, r in new.items():
        sk = r["median"].get("sk")
        s_new = r.get("speedup", {}).get("mlx")
        s_pat = r.get("speedup", {}).get("mlx_patched")
        chk = r.get("check", {}).get("mlx", {})
        ok = chk.get("rank") == chk.get("rank_ref") and abs(chk.get("obj_rel", 0)) < 1e-6
        if not ok:
            bad.append(label(key))
        o = old.get(key)
        s_old = o.get("speedup", {}).get("mlx") if o else None
        n_cells += 1
        wins += bool(s_new and s_new > 1)
        sp_new.append(s_new)
        sp_pat.append(s_pat)
        if s_old:
            sp_old.append(s_old)
        cols = [label(key), f"{sk*1e3:.1f}"]
        if old:
            cols.append(f"{s_old:.2f}x" if s_old else "")
        cols += [f"{s_new:.2f}x" if s_new else "", f"{s_pat:.2f}x" if s_pat else "", "yes" if ok else "**no**"]
        print("| " + " | ".join(cols) + " |")
    print()
    print(f"{title}: {n_cells} cells; rewrite faster than stock in {wins}; geomean {geo(sp_new):.2f}x "
          f"(min {min(x for x in sp_new if x):.2f}x); patched geomean {geo(sp_pat):.2f}x "
          f"(min {min(x for x in sp_pat if x):.2f}x)"
          + (f"; PR 4 geomean {geo(sp_old):.2f}x (min {min(sp_old):.2f}x)" if sp_old else ""))
    if bad:
        print("rank/objective mismatches:", bad)


if __name__ == "__main__":
    main()
