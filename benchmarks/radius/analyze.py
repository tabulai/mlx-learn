#!/usr/bin/env python
"""Summarize radius benchmark JSONL produced by bench_radius.py.

For every config it reports times and ratios, all written as speedups, where a
value above 1 means mlxlearn is faster:

* Layer 1 (``mlx``) against ``sk_auto``, ``sk_brute``, ``sk_auto_nj`` and the best
  scikit-learn variant;
* each patched variant against the stock variant its user configured
  (``user_stock``: ``sk_auto`` for default users, ``sk_auto_nj`` for ``n_jobs=-1``
  users, ``sk_brute`` for ``algorithm="brute"`` users);
* same-session controls, as ``ctl_<v>`` time divided by ``<v>`` time (new code
  against old code);
* regret, the automatic route's time divided by the best forced route's time (only
  with ``--routes`` variants). For this one, lower is better.

It then aggregates win counts and geometric means per region (low-d <= 8,
mid-d 9-15, high-d >= 16, real data, ops), checks the acceptance bars, lists the
routes taken, and reports correctness flags and errors. ``--md out.md`` also
writes the report as Markdown.

``--metric`` picks the time compared: the warm median (default), the warm
minimum, the cold first query, ``e2e`` (fit + cold first query), or ``paired``.
``paired`` is the median, over the warm rounds both sides ran, of the per-round
ratio. On a laptop, short gapped calls switch between fast and slow phases that
span whole rounds, so the paired ratio is the more robust estimate there.

Usage::

    python benchmarks/radius/analyze.py results.jsonl [more.jsonl ...] \\
        [--label L] [--metric warm_med|warm_min|cold_med|e2e|paired] [--md summary.md]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

REAL_DATA = ("mnist", "clip", "spacev")
REGIONS = ("low-d (<=8)", "mid-d (9-15)", "high-d (>=16)", "real", "ops")
METRICS = ("warm_med", "warm_min", "cold_med", "e2e", "paired")
SK_VARIANTS = ("sk_auto", "sk_brute", "sk_auto_nj")
#: The stock variant each patched user's configuration runs (bench_radius.REGISTRY).
DEFAULT_STOCK = {"mlx_patched": "sk_auto", "mlx_patched_nj": "sk_auto_nj",
                 "mlx_patched_brute": "sk_brute"}
ROUTE_PREFIX = "mlx_route="
#: Acceptance thresholds (design plan, M1/M4 gates).
PATCHED_GATE = 0.95
CONTROL_GATE = 1 / 1.05
REGRET_GATE = 1.25


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def config_key(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k not in ("d_out",))


def canonical_key(cfg):
    """``config_key`` without the implied ``d`` of real datasets (as bench_radius)."""
    if cfg.get("data") in REAL_DATA:
        cfg = {k: v for k, v in cfg.items() if k != "d"}
    return config_key(cfg)


def load(paths, label=None):
    """Records with summaries and error records; the last record per (label, config) wins."""
    latest = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "cfg" not in r:
                    continue
                if label and r.get("label") != label:
                    continue
                latest[(r.get("label"), canonical_key(r["cfg"]))] = r
    rows = [r for r in latest.values() if "summary" in r]
    errors = [r for r in latest.values() if "error" in r]
    return rows, errors


# ---------------------------------------------------------------------------
# per-config cells
# ---------------------------------------------------------------------------

def region(rec) -> str:
    c = rec["cfg"]
    d = rec.get("d", c.get("d"))
    if c["op"] != "rn":
        return "ops"
    if c["data"] in REAL_DATA:
        return "real"
    if d <= 8:
        return "low-d (<=8)"
    if d <= 15:
        return "mid-d (9-15)"
    return "high-d (>=16)"


def family(name: str) -> str:
    if name.startswith("sk_"):
        return "sk"
    if name.startswith("ctl_"):
        return "ctl"
    if name.startswith(ROUTE_PREFIX):
        return "route"
    if name.startswith("mlx_patched"):
        return "patched"
    return "mlx"


def metric_value(entry, metric):
    if not entry:
        return None
    if metric == "e2e":
        cold, fit = entry.get("cold_med"), entry.get("fit_med")
        return cold + fit if cold is not None and fit is not None else None
    value = entry.get(metric)
    return value if value else None


def stock_of(rec, name, times):
    """The user's stock variant for a patched (or control-patched) variant."""
    stored = rec.get("user_stock", {}).get(name)
    if stored:
        return stored if stored in times else None
    base = name[4:] if name.startswith("ctl_") else name
    stock = DEFAULT_STOCK.get(base)
    if stock is None:
        return None
    if stock in times:
        return stock
    d = rec.get("d", rec["cfg"].get("d")) or 0
    if stock == "sk_auto_nj" and d > 15 and "sk_auto" in times:
        return "sk_auto"
    return None


def _bad(check: dict) -> bool:
    """A correctness flag against scikit-learn brute force (boundary-only
    membership disagreements still count; inspect ``off_boundary`` for those)."""
    return bool(check.get("bad_rows") or check.get("off_boundary") or check.get("disagree")
                or check.get("nan_mismatch") or check.get("same_pattern") is False
                or check.get("unsorted_rows"))


def _differs_from_counterpart(check: dict) -> bool:
    """A control output that differs from its in-process counterpart's output."""
    return any(v is False for k, v in check.items()
               if k.startswith("same_membership_as_mlx") or k.startswith("same_as_"))


@dataclass
class Cell:
    rec: dict
    key: str
    label: str
    region: str
    pairs: int | None
    times: dict
    values: dict = field(default_factory=dict)  # comparison name -> ratio
    stocks: dict = field(default_factory=dict)  # patched variant -> user stock name


def _round_samples(rec, name) -> dict:
    """``{round: seconds}`` of a variant's timed warm calls."""
    values = (rec.get("warm") or {}).get(name) or []
    rounds = (rec.get("warm_round") or {}).get(name)
    if rounds is None or len(rounds) != len(values):
        rounds = range(1, len(values) + 1)  # older records: one sample per timed round
    return dict(zip(rounds, values, strict=False))


def ratio_function(rec, times, metric):
    """``ratio(num, den)``: the best (minimum) time of ``num`` over that of ``den``.

    With ``metric="paired"`` it is the median, over the warm rounds both sides ran,
    of the per-round ratio. This is robust to machine-wide speed phases that change
    between rounds.
    """
    if metric != "paired":
        def ratio(num, den):
            a = min((times[n] for n in num if n in times), default=None)
            b = min((times[n] for n in den if n in times), default=None)
            return a / b if a and b else None
        return ratio
    cache = {}

    def samples(name):
        if name not in cache:
            cache[name] = _round_samples(rec, name)
        return cache[name]

    def ratio(num, den):
        rounds = set().union(*(samples(n) for n in num)) & set().union(*(samples(n) for n in den))
        per_round = []
        for r in sorted(rounds):
            a = min((samples(n)[r] for n in num if r in samples(n)), default=None)
            b = min((samples(n)[r] for n in den if r in samples(n)), default=None)
            if a and b:
                per_round.append(a / b)
        return statistics.median(per_round) if per_round else None

    return ratio


def build_cell(rec, metric="warm_med") -> Cell:
    c = rec["cfg"]
    weights = "/" + c["weights"] if "weights" in c else ""
    extras = "".join(f" {k}={c[k]}" for k in ("offset", "order", "queries") if k in c)
    key = (f"{c['data']}/{c['op']}{weights} n={c['n']} d={rec.get('d', c.get('d'))} "
           f"nq={c['nq']} m={c.get('m')}{extras}")
    times = {}
    for name, entry in rec.get("summary", {}).items():
        value = metric_value(entry, "warm_med" if metric == "paired" else metric)
        if value is not None:
            times[name] = value
    cell = Cell(rec, key, rec.get("label", ""), region(rec),
                (rec.get("ref_summary") or {}).get("pairs"), times)
    ratio = ratio_function(rec, times, metric)
    sk = [n for n in times if family(n) == "sk"]
    v = cell.values
    if "mlx" in times:
        for base in SK_VARIANTS:
            v[f"mlx vs {base}"] = ratio([base], ["mlx"])
        v["mlx vs sk_best"] = ratio(sk, ["mlx"])
    for name in times:
        if family(name) == "patched":
            stock = stock_of(rec, name, times)
            cell.stocks[name] = stock
            v[f"{name} vs user stock"] = ratio([stock], [name]) if stock else None
            v[f"{name} vs sk_best"] = ratio(sk, [name])
        elif family(name) == "ctl" and name[4:] in times:
            v[f"{name[4:]}: new vs ctl"] = ratio([name], [name[4:]])
    forced = [n for n in times if family(n) == "route"]
    if forced:
        engines = [n for n in forced if n != ROUTE_PREFIX + "stock"]
        if "mlx" in times and engines:
            v["regret mlx"] = ratio(["mlx"], engines)
        options = list(forced)
        stock = stock_of(rec, "mlx_patched", times)
        if stock:
            options.append(stock)
        if "mlx_patched" in times:
            v["regret mlx_patched"] = ratio(["mlx_patched"], options)
    cell.values = {k: x for k, x in v.items() if x is not None}
    return cell


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def gate_of(comparison: str):
    """``(threshold, higher_is_better, label)`` for a comparison."""
    if comparison.startswith("regret"):
        return REGRET_GATE, False, f"<={REGRET_GATE:g}"
    if comparison.endswith("new vs ctl"):
        return CONTROL_GATE, True, "<=5% slower"
    if comparison.endswith("vs user stock"):
        return PATCHED_GATE, True, f">={PATCHED_GATE:g}"
    return 1.0, True, ">=1.0"


def stats(values) -> dict:
    vals = [x for x in values if x is not None and x > 0 and math.isfinite(x)]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "geo": math.exp(sum(math.log(x) for x in vals) / len(vals)),
            "min": min(vals), "max": max(vals), "wins": sum(x >= 1.0 for x in vals)}


def comparison_order(names):
    def rank(name):
        if name.endswith("new vs ctl"):
            group = 2
        elif name.startswith("regret"):
            group = 3
        elif name.startswith("mlx vs"):
            group = 0
        else:
            group = 1
        return (group, name)
    return sorted(names, key=rank)


def aggregate(cells) -> dict:
    """``{comparison: {region: stats}}``, with an ``all`` region."""
    grouped = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        for name, value in cell.values.items():
            grouped[name][cell.region].append(value)
            grouped[name]["all"].append(value)
    out = {}
    for name in comparison_order(grouped):
        threshold, higher, _ = gate_of(name)
        out[name] = {}
        for reg in (*REGIONS, "all"):
            values = grouped[name].get(reg, [])
            s = stats(values)
            if s["n"]:
                s["pass"] = sum((x >= threshold) if higher else (x <= threshold)
                                for x in values if x is not None)
            out[name][reg] = s
    return out


def acceptance(cells) -> list[tuple[str, int, int]]:
    """``(bar, passing cells, measured cells)`` for the plan's acceptance bars."""
    bars = []
    l1 = [c for c in cells if "mlx vs sk_auto" in c.values and "mlx vs sk_brute" in c.values]
    bars.append(("Layer 1 >= 1.0x both sk_auto and sk_brute",
                 sum(c.values["mlx vs sk_auto"] >= 1 and c.values["mlx vs sk_brute"] >= 1
                     for c in l1), len(l1)))
    for name in sorted({k for c in cells for k in c.values if k.endswith("vs user stock")}):
        have = [c.values[name] for c in cells if name in c.values]
        bars.append((f"{name.replace(' vs user stock', '')} >= {PATCHED_GATE:g}x user stock",
                     sum(x >= PATCHED_GATE for x in have), len(have)))
    for name in sorted({k for c in cells for k in c.values if k.endswith("new vs ctl")}):
        have = [c.values[name] for c in cells if name in c.values]
        bars.append((f"{name.split(':')[0]} not >5% slower than control",
                     sum(x >= CONTROL_GATE for x in have), len(have)))
    for name in sorted({k for c in cells for k in c.values if k.startswith("regret")}):
        have = [c.values[name] for c in cells if name in c.values]
        bars.append((f"{name} <= {REGRET_GATE:g}", sum(x <= REGRET_GATE for x in have),
                     len(have)))
        bars.append((f"{name} <= 2", sum(x <= 2 for x in have), len(have)))
    return bars


def route_table(cells) -> list[tuple[str, str, str, int, str]]:
    """``(variant, region, route, cells, example reason)`` for the warm routes taken."""
    counts = Counter()
    reasons = {}
    for cell in cells:
        rec = cell.rec
        for slot, routes in rec.get("routes", {}).items():
            name, _, phase = slot.rpartition(":")
            if phase != "warm" or family(name) == "sk":
                continue
            route = ",".join(routes) if routes else "-"
            counts[(name, cell.region, route)] += 1
            plan = rec.get("plans", {}).get(slot) or {}
            if plan.get("reason"):
                reasons.setdefault((name, route), str(plan["reason"]))
    return [(name, reg, route, k, reasons.get((name, route), ""))
            for (name, reg, route), k in sorted(counts.items())]


def forced_route_honored(cells) -> list[tuple[str, int, int]]:
    """For each forced-route variant: cells where the recorded route matches."""
    seen = defaultdict(lambda: [0, 0])
    for cell in cells:
        for slot, routes in cell.rec.get("routes", {}).items():
            name, _, phase = slot.rpartition(":")
            if phase != "warm" or family(name) != "route":
                continue
            wanted = name[len(ROUTE_PREFIX):]
            seen[name][1] += 1
            seen[name][0] += bool(routes) and all(r == wanted for r in routes)
    return [(name, ok, total) for name, (ok, total) in sorted(seen.items())]


def correctness(cells) -> list[tuple[str, int, int, int]]:
    """``(variant, flagged cells, cells differing from the counterpart, checked cells)``."""
    flagged = Counter()
    differs = Counter()
    checked = Counter()
    for cell in cells:
        for name, check in cell.rec.get("check", {}).items():
            checked[name] += 1
            flagged[name] += _bad(check)
            differs[name] += _differs_from_counterpart(check)
    return [(name, flagged[name], differs[name], checked[name]) for name in sorted(checked)]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def fmt_ms(v):
    if v is None:
        return "-"
    return f"{v * 1e3:.1f}" if v < 10 else f"{v:.2f}s"


def fmt_x(v):
    return "-" if v is None else f"{v:.2f}"


def config_rows(cells, multi_label):
    header = ["config", "pairs", "sk_auto", "sk_brute", "sk_nj", "mlx", "patched",
              "mlx/auto", "mlx/brute", "mlx/best", "pat/user", "new/ctl mlx",
              "new/ctl pat", "regret", "route mlx", "route pat (cold>warm)", "bad"]
    if multi_label:
        header.insert(0, "label")
    rows = []
    for cell in cells:
        t, v, rec = cell.times, cell.values, cell.rec
        routes = rec.get("routes", {})
        bad = sorted(name for name, check in rec.get("check", {}).items() if _bad(check))
        row = [cell.key, "-" if cell.pairs is None else str(cell.pairs),
               fmt_ms(t.get("sk_auto")), fmt_ms(t.get("sk_brute")), fmt_ms(t.get("sk_auto_nj")),
               fmt_ms(t.get("mlx")), fmt_ms(t.get("mlx_patched")),
               fmt_x(v.get("mlx vs sk_auto")), fmt_x(v.get("mlx vs sk_brute")),
               fmt_x(v.get("mlx vs sk_best")), fmt_x(v.get("mlx_patched vs user stock")),
               fmt_x(v.get("mlx: new vs ctl")), fmt_x(v.get("mlx_patched: new vs ctl")),
               fmt_x(v.get("regret mlx")),
               ",".join(routes.get("mlx:warm", [])) or "-",
               (",".join(routes.get("mlx_patched:cold", [])) + ">"
                + ",".join(routes.get("mlx_patched:warm", []))),
               " ".join(bad)]
        if multi_label:
            row.insert(0, cell.label)
        rows.append(row)
    return header, rows


def region_rows(agg):
    header = ["comparison", "region", "n", "wins", "gate", "pass", "geomean", "min", "max"]
    rows = []
    for name, by_region in agg.items():
        _, _, gate_label = gate_of(name)
        for reg in (*REGIONS, "all"):
            s = by_region.get(reg, {"n": 0})
            if not s["n"]:
                continue
            # A regret is a cost (lower is better): "wins" does not apply.
            wins = "-" if name.startswith("regret") else f"{s['wins']}/{s['n']}"
            rows.append([name, reg, str(s["n"]), wins, gate_label,
                         f"{s['pass']}/{s['n']}", f"{s['geo']:.2f}", f"{s['min']:.2f}",
                         f"{s['max']:.2f}"])
    return header, rows


def text_table(header, rows) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=False)] if rows else \
        [len(h) for h in header]
    lines = ["  ".join(str(x).ljust(w) for x, w in zip(header, widths, strict=False)).rstrip()]
    lines += ["  ".join(str(x).ljust(w) for x, w in zip(row, widths, strict=False)).rstrip() for row in rows]
    return "\n".join(lines)


def md_table(header, rows) -> str:
    def esc(x):
        return str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(esc(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(esc(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def report(cells, errors, metric, *, markdown=False) -> str:
    table = md_table if markdown else text_table

    def heading(text):
        return f"\n## {text}\n" if markdown else f"\n== {text} =="

    multi_label = len({c.label for c in cells}) > 1
    out = []
    labels = sorted({c.label for c in cells})
    title = f"Radius benchmark summary ({metric}; labels: {', '.join(labels) or '-'})"
    out.append(f"# {title}" if markdown else title)
    out.append(f"{len(cells)} configs, {len(errors)} errors. Ratios are speedups "
               f"(baseline time / variant time); regret is auto time / best forced route.")
    out.append(heading("Acceptance bars"))
    bars = acceptance(cells)
    out.append(table(["bar", "pass", "cells", "share"],
                     [[b, str(k), str(n), f"{(k / n if n else 0):.0%}"] for b, k, n in bars]))
    out.append(heading("Per region"))
    out.append(table(*region_rows(aggregate(cells))))
    out.append(heading("Routes taken (warm calls)"))
    out.append(table(["variant", "region", "route", "cells", "reason (example)"],
                     [[a, b, c, str(k), r[:80]] for a, b, c, k, r in route_table(cells)]))
    honored = forced_route_honored(cells)
    if honored:
        out.append(heading("Forced routes honored"))
        out.append(table(["variant", "matching", "cells"],
                         [[n, str(k), str(t)] for n, k, t in honored]))
    out.append(heading("Correctness flags"))
    out.append(table(["variant", "flagged vs sk_brute", "control differs from new", "checked"],
                     [[n, str(k), str(x) if n.startswith("ctl_") else "-", str(t)]
                      for n, k, x, t in correctness(cells)]))
    if errors or any(c.rec.get("variant_errors") for c in cells):
        out.append(heading("Errors"))
        rows = [[canonical_key(e["cfg"]), "config", e["error"][:160]] for e in errors]
        for cell in cells:
            for name, text in cell.rec.get("variant_errors", {}).items():
                rows.append([cell.key, name, str(text).splitlines()[0][:160] if text else ""])
        out.append(table(["config", "variant", "error"], rows))
    out.append(heading("Per config"))
    out.append(table(*config_rows(cells, multi_label)))
    return "\n".join(out) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Summarize bench_radius.py JSONL output.")
    ap.add_argument("paths", nargs="+", help="JSONL files")
    ap.add_argument("--label", default=None, help="only records with this label")
    ap.add_argument("--metric", choices=METRICS, default="warm_med",
                    help="time compared: warm median (default), warm minimum, cold first query, "
                         "e2e = fit + cold first query, or paired = median over warm rounds of "
                         "the per-round ratio")
    ap.add_argument("--md", default=None, help="also write the report as Markdown here")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows, errors = load(args.paths, args.label)
    cells = [build_cell(r, args.metric) for r in rows]
    sys.stdout.write(report(cells, errors, args.metric))
    if args.md:
        with open(args.md, "w") as f:
            f.write(report(cells, errors, args.metric, markdown=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
