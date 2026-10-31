"""Deterministic serial order and descriptive phase diagnostics; no workloads."""

from __future__ import annotations

import math
import statistics


def williams_orders(names):
    """Position/predecessor-balanced complete blocks (2n rows for odd n)."""
    names = list(names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two distinct workers are required")
    n = len(names)
    base = [0]
    lo, hi = 1, n - 1
    while lo <= hi:
        base.append(lo)
        lo += 1
        if lo <= hi:
            base.append(hi)
            hi -= 1
    rows = [[names[(item + offset) % n] for item in base] for offset in range(n)]
    return rows + [list(reversed(row)) for row in rows] if n % 2 else rows


def round_order(starting_order, pair, policy):
    if policy == "legacy":
        return list(starting_order) if pair % 2 == 0 else list(reversed(starting_order))
    if policy == "balanced":
        rows = williams_orders(starting_order)
        return rows[pair % len(rows)]
    raise ValueError(f"Unknown order policy: {policy}")


def placement(order, previous_request):
    return {name: {"position": index,
                   "within_round_predecessor": order[index - 1] if index else None,
                   "previous_request": order[index - 1] if index else previous_request}
            for index, name in enumerate(order)}


def _median(values):
    return statistics.median(values) if values else None


def _valid_time(sample):
    value = sample.get("ms")
    return "error" not in sample and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def phase_diagnostics(experiment, variants):
    """Keep every round; phase splits diagnose order, not significance or clocks."""
    count = len(experiment["rounds"])
    edge = min(10, count // 3) if count >= 3 else 1
    groups = {"all": list(range(count)), "first": list(range(edge)),
              "middle": list(range(edge, count - edge)), "last": list(range(max(edge, count - edge), count)),
              "even_rounds": list(range(0, count, 2)), "odd_rounds": list(range(1, count, 2))}
    samples = {v: experiment["variants"][v]["samples"] for v in variants}

    def summarize(indices, variant):
        valid = [i for i in indices if _valid_time(samples[variant][i])]
        paired = [i for i in valid if _valid_time(samples["current"][i])]
        return {"rounds": indices, "valid_count": len(valid),
                "valid_rounds": valid, "invalid_rounds": [i for i in indices if i not in valid],
                "invalid_count": len(indices) - len(valid),
                "paired_valid_rounds": paired, "paired_valid_count": len(paired),
                "paired_invalid_rounds": [i for i in indices if i not in paired],
                "paired_invalid_count": len(indices) - len(paired),
                "median_ms": _median([samples[variant][i]["ms"] for i in valid]),
                "median_ratio_over_current": _median([
                    samples[variant][i]["ms"] / samples["current"][i]["ms"] for i in paired]),
                "median_parent_seconds_since_warmup_probe": _median([
                    samples[variant][i]["order_context"]["seconds_since_warmup_probe"] for i in valid])}

    return {variant: {
        "phase": {name: summarize(indices, variant) for name, indices in groups.items()},
        "position": {str(position): summarize([i for i, row in enumerate(experiment["rounds"])
                        if row["placement"][variant]["position"] == position], variant)
                     for position in range(len(variants))},
        "within_round_predecessor": {str(previous): summarize([
            i for i, row in enumerate(experiment["rounds"])
            if row["placement"][variant]["within_round_predecessor"] == previous], variant)
            for previous in [None, *(v for v in variants if v != variant)]},
    } for variant in variants}
