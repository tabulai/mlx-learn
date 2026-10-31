"""Read existing KNN reports and identify every retained paired-round miss.

This is a CPU-only report audit. It never reruns, trims, reorders or replaces
timings, and relative-to-median labels describe observations rather than causes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .knn_acceleration import TARGET_OPERATIONS
from .knn_pipeline import ROOT


def audit_reports(summaries, *, target_speedup=2.0):
    rows = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        for report_path in summary["reports"]:
            report = json.loads(Path(report_path).read_text())
            variants = report["cases"][0]["variants"]
            for operation in TARGET_OPERATIONS:
                current = [s["ms"] for s in variants["current"][operation]["samples"]]
                default = [s["ms"] for s in variants["sklearn_default"][operation]["samples"]]
                current_median, default_median = statistics.median(current), statistics.median(default)
                for index, (a, b) in enumerate(zip(current, default, strict=True)):
                    current_high, sklearn_low = a > 1.1 * current_median, b < 0.9 * default_median
                    category = ("both" if current_high and sklearn_low else
                                "current_above_110pct_median" if current_high else
                                "sklearn_below_90pct_median" if sklearn_low else
                                "neither_10pct_threshold")
                    rows.append({
                        "run": summary_path.name,
                        "base_seed": report["seed"],
                        "operation": operation,
                        "round": index + 1,
                        "variant_order": "forward" if (report["warmup"] + index) % 2 == 0 else "reverse",
                        "current_ms": a,
                        "sklearn_default_ms": b,
                        "speedup": b / a,
                        "pass": b / a >= target_speedup,
                        "current_over_own_median": a / current_median,
                        "sklearn_over_own_median": b / default_median,
                        "maximum_current_ms_for_target": b / target_speedup,
                        "required_current_reduction_fraction": max(0, 1 - b / (target_speedup * a)),
                        "descriptive_category": category,
                    })
    misses = [r for r in rows if not r["pass"]]
    operations = {}
    for operation in TARGET_OPERATIONS:
        selected = [r for r in rows if r["operation"] == operation]
        current = [r["current_ms"] for r in selected]
        default = [r["sklearn_default_ms"] for r in selected]
        operations[operation] = {
            "round_count": len(selected),
            "miss_count": sum(not r["pass"] for r in selected),
            "minimum_paired_speedup": min(r["speedup"] for r in selected),
            "current_min_ms": min(current),
            "current_max_ms": max(current),
            "sklearn_default_min_ms": min(default),
            "sklearn_default_max_ms": max(default),
            "current_ceiling_ms_using_fastest_observed_sklearn": min(default) / target_speedup,
            "worst_current_divided_by_ceiling": max(current) / (min(default) / target_speedup),
        }
    return {
        "target_speedup": target_speedup,
        "sources": [str(p) for p in summaries],
        "policy": "All retained timed samples; first-call observations and warmups are separate. "
                  "Categories use ±10% of each operation/seed/run's own median. They are "
                  "descriptive and do not establish the cause of a timing change. The "
                  "fastest-sklearn ceiling is a stronger unpaired envelope, not the gate.",
        "round_count": len(rows),
        "pass_count": len(rows) - len(misses),
        "miss_count": len(misses),
        "minimum_paired_speedup": min(r["speedup"] for r in rows),
        "maximum_required_current_reduction_fraction": max(
            r["required_current_reduction_fraction"] for r in rows
        ),
        "miss_categories": dict(Counter(r["descriptive_category"] for r in misses)),
        "miss_round_numbers": dict(Counter(r["round"] for r in misses)),
        "miss_variant_orders": dict(Counter(r["variant_order"] for r in misses)),
        "operations": operations,
        "misses": sorted(misses, key=lambda r: r["speedup"]),
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+", help="Existing acceleration summary JSONs")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/knn-tail-audit.json")
    args = parser.parse_args()
    result = audit_reports(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['pass_count']}/{result['round_count']} retained pairs pass; "
          f"worst {result['minimum_paired_speedup']:.3f}x")
    for operation, data in result["operations"].items():
        print(f"{operation}: {data['miss_count']}/{data['round_count']} misses; "
              f"fastest-sklearn ceiling {data['current_ceiling_ms_using_fastest_observed_sklearn']:.3f} ms")


if __name__ == "__main__":
    main()
