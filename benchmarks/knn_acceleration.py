"""Audit the 50,000 x 32 KNN acceleration across three independent datasets.

    python -m benchmarks.knn_acceleration --output benchmarks/results/knn-acceleration-2026-10-28.json

Each dataset uses the existing isolated, interleaved pipeline benchmark. The
reference is the working-tree implementation preceding this acceleration task,
not the much older pre-KNN baseline. All first-call observations, warmed samples
and accuracy checks are retained in adjacent per-seed JSON reports.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .knn_pipeline import CASES, ROOT, run, sha256

CASE = "batch_50000_32_k5"
REFERENCE = ROOT / "benchmarks/reference/knn-pre-acceleration-2026-10-28.zip"
TARGET_OPERATIONS = (
    "fit_query", "kneighbors", "indices", "predict_uniform", "proba_uniform",
    "regress_uniform", "predict_distance", "proba_distance", "regress_distance",
)


def summarize_reports(reports, *, target_speedup, paired_rounds=True):
    """Gate interleaved pairs; explicit unpaired diagnostics retain their median gate."""
    if not reports or not math.isfinite(target_speedup) or target_speedup <= 0:
        raise ValueError("Reports and a finite positive target speedup are required")
    first = reports[0]
    operations = [op for op in first["cases"][0]["variants"]["current"] if op != "cold"]
    summary = {}
    for operation in operations:
        samples = []
        for report in reports:
            variants = report["cases"][0]["variants"]
            current = variants["current"][operation]["median_ms"]
            times = {name: value[operation]["median_ms"]
                     for name, value in variants.items() if operation in value}
            current_rounds = [sample["ms"] for sample in variants["current"][operation]["samples"]]
            default_rounds = [sample["ms"] for sample in variants["sklearn_default"][operation]["samples"]]
            if len(current_rounds) != len(default_rounds):
                raise ValueError("Compared variants must retain the same number of rounds")
            if not current_rounds or any(
                not math.isfinite(value) or value <= 0
                for value in (*current_rounds, *default_rounds)
            ):
                raise ValueError("Compared variants must retain finite positive timed samples")
            paired_ratios = [default / current for default, current
                             in zip(default_rounds, current_rounds, strict=True)]
            paired_misses = [
                {
                    "round": position + 1,
                    "current_ms": current_ms,
                    "sklearn_default_ms": default_ms,
                    "speedup_vs_sklearn_default": ratio,
                    "maximum_current_ms_for_target": default_ms / target_speedup,
                }
                for position, (current_ms, default_ms, ratio) in enumerate(
                    zip(current_rounds, default_rounds, paired_ratios, strict=True)
                )
                if ratio < target_speedup
            ]
            split = (len(current_rounds) + 1) // 2
            halves = {}
            for name, selection in (("first", slice(None, split)), ("second", slice(split, None))):
                if current_rounds[selection]:
                    halves[name] = {
                        "round_count": len(current_rounds[selection]),
                        "current_median_ms": statistics.median(current_rounds[selection]),
                        "sklearn_default_median_ms": statistics.median(default_rounds[selection]),
                        "speedup_vs_sklearn_default": (
                            statistics.median(default_rounds[selection])
                            / statistics.median(current_rounds[selection])
                        ),
                    }
            samples.append({
                "base_seed": report["seed"],
                "dataset_seed": report["seed"] + next(i for i, c in enumerate(CASES) if c.name == CASE),
                "median_ms": times,
                "speedup_vs_previous": times["reference"] / current,
                "speedup_vs_sklearn_1": times["sklearn_1"] / current,
                "speedup_vs_sklearn_default": times["sklearn_default"] / current,
                "paired_round_speedups_vs_sklearn_default": paired_ratios,
                "paired_rounds_meeting_target": sum(ratio >= target_speedup for ratio in paired_ratios),
                "paired_round_count": len(paired_ratios),
                "minimum_paired_round_speedup_vs_sklearn_default": min(paired_ratios),
                "all_paired_rounds_meet_target": not paired_misses,
                "paired_round_misses": paired_misses,
                "halves": halves,
                "current_range_ms": [variants["current"][operation]["min_ms"],
                                     variants["current"][operation]["max_ms"]],
                "sklearn_default_range_ms": [variants["sklearn_default"][operation]["min_ms"],
                                             variants["sklearn_default"][operation]["max_ms"]],
            })
        ratios = [sample["speedup_vs_sklearn_default"] for sample in samples]
        summary[operation] = {
            "seeds": samples,
            "minimum_speedup_vs_sklearn_default": min(ratios),
            "maximum_speedup_vs_sklearn_default": max(ratios),
            "all_seed_medians_meet_target": all(ratio >= target_speedup for ratio in ratios),
            "minimum_paired_round_speedup_vs_sklearn_default": min(
                sample["minimum_paired_round_speedup_vs_sklearn_default"] for sample in samples
            ),
            "all_paired_rounds_meet_target": all(
                sample["all_paired_rounds_meet_target"] for sample in samples
            ),
        }
    result = {
        "operations": summary,
        "implementation_quality_pass": all(report["implementation_quality_pass"] for report in reports),
        "quality_by_variant": {
            variant: all(report["quality_by_variant"][variant] for report in reports)
            for variant in first["quality_by_variant"]
        },
        "source_integrity_pass": (
            all(report["source_integrity_pass"] for report in reports)
            and all(report["workers"]["current"]["source_hashes"]
                    == first["workers"]["current"]["source_hashes"] for report in reports)
        ),
        "all_target_medians_pass": all(
            summary[operation]["all_seed_medians_meet_target"] for operation in TARGET_OPERATIONS
        ),
        "all_target_operations_pass": all(
            summary[operation]["all_paired_rounds_meet_target"] for operation in TARGET_OPERATIONS
        ),
        "minimum_target_paired_round_speedup": min(
            summary[operation]["minimum_paired_round_speedup_vs_sklearn_default"]
            for operation in TARGET_OPERATIONS
        ),
        "target_paired_round_count": sum(
            sample["paired_round_count"]
            for operation in TARGET_OPERATIONS for sample in summary[operation]["seeds"]
        ),
        "target_paired_rounds_meeting_target": sum(
            sample["paired_rounds_meeting_target"]
            for operation in TARGET_OPERATIONS for sample in summary[operation]["seeds"]
        ),
    }
    if not paired_rounds:
        # Continuous variant blocks are not simultaneous/interleaved pairs.
        # Preserve their separately documented median-only diagnostic policy.
        result["all_target_operations_pass"] = result["all_target_medians_pass"]
        for name in ("minimum_target_paired_round_speedup", "target_paired_round_count",
                     "target_paired_rounds_meeting_target"):
            result.pop(name)
        for operation in result["operations"].values():
            operation.pop("minimum_paired_round_speedup_vs_sklearn_default")
            operation.pop("all_paired_rounds_meet_target")
            for sample in operation["seeds"]:
                for name in tuple(sample):
                    if "paired_round" in name:
                        sample.pop(name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43],
                        help="Base seeds; this case adds 10, giving datasets 51/52/53 by default.")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--target-speedup", type=float, default=2.0)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-acceleration-2026-10-28.json")
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or not math.isfinite(args.target_speedup)
            or args.target_speedup <= 0):
        parser.error("seeds must be distinct and target speedup finite and positive")
    reports, paths = [], []
    for seed in args.seeds:
        output = args.output.with_name(f"{args.output.stem}-seed{seed}{args.output.suffix}")
        options = argparse.Namespace(
            profile="full", case=[CASE], seed=seed, repeats=args.repeats, warmup=args.warmup,
            threads="both", resident=True, reference_archive=args.reference_archive, output=output,
        )
        run(options)
        reports.append(json.loads(output.read_text()))
        paths.append(str(output))
    result = summarize_reports(reports, target_speedup=args.target_speedup)
    result.update({
        "schema_version": 2,
        "case": reports[0]["cases"][0]["case"],
        "reference": reports[0]["reference"],
        "hardware": reports[0]["workers"]["current"]["hardware"],
        "base_seeds": args.seeds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "target_speedup_vs_sklearn_default": args.target_speedup,
        "target_operations": list(TARGET_OPERATIONS),
        "target_policy": (
            "Every retained timed pair for each named operation must reach the target ratio of "
            "sklearn-default time to current time on every dataset. A single paired-round miss "
            "fails the gate even when every median passes. The existing warmup count and "
            "interleaved sample order are unchanged. First-call compilation observations remain "
            "separate, and are not retained samples. Fit alone is reported separately; fresh "
            "fit_query includes preparation. Paired-round counts compare corresponding "
            "interleaved rounds; "
            "halves divide retained rounds chronologically (the first half takes an odd extra round). "
            "These ratios and counts are measurements, not confidence bounds."
        ),
        "reports": paths,
        "harness_sha256": {
            "knn_pipeline.py": sha256(Path(__file__).with_name("knn_pipeline.py").read_bytes()),
            "knn_acceleration.py": sha256(Path(__file__).read_bytes()),
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for operation, value in result["operations"].items():
        ratios = ", ".join(f"{s['speedup_vs_sklearn_default']:.2f}x" for s in value["seeds"])
        rounds = sum(s["paired_round_count"] for s in value["seeds"])
        passed = sum(s["paired_rounds_meeting_target"] for s in value["seeds"])
        worst = value["minimum_paired_round_speedup_vs_sklearn_default"]
        print(f"{operation}: median versus sklearn default: {ratios}; "
              f"paired rounds {passed}/{rounds}, worst {worst:.3f}x")
    print(f"implementation_quality_pass={result['implementation_quality_pass']} "
          f"source_integrity_pass={result['source_integrity_pass']} "
          f"all_target_operations_pass={result['all_target_operations_pass']}: {args.output}")
    # Missing the requested speedup remains explicit in the JSON and exit code;
    # do not select only the favorable seeds or substitute a weaker comparator.
    return 0 if (result["implementation_quality_pass"] and result["source_integrity_pass"]
                 and result["all_target_operations_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
