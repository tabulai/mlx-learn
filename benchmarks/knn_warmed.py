"""Continuous one-second warmup followed by 25 audited KNN calls.

    python -m benchmarks.knn_warmed

This additional protocol measures warmed sustained use. It does not replace
cold, short-warmup or interleaved observations, and changes no runtime policy.
Every variant receives the same minimum continuous warmup duration per operation.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

from .knn_acceleration import CASE, TARGET_OPERATIONS, summarize_reports
from .knn_pipeline import CASES, ROOT, Worker, extract_reference, make_fixture, sha256

REFERENCE = ROOT / "benchmarks/reference/knn-pre-v2-2026-10-28.zip"


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("warmup seconds must be finite and positive")
    return seconds


def run_seed(args, seed, order):
    case = next(case for case in CASES if case.name == CASE)
    report = {
        "schema_version": 1,
        "protocol": "continuous_time_warmup",
        "seed": seed,
        "dataset_seed": seed + CASES.index(case),
        "repeats": args.repeats,
        "warmup": 0,
        "warmup_seconds": args.warmup_seconds,
        "platform": platform.platform(),
        "variant_order": order,
        "timing_policy": "Consecutive calls inside each persistent isolated worker; no IPC between "
        "samples, synchronized every call, every output audited outside timing. "
        "Each operation/variant first runs continuously for at least the configured warmup_seconds. "
        "All warmup outputs are audited. Full public validation and NumPy I/O included; no output caching. "
        "Whole variant blocks run in a fixed order rotated across datasets.",
        "cold_policy": "First fit/query in each worker after imports, before steady blocks; "
        "startup and input construction excluded. Reported separately.",
        "fit_policy": "Fit measures eager work; fit_query constructs and queries a fresh estimator "
        "for every sample, including deferred preparation.",
        "query_tolerances": {"atol": 1e-6, "rtol": 1e-6, "indices": "exact equality"},
        "accuracy_reference": "sklearn brute float64 on serialized float32 inputs; direct float64 "
        "selected distances and derived predictions, unchanged from mixed audit.",
        "cases": [{"case": asdict(case), "variants": {name: {} for name in order}}],
    }
    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-warmed-") as folder:
        folder = Path(folder)
        reference = folder / "reference"
        report["reference"] = extract_reference(args.reference_archive, reference)
        fixtures = folder / "fixtures"
        fixtures.mkdir()
        report["fixtures"] = [make_fixture(case, report["dataset_seed"], fixtures)]
        workers = {}
        try:
            for name in order:
                workers[name] = Worker(
                    name, reference if name == "reference" else ROOT / "src", fixtures
                )
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            variants = report["cases"][0]["variants"]
            for name in order:
                variants[name]["cold"] = workers[name].request({"op": "cold", "case": asdict(case)})
            for operation in ("fit", *TARGET_OPERATIONS):
                for name in order:
                    variants[name][operation] = workers[name].request(
                        {
                            "op": "block",
                            "operation": operation,
                            "case": asdict(case),
                            "repeats": args.repeats,
                            "warmup": 0,
                            "warmup_seconds": args.warmup_seconds,
                        }
                    )
            report["source_integrity_pass"] = all(
                worker.request({"op": "integrity"}) == worker.info["source_hashes"]
                for worker in workers.values()
            )
            report["quality_by_variant"] = {
                name: values["cold"]["audit"]["pass"]
                and all(
                    entry["quality_pass"] and all(a["pass"] for a in entry["warmup_audits"])
                    for operation, entry in values.items()
                    if operation != "cold"
                )
                for name, values in variants.items()
            }
            report["quality_pass"] = all(report["quality_by_variant"].values())
            report["implementation_quality_pass"] = all(
                report["quality_by_variant"][name] for name in ("current", "reference")
            )
            report["warmup_policy_pass"] = all(
                entry["warmup_min_seconds"] == args.warmup_seconds
                and entry["warmup_elapsed_seconds"] >= args.warmup_seconds
                and entry["warmup_count"] == len(entry["warmup_audits"])
                and len(entry["samples"]) == args.repeats
                for values in variants.values()
                for operation, entry in values.items() if operation != "cold"
            )
        finally:
            for worker in workers.values():
                worker.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup-seconds", type=positive_seconds, default=1.0)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/results/knn-v2-warmed-2026-10-28.json"
    )
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("require positive repeats and distinct seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    variants = ["current", "reference", "sklearn_1", "sklearn_default"]
    reports, paths = [], []
    for index, seed in enumerate(args.seeds):
        rotation = index % len(variants)
        report = run_seed(args, seed, variants[rotation:] + variants[:rotation])
        path = args.output.with_name(f"{args.output.stem}-seed{seed}{args.output.suffix}")
        path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        paths.append(str(path))
        print(
            f"Continuously warmed dataset {report['dataset_seed']}: quality={report['quality_pass']}, "
            f"source_integrity={report['source_integrity_pass']}: {path}",
            flush=True,
        )
    summary = summarize_reports(reports, target_speedup=2, paired_rounds=False)
    summary["warmup_policy_pass"] = all(report["warmup_policy_pass"] for report in reports)
    summary.update(
        {
            "schema_version": 1,
            "protocol": "continuous_time_warmup",
            "reports": paths,
            "case": reports[0]["cases"][0]["case"],
            "base_seeds": args.seeds,
            "repeats": args.repeats,
            "warmup": 0,
            "warmup_seconds": args.warmup_seconds,
            "hardware": reports[0]["workers"]["current"]["hardware"],
            "reference": reports[0]["reference"],
            "timing_policy": reports[0]["timing_policy"],
            "target_speedup_vs_sklearn_default": 2,
            "target_operations": list(TARGET_OPERATIONS),
            "target_policy": "Every target operation must reach 2x default-sklearn median on every "
            "dataset. Fit-only is reported separately. Ratios are descriptive, not "
            "confidence bounds. Continuous-warmup results do not replace cold, short-warmup or mixed-workload results.",
            "mixed_protocol_report": str(
                ROOT / "benchmarks/results/knn-v2-acceleration-2026-10-28.json"
            ),
            "harness_sha256": {
                name: sha256(Path(__file__).with_name(name).read_bytes())
                for name in ("knn_pipeline.py", "knn_acceleration.py", "knn_warmed.py")
            },
        }
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for operation, values in summary["operations"].items():
        print(
            operation,
            "default-sklearn ratios:",
            "/".join(f"{seed['speedup_vs_sklearn_default']:.2f}x" for seed in values["seeds"]),
        )
    print("all_target_operations_pass=", summary["all_target_operations_pass"])
    return (
        0
        if (
            summary["implementation_quality_pass"]
            and summary["warmup_policy_pass"]
            and summary["source_integrity_pass"]
            and summary["all_target_operations_pass"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
