"""Audited automatic dispatch, current forced GPU and preceding GPU comparison.

All timed work is serial. No estimator hooks, output caches, oracle changes or
fixture transforms are used. Freeze current source before running this harness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .knn_128_control import _hash_files, _valid_time
from .knn_acceleration import TARGET_OPERATIONS
from .knn_backend_audit import backend_counts, validate_record_backend
from .knn_broad import BroadWorker, request, rotate, save
from .knn_broad_fixtures import make_fixture, workload_specs
from .knn_gaussian_fresh_control import select_specs as preservation_specs
from .knn_order import phase_diagnostics, placement, round_order, williams_orders
from .knn_pipeline import ROOT, extract_reference, sha256, source_hashes
from .knn_reference_runtime import (
    install_reference_runtime,
    reference_runtime_unchanged,
    validate_reference_worker,
)

VARIANTS = ("current", "current_b", "current_gpu", "previous_gpu")
WORKER_MODES = {"current": "auto", "current_b": "auto",
                "current_gpu": "gpu", "previous_gpu": "gpu"}
EXTENDED_VARIANTS = ("current", "current_b", "current_gpu", "previous_auto", "previous_gpu")
EXTENDED_WORKER_MODES = {**WORKER_MODES, "previous_auto": "auto"}
OPERATIONS = tuple(TARGET_OPERATIONS)
PRESERVATION_OPERATIONS = ("kneighbors", "fit_query")
PREVIOUS = ROOT / "benchmarks/reference/knn-small-final-2026-10-30.zip"
TARGET_FAMILIES = frozenset({
    "shape_single_8_k5", "shape_single_32_k5", "real_iris_raw_k5",
    "real_wine_raw_k5", "real_breast_cancer_raw_k5", "real_digits_raw_k5",
})


def _sample_pass(sample):
    try:
        validate_record_backend(sample, "auto" if sample.get("execution_policy") == "auto" else "gpu")
    except ValueError:
        return False
    return bool(sample.get("audit", {}).get("pass")) and "error" not in sample


def _summarize(samples, expected_count):
    values = [sample["ms"] for sample in samples if _valid_time(sample)]
    return {
        "median_ms": statistics.median(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "completed_samples": len(values),
        "expected_samples": expected_count,
        "quality_pass": len(samples) == expected_count
        and len(values) == expected_count and all(_sample_pass(sample) for sample in samples),
        "samples": samples,
    }


def _warmup_pass(block, duration):
    audits = block.get("warmup_audits", [])
    return (
        block.get("warmup_min_seconds") == duration
        and block.get("warmup_elapsed_seconds", -1) >= duration
        and block.get("warmup_count", 0) == len(audits)
        and len(audits) > 0
        and all(audit.get("pass", False) for audit in audits)
        and bool(block.get("warmup_quality_pass"))
        and bool(block.get("quality_pass"))
        and _summarize(block.get("samples", []), 1)["quality_pass"]
    )


def _paired_ratios(samples, comparator, expected_count):
    values = []
    pairs = []
    for pair, (current, other) in enumerate(zip(samples["current"], samples[comparator], strict=True)):
        valid = _valid_time(current) and _valid_time(other)
        ratio = other["ms"] / current["ms"] if valid else None
        pairs.append({"pair": pair, "ratio": ratio,
                      "quality_pass": valid and _sample_pass(current) and _sample_pass(other)})
        if ratio is not None:
            values.append(ratio)
    return {
        "ratio_definition": f"{comparator} milliseconds / current milliseconds",
        "paired_ratio_median": statistics.median(values) if values else None,
        "paired_ratio_min": min(values) if values else None,
        "paired_ratio_max": max(values) if values else None,
        "completed_pairs": len(values),
        "expected_pairs": expected_count,
        "quality_pass": len(pairs) == expected_count and all(pair["quality_pass"] for pair in pairs),
        "pairs": pairs,
    }



def operations_for(spec):
    """Cover every target API, plus search/fresh setup on preservation fixtures."""
    return OPERATIONS if spec["family"] in TARGET_FAMILIES else PRESERVATION_OPERATIONS


def select_specs(names=None):
    all_specs = ([spec for spec in workload_specs() if spec["family"] in TARGET_FAMILIES]
                 + preservation_specs())
    if names:
        missing = set(names) - {spec["case"]["name"] for spec in all_specs}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
        return [spec for spec in all_specs if spec["case"]["name"] in names]
    return all_specs


def family_summary(cases, variant_names=VARIANTS):
    families = defaultdict(list)
    for case in cases:
        families[case["spec"]["family"]].append(case)
    result = []
    for name, members in families.items():
        operations = {}
        for operation in operations_for(members[0]["spec"]):
            entries = [c["experiments"][operation] for c in members]
            variants = {}
            for variant in variant_names:
                values = [e["variants"][variant]["median_ms"] for e in entries]
                variants[variant] = {"min": min(values), "max": max(values)} if all(
                    v is not None for v in values) else {"min": None, "max": None}
            comparisons = {}
            median_comparisons = {}
            for other in variant_names[1:]:
                values = [e["comparisons"][f"current_vs_{other}"]["paired_ratio_median"]
                          for e in entries]
                comparisons[other] = {"min": min(values), "max": max(values)} if all(
                    v is not None for v in values) else {"min": None, "max": None}
                values = [e["comparisons"][f"current_vs_{other}"]["ratio_of_medians"]
                          for e in entries]
                median_comparisons[other] = {"min": min(values), "max": max(values)} if all(
                    v is not None for v in values) else {"min": None, "max": None}
            operations[operation] = {
                "median_ms_ranges": variants,
                "median_paired_ratio_ranges": comparisons,
                "ratio_of_medians_ranges": median_comparisons,
            }
        result.append({"family": name, "seeds": [c["spec"]["seed"] for c in members],
                       "operations": operations})
    return result


def routing_summary(cases, variant_names=VARIANTS):
    result = {}
    for variant in variant_names:
        phases = {"cold": [], "warmup": [], "probe": [], "retained": []}
        for case in cases:
            phases["cold"].append(case["cold"][variant])
            for experiment in case["experiments"].values():
                warmup = experiment["warmup"][variant]
                phases["warmup"].extend(warmup.get("warmup_backend_records", []))
                phases["probe"].extend(warmup.get("samples", []))
                phases["retained"].extend(experiment["variants"][variant]["samples"])
        result[variant] = {phase: backend_counts(records) for phase, records in phases.items()}
    return result


def run(args):
    specs = select_specs(args.case)
    include_auto = args.previous_auto_archive is not None
    variants = EXTENDED_VARIANTS if include_auto else VARIANTS
    modes = EXTENDED_WORKER_MODES if include_auto else WORKER_MODES
    manifest = json.loads(args.source_manifest.read_text())
    before = source_hashes(ROOT / "src")
    if before != manifest["files"]:
        raise RuntimeError("Current source differs from --source-manifest")
    harness_paths = [ROOT / "benchmarks" / name for name in (
        "knn_auto_control.py", "knn_backend_audit.py", "knn_gaussian_fresh_control.py", "knn_128_control.py", "knn_broad.py",
        "knn_broad_worker.py", "knn_broad_fixtures.py", "knn_pipeline.py",
        "knn_review_distributions.py", "knn_acceleration.py", "knn_reference_runtime.py",
        "knn_freeze_runtime.py", "knn_freeze.py", "knn_order.py",
    )]
    current_archive = Path(manifest["archive"]["path"])
    if not current_archive.is_absolute():
        current_archive = ROOT / current_archive
    if sha256(current_archive.read_bytes()) != manifest["archive"]["sha256"]:
        raise RuntimeError("Current source archive hash differs from --source-manifest")
    archives = {"previous_gpu": args.previous_archive}
    if include_auto:
        archives["previous_auto"] = args.previous_auto_archive
    runtime_archives = {"previous_auto": args.previous_auto_runtime} if include_auto else {}
    archive_paths = [args.source_manifest, current_archive]
    for archive in runtime_archives.values():
        archive_paths.extend((archive, archive.with_suffix(".manifest.json")))
    for archive in archives.values():
        archive_paths.extend((archive, archive.with_suffix(".manifest.json")))
    report = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "planned_cases": len(specs), "pairs": args.pairs,
        "warmup_seconds_per_variant_operation": args.warmup_seconds,
        "operations": list(OPERATIONS), "variants": list(variants),
        "worker_modes": modes, "backend_telemetry_version": 1,
        "order_policy": args.order, "order_telemetry_version": 1,
        "order_design": "Legacy alternates one order and its reverse. Balanced uses a Williams "
        "complete block: five workers have ten rows, balancing every position and each directed "
        "within-round predecessor twice per block. First 30 of 31 rounds are balanced; round 31 "
        "is retained. Cross-round predecessors and parent timing context are recorded separately. "
        "No clock state is inferred from these descriptive phase diagnostics.",
        "order_block": williams_orders(variants) if args.order == "balanced" else None,
        "operations_by_case": {spec["case"]["name"]: list(operations_for(spec)) for spec in specs},
        "planned_case_operation_count": sum(len(operations_for(spec)) for spec in specs),
        "source_manifest": str(args.source_manifest.resolve()), "source_snapshot": manifest["archive"],
        "reference_archives": {name: str(path.resolve()) for name, path in archives.items()},
        "reference_runtime_archives": {name: str(path.resolve()) for name, path in runtime_archives.items()},
        "reference_roles": {
            **({"previous_auto": "Preceding automatic source with its archived, ABI-matched native binary."} if include_auto else {}),
            "previous_gpu": "Preceding small-workload GPU fix, including Gaussian/clustered/offset and precision gains.",
            "current_gpu": "Identical current source with explicitly forced GPU; isolates automatic routing.",
            "current_b": "Identical current source in a separate worker; A/A timing control.",
        },
        "dataset_policy": "Unchanged broad fixtures for single 8D/32D queries and raw Iris, Wine, "
                          "Breast Cancer and Digits: 18 cases, all nine target operations. Preservation "
                          "uses all 24 broad distribution fixtures plus six expanded Gaussian seeds "
                          "from the preceding control: 30 cases, query and fresh-fit operations. "
                          "Default total: 48 cases and 222 case-operations. No fixture transforms.",
        "source_hashes_before": before, "harness_hashes_before": _hash_files(harness_paths),
        "archive_hashes_before": _hash_files(archive_paths),
        "timing_policy": f"{len(variants)} persistent isolated public-API workers: current automatic routing, "
        "an identical automatic A/A control, current forced GPU, preceding frozen forced GPU, "
        "and (when selected) preceding automatic source with its exact archived native binary. "
        "All calls serial; continuous per-worker warmup, then "
        f"{args.order} interleaved paired rounds. Starting labels rotate by dataset and operation. "
        "Every call includes validation, NumPy conversion and synchronization. IPC and audits "
        "are outside timing. Ratios are descriptive; A/A exposes process/timing variation.",
        "warmup_policy": "Default continuous warmup is 0.25 seconds (the preceding small control used 1 second); "
        "this is configurable and absolute timings must not be compared across those runs. "
        "Each worker/operation warms continuously for at least the requested "
        "duration. Its one post-warmup probe is audited and retained separately, excluded from "
        "paired ratios. All warmup audits and probe samples remain in the report.",
        "fit_query_policy": "Fresh fit plus first query each call includes all deferred preparation. "
        "Repeated kneighbors reuses the estimator established by cold. No result caching.",
        "cold_policy": "First fit/query per fixture in a persistent worker, not fresh-process startup. "
        "Earlier fixtures may populate compiler caches. Cold times remain separate.",
        "accuracy_reference": "Identical knn_broad_fixtures: exhaustive direct float64 differences "
        "on serialized float32 coordinates, canonical distance/index ordering. Exact indices; "
        "distance atol/rtol 1e-6. No centering, rescaling or jitter of benchmark inputs.",
        "acceptance_policy": "Every automatic, A/A, forced-GPU and frozen-reference cold, warmup, probe and retained "
        "audit must pass, with complete valid timing counts and source/fixture/harness/archive "
        "integrity. Speedups do not alter accuracy gates or filter failed records.",
        "cases": [],
    }
    save(report, args.output)
    workers = {}
    try:
        with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-auto-control-") as tmp:
            temp = Path(tmp)
            roots = {"current": ROOT / "src", "current_b": ROOT / "src",
                     "current_gpu": ROOT / "src", "previous_gpu": temp / "previous_gpu",
                     **({"previous_auto": temp / "previous_auto"} if include_auto else {})}
            references = {name: extract_reference(archive, roots[name])
                          for name, archive in archives.items()}
            report["references"] = references
            runtimes = {name: install_reference_runtime(archive, roots[name], references[name])
                        for name, archive in runtime_archives.items()}
            report["reference_runtimes"] = runtimes
            fixtures = temp / "fixtures"
            fixtures.mkdir()
            report["fixtures"] = [make_fixture(spec, fixtures) for spec in specs]
            fixture_paths = [fixtures / f"{s['case']['name']}.npz" for s in specs]
            report["fixture_hashes_before"] = {p.name: sha256(p.read_bytes()) for p in fixture_paths}
            for name in variants:
                # Separate source roots and explicit policies distinguish implementations.
                workers[name] = BroadWorker("current", fixtures, source_root=roots[name],
                                            current_mode=modes[name])
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            for name, runtime in runtimes.items():
                validate_reference_worker(report["workers"][name], runtime)
            expected_hashes = {"current": before, "current_b": before, "current_gpu": before,
                               **{name: value["files"] for name, value in references.items()}}
            report["worker_initial_source_match"] = {
                name: worker.info["source_hashes"] == expected_hashes[name]
                for name, worker in workers.items()
            }
            if not all(report["worker_initial_source_match"].values()):
                raise RuntimeError("A worker loaded source that does not match its manifest")
            for case_index, spec in enumerate(specs):
                case = spec["case"]
                entry = {"spec": spec, "case": case, "cold": {}, "experiments": {}}
                report["cases"].append(entry)
                for name in rotate(variants, case_index):
                    entry["cold"][name] = request(workers[name], {"op": "cold", "case": case})
                for operation_index, operation in enumerate(operations_for(spec)):
                    order = rotate(variants, case_index + operation_index)
                    experiment = {"operation": operation, "starting_order": order,
                                  "warmup": {}, "rounds": [], "variants": {}}
                    entry["experiments"][operation] = experiment
                    warmup_completed = {}
                    for name in order:
                        experiment["warmup"][name] = request(workers[name], {
                            "op": "block", "operation": operation, "case": case,
                            "warmup": 0, "warmup_seconds": args.warmup_seconds, "repeats": 1,
                        })
                        warmup_completed[name] = time.perf_counter_ns()
                    experiment["warmup_probe_completed_at_ns"] = warmup_completed
                    samples = {name: [] for name in variants}
                    previous_request = order[-1]
                    for pair in range(args.pairs):
                        pair_order = round_order(order, pair, args.order)
                        positions = placement(pair_order, previous_request)
                        experiment["rounds"].append({"pair": pair, "order": pair_order,
                                                     "placement": positions})
                        for name in pair_order:
                            requested = time.perf_counter_ns()
                            sample = request(workers[name], {"op": operation, "case": case})
                            sample["order_context"] = {**positions[name], "pair": pair,
                                "requested_at_ns": requested, "completed_at_ns": time.perf_counter_ns(),
                                "seconds_since_warmup_probe": (requested - warmup_completed[name]) / 1e9}
                            samples[name].append(sample)
                        previous_request = pair_order[-1]
                    experiment["variants"] = {
                        name: _summarize(samples[name], args.pairs) for name in variants
                    }
                    experiment["comparisons"] = {
                        f"current_vs_{other}": _paired_ratios(samples, other, args.pairs)
                        for other in variants[1:]
                    }
                    for other in variants[1:]:
                        current_ms = experiment["variants"]["current"]["median_ms"]
                        other_ms = experiment["variants"][other]["median_ms"]
                        experiment["comparisons"][f"current_vs_{other}"]["ratio_of_medians"] = (
                            other_ms / current_ms if current_ms is not None and other_ms is not None else None
                        )
                    experiment["quality_by_variant"] = {
                        name: _warmup_pass(experiment["warmup"][name], args.warmup_seconds)
                        and experiment["variants"][name]["quality_pass"] for name in variants
                    }
                    experiment["phase_diagnostics"] = phase_diagnostics(experiment, variants)
                save(report, args.output)
                print(f"case {case_index+1}/{len(specs)} {case['name']}: " + str({
                    op: {name: value["paired_ratio_median"]
                         for name, value in entry["experiments"][op]["comparisons"].items()}
                    for op in operations_for(spec)
                }), flush=True)
            report["worker_source_hashes_after"] = {
                name: request(worker, {"op": "integrity"}) for name, worker in workers.items()
            }
            report["source_integrity_pass"] = all(report["worker_initial_source_match"].values()) and all(
                report["worker_source_hashes_after"][name] == expected_hashes[name] for name in variants
            )
            report["reference_runtime_integrity_pass"] = all(
                reference_runtime_unchanged(roots[name], runtime) for name, runtime in runtimes.items()
            )
            report["fixture_hashes_after"] = {p.name: sha256(p.read_bytes()) for p in fixture_paths}
            report["fixture_integrity_pass"] = report["fixture_hashes_before"] == report["fixture_hashes_after"]
            report["quality_by_variant"] = {
                name: all(_sample_pass(entry["cold"][name]) and all(
                    experiment["quality_by_variant"][name] for experiment in entry["experiments"].values()
                ) for entry in report["cases"]) for name in variants
            }
            report["quality_pass"] = all(report["quality_by_variant"].values())
            report["complete_case_count_pass"] = len(report["cases"]) == len(specs) and all(
                set(entry["experiments"]) == set(operations_for(entry["spec"])) for entry in report["cases"]
            )
            report["families"] = family_summary(report["cases"], variants)
            report["routing"] = routing_summary(report["cases"], variants)
            report["status"] = "complete"
    except Exception:
        report["error"] = traceback.format_exc()
        report["status"] = "error"
    finally:
        for worker in workers.values():
            worker.close()
        report["source_hashes_after"] = source_hashes(ROOT / "src")
        report["current_source_integrity_pass"] = before == report["source_hashes_after"]
        report["harness_hashes_after"] = _hash_files(harness_paths)
        report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
        report["archive_hashes_after"] = _hash_files(archive_paths)
        report["archive_integrity_pass"] = report["archive_hashes_before"] == report["archive_hashes_after"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["acceptance_pass"] = report["status"] == "complete" and all(report.get(key, False) for key in (
            "quality_pass", "complete_case_count_pass", "source_integrity_pass", "current_source_integrity_pass",
            "fixture_integrity_pass", "harness_integrity_pass", "archive_integrity_pass",
            "reference_runtime_integrity_pass",
        ))
        save(report, args.output)
    print(f"acceptance_pass={report['acceptance_pass']}: {args.output}", flush=True)
    return 0 if report["acceptance_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup-seconds", type=float, default=0.25)
    parser.add_argument("--order", choices=("legacy", "balanced"), default="legacy",
                        help="Legacy preserves historical order; balanced rotates positions/predecessors")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--previous-archive", "--previous-gpu-archive", type=Path, default=PREVIOUS)
    parser.add_argument("--previous-auto-archive", type=Path)
    parser.add_argument("--previous-auto-runtime", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps(select_specs(args.case), indent=2))
        return
    if (args.previous_auto_archive is None) != (args.previous_auto_runtime is None):
        parser.error("--previous-auto-archive and --previous-auto-runtime must be supplied together")
    if args.source_manifest is None or args.output is None:
        parser.error("--source-manifest and --output are required for a benchmark run")
    if args.pairs < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        parser.error("require positive pairs and finite positive warmup seconds")
    if args.case and len(set(args.case)) != len(args.case):
        parser.error("case names must be distinct")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
