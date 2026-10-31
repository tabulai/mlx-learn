"""Broad, audited public-API kNN comparison against sklearn brute and auto.

Run all GPU work serially; never change production sources during a benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .knn_acceleration import TARGET_OPERATIONS
from .knn_broad_fixtures import make_fixture, workload_specs
from .knn_pipeline import ROOT, Worker, sha256, source_hashes

VARIANTS = ("current", "sklearn_default", "sklearn_1", "sklearn_auto")
OPERATIONS = ("fit", *TARGET_OPERATIONS)
SOURCE_MANIFEST = ROOT / "benchmarks/reference/knn-128-final-2026-10-30.manifest.json"


class BroadWorker(Worker):
    def __init__(self, variant, fixtures, *, source_root=None, current_mode="gpu"):
        if current_mode not in {"gpu", "auto"}:
            raise ValueError(f"Unknown current mode: {current_mode}")
        source_root = Path(source_root or ROOT / "src").resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source_root) + os.pathsep + str(ROOT)
        self.process = subprocess.Popen(
            [sys.executable, str(ROOT / "benchmarks/knn_broad_worker.py"),
             "--worker", variant, "--source-root", str(source_root),
             "--fixtures", str(fixtures), "--current-mode", current_mode], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        self.info = self.request({"op": "info"})


def request(worker, query):
    try:
        return worker.request(query)
    except Exception:
        return {"error": traceback.format_exc(), "audit": {"pass": False}}


def valid_time(sample):
    value = sample.get("ms")
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def summarize(samples, warmups, count):
    times = [sample["ms"] for sample in samples if valid_time(sample)]
    return {
        "median_ms": statistics.median(times) if times else None,
        "min_ms": min(times) if times else None, "max_ms": max(times) if times else None,
        "completed_samples": len(times), "expected_samples": count,
        "quality_pass": len(times) == count and all(s.get("audit", {}).get("pass", False) for s in samples),
        "warmup_quality_pass": all(s.get("audit", {}).get("pass", False) for s in warmups),
        "samples": samples, "warmup_samples": warmups,
    }


def compare(current, baseline, count, target):
    pairs = []
    for i, (a, b) in enumerate(zip(current["samples"], baseline["samples"], strict=True)):
        available = valid_time(a) and valid_time(b)
        ratio = b["ms"] / a["ms"] if available else None
        pairs.append({"round": i, "speedup": ratio,
                      "meets_target": available and ratio >= target,
                      "both_strict_audits_pass": bool(a.get("audit", {}).get("pass"))
                      and bool(b.get("audit", {}).get("pass"))})
    ratios = [p["speedup"] for p in pairs if p["speedup"] is not None]
    median_ratio = (baseline["median_ms"] / current["median_ms"]
                    if baseline["median_ms"] is not None and current["median_ms"] is not None else None)
    return {
        "ratio_definition": "sklearn milliseconds / current milliseconds",
        "ratio_of_medians": median_ratio,
        "median_of_paired_ratios": statistics.median(ratios) if ratios else None,
        "median_meets_target": median_ratio is not None and median_ratio >= target,
        "completed_pairs": len(ratios), "expected_pairs": count,
        "pairs_meeting_target": sum(p["meets_target"] for p in pairs),
        "all_pairs_meet_target": len(ratios) == count and all(p["meets_target"] for p in pairs),
        "minimum_pair_speedup": min(ratios) if ratios else None,
        "both_strict_quality_pass": current["quality_pass"] and baseline["quality_pass"]
        and current["warmup_quality_pass"] and baseline["warmup_quality_pass"],
        "pairs": pairs,
    }


def rotate(names, offset):
    offset %= len(names)
    return list(names[offset:]) + list(names[:offset])


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def hashes(paths):
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in paths}


def run(args):
    specs = workload_specs()
    if args.case:
        specs = [s for s in specs if s["case"]["name"] in args.case]
        missing = set(args.case) - {s["case"]["name"] for s in specs}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
    harnesses = [ROOT / "benchmarks" / name for name in (
        "knn_broad.py", "knn_broad_worker.py", "knn_broad_fixtures.py",
        "knn_pipeline.py", "knn_review_distributions.py", "knn_acceleration.py",
        "knn_backend_audit.py",
    )]
    before = source_hashes(ROOT / "src")
    manifest_path = args.source_manifest.resolve()
    frozen = json.loads(manifest_path.read_text())
    if before != frozen["files"]:
        raise RuntimeError(f"Production differs from the selected source manifest: {manifest_path}")
    report = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "variants": list(VARIANTS), "operations": list(OPERATIONS),
        "current_mode": args.current_mode,
        "backend_telemetry_version": 1,
        "target_operations": list(TARGET_OPERATIONS), "target_speedup": args.target,
        "repeats": args.repeats, "warmup": args.warmup, "planned_cases": len(specs),
        "source_manifest": str(manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT)
                               else manifest_path), "source_snapshot": frozen["archive"],
        "source_hashes_before": before, "harness_hashes_before": hashes(harnesses),
        "protocol": "Persistent isolated public NumPy API workers. All GPU/CPU timed calls are serial. "
        "Cold first fit/query is separate; every operation gets the stated count of warmups per "
        "variant. Variant starting order rotates by dataset and operation, and each round reverses "
        "it. Each call is synchronized and audited outside timing. fit_query includes a fresh fit "
        "and all first-query preparation. No result caching. Inputs are the same float32 arrays "
        "for all variants, with no rescaling or jitter. Real data use raw features and stratified "
        "holdouts. Synthetic inputs and real-data split seeds are preserved per fixture.",
        "accuracy_reference": "Exhaustive direct float64 differences on actual float32 coordinates; "
        "canonical distance/index order. Strict index and prediction mismatches remain failures. "
        "A separate query audit recognizes exact-distance-valid alternative tie choices; it never "
        "changes strict pass. Prediction mismatches on tied fixtures are marked ambiguous, not "
        "automatically accepted. Distance atol/rtol1e-6; existing prediction tolerance retained.",
        "two_x_policy": "Nine search/inference operations, excluding fit-only. Report both ratio "
        "of per-variant medians and EVERY retained paired call>=2x. Do not average away misses. "
        "Speed results and accuracy qualifications are separate; failures remain in raw records.",
        "cases": [],
    }
    save(report, args.output)
    workers = {}
    try:
        with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-broad-") as tmp:
            fixtures = Path(tmp)
            report["fixtures"] = []
            for i, spec in enumerate(specs):
                report["fixtures"].append(make_fixture(spec, fixtures))
                print(f"fixture {i+1}/{len(specs)} {spec['case']['name']}", flush=True)
            fixture_paths = [fixtures / f"{s['case']['name']}.npz" for s in specs]
            report["fixture_hashes_before"] = {p.name: sha256(p.read_bytes()) for p in fixture_paths}
            save(report, args.output)
            for variant in VARIANTS:
                workers[variant] = BroadWorker(variant, fixtures, current_mode=args.current_mode)
            report["workers"] = {name: w.info for name, w in workers.items()}
            report["initial_worker_source_integrity_pass"] = all(w.info["source_hashes"] == before for w in workers.values())
            for case_index, spec in enumerate(specs):
                entry = {"spec": spec, "case": spec["case"], "cold": {}, "operations": {}}
                report["cases"].append(entry)
                for name in rotate(VARIANTS, case_index):
                    entry["cold"][name] = request(workers[name], {"op": "cold", "case": spec["case"]})
                for op_index, operation in enumerate(OPERATIONS):
                    order = rotate(VARIANTS, case_index + op_index)
                    samples = {v: [] for v in VARIANTS}
                    warmups = {v: [] for v in VARIANTS}
                    rounds = []
                    for r in range(args.warmup + args.repeats):
                        names = order if r % 2 == 0 else list(reversed(order))
                        if r >= args.warmup:
                            rounds.append({"round": r-args.warmup, "order": names})
                        for name in names:
                            sample = request(workers[name], {"op": operation, "case": spec["case"]})
                            (samples if r >= args.warmup else warmups)[name].append(sample)
                    variants = {v: summarize(samples[v], warmups[v], args.repeats) for v in VARIANTS}
                    entry["operations"][operation] = {"variants": variants, "rounds": rounds,
                        "comparisons": {v: compare(variants["current"], variants[v], args.repeats, args.target)
                                        for v in VARIANTS if v != "current"}}
                save(report, args.output)
                query = entry["operations"]["kneighbors"]["comparisons"]
                print(f"case {case_index+1}/{len(specs)} {spec['case']['name']}: "
                      + str({v: r['ratio_of_medians'] for v, r in query.items()}), flush=True)
            report["worker_source_hashes_after"] = {v: request(w, {"op": "integrity"}) for v, w in workers.items()}
            report["worker_source_integrity_pass"] = report["initial_worker_source_integrity_pass"] and all(
                x == before for x in report["worker_source_hashes_after"].values())
            report["fixture_hashes_after"] = {p.name: sha256(p.read_bytes()) for p in fixture_paths}
            report["fixture_integrity_pass"] = report["fixture_hashes_after"] == report["fixture_hashes_before"]
            report["quality_by_variant"] = {v: all(
                c["cold"][v].get("audit", {}).get("pass", False) and all(
                    e["variants"][v]["quality_pass"] and e["variants"][v]["warmup_quality_pass"]
                    for e in c["operations"].values()) for c in report["cases"]
            ) for v in VARIANTS}
            report["gate_by_comparator"] = {}
            for v in VARIANTS[1:]:
                cells = [c["operations"][op]["comparisons"][v] for c in report["cases"] for op in TARGET_OPERATIONS]
                report["gate_by_comparator"][v] = {
                    "median_cells_meeting_target": sum(c["median_meets_target"] for c in cells),
                    "median_cell_count": len(cells),
                    "paired_calls_meeting_target": sum(c["pairs_meeting_target"] for c in cells),
                    "paired_call_count": len(cells)*args.repeats,
                    "all_medians_meet_target": all(c["median_meets_target"] for c in cells),
                    "every_pair_meets_target": all(c["all_pairs_meet_target"] for c in cells),
                }
            report["status"] = "complete"
    except Exception:
        report["error"] = traceback.format_exc()
        report["status"] = "error"
    finally:
        for worker in workers.values():
            worker.close()
        report["source_hashes_after"] = source_hashes(ROOT / "src")
        report["source_integrity_pass"] = before == report["source_hashes_after"]
        report["harness_hashes_after"] = hashes(harnesses)
        report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["accuracy_and_integrity_pass"] = report["status"] == "complete" and all(report.get(k, False) for k in (
            "worker_source_integrity_pass", "fixture_integrity_pass", "source_integrity_pass", "harness_integrity_pass",
        )) and report.get("quality_by_variant", {}).get("current", False)
        save(report, args.output)
    print("accuracy_and_integrity_pass=", report["accuracy_and_integrity_pass"], flush=True)
    print(report.get("gate_by_comparator", {}), flush=True)
    return 0 if report["accuracy_and_integrity_pass"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--target", type=float, default=2.0)
    p.add_argument("--current-mode", choices=("gpu", "auto"), default="gpu",
                   help="Current mlxlearn policy; gpu preserves the historical forced-GPU benchmark")
    p.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST,
                   help="Frozen production manifest required to match the benchmarked source")
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/knn-broad-2026-10-30.json")
    p.add_argument("--list-cases", action="store_true")
    args = p.parse_args()
    if args.list_cases:
        print(json.dumps(workload_specs(), indent=2))
        return
    if args.repeats < 1 or args.warmup < 0 or not math.isfinite(args.target) or args.target <= 0:
        p.error("invalid repeat/warmup/target")
    start = time.perf_counter()
    code = run(args)
    print(f"elapsed_seconds={time.perf_counter()-start:.3f}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
