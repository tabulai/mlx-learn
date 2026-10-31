"""Fresh-model first/second-query latency in initialized isolated processes."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .knn_backend_audit import validate_record_backend
from .knn_broad import hashes, request, rotate, save
from .knn_broad_fixtures import make_fixture
from .knn_cold import VARIANTS, select_specs
from .knn_order import placement, round_order
from .knn_pipeline import ROOT, Worker, extract_reference, sha256, source_hashes
from .knn_reference_runtime import (
    current_abi,
    install_reference_runtime,
    reference_runtime_unchanged,
    validate_reference_worker,
)

METRICS = ("fit_ms", "first_query_ms", "fit_first_query_ms", "second_query_ms")


class FirstQueryWorker(Worker):
    def __init__(self, variant, fixtures, source_root):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source_root) + os.pathsep + str(ROOT)
        started = time.perf_counter_ns()
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "benchmarks/knn_first_query_worker.py"),
             "--variant", variant, "--source-root", str(source_root), "--fixtures", str(fixtures)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        try:
            message = json.loads(self.process.stdout.readline())
            if message.get("kind") != "initialized":
                raise ValueError("First-query worker did not send its import/setup record")
            self.setup = {**message["setup"],
                          "parent_process_start_to_initialized_ms": (time.perf_counter_ns() - started) / 1e6}
        except Exception:
            self.close()
            raise


def valid(sample):
    return "error" not in sample and all(isinstance(sample.get("timings", {}).get(m), (int, float))
        and math.isfinite(sample["timings"][m]) and sample["timings"][m] > 0 for m in METRICS)


def summarize(samples, expected):
    completed = [s for s in samples if valid(s)]
    scopes = {"all_rounds": samples, "first_model_for_fixture": samples[:1],
              "subsequent_fresh_models": samples[1:]}
    return {"expected_samples": expected, "completed_samples": len(completed),
            "quality_pass": len(completed) == len(samples) == expected
            and all(s[phase]["audit"].get("pass", False) for s in completed for phase in ("first", "second")),
            "metrics": {scope: {m: {
                "median_ms": statistics.median(s["timings"][m] for s in selected if valid(s))
                if any(valid(s) for s in selected) else None,
                "min_ms": min(s["timings"][m] for s in selected if valid(s))
                if any(valid(s) for s in selected) else None,
                "max_ms": max(s["timings"][m] for s in selected if valid(s))
                if any(valid(s) for s in selected) else None,
                "completed_samples": sum(valid(s) for s in selected), "retained_samples": len(selected),
            } for m in METRICS} for scope, selected in scopes.items()}, "samples": samples}


def comparisons(variants):
    current = variants["current"]
    output = {}
    for name, other in variants.items():
        if name == "current":
            continue
        output[name] = {}
        for metric in METRICS:
            a = current["metrics"]["all_rounds"][metric]["median_ms"]
            b = other["metrics"]["all_rounds"][metric]["median_ms"]
            paired = [y["timings"][metric] / x["timings"][metric] if valid(x) and valid(y) else None
                      for x, y in zip(current["samples"], other["samples"], strict=True)]
            present = [x for x in paired if x is not None]
            output[name][metric] = {"ratio_of_medians": b / a if a is not None and b is not None else None,
                                   "median_paired_ratio": statistics.median(present) if present else None,
                                   "paired_ratios": paired}
    return output


def validate(report, *, expected_cases=18, expected_repeats=30):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(report["status"] == "complete", "Incomplete first-query evidence")
    require(tuple(report["variants"]) in (VARIANTS, (*VARIANTS, "previous_auto")), "Wrong variants")
    require(report["metrics"] == list(METRICS), "Wrong first-query metrics")
    require(expected_cases is None or report["planned_cases"] == expected_cases, "Wrong case count")
    require(expected_repeats is None or report["repeats"] == expected_repeats, "Wrong repeat count")
    require(report["planned_cases"] == len(report["cases"]), "Incomplete first-query cases")
    names = [c["case"]["name"] for c in report["cases"]]
    require(len(set(names)) == len(names), "Duplicate first-query fixture")
    require([c["spec"] for c in report["cases"]] == select_specs(names), "Changed fixtures")
    fixtures = {f["case"]["name"]: f for f in report["fixtures"]}
    require(set(fixtures) == set(names), "Missing fixture metadata")
    for stem in ("source", "harness", "archive", "native", "fixture"):
        require(report[f"{stem}_hashes_before"] == report[f"{stem}_hashes_after"]
                and report[f"{stem}_integrity_pass"], f"First-query {stem} changed")
    require(report["reference_runtime_integrity_pass"], "Reference runtime changed")
    repeats, variants = report["repeats"], report["variants"]
    pids = set()
    for name in variants:
        setup, worker = report["setup"][name], report["workers"][name]
        require(setup["model_prewarm_count"] == setup["query_prewarm_count"] == 0, "Worker was prewarmed")
        require(not any(setup["loaded_modules_before_imports"].values()), "Imports preceded startup record")
        require(setup["pid"] not in pids, "Variants shared a worker")
        pids.add(setup["pid"])
        require(worker["model_count"] == repeats * len(names), "Missing fresh models")
        require(worker["fixture_round_counts"] == dict.fromkeys(names, repeats), "Wrong worker round counts")
        expected = report["reference"]["files"] if name == "previous_auto" else report["source_hashes_before"]
        require(worker["source_hashes"] == expected, "Wrong worker source")
        require(all(worker["setup"][k] == v for k, v in setup.items() if k != "parent_process_start_to_initialized_ms"),
                "Worker setup record changed")
        if not name.startswith("sklearn"):
            native = worker["native_cpu_extension"]
            binary = native["binary"]["sha256"] if native["available"] else None
            require(binary == report["native_hashes_before"][name], "Wrong native binary")
            require((native["source"]["sha256"] if native["source"] else None)
                    == expected.get("mlxlearn/_kernels/_neighbors_cpu_native.cpp"), "Wrong native C++ source")
        if name == "previous_auto":
            validate_reference_worker(worker, report["reference_runtime"])
    previous = None
    for ci, case in enumerate(report["cases"]):
        require(set(case["variants"]) == set(variants), "Missing variant")
        require(len(case["rounds"]) == repeats, "Missing fresh-model round")
        require(case["case"] == case["spec"]["case"], "Changed case metadata")
        require(fixtures[case["case"]["name"]]["fixture_sha256"] ==
                report["fixture_hashes_before"][case["case"]["name"] + ".npz"], "Wrong fixture hash")
        for ri, row in enumerate(case["rounds"]):
            order = round_order(rotate(variants, ci), ri, "balanced")
            require(row == {"round": ri, "order": order, "placement": placement(order, previous)},
                    "Wrong fresh-model order/position")
            previous = order[-1]
        for name, stats in case["variants"].items():
            require(len(stats["samples"]) == repeats and stats == summarize(stats["samples"], repeats),
                    "Invalid fresh-model summary")
            for ri, sample in enumerate(stats["samples"]):
                if "error" in sample:
                    continue
                require(valid(sample), "Invalid first-query timing")
                require(sample["round"] == ri and sample["model_number"] == ci * repeats + ri + 1,
                        "Model was reused or skipped")
                require(sample["first_model_in_worker"] == (ci == ri == 0)
                        and sample["first_model_for_fixture"] == (ri == 0), "Incorrect first-model flags")
                times = sample["timings"]
                require(math.isclose(times["fit_first_query_ms"], times["fit_ms"] + times["first_query_ms"], rel_tol=1e-10),
                        "First-query preparation missing from combined timing")
                for phase in ("first", "second"):
                    validate_record_backend(sample[phase], "auto", is_sklearn=name.startswith("sklearn"))
        require(case["comparisons"] == comparisons(case["variants"]), "Wrong first-query comparisons")
    quality = {name: all(c["variants"][name]["quality_pass"] for c in report["cases"]) for name in variants}
    timing = all(c["variants"][name]["completed_samples"] == repeats for c in report["cases"] for name in variants)
    require(report["quality_by_variant"] == quality and report["timing_count_pass"] == timing, "Wrong quality/count gate")
    require(report["acceptance_pass"] == (timing and all(quality[n] for n in variants if not n.startswith("sklearn"))),
            "Wrong first-query acceptance")


def run(args):
    specs = select_specs(args.case)
    variants = (*VARIANTS, "previous_auto") if args.previous_auto_archive else VARIANTS
    manifest = json.loads(args.source_manifest.read_text())
    before = source_hashes(ROOT / "src")
    if before != manifest["files"]:
        raise ValueError("Current source differs from frozen manifest")
    harnesses = [ROOT / "benchmarks" / name for name in (
        "knn_first_query.py", "knn_first_query_worker.py", "knn_cold.py", "knn_order.py", "knn_auto_control.py",
        "knn_reference_runtime.py", "knn_broad_worker.py", "knn_broad.py", "knn_broad_fixtures.py",
        "knn_backend_audit.py", "knn_pipeline.py", "knn_freeze.py", "knn_freeze_runtime.py",
        "knn_review_distributions.py", "knn_acceleration.py", "knn_gaussian_fresh_control.py", "knn_128_control.py",
    )]
    archive = Path(manifest["archive"]["path"])
    archive = archive if archive.is_absolute() else ROOT / archive
    if sha256(archive.read_bytes()) != manifest["archive"]["sha256"]:
        raise ValueError("Current source archive changed")
    archives = [archive, args.source_manifest]
    if args.previous_auto_archive:
        for path in (args.previous_auto_archive, args.previous_auto_runtime):
            archives.extend((path, path.with_suffix(".manifest.json")))
    report = {"schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "variants": list(variants), "metrics": list(METRICS), "planned_cases": len(specs), "repeats": args.repeats,
              "source_snapshot": manifest["archive"], "source_hashes_before": before,
              "harness_hashes_before": hashes(harnesses),
              "archive_hashes_before": {str(p.resolve()): sha256(p.read_bytes()) for p in archives},
              "protocol": "Initialized isolated persistent processes; normal NumPy/estimator imports and configuration "
              "are recorded separately. No model or query prewarm. Every request constructs and fits a new estimator, "
              "then times its first query and separately its second query. Public NumPy results establish completion; "
              "no harness MLX import/device probe/synchronization runs before the first query. Both outputs are audited "
              "after both timers. All requested rounds (default 30), including first model per fixture and first model in worker, remain "
              "visible. Earlier models may initialize runtime/compiler caches. This is model-first-use in an initialized "
              "process, not fresh-process or machine-cold latency. Five-worker Williams order has a complete balanced "
              "10-round block, repeated three times by default. Any incomplete final block is also retained. "
              "No classification/regression operations are added here; "
              "the separate full broad and preservation control panels retain those operations and fixtures.",
              "cases": []}
    workers = {}
    save(report, args.output)
    try:
        with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-first-query-") as tmp:
            temp = Path(tmp)
            roots = dict.fromkeys(variants, ROOT / "src")
            runtime = None
            if args.previous_auto_archive:
                roots["previous_auto"] = temp / "previous_auto"
                reference = extract_reference(args.previous_auto_archive, roots["previous_auto"])
                runtime = install_reference_runtime(args.previous_auto_runtime, roots["previous_auto"], reference)
                report.update(reference=reference, reference_runtime=runtime,
                              reference_archive=str(args.previous_auto_archive.resolve()),
                              reference_runtime_archive=str(args.previous_auto_runtime.resolve()))
            fixtures = temp / "fixtures"
            fixtures.mkdir()
            report["fixtures"] = [make_fixture(spec, fixtures) for spec in specs]
            paths = [fixtures / (s["case"]["name"] + ".npz") for s in specs]
            report["fixture_hashes_before"] = {p.name: sha256(p.read_bytes()) for p in paths}
            binary_relative = "mlxlearn/_kernels/_neighbors_cpu_native" + current_abi()["EXT_SUFFIX"]
            binaries = {v: roots[v] / binary_relative for v in variants if not v.startswith("sklearn")}
            report["native_hashes_before"] = {v: sha256(p.read_bytes()) if p.is_file() else None for v, p in binaries.items()}
            for name in variants:
                workers[name] = FirstQueryWorker(name, fixtures, roots[name])
            report["setup"] = {name: worker.setup for name, worker in workers.items()}
            previous = None
            for ci, spec in enumerate(specs):
                samples = {v: [] for v in variants}
                case = {"case": spec["case"], "spec": spec, "rounds": [],
                        "variants": {v: {"samples": samples[v]} for v in variants}}
                report["cases"].append(case)
                for ri in range(args.repeats):
                    order = round_order(rotate(variants, ci), ri, "balanced")
                    positions = placement(order, previous)
                    case["rounds"].append({"round": ri, "order": order, "placement": positions})
                    for name in order:
                        sample = request(workers[name], {"op": "fresh_model", "case": spec["case"], "round": ri})
                        samples[name].append(sample)
                    previous = order[-1]
                case["variants"] = {v: summarize(samples[v], args.repeats) for v in variants}
                case["comparisons"] = comparisons(case["variants"])
                save(report, args.output)
                print(f"case {ci+1}/{len(specs)} {spec['case']['name']}", flush=True)
            report["workers"] = {name: worker.request({"op": "info"}) for name, worker in workers.items()}
            report["native_hashes_after"] = {v: sha256(p.read_bytes()) if p.is_file() else None for v, p in binaries.items()}
            report["native_integrity_pass"] = report["native_hashes_before"] == report["native_hashes_after"]
            report["reference_runtime_integrity_pass"] = runtime is None or reference_runtime_unchanged(roots["previous_auto"], runtime)
            report["fixture_hashes_after"] = {p.name: sha256(p.read_bytes()) for p in paths}
            report["fixture_integrity_pass"] = report["fixture_hashes_before"] == report["fixture_hashes_after"]
            report["quality_by_variant"] = {v: all(c["variants"][v]["quality_pass"] for c in report["cases"]) for v in variants}
            report["timing_count_pass"] = all(c["variants"][v]["completed_samples"] == args.repeats for c in report["cases"] for v in variants)
            report["status"] = "complete"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    finally:
        for worker in workers.values():
            worker.close()
        for stem, values in (
            ("source", source_hashes(ROOT / "src")), ("harness", hashes(harnesses)),
            ("archive", {str(p.resolve()): sha256(p.read_bytes()) for p in archives}),
        ):
            report[f"{stem}_hashes_after"] = values
            report[f"{stem}_integrity_pass"] = values == report[f"{stem}_hashes_before"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["acceptance_pass"] = report["status"] == "complete" and all(report.get(k, False) for k in (
            "source_integrity_pass", "harness_integrity_pass", "archive_integrity_pass", "native_integrity_pass",
            "reference_runtime_integrity_pass", "fixture_integrity_pass", "timing_count_pass",
        )) and all(report.get("quality_by_variant", {}).get(v, False) for v in variants if not v.startswith("sklearn"))
        if report["status"] == "complete":
            try:
                validate(report, expected_cases=len(specs), expected_repeats=args.repeats)
            except Exception:
                report.update(status="error", acceptance_pass=False, validation_error=traceback.format_exc())
        save(report, args.output)
    print(f"acceptance_pass={report['acceptance_pass']}: {args.output}", flush=True)
    return 0 if report["acceptance_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--previous-auto-archive", type=Path)
    parser.add_argument("--previous-auto-runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Require positive repeats")
    if bool(args.previous_auto_archive) != bool(args.previous_auto_runtime):
        parser.error("Provide both previous-auto source and native runtime")
    if args.case and len(set(args.case)) != len(args.case):
        parser.error("Case names must be distinct")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing result")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
