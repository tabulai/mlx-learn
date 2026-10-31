"""Audited fresh-process first fit/query against sklearn and frozen auto.

Each retained observation starts a separate interpreter and performs one first
fit/query. No estimator/runtime metadata request or warmup precedes it. All
fixtures and numerical oracles are shared with the unchanged broad panel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .knn_auto_control import TARGET_FAMILIES
from .knn_backend_audit import validate_record_backend
from .knn_broad import hashes, rotate, save
from .knn_broad_fixtures import make_fixture, workload_specs
from .knn_pipeline import ROOT, extract_reference, sha256, source_hashes
from .knn_reference_runtime import (
    current_abi,
    install_reference_runtime,
    reference_runtime_unchanged,
    validate_reference_worker,
)

VARIANTS = ("current", "sklearn_default", "sklearn_1", "sklearn_auto")
METRICS = ("numpy_import_ms", "fixture_load_ms", "estimator_setup_ms", "fit_ms", "query_ms",
           "fit_query_ms", "estimator_to_result_ms", "worker_entry_to_result_ms",
           "process_start_to_result_ms", "process_elapsed_ms")


def select_specs(names=None):
    specs = [s for s in workload_specs() if s["family"] in TARGET_FAMILIES]
    if names:
        missing = set(names) - {s["case"]["name"] for s in specs}
        if missing:
            raise ValueError(f"Unknown targeted cases: {sorted(missing)}")
        specs = [s for s in specs if s["case"]["name"] in names]
    return specs


def parse_launch(first, remaining, *, returncode, elapsed_ms, result_ms):
    """Validate the two-stage protocol without dropping failed observations."""
    lines = ([first] if first else []) + remaining.splitlines()
    messages = [json.loads(line) for line in lines if line.strip()]
    if returncode != 0 or len(messages) != 2 or [m.get("kind") for m in messages] != [
        "result_ready", "audited_result",
    ]:
        return {"error": "Fresh-process launch failed", "returncode": returncode,
                "messages": messages, "audit": {"pass": False}, "process_elapsed_ms": elapsed_ms}
    ready, audited = messages
    timings = dict(ready["timings"], process_start_to_result_ms=result_ms, process_elapsed_ms=elapsed_ms)
    if not all(isinstance(timings.get(k), (int, float)) and math.isfinite(timings[k]) and timings[k] > 0 for k in METRICS):
        raise ValueError("Invalid fresh-process timing")
    if not math.isclose(timings["fit_query_ms"], timings["fit_ms"] + timings["query_ms"], rel_tol=1e-10):
        raise ValueError("First fit/query phase timing does not reconcile")
    if not math.isclose(timings["estimator_to_result_ms"], timings["estimator_setup_ms"] + timings["fit_query_ms"], rel_tol=1e-10):
        raise ValueError("Estimator setup time was omitted from first-result latency")
    if not math.isclose(timings["worker_entry_to_result_ms"], (ready["result_ready_ns"] - ready["entry_ns"]) / 1e6, rel_tol=1e-10):
        raise ValueError("Worker result timestamp disagrees with first-result latency")
    if timings["process_elapsed_ms"] < timings["process_start_to_result_ms"]:
        raise ValueError("Result marker follows process exit")
    return {**audited, "timings": timings, "loaded_modules": ready["loaded_modules"],
            "entry_ns": ready["entry_ns"], "result_ready_ns": ready["result_ready_ns"]}


def launch(variant, source_root, fixture, k, *, timeout=120):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + str(ROOT)
    command = [sys.executable, "-u", str(ROOT / "benchmarks/knn_cold_worker.py"),
               "--variant", variant, "--source-root", str(source_root), "--fixture", str(fixture), "--k", str(k)]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env)
    try:
        buffer = b""
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in buffer:
                remaining_time = timeout - (time.perf_counter() - started)
                if remaining_time <= 0 or not selector.select(remaining_time):
                    raise TimeoutError("Fresh-process result marker timed out")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                buffer += chunk
        first, _separator, pending = buffer.partition(b"\n")
        result_ms = (time.perf_counter() - started) * 1000
        remaining, stderr = process.communicate(timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = parse_launch(first.decode(), (pending + remaining).decode(), returncode=process.returncode,
                              elapsed_ms=elapsed_ms, result_ms=result_ms)
        result.update(pid=process.pid, command=command, stderr=stderr.decode(errors="replace"))
        return result
    except Exception:
        process.kill()
        stdout, stderr = process.communicate()
        return {"error": traceback.format_exc(), "audit": {"pass": False}, "pid": process.pid,
                "initial_output": buffer.decode(errors="replace"),
                "command": command, "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}


def summarize(samples, expected):
    completed = [s for s in samples if "timings" in s and "error" not in s]
    return {
        "expected_samples": expected, "completed_samples": len(completed),
        "quality_pass": len(samples) == len(completed) == expected
        and all(s.get("audit", {}).get("pass", False) for s in samples),
        "metrics": {metric: {
            "median_ms": statistics.median(s["timings"][metric] for s in completed) if completed else None,
            "min_ms": min(s["timings"][metric] for s in completed) if completed else None,
            "max_ms": max(s["timings"][metric] for s in completed) if completed else None,
        } for metric in METRICS}, "samples": samples,
    }


def comparisons(variants):
    current = variants["current"]
    result = {}
    for name, other in variants.items():
        if name == "current":
            continue
        result[name] = {}
        for metric in METRICS:
            a, b = current["metrics"][metric]["median_ms"], other["metrics"][metric]["median_ms"]
            paired = [y["timings"][metric] / x["timings"][metric]
                      if "timings" in x and "timings" in y else None
                      for x, y in zip(current["samples"], other["samples"], strict=True)]
            ratios = [r for r in paired if r is not None]
            result[name][metric] = {
                "ratio_of_medians": b / a if a is not None and b is not None else None,
                "median_paired_ratio": statistics.median(ratios) if ratios else None,
                "paired_ratios": paired,
            }
    return result


def validate(report):
    if report["status"] != "complete" or len(report["cases"]) != report["planned_cases"]:
        raise ValueError("Incomplete fresh-process report")
    if tuple(report["variants"]) not in (VARIANTS, (*VARIANTS, "previous_auto")) or report["metrics"] != list(METRICS):
        raise ValueError("Unexpected fresh-process variant/metric inventory")
    expected_specs = select_specs([c["case"]["name"] for c in report["cases"]])
    if [c["spec"] for c in report["cases"]] != expected_specs:
        raise ValueError("Fresh-process fixtures changed")
    for case in report["cases"]:
        if set(case["variants"]) != set(report["variants"]):
            raise ValueError("Missing cold variant")
        if len(case["rounds"]) != report["repeats"]:
            raise ValueError("Missing cold launch round")
        for index, entry in enumerate(case["rounds"]):
            if entry["round"] != index or sorted(entry["order"]) != sorted(report["variants"]):
                raise ValueError("Invalid cold launch order")
        for name, stats in case["variants"].items():
            if stats != summarize(stats["samples"], report["repeats"]):
                raise ValueError("Cold summary disagrees with retained launches")
            for sample in stats["samples"]:
                validate_record_backend(sample, "auto", is_sklearn=name.startswith("sklearn"))
                if "error" not in sample and any(sample["loaded_modules"]["before_input"].values()):
                    raise ValueError("Heavy modules were already loaded before fresh-process input setup")
        if case["comparisons"] != comparisons(case["variants"]):
            raise ValueError("Cold speed ratios disagree with retained launches")
    quality = {name: all(c["variants"][name]["quality_pass"] for c in report["cases"]) for name in report["variants"]}
    if report["quality_by_variant"] != quality:
        raise ValueError("Cold overall accuracy gate disagrees with launches")
    timing_pass = all(c["variants"][name]["completed_samples"] == report["repeats"]
                      for c in report["cases"] for name in report["variants"])
    if report["timing_count_pass"] != timing_pass:
        raise ValueError("Cold completed timing count disagrees with launches")
    expected = len(report["cases"]) * len(report["variants"]) * report["repeats"]
    actual = sum(len(c["variants"][name]["samples"]) for c in report["cases"] for name in report["variants"])
    if report["expected_launches"] != expected or report["completed_launches"] != actual or actual != expected:
        raise ValueError("Cold launch inventory is incomplete")


def run(args):
    specs = select_specs(args.case)
    variants = (*VARIANTS, "previous_auto") if args.previous_auto_archive else VARIANTS
    manifest = json.loads(args.source_manifest.read_text())
    before = source_hashes(ROOT / "src")
    if before != manifest["files"]:
        raise ValueError("Current source differs from frozen manifest")
    harnesses = [ROOT / "benchmarks" / name for name in (
        "knn_cold.py", "knn_cold_worker.py", "knn_reference_runtime.py", "knn_broad_worker.py",
        "knn_broad_fixtures.py", "knn_broad.py", "knn_backend_audit.py", "knn_auto_control.py",
        "knn_pipeline.py", "knn_freeze.py", "knn_freeze_runtime.py", "knn_review_distributions.py",
        "knn_acceleration.py", "knn_gaussian_fresh_control.py", "knn_128_control.py", "knn_order.py",
    )]
    archives = [args.source_manifest]
    source_archive = Path(manifest["archive"]["path"])
    source_archive = source_archive if source_archive.is_absolute() else ROOT / source_archive
    archives.append(source_archive)
    if sha256(source_archive.read_bytes()) != manifest["archive"]["sha256"]:
        raise ValueError("Current source archive changed")
    if args.previous_auto_archive:
        for p in (args.previous_auto_archive, args.previous_auto_runtime):
            archives.extend((p, p.with_suffix(".manifest.json")))
    report = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "variants": list(variants), "repeats": args.repeats, "planned_cases": len(specs),
        "expected_launches": len(specs) * len(variants) * args.repeats,
        "source_manifest": str(args.source_manifest.resolve()), "source_snapshot": manifest["archive"],
        "source_hashes_before": before, "harness_hashes_before": hashes(harnesses),
        "archive_hashes_before": {str(p.resolve()): sha256(p.read_bytes()) for p in archives},
        "protocol": "Each retained launch starts a new interpreter and performs exactly one first public "
        "NearestNeighbors fit/query. There are no warmups, metadata preflight requests, harness-driven native preimports "
        "or external pre-fit MLX synchronization. NumPy import and fixture X/Q loading are reported separately "
        "from estimator import/setup and API calls. Necessary transitive library imports remain charged. "
        "The first stdout result marker follows returned NumPy neighbors, before audits and runtime metadata. "
        "Parent process-start-to-result includes interpreter startup, fixture input loading, estimator imports, "
        "fit/query and result-marker IPC. Process elapsed also includes post-result audits/metadata/exit. "
        "This is process-cold, not a claim of cold OS/filesystem/Metal disk caches. All launches are serial.",
        "accuracy_reference": "Unchanged broad fixture oracles and strict canonical audit_search; every launch "
        "is audited after timing. Accuracy failures never remove timings or speed denominators.",
        "metrics": list(METRICS), "cases": [],
    }
    save(report, args.output)
    try:
        with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-process-cold-") as tmp:
            temp = Path(tmp)
            roots = dict.fromkeys(variants, ROOT / "src")
            runtime = None
            if args.previous_auto_archive:
                roots["previous_auto"] = temp / "previous_auto"
                reference = extract_reference(args.previous_auto_archive, roots["previous_auto"])
                runtime = install_reference_runtime(args.previous_auto_runtime, roots["previous_auto"], reference)
                report["reference"] = reference
                report["reference_runtime"] = runtime
                report["reference_archive"] = str(args.previous_auto_archive.resolve())
                report["reference_runtime_archive"] = str(args.previous_auto_runtime.resolve())
            fixtures = temp / "fixtures"
            fixtures.mkdir()
            report["fixtures"] = [make_fixture(spec, fixtures) for spec in specs]
            paths = [fixtures / (s["case"]["name"] + ".npz") for s in specs]
            report["fixture_hashes_before"] = {p.name: sha256(p.read_bytes()) for p in paths}
            binary_relative = "mlxlearn/_kernels/_neighbors_cpu_native" + current_abi()["EXT_SUFFIX"]
            binary_paths = {name: roots[name] / binary_relative for name in variants if not name.startswith("sklearn")}
            report["native_hashes_before"] = {name: sha256(p.read_bytes()) if p.is_file() else None for name, p in binary_paths.items()}
            for ci, spec in enumerate(specs):
                samples = {name: [] for name in variants}
                entry = {"case": spec["case"], "spec": spec, "rounds": [],
                         "variants": {name: {"samples": samples[name]} for name in variants}}
                report["cases"].append(entry)
                for repeat in range(args.repeats):
                    order = rotate(variants, ci + repeat)
                    if repeat % 2:
                        order.reverse()
                    entry["rounds"].append({"round": repeat, "order": order})
                    for name in order:
                        sample = launch(name, roots[name], paths[ci], spec["case"]["k"], timeout=args.timeout)
                        samples[name].append(sample)
                        if "error" not in sample:
                            expected = report["reference"]["files"] if name == "previous_auto" else before
                            if sample["worker"]["source_hashes"] != expected:
                                raise ValueError("Cold worker loaded incorrect source")
                            if sample["fixture_sha256"] != report["fixture_hashes_before"][paths[ci].name]:
                                raise ValueError("Cold fixture changed")
                            validate_record_backend(sample, "auto", is_sklearn=name.startswith("sklearn"))
                            if name == "previous_auto":
                                validate_reference_worker(sample["worker"], runtime)
                            if name in binary_paths:
                                native = sample["worker"]["native_cpu_extension"]
                                actual = native["binary"]["sha256"] if native["available"] else None
                                if actual != report["native_hashes_before"][name]:
                                    raise ValueError("Cold worker loaded a different native binary")
                entry["variants"] = {name: summarize(samples[name], args.repeats) for name in variants}
                entry["comparisons"] = comparisons(entry["variants"])
                save(report, args.output)
                print(f"case {ci+1}/{len(specs)} {spec['case']['name']}", flush=True)
            report["native_hashes_after"] = {name: sha256(p.read_bytes()) if p.is_file() else None for name, p in binary_paths.items()}
            report["native_integrity_pass"] = report["native_hashes_before"] == report["native_hashes_after"]
            report["reference_runtime_integrity_pass"] = runtime is None or reference_runtime_unchanged(roots["previous_auto"], runtime)
            report["fixture_hashes_after"] = {p.name: sha256(p.read_bytes()) for p in paths}
            report["fixture_integrity_pass"] = report["fixture_hashes_before"] == report["fixture_hashes_after"]
            report["quality_by_variant"] = {name: all(c["variants"][name]["quality_pass"] for c in report["cases"]) for name in variants}
            report["completed_launches"] = sum(len(c["variants"][name]["samples"]) for c in report["cases"] for name in variants)
            report["timing_count_pass"] = all(c["variants"][name]["completed_samples"] == args.repeats
                                              for c in report["cases"] for name in variants)
            report["status"] = "complete"
            validate(report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    finally:
        report["source_hashes_after"] = source_hashes(ROOT / "src")
        report["source_integrity_pass"] = report["source_hashes_before"] == report["source_hashes_after"]
        report["harness_hashes_after"] = hashes(harnesses)
        report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
        report["archive_hashes_after"] = {str(p.resolve()): sha256(p.read_bytes()) for p in archives}
        report["archive_integrity_pass"] = report["archive_hashes_before"] == report["archive_hashes_after"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["acceptance_pass"] = report["status"] == "complete" and all(report.get(k, False) for k in (
            "source_integrity_pass", "harness_integrity_pass", "archive_integrity_pass",
            "native_integrity_pass", "reference_runtime_integrity_pass", "fixture_integrity_pass", "timing_count_pass",
        )) and all(report.get("quality_by_variant", {}).get(name, False) for name in variants if not name.startswith("sklearn"))
        save(report, args.output)
    print(f"acceptance_pass={report['acceptance_pass']}: {args.output}", flush=True)
    return 0 if report["acceptance_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--previous-auto-archive", type=Path)
    parser.add_argument("--previous-auto-runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Require positive repeats and finite positive timeout")
    if bool(args.previous_auto_archive) != bool(args.previous_auto_runtime):
        parser.error("Provide both previous-auto source and native runtime archives")
    if args.case and len(set(args.case)) != len(args.case):
        parser.error("Case names must be distinct")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
