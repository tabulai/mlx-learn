"""Independently verify final automatic-dispatch evidence without timed work.

Read-only with respect to source, harnesses, raw evidence and fixtures. Writes a
compact derived audit, retaining every target/preservation cell and threshold
miss. The raw control may be large; this loads each input once and streams file
hashes instead of retaining a second copy of raw bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from .knn_auto_control import TARGET_FAMILIES
from .knn_auto_control_tables import validate as validate_control
from .knn_broad_summary import summarize_report, valid_time
from .knn_cold_summary import summarize_report as summarize_cold_report
from .knn_first_query_tables import summarize_report as summarize_first_query_report
from .knn_freeze_runtime import verify_runtime_archive
from .knn_pipeline import ROOT
from .knn_reference_runtime import validate_reference_worker

COMPARATORS = ("sklearn_default", "sklearn_1", "sklearn_auto")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_report(path):
    # json.load avoids explicitly holding a separate raw byte buffer.
    with Path(path).open() as stream:
        return json.load(stream)


def fixture_inventory(report):
    fixtures = {f["case"]["name"]: f for f in report["fixtures"]}
    require(len(fixtures) == len(report["fixtures"]), "Duplicate fixture metadata")
    cases = {c["case"]["name"]: c["spec"] for c in report["cases"]}
    require(set(fixtures) == set(cases), "Case and fixture inventories disagree")
    for name, fixture in fixtures.items():
        require(fixture["case"] == cases[name]["case"], f"Fixture case changed: {name}")
        require(set(fixture["array_sha256"]) == set(fixture["array_shapes"])
                == set(fixture["array_dtypes"]), f"Array inventory differs: {name}")
        require(report["fixture_hashes_before"][name + ".npz"] == fixture["fixture_sha256"],
                f"Serialized fixture hash differs: {name}")
    return fixtures, cases


def check_integrity(report, *, control=False):
    for stem in ("source", "harness", "fixture"):
        require(report[f"{stem}_hashes_before"] == report[f"{stem}_hashes_after"],
                f"{stem} changed during timing")
    expected = dict.fromkeys(report["variants"], report["source_hashes_before"])
    if control:
        require(report["archive_hashes_before"] == report["archive_hashes_after"],
                "An archive changed during control timing")
        expected.update({name: reference["files"] for name, reference in report["references"].items()})
    for variant, hashes in expected.items():
        require(report["workers"][variant]["source_hashes"] == hashes,
                f"Wrong initial worker source: {variant}")
        require(report["worker_source_hashes_after"][variant] == hashes,
                f"Worker source changed: {variant}")


def check_archives(broad, control):
    """Verify the archived bytes as well as equality of recorded hash maps."""
    archive_path = Path(broad["source_snapshot"]["path"])
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    expected = [(archive_path, broad["source_snapshot"]["sha256"], broad["source_hashes_before"])]
    for name, archive in control["reference_archives"].items():
        reference = control["references"][name]
        expected.append((Path(archive), reference["archive"]["sha256"], reference["files"]))
    for archive, expected_hash, files in expected:
        manifest = read_report(archive.with_suffix(".manifest.json"))
        require(manifest["files"] == files and manifest["archive"]["sha256"] == expected_hash,
                f"Recorded archive manifest changed: {archive}")
        require(file_hash(archive) == expected_hash, f"Archive hash changed: {archive}")
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            require(len(names) == len(set(names)) and set(names) == set(files),
                    f"Archive inventory changed: {archive}")
            require(all(hashlib.sha256(source.read(name)).hexdigest() == files[name] for name in names),
                    f"Archive source bytes changed: {archive}")


def check_native_binary_metadata(broad, control):
    """Check current runtime binaries; frozen GPU temp paths are never required."""
    workers = {"broad/current": broad["workers"]["current"], **{
        f"control/{name}": control["workers"][name]
        for name in ("current", "current_b", "current_gpu")
    }}
    records = {name: worker.get("native_cpu_extension") for name, worker in workers.items()}
    relative = "mlxlearn/_kernels/_neighbors_cpu_native.cpp"
    if all(record is None for record in records.values()):
        require(relative not in broad["source_hashes_before"], "Native C++ source has no runtime binary metadata")
        return {"available": False, "metadata_recorded": False}
    require(all(isinstance(record, dict) for record in records.values()), "Missing native CPU worker metadata")
    first = next(iter(records.values()))
    binary_hash = first["binary"]["sha256"] if first["binary"] is not None else None
    source_hash = first["source"]["sha256"] if first["source"] is not None else None
    require(source_hash == broad["source_hashes_before"].get(relative),
            "Native C++ metadata differs from frozen source inventory")
    checked_files = {}
    for name, record in records.items():
        require(record["module"] == "mlxlearn._kernels._neighbors_cpu_native", "Wrong native CPU module")
        require(record["available"] == first["available"], "Workers disagree on native CPU availability")
        require((record["binary"]["sha256"] if record["binary"] is not None else None) == binary_hash,
                f"Native CPU binaries differ: {name}")
        require((record["source"]["sha256"] if record["source"] is not None else None) == source_hash,
                f"Native C++ sources differ: {name}")
        require(record["available"] == (record["binary"] is not None), "Inconsistent native availability")
        root = Path(workers[name]["source_root"]).resolve()
        for kind in ("source", "binary"):
            item = record[kind]
            if item is None:
                continue
            path = Path(item["path"]).resolve()
            require(path.is_relative_to(root), f"Native {kind} is outside its worker source: {name}")
            if kind == "source":
                require(item["relative_path"] == relative and str(path.relative_to(root)) == relative,
                        f"Unexpected native C++ path: {name}")
            require(path.is_file(), f"Current native {kind} is missing after the run: {path}")
            if str(path) not in checked_files:
                checked_files[str(path)] = file_hash(path)
            require(checked_files[str(path)] == item["sha256"], f"Native {kind} changed after timing: {name}")
    return {"available": first["available"], "metadata_recorded": True,
            "binary_sha256": binary_hash, "cpp_sha256": source_hash,
            "all_four_current_source_workers_match": True,
            "current_files_verified_after_run": checked_files,
            "historical_gpu_reference_paths_required": False}


def check_runtime_artifact(archive, broad, native):
    if archive is None:
        return None
    manifest = verify_runtime_archive(archive)
    require(native["available"], "A native runtime artifact requires native worker metadata")
    require(manifest["source_files"] == broad["source_hashes_before"], "Runtime artifact source differs from benchmark")
    require(manifest["source_archive"]["sha256"] == broad["source_snapshot"]["sha256"],
            "Runtime artifact is linked to a different source archive")
    require(manifest["native_extension"]["binary"]["sha256"] == native["binary_sha256"],
            "Archived binary differs from measured worker binary")
    require(manifest["native_extension"]["source"]["sha256"] == native["cpp_sha256"],
            "Archived C++ differs from measured source")
    return {"path": str(Path(archive).resolve()), "sha256": manifest["archive"]["sha256"],
            "manifest_sha256": file_hash(Path(archive).with_suffix(".manifest.json")),
            "member_count": len(manifest["files"]), "matches_measured_source_and_binary": True,
            "abi": manifest["abi"], "restore_policy": manifest["restore_policy"]}


def broad_cells(report, *, target_only=False):
    result = []
    for ci, case in enumerate(report["cases"]):
        if target_only and case["spec"]["family"] not in TARGET_FAMILIES:
            continue
        for operation in report["target_operations"]:
            entry = case["operations"][operation]
            ratios = {v: entry["comparisons"][v]["ratio_of_medians"] for v in COMPARATORS}
            result.append({
                "case": case["case"]["name"], "family": case["spec"]["family"],
                "distribution": case["spec"]["distribution"], "seed": case["spec"]["seed"],
                "operation": operation,
                "median_ms": {v: entry["variants"][v]["median_ms"] for v in report["variants"]},
                "speedup_by_comparator": ratios,
                "median_paired_speedup_by_comparator": {
                    v: entry["comparisons"][v]["median_of_paired_ratios"] for v in COMPARATORS
                },
                "query_backends": dict(Counter(
                    s.get("query_backend", "missing") for s in entry["variants"]["current"]["samples"]
                )),
                "raw_pointer": f"/cases/{ci}/operations/{operation}",
            })
    return result


def threshold_counts(cells):
    result = {}
    for comparator in COMPARATORS:
        ratios = [c["speedup_by_comparator"][comparator] for c in cells]
        available = [r for r in ratios if r is not None]
        slow = [c for c in cells if c["speedup_by_comparator"][comparator] is None
                or c["speedup_by_comparator"][comparator] < 1]
        result[comparator] = {
            "cell_count": len(cells), "available_ratios": len(available),
            "faster_than_baseline": sum(r > 1 for r in available),
            "at_least_baseline": sum(r >= 1 for r in available),
            "at_least_two_x": sum(r >= 2 for r in available),
            "minimum_speedup": min(available) if available else None,
            "maximum_speedup": max(available) if available else None,
            "slower_or_missing_cells": slow,
        }
    return result


def paired_ratio(samples, numerator, denominator):
    ratios = [a["ms"] / b["ms"] for a, b in zip(
        samples[numerator], samples[denominator], strict=True,
    ) if valid_time(a) and valid_time(b)]
    return {"median": statistics.median(ratios) if ratios else None,
            "min": min(ratios) if ratios else None, "max": max(ratios) if ratios else None,
            "pair_count": len(ratios), "ratios": ratios}


def control_cells(report, *, preservation):
    cells = []
    for ci, case in enumerate(report["cases"]):
        if (case["spec"]["family"] not in TARGET_FAMILIES) != preservation:
            continue
        for operation, experiment in case["experiments"].items():
            samples = {v: experiment["variants"][v]["samples"] for v in report["variants"]}
            ratios = {
                "auto_aa_over_auto": paired_ratio(samples, "current_b", "current"),
                "current_gpu_over_auto": paired_ratio(samples, "current_gpu", "current"),
                "current_gpu_over_auto_aa": paired_ratio(samples, "current_gpu", "current_b"),
                "previous_gpu_over_auto": paired_ratio(samples, "previous_gpu", "current"),
                "previous_gpu_over_auto_aa": paired_ratio(samples, "previous_gpu", "current_b"),
                "previous_gpu_over_current_gpu": paired_ratio(samples, "previous_gpu", "current_gpu"),
            }
            baselines = ["current_gpu", "previous_gpu"]
            if "previous_auto" in report["variants"]:
                baselines.append("previous_auto")
                ratios.update({
                    "previous_auto_over_auto": paired_ratio(samples, "previous_auto", "current"),
                    "previous_auto_over_auto_aa": paired_ratio(samples, "previous_auto", "current_b"),
                    "previous_gpu_over_previous_auto": paired_ratio(samples, "previous_gpu", "previous_auto"),
                })
            slower_both = {}
            for baseline in baselines:
                a, b = (ratios[f"{baseline}_over_{v}"]["median"] for v in ("auto", "auto_aa"))
                slower_both[baseline] = a is not None and b is not None and a < 1 and b < 1
            cells.append({
                "case": case["case"]["name"], "family": case["spec"]["family"],
                "distribution": case["spec"]["distribution"], "seed": case["spec"]["seed"],
                "operation": operation,
                "median_ms": {v: experiment["variants"][v]["median_ms"] for v in report["variants"]},
                "paired_ratios": ratios, "auto_slower_in_both_replicas": slower_both,
                "ratio_of_medians": {v: experiment["comparisons"][f"current_vs_{v}"]["ratio_of_medians"]
                                     for v in report["variants"][1:]},
                "raw_pointer": f"/cases/{ci}/experiments/{operation}",
            })
    return cells


def preservation_summary(cells):
    result = {}
    for distribution in sorted({c["distribution"] for c in cells}):
        selected = [c for c in cells if c["distribution"] == distribution]
        ranges = {}
        for key in selected[0]["paired_ratios"]:
            values = [c["paired_ratios"][key]["median"] for c in selected]
            present = [v for v in values if v is not None]
            ranges[key] = {"min": min(present) if present else None,
                           "max": max(present) if present else None,
                           "at_least_one_x": sum(v >= 1 for v in present)}
        result[distribution] = {
            "cell_count": len(selected), "median_paired_ratio_ranges": ranges,
            "auto_slower_in_both_replicas": {
                v: [c for c in selected if c["auto_slower_in_both_replicas"][v]]
                for v in selected[0]["auto_slower_in_both_replicas"]
            },
        }
    return result


def control_audit_counts(report):
    result = {}
    for variant in report["variants"]:
        counts = {phase: {"audits": 0, "strict_passes": 0}
                  for phase in ("cold", "warmup", "probe", "retained")}

        def add(phase, audits, counts=counts):
            counts[phase]["audits"] += len(audits)
            counts[phase]["strict_passes"] += sum(bool(a.get("pass")) for a in audits)

        for case in report["cases"]:
            add("cold", [case["cold"][variant].get("audit", {})])
            for experiment in case["experiments"].values():
                block = experiment["warmup"][variant]
                add("warmup", block.get("warmup_audits", []))
                add("probe", [s.get("audit", {}) for s in block.get("samples", [])])
                add("retained", [s.get("audit", {}) for s in experiment["variants"][variant]["samples"]])
        result[variant] = counts
    return result


def verify(broad, control, previous, *, broad_path, broad_sha256, runtime_archive=None, previous_auto=None, cold=None,
           first_query=None):
    summary = summarize_report(broad, broad_path, broad_sha256, expected_cases=72, expected_repeats=15)
    validate_control(control, expected_cases=48)
    require(broad.get("current_mode") == "auto", "Final broad evidence must use automatic dispatch")
    require(previous.get("current_mode", "gpu") == "gpu", "Previous evidence must remain labelled GPU")
    require(previous["status"] == "complete", "Previous evidence is incomplete")
    require(control["pairs"] == 31, "Expected all 31 control pairs")
    require(control["planned_case_operation_count"] == 222, "Expected all 222 control operations")
    require(broad["warmup"] == 5, "Expected all five broad warmup requests")
    require(broad["source_hashes_before"] == control["source_hashes_before"], "Control/broad source mismatch")
    check_integrity(broad)
    check_integrity(control, control=True)
    check_archives(broad, control)
    reference_runtime_checks = {}
    for name, runtime in control.get("reference_runtimes", {}).items():
        validate_reference_worker(control["workers"][name], runtime)
        saved = verify_runtime_archive(control["reference_runtime_archives"][name])
        require(saved == runtime, f"Reference native runtime manifest changed: {name}")
        require(runtime["source_files"] == control["references"][name]["files"], "Reference runtime source mismatch")
        require(runtime["source_archive"]["sha256"] == control["references"][name]["archive"]["sha256"],
                "Reference runtime is linked to a different source archive")
        require(control["reference_runtime_integrity_pass"], "Reference native runtime changed during timing")
        reference_runtime_checks[name] = {"binary_sha256": runtime["native_extension"]["binary"]["sha256"],
                                          "cpp_sha256": runtime["native_extension"]["source"]["sha256"],
                                          "archive_sha256": runtime["archive"]["sha256"], "pass": True}
    native = check_native_binary_metadata(broad, control)
    runtime = check_runtime_artifact(runtime_archive, broad, native)
    cold_summary = None
    if cold is not None:
        cold_summary = summarize_cold_report(cold, broad=broad, expected_cases=18, expected_repeats=5)
        require(cold["native_hashes_before"]["current"] == native["binary_sha256"],
                "Cold and broad workers used different current native binaries")
        if "previous_auto" in control["variants"]:
            require(cold["reference"]["files"] == control["references"]["previous_auto"]["files"],
                    "Cold and control used different previous automatic sources")
            require(cold["native_hashes_before"]["previous_auto"] ==
                    control["reference_runtimes"]["previous_auto"]["native_extension"]["binary"]["sha256"],
                    "Cold and control used different previous automatic binaries")
    first_query_summary = None
    if first_query is not None:
        first_query_summary = summarize_first_query_report(first_query, broad=broad, expected_cases=18, expected_repeats=30)
        require(first_query["native_hashes_before"]["current"] == native["binary_sha256"],
                "First-query and broad current native binaries differ")
        if "previous_auto" in control["variants"]:
            require(first_query["reference"]["files"] == control["references"]["previous_auto"]["files"]
                    and first_query["native_hashes_before"]["previous_auto"] ==
                    control["reference_runtimes"]["previous_auto"]["native_extension"]["binary"]["sha256"],
                    "First-query and control prior automatic runtimes differ")
    current_fixtures, current_specs = fixture_inventory(broad)
    old_fixtures, old_specs = fixture_inventory(previous)
    require(len(current_fixtures) == len(old_fixtures) == 72, "Need all 72 current/previous fixtures")
    require(current_specs == old_specs, "Broad case specifications differ from previous GPU evidence")
    require(current_fixtures == old_fixtures, "Broad arrays/oracles/fixture metadata differ from previous evidence")
    if previous_auto is not None:
        require(previous_auto.get("current_mode") == "auto" and previous_auto["status"] == "complete",
                "Previous auto evidence must be a complete automatic run")
        auto_fixtures, auto_specs = fixture_inventory(previous_auto)
        require(auto_fixtures == current_fixtures and auto_specs == current_specs,
                "Previous automatic fixtures differ from current broad panel")
        require(control["references"]["previous_auto"]["files"] == previous_auto["source_hashes_before"],
                "Previous automatic control source differs from preceding broad run")
    controlled, _ = fixture_inventory(control)
    overlap = set(controlled) & set(current_fixtures)
    require(len(overlap) == 42, "Expected 42 shared fixtures plus six expanded Gaussian controls")
    require(all(controlled[name] == current_fixtures[name] for name in overlap),
            "Control and broad share a changed fixture")
    require(control["references"]["previous_gpu"]["files"] == previous["source_hashes_before"],
            "Control frozen GPU differs from previous broad GPU source")
    target = broad_cells(broad, target_only=True)
    all_cells = broad_cells(broad)
    require(len(target) == 162 and len(all_cells) == 648, "Incomplete headline/target inventory")
    preservation = control_cells(control, preservation=True)
    controlled_targets = control_cells(control, preservation=False)
    require(len(preservation) == 60 and len(controlled_targets) == 162, "Incomplete control inventory")
    audits = control_audit_counts(control)
    for variant, phases in audits.items():
        require(phases["cold"]["audits"] == 48 and phases["probe"]["audits"] == 222
                and phases["retained"]["audits"] == 222 * 31, f"Missing control audits: {variant}")
        for phase, counts in phases.items():
            require(counts["audits"] == control["routing"][variant][phase]["request_count"],
                    f"Routing/audit count differs: {variant}/{phase}")
    for variant, quality in summary["quality_by_variant"].items():
        require(quality["all_phases"]["audit_count"] == 14472, f"Missing full broad audits: {variant}")
    return {
        "schema_version": 1, "evidence_validation_pass": True,
        "accuracy_and_integrity_pass": broad["accuracy_and_integrity_pass"] and control["acceptance_pass"]
        and (cold_summary is None or cold_summary["acceptance_pass"])
        and (first_query_summary is None or first_query_summary["acceptance_pass"]),
        "fixture_identity": {"all_72_broad_fixtures_match_previous": True,
                             "all_42_shared_control_fixtures_match_broad": True,
                             "extra_gaussian_controls": sorted(set(controlled) - overlap)},
        "current_and_previous_archive_bytes_verified": True,
        "native_cpu_extension": native,
        "reference_native_runtimes": reference_runtime_checks,
        "runtime_artifact": runtime,
        "fresh_process_cold": cold_summary,
        "initialized_process_first_query": first_query_summary,
        "control_order_policy": control.get("order_policy", "legacy"),
        "broad_routing": summary["routing"], "control_routing": control["routing"],
        "broad_accuracy": {v: q["all_phases"] for v, q in summary["quality_by_variant"].items()},
        "control_accuracy_counts": audits,
        "all_648_cells": threshold_counts(all_cells),
        "target_162_cells": target, "target_thresholds": threshold_counts(target),
        "controlled_target_162_cells": controlled_targets,
        "preservation_60_cells": preservation,
        "preservation_by_distribution": preservation_summary(preservation),
        "interpretation": "Speedup is comparator/current time. Every below-1 result is retained. "
        "Both independent automatic workers are compared with each GPU control on corresponding "
        "rounds; A/A ratios are shown directly. Slower in both replicas is descriptive, not a "
        "statistical significance claim. Accuracy failures never filter timing denominators.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--previous-broad", type=Path, required=True)
    parser.add_argument("--previous-auto-broad", type=Path)
    parser.add_argument("--cold", type=Path, help="Optional completed 18-case/five-launch fresh-process report")
    parser.add_argument("--first-query", type=Path, help="Optional initialized-process 18-case/30-fresh-model report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path,
                        help="Optional frozen native runtime ZIP; verify against measured source/binary")
    args = parser.parse_args()
    paths = {"broad": args.broad, "control": args.control, "previous_broad": args.previous_broad}
    if args.previous_auto_broad:
        paths["previous_auto_broad"] = args.previous_auto_broad
    if args.cold:
        paths["cold"] = args.cold
    if args.first_query:
        paths["first_query"] = args.first_query
    if args.runtime_archive:
        paths.update(runtime_archive=args.runtime_archive,
                     runtime_manifest=args.runtime_archive.with_suffix(".manifest.json"))
    hashes = {key: file_hash(path) for key, path in paths.items()}
    report = verify(read_report(args.broad), read_report(args.control), read_report(args.previous_broad),
                    broad_path=args.broad, broad_sha256=hashes["broad"], runtime_archive=args.runtime_archive,
                    previous_auto=read_report(args.previous_auto_broad) if args.previous_auto_broad else None,
                    cold=read_report(args.cold) if args.cold else None,
                    first_query=read_report(args.first_query) if args.first_query else None)
    require(hashes == {key: file_hash(path) for key, path in paths.items()}, "Raw evidence changed during verification")
    report["evidence"] = {key: {"path": str(path.resolve()), "sha256": hashes[key]}
                          for key, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "evidence_validation_pass": True,
                      "accuracy_and_integrity_pass": report["accuracy_and_integrity_pass"],
                      "target_slow_cell_counts": {
                          v: len(r["slower_or_missing_cells"]) for v, r in report["target_thresholds"].items()
                      }}, sort_keys=True))
    raise SystemExit(0 if report["accuracy_and_integrity_pass"] else 1)


if __name__ == "__main__":
    main()
