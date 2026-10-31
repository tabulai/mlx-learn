"""Reporting-only audit driver. Does not execute estimators or mutate raw evidence."""

import hashlib
import json
import os
import statistics
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.knn_auto_control_tables import validate
from benchmarks.knn_auto_verify import (
    check_archives,
    check_integrity,
    check_native_binary_metadata,
    check_runtime_artifact,
    control_audit_counts,
    file_hash,
    fixture_inventory,
)
from benchmarks.knn_broad_report import table
from benchmarks.knn_freeze_runtime import verify_runtime_archive
from benchmarks.knn_pipeline import source_hashes

PREFIX = ROOT / "benchmarks/results/knn-startup-final2-2026-10-30"
BASES = ("previous_auto", "current_gpu", "previous_gpu")
REPLICAS = ("current", "current_b")
METRICS = ("ratio_of_medians", "median_paired_ratio")
LABELS = {"current": "Current auto", "current_b": "Current auto A/A",
          "previous_auto": "Immediate prior auto", "current_gpu": "Current forced GPU",
          "previous_gpu": "Historical forced GPU", "sklearn_default": "Default-thread brute",
          "sklearn_1": "Single-thread brute", "sklearn_auto": "sklearn auto"}
FAMILIES = {"real_digits_raw_k5": "Digits", "real_breast_cancer_raw_k5": "Breast Cancer",
            "distribution_offset_1e4_4096x128_k5": "Offset +1e4, 4096×128",
            "distribution_gaussian_4096x128_k5": "Gaussian 4096×128",
            "distribution_clusters_4096x128_k5": "Clusters 4096×128"}


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def save_new(path, value):
    with Path(path).open("x") as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


def cells(raw):
    result = []
    for ci, case in enumerate(raw["cases"]):
        for operation, experiment in case["experiments"].items():
            samples = {v: experiment["variants"][v]["samples"] for v in raw["variants"]}
            assert all(len(s) == 31 for s in samples.values())
            medians = {v: statistics.median(s["ms"] for s in rows) for v, rows in samples.items()}
            comparisons = {}
            for base in BASES:
                comparisons[base] = {}
                for replica in REPLICAS:
                    ratios = [x["ms"] / y["ms"] for x, y in zip(samples[base], samples[replica], strict=True)]
                    comparisons[base][replica] = {"ratio_of_medians": medians[base] / medians[replica],
                        "median_paired_ratio": statistics.median(ratios), "all_paired_ratios": ratios}
            aa = [x["ms"] / y["ms"] for x, y in zip(samples["current_b"], samples["current"], strict=True)]
            result.append({"case": case["case"]["name"], "family": case["spec"]["family"],
                "seed": case["spec"]["seed"], "operation": operation, "median_ms": medians,
                "comparisons": comparisons,
                "auto_aa_over_auto": {"ratio_of_medians": medians["current_b"] / medians["current"],
                    "median_paired_ratio": statistics.median(aa), "all_paired_ratios": aa},
                "phase_diagnostics": experiment["phase_diagnostics"],
                "raw_pointer": f"/cases/{ci}/experiments/{operation}"})
    return result


def key(cell):
    return cell["case"], cell["operation"]


def flagged(cell, base, threshold=1 / 1.05):
    return all(cell["comparisons"][base][replica][metric] < threshold for replica in REPLICAS for metric in METRICS)


def ranges(selected):
    return {base: {replica: {metric: {
        "minimum": min(c["comparisons"][base][replica][metric] for c in selected),
        "maximum": max(c["comparisons"][base][replica][metric] for c in selected),
        "faster_count": sum(c["comparisons"][base][replica][metric] > 1 for c in selected),
        "cell_count": len(selected),
        "slower_cells": [c for c in selected if c["comparisons"][base][replica][metric] < 1],
    } for metric in METRICS} for replica in REPLICAS} for base in BASES}


def fmt(value):
    value_text = f"{value:.4f}×"
    return "**" + value_text + "**" if value < 1 else value_text


def row_label(cell):
    return [FAMILIES.get(cell["family"], cell["family"]), cell["seed"], cell["operation"]]


def compact(cell, base, replica):
    value = cell["comparisons"][base][replica]
    return fmt(value["ratio_of_medians"]) + " / " + fmt(value["median_paired_ratio"])


def render_repeat(audit, output, title, protocol_name, predeclaration_path):
    rows = audit["all_cells"]
    variants = audit["variants"]
    link = lambda path, label: f"[{label}]({os.path.relpath(path, output.parent)})"
    lines = [f"# {title}",
        f"All **{len(rows)} operation cells**, both automatic replicas and all five workers are retained. "
        "Accuracy, runtime identity and numerical-oracle validation pass. The two isolated >5% full-control "
        "flags were covered by these predeclared panels; the panel-specific crosschecks below retain "
        "both the full result and the repeat, including smaller or inconsistent slowdowns.",
        f"This is the `{protocol_name}` portion of the " + link(predeclaration_path, "saved predeclaration") +
        f" (SHA-256 `{audit['predeclaration_sha256']}`). The original Digits/Breast/offset panel was retained, "
        "and all six Gaussian 128D seeds plus all three clustered 128D seeds were added before the final "
        "preservation results. The exact requested inventories are verified, with no seed or operation removed.",
        "Each operation uses 31 serial interleaved rounds across current auto, identical current auto A/A, "
        "current forced GPU, immediate prior auto (`knn-latency-final-2026-10-30` plus its exact archived native "
        "runtime), and historical forced GPU (`knn-small-final-2026-10-30`). This panel uses the legacy "
        "forward/reverse ordering and 1 second of continuous warmup. The full 48-fixture control uses balanced "
        "Williams ordering and 0.25 seconds of warmup. Absolute milliseconds across these runs are not directly "
        "comparable; ratios compare implementations within their own run. Neither run replaces the other, "
        "and no faster window or run is selected.",
        "M/P means ratio of medians / median paired ratio. Ratios are comparator/current; below 1× means "
        "the automatic replica is slower and is bold. A >5% runtime slowdown means ratio <1/1.05 for BOTH "
        "statistics and BOTH independent automatic replicas. This recurrence screen is descriptive, not a "
        "confidence interval or significance test.", "## Full-control exceptions and repeat crosscheck"]
    counts, cross = [], []
    for base, entry in audit["cross_run_flags"].items():
        counts.append([LABELS[base], entry["full_control_flag_count"], entry["selected_full_flag_count"],
                       len(entry["recurring"]), len(entry["not_recurring"]), len(entry["new_in_repeat"])])
        for category in ("recurring", "not_recurring", "new_in_repeat"):
            for value in entry[category]:
                old, new = value["full"], value["repeat"]
                cross.append([LABELS[base], category.replace("_", " "), *row_label(new),
                    compact(old, base, "current"), compact(old, base, "current_b"),
                    compact(new, base, "current"), compact(new, base, "current_b")])
    lines += [table(["Comparator", "Flags in full 222 cells", "Covered here", "Recurring", "Not recurring", "New flags"], counts),
        table(["Comparator", "Classification", "Family", "Seed", "Operation", "Full current M/P", "Full A/A M/P",
               "Repeat current M/P", "Repeat A/A M/P"], cross) if cross else "No >5% flags in this matched panel."]
    slow = [[LABELS[base], LABELS[replica], *row_label(cell), compact(cell, base, replica)]
            for cell in rows for base in BASES for replica in REPLICAS
            if cell["comparisons"][base][replica]["ratio_of_medians"] < 1
            or cell["comparisons"][base][replica]["median_paired_ratio"] < 1]
    lines += ["## Every slower observation in either statistic",
        table(["Comparator", "Replica", "Family", "Seed", "Operation", "M/P"], slow)
        if slow else "No below-1 ratios in either statistic for either replica.",
        "## Every repeat timing cell", "All times are median milliseconds from this repeat only.",
        table(["Family", "Seed", "Operation", *(LABELS[v] + " ms" for v in variants), "A/A over auto M/P"], [
            [*row_label(c), *(f"{c['median_ms'][v]:.6f}" for v in variants),
             fmt(c["auto_aa_over_auto"]["ratio_of_medians"]) + " / " + fmt(c["auto_aa_over_auto"]["median_paired_ratio"])] for c in rows])]
    for base in BASES:
        lines += ["## Every cell against " + LABELS[base],
            table(["Family", "Seed", "Operation", "Current M", "Current P", "A/A M", "A/A P"], [
                [*row_label(c), *(fmt(c["comparisons"][base][replica][metric]) for replica in REPLICAS for metric in METRICS)] for c in rows])]
    lines += ["## Accuracy and source identity",
        "All cold, warmup, probe and retained strict audits remain in the raw evidence. Every worker passes; "
        "no accuracy failure filters a timing denominator. Each repeat fixture and oracle record exactly "
        "matches the corresponding full-control fixture. Source, runtime and harness archive bytes, measured "
        "worker identities, and current source/native/harness files have been independently verified.",
        table(["Worker", "Cold passes/audits", "Warmup passes/audits", "Probe passes/audits", "Retained passes/audits"], [
            [LABELS[v], *(f"{audit['audit_counts'][v][p]['strict_passes']}/{audit['audit_counts'][v][p]['audits']}"
                         for p in ("cold", "warmup", "probe", "retained"))] for v in variants]),
        "All raw paired ratios, actual routing, first/middle/last summaries, position and predecessor "
        "diagnostics remain in the independent audit. They diagnose timing sensitivity without claiming "
        "an identified clock mechanism.", "## Evidence",
        link(Path(audit["raw_evidence"]["path"]), "Every raw observation") + f" — SHA-256 `{audit['raw_evidence']['sha256']}`.",
        link(Path(audit["audit_path"]), "Independent audit and complete crossmatching") + ".",
        link(Path(str(PREFIX) + "-control.json"), "Full balanced control") + "."]
    result = "\n\n".join(lines) + "\n"
    # All requested operation cells must appear in each complete timing/ratio table.
    for section in result.split("\n\n## "):
        if section.startswith("Every repeat timing cell") or section.startswith("Every cell against"):
            body_rows = [line for line in section.splitlines() if any(line.startswith("| " + family + " |") for family in FAMILIES.values())]
            assert len(body_rows) == len(rows)
    save_new(output, result)


def main():
    full_path = Path(str(PREFIX) + "-control.json")
    full = load(full_path)
    validate(full, expected_cases=48)
    full_cells = cells(full)
    full_map = {key(c): c for c in full_cells}
    full_fixtures, _ = fixture_inventory(full)
    source_files = full["source_hashes_before"]
    assert source_hashes(ROOT / "src") == source_files
    broad = load(str(PREFIX) + "-broad.json")
    assert broad["source_hashes_before"] == source_files
    declaration_path = ROOT / "benchmarks/results/knn-startup-final2-legacy-predeclaration-2026-10-30.json"
    declaration = load(declaration_path)
    declaration_hash = file_hash(declaration_path)
    harness_path = ROOT / "benchmarks/reference/knn-startup-final2-2026-10-30-harnesses.zip"
    harness = load(harness_path.with_suffix(".manifest.json"))
    assert file_hash(harness_path) == harness["archive"]["sha256"]
    with zipfile.ZipFile(harness_path) as archive:
        assert set(archive.namelist()) == set(harness["files"])
        assert all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in harness["files"].items())
    assert all(file_hash(ROOT / path) == value for path, value in harness["files"].items())
    device_checks = {}
    for baseline in ("knn-small-final-2026-10-30", "knn-latency-final-2026-10-30"):
        previous = load(ROOT / "benchmarks/reference" / (baseline + ".manifest.json"))
        current_device = {k: v for k, v in source_files.items() if k.startswith("mlxlearn_device/")}
        assert current_device == {k: v for k, v in previous["files"].items() if k.startswith("mlxlearn_device/")}
        device_checks[baseline] = {"all_device_files_identical": True, "file_count": len(current_device)}
    repeats = {}
    for plan in declaration["runs"]:
        name = plan["name"]
        raw_path = Path(str(PREFIX) + "-" + name + ".json")
        raw = load(raw_path)
        validate(raw, expected_cases=plan["case_count"])
        assert sorted(c["case"]["name"] for c in raw["cases"]) == sorted(plan["case_names"])
        assert raw["planned_case_operation_count"] == plan["operation_cells"]
        assert raw["order_policy"] == declaration["order"] == "legacy"
        assert raw["pairs"] == declaration["pairs"] == 31
        assert raw["warmup_seconds_per_variant_operation"] == declaration["warmup_seconds"] == 1
        check_integrity(raw, control=True)
        check_archives(broad, raw)
        assert raw["source_hashes_before"] == source_files
        assert raw["references"] == full["references"] and raw["reference_runtimes"] == full["reference_runtimes"]
        fixtures, _ = fixture_inventory(raw)
        assert all(f == full_fixtures[k] for k, f in fixtures.items())
        native = check_native_binary_metadata(broad, raw)
        runtime = check_runtime_artifact(ROOT / "benchmarks/reference/knn-startup-final2-2026-10-30-runtime.zip", broad, native)
        for variant, reference in raw["reference_runtimes"].items():
            assert verify_runtime_archive(raw["reference_runtime_archives"][variant]) == reference
        for path, expected in raw["harness_hashes_before"].items():
            path = Path(path)
            relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
            assert harness["files"][relative] == expected == file_hash(ROOT / relative)
        repeat_cells = cells(raw)
        repeat_map = {key(c): c for c in repeat_cells}
        flags = {}
        for base in BASES:
            full_flags = {key(c) for c in full_cells if flagged(c, base)}
            selected = full_flags & set(repeat_map)
            new = {key(c) for c in repeat_cells if flagged(c, base)}
            cross = lambda keys: [{"full": full_map[k], "repeat": repeat_map[k]} for k in sorted(keys)]
            flags[base] = {"full_control_flag_count": len(full_flags), "selected_full_flag_count": len(selected),
                "recurring": cross(selected & new), "not_recurring": cross(selected - new),
                "new_in_repeat": cross(new - selected), "full_flags_not_covered_here": [full_map[k] for k in sorted(full_flags - set(repeat_map))]}
        audit_path = Path(str(PREFIX) + "-" + name + "-audit.json")
        audit = {"schema_version": 1, "validation_pass": True, "acceptance_pass": raw["acceptance_pass"],
            "variants": raw["variants"], "case_count": len(raw["cases"]), "operation_cells": len(repeat_cells),
            "order": "legacy", "pairs": 31, "warmup_seconds": 1,
            "predeclaration_sha256": declaration_hash, "declared_inventory_matches": True,
            "audit_counts": control_audit_counts(raw), "routing": raw["routing"],
            "native": native, "runtime": runtime, "source_fixture_harness_archive_and_disk_identity_pass": True,
            "cross_run_flags": flags, "all_cells": repeat_cells,
            "ranges": ranges(repeat_cells), "family_ranges": {
                family: ranges([c for c in repeat_cells if c["family"] == family]) for family in sorted({c["family"] for c in repeat_cells})},
            "threshold": "Comparator/current ratio <1/1.05 in both statistics and both independent automatic replicas; descriptive, not significance.",
            "raw_evidence": {"path": str(raw_path), "sha256": file_hash(raw_path)},
            "full_control_sha256": file_hash(full_path), "audit_path": str(audit_path),
            "driver": {"path": str(Path(__file__).resolve()), "sha256": file_hash(__file__)}}
        assert audit["acceptance_pass"] and all(v["audits"] == v["strict_passes"] for phases in audit["audit_counts"].values() for v in phases.values())
        save_new(audit_path, audit)
        suffix = "legacy-control" if name == "legacy-control" else "preservation-repeat"
        output = ROOT / "docs" / ("knn-startup-" + suffix + "-2026-10-30.md")
        title = "Predeclared Digits, Breast Cancer and offset legacy-order control" if name == "legacy-control" else "Predeclared Gaussian and clustered 128D preservation repeat"
        render_repeat(audit, output, title, name, declaration_path)
        repeats[name] = {"audit_path": str(audit_path), "report_path": str(output),
                        "audit_sha256": file_hash(audit_path), "report_sha256": file_hash(output),
                        "cross_run_flag_counts": {b: {"selected_full": e["selected_full_flag_count"],
                            "recurring": len(e["recurring"]), "not_recurring": len(e["not_recurring"]),
                            "new": len(e["new_in_repeat"])} for b, e in flags.items()},
                        "audit_counts": audit["audit_counts"]}
        print(json.dumps({"panel": name, "validation_pass": True, "flags": repeats[name]["cross_run_flag_counts"]}), flush=True)
        del raw

    independent = load(str(PREFIX) + "-independent-audit.json")
    cold = load(str(PREFIX) + "-cold.json")
    first = load(str(PREFIX) + "-first-query.json")
    all_measured_harnesses = {"broad": broad, "balanced_control": full, "cold": cold, "first_query": first}
    for report in all_measured_harnesses.values():
        for path, expected in report["harness_hashes_before"].items():
            path = Path(path)
            relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
            assert harness["files"][relative] == expected == file_hash(ROOT / relative)
    cold_metrics = {}
    for metric in ("query_ms", "fit_query_ms", "estimator_setup_ms", "estimator_to_result_ms", "process_start_to_result_ms"):
        cold_metrics[metric] = {}
        for variant in cold["variants"]:
            rows = []
            for case in cold["cases"]:
                current = statistics.median(s["timings"][metric] for s in case["variants"]["current"]["samples"])
                other = statistics.median(s["timings"][metric] for s in case["variants"][variant]["samples"])
                rows.append({"case": case["case"]["name"], "current_ms": current, "comparator_ms": other,
                             "ratio": other / current, "milliseconds_saved": other - current,
                             "fractional_time_reduction": 1 - current / other})
            cold_metrics[metric][variant] = {"all_cells": rows, "cell_count": len(rows),
                "current_median_ms_range": [min(r["current_ms"] for r in rows), max(r["current_ms"] for r in rows)],
                "comparator_median_ms_range": [min(r["comparator_ms"] for r in rows), max(r["comparator_ms"] for r in rows)],
                "ratio_range": [min(r["ratio"] for r in rows), max(r["ratio"] for r in rows)],
                "milliseconds_saved_range": [min(r["milliseconds_saved"] for r in rows), max(r["milliseconds_saved"] for r in rows)],
                "fractional_time_reduction_range": [min(r["fractional_time_reduction"] for r in rows), max(r["fractional_time_reduction"] for r in rows)],
                "faster_medians": sum(r["ratio"] > 1 for r in rows), "slower_cells": [r for r in rows if r["ratio"] < 1]}
    cold_imports = independent["fresh_process_cold"]["sklearn_neighbors_loaded_before_fit_counts"]
    first_imports = independent["initialized_process_first_query"]["sklearn_neighbors_before_fit"]
    assert cold_imports["current"] == {"False": 90}
    assert first_imports["current"] == {"False": 540}
    assert all(s["loaded_modules"]["before_fit"]["sklearn.neighbors"] is False for c in cold["cases"] for s in c["variants"]["current"]["samples"])
    assert first["setup"]["current"]["loaded_modules_after_imports"]["sklearn.neighbors"] is False
    assert all(s["loaded_modules_before_fit"]["sklearn.neighbors"] is False for c in first["cases"] for s in c["variants"]["current"]["samples"])
    first_summary = independent["initialized_process_first_query"]
    first_compact = {v: {metric: {k: value for k, value in entry.items() if k != "all_cells"}
                        for metric, entry in values.items()} for v, values in first_summary["comparisons"].items()}
    final = {"schema_version": 1, "validation_pass": True,
        "harness_archive_sha256": file_hash(harness_path), "all_harness_archive_and_current_file_count": len(harness["files"]),
        "current_source_harness_runtime_identity_pass": True, "device_source_identity": device_checks,
        "repeat_panels": repeats, "cold_latency": cold_metrics,
        "first_query_comparisons": first_compact,
        "stock_neighbor_imports": {"cold_before_fit": cold_imports, "first_query_before_fit": first_imports,
                                  "current_first_query_after_imports": False},
        "precision": {"broad": independent["broad_accuracy"], "balanced_control": independent["control_accuracy_counts"],
                      "cold": independent["fresh_process_cold"]["audit_counts"],
                      "first_query": first_summary["audit_counts"],
                      **{name: value["audit_counts"] for name, value in repeats.items()}},
        "full_control_flags": {b: [c for c in full_cells if flagged(c, b)] for b in BASES},
        "driver": {"path": str(Path(__file__).resolve()), "sha256": file_hash(__file__)},
        "interpretation": "Cold metrics include normal public imports in setup and total latency; initialized-process fresh-model "
        "metrics deliberately separate startup. Repeats use legacy order and longer warmup, preserve all raw observations, "
        "and are not replacements for balanced results. No blanket zero-regression or universal2x claim is made."}
    save_new(Path(str(PREFIX) + "-additional-audit.json"), final)
    print(json.dumps({"startup_prior_auto": {m: {k: v for k, v in values["previous_auto"].items() if k not in ("all_cells", "slower_cells")}
        for m, values in cold_metrics.items()}, "first_query_wins": {v: {m: e["faster_medians"] for m, e in values.items()} for v, values in first_compact.items()},
        "driver_sha256": file_hash(__file__)}), flush=True)


if __name__ == "__main__":
    main()
