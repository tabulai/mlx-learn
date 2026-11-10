"""Render the complete current sklearn refresh without running numerical code.

Only the standard library is imported. Ratios are recomputed from recorded
medians; failed quality checks never remove a timing or become qualified wins.
Use --allow-missing-panels only while developing the report, never for final
evidence. This module does not import any estimator or benchmark runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PANELS = (
    "knn-broad-summary", "knn-cold-summary", "knn-first-query-summary",
    "svc-seed41", "svc-seed42", "lr-default", "lr-requested1",
    "lr-strict4096-default", "lr-strict4096-requested1", "grid-default-seed41",
)
OPTIONAL_PANELS = ("lr-accuracy-matched-default", "lr-accuracy-matched-requested1")
VARIANTS = ("direct_auto", "patched_auto", "forced_gpu")
NAMES = {
    "direct_auto": "Direct auto", "patched_auto": "Patched auto",
    "forced_gpu": "Forced GPU", "forced_cpu": "Forced CPU",
    "sklearn": "sklearn", "sklearn_brute": "sklearn brute/default",
    "sklearn_default": "sklearn brute/default",
    "sklearn_auto": "sklearn auto/default",
    "sklearn_1": "sklearn brute/threadpoolctl-limited",
}
SVC_CASES = {"rbf_narrow", "rbf_wide", "linear", "poly", "sigmoid",
             "rbf_multiclass", "rbf_bounded_cache", "linear_medium"}
LR_CHECK_NAMES = {
    "completed": "fit/scoring did not complete", "finite": "nonfinite/missing score",
    "gradient": "gradient exceeds tolerance", "objective": "objective gap exceeds tolerance",
    "predictions": "prediction agreement below threshold", "convergence": "nonconvergence",
    "backend": "unexpected backend", "solver_observed": "solver provenance missing",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(reference, current):
    return reference / current if finite(reference) and finite(current) and reference > 0 and current > 0 else None


def clean(value):
    """JSON output is strict even when a raw failure records NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    if not finite(value):
        return "missing"
    return f"{value:.4g}"


def ratio_text(value):
    return number(value) + "×" if finite(value) else "missing"


def interval(low, high, *, ratios=False):
    formatter = ratio_text if ratios else number
    return formatter(low) if low == high else formatter(low) + "–" + formatter(high)


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def linked(path, destination, label):
    relative = os.path.relpath(Path(path).resolve(), destination.parent.resolve())
    return f"[{label}]({relative})"


def backend_counts(details):
    return dict(Counter(str(item.get("backend") or "unknown") for item in details))


def backend_text(counts):
    return ", ".join(f"{key}: {count}" for key, count in sorted(counts.items())) or "unknown"


def lr_ms(block, variant):
    return (block.get("timings", {}).get(variant, {}).get("summary") or {}).get("median_ms")


def value_range(values):
    values = [value for value in values if finite(value)]
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def lr_scores(block, variant):
    scores = [item.get("score") or {} for item in block.get("quality_each_fit", {}).get(variant, [])]
    return {key: value_range(item.get(key) for item in scores) for key in (
        "objective_float64", "gradient_max_abs_float64", "objective_relative_gap_to_sklearn",
        "prediction_agreement_sklearn")}


def lr_fit_reasons(row, variant):
    if "error" in row:
        return ["case error: " + str(row["error"])]
    block = row.get("fit", {})
    reasons = set()
    audits = block.get("quality_each_fit", {}).get(variant, [])
    if not audits:
        reasons.add("fit audits missing")
    for audit in audits:
        for check, passed in audit.get("checks", {}).items():
            if not passed:
                reasons.add(LR_CHECK_NAMES.get(check, check))
        if audit.get("scoring_error"):
            reasons.add("scoring error: " + str(audit["scoring_error"]))
        if not audit.get("own_checks_pass", False) and not audit.get("checks"):
            reasons.add("fit quality failed")
    for detail in block.get("fit_details", {}).get(variant, []):
        if detail.get("error"):
            reasons.add("fit error: " + str(detail["error"]))
        for warning in detail.get("warnings", []):
            if "Convergence" in warning.get("category", ""):
                reasons.add(warning["category"] + ": " + warning["message"])
    if not block.get("timings", {}).get(variant, {}).get("complete", False):
        reasons.add("incomplete fit timings")
    return sorted(reasons)


def lr_inference_reasons(block, variant):
    reasons = set()
    for audit in block.get("quality_each_call", {}).get(variant, []):
        if audit.get("passed", False):
            continue
        reasons.add(audit.get("reason", "inference output check failed"))
        for key in ("probabilities_valid", "probabilities_close"):
            if audit.get(key) is False:
                reasons.add(key.replace("_", " ") + " failed")
        agreement = audit.get("prediction_agreement_sklearn")
        if finite(agreement) and agreement < .995:
            reasons.add("inference prediction agreement below 0.995")
    for detail in block.get("call_details", {}).get(variant, []):
        if detail.get("error"):
            reasons.add("inference error: " + str(detail["error"]))
    if not block.get("timings", {}).get(variant, {}).get("complete", False):
        reasons.add("incomplete inference timings")
    return sorted(reasons)


def measurement(*, panel, case, operation, variant, baseline, current_ms, reference_ms,
                qualified, reasons=(), dtype=None, seed=None, queries=None,
                backends=None, shape=None, policy=None):
    raw = ratio(reference_ms, current_ms)
    reasons = set(reasons)
    if raw is None:
        reasons.add("missing/nonpositive/nonfinite timing")
    if not qualified and not reasons:
        reasons.add("recorded quality gate failed")
    return {
        "panel": panel, "case": case, "operation": operation, "variant": variant,
        "baseline": baseline, "dtype": dtype, "seed": seed, "queries": queries,
        "shape": shape, "policy": policy, "fit_backend_counts": backends or {},
        "current_ms": current_ms, "reference_ms": reference_ms, "raw_speedup": raw,
        "qualified": bool(qualified and raw is not None and not reasons), "failure_causes": sorted(reasons),
    }


def parse_logistic(name, raw):
    rows, measurements = [], []
    policy = "requested1" if name.endswith("requested1") else "default"
    for row in raw["results"]:
        case = row["case"]
        stock_reasons = lr_fit_reasons(row, "sklearn")
        fits = row.get("fit", {})
        parsed = {"case": case, "dtype": row["dtype"], "seed": row["seed"],
                  "sklearn_ms": lr_ms(fits, "sklearn"), "variants": {},
                  "sklearn_failure_causes": stock_reasons, "sklearn_scores": lr_scores(fits, "sklearn"),
                  "input_sha256": row.get("input_sha256")}
        for variant in VARIANTS:
            own = lr_fit_reasons(row, variant)
            reasons = own + ["sklearn reference: " + reason for reason in stock_reasons]
            details = fits.get("fit_details", {}).get(variant, [])
            backends = backend_counts(details)
            fit = measurement(
                panel=name, case=case["name"], operation="fit", variant=variant,
                baseline="sklearn", current_ms=lr_ms(fits, variant),
                reference_ms=parsed["sklearn_ms"],
                qualified=fits.get("speedups", {}).get(variant, {}).get("quality_qualified", False)
                and raw.get("source_integrity_pass", False),
                reasons=reasons, dtype=row["dtype"], seed=row["seed"], policy=policy,
                shape=[case["n_samples"], case["n_features"]], backends=backends)
            if not raw.get("source_integrity_pass", False):
                fit["failure_causes"].append("source integrity failed/incomplete")
            fit["implementations"] = dict(Counter(d.get("implementation", "unknown") for d in details))
            fit["quality_scores"] = lr_scores(fits, variant)
            parsed["variants"][variant] = fit
            measurements.append(fit)
            inference = row.get("inference", {})
            query_counts = raw["parameters"].get("queries") or [int(q) for q in inference]
            for query_count in query_counts:
                operations = inference.get(str(query_count), {})
                for operation in ("predict", "predict_proba"):
                    block = operations.get(operation, {})
                    causes = ["fit prerequisite: " + reason for reason in reasons]
                    causes += lr_inference_reasons(block, variant)
                    causes += ["sklearn reference: " + reason for reason in lr_inference_reasons(block, "sklearn")]
                    if not raw.get("source_integrity_pass", False):
                        causes.append("source integrity failed/incomplete")
                    entry = measurement(
                        panel=name, case=case["name"], operation=operation, variant=variant,
                        baseline="sklearn", current_ms=lr_ms(block, variant), reference_ms=lr_ms(block, "sklearn"),
                        qualified=block.get("speedups", {}).get(variant, {}).get("quality_qualified", False)
                        and raw.get("source_integrity_pass", False), reasons=causes,
                        dtype=row["dtype"], seed=row["seed"], queries=int(query_count),
                        shape=fit["shape"], backends=backends, policy=policy)
                    entry["inference_execution"] = "public host-array inference; fit backend does not imply GPU inference"
                    entry["probability_difference_range"] = value_range(
                        audit.get("max_probability_difference_sklearn")
                        for audit in block.get("quality_each_call", {}).get(variant, []))
                    measurements.append(entry)
        rows.append(parsed)
    return {"fit_rows": rows, "measurements": measurements, "parameters": raw["parameters"],
            "thread_environment": raw.get("thread_environment_before_import", {}),
            "source_integrity_pass": raw.get("source_integrity_pass", False),
            "sessions": raw.get("sessions", []), "complete": raw.get("complete", False),
            "audit_schema_version": raw.get("schema_version"), "quality_certificate": raw.get("quality_certificate"),
            "production_and_native_hashes": {path: value for path, value in raw.get("source_integrity_before", {}).items()
                                              if path.startswith("src/")}}


def boolean_failures(audit):
    return [key.replace("_", " ") for key, value in audit.items()
            if isinstance(value, bool) and not value
            and key not in {"passed", "valid_exact_tie_alternative", "indices_match_direct_float64"}]


def grid_integrity_reasons(result, identity):
    reasons = set()
    if result.get("source_identity") != identity.get("sources"):
        reasons.add("worker source/native integrity mismatch")
    environment = result.get("environment", {})
    for field in ("python", "platform", "thread_policy"):
        if environment.get(field) != identity.get(field):
            reasons.add("worker runtime mismatch: " + field)
    for package, field in (("numpy", "numpy"), ("scipy", "scipy"), ("scikit-learn", "sklearn"), ("mlx", "mlx")):
        if environment.get(field) != identity.get("packages", {}).get(package):
            reasons.add("worker package mismatch: " + package)
    if len(result.get("warmup_trials", [])) != identity["warmup"]:
        reasons.add("warmup audit count mismatch")
    if len(result.get("measured_trials", [])) != identity["repeats"]:
        reasons.add("retained audit count mismatch")
    return reasons


def grid_timing_reasons(result, variant, repeats):
    timing = result.get("timings", {}).get(variant, {})
    samples = timing.get("samples_s", [])
    reasons = set()
    if len(samples) != repeats or not all(finite(value) and value > 0 for value in samples):
        reasons.add(variant + ": missing/invalid timing samples")
    elif not finite(timing.get("median_s")) or not math.isclose(
        timing["median_s"], statistics.median(samples), rel_tol=1e-12
    ):
        reasons.add(variant + ": median differs from samples")
    return reasons


def parse_grid(name, raw):
    rows, measurements = [], []
    for case in raw["identity"]["cases"]:
        result = raw["results"].get(case["id"])
        parsed = {"case": case, "variants": {}, "reference_ms": {},
                  "recorded_qualified_speedup": (result or {}).get("qualified_speedup"),
                  "recorded_case_quality_passed": (result or {}).get("quality_passed", False)}
        baselines = ("sklearn_brute", "sklearn_auto") if case["algorithm"] == "knn" else ("sklearn",)
        failure_causes = defaultdict(set)
        if result is None:
            for variant in ("direct_auto", "patched_auto", *baselines):
                failure_causes[variant].add("worker error: " + str(raw.get("errors", {}).get(case["id"], "missing result")))
        else:
            for trial in result["warmup_trials"] + result["measured_trials"]:
                for variant, sample in trial.items():
                    audit = sample["audit"]
                    if not audit.get("passed", False):
                        reasons = boolean_failures(audit) or ["quality audit failed"]
                        failure_causes[variant].update(variant + ": " + reason for reason in reasons)
                    for warning in sample.get("model", {}).get("warnings", []):
                        if "ConvergenceWarning" in warning.get("category", ""):
                            failure_causes[variant].add(variant + ": " + warning["message"])
            for variant, setup in result.get("setup_models", {}).items():
                for warning in setup.get("warnings", []):
                    if "ConvergenceWarning" in warning.get("category", ""):
                        failure_causes[variant].add(variant + " setup: " + warning["message"])
        for baseline in baselines:
            seconds = (result or {}).get("timings", {}).get(baseline, {}).get("median_s")
            parsed["reference_ms"][baseline] = seconds * 1000 if finite(seconds) else None
        for variant in ("direct_auto", "patched_auto"):
            seconds = (result or {}).get("timings", {}).get(variant, {}).get("median_s")
            current_ms = seconds * 1000 if finite(seconds) else None
            details = [trial[variant]["model"] for trial in (result or {}).get("measured_trials", [])]
            backends = backend_counts(details)
            query_backends = dict(Counter(str(d.get("query_backend")) for d in details))
            parsed["variants"][variant] = {}
            for baseline in baselines:
                causes = failure_causes[variant] | failure_causes[baseline]
                if result is not None:
                    causes |= grid_integrity_reasons(result, raw["identity"])
                    causes |= grid_timing_reasons(result, variant, raw["identity"]["repeats"])
                    causes |= grid_timing_reasons(result, baseline, raw["identity"]["repeats"])
                entry = measurement(
                    panel=name, case=case["id"], operation=case["algorithm"] + "/" + case["operation"],
                    variant=variant, baseline=baseline, current_ms=current_ms,
                    reference_ms=parsed["reference_ms"][baseline],
                    qualified=result is not None and not causes, reasons=causes,
                    dtype=(result or {}).get("input_dtype", "float64"), seed=raw["identity"]["seed"],
                    queries=None if case["operation"] == "fit" else (
                        case["n_queries"] if case["algorithm"] == "knn" else (result or {}).get("query_rows")),
                    shape=[case["n_samples"], case["n_features"]], policy=raw["identity"]["thread_policy"],
                    backends=backends)
                entry["query_backend_counts"] = query_backends
                entry["recorded_case_quality_passed"] = (result or {}).get("quality_passed", False)
                entry["qualification_policy"] = "recomputed per current/comparator pair from both audits, timing counts and recorded source/runtime integrity; original aggregate gate retained"
                parsed["variants"][variant][baseline] = entry
                measurements.append(entry)
        rows.append(parsed)
    return {"rows": rows, "measurements": measurements, "complete": raw.get("complete", False),
            "errors": raw.get("errors", {}),
            "failed_attempts": [a for a in raw.get("attempts", []) if a.get("returncode")],
            "identity": raw["identity"]}


def parse_svc(name, raw):
    rows, measurements = [], []
    for case in raw["cases"]:
        reasons = set()
        for variant, samples in case["quality_samples_including_first"].items():
            if variant == "previous":
                continue
            for audit in samples:
                if audit.get("fit_status") != 0:
                    reasons.add(variant + ": nonconvergence")
                for check in ("first_repeat_predictions_equal", "all_margins_finite"):
                    if not audit.get(check, False):
                        reasons.add(variant + ": " + check.replace("_", " "))
                if variant == "current":
                    for baseline in ("sklearn",):
                        if audit.get("agreement_with_" + baseline, 0) < case["quality_thresholds"]["prediction_agreement"]:
                            reasons.add(baseline + " comparison: prediction agreement below threshold")
                        if not audit.get("margins_close_to_" + baseline, False):
                            reasons.add(baseline + " comparison: decision margin mismatch")
        current_worker = case["workers"]["current"]
        stock_worker = case["workers"]["sklearn"]
        current_env, stock_env = current_worker["environment"], stock_worker["environment"]
        if current_env.get("source_sha256") != raw["current_source_sha256"]:
            reasons.add("current source/native provenance mismatch")
        if not current_env.get("native_extensions"):
            reasons.add("native module provenance missing")
        if any(audit.get("backend") != case["backend"] for audit in case["quality_samples_including_first"]["current"]):
            reasons.add("observed backend differs from requested backend")
        for field in ("python", "numpy", "scipy", "sklearn", "mlx", "platform", "thread_environment"):
            if current_env.get(field) != stock_env.get(field):
                reasons.add("worker runtime mismatch: " + field)
        for worker in (current_worker, stock_worker):
            if worker.get("fixture_sha256") != case["fixture_sha256"]:
                reasons.add("worker fixture mismatch")
            if worker["environment"].get("harness_sha256") != raw["harness_sha256"]:
                reasons.add("worker harness mismatch")
        for extension in current_env.get("native_extensions", {}).values():
            try:
                relative = Path(extension["path"]).relative_to(current_env["source_root"]).as_posix()
            except ValueError:
                reasons.add("native module outside source root")
                continue
            if raw["current_extension_sha256"].get(relative) != extension["sha256"]:
                reasons.add("native binary mismatch")
        actual = backend_counts(case["quality_samples_including_first"]["current"][1:])
        variant = "forced_gpu" if case["backend"] == "mlx" else "forced_cpu"
        parsed = {"case": case["name"], "seed": case["seed"], "shape": case["shape"],
                  "cache_size_mb": case["cache_size_mb"], "classes": case["classes"],
                  "backends": actual, "metrics": {}, "first_call": case["first_call"],
                  "historical_control_retained": True, "recorded_case_quality_passed": case["quality_passed"]}
        for operation in ("fit", "first_predict", "repeat_predict"):
            metric_reasons = set(reasons)
            for variant_name in ("current", "sklearn"):
                timing = case[operation][variant_name]
                samples = timing["samples_s"]
                if len(samples) != raw["repeats"] or not all(finite(s) and s > 0 for s in samples):
                    metric_reasons.add(variant_name + ": missing/invalid timing samples")
                elif not math.isclose(timing["median_ms"], statistics.median(samples) * 1000, rel_tol=1e-12):
                    metric_reasons.add(variant_name + ": median differs from samples")
            entry = measurement(
                panel=name, case=case["name"], operation=operation, variant=variant, baseline="sklearn",
                current_ms=case[operation]["current"]["median_ms"], reference_ms=case[operation]["sklearn"]["median_ms"],
                qualified=not metric_reasons, reasons=metric_reasons, dtype="float32", seed=case["seed"],
                queries=None if operation == "fit" else case["query_rows"], shape=case["shape"],
                backends=actual, policy="pre-import thread limits 1")
            parsed["metrics"][operation] = entry
            measurements.append(entry)
        rows.append(parsed)
    return {"rows": rows, "measurements": measurements, "timing_scope": raw["timing_scope"],
            "seed": raw["seed"], "repeats": raw["repeats"]}


def summarize_measurements(entries):
    """Descriptive cell ratios; never pool operations or thread policies."""
    groups = defaultdict(list)
    keys = ("panel", "operation", "variant", "baseline", "dtype", "queries", "policy")
    for entry in entries:
        groups[tuple(entry.get(key) for key in keys)].append(entry)
    summaries = []
    for identity, rows in groups.items():
        raw = [r["raw_speedup"] for r in rows if finite(r["raw_speedup"])]
        qualified = [r["raw_speedup"] for r in rows if r["qualified"]]
        summaries.append(dict(zip(keys, identity, strict=True)) | {
            "cells": len(rows), "available_timings": len(raw),
            "raw_wins": sum(value > 1 for value in raw),
            "raw_median_ratio": statistics.median(raw) if raw else None,
            "raw_min_ratio": min(raw) if raw else None, "raw_max_ratio": max(raw) if raw else None,
            "qualified_cells": len(qualified), "qualified_wins": sum(value > 1 for value in qualified),
            "qualified_median_ratio": statistics.median(qualified) if qualified else None,
            "qualified_min_ratio": min(qualified) if qualified else None,
            "qualified_max_ratio": max(qualified) if qualified else None,
        })
    return summaries


def parse_knn(name, raw):
    result = {"case_count": raw["case_count"], "repeats": raw["repeats"], "operations": []}
    if name == "knn-broad-summary":
        result.update(quality=raw["quality_by_variant"], routing=raw["routing"],
                      acceptance=raw["accuracy_and_integrity_pass"])
        for operation, block in raw["per_operation"].items():
            for baseline, comparison in block["comparators"].items():
                # Every cell below the target, or missing it in a paired round,
                # is retained in misses. Cells omitted from misses all meet 2x.
                below_or_equal_one = sum(
                    item["ratio_of_medians"] is None or item["ratio_of_medians"] <= 1
                    for item in comparison["misses"])
                result["operations"].append({
                    "operation": operation, "baseline": baseline,
                    "cells": comparison["median_cell_count"],
                    "raw_wins": comparison["median_cell_count"] - below_or_equal_one,
                    "raw_median_ratio": comparison["median_across_cell_ratios"],
                    "raw_min_ratio": comparison["ratio_of_medians_range"]["min"],
                    "raw_max_ratio": comparison["ratio_of_medians_range"]["max"],
                    "two_x_cells": comparison["median_cells_meeting_target"],
                    "strict_comparison_pass": raw["accuracy_and_integrity_pass"]
                    and raw["quality_by_variant"]["current"]["strict_quality_pass"]
                    and raw["quality_by_variant"][baseline]["strict_quality_pass"],
                    "current_ms_range": block["median_ms_range_by_variant"]["current"],
                    "reference_ms_range": block["median_ms_range_by_variant"][baseline],
                })
    else:
        result.update(quality=raw["audit_counts"], routing=raw.get("routing", raw.get("query_backend_counts")),
                      acceptance=raw["acceptance_pass"])
        for baseline, metrics in raw["comparisons"].items():
            for operation, comparison in metrics.items():
                cells = comparison["all_cells"]
                ratios = [c["ratio_of_medians"] for c in cells if finite(c["ratio_of_medians"])]
                strict = True
                for variant in ("current", baseline):
                    counts = raw["audit_counts"][variant]
                    if "requests" in counts:
                        strict &= counts.get("strict_passes", 0) == counts["requests"]
                    else:
                        strict &= counts.get("first_strict_passes", 0) == counts.get("models_requested")
                        strict &= counts.get("second_strict_passes", 0) == counts.get("models_requested")
                current = [c["current_ms"] for c in cells if finite(c["current_ms"])]
                reference = [c["comparator_ms"] for c in cells if finite(c["comparator_ms"])]
                result["operations"].append({
                    "operation": operation, "baseline": baseline, "cells": len(cells),
                    "raw_wins": sum(r > 1 for r in ratios),
                    "raw_median_ratio": statistics.median(ratios) if ratios else None,
                    "raw_min_ratio": min(ratios) if ratios else None, "raw_max_ratio": max(ratios) if ratios else None,
                    "strict_comparison_pass": bool(strict and raw["acceptance_pass"]), "all_cells": cells,
                    "current_ms_range": {"min": min(current) if current else None, "max": max(current) if current else None},
                    "reference_ms_range": {"min": min(reference) if reference else None, "max": max(reference) if reference else None},
                })
    return result


def validate_panel(name, raw):
    """Coverage is independent of quality: completed failed checks are valid evidence."""
    require(raw.get("schema_version") == (2 if name.startswith("lr-") else 1), f"{name}: unsupported schema")
    if name.startswith("lr-"):
        require(raw.get("quality_certificate") == "float64-original-data-v2",
                f"{name}: missing corrected float64 quality certificate")
    if name.startswith("knn-"):
        require(raw["case_count"] == (72 if "broad" in name else 18), f"{name}: incomplete fixture inventory")
        require(raw["repeats"] == (15 if "broad" in name else 5 if "cold" in name else 30), f"{name}: repeat count changed")
        if "broad" in name:
            require(raw.get("status") == "complete" and raw.get("current_mode") == "auto", f"{name}: not completed auto panel")
            require(len(raw["per_operation"]) == 10, f"{name}: operation inventory changed")
        else:
            require(raw.get("evidence_validation_pass"), f"{name}: evidence validation failed")
    elif name.startswith("svc-"):
        require({c["name"] for c in raw["cases"]} == SVC_CASES and len(raw["cases"]) == 8,
                f"{name}: expected all eight kernel/cache cases")
        require(raw["repeats"] == 7, f"{name}: expected seven repeats")
        require(raw["seed"] == int(name.removeprefix("svc-seed")), f"{name}: wrong seed")
        for case in raw["cases"]:
            for variant in ("current", "previous", "sklearn"):
                require(len(case["quality_samples_including_first"][variant]) == 8,
                        f"{name}/{case['name']}/{variant}: missing audits")
                for operation in ("fit", "first_predict", "repeat_predict"):
                    stats = case[operation][variant]
                    samples = stats["samples_s"]
                    require(len(samples) == 7 and all(finite(s) and s > 0 for s in samples),
                            f"{name}/{case['name']}/{variant}/{operation}: incomplete timing")
                    require(math.isclose(stats["median_ms"], statistics.median(samples) * 1000, rel_tol=1e-12),
                            f"{name}/{case['name']}/{variant}/{operation}: invalid median")
    elif name.startswith("lr-"):
        parameters = raw["parameters"]
        expected = {(c["name"], dtype, seed) for c in parameters["cases"]
                    for dtype in parameters["dtypes"] for seed in parameters["seeds"]}
        actual = [(r["case"]["name"], r["dtype"], r["seed"]) for r in raw["results"]]
        require(raw.get("complete") and len(actual) == len(expected) and set(actual) == expected,
                f"{name}: incomplete/duplicate LR rows")
        require(set(parameters["dtypes"]) == {"float32", "float64"}, f"{name}: expected both dtypes")
        require(len(parameters["cases"]) == (3 if "accuracy-matched" in name else 1 if "strict4096" in name else 25),
                f"{name}: wrong fixture count")
        require(parameters["seeds"] == [41], f"{name}: expected seed 41")
        require(parameters["repeats"] == 7 and parameters["warmup"] == 1, f"{name}: repeat/warmup count changed")
        require(set(parameters["queries"]) == {1, 256, 4096}, f"{name}: query counts changed")
        require(parameters["requested_thread_limit"] == (1 if name.endswith("requested1") else None), f"{name}: wrong thread policy")
        for row in raw["results"]:
            if "error" in row:
                continue  # A failed attempted case is visible as missing timings, never dropped.
            validate_lr_block(row["fit"], parameters["repeats"], f"{name}/{row['case']['name']}/fit", fit=True)
            require(set(row["inference"]) == {str(q) for q in parameters["queries"]},
                    f"{name}/{row['case']['name']}: missing inference query counts")
            for query, operations in row["inference"].items():
                require(set(operations) == {"predict", "predict_proba"}, f"{name}/{query}: missing inference operation")
                for operation, block in operations.items():
                    validate_lr_block(block, parameters["repeats"], f"{name}/{row['case']['name']}/{query}/{operation}", fit=False)
        if "accuracy-matched" in name:
            require(parameters["tol"] == 1e-6 and {c["name"] for c in parameters["cases"]} == {
                "grid_1000_1024", "multiclass_weighted_wide", "wide_20k_4096"},
                f"{name}: accuracy-matched supplement identity changed")
        elif "strict4096" in name:
            require(parameters["tol"] == 1e-5 and parameters["cases"][0]["n_features"] == 4096,
                    f"{name}: strict supplement identity changed")
        else:
            require(parameters["tol"] == 1e-4, f"{name}: tolerance changed")
    else:
        expected = {c["id"] for c in raw["identity"]["cases"]}
        require(expected == grid_ids() and len(raw["identity"]["cases"]) == 84
                and raw.get("complete") and set(raw["results"]) == expected,
                f"{name}: expected completed 84-cell grid")
        require(raw["identity"]["thread_policy"] == "default" and raw["identity"]["seed"] == 41,
                f"{name}: wrong grid policy/seed")
        require(raw["identity"]["repeats"] == 7 and raw["identity"]["warmup"] == 1,
                f"{name}: wrong grid repetition/warmup count")
        for case in raw["identity"]["cases"]:
            result = raw["results"][case["id"]]
            require(result["case"] == case and result["source_identity"] == raw["identity"]["sources"],
                    f"{name}/{case['id']}: result case/source mismatch")
            variants = {"direct_auto", "patched_auto", "sklearn_brute", "sklearn_auto"} if case["algorithm"] == "knn" else {
                "direct_auto", "patched_auto", "sklearn"}
            require(set(result["timings"]) == variants, f"{name}/{case['id']}: missing variants")
            require(len(result["warmup_trials"]) == 1 and len(result["measured_trials"]) == 7,
                    f"{name}/{case['id']}: missing trial audits")
            for trial in result["warmup_trials"] + result["measured_trials"]:
                require(set(trial) == variants, f"{name}/{case['id']}: incomplete trial variants")
            for variant, timing in result["timings"].items():
                samples = timing["samples_s"]
                require(len(samples) == 7 and all(finite(s) and s > 0 for s in samples),
                        f"{name}/{case['id']}/{variant}: missing/invalid samples")
                require(math.isclose(timing["median_s"], statistics.median(samples), rel_tol=1e-12),
                        f"{name}/{case['id']}/{variant}: median differs from samples")


def grid_ids():
    """The frozen published shape inventory, without importing numerical modules."""
    names = {f"knn/fit/{n}x{d}" for n in (500, 2000, 8000, 32000, 128000, 512000) for d in (8, 64, 768)}
    names.update(f"knn/kneighbors/{n}x{d}/q{q}" for n in (500, 2000, 8000, 32000, 128000, 512000)
                 for d in (8, 64, 768) for q in (1, 256, 4096))
    names.update(f"svc/fit/{n}x{d}" for n in (500, 2000, 8000, 32000) for d in (16, 128))
    names.update(f"svc/predict/{n}x{d}" for n in (2000, 8000) for d in (16, 128))
    return names


def validate_lr_block(block, repeats, context, *, fit):
    variants = {*VARIANTS, "sklearn"}
    require(set(block["timings"]) == variants and set(block["speedups"]) == variants,
            f"{context}: missing variants")
    details = block["fit_details" if fit else "call_details"]
    audits = block["quality_each_fit" if fit else "quality_each_call"]
    require(set(details) == variants and set(audits) == variants, f"{context}: missing audits/details")
    for variant in variants:
        timing = block["timings"][variant]
        require(len(timing["samples_s"]) == len(details[variant]) == len(audits[variant]) == repeats,
                f"{context}/{variant}: sample/audit count mismatch")
        valid = [s for s in timing["samples_s"] if finite(s)]
        if valid:
            require(timing["summary"] is not None and math.isclose(
                timing["summary"]["median_ms"], statistics.median(valid) * 1000, rel_tol=1e-12),
                f"{context}/{variant}: median differs from raw samples")
        else:
            require(timing["summary"] is None, f"{context}/{variant}: summary without observations")


def build_summary(directory, *, allow_missing=False):
    panels, inputs, incomplete, measurements = {}, {}, {}, []
    required = set(PANELS)
    archived_manifest = directory / "initial-lr-gradient-audit" / "manifest.json"
    if archived_manifest.is_file():
        archived_files = json.loads(archived_manifest.read_text()).get("files", {})
        required.update(name for name in OPTIONAL_PANELS if name + ".json" in archived_files)
    for name in (*PANELS, *OPTIONAL_PANELS):
        path = directory / (name + ".json")
        if not path.is_file():
            if name in required:
                incomplete[name] = "missing file"
            continue
        data = path.read_bytes()
        raw = json.loads(data)
        inputs[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        try:
            validate_panel(name, raw)
        except (KeyError, TypeError, ValueError) as exc:
            incomplete[name] = str(exc)
            if not allow_missing:
                continue
        if name.startswith("knn-"):
            parsed = parse_knn(name, raw)
        elif name.startswith("lr-"):
            parsed = parse_logistic(name, raw)
        elif name.startswith("svc-"):
            parsed = parse_svc(name, raw)
        else:
            parsed = parse_grid(name, raw)
        panels[name] = parsed
        measurements.extend(parsed.get("measurements", []))
    require(allow_missing or not incomplete, "Final report requires every completed panel: " + json.dumps(incomplete, sort_keys=True))
    environment_path = directory / "environment.json"
    source_path = directory / "source-state.json"
    return clean({
        "schema_version": 1, "complete": not incomplete, "development_partial": bool(incomplete),
        "missing_or_incomplete_panels": incomplete, "input_files": inputs, "panels": panels,
        "operation_summaries": summarize_measurements(measurements),
        "failure_groups": grouped_failures(measurements),
        "environment": json.loads(environment_path.read_text()) if environment_path.is_file() else {},
        "source_state": json.loads(source_path.read_text()) if source_path.is_file() else {},
        "superseded_lr_audit": superseded_lr_provenance(directory, panels),
        "ratio_definition": "sklearn median milliseconds / current median milliseconds",
        "aggregation_policy": "Normalized LR/SVC/grid summaries describe case ratios within one panel, operation, dtype, policy, query count and variant. KNN broad and latency summaries span their heterogeneous fixture inventories, including different shapes/query counts. No average combines operations or policies. Unqualified timings remain visible.",
        "quality_audit_note": "Official LR inputs require schema_version=2 and quality_certificate=float64-original-data-v2. Original X, coefficients, intercepts and weights are explicitly promoted before float64 logits, residuals, objective and gradient calculations. The six earlier panels are archived as superseded audit evidence and never blended into official speed statistics.",
    })


def superseded_lr_provenance(directory, panels):
    """Verify archived evidence and the unchanged data/timing configuration."""
    archive = directory / "initial-lr-gradient-audit"
    path = archive / "manifest.json"
    if not path.is_file():
        return None
    data = path.read_bytes()
    manifest = json.loads(data)
    require(isinstance(manifest.get("files"), dict), "Superseded LR archive has no file inventory")
    comparisons = {}
    for name, expected in manifest["files"].items():
        require(Path(name).name == name, "Unsafe superseded archive member")
        archived = archive / name
        payload = archived.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected, f"Superseded LR evidence changed: {name}")
        panel_name = name.removesuffix(".json")
        if panel_name not in panels or not panel_name.startswith("lr-"):
            continue
        original = json.loads(payload)
        current = panels[panel_name]
        if not current.get("complete", False):
            continue
        require(original.get("schema_version") == 1, f"Superseded LR panel is not original schema 1: {panel_name}")
        keys = ("cases", "seeds", "dtypes", "queries", "repeats", "warmup", "tol", "max_iter", "requested_thread_limit")
        require(all(original["parameters"][key] == current["parameters"][key] for key in keys),
                f"Corrected LR timing/data parameters differ from archived panel: {panel_name}")
        require(original["source_integrity_before"]["benchmarks/logistic_refresh.py"]
                == manifest["original_lr_driver_sha256"], f"Archived LR driver differs: {panel_name}")
        original_source = {key: value for key, value in original["source_integrity_before"].items() if key.startswith("src/")}
        require(original_source == current["production_and_native_hashes"],
                f"Production/native hashes changed between original and corrected LR panel: {panel_name}")
        old_rows = {(row["case"]["name"], row["dtype"], row["seed"]): row.get("input_sha256")
                    for row in original["results"]}
        new_rows = {(row["case"]["name"], row["dtype"], row["seed"]): row.get("input_sha256")
                    for row in current["fit_rows"]}
        require(old_rows == new_rows, f"Corrected LR fixtures differ from archived panel: {panel_name}")
        comparisons[panel_name] = {"same_timing_parameters": True, "same_fixture_hashes": True,
                                  "same_production_and_native_hashes": True,
                                  "original_schema_version": original["schema_version"],
                                  "official_schema_version": current["audit_schema_version"]}
    if "original_harness_archive" in manifest:
        snapshot = Path(__file__).resolve().parents[1] / manifest["original_harness_archive"]
        require(hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest["original_harness_sha256"],
                "Original LR harness archive hash changed")
    return {"manifest_path": str(path.resolve()), "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "manifest": manifest, "verified_all_archived_file_hashes": True,
            "official_vs_superseded": comparisons, "included_in_speed_statistics": False}


def grouped_failures(measurements):
    groups = {}
    for entry in measurements:
        if entry["qualified"]:
            continue
        key = (entry["panel"], entry["case"], entry["dtype"], entry["seed"], entry["variant"],
               entry["baseline"], tuple(entry["failure_causes"]))
        group = groups.setdefault(key, {k: entry[k] for k in (
            "panel", "case", "dtype", "seed", "variant", "baseline", "failure_causes")})
        group.setdefault("operations", set()).add(entry["operation"])
        if entry["queries"] is not None:
            group.setdefault("query_counts", set()).add(entry["queries"])
    return [{**group, "operations": sorted(group["operations"]), "query_counts": sorted(group.get("query_counts", []))}
            for group in groups.values()]


def sample_cell(entry):
    suffix = "" if entry["qualified"] else " **U**"
    return number(entry["current_ms"]) + " / " + ratio_text(entry["raw_speedup"]) + suffix


def summary_rows(entries):
    return [[entry["panel"], entry["operation"], entry["dtype"] or "—",
             entry["queries"] if entry["queries"] is not None else "—", NAMES.get(entry["variant"], entry["variant"]),
             NAMES.get(entry["baseline"], entry["baseline"]),
             ratio_text(entry["raw_median_ratio"]), interval(entry["raw_min_ratio"], entry["raw_max_ratio"], ratios=True),
             f"{entry['raw_wins']}/{entry['cells']}", f"{entry['qualified_cells']}/{entry['cells']}",
             f"{entry['qualified_wins']}/{entry['qualified_cells']}", ratio_text(entry["qualified_median_ratio"])]
            for entry in entries]


SUMMARY_HEADERS = ["Panel", "Operation", "Input", "Queries", "Variant", "Comparator",
                   "Raw median ratio", "Raw ratio range", "Raw wins/cells", "Qualified/cells",
                   "Qualified wins/qualified", "Qualified median ratio"]


def render(summary, output, directory, summary_path):
    panels = summary["panels"]
    lines = ["# Current mlxlearn speedups versus scikit-learn",
             "Complete current performance panels on the recorded machine. **Speedup = sklearn median time / "
             "mlxlearn median time**; above 1× is faster. Fit, prediction, probability estimation, first query "
             "and process startup stay separate. Every median uses the retained observations in its source panel.",
             "**U** marks an unqualified comparison: its raw timing remains visible, but failed accuracy, "
             "convergence, reference or integrity checks prevent a speed claim. A patched estimator that used "
             "sklearn is a fallback comparison, never a GPU win. CPU/MLX labels below describe observed fit "
             "routing; KNN additionally reports actual query routing. LR host-array inference must not be "
             "described as GPU execution merely because fitting used the GPU.",
             "The tables' medians across case ratios are **descriptive**, not significance tests. Raw "
             "summaries include failures; qualified summaries show their explicit denominators. No average "
             "combines different operations or thread policies. LR and full-grid summaries additionally "
             "separate input dtypes and query counts. KNN broad/latency aggregates describe heterogeneous "
             "fixture inventories, with different training shapes, query counts and, in the broad panel, k values."]
    environment = summary.get("environment", {})
    if environment:
        lines.insert(2, f"Machine: **{environment.get('cpu', 'recorded in evidence')}**, "
                     f"{number(environment.get('memory_bytes', 0) / 2**30)} GiB; "
                     f"{environment.get('platform', '')}. Python {environment.get('python', '').split()[0]}; "
                     f"MLX {environment.get('mlx')}, NumPy {environment.get('numpy')}, "
                     f"SciPy {environment.get('scipy')}, scikit-learn {environment.get('sklearn')}. "
                     f"Source commit `{summary.get('source_state', {}).get('commit', 'see evidence')}`. "
                     "Benchmark workers ran serially.")
    if not summary["complete"]:
        lines += ["**DEVELOPMENT / INCOMPLETE — not a final benchmark report.**",
                  table(["Missing or incomplete panel", "Reason"], summary["missing_or_incomplete_panels"].items())]
    lines += ["## Operation overview", table(SUMMARY_HEADERS, summary_rows([
        e for e in summary["operation_summaries"] if e["operation"] == "fit"
        or e["panel"].startswith("svc-") or e["panel"].startswith("grid-")]))]
    lines += render_knn(panels)
    lines += ["## Logistic regression",
              "The default and requested-one-thread panels each retain 25 fixtures × two public input dtypes. "
              "Direct automatic, patched automatic and forced GPU are separate comparisons. The strict "
              "4,096-feature supplements use their own tolerance and never replace a failed default-tolerance row. "
              "Optional accuracy-matched supplements use tol=1e-6 on the three designated difficult fixtures. "
              "A failure of agreement with sklearn does not establish which optimizer is more accurate: "
              "independent objective and gradient diagnostics are retained in the summary. Different stopping "
              "points can produce different probabilities even when one has a lower objective. "
              "Thread environment and loaded pool information are retained in each input JSON. The requested "
              "one-thread panel controls Accelerate only when VECLIB_MAXIMUM_THREADS was set before process startup.",
              "The objective gate is **one-sided**: `(current objective − sklearn objective) / abs(sklearn objective) "
              "≤ 1e-5`. A lower current objective passes; objective equality is not required. Fits also require "
              "the independently evaluated maximum gradient to meet the requested tolerance, convergence, "
              "at least 99.5% training prediction agreement, and a passing sklearn reference. Probability "
              "comparisons separately require absolute tolerance 0.002 plus relative tolerance 1e-7, valid "
              "normalization and at least 99.5% class agreement.",
              "**Corrected float64 certificate:** every official LR panel uses schema version 2 and "
              "`float64-original-data-v2`. Training inputs, coefficients, intercepts and weights are "
              "explicitly promoted before computing logits, residuals, objectives and gradients. The six "
              "panels were rerun with unchanged timing parameters, seeds and thread policies. Their earlier "
              "runs are preserved as superseded audit evidence and are not combined with the official results.",
              "Fit cells show **current milliseconds / sklearn-current ratio**. Backend counts cover measured fits; "
              "the stock milliseconds are shared by all three comparisons in that row."]
    for name, panel in panels.items():
        if not name.startswith("lr-"):
            continue
        parameters = panel["parameters"]
        lines += ["### " + name,
                  f"tol={parameters['tol']:g}; max_iter={parameters['max_iter']}; "
                  f"pre-import thread environment: `{json.dumps(panel['thread_environment'], sort_keys=True)}`.",
                  table(["Case", "Shape", "Input", "Seed", "sklearn ms", "Direct auto ms / ratio",
                         "Patched auto ms / ratio", "Forced GPU ms / ratio", "Observed fit backends"], [
                      [row["case"]["name"], f"{row['case']['n_samples']}×{row['case']['n_features']}", row["dtype"], row["seed"],
                       number(row["sklearn_ms"]) + (" **U**" if row["sklearn_failure_causes"] else ""),
                       *(sample_cell(row["variants"][variant]) for variant in VARIANTS),
                       "; ".join(NAMES[variant] + "=" + backend_text(row["variants"][variant]["fit_backend_counts"])
                                 for variant in VARIANTS)] for row in panel["fit_rows"]])]
    lines += ["### LR inference by query count",
              "Each fit fixture contributes one separately qualified cell per operation and query count. "
              "Inference uses the final fitted model; fit time is excluded. Fit/reference failures invalidate "
              "the corresponding inference comparison even if the output check alone passes.",
              table(SUMMARY_HEADERS, summary_rows([entry for entry in summary["operation_summaries"]
                    if entry["panel"].startswith("lr-") and entry["operation"] != "fit"]))]
    lines += render_lr_diagnostics(panels)
    lines += render_svc(panels)
    lines += render_grid(panels)
    lines += ["## Unqualified comparisons and causes",
              "All failed cells remain in the tables and machine-readable summary. Repeated identical causes "
              "are grouped below; query counts and operations identify every affected comparison."]
    failures = summary["failure_groups"]
    lines += [table(["Panel", "Case", "Input / seed", "Variant / comparator", "Operations / queries", "Causes"], [
        [row["panel"], row["case"], f"{row['dtype']} / {row['seed']}",
         NAMES.get(row["variant"], row["variant"]) + " / " + NAMES.get(row["baseline"], row["baseline"]),
         ", ".join(row["operations"]) + " / " + ", ".join(map(str, row["query_counts"])),
         "; ".join(row["failure_causes"])] for row in failures]) if failures else "No unqualified LR/SVC/grid comparisons."]
    lines += ["## Evidence and reproducibility",
              "Large result files are distributed in a verified release asset. "
              "[Restore the evidence](../benchmarks/evidence/sklearn-refresh-evidence-2026-10-31.md) "
              "before following links to those files or regenerating this report.",
              linked(summary_path, output, "Machine-readable summary, all normalized comparisons and failure causes"),
              table(["Input panel", "SHA-256"], [
                  [linked(info["path"], output, name), "`" + info["sha256"] + "`"]
                  for name, info in summary["input_files"].items()])]
    superseded = summary.get("superseded_lr_audit")
    if superseded:
        lines += [linked(superseded["manifest_path"], output, "Superseded original LR panels and archive manifest")
                  + ". Every archived file hash was verified, along with matching original/current "
                  "fixture hashes, production/native hashes and timing/data parameters. These original panels are provenance only; "
                  "none of their timings enter the tables above."]
    for name in ("environment.json", "run-plan.json", "source-state.json", "queue-state.json",
                 "supplemental-queue-state.json", "corrected-lr-queue-state.json", "final-integrity.json"):
        if (directory / name).exists():
            lines.append(linked(directory / name, output, name))
    archive_root = Path(__file__).resolve().parents[1] / "benchmarks" / "reference"
    for suffix, label in ((".zip", "Final benchmark and report helper archive"),
                          (".manifest.json", "Final benchmark and report helper archive manifest")):
        path = archive_root / ("sklearn-refresh-harnesses-final-2026-10-31" + suffix)
        if path.is_file():
            lines.append(linked(path, output, label))
    lines += ["The linked panels retain source and native integrity checks for each timing run. Source hashes, "
              "native hashes, package versions, thread settings, warmups and actual backend records remain "
              "in the linked raw evidence. Historical SVC implementation controls are retained there but "
              "are not the sklearn comparison reported here. Full-grid KNN fit rows are deduplicated by "
              "training shape: 18 measured fits represent the original 54 repeated query-count fit labels."]
    return "\n\n".join(lines) + "\n"


def render_knn(panels):
    lines = ["## KNN: broad workload, first query and startup",
             "The broad panel uses 72 float32 fixtures, including raw Iris, Wine, Breast Cancer and Digits "
             "and synthetic shape/distribution cases. It retains ten operations and three sklearn controls. "
             "The **threadpoolctl-limited** control does not manage Apple Accelerate; it must not be labeled "
             "a verified globally single-threaded process. Default brute and auto retain their default pools.",
             "KNN ratios below are raw descriptive comparisons, retaining every comparator audit failure. "
             "Strict comparator failures include exact-distance-valid tie alternatives and distance/prediction "
             "mismatches; the audit table separates their recorded causes. A passing current implementation "
             "does not turn a failed comparator check into a qualified aggregate."]
    for name, panel in panels.items():
        if not name.startswith("knn-"):
            continue
        lines += ["### " + name.removesuffix("-summary"),
                  f"{panel['case_count']} fixtures; {panel['repeats']} retained repetitions; "
                  f"recorded current accuracy/integrity acceptance: **{panel['acceptance']}**.",
                  table(["Operation / phase", "Comparator", "Current ms range", "sklearn ms range", "Raw median ratio", "Raw ratio range", "Raw wins/cells", "Aggregate qualification"], [
                      [entry["operation"], NAMES.get(entry["baseline"], entry["baseline"]),
                       interval(entry["current_ms_range"]["min"], entry["current_ms_range"]["max"]),
                       interval(entry["reference_ms_range"]["min"], entry["reference_ms_range"]["max"]),
                       ratio_text(entry["raw_median_ratio"]), interval(entry["raw_min_ratio"], entry["raw_max_ratio"], ratios=True),
                       f"{entry['raw_wins']}/{entry['cells']}",
                       "strict pass" if entry["strict_comparison_pass"] else "**U** — strict audit failures retained"]
                      for entry in panel["operations"]])]
        if "broad" in name:
            lines.append(table(["Variant", "Strict passed / requests", "Recorded failure causes"], [
                [NAMES.get(variant, variant), f"{audit['all_phases']['strict_passes']}/{audit['all_phases']['audit_count']}",
                 json.dumps(audit["all_phases"]["failure_kind_counts"], sort_keys=True)]
                for variant, audit in panel["quality"].items()]))
            routing = panel["routing"]["by_phase"]["retained"]
            lines.append("Actual retained query backends: `" + json.dumps(routing["query_backend_counts"], sort_keys=True)
                         + "`; fitted-storage backends: `" + json.dumps(routing["fit_backend_counts"], sort_keys=True) + "`.")
        else:
            lines.append(table(["Variant", "Recorded query audits", "Actual query routing"], [
                [NAMES.get(variant, variant), json.dumps(audit, sort_keys=True),
                 json.dumps((panel["routing"] or {}).get(variant, {}), sort_keys=True)]
                for variant, audit in panel["quality"].items()]))
    lines += ["Fresh-model first/second-query measurements use initialized workers; they exclude interpreter "
              "and import startup. Fresh-process measurements create a new interpreter for every observation. "
              "Process-start-to-result includes imports, fixture loading and IPC; process elapsed additionally "
              "includes post-result audits. Fresh processes do not imply cold OS, filesystem or Metal disk caches."]
    return lines


def render_lr_diagnostics(panels):
    selected = {"wide_20k_4096", "grid_1000_1024", "multiclass_weighted_wide"}
    rows = []
    for name, panel in panels.items():
        if not name.startswith("lr-"):
            continue
        proba = {(entry["case"], entry["dtype"], entry["seed"]): entry
                 for entry in panel["measurements"] if entry["operation"] == "predict_proba"
                 and entry["queries"] == 4096 and entry["variant"] == "direct_auto"}
        for row in panel["fit_rows"]:
            if row["case"]["name"] not in selected:
                continue
            fitted = row["variants"]["direct_auto"]
            output = proba.get((row["case"]["name"], row["dtype"], row["seed"]), {})
            scores, stock = fitted["quality_scores"], row["sklearn_scores"]
            rows.append([name, row["case"]["name"], row["dtype"],
                         number(stock["gradient_max_abs_float64"]["max"]),
                         number(scores["gradient_max_abs_float64"]["max"]),
                         number(scores["objective_relative_gap_to_sklearn"]["max"]),
                         number(output.get("probability_difference_range", {}).get("max")),
                         "qualified" if fitted["qualified"] else "**U**",
                         "qualified" if output.get("qualified") else "**U**"])
    return ["### Difficult-fixture accuracy diagnostics",
            "This table follows **direct automatic** only, with primary and tighter-tolerance measurements "
            "kept in separate rows. Values are maxima across retained fits/calls, not a selected faster "
            "observation. Negative objective gaps favor the current model under the one-sided objective "
            "gate. A certified float64 stock gradient above the panel tolerance disqualifies the comparison "
            "even when current checks pass. "
            "Probability deltas use all 4,096 held-out queries.",
            table(["Panel", "Case", "Input", "sklearn max gradient", "Current max gradient",
                   "Signed relative objective gap", "Maximum probability delta", "Fit", "4,096-query probabilities"], rows)]


def render_svc(panels):
    rows, first = [], []
    for name, panel in panels.items():
        if not name.startswith("svc-"):
            continue
        for row in panel["rows"]:
            rows.append([row["case"], row["seed"], "×".join(map(str, row["shape"])), row["cache_size_mb"],
                         backend_text(row["backends"]), *(
                             number(row["metrics"][operation]["reference_ms"]) + " / " + sample_cell(row["metrics"][operation])
                             for operation in ("fit", "first_predict", "repeat_predict"))])
            first.append([row["case"], row["seed"], *(
                number(row["first_call"]["sklearn"][operation + "_s"] * 1000) + " / "
                + number(row["first_call"]["current"][operation + "_s"] * 1000)
                for operation in ("fit", "first_predict", "repeat_predict"))])
    return ["## SVC kernel/cache panel",
            "Eight cases × seeds 41/42, seven retained fresh fits each. Matching CPU/GPU routes are forced "
            "per case, with pre-import CPU thread controls set to one. These are direct estimator comparisons. "
            "Each retained fit has separately timed first and repeated predictions on 257 held-out queries. "
            "Quality checks include convergence, prediction agreement and continuous OVO margin agreement. "
            "The original aggregate quality gate, including the historical native control, is retained in "
            "the evidence; the comparisons here are qualified from current and sklearn checks and provenance only.",
            "Each metric cell is **sklearn ms / current ms / ratio**; U marks an unqualified raw ratio.",
            table(["Case", "Seed", "Train shape", "Cache MB", "Observed fit backend", "Fit", "First prediction", "Repeat prediction"], rows),
            "### SVC first calls, separate from retained medians",
            "Each cell below is sklearn ms / current ms for the first model in its worker. Imports and input "
            "generation are excluded. These observations are not the dedicated fresh-process startup measure.",
            table(["Case", "Seed", "First fit ms", "First prediction ms", "Repeat prediction ms"], first)]


def render_grid(panels):
    lines = ["## Full KNN/SVC shape grid",
             "84 operation cases: 18 distinct KNN fits, 54 KNN query configurations, eight SVC fits and four "
             "SVC predictions. Public inputs are float64, with original and canonical-float32 neighbor "
             "geometry checked separately. **KNN correctness samples up to nine evenly spaced query rows**, "
             "rather than claiming an exhaustive audit of all 4,096 queries. SVC queries are independent "
             "held-out Gaussian rows. Each case runs in a new worker; seven retained calls follow one warmup.",
             "Fit and query routes are shown separately; fit rows' query route belongs to their untimed accuracy check. "
             "Cells show current ms / raw ratio. KNN columns compare "
             "against both brute and auto sklearn; SVC uses stock SVC. Raw timing remains visible when any "
             "comparison's gate fails, including its stock reference. Qualification is evaluated per variant/comparator "
             "pair across all retained audits; an unrelated variant's failure does not erase a passing pair."]
    panel = panels.get("grid-default-seed41")
    if panel is None:
        return lines
    rows = []
    for row in panel["rows"]:
        baselines = list(row["reference_ms"])
        routes = []
        cells = []
        for variant in ("direct_auto", "patched_auto"):
            entry = row["variants"][variant][baselines[0]]
            routes.append(NAMES[variant] + ": fit=" + backend_text(entry["fit_backend_counts"])
                          + "; query=" + backend_text(entry["query_backend_counts"]))
            cells.append("; ".join(baseline.removeprefix("sklearn_") + " " + sample_cell(row["variants"][variant][baseline])
                                   for baseline in baselines))
        rows.append([row["case"]["id"], "; ".join(name + "=" + number(ms) for name, ms in row["reference_ms"].items()),
                     *cells, "; ".join(routes)])
    lines.append(table(["Case / operation", "sklearn ms", "Direct auto ms / ratio", "Patched auto ms / ratio", "Observed fit/query routes"], rows))
    if panel["failed_attempts"]:
        lines += ["Failed worker attempts are retained even if a later resume succeeded:",
                  table(["Case", "Started", "Exit code"], [[a["case"], a["started_at"], a["returncode"]] for a in panel["failed_attempts"]])]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing the completed panel JSON files")
    parser.add_argument("--output", type=Path, required=True, help="Markdown destination")
    parser.add_argument("--summary", type=Path, required=True, help="Machine-readable summary destination")
    parser.add_argument("--allow-missing-panels", action="store_true", help="Development only: visibly mark an incomplete report")
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.input, allow_missing=args.allow_missing_panels)
        markdown = render(summary, args.output, args.input, args.summary)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"Cannot render refresh evidence: {exc}\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    args.output.write_text(markdown)
    print(json.dumps({"complete": summary["complete"], "panels": len(summary["panels"]),
                      "summary": str(args.summary), "report": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
