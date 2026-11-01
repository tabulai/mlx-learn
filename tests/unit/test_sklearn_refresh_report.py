"""Reporting protocol tests discovered by the regular unit-test suite."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from benchmarks import sklearn_refresh_report as report


def logistic_payload():
    variants = (*report.VARIANTS, "sklearn")
    fit = {
        "timings": {v: {"summary": {"median_ms": 10 if v == "sklearn" else 2}, "complete": True} for v in variants},
        "fit_details": {v: [{"backend": "sklearn" if v in {"sklearn", "patched_auto"} else "mlx",
                             "implementation": "fixture", "warnings": [], "error": None}] for v in variants},
        "quality_each_fit": {v: [{"checks": {"gradient": True, "objective": True}, "own_checks_pass": True,
                                  "reference_valid": True, "passed": True}] for v in variants},
        "speedups": {v: {"quality_qualified": True} for v in variants},
    }
    inference = {
        "timings": copy.deepcopy(fit["timings"]),
        "call_details": {v: [{"error": None}] for v in variants},
        "quality_each_call": {v: [{"passed": True}] for v in variants},
        "speedups": copy.deepcopy(fit["speedups"]),
    }
    row = {"case": {"name": "fixture", "n_samples": 10, "n_features": 3}, "dtype": "float32", "seed": 41,
           "fit": fit, "inference": {"256": {"predict": copy.deepcopy(inference), "predict_proba": inference}}}
    return {"results": [row], "source_integrity_pass": True, "parameters": {}, "complete": True}


def test_probability_failure_does_not_disqualify_fit_or_prediction():
    raw = logistic_payload()
    block = raw["results"][0]["inference"]["256"]["predict_proba"]
    block["speedups"]["forced_gpu"]["quality_qualified"] = False
    block["quality_each_call"]["forced_gpu"] = [{"passed": False, "probabilities_close": False}]
    parsed = report.parse_logistic("lr-default", raw)
    cells = {(r["variant"], r["operation"]): r for r in parsed["measurements"]}
    assert cells["forced_gpu", "fit"]["qualified"]
    assert cells["forced_gpu", "predict"]["qualified"]
    assert not cells["forced_gpu", "predict_proba"]["qualified"]
    assert cells["direct_auto", "predict_proba"]["qualified"]
    assert cells["forced_gpu", "predict_proba"]["raw_speedup"] == 5
    assert "probabilities close failed" in cells["forced_gpu", "predict_proba"]["failure_causes"]
    assert cells["patched_auto", "fit"]["fit_backend_counts"] == {"sklearn": 1}


def test_reference_failure_retains_time_and_blocks_claim_even_if_flag_is_stale():
    raw = logistic_payload()
    raw["results"][0]["fit"]["quality_each_fit"]["sklearn"][0]["checks"]["gradient"] = False
    parsed = report.parse_logistic("lr-default", raw)
    entry = parsed["fit_rows"][0]["variants"]["direct_auto"]
    assert not entry["qualified"]
    assert entry["raw_speedup"] == 5
    assert "sklearn reference: gradient exceeds tolerance" in entry["failure_causes"]


def test_summaries_never_pool_operations_or_dtypes():
    base = {"panel": "lr-default", "case": "one", "variant": "direct_auto", "baseline": "sklearn",
            "current_ms": 2, "reference_ms": 10, "qualified": True, "policy": "default"}
    rows = [report.measurement(**base, operation="fit", dtype="float32"),
            report.measurement(**base, operation="fit", dtype="float64"),
            report.measurement(**base, operation="predict", dtype="float32", queries=256)]
    summaries = report.summarize_measurements(rows)
    assert len(summaries) == 3
    assert all(row["cells"] == row["qualified_cells"] == 1 for row in summaries)


def test_final_report_rejects_missing_panels_but_marks_development(tmp_path):
    with pytest.raises(ValueError, match="every completed panel"):
        report.build_summary(tmp_path)
    partial = report.build_summary(tmp_path, allow_missing=True)
    assert not partial["complete"]
    assert set(partial["missing_or_incomplete_panels"]) == set(report.PANELS)
    text = report.render(partial, tmp_path / "report.md", tmp_path, tmp_path / "summary.json")
    assert "DEVELOPMENT / INCOMPLETE" in text
    json.dumps(partial, allow_nan=False)


def test_nonfinite_or_failed_timings_never_become_qualified():
    entry = report.measurement(panel="test", case="case", operation="fit", variant="direct_auto", baseline="sklearn",
                               current_ms=float("nan"), reference_ms=1, qualified=True)
    assert entry["raw_speedup"] is None and not entry["qualified"]
    json.dumps(report.clean(entry), allow_nan=False)


def test_failed_case_keeps_missing_inference_in_denominators():
    raw = logistic_payload()
    raw["parameters"]["queries"] = [1, 256, 4096]
    row = raw["results"][0]
    row.pop("fit")
    row.pop("inference")
    row["error"] = {"type": "MemoryError", "message": "recorded failure"}
    parsed = report.parse_logistic("lr-default", raw)
    assert len(parsed["measurements"]) == 3 * (1 + 3 * 2)
    assert not any(entry["qualified"] for entry in parsed["measurements"])
    assert all(entry["raw_speedup"] is None for entry in parsed["measurements"])


def test_knn_raw_win_count_retains_all_misses():
    comparison = {"median_cell_count": 4, "median_across_cell_ratios": 2.5,
                  "ratio_of_medians_range": {"min": .5, "max": 4}, "median_cells_meeting_target": 3,
                  "misses": [{"ratio_of_medians": .5}, {"ratio_of_medians": 3}]}
    raw = {"case_count": 4, "repeats": 15, "quality_by_variant": {
        "current": {"strict_quality_pass": True}, "sklearn_default": {"strict_quality_pass": False}},
        "routing": {}, "accuracy_and_integrity_pass": True,
        "per_operation": {"kneighbors": {"comparators": {"sklearn_default": comparison},
            "median_ms_range_by_variant": {"current": {"min": 1, "max": 2},
                                           "sklearn_default": {"min": .5, "max": 8}}}}}
    parsed = report.parse_knn("knn-broad-summary", raw)
    assert parsed["operations"][0]["raw_wins"] == 3
    assert not parsed["operations"][0]["strict_comparison_pass"]


def test_frozen_grid_inventory_has_exact_operation_counts():
    names = report.grid_ids()
    assert len(names) == 84
    assert sum(name.startswith("knn/fit/") for name in names) == 18
    assert sum(name.startswith("knn/kneighbors/") for name in names) == 54
    assert sum(name.startswith("svc/fit/") for name in names) == 8
    assert sum(name.startswith("svc/predict/") for name in names) == 4


def test_grid_qualifies_pairs_without_hiding_aggregate_failure():
    case = {"id": "svc/predict/2000x16", "algorithm": "svc", "operation": "predict",
            "n_samples": 2000, "n_features": 16, "n_queries": 0}
    variants = ("direct_auto", "patched_auto", "sklearn")
    trial = {v: {"audit": {"passed": v != "direct_auto", "reported_kkt_passed": v != "direct_auto"},
                  "model": {"backend": "cpu" if v == "direct_auto" else "sklearn", "warnings": []}}
             for v in variants}
    identity = {"cases": [case], "seed": 41, "repeats": 7, "warmup": 1,
                "thread_policy": "default", "sources": {"source": "hash"},
                "python": "test", "platform": "test", "packages": {
                    "numpy": "test", "scipy": "test", "scikit-learn": "test", "mlx": "test"}}
    result = {"case": case, "quality_passed": False, "qualified_speedup": None,
              "timings": {v: {"median_s": 1, "samples_s": [1] * 7} for v in variants},
              "warmup_trials": [copy.deepcopy(trial)], "measured_trials": [copy.deepcopy(trial) for _ in range(7)],
              "source_identity": identity["sources"], "input_dtype": "float64", "query_rows": 1000,
              "environment": {"python": "test", "platform": "test", "thread_policy": "default",
                              "numpy": "test", "scipy": "test", "sklearn": "test", "mlx": "test"}}
    raw = {"identity": identity, "results": {case["id"]: result}}
    parsed = report.parse_grid("grid-default-seed41", raw)
    pairs = {entry["variant"]: entry for entry in parsed["measurements"]}
    assert not pairs["direct_auto"]["qualified"]
    assert pairs["patched_auto"]["qualified"]
    assert parsed["rows"][0]["recorded_qualified_speedup"] is None
    result["source_identity"] = {"source": "changed"}
    assert not any(entry["qualified"] for entry in report.parse_grid("grid-default-seed41", raw)["measurements"])


@pytest.mark.parametrize("payload, message", [
    ({"schema_version": 1}, "unsupported schema"),
    ({"schema_version": 2}, "corrected float64 quality certificate"),
])
def test_final_lr_requires_corrected_certificate(payload, message):
    with pytest.raises(ValueError, match=message):
        report.validate_panel("lr-default", payload)


def test_archived_original_lr_evidence_is_verified_but_never_merged(tmp_path):
    archive = tmp_path / "initial-lr-gradient-audit"
    archive.mkdir()
    parameters = {key: [] for key in ("cases", "seeds", "dtypes", "queries")}
    parameters.update(repeats=7, warmup=1, tol=1e-4, max_iter=200, requested_thread_limit=None)
    original = {"schema_version": 1, "parameters": parameters, "results": [],
                "source_integrity_before": {"benchmarks/logistic_refresh.py": "old-driver", "src/module.py": "unchanged"}}
    saved = archive / "lr-default.json"
    saved.write_text(json.dumps(original))
    manifest = {"files": {saved.name: hashlib.sha256(saved.read_bytes()).hexdigest()},
                "original_lr_driver_sha256": "old-driver"}
    (archive / "manifest.json").write_text(json.dumps(manifest))
    panels = {"lr-default": {"parameters": parameters, "fit_rows": [], "audit_schema_version": 2, "complete": True,
                             "production_and_native_hashes": {"src/module.py": "unchanged"}}}
    provenance = report.superseded_lr_provenance(tmp_path, panels)
    assert provenance["verified_all_archived_file_hashes"]
    assert not provenance["included_in_speed_statistics"]
    assert provenance["official_vs_superseded"]["lr-default"]["same_production_and_native_hashes"]
    saved.write_text(saved.read_text() + " ")
    with pytest.raises(ValueError, match="Superseded LR evidence changed"):
        report.superseded_lr_provenance(tmp_path, panels)


def test_archived_optional_panel_requires_its_corrected_replacement(tmp_path):
    archive = tmp_path / "initial-lr-gradient-audit"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({
        "files": {"lr-accuracy-matched-default.json": "original-hash"}}))
    with pytest.raises(ValueError, match="lr-accuracy-matched-default"):
        report.build_summary(tmp_path)
