"""CPU-only protocol, five-worker schema and archived-runtime isolation tests."""

import copy
import json
import sysconfig
from types import SimpleNamespace

import pytest

from benchmarks import test_knn_freeze_runtime as runtime_helpers
from benchmarks.knn_auto_control import (
    EXTENDED_VARIANTS,
    EXTENDED_WORKER_MODES,
    _paired_ratios,
    family_summary,
    routing_summary,
)
from benchmarks.knn_auto_control_tables import validate as validate_control
from benchmarks.knn_broad_worker import backend_metadata
from benchmarks.knn_cold import (
    METRICS,
    VARIANTS,
    comparisons,
    parse_launch,
    select_specs,
    summarize,
)
from benchmarks.knn_cold_summary import summarize_report
from benchmarks.knn_pipeline import extract_reference, sha256, source_hashes
from benchmarks.knn_reference_runtime import (
    install_reference_runtime,
    reference_runtime_unchanged,
    validate_reference_worker,
)
from benchmarks.test_knn_auto_harness import fake_control

freeze = runtime_helpers.freeze
runtime_inputs = runtime_helpers.runtime_inputs


def extended_control():
    report = fake_control()
    report["variants"] = list(EXTENDED_VARIANTS)
    report["worker_modes"] = EXTENDED_WORKER_MODES
    root = "/tmp/isolated-fake-reference"
    cpp = "mlxlearn/_kernels/_neighbors_cpu_native.cpp"
    relative = "mlxlearn/_kernels/_neighbors_cpu_native" + sysconfig.get_config_var("EXT_SUFFIX")
    runtime = {"native_extension": {"source": {"sha256": "cpp"}, "binary": {"sha256": "binary"}},
               "source_files": {cpp: "cpp"}, "abi": {"EXT_SUFFIX": sysconfig.get_config_var("EXT_SUFFIX")}}
    worker = {"current_mode": "auto", "source_root": root, "source_hashes": {cpp: "cpp"},
              "native_cpu_extension": {"available": True, "source": {"sha256": "cpp", "path": root + "/" + cpp},
                                       "binary": {"sha256": "binary", "path": root + "/" + relative}}}
    report["workers"]["previous_auto"] = worker
    report["reference_runtimes"] = {"previous_auto": runtime}
    report["reference_runtime_integrity_pass"] = True
    report["quality_by_variant"]["previous_auto"] = True
    for case in report["cases"]:
        case["cold"]["previous_auto"] = copy.deepcopy(case["cold"]["current"])
        for experiment in case["experiments"].values():
            experiment["variants"]["previous_auto"] = copy.deepcopy(experiment["variants"]["current"])
            experiment["warmup"]["previous_auto"] = copy.deepcopy(experiment["warmup"]["current"])
            experiment["quality_by_variant"]["previous_auto"] = True
            for entry in experiment["rounds"]:
                entry["order"] = list(EXTENDED_VARIANTS)
            samples = {v: stats["samples"] for v, stats in experiment["variants"].items()}
            experiment["comparisons"]["current_vs_previous_auto"] = _paired_ratios(samples, "previous_auto", report["pairs"]) | {"ratio_of_medians": 1.0}
    report["families"] = family_summary(report["cases"], EXTENDED_VARIANTS)
    report["routing"] = routing_summary(report["cases"], EXTENDED_VARIANTS)
    return report


def test_extended_control_keeps_all_five_workers_and_audits():
    report = extended_control()
    validate_control(report, expected_cases=1)
    assert report["routing"]["previous_auto"]["retained"]["request_count"] == 18
    report["workers"]["previous_auto"]["native_cpu_extension"]["binary"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="wrong native binary"):
        validate_control(report, expected_cases=1)


def test_archived_runtime_is_restored_only_inside_reference(runtime_inputs, monkeypatch):
    root, _, source_archive, binary, _, _, source_manifest, _ = runtime_inputs
    runtime = freeze(runtime_inputs)
    import benchmarks.knn_reference_runtime as module
    monkeypatch.setattr(module, "current_abi", lambda: runtime["abi"])
    destination = root / "isolated"
    extract_reference(source_archive, destination)
    restored = install_reference_runtime(root / "runtime.zip", destination, source_manifest)
    relative = runtime["native_extension"]["binary"]["archive_member"].removeprefix("src/")
    assert (destination / relative).read_bytes() == binary.read_bytes()
    assert source_hashes(destination) == source_manifest["files"]
    assert reference_runtime_unchanged(destination, restored)
    worker = {"source_root": str(destination), "source_hashes": source_manifest["files"],
              "native_cpu_extension": {"available": True,
                 "binary": {"sha256": sha256(binary.read_bytes()), "path": str(destination / relative)},
                 "source": {"sha256": source_manifest["files"]["mlxlearn/_kernels/_neighbors_cpu_native.cpp"],
                            "path": str(destination / "mlxlearn/_kernels/_neighbors_cpu_native.cpp")}}}
    validate_reference_worker(worker, restored)
    with pytest.raises(FileExistsError):
        install_reference_runtime(root / "runtime.zip", destination, source_manifest)


def test_reference_runtime_rejects_wrong_source_and_abi(runtime_inputs, monkeypatch):
    root, _, source_archive, _, _, _, source_manifest, _ = runtime_inputs
    freeze(runtime_inputs)
    destination = root / "isolated"
    extract_reference(source_archive, destination)
    changed = copy.deepcopy(source_manifest)
    changed["archive"]["sha256"] = "other source archive"
    with pytest.raises(ValueError, match="same implementation"):
        install_reference_runtime(root / "runtime.zip", destination, changed)
    import benchmarks.knn_reference_runtime as module
    current = module.current_abi()
    monkeypatch.setattr(module, "current_abi", lambda: dict(current, SOABI="incompatible"))
    with pytest.raises(ValueError, match="ABI mismatch"):
        install_reference_runtime(root / "runtime.zip", destination, source_manifest)


def ready_messages(*, passed=True):
    timings = dict.fromkeys(METRICS[:-2], 1.0)
    timings["fit_query_ms"] = 2.0
    timings["estimator_to_result_ms"] = 3.0
    timings["worker_entry_to_result_ms"] = 5.0
    ready = {"kind": "result_ready", "timings": timings, "entry_ns": 100, "result_ready_ns": 5000100,
             "loaded_modules": {"before_input": {"numpy": False}, "after_input": {"numpy": True},
                                "before_fit": {"mlxlearn._kernels._neighbors_cpu_native": True}}}
    model = SimpleNamespace(execution_backend_="cpu", execution_policy_="auto", last_query_backend_="cpu")
    audited = {"kind": "audited_result", "audit": {"pass": passed},
               **backend_metadata(model, current_mode="auto", is_sklearn=False, operation="cold")}
    return ready, audited


def test_first_result_marker_excludes_audit_and_preserves_import_cost():
    ready, audited = ready_messages()
    result = parse_launch(json.dumps(ready), json.dumps(audited), returncode=0, elapsed_ms=50, result_ms=10)
    assert result["timings"]["process_start_to_result_ms"] == 10
    assert result["timings"]["process_elapsed_ms"] == 50
    assert result["timings"]["estimator_to_result_ms"] == 3
    assert result["loaded_modules"]["before_fit"]["mlxlearn._kernels._neighbors_cpu_native"]


def test_cold_accuracy_failures_retain_timing_denominators():
    ready, audited = ready_messages(passed=False)
    result = parse_launch(json.dumps(ready), json.dumps(audited), returncode=0, elapsed_ms=50, result_ms=10)
    stats = summarize([result], 1)
    assert not stats["quality_pass"] and stats["completed_samples"] == 1
    other = copy.deepcopy(stats)
    ratios = comparisons({"current": stats, "previous_auto": other})
    assert ratios["previous_auto"]["fit_query_ms"]["ratio_of_medians"] == 1


def test_cold_summary_checks_every_launch_and_rejects_process_reuse():
    spec = select_specs()[0]
    variants = {}
    for index, name in enumerate(VARIANTS):
        ready, audited = ready_messages()
        audited.update(backend_metadata(None, current_mode="auto", is_sklearn=True, operation="cold")
                       if name.startswith("sklearn") else {})
        sample = parse_launch(json.dumps(ready), json.dumps(audited), returncode=0, elapsed_ms=50, result_ms=10)
        sample.update(pid=index + 100, fixture_sha256="fixture", worker={
            "source_hashes": {"fake.py": "source"}, "native_cpu_extension": {"available": False},
        })
        variants[name] = summarize([sample], 1)
    case = {"case": spec["case"], "spec": spec, "variants": variants,
            "rounds": [{"round": 0, "order": list(VARIANTS)}], "comparisons": comparisons(variants)}
    report = {"status": "complete", "planned_cases": 1, "repeats": 1, "variants": list(VARIANTS),
              "metrics": list(METRICS), "cases": [case], "quality_by_variant": dict.fromkeys(VARIANTS, True),
              "timing_count_pass": True, "expected_launches": 4, "completed_launches": 4,
              "reference_runtime_integrity_pass": True, "acceptance_pass": True,
              "fixtures": [{"case": spec["case"], "fixture_sha256": "fixture"}], "protocol": "fabricated test"}
    for stem in ("source", "harness", "archive", "fixture", "native"):
        values = {"current": None} if stem == "native" else {"fake.py": "source"}
        report[f"{stem}_hashes_before"] = report[f"{stem}_hashes_after"] = values
        report[f"{stem}_integrity_pass"] = True
    summary = summarize_report(report, expected_cases=1, expected_repeats=1)
    assert all(c["requests"] == c["strict_passes"] == 1 for c in summary["audit_counts"].values())
    variants["sklearn_default"]["samples"][0]["pid"] = variants["current"]["samples"][0]["pid"]
    with pytest.raises(ValueError, match="process was reused"):
        summarize_report(report, expected_cases=1, expected_repeats=1)


@pytest.mark.parametrize("change", ["wrong_order", "extra_response", "missing_audit", "nonzero_exit"])
def test_cold_protocol_rejects_non_first_use_or_incomplete_response(change):
    ready, audited = map(json.dumps, ready_messages())
    first, remaining, code = ready, audited, 0
    if change == "wrong_order":
        first, remaining = remaining, first
    elif change == "extra_response":
        remaining += "\n" + audited
    elif change == "missing_audit":
        remaining = ""
    else:
        code = 1
    result = parse_launch(first, remaining, returncode=code, elapsed_ms=50, result_ms=10)
    assert not result["audit"]["pass"]
    assert "messages" in result
