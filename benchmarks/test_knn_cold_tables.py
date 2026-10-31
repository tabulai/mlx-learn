"""Static report checks: cost relocation and failed cells remain visible."""

import json
import sysconfig
from pathlib import Path

import pytest

from benchmarks.knn_broad_worker import backend_metadata
from benchmarks.knn_cold import (
    METRICS,
    VARIANTS,
    comparisons,
    parse_launch,
    select_specs,
    summarize,
)
from benchmarks.knn_cold_tables import SECTIONS, build_report
from benchmarks.test_knn_followup_benchmarks import ready_messages


def fake_cold_report(monkeypatch):
    import benchmarks.knn_cold_summary as module

    spec = select_specs()[0]
    names = (*VARIANTS, "previous_auto")
    source = {"mlxlearn/_kernels/_neighbors_cpu_native.cpp": "cpp"}
    root = "/tmp/isolated-cold-table-test"
    binary = "mlxlearn/_kernels/_neighbors_cpu_native" + sysconfig.get_config_var("EXT_SUFFIX")
    runtime = {"source_files": source, "abi": {"EXT_SUFFIX": sysconfig.get_config_var("EXT_SUFFIX")},
               "native_extension": {"source": {"sha256": "cpp"}, "binary": {"sha256": "binary"}}}
    monkeypatch.setattr(module, "verify_runtime_archive", lambda _: runtime)
    variants = {}
    for index, name in enumerate(names):
        if name == "sklearn_1":
            sample = {"error": "deliberate synthetic worker failure", "audit": {"pass": False}}
        else:
            ready, audited = ready_messages(passed=name != "sklearn_default")
            scale = .5 if name == "sklearn_default" else 2 if name == "sklearn_auto" else 1
            ready["timings"] = {key: value * scale for key, value in ready["timings"].items()}
            ready["result_ready_ns"] = ready["entry_ns"] + int(ready["timings"]["worker_entry_to_result_ms"] * 1e6)
            audited.update(backend_metadata(None, current_mode="auto", is_sklearn=True, operation="cold")
                           if name.startswith("sklearn") else {})
            sample = parse_launch(json.dumps(ready), json.dumps(audited), returncode=0,
                                  elapsed_ms=50 * scale, result_ms=10 * scale)
            sample.update(pid=index + 100, fixture_sha256="fixture", worker={
                "source_root": root, "source_hashes": source,
                "native_cpu_extension": {"available": True,
                    "source": {"sha256": "cpp", "path": root + "/mlxlearn/_kernels/_neighbors_cpu_native.cpp"},
                    "binary": {"sha256": "binary", "path": root + "/" + binary}},
            })
        variants[name] = summarize([sample], 1)
    case = {"case": spec["case"], "spec": spec, "variants": variants,
            "rounds": [{"round": 0, "order": list(names)}], "comparisons": comparisons(variants)}
    report = {"status": "complete", "planned_cases": 1, "repeats": 1, "variants": list(names),
              "metrics": list(METRICS), "cases": [case],
              "quality_by_variant": {v: s["quality_pass"] for v, s in variants.items()},
              "timing_count_pass": False, "expected_launches": 5, "completed_launches": 5,
              "reference_runtime_integrity_pass": True, "acceptance_pass": False,
              "reference_runtime": runtime, "reference_runtime_archive": "synthetic.zip",
              "reference": {"files": source},
              "fixtures": [{"case": spec["case"], "fixture_sha256": "fixture"}], "protocol": "fabricated test"}
    for stem in ("source", "harness", "archive", "fixture", "native"):
        values = {"current": "binary", "previous_auto": "binary"} if stem == "native" else source
        report[f"{stem}_hashes_before"] = report[f"{stem}_hashes_after"] = values
        report[f"{stem}_integrity_pass"] = True
    return report


def test_renderer_retains_every_cost_failure_and_missing_cell(monkeypatch):
    raw = fake_cold_report(monkeypatch)
    text, summary = build_report(raw, raw_path=Path("raw.json"), output_path=Path("report.md"),
                                 expected_cases=1, expected_repeats=1)
    for _, heading, _ in SECTIONS:
        assert "## " + heading in text
    assert "Preceding automatic ms / ratio" in text
    assert "**0.500×**" in text and "**missing**" in text
    assert "5 separate process launches" in text
    assert "/cases/0/variants/sklearn_default/samples/0" in text
    assert "/cases/0/variants/sklearn_1/samples/0" in text
    assert "strict_audit_failure" in text and "worker_error" in text
    assert '"True": 1' in text
    assert "Moving native initialization into imports does not remove it" in text
    assert "Full process elapsed includes audits, metadata" in text
    assert summary["audit_counts"]["sklearn_default"]["completed_timings"] == 1
    assert summary["audit_counts"]["sklearn_default"]["strict_passes"] == 0
    assert not summary["acceptance_pass"]


def test_renderer_validates_raw_values_before_publishing(monkeypatch):
    raw = fake_cold_report(monkeypatch)
    raw["cases"][0]["variants"]["current"]["metrics"]["query_ms"]["median_ms"] = 999
    with pytest.raises(ValueError, match="summary disagrees"):
        build_report(raw, raw_path=Path("raw.json"), output_path=Path("report.md"),
                     expected_cases=1, expected_repeats=1)
