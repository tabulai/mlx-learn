"""CPU-only evidence-contract tests; no estimator or benchmark runtime is loaded."""

import copy
from types import SimpleNamespace

import pytest

from benchmarks.knn_auto_control import (
    VARIANTS,
    WORKER_MODES,
    _paired_ratios,
    _summarize,
    family_summary,
    operations_for,
    routing_summary,
    select_specs,
)
from benchmarks.knn_auto_control_tables import validate
from benchmarks.knn_backend_audit import validate_record_backend
from benchmarks.knn_broad_worker import backend_metadata
from benchmarks.knn_freeze import freeze_source, verify_source
from benchmarks.knn_small_control import select_specs as previous_specs


def sample(mode, *, passed=True):
    model = SimpleNamespace(execution_backend_="cpu" if mode == "auto" else "mlx",
                            execution_policy_="auto" if mode == "auto" else "fixed",
                            last_query_backend_="cpu" if mode == "auto" else "mlx")
    return {"ms": 1.0, "audit": {"pass": passed}, **backend_metadata(
        model, current_mode=mode, is_sklearn=False, operation="kneighbors",
    )}


def fake_control(*, passed=True):
    """One actual case specification with fabricated timings, for schema testing."""
    spec = select_specs()[0]
    pairs, duration = 2, 0.25
    cold = {v: sample(WORKER_MODES[v], passed=passed) for v in VARIANTS}
    experiments = {}
    for operation in operations_for(spec):
        samples = {v: [copy.deepcopy(cold[v]) for _ in range(pairs)] for v in VARIANTS}
        variants = {v: _summarize(samples[v], pairs) for v in VARIANTS}
        warmup = {v: {
            "warmup_count": 1, "warmup_audits": [{"pass": passed}],
            "warmup_elapsed_seconds": duration, "warmup_min_seconds": duration,
            "warmup_quality_pass": passed, "quality_pass": passed,
            "samples": [copy.deepcopy(cold[v])],
            "warmup_backend_records": [copy.deepcopy(cold[v])],
        } for v in VARIANTS}
        comparisons = {f"current_vs_{v}": _paired_ratios(samples, v, pairs)
                       | {"ratio_of_medians": 1.0} for v in VARIANTS[1:]}
        experiments[operation] = {
            "variants": variants, "warmup": warmup, "comparisons": comparisons,
            "quality_by_variant": dict.fromkeys(VARIANTS, passed),
            "rounds": [{"pair": p, "order": list(VARIANTS)} for p in range(pairs)],
        }
    cases = [{"case": spec["case"], "spec": spec, "cold": cold, "experiments": experiments}]
    return {
        "status": "complete", "schema_version": 1, "variants": list(VARIANTS),
        "worker_modes": WORKER_MODES, "backend_telemetry_version": 1,
        "workers": {v: {"current_mode": WORKER_MODES[v]} for v in VARIANTS},
        "cases": cases, "planned_cases": 1, "fixtures": [{}],
        "operations_by_case": {spec["case"]["name"]: list(operations_for(spec))},
        "planned_case_operation_count": len(operations_for(spec)),
        "pairs": pairs, "warmup_seconds_per_variant_operation": duration,
        "quality_by_variant": dict.fromkeys(VARIANTS, passed), "quality_pass": passed,
        "families": family_summary(cases), "routing": routing_summary(cases),
        "complete_case_count_pass": True, "source_integrity_pass": True,
        "current_source_integrity_pass": True, "fixture_integrity_pass": True,
        "harness_integrity_pass": True, "archive_integrity_pass": True,
        "acceptance_pass": passed,
    }


def test_unchanged_control_inventory():
    specs = select_specs()
    assert specs == previous_specs()
    assert len(specs) == 48
    assert sum(len(operations_for(s)) for s in specs) == 222


@pytest.mark.parametrize("passed", [True, False])
def test_cpu_schema_retains_accuracy_failures(passed):
    validate(fake_control(passed=passed), expected_cases=1)


@pytest.mark.parametrize("mutation", ["missing_query", "fallback", "wrong_label", "wrong_policy"])
def test_rejects_misleading_auto_records(mutation):
    record = sample("auto")
    if mutation == "missing_query":
        record.pop("query_backend")
    elif mutation == "fallback":
        record.update(backend="sklearn", query_backend="sklearn")
    elif mutation == "wrong_label":
        record["backend"] = "mlx"
    else:
        record["execution_policy"] = "fixed"
    with pytest.raises(ValueError):
        validate_record_backend(record, "auto")


def test_auto_storage_and_query_can_differ():
    model = SimpleNamespace(execution_backend_="cpu", execution_policy_="auto",
                            last_query_backend_="mlx")
    record = backend_metadata(model, current_mode="auto", is_sklearn=False, operation="fit")
    assert record["fit_backend"] == "cpu"
    assert record["query_backend"] == "mlx"
    assert record["query_backend_phase"] == "untimed_fit_audit"
    validate_record_backend(record, "auto")


def test_legacy_gpu_telemetry_fallback_is_explicit_only():
    model = SimpleNamespace(execution_backend_="mlx")
    record = backend_metadata(model, current_mode="gpu", is_sklearn=False, operation="kneighbors")
    validate_record_backend(record, "gpu")
    with pytest.raises(RuntimeError):
        backend_metadata(model, current_mode="auto", is_sklearn=False, operation="kneighbors")


@pytest.mark.parametrize("mutation", ["missing_warmup", "mislabel_worker", "changed_ratio", "changed_routes"])
def test_validator_rejects_inconsistent_evidence(mutation):
    report = fake_control()
    experiment = next(iter(report["cases"][0]["experiments"].values()))
    if mutation == "missing_warmup":
        experiment["warmup"]["current"]["warmup_backend_records"] = []
    elif mutation == "mislabel_worker":
        report["workers"]["current"]["current_mode"] = "gpu"
    elif mutation == "changed_ratio":
        experiment["comparisons"]["current_vs_previous_gpu"]["ratio_of_medians"] = 5
    else:
        report["routing"]["current"]["retained"]["query_backend_counts"] = {"mlx": 18}
    with pytest.raises(ValueError):
        validate(report, expected_cases=1)


def test_source_freeze_is_reproducible_and_refuses_overwrite(tmp_path):
    source = tmp_path / "src"
    module = source / "mlxlearn" / "example.py"
    module.parent.mkdir(parents=True)
    module.write_text("value = 1\n")
    first, second = (tmp_path / name for name in ("first.zip", "second.zip"))
    for path in (first, second):
        freeze_source(source, path, source_state="test fixture", revision="test")
    assert first.read_bytes() == second.read_bytes()
    verify_source(first, source)
    with pytest.raises(FileExistsError):
        freeze_source(source, first, source_state="test fixture", revision="test")
    module.write_text("value = 2\n")
    with pytest.raises(ValueError, match="Current source"):
        verify_source(first, source)
