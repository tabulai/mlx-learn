"""CPU-only protocol tests; no estimator imports or timed workload launches."""

import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.knn_auto_control_tables import validate as validate_control
from benchmarks.knn_broad_worker import backend_metadata
from benchmarks.knn_first_query import (
    METRICS,
    VARIANTS,
    comparisons,
    select_specs,
    summarize,
    validate,
)
from benchmarks.knn_first_query_tables import render
from benchmarks.knn_order import phase_diagnostics, placement, round_order, williams_orders
from benchmarks.test_knn_followup_benchmarks import extended_control


def test_williams_position_and_predecessor_balance_all31_retained():
    names = list("abcde")
    rows = [round_order(names, i, "balanced") for i in range(31)]
    assert len(williams_orders(names)) == 10
    for position in range(5):
        assert Counter(r[position] for r in rows[:30]) == dict.fromkeys(names, 6)
        assert sorted(Counter(r[position] for r in rows).values()) == [6, 6, 6, 6, 7]
    adjacency = Counter((a, b) for row in rows[:30] for a, b in zip(row, row[1:], strict=False))
    assert len(adjacency) == 20 and set(adjacency.values()) == {6}
    assert round_order(names, 0, "legacy") == names
    assert round_order(names, 1, "legacy") == list(reversed(names))


def ordered_control():
    report = extended_control()
    names = report["variants"]
    report.update(order_policy="balanced", order_telemetry_version=1, order_block=williams_orders(names))
    for case in report["cases"]:
        for experiment in case["experiments"].values():
            experiment["starting_order"] = list(names)
            experiment["warmup_probe_completed_at_ns"] = dict.fromkeys(names, 100)
            previous, tick = names[-1], 1000
            for ri, row in enumerate(experiment["rounds"]):
                order = round_order(names, ri, "balanced")
                positions = placement(order, previous)
                row.update(order=order, placement=positions)
                for name in order:
                    experiment["variants"][name]["samples"][ri]["order_context"] = {
                        **positions[name], "pair": ri, "requested_at_ns": tick, "completed_at_ns": tick + 1,
                        "seconds_since_warmup_probe": (tick - 100) / 1e9}
                    tick += 10
                previous = order[-1]
            experiment["phase_diagnostics"] = phase_diagnostics(experiment, names)
    return report


def test_control_order_metadata_validates_and_cannot_be_relabelled():
    report = ordered_control()
    validate_control(report, expected_cases=1)
    first = next(iter(report["cases"][0]["experiments"].values()))
    first["rounds"][0]["order"].reverse()
    with pytest.raises(ValueError, match="order differs"):
        validate_control(report, expected_cases=1)


def test_control_phase_diagnostics_are_recomputed():
    report = ordered_control()
    first = next(iter(report["cases"][0]["experiments"].values()))
    first["phase_diagnostics"]["current"]["phase"]["all"]["median_ms"] = 5
    with pytest.raises(ValueError, match="diagnostics differ"):
        validate_control(report, expected_cases=1)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), "1", 0, -1])
def test_phase_diagnostics_retains_invalid_denominator_rounds(bad):
    report = ordered_control()
    experiment = next(iter(report["cases"][0]["experiments"].values()))
    experiment["variants"]["current"]["samples"][0]["ms"] = bad
    experiment["variants"]["current_gpu"]["samples"][1]["audit"]["pass"] = False
    result = phase_diagnostics(experiment, report["variants"])
    current = result["current"]["phase"]["all"]
    other = result["current_gpu"]["phase"]["all"]
    assert current["invalid_rounds"] == [0] and current["invalid_count"] == 1
    assert other["valid_rounds"] == [0, 1]
    assert other["paired_invalid_rounds"] == [0] and other["paired_invalid_count"] == 1
    assert other["paired_valid_rounds"] == [1] and other["median_ratio_over_current"] == 1


def fake_first_query():
    spec = select_specs()[0]
    repeats = 2
    model = SimpleNamespace(execution_backend_="cpu", execution_policy_="auto", last_query_backend_="cpu")
    variants, workers, setup = {}, {}, {}
    source = {"fake.py": "source"}
    for i, name in enumerate(VARIANTS):
        metadata = backend_metadata(model, current_mode="auto", is_sklearn=name.startswith("sklearn"), operation="kneighbors")
        samples = []
        for ri in range(repeats):
            phase = {**metadata, "audit": {"pass": True}}
            samples.append({"round": ri, "model_number": ri + 1, "first_model_in_worker": ri == 0,
                "first_model_for_fixture": ri == 0, "loaded_modules_before_fit": {"mlxlearn._kernels._neighbors_cpu_native": False},
                "timings": {"fit_ms": 1.0, "first_query_ms": 2.0, "fit_first_query_ms": 3.0, "second_query_ms": 1.0},
                "first": copy.deepcopy(phase), "second": copy.deepcopy(phase)})
        variants[name] = summarize(samples, repeats)
        setup[name] = {"model_prewarm_count": 0, "query_prewarm_count": 0, "pid": 100 + i,
            "loaded_modules_before_imports": {"numpy": False}, "numpy_import_ms": 1.0,
            "estimator_import_setup_ms": 2.0, "worker_entry_to_initialized_ms": 3.0,
            "parent_process_start_to_initialized_ms": 4.0}
        workers[name] = {"setup": copy.deepcopy(setup[name]), "source_hashes": source,
            "model_count": repeats, "fixture_round_counts": {spec["case"]["name"]: repeats},
            "native_cpu_extension": {"available": False, "source": None, "binary": None}}
    rows, previous = [], None
    for ri in range(repeats):
        order = round_order(VARIANTS, ri, "balanced")
        rows.append({"round": ri, "order": order, "placement": placement(order, previous)})
        previous = order[-1]
    case = {"case": spec["case"], "spec": spec, "variants": variants, "rounds": rows,
            "comparisons": comparisons(variants)}
    report = {"status": "complete", "variants": list(VARIANTS), "metrics": list(METRICS),
        "planned_cases": 1, "repeats": repeats, "cases": [case], "setup": setup, "workers": workers,
        "fixtures": [{"case": spec["case"], "fixture_sha256": "fixture"}], "reference_runtime_integrity_pass": True,
        "quality_by_variant": dict.fromkeys(VARIANTS, True), "timing_count_pass": True, "acceptance_pass": True,
        "protocol": "Synthetic initialized-process evidence"}
    for stem in ("source", "harness", "archive", "native", "fixture"):
        values = ({"current": None} if stem == "native" else {spec["case"]["name"] + ".npz": "fixture"}
                  if stem == "fixture" else source)
        report[f"{stem}_hashes_before"] = report[f"{stem}_hashes_after"] = values
        report[f"{stem}_integrity_pass"] = True
    return report


def test_fresh_model_all_first_second_calls_retained_and_rendered():
    report = fake_first_query()
    validate(report, expected_cases=1, expected_repeats=2)
    text, summary = render(report, Path("raw.json"), Path("report.md"), expected_cases=1, expected_repeats=2)
    assert summary["audit_counts"]["current"]["first_queries"] == 2
    assert summary["audit_counts"]["current"]["second_queries"] == 2
    assert "first_model_for_fixture" in text and "subsequent_fresh_models" in text
    assert "Estimator import/setup ms" in text
    assert "first_query_ms" in text and "second_query_ms" in text


@pytest.mark.parametrize("mutation", ["prewarm", "reuse", "timing", "missing_second_audit", "ratio"])
def test_fresh_model_protocol_rejects_hidden_work_or_missing_evidence(mutation):
    report = fake_first_query()
    sample = report["cases"][0]["variants"]["current"]["samples"][0]
    if mutation == "prewarm":
        report["setup"]["current"]["query_prewarm_count"] = 1
    elif mutation == "reuse":
        sample["model_number"] = 2
    elif mutation == "timing":
        sample["timings"]["fit_first_query_ms"] = 4
    elif mutation == "missing_second_audit":
        del sample["second"]
    else:
        report["cases"][0]["comparisons"]["sklearn_default"]["first_query_ms"]["ratio_of_medians"] = 9
    with pytest.raises((ValueError, KeyError)):
        validate(report, expected_cases=1, expected_repeats=2)


def test_failed_numerical_audit_does_not_remove_first_query_timing():
    report = fake_first_query()
    case = report["cases"][0]
    samples = case["variants"]["sklearn_default"]["samples"]
    samples[0]["first"]["audit"]["pass"] = False
    case["variants"]["sklearn_default"] = summarize(samples, 2)
    report["quality_by_variant"]["sklearn_default"] = False
    validate(report, expected_cases=1, expected_repeats=2)
    assert case["variants"]["sklearn_default"]["completed_samples"] == 2
    assert case["comparisons"]["sklearn_default"]["first_query_ms"]["ratio_of_medians"] == 1
