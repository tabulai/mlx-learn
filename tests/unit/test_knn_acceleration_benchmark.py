"""Pure-report checks for accuracy-qualified, multi-seed KNN speed claims."""

from __future__ import annotations

import copy
import json
import statistics
import sys

import pytest
from benchmarks import knn_acceleration as benchmark


def _measurement(values):
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values), "max_ms": max(values),
        "samples": [{"ms": value} for value in values],
    }


def _report(seed):
    variants = {
        name: {operation: _measurement([elapsed] * 9)
               for operation in ("fit", *benchmark.TARGET_OPERATIONS)}
        for name, elapsed in (("current", 1.0), ("reference", 4.0),
                              ("sklearn_1", 12.0), ("sklearn_default", 3.0))
    }
    return {
        "seed": seed,
        "cases": [{"case": {"name": benchmark.CASE}, "variants": variants}],
        "reference": {"revision": "frozen-fixture"},
        "workers": {"current": {"source_hashes": {"neighbors.py": "a"}, "hardware": {}}},
        "implementation_quality_pass": True,
        "quality_by_variant": dict.fromkeys(variants, True),
        "source_integrity_pass": True,
    }


def test_every_search_operation_and_seed_must_pass():
    reports = [_report(seed) for seed in (41, 42, 43)]
    # Fit alone remains reported but does not replace the fresh-fit/query target.
    reports[0]["cases"][0]["variants"]["current"]["fit"] = _measurement([100] * 9)
    assert benchmark.summarize_reports(reports, target_speedup=2)["all_target_operations_pass"]
    reports[1]["cases"][0]["variants"]["sklearn_default"]["regress_distance"] = _measurement([1.99] * 9)
    summary = benchmark.summarize_reports(reports, target_speedup=2)
    assert not summary["all_target_operations_pass"]
    assert summary["operations"]["regress_distance"]["minimum_speedup_vs_sklearn_default"] == 1.99


def test_round_counts_and_halves_expose_late_slowdown():
    report = _report(41)
    report["cases"][0]["variants"]["current"]["kneighbors"] = _measurement([1] * 5 + [4] * 4)
    summary = benchmark.summarize_reports([report], target_speedup=2)
    query = summary["operations"]["kneighbors"]["seeds"][0]
    assert query["paired_round_count"] == 9
    assert query["paired_rounds_meeting_target"] == 5
    assert query["halves"]["first"]["speedup_vs_sklearn_default"] == 3
    assert query["halves"]["second"]["speedup_vs_sklearn_default"] == 0.75
    assert summary["all_target_medians_pass"]
    assert not summary["all_target_operations_pass"]


@pytest.mark.parametrize("operation", benchmark.TARGET_OPERATIONS)
@pytest.mark.parametrize("round_index", range(9))
def test_single_timed_miss_fails_even_when_all_medians_and_halves_pass(operation, round_index):
    reports = [_report(seed) for seed in (41, 42, 43)]
    durations = [1] * 9
    durations[round_index] = 2
    reports[1]["cases"][0]["variants"]["current"][operation] = _measurement(durations)
    summary = benchmark.summarize_reports(reports, target_speedup=2)
    assert summary["all_target_medians_pass"]
    assert not summary["all_target_operations_pass"]
    assert summary["target_paired_round_count"] == 243
    assert summary["target_paired_rounds_meeting_target"] == 242
    assert summary["minimum_target_paired_round_speedup"] == 1.5
    measurement = summary["operations"][operation]["seeds"][1]
    assert all(value["speedup_vs_sklearn_default"] == 3
               for value in measurement["halves"].values())
    assert measurement["paired_round_misses"] == [{
        "round": round_index + 1,
        "current_ms": 2,
        "sklearn_default_ms": 3,
        "speedup_vs_sklearn_default": 1.5,
        "maximum_current_ms_for_target": 1.5,
    }]


def test_exact_target_passes_and_faster_sklearn_sample_is_not_hidden():
    report = _report(41)
    report["cases"][0]["variants"]["sklearn_default"]["fit_query"] = _measurement([2] * 9)
    assert benchmark.summarize_reports([report], target_speedup=2)["all_target_operations_pass"]
    report["cases"][0]["variants"]["sklearn_default"]["fit_query"] = _measurement([2] * 8 + [1.999])
    summary = benchmark.summarize_reports([report], target_speedup=2)
    assert summary["all_target_medians_pass"]
    assert not summary["all_target_operations_pass"]
    assert summary["minimum_target_paired_round_speedup"] == 1.999


def test_first_call_observations_remain_separate_from_retained_pairs():
    report = _report(41)
    report["cases"][0]["variants"]["current"]["cold"] = {"fit_ms": 10, "query_ms": 100}
    report["cases"][0]["variants"]["sklearn_default"]["cold"] = {"fit_ms": 1, "query_ms": 1}
    summary = benchmark.summarize_reports([report], target_speedup=2)
    assert summary["all_target_operations_pass"]
    assert summary["target_paired_round_count"] == 81
    assert report["cases"][0]["variants"]["current"]["cold"]["query_ms"] == 100


def test_unpaired_diagnostic_blocks_keep_their_separate_median_policy():
    report = _report(41)
    report["cases"][0]["variants"]["current"]["fit_query"] = _measurement([1] * 8 + [2])
    paired = benchmark.summarize_reports([report], target_speedup=2)
    unpaired = benchmark.summarize_reports([report], target_speedup=2, paired_rounds=False)
    assert not paired["all_target_operations_pass"]
    assert unpaired["all_target_operations_pass"]
    assert unpaired["all_target_medians_pass"]
    assert not any("paired_round" in key for key in unpaired)
    for operation in unpaired["operations"].values():
        assert not any("paired_round" in key for key in operation)
        for sample in operation["seeds"]:
            assert not any("paired_round" in key for key in sample)


@pytest.mark.parametrize("values", [[], [0] * 9, [float("nan")] * 9, [float("inf")] * 9])
def test_invalid_or_missing_timed_samples_cannot_pass(values):
    report = _report(41)
    measurement = report["cases"][0]["variants"]["current"]["kneighbors"]
    measurement["samples"] = [{"ms": value} for value in values]
    with pytest.raises(ValueError):
        benchmark.summarize_reports([report], target_speedup=2)


@pytest.mark.parametrize("failure", [None, "accuracy", "source", "speed", "single_round"])
def test_cli_success_requires_accuracy_stable_source_and_speed(tmp_path, monkeypatch, failure):
    def fake_run(options):
        assert options.repeats == 9
        assert options.warmup == 3
        assert options.threads == "both"
        assert options.resident is True
        report = copy.deepcopy(_report(options.seed))
        if options.seed == 42:
            if failure == "accuracy":
                report["implementation_quality_pass"] = False
                report["quality_by_variant"]["current"] = False
            elif failure == "source":
                # Each process can pass its own check while source changes between seeds.
                report["workers"]["current"]["source_hashes"]["neighbors.py"] = "b"
            elif failure == "speed":
                report["cases"][0]["variants"]["sklearn_default"]["fit_query"] = _measurement([1] * 9)
            elif failure == "single_round":
                report["cases"][0]["variants"]["current"]["fit_query"] = _measurement([1] * 8 + [2])
        options.output.write_text(json.dumps(report))
        return 0

    output = tmp_path / "summary.json"
    monkeypatch.setattr(benchmark, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["knn_acceleration", "--output", str(output)])
    assert benchmark.main() == (0 if failure is None else 1)
    summary = json.loads(output.read_text())
    assert len(summary["reports"]) == 3
