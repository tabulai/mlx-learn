"""Small CPU-only tests for report verification, with fabricated evidence."""

import copy

import pytest

from benchmarks.knn_auto_verify import (
    COMPARATORS,
    control_audit_counts,
    control_cells,
    fixture_inventory,
    paired_ratio,
    threshold_counts,
)
from benchmarks.test_knn_auto_harness import fake_control


def test_every_slow_cell_remains_visible():
    cells = [{"speedup_by_comparator": dict.fromkeys(COMPARATORS, r)}
             for r in (0.5, 1.0, 2.0, None)]
    result = threshold_counts(cells)
    for counts in result.values():
        assert counts["cell_count"] == 4
        assert counts["faster_than_baseline"] == 1
        assert counts["at_least_baseline"] == 2
        assert counts["at_least_two_x"] == 1
        assert counts["slower_or_missing_cells"] == [cells[0], cells[3]]


def test_both_auto_replicas_are_measured_separately():
    report = fake_control()
    experiment = next(iter(report["cases"][0]["experiments"].values()))
    for sample in experiment["variants"]["current_b"]["samples"]:
        sample["ms"] = 0.25
    for sample in experiment["variants"]["previous_gpu"]["samples"]:
        sample["ms"] = 0.5
    first = control_cells(report, preservation=False)[0]
    assert first["paired_ratios"]["previous_gpu_over_auto"]["median"] == 0.5
    assert first["paired_ratios"]["previous_gpu_over_auto_aa"]["median"] == 2.0
    assert not first["auto_slower_in_both_replicas"]["previous_gpu"]


def test_audit_counts_include_every_phase_and_failed_audit():
    report = fake_control(passed=False)
    counts = control_audit_counts(report)
    for phases in counts.values():
        assert {phase: c["audits"] for phase, c in phases.items()} == {
            "cold": 1, "warmup": 9, "probe": 9, "retained": 18,
        }
        assert all(c["strict_passes"] == 0 for c in phases.values())


def test_failed_numerical_audit_does_not_filter_speed_ratio():
    samples = {"a": [{"ms": 1.0, "audit": {"pass": False}}, {"ms": float("nan")}],
               "b": [{"ms": 0.5, "audit": {"pass": True}}, {"ms": 2.0}]}
    result = paired_ratio(samples, "a", "b")
    assert result["ratios"] == [2.0]
    assert result["pair_count"] == 1


def test_fixture_inventory_rejects_missing_oracle_array():
    spec = {"case": {"name": "fixture"}}
    fixture = {"case": spec["case"], "fixture_sha256": "zip-hash",
               "array_sha256": {"X": "x-hash", "indices": "oracle-hash"},
               "array_shapes": {"X": [2, 1], "indices": [1, 1]},
               "array_dtypes": {"X": "float32", "indices": "int64"}}
    report = {"fixtures": [fixture], "cases": [{"case": spec["case"], "spec": spec}],
              "fixture_hashes_before": {"fixture.npz": "zip-hash"}}
    fixture_inventory(report)
    changed = copy.deepcopy(report)
    changed["fixtures"][0]["array_sha256"].pop("indices")
    with pytest.raises(ValueError, match="Array inventory"):
        fixture_inventory(changed)
