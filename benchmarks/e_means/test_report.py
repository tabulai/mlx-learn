"""Pure-CPU checks that the report cannot invent or mismatch measurements."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("emeans_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def row(method="metal", epochs=3, stage="core1m"):
    return {
        "status": "completed", "n": 1_000_000, "d": 96, "k": 16384,
        "config": {"dataset": "deep1b-1M", "method": method, "seed": 10, "precision": "float32",
                   "assignment": "auto", "warmup": True, "lloyd_iters": 25, "epochs": epochs,
                   "recipe": "paper", "finalization": "last_epoch"},
        "train_seconds": 2 if method == "metal" else 10,
        "train_seconds_samples": [2] if method == "metal" else [10], "test_mse": .9 if method == "metal" else 1,
        "initial_centroids_sha256": "initial", "source_sha256": {"kernel": "source"},
        "versions": {"python": "3.14.2"}, "hardware": {"cpu": "M4 Max"}, "timing_scope": "fit",
        "dataset": {"train_sample_sha256": "train", "test_sha256": "test"},
        "_path": Path(f"/repo/benchmarks/e_means/results/{stage}/{method}-{epochs}.json"),
    }


def test_paired_speedup_is_derived_from_measured_times():
    pairs, unpaired = report.matched_pairs([row(), row("metal-lloyd")])
    assert not unpaired
    assert pairs[0]["speedup"] == 5
    assert pairs[0]["mse_change_pct"] == pytest.approx(-10)


@pytest.mark.parametrize("field", ["source_sha256", "versions", "hardware", "dataset", "initial_centroids_sha256"])
def test_different_provenance_is_never_paired(field):
    a, b = row(), row("metal-lloyd")
    b[field] = "different"
    pairs, unpaired = report.matched_pairs([a, b])
    assert not pairs
    assert unpaired == [a]


def test_warmed_and_unwarmed_fits_cannot_be_paired():
    a, b = row(), row("metal-lloyd")
    b["config"]["warmup"] = False
    assert not report.matched_pairs([a, b])[0]


def test_repeated_measurement_of_same_seed_is_not_another_quality_seed():
    first, longer = row(), row()
    longer["train_seconds_samples"] = [1.9, 2, 2.1]
    longer["_path"] = longer["_path"].with_name("longer.json")
    pairs, _ = report.matched_pairs([first, longer, row("metal-lloyd")])
    groups = report.paired_groups(pairs)
    assert groups[0]["seeds"] == [10]
    assert len(groups[0]["pairs"]) == 1
    assert len(groups[0]["pairs"][0]["emeans"]["train_seconds_samples"]) == 3


def test_quality_crossing_uses_actual_loss_not_fastest_runtime():
    early = row(epochs=1, stage="convergence")
    early["test_mse"] = 1.1
    later = row(epochs=2, stage="convergence")
    later["test_mse"] = .95
    pairs, _ = report.matched_pairs([early, later, row(), row("metal-lloyd")])
    table = "\n".join(report.crossing_table(pairs, Path("/repo/docs/report.md")))
    assert "| DEEP | 1, 2, 3 | 2 |" in table


def test_bad_recorded_median_is_rejected(tmp_path):
    item = copy.deepcopy(row())
    item.pop("_path")
    item["train_seconds"] = 99
    (tmp_path / "run.json").write_text(json.dumps(item))
    with pytest.raises(ValueError, match="median"):
        report.evidence_rows(tmp_path)


def test_protocol_description_labels_actual_cold_fit():
    item = row(stage="vibe")
    item["config"]["warmup"] = False
    text = report.protocol_description([item], "vibe")
    assert "without a full warm-up" in text
    assert "timed fits per configuration: 1" in text
