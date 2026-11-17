"""Pure-CPU checks of fresh gap-report pairing, calculations and artifacts."""

import copy
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_report as report  # noqa: E402
from test_gap_bench import record as runtime_record  # noqa: E402
from test_h100 import row as protocol_record  # noqa: E402


def row(mode="previous", seed=10):
    value = protocol_record("optimized" if mode == "optimized" else "baseline", seed)
    runtime = runtime_record(mode, seed)
    c = value["config"]
    shared = {key: digest for key, digest in value["source_sha256"].items() if not key.startswith("src/")}
    shared["benchmarks/e_means/gap_bench.py"] = "gap"
    c.update(mode=mode, assignment="auto", shared_source_sha256=shared,
             runtime_comparison=runtime["config"]["runtime_comparison"], archive_sha256="archive")
    c["expected_source_sha256"] = {**shared, **c["runtime_comparison"][mode]}
    value.update(kind=report.bench.KIND, shared_source_sha256=shared,
                 runtime_source_sha256=c["runtime_comparison"][mode], loaded_runtime=runtime["loaded_runtime"],
                 source_sha256=c["expected_source_sha256"])
    value["recipe_resolved"]["assignment"] = "auto"
    value["_path"] = Path(f"/repo/results/{mode}-s{seed}.json")
    return value


def build(rows):
    valid, pairs, unpaired, excluded = report.validate_and_pair(rows)
    return valid, pairs, unpaired, excluded, report.groups(pairs)


def test_pairs_only_fresh_runtime_variants_and_uses_median_of_ratios():
    a, b, c, d = row(), row("optimized"), row(seed=11), row("optimized", seed=11)
    c["train_seconds"], c["train_seconds_samples"] = 9, [9]
    valid, pairs, unpaired, excluded, grouped = build([a, b, c, d])
    assert len(valid) == 4 and len(pairs) == 2 and not unpaired and not excluded
    assert grouped[0]["speedup"]["median"] == 6
    assert grouped[0]["previous_seconds"]["median"] == 6
    assert grouped[0]["optimized_seconds"]["median"] == 1
    assert grouped[0]["paired_seeds"] == [10, 11]


@pytest.mark.parametrize("change", ["historical", "override", "off_manifest", "proof", "source"])
def test_invalid_evidence_is_quarantined(change):
    value = row()
    if change == "historical":
        value["kind"] = "h100_protocol_metal"
    elif change == "override":
        value["config"]["rows"] = 1000
    elif change == "off_manifest":
        value["config"]["seed"] = 999
    elif change == "proof":
        value["loaded_runtime"]["verified"] = False
    else:
        value["source_sha256"] = {"unexpected": "source"}
    valid, pairs, unpaired, excluded, _ = build([value, row("optimized")])
    assert len(valid) == len(unpaired) == len(excluded) == 1
    assert not pairs


def test_ambiguous_duplicates_are_all_excluded():
    original = row()
    valid, pairs, unpaired, excluded, _ = build([original, copy.deepcopy(original), row("optimized")])
    assert len(excluded) == 2 and len(unpaired) == 1 and len(valid) == 1 and not pairs
    assert all("duplicate" in reason for _, reason in excluded)


def test_conflicting_campaign_sources_across_seeds_are_not_pooled():
    first = [row(), row("optimized")]
    second = [row(seed=11), row("optimized", seed=11)]
    for record in second:
        c = record["config"]
        c["runtime_comparison"]["optimized"]["unused.py"] = "new-version"
        if c["mode"] == "optimized":
            c["expected_source_sha256"]["unused.py"] = "new-version"
    valid, pairs, unpaired, excluded, grouped = build(first + second)
    assert not valid and not pairs and not unpaired and not grouped
    assert len(excluded) == 4
    assert all("conflicting" in reason for _, reason in excluded)


def test_different_runtime_campaigns_across_datasets_fail_loudly():
    first = [row(), row("optimized")]
    second = [row(), row("optimized")]
    for record in second:
        c = record["config"]
        c["dataset"] = c["protocol_config"]["dataset"] = "different-dataset"
        c["runtime_comparison"]["optimized"]["unused.py"] = "new-version"
        if c["mode"] == "optimized":
            c["expected_source_sha256"]["unused.py"] = "new-version"
    with pytest.raises(RuntimeError, match="Mixed campaign"):
        build(first + second)


def test_unpaired_extra_seed_does_not_change_any_summary():
    valid, pairs, unpaired, excluded, grouped = build([row(), row("optimized"), row("optimized", seed=11)])
    assert len(valid) == 3 and len(unpaired) == 1 and not excluded
    assert grouped[0]["pair_count"] == 1 and grouped[0]["optimized_seconds"]["sample_sd"] is None
    assert grouped[0]["optimized_mlx_active_complete_seeds"] == 1


def test_memory_metric_missing_for_one_seed_withholds_group_and_keeps_counts():
    a, b, c, d = row(), row("optimized"), row(seed=11), row("optimized", seed=11)
    d["mlx_memory_samples"] = []
    _, _, _, _, grouped = build([a, b, c, d])
    assert grouped[0]["optimized_mlx_active_bytes"] is None
    assert grouped[0]["optimized_mlx_active_complete_seeds"] == 1
    assert grouped[0]["optimized_footprint_bytes"] is not None


def test_report_keeps_quality_range_tradeoffs_and_h100_caveat():
    a, b = row(), row("optimized")
    a["test_mse"], b["test_mse"] = 100, 100.02
    b["mlx_memory_samples"] = [{"peak_active_bytes": 4 * 2**30}] * 3
    for value in (a, b):
        value["config"]["protocol_config"]["reported_h100_emeans_seconds"] = .1
    valid, pairs, unpaired, excluded, grouped = build([a, b])
    text = report.make_report(valid, pairs, unpaired, excluded, grouped, Path("/repo/report.md"), Path("/repo/results"))
    assert grouped[0]["contextual_apple_h100_ratio"] == 10
    assert "not an equal-work or equal-quality hardware speedup" in text
    assert "not the speedup baseline" in text
    assert "+0.02% to +0.02%" in text
    assert "Not all memory metrics improve" in text
    assert "Repeated calls are not independent seeds" in text
    assert "Partial snapshot: 1/17 datasets and 1/51 planned seed pairs" in text
    assert "Higher-dimension assignment kernels are unchanged" in text


def test_main_writes_separate_csvs_manifest_and_report_without_plotting(tmp_path, monkeypatch):
    folder = tmp_path / "fresh"
    folder.mkdir()
    for record in (row(), row("optimized")):
        path = folder / record.pop("_path").name
        path.write_text(json.dumps(record))
    output = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", [report.__file__, "--results", str(folder), "--output", str(output)])
    report.main()
    manifest = json.loads((folder / "report_manifest.json").read_text())
    assert manifest["paired_seeds"] == manifest["paired_datasets"] == 1
    assert len(manifest["evidence"]) == 2
    with (folder / "pairs.csv").open() as handle:
        pair, = list(csv.DictReader(handle))
    assert float(pair["speedup"]) == 3
    assert Path(pair["previous_evidence_path"]).exists()
    assert output.exists()
    # Generated manifests are never treated as new benchmark measurements.
    assert len(report.evidence_rows(folder)) == 2


def test_empty_results_produce_pending_report_not_fabricated_values(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [report.__file__, "--results", str(tmp_path / "empty"),
                                     "--output", str(tmp_path / "report.md")])
    report.main()
    assert "0/17 datasets" in (tmp_path / "report.md").read_text()
