"""CPU-only checks for the Lloyd-25 reference addendum driver and report."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from tests.unit.test_kmeans_report import campaign as frozen_campaign

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/e_means"
sys.path.insert(0, str(BENCH))
import kmeans_lloyd25 as driver  # noqa: E402
import kmeans_lloyd25_report as report  # noqa: E402

PROTOCOL = BENCH / "kmeans_lloyd25_protocol.json"
LLOYD, EMEANS = driver.MODES


def campaign(lloyd_floor=1.5):
    """The frozen campaign's Metal rows, re-budgeted as the addendum's two arms.

    Lloyd inertia after ``u`` updates is ``n * (lloyd_floor + 2**-u)`` and each
    iteration takes 10 s; e-means trains for 20 s and ends at ``1.6 * n``.
    """
    protocol, entries = driver.load_protocol(PROTOCOL)
    digest = report.common.sha256(PROTOCOL)
    rows = []
    for row in frozen_campaign()[0]:
        config, n = row["config"], row["n"]
        if config["mode"] not in driver.MODES:
            continue
        row["source_sha256"]["benchmarks/e_means/kmeans_lloyd25_protocol.json"] = digest
        config.update(addendum=driver.KIND, addendum_source_sha256={"driver": "9" * 64})
        if config["mode"] == LLOYD:
            budget = driver.protocol_entry(config["protocol_config"], LLOYD, protocol["lloyd_iterations"])
            config.update(epochs=budget["epochs"], protocol_config=budget)
            inertias = [n * (lloyd_floor + 2.0**-updates) for updates in range(26)]
            row.update(actual_iterations=25, train_seconds=260.0, train_seconds_samples=[260.0],
                       final_inertia=inertias[25],
                       iteration_history=[{"iteration": i + 1, "seconds": 10.0, "inertia_before_update": inertias[i]}
                                          for i in range(25)])
        else:
            epochs = config["epochs"]
            row.update(train_seconds=30.0, train_seconds_samples=[30.0], test_mse=0.99, final_inertia=1.6 * n,
                       iteration_history=[{"epoch": i + 1, "cumulative_seconds": 20.0 * (i + 1) / epochs}
                                          for i in range(epochs)])
        rows.append(row)
    return rows, protocol, entries, digest


def test_addendum_changes_only_the_lloyd_budget():
    protocol, entries = driver.load_protocol(PROTOCOL)
    frozen = json.loads((BENCH / "kmeans_protocol.json").read_text())["configurations"]
    assert protocol["lloyd_iterations"] == 25 and entries == frozen
    for entry in entries:
        assert driver.protocol_entry(entry, EMEANS, 25) is entry
        assert driver.protocol_entry(entry, LLOYD, 25) == {**entry, "epochs": 25}


def test_addendum_refuses_a_different_frozen_protocol(tmp_path):
    changed = json.loads(PROTOCOL.read_text())
    changed["applies_to"]["sha256"] = "0" * 64
    path = tmp_path / "addendum.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Frozen comparison protocol differs"):
        driver.load_protocol(path)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(driver.benchmark, "sources", lambda _: {"driver": "hash"})
    monkeypatch.setattr(driver.previous, "reference_sources", lambda _: {"reference": "hash"})
    monkeypatch.setattr(driver.benchmark, "sklearn_sources", lambda: {"sklearn": "hash"})
    monkeypatch.setattr(driver.base, "package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(driver.base, "host_identity", lambda: {"cpu": "test"})
    monkeypatch.setattr(driver.benchmark, "measured_run", lambda config, timeout: seen.append(copy.deepcopy(config)))

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", [driver.__file__, "--out", str(tmp_path), *arguments])
        driver.main()
        return seen

    return run


def test_default_cli_runs_lloyd_at_25_and_emeans_at_frozen_budgets(cli):
    seen = cli()
    assert len(seen) == 10 and all(c["addendum"] == driver.KIND and not c["diagnostic_subset"] for c in seen)
    assert [c["epochs"] for c in seen if c["mode"] == LLOYD] == [25] * 5
    assert [c["epochs"] for c in seen if c["mode"] == EMEANS] == [2, 2, 2, 2, 3]
    assert all(c["epochs"] == c["protocol_config"]["epochs"] and c["epochs_override"] is None for c in seen)
    # The first arm alternates between datasets.
    assert [c["mode"] for c in seen[:4]] == [LLOYD, EMEANS, EMEANS, LLOYD]


def test_cli_runs_named_datasets_in_the_given_order(cli):
    seen = cli("--datasets", "sift1b-10M-div512", "deep1b-10M", "--modes", LLOYD)
    assert [c["dataset"] for c in seen] == ["sift1b-10M-div512", "deep1b-10M"]


def test_cli_overrides_mark_a_diagnostic_subset(cli):
    seen = cli("--rows", "1000", "--k", "16", "--lloyd-iterations", "4", "--datasets", "deep1b-10M")
    lloyd, emeans = seen
    assert all(c["diagnostic_subset"] and c["rows"] == 1000 and c["k_override"] == 16 for c in seen)
    assert (lloyd["epochs"], lloyd["epochs_override"]) == (4, 4)
    assert (emeans["epochs"], emeans["epochs_override"]) == (2, None)


def test_record_is_withdrawn_if_addendum_sources_change_during_a_fit(cli, monkeypatch):
    hashes = iter([{"driver": "before"}, {"driver": "after"}])
    monkeypatch.setattr(driver, "addendum_sources", lambda: next(hashes))
    monkeypatch.setattr(driver.benchmark, "measured_run", lambda config, timeout: Path(config["output"]).write_text("{}"))
    with pytest.raises(RuntimeError, match="sources changed"):
        cli("--datasets", "deep1b-10M", "--modes", LLOYD)
    assert not list(Path(sys.argv[2]).glob("*.json"))


def test_complete_campaign_pairs_ratios_and_inertia_crossing():
    rows, protocol, entries, digest = campaign()
    pairs, validation = report.validate(rows, protocol, entries, digest)
    assert validation["complete"] and validation["accepted_records"] == 10
    summary = report.summarize(pairs)
    first = summary[0]
    assert len(summary) == 5 and (first["lloyd_budget"], first["lloyd_iterations"], first["emeans_epochs"]) == (25, 25, 2)
    assert first["runtime_ratio"] == 260 / 30 and first["training_ratio"] == 250 / 20
    assert first["test_mse_change_pct"] == pytest.approx(-1.0)
    # 1.5 + 2**-u first falls to 1.6 after four updates, which cost 40 s.
    assert first["crossing_updates"] == 4 and first["matched_training_ratio"] == 2.0
    assert first["inertia_change_pct"] == pytest.approx(100 * (1.6 / (1.5 + 2.0**-25) - 1))
    text = report.headline(summary)
    assert "8.67× to 8.67×" in text and "12.50× to 12.50×" in text and "equal or lower on 5/5" in text


def test_lloyd_that_never_reaches_emeans_inertia_reports_a_lower_bound(tmp_path):
    rows, protocol, entries, digest = campaign(lloyd_floor=1.7)
    pairs, validation = report.validate(rows, protocol, entries, digest)
    summary = report.summarize(pairs)
    assert all(row["crossing_updates"] is None and row["matched_training_ratio"] is None for row in summary)
    for row in summary:
        row.update(lloyd_evidence_path=str(tmp_path / "lloyd.json"), emeans_evidence_path=str(tmp_path / "emeans.json"))
    text = report.markdown(summary, validation, protocol, tmp_path / "report.md", tmp_path, PROTOCOL)
    assert "| 10.00 [10.00, 10.00] |" in text and "| not reached in 25 | > 12.50× |" in text and "did not reach the e-means training inertia within its budget on 5/5" in text


def test_missing_arm_is_incomplete_and_unpaired():
    rows, protocol, entries, digest = campaign()
    pairs, validation = report.validate(rows[:-1], protocol, entries, digest)
    assert not validation["complete"] and len(pairs) == 4
    assert validation["missing"] == [{"dataset": "sift1b-10M-div512", "seed": 10, "mode": EMEANS}]


@pytest.mark.parametrize("change", ["matched_budget", "not_addendum", "history", "protocol_hash", "memory"])
def test_bad_evidence_is_rejected(change):
    rows, protocol, entries, digest = campaign()
    lloyd = rows[0]
    if change == "matched_budget":
        frozen_entry = entries[0]
        lloyd["config"].update(epochs=frozen_entry["epochs"], protocol_config=frozen_entry)
        lloyd.update(actual_iterations=2, iteration_history=lloyd["iteration_history"][:2])
    elif change == "not_addendum":
        del lloyd["config"]["addendum"]
    elif change == "history":
        lloyd["iteration_history"] = lloyd["iteration_history"][:-1]
    elif change == "protocol_hash":
        digest = "0" * 64
    else:
        lloyd["memory_samples"][0]["sampling_errors"] = ["lost observation"]
    _, validation = report.validate(rows, protocol, entries, digest)
    assert not validation["complete"] and validation["rejected"]


@pytest.mark.parametrize("change", ["initial", "cache"])
def test_arms_must_share_data_initialization_and_conditions(change):
    rows, protocol, entries, digest = campaign()
    if change == "initial":
        rows[0]["initial_centroids_sha256"] = "9" * 64
    else:
        rows[0]["config"]["cache"] = "/different"
    pairs, validation = report.validate(rows, protocol, entries, digest)
    assert len(pairs) == 4 and validation["accepted_records"] == 8 and len(validation["rejected"]) == 2


def test_mixed_sources_reject_the_entire_campaign():
    rows, protocol, entries, digest = campaign()
    rows[0]["config"]["addendum_source_sha256"] = {"driver": "8" * 64}
    pairs, validation = report.validate(rows, protocol, entries, digest)
    assert not pairs and validation["accepted_records"] == 0


def test_partial_report_is_written_but_refused_when_completeness_is_required(tmp_path):
    rows = campaign()[0][:2]
    for index, row in enumerate(rows):
        (tmp_path / f"record{index}.json").write_text(json.dumps(row))
    validation = report.generate(tmp_path, tmp_path / "report.md", PROTOCOL)
    assert not validation["complete"] and validation["accepted_records"] == 2
    text = (tmp_path / "report.md").read_text()
    assert "INCOMPLETE: 2/10" in text and "Across the 1 paired dataset," in text
    assert "below: DINO10B prefix, SimSearchNet++, SIFT, SPACEV." in text
    assert (tmp_path / "summary.csv").read_text().count("\n") == 2
    with pytest.raises(ValueError, match="Incomplete"):
        report.generate(tmp_path, tmp_path / "final.md", PROTOCOL, require_complete=True)
    assert not (tmp_path / "final.md").exists()
