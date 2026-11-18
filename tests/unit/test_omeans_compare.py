"""Contracts of the o-means comparison driver and report. No GPU is used."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/e_means"
sys.path.insert(0, str(BENCH))
import omeans_compare as bench  # noqa: E402
import omeans_report as report  # noqa: E402


def _fit(mode, epochs, seconds, error, fraction=1.0, n=1000, k=10):
    return {"mode": mode, "epochs": epochs, "seconds": seconds, "test_mse": error,
            "evaluated_fraction": fraction, "evaluated_distances": fraction * n * k * epochs}


def _record(dataset="data", precision="float32", seed=10, fits=None, **changes):
    record = {
        "kind": bench.KIND, "dataset": dataset, "n": 1000, "d": 4, "k": 10,
        "precision": precision, "seed": seed, "rows_limit": None,
        "fits": fits if fits is not None else [
            _fit("metal-emeans", 1, 4.0, 10.0), _fit("metal-emeans", 2, 8.0, 9.0),
            _fit("metal-omeans", 1, 1.0, 10.5, 0.2), _fit("metal-omeans", 2, 2.0, 9.5, 0.2),
            _fit("metal-omeans", 4, 4.0, 8.5, 0.2),
            _fit("metal-omeans-dense", 1, 4.1, 9.9),
        ],
        "sources": {"src/mlxlearn/cluster/_o_means.py": "a"}, "versions": {"mlx": "0.32.4", "python": "3.14"},
        "host": {"cpu": "Apple M4 Max"}, "device": {"device_name": "Apple M4 Max"},
        "protocol_sha256": "p",
    }
    record.update(changes)
    return record


def test_protocol_lists_every_mode_and_positive_budgets(tmp_path):
    protocol = bench.load_protocol(bench.DEFAULT_PROTOCOL)
    assert set(protocol["epochs"]) == set(bench.MODES)
    assert protocol["epochs"]["metal-emeans"] == [1, 2, 3]
    names = [entry["dataset"] for entry in protocol["configurations"]]
    assert len(names) == len(set(names)) and all(entry["k"] > 0 for entry in protocol["configurations"])
    wrong = tmp_path / "protocol.json"
    wrong.write_text(json.dumps({**protocol, "kind": "other"}))
    with pytest.raises(ValueError, match="protocol"):
        bench.load_protocol(wrong)
    wrong.write_text(json.dumps({**protocol, "epochs": {"metal-emeans": [1, 0]}}))
    with pytest.raises(ValueError, match="epoch budgets"):
        bench.load_protocol(wrong)


def test_sources_cover_the_measured_packages_and_the_driver():
    captured = bench.sources()
    assert "benchmarks/e_means/omeans_compare.py" in captured
    assert "benchmarks/e_means/omeans_protocol.json" in captured
    for path in ("src/mlxlearn/cluster/_o_means.py", "src/mlxlearn/_kernels/o_means.py",
                 "src/mlxlearn/_kernels/e_means.py", "src/mlxlearn_device/_binary32.py"):
        assert path in captured
    assert all(len(digest) == 64 for digest in captured.values())


def test_timed_calls_and_record_names():
    assert [bench.expected_calls(s) for s in (0.0, 1.99, 2.0, 60.0)] == [3, 3, 1, 1]
    assert bench.record_name("spacev-1M", "bfloat16", 11) == "spacev-1M-bfloat16-s11.json"


def test_models_share_the_recipe_and_differ_only_in_the_search():
    pytest.importorskip("mlx.core")
    emeans = bench.model_for("metal-emeans", 64, 2, "throughput", "float32", 10)
    omeans = bench.model_for("metal-omeans", 64, 2, "throughput", "float32", 10)
    dense = bench.model_for("metal-omeans-dense", 64, 2, "throughput", "float32", 10)
    assert emeans.recipe == "faiss" and type(emeans).__name__ == "EMeans"
    assert (omeans.assignment, dense.assignment) == ("two_level", "dense")
    for model in (emeans, omeans, dense):
        assert (model.max_iter, model.batch_law, model.random_state, model.compute_labels) == (
            2, "throughput", 10, False)


def test_campaign_skips_measured_records_and_stops_on_a_failed_worker(tmp_path, monkeypatch):
    captured = bench.sources()
    calls = []

    def fake_run(command, **options):
        calls.append(json.loads(command[-1]))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    args = SimpleNamespace(
        protocol=bench.DEFAULT_PROTOCOL, datasets=["spacev-1M"], modes=list(bench.MODES),
        precision=["float32"], seeds=[10], rows=None, k=None, threads=8, timeout=10.0,
        cache=tmp_path, faiss_source=tmp_path, out=tmp_path, force=False)
    (tmp_path / "spacev-1M-float32-s10.json").write_text(json.dumps({"sources": captured}))
    bench.run(args)
    assert calls == []
    # A record from other sources is measured again; a failed worker stops the campaign.
    (tmp_path / "spacev-1M-float32-s10.json").write_text(json.dumps({"sources": {}}))
    with pytest.raises(SystemExit, match="no record was written"):
        bench.run(args)
    assert len(calls) == 1 and calls[0]["expected_sources"] == captured
    assert calls[0]["k"] == 16384 and calls[0]["modes"] == list(bench.MODES)
    with pytest.raises(SystemExit, match="Not in the protocol"):
        bench.run(SimpleNamespace(**{**vars(args), "datasets": ["unknown"]}))


def test_crossing_interpolates_log_time_and_reports_the_measured_budget():
    points = [{"epochs": e, "seconds": s, "error": err, "distances": s * 100}
              for e, s, err in [(1, 1.0, 10.5), (2, 2.0, 9.5), (4, 4.0, 8.5)]]
    reached = report.crossing(points, 10.0)
    assert reached["measured"] == 2.0 and reached["epochs"] == 2 and not reached["bounded"]
    assert reached["interpolated"] == pytest.approx(math.sqrt(2.0))
    assert reached["distances"] == pytest.approx(100 * math.sqrt(2.0))
    assert report.crossing(points, 9.5)["interpolated"] == pytest.approx(2.0)
    first = report.crossing(points, 11.0)
    assert first["bounded"] and first["interpolated"] == first["measured"] == 1.0
    assert report.crossing(points, 8.0) is None
    # A later budget may be worse than an earlier one; the first to reach the target counts.
    uneven = [{"epochs": e, "seconds": s, "error": err, "distances": s}
              for e, s, err in [(1, 1.0, 10.0), (1.6, 1.6, 9.0), (2, 2.0, 9.4), (3, 3.0, 8.8)]]
    assert report.crossing(uneven, 9.2)["epochs"] == 1.6
    assert report.crossing(uneven, 8.9)["epochs"] == 3


def test_error_at_a_time_budget():
    points = [{"seconds": 1.0, "error": 10.0}, {"seconds": 4.0, "error": 8.0}]
    assert report.error_at(points, 0.5) is None
    assert report.error_at(points, 2.0) == pytest.approx(9.0)
    assert report.error_at(points, 4.0) == pytest.approx(8.0)
    assert report.error_at(points, 100.0) == 8.0


def test_validation_rejects_mixed_campaigns_duplicates_and_subsets():
    assert report.validate([_record(), _record(seed=11)]) == {"records": 2, "datasets": 1, "seeds": [10, 11]}
    with pytest.raises(ValueError, match="No o-means"):
        report.validate([])
    with pytest.raises(ValueError, match="mixed source"):
        report.validate([_record(), _record(seed=11, sources={"changed": "b"})])
    with pytest.raises(ValueError, match="mixed source"):
        report.validate([_record(), _record(seed=11, device={"device_name": "Apple M5 Ultra"})])
    with pytest.raises(ValueError, match="duplicate"):
        report.validate([_record(), _record()])
    with pytest.raises(ValueError, match="diagnostic subset"):
        report.validate([_record(rows_limit=1000)])
    with pytest.raises(ValueError, match="invalid measurement"):
        report.validate([_record(fits=[_fit("metal-emeans", 1, 0.0, 1.0)])])


def test_summary_and_markdown_from_two_seeds(tmp_path):
    records = [_record(), _record(seed=11)]
    for index, record in enumerate(records):
        (tmp_path / f"r{index}.json").write_text(json.dumps(record))
    (tmp_path / "other.json").write_text(json.dumps({"kind": "something-else"}))
    loaded = report.load(tmp_path)
    assert len(loaded) == 2
    curves, shapes = report.aggregate(loaded)
    rows = report.summarize(curves, shapes)
    assert [row["epochs"] for row in rows] == [1, 2] and rows[0]["seeds"] == 2
    one, two = rows
    # e-means reaches 10.0 in 4 s; o-means crosses it between its 1 s and 2 s budgets.
    assert one["speedup"] == pytest.approx(4.0 / math.sqrt(2.0))
    assert one["speedup_measured"] == pytest.approx(2.0) and one["omeans_epochs_measured"] == 2
    assert one["distance_ratio"] == pytest.approx(0.2 * math.sqrt(2.0))
    assert one["equal_epochs_time_ratio"] == pytest.approx(4.0)
    assert one["equal_epochs_error"] == pytest.approx(0.05)
    assert one["dense_equal_epochs_error"] == pytest.approx(-0.01)
    assert one["error_at_equal_time"] == pytest.approx(8.5 / 10.0 - 1)
    assert two["dense_equal_epochs_error"] is None
    assert two["speedup"] == pytest.approx(8.0 / math.exp(math.log(2.0) + 0.5 * math.log(2.0)))
    text = report.markdown(rows, report.validate(loaded), loaded[0]["device"], loaded[0]["versions"])
    assert "## float32" in text and "| data | 1,000 | 4 | 10 |" in text
    assert "Geometric mean" in text and "Apple M4 Max" in text
    assert report.geometric_mean([2.0, 8.0, None]) == pytest.approx(4.0)
    assert report.geometric_mean([None]) is None
