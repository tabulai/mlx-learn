"""CPU-only evidence, pairing and archive checks for the four-method report."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks/e_means"
sys.path.insert(0, str(BENCH))
import kmeans_report as report  # noqa: E402


def campaign():
    protocol = json.loads((BENCH / "kmeans_protocol.json").read_text())
    protocol_hash = report.common.sha256(BENCH / "kmeans_protocol.json")
    sources = {"benchmarks/e_means/kmeans_protocol.json": protocol_hash, "driver": "f" * 64}
    rows = []
    for entry in protocol["configurations"]:
        n, d, k, epochs = (entry[key] for key in ("n", "d", "k", "epochs"))
        for mode in report.MODES:
            seconds = {"metal-kmeans": 10., "sklearn-kmeans": 100., "metal-emeans": 8., "faiss-emeans": 80.}[mode]
            config = {"dataset": entry["dataset"], "mode": mode, "seed": 10, "epochs": epochs,
                      "protocol_config": entry, "precision": "float32", "n_init": 1, "tol": 0.,
                      "warmup": True, "final_labels": True, "threads": 8, "train_score_rows": 10000,
                      "memory": True, "memory_interval_seconds": .01, "rows": None, "k_override": None,
                      "epochs_override": None, "diagnostic_subset": False, "expected_source_sha256": sources,
                      "expected_reference_source_sha256": {"ref": "e" * 64},
                      "expected_sklearn_sha256": {"sklearn": "d" * 64},
                      "expected_versions": {"python": "test"}, "expected_host": {"cpu": "test"},
                      "output": f"/{entry['dataset']}-{mode}.json", "worker_output": "/worker.tmp"}
            memory = {"complete": True, "sampling_errors": [], "sample_interval_seconds": .01,
                      "maximum_sample_gap_seconds": .02, "duration_seconds": seconds + .01, "sample_count": 3,
                      "peak_physical_footprint_bytes": 120, "peak_rss_bytes": 100,
                      "physical_footprint_peak_increase_bytes": 20, "rss_peak_increase_bytes": 10,
                      "baseline": {"physical_footprint_bytes": 100, "rss_bytes": 90},
                      "end": {"physical_footprint_bytes": 110, "rss_bytes": 95,
                              "lifetime_peak_physical_footprint_bytes": 200}}
            metal = mode.startswith("metal-")
            recipe = {"nepochs": epochs, "batch_law": "quality", "batch_size": 65536,
                      "alpha0": .0705061840172112 if epochs == 2 else .06754829074955014,
                      "epoch_rows": [n] * epochs, "epoch_steps": [153] * epochs, "total_steps": 153 * epochs,
                      "alpha_min_factor": .1, "init_usage": 2**-16, "split_epsilon": 1e-4,
                      "unused_threshold": .1, "check_unused_every": 5, "tail_policy": "keep", "seed": 10,
                      "precision": "float32", "finalization": "last_epoch", "assignment": "auto"}
            row = {"status": "completed", "kind": report.benchmark.KIND, "config": config,
                   "dataset": entry["dataset_identity"], "n": n, "d": d, "k": k,
                   "train_seconds": seconds, "train_seconds_samples": [seconds], "warmup_seconds": seconds,
                   "quality_fit_index": 0, "test_mse": 1., "train_sample_mse": .8, "final_inertia": .8 * n,
                   "label_sample_mse": .8, "label_sample_agreement": 1., "actual_iterations": epochs,
                   "final_labels": {"shape": [n], "dtype": "int64" if metal else "int32",
                                    "semantics": "nearest final centroids; no update after final assignment"},
                   "initial_centroids_sha256": "a" * 64, "final_centroids_sha256": "b" * 64,
                   "final_labels_sha256": "c" * 64, "train_score_sample_sha256": "d" * 64,
                   "train_score_rows": 10000, "full_host_input_released": metal,
                   "effective_omp_threads": 8, "sklearn_effective_threads": 8 if mode == "sklearn-kmeans" else None,
                   "thread_environment": {"OMP_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": None},
                   "memory_samples": [memory], "mlx_memory_samples": [{"peak_active_bytes": 80, "active_after_fit_bytes": 70,
                                                                         "cached_after_fit_bytes": 20} if metal else None],
                   "preparation_process_memory": {"lifetime_peak_physical_footprint_bytes": 190},
                   "metal_device": {"name": "test GPU"} if metal else None,
                   "recipe_resolved": recipe if mode.endswith("emeans") else None,
                   "iteration_history": [{"epoch": i} for i in range(epochs)] if mode.endswith("emeans") else [],
                   "timing_scope": "test timing", "memory_scope": "test memory", "numerical_notes": "test differences"}
            for key, config_key in (("source_sha256", "expected_source_sha256"), ("reference_source_sha256", "expected_reference_source_sha256"),
                                    ("sklearn_source_sha256", "expected_sklearn_sha256"), ("versions", "expected_versions"), ("hardware", "expected_host")):
                row[key] = config[config_key]
            rows.append(copy.deepcopy(row))
    return rows, protocol, protocol_hash


def test_complete_twenty_records_and_ratios():
    rows, protocol, digest = campaign()
    groups, validation = report.validate(rows, protocol, digest)
    assert validation["complete"] and validation["accepted_records"] == 20
    summary, ratios = report.summarize(groups)
    assert len(summary) == 20 and len(ratios) == 5
    assert ratios[0]["sklearn_over_metal_kmeans"] == 10
    assert ratios[0]["metal_kmeans_over_metal_emeans"] == 1.25
    assert summary[0]["preparation_lifetime_peak_footprint_bytes"] == 190
    assert summary[0]["peak_physical_footprint_bytes"] == 120


def test_missing_measurement_stays_pending():
    rows, protocol, digest = campaign()
    _, validation = report.validate(rows[:-1], protocol, digest)
    assert not validation["complete"]
    assert validation["accepted_records"] == 19
    assert len(validation["missing"]) == 1 and not validation["rejected"]


def test_markdown_names_passes_instead_of_minibatch_updates(tmp_path):
    rows, protocol, digest = campaign()
    groups, validation = report.validate(rows, protocol, digest)
    summary, comparisons = report.summarize(groups)
    for row in summary:
        row["evidence_path"] = str(tmp_path / "measurement.json")
    text = report.markdown(summary, comparisons, validation, tmp_path / "report.md", tmp_path, BENCH / "kmeans_protocol.json")
    assert "Budget / actual passes" in text
    assert "Budget / actual updates" not in text


def test_headline_uses_observed_ranges_and_small_percentages():
    rows = [{"sklearn_over_metal_kmeans": speed, "metal_kmeans_over_metal_emeans": ratio,
             "metal_kmeans_vs_sklearn_mse_change_pct": delta,
             "metal_emeans_vs_metal_kmeans_mse_change_pct": -2.0 - index}
            for index, (speed, ratio, delta) in enumerate(zip(
                [5, 7, 9, 11, 13], [.9, 1., 1.1, 1.2, 1.3], [-.0001, 0., .0002, .0003, .000490065], strict=True))]
    text = report.headline(rows)
    assert "5–13×" in text and "0.9–1.3×" in text
    assert "-0.0001% to +0.00049007%" in text
    assert "-6% to -2%" in text


def test_partial_report_omits_completed_campaign_headline(tmp_path):
    rows, protocol, digest = campaign()
    groups, validation = report.validate(rows[:-1], protocol, digest)
    summary, comparisons = report.summarize(groups)
    for row in summary:
        row["evidence_path"] = str(tmp_path / "measurement.json")
    text = report.markdown(summary, comparisons, validation, tmp_path / "report.md", tmp_path, BENCH / "kmeans_protocol.json")
    assert "Across the five datasets, Metal KMeans ran" not in text


def test_optional_figure_links_use_report_relative_paths(tmp_path):
    figures = tmp_path / "benchmarks/e_means/kmeans_figures"
    figures.mkdir(parents=True)
    output = tmp_path / "docs/report.md"
    assert report.figure_markdown(output, figures) == []
    (figures / "comparison.png").touch()
    (figures / "comparison.pdf").touch()
    text = "\n".join(report.figure_markdown(output, figures))
    assert "![Runtime, fit memory and test MSE for all four methods](../benchmarks/e_means/kmeans_figures/comparison.png)" in text
    assert "[Download the comparison figure as PDF](../benchmarks/e_means/kmeans_figures/comparison.pdf)" in text


def test_small_mse_difference_remains_visible_in_table(tmp_path):
    rows, protocol, digest = campaign()
    rows[0]["test_mse"] = 1.00000490065
    groups, validation = report.validate(rows, protocol, digest)
    summary, comparisons = report.summarize(groups)
    for row in summary:
        row["evidence_path"] = str(tmp_path / "measurement.json")
    text = report.markdown(summary, comparisons, validation, tmp_path / "report.md", tmp_path, BENCH / "kmeans_protocol.json")
    assert "+0.00049007%" in text


def test_duplicate_records_all_rejected():
    rows, protocol, digest = campaign()
    _, validation = report.validate(rows + [copy.deepcopy(rows[0])], protocol, digest)
    assert validation["accepted_records"] == 19
    assert len(validation["rejected"]) == 2


@pytest.mark.parametrize("change", ["initial", "score_sample", "common_config", "emeans_recipe"])
def test_incompatible_dataset_group_excluded(change):
    rows, protocol, digest = campaign()
    if change == "initial":
        rows[0]["initial_centroids_sha256"] = "9" * 64
    elif change == "score_sample":
        rows[0]["train_score_sample_sha256"] = "9" * 64
    elif change == "common_config":
        rows[0]["config"]["cache"] = "/different"
    else:
        rows[2]["recipe_resolved"]["alpha0"] *= 2
    _, validation = report.validate(rows, protocol, digest)
    assert validation["accepted_records"] == 16
    assert len(validation["rejected"]) == 4


def test_mixed_sources_reject_entire_campaign():
    rows, protocol, digest = campaign()
    rows[0]["source_sha256"]["driver"] = "9" * 64
    rows[0]["config"]["expected_source_sha256"] = rows[0]["source_sha256"]
    _, validation = report.validate(rows, protocol, digest)
    assert validation["accepted_records"] == 0


@pytest.mark.parametrize("change", ["online_labels", "memory_error", "archive_mismatch", "protocol_hash", "identity", "precision"])
def test_bad_evidence_is_rejected(change):
    rows, protocol, digest = campaign()
    archive = None
    if change == "online_labels":
        rows[0]["final_labels"]["semantics"] = "online"
    elif change == "memory_error":
        rows[0]["memory_samples"][0]["sampling_errors"] = ["lost observation"]
    elif change == "archive_mismatch":
        archive = {"source_sha256": {"different": "0" * 64}}
    elif change == "protocol_hash":
        digest = "0" * 64
    elif change == "identity":
        rows[0]["dataset"] = {**rows[0]["dataset"], "test_sha256": "0" * 64}
    else:
        rows[0]["config"]["precision"] = "bfloat16"
    _, validation = report.validate(rows, protocol, digest, archive)
    assert not validation["complete"] and validation["rejected"]


def test_archive_digest_and_source_members_are_checked(tmp_path):
    archive = tmp_path / "snapshot.zip"
    source = b"frozen source"
    with zipfile.ZipFile(archive, "w") as contents:
        contents.writestr("source.py", source)
    metadata = {"archive_sha256": report.common.sha256(archive),
                "source_sha256": {"source.py": hashlib.sha256(source).hexdigest()}}
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(metadata))
    assert report.archive_evidence(path)[1]["verified"]
    metadata["source_sha256"]["source.py"] = "0" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="member differs"):
        report.archive_evidence(path)
    metadata["archive_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest mismatch"):
        report.archive_evidence(path)


def test_incomplete_refuses_final_report(tmp_path):
    source = b"source"
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as contents:
        contents.writestr("source.py", source)
    metadata = tmp_path / "archive.json"
    metadata.write_text(json.dumps({"archive_sha256": report.common.sha256(archive),
                                    "source_sha256": {"source.py": hashlib.sha256(source).hexdigest()}}))
    output = tmp_path / "final.md"
    with pytest.raises(ValueError, match="Incomplete"):
        report.generate(tmp_path / "results", output, BENCH / "kmeans_protocol.json", metadata, require_complete=True)
    assert not output.exists()
    assert (tmp_path / "results/validation.json").exists()
