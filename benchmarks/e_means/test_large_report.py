"""CPU-only evidence validation; no MLX imports or dataset loading."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import large_report as report  # noqa: E402

PROTOCOL_PATH = HERE / "large_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
PROTOCOL_SHA = report.common.sha256(PROTOCOL_PATH)
HASH = "a" * 64
SHARED = json.loads((HERE / "reference/large-comparison-source.json").read_text())


def measurement(dataset=0, mode="metal-fp32"):
    entry = copy.deepcopy(PROTOCOL["configurations"][dataset])
    sources = copy.deepcopy(SHARED["source_sha256"])
    reference = copy.deepcopy(SHARED["reference_source_sha256"])
    recipe = {**entry["recipe_expected"], "batch_law": "quality", "nepochs": entry["epochs"], "seed": 10}
    precision = "bfloat16" if mode == "metal-bf16" else "float32"
    if mode != "faiss-fp32":
        recipe.update(precision=precision, finalization="last_epoch", assignment="auto")
    config = {"dataset": entry["dataset"], "mode": mode, "seed": 10, "recipe": "faiss", "batch_law": "quality",
              "finalization": "last_epoch", "capture_assignments": True, "warmup": True,
              "threads": 8, "train_score_rows": 10000, "memory": True, "memory_interval_seconds": .01,
              "rows": None, "k_override": None, "epochs_override": None, "diagnostic_subset": False,
              "precision": precision, "epochs": entry["epochs"], "protocol_config": entry,
              "expected_source_sha256": copy.deepcopy(sources), "expected_reference_source_sha256": copy.deepcopy(reference),
              "expected_versions": {"numpy": "fixture"}, "expected_host": {"chip": "fixture"}}
    memory = {"complete": True, "sampling_errors": [], "peak_physical_footprint_bytes": 4000,
              "peak_rss_bytes": 5000, "sample_interval_seconds": .01, "maximum_sample_gap_seconds": .011,
              "duration_seconds": 3.1, "sample_count": 300, "physical_footprint_peak_increase_bytes": 1000,
              "rss_peak_increase_bytes": 1500, "baseline": {"physical_footprint_bytes": 3000, "rss_bytes": 3500},
              "end": {"physical_footprint_bytes": 3500, "rss_bytes": 4000,
                      "lifetime_peak_physical_footprint_bytes": 12000}}
    row = {"status": "completed", "kind": report.KIND, "config": config, "dataset": entry["dataset_identity"],
           **{key: entry[key] for key in ("n", "d", "k")},
           "train_seconds": 3.0, "train_seconds_samples": [3.0], "test_mse": .25, "train_sample_mse": .2,
           "quality_fit_index": 0, "train_score_rows": 10000, "train_score_sample_sha256": HASH,
           "initial_centroids_sha256": HASH, "final_centroids_sha256": HASH,
           "source_sha256": sources, "reference_source_sha256": reference, "versions": {"numpy": "fixture"},
           "hardware": {"chip": "fixture"}, "faiss_reference_revision": report.REVISION,
           "faiss_compile_options": "fixture", "effective_omp_threads": 8,
           "thread_environment": {"OMP_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": None},
           "timing_scope": "fixture", "scoring": "fixture", "memory_scope": "fixture", "memory_policy": "fixture",
           "backend_timing_notes": "fixture", "reference_backend": "fixture", "numerical_differences": ["fixture"],
           "full_host_input_released": mode != "faiss-fp32", "metal_device": None if mode == "faiss-fp32" else {"name": "fixture"},
           "recipe_resolved": recipe, "algorithm_comparison": {key: recipe[key] for key in report.RECIPE_FIELDS},
           "iteration_history": [{"rows": rows, "steps": sum(recipe["epoch_steps"][:i + 1])}
                                 for i, rows in enumerate(recipe["epoch_rows"])],
           "captured_assignments": {"shape": [entry["n"]], "dtype": "int64",
                                    "semantics": "online final-pass labels; no extra reassignment"},
           "n_steps": recipe["total_steps"], "memory_samples": [memory],
           "mlx_memory_samples": [None if mode == "faiss-fp32" else
                                  {"peak_active_bytes": 2000, "active_after_fit_bytes": 1000, "cached_after_fit_bytes": 500}],
           "preparation_process_memory": {"lifetime_peak_physical_footprint_bytes": 9000}}
    return row


def campaign():
    return [measurement(dataset, mode) for dataset in range(5) for mode in report.MODES]


def test_complete_campaign_and_no_bf16_cpu_ratio():
    rows = campaign()
    groups, validation = report.validate(rows, PROTOCOL, PROTOCOL_SHA)
    assert validation["complete"]
    assert validation["accepted_records"] == 15
    summary = report.summarize(groups)
    assert len(summary) == 5
    assert all(row["cpu_over_metal_fp32"] == row["metal_fp32_over_metal_bf16"] == 1 for row in summary)
    assert not any("cpu" in field and "bf16" in field for field in report.CSV_FIELDS)


@pytest.mark.parametrize("mode", report.MODES)
def test_individual_records_validate(mode):
    assert report.row_problem(measurement(mode=mode), PROTOCOL, PROTOCOL_SHA) is None


@pytest.mark.parametrize("field,value", [("precision", "float16"), ("epochs", 20), ("threads", 4),
                                         ("batch_law", "throughput"), ("warmup", False), ("seed", 11),
                                         ("rows", 1000), ("diagnostic_subset", True), ("capture_assignments", False)])
def test_rejects_protocol_changes(field, value):
    row = measurement()
    row["config"][field] = value
    assert report.row_problem(row, PROTOCOL, PROTOCOL_SHA)


@pytest.mark.parametrize("change", ["dataset", "init", "sample", "recipe", "config"])
def test_mismatched_modes_are_not_paired(change):
    rows = [measurement(mode=mode) for mode in report.MODES]
    if change == "dataset":
        rows[1]["dataset"] = {**rows[1]["dataset"], "rescaling": "different"}
    elif change == "init":
        rows[1]["initial_centroids_sha256"] = "b" * 64
    elif change == "sample":
        rows[1]["train_score_sample_sha256"] = "b" * 64
    elif change == "recipe":
        rows[1]["recipe_resolved"]["alpha0"] *= 2
    else:
        rows[1]["config"]["unrecorded_policy"] = "different"
    groups, validation = report.validate(rows, PROTOCOL, PROTOCOL_SHA)
    assert not validation["complete"]
    assert not report.summarize(groups)
    assert validation["rejected"]


def test_duplicate_never_selects_latest():
    rows = campaign()
    rows.append(copy.deepcopy(rows[0]))
    groups, validation = report.validate(rows, PROTOCOL, PROTOCOL_SHA)
    assert not validation["complete"]
    assert len([r for r in validation["rejected"] if "duplicate" in r["reason"]]) == 2
    assert len(report.summarize(groups)) == 4


@pytest.mark.parametrize("field,expected", [("versions", "expected_versions"), ("hardware", "expected_host"),
                                             ("source_sha256", "expected_source_sha256")])
def test_self_consistent_but_mixed_campaign_rejected(field, expected):
    rows = campaign()
    changed = dict(rows[-1][field])
    changed[next(iter(changed))] = "b" * 64
    rows[-1][field] = rows[-1]["config"][expected] = changed
    groups, validation = report.validate(rows, PROTOCOL, PROTOCOL_SHA)
    assert not groups
    assert not validation["complete"]
    assert any("mixed" in r["reason"] for r in validation["rejected"])


def test_recipe_numeric_tolerance():
    row = measurement(mode="faiss-fp32")
    row["recipe_resolved"]["alpha0"] *= 1 + 1e-14
    row["algorithm_comparison"]["alpha0"] = row["recipe_resolved"]["alpha0"]
    assert report.row_problem(row, PROTOCOL, PROTOCOL_SHA) is None
    assert report.pair_problem(measurement(), row) is None


def test_adaptive_calls_and_median():
    row = measurement()
    row["train_seconds_samples"] = [1.5, 1.8, 1.2]
    row["train_seconds"] = 1.5
    row["memory_samples"] *= 3
    row["mlx_memory_samples"] *= 3
    assert report.row_problem(row, PROTOCOL, PROTOCOL_SHA) is None
    row["train_seconds"] = 1.8
    assert "median" in report.row_problem(row, PROTOCOL, PROTOCOL_SHA)
    row["train_seconds_samples"] = [1.5]
    assert "adaptive" in report.row_problem(row, PROTOCOL, PROTOCOL_SHA)


@pytest.mark.parametrize("mutation", ["incomplete", "count", "interval", "increase", "allocator", "error"])
def test_memory_integrity(mutation):
    row = measurement()
    sample = row["memory_samples"][0]
    if mutation == "incomplete":
        sample["complete"] = False
    elif mutation == "count":
        row["memory_samples"].append(copy.deepcopy(sample))
    elif mutation == "interval":
        sample["sample_interval_seconds"] = .1
    elif mutation == "increase":
        sample["physical_footprint_peak_increase_bytes"] = 999
    elif mutation == "allocator":
        row["mlx_memory_samples"][0]["peak_active_bytes"] = 0
    else:
        sample["sampling_errors"] = ["missing process"]
    assert report.row_problem(row, PROTOCOL, PROTOCOL_SHA)


def test_memory_maximum_and_lifetime_are_separate():
    group = {mode: measurement(mode=mode) for mode in report.MODES}
    metal = group["metal-fp32"]
    metal["memory_samples"].append(copy.deepcopy(metal["memory_samples"][0]))
    metal["memory_samples"][1]["peak_physical_footprint_bytes"] = 6000
    metal["memory_samples"][1]["end"]["lifetime_peak_physical_footprint_bytes"] = 15000
    summary = report.summarize([group])[0]
    assert summary["metal_fp32_peak_physical_footprint_bytes"] == 6000
    assert summary["metal_fp32_lifetime_peak_footprint_through_fit_bytes"] == 15000
    assert summary["metal_fp32_lifetime_peak_footprint_through_preparation_bytes"] == 9000


def test_zero_reference_mse_has_no_invented_percentage():
    assert report.change_pct(1, 0) is None
    assert report.change_pct(0, 0) is None


def test_strict_generation_preserves_existing_publication(tmp_path):
    output = tmp_path / "report.md"
    output.write_text("existing final report")
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("existing final csv")
    with pytest.raises(ValueError, match="Incomplete"):
        report.generate(tmp_path, output, PROTOCOL_PATH, require_complete=True)
    assert output.read_text() == "existing final report"
    assert csv_path.read_text() == "existing final csv"
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["accepted_records"] == 0
    assert len(validation["missing"]) == 15


def test_complete_generation_writes_raw_hashes_and_csv(tmp_path):
    for index, row in enumerate(campaign()):
        (tmp_path / f"raw-{index}.json").write_text(json.dumps(row))
    output = tmp_path / "report.md"
    validation = report.generate(tmp_path, output, PROTOCOL_PATH, require_complete=True)
    assert validation["complete"]
    assert len(validation["raw_evidence"]) == 15
    assert all(report.digest(item["sha256"]) for item in validation["raw_evidence"])
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert set(rows[0]) == set(report.CSV_FIELDS)
    prose = output.read_text()
    assert "single-seed" in prose and "No H100 timing" in prose
    assert "76.3 GiB before host release, plus other allocations" in prose and "not include scoring" in prose
    assert "observed process lifetime footprint through preparation" in prose
    assert validation["frozen_shared_sources"]["verified_members"] == len(SHARED["source_sha256"])
    assert validation["frozen_shared_sources"]["matches_all_records"]
    assert "index-boundary probe" in prose


def test_import_does_not_load_numerical_runtimes():
    code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import large_report; "
            "assert not any(m == 'numpy' or m == 'mlx' or m.startswith('mlx.') for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)


def make_archive(folder, members=None, sources=None):
    members = members if members is not None else [("source.py", b"original\n")]
    archive = folder / "snapshot.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, contents in members:
            stream.writestr(name, contents)
    metadata = {"archive_sha256": report.common.sha256(archive),
                "source_sha256": sources if sources is not None else
                    {name: hashlib.sha256(contents).hexdigest() for name, contents in members}}
    path = archive.with_suffix(".json")
    path.write_text(json.dumps(metadata))
    return path


def test_source_archive_verifies_digest_and_all_member_bytes(tmp_path):
    path = make_archive(tmp_path)
    result = report.verify_source_archive(path)
    assert result["verified_members"] == 1
    assert result["archive_sha256"] == report.common.sha256(path.with_suffix(".zip"))
    assert result["metadata_sha256"] == report.common.sha256(path)


def test_source_archive_requires_published_digest(tmp_path):
    path = make_archive(tmp_path)
    with pytest.raises(ValueError, match="published snapshot"):
        report.verify_source_archive(path, HASH)


def test_source_archive_rejects_wrong_whole_archive_digest(tmp_path):
    path = make_archive(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["archive_sha256"] = HASH
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="recorded digest"):
        report.verify_source_archive(path)


def test_source_archive_rejects_changed_member_despite_updated_zip_digest(tmp_path):
    path = make_archive(tmp_path, sources={"source.py": HASH})
    with pytest.raises(ValueError, match="member bytes"):
        report.verify_source_archive(path)


@pytest.mark.parametrize("sources", [{"other.py": HASH}, {"source.py": HASH, "extra.py": HASH}])
def test_source_archive_rejects_missing_or_unrecorded_members(tmp_path, sources):
    path = make_archive(tmp_path, sources=sources)
    with pytest.raises(ValueError, match="member names"):
        report.verify_source_archive(path)


def test_source_archive_rejects_duplicate_members(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        path = make_archive(tmp_path, members=[("source.py", b"first"), ("source.py", b"second")])
    with pytest.raises(ValueError, match="duplicate member"):
        report.verify_source_archive(path)


@pytest.mark.parametrize("field", ["source_sha256", "reference_source_sha256"])
def test_generation_rejects_self_consistent_source_maps_unmatched_to_archive(tmp_path, field):
    row = measurement()
    key = "benchmarks/e_means/large_compare.py" if field == "source_sha256" else "contrib/e_means/base.py"
    row[field][key] = row["config"]["expected_" + field][key] = HASH
    (tmp_path / "raw.json").write_text(json.dumps(row))
    output = tmp_path / "report.md"
    with pytest.raises(ValueError, match="Incomplete"):
        report.generate(tmp_path, output, PROTOCOL_PATH, require_complete=True)
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["accepted_records"] == 0
    assert not validation["frozen_shared_sources"]["matches_all_records"]
    assert "source map" in validation["rejected"][0]["reason"]
    assert not output.exists()
