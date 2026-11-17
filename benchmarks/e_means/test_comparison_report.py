"""Pure-CPU comparison-report checks; no dataset or MLX imports."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("comparison_report", Path(__file__).with_name("comparison_report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def row(method="metal", trial=0, seed=10):
    sources = dict.fromkeys(report.REQUIRED_SOURCES, "source")
    versions, host = {"python": "3.14.2"}, {"cpu": "Apple M4 Max", "ram_bytes": 128 * 2**30}
    return {
        "status": "completed", "kind": "faiss_direct_comparison", "n": 1_000_000, "d": 96, "k": 16384,
        "config": {"dataset": "deep1b-1M", "method": method, "epochs": 3, "seed": seed, "trial": trial,
                   "threads": 8, "veclib_threads": None, "rows": None, "k": None,
                   "precision": "float32", "recipe": "faiss", "finalization": "last_epoch",
                   "assignment": "auto", "warmup": False, "train_score_rows": 10000,
                   "memory_interval_seconds": .01, "expected_source_sha256": sources.copy(),
                   "expected_versions": versions.copy(), "expected_host": host.copy()},
        "train_seconds": 2 if method == "metal" else 10,
        "test_mse": .99 if method == "metal" else 1.,
        "initial_centroids_sha256": f"initial-{seed}", "source_sha256": sources,
        "versions": versions, "hardware": host,
        "dataset": {"train_sample_sha256": "train", "test_sha256": "test"},
        "faiss_reference_revision": "faiss-revision", "faiss_compile_options": "OPTIMIZE GENERIC",
        "effective_omp_threads": 8, "thread_environment": {"OMP_NUM_THREADS": "8", "VECLIB_MAXIMUM_THREADS": None},
        "timing_scope": "first fit", "memory_scope": "fit window", "scoring": "shared",
        "recipe_resolved": {
            "name": "faiss" if method == "metal" else "canonical_v1", "backend": method,
            "batch_law": "throughput", "batch_size": 65536, "alpha0": .3608988199706412,
            "nepochs": 3, "epoch_rows": [1_000_000] * 3, "epoch_steps": [16] * 3,
            "total_steps": 48, "alpha_min_factor": .1, "init_usage": 2**-16,
            "split_epsilon": 1e-4, "unused_threshold": .1, "check_unused_every": 5,
            "tail_policy": "keep", "seed": seed,
        },
        "memory": {
            "peak_physical_footprint_bytes": 3 * 2**30, "peak_rss_bytes": 4 * 2**30,
            "physical_footprint_peak_increase_bytes": 2 * 2**30, "rss_peak_increase_bytes": 2 * 2**30,
            "baseline": {"physical_footprint_bytes": 2**30, "rss_bytes": 2 * 2**30},
            "end": {"physical_footprint_bytes": 2**30, "rss_bytes": 2 * 2**30},
            "sample_interval_seconds": .01, "sample_count": 200, "duration_seconds": 2.01,
            "maximum_sample_gap_seconds": .011, "complete": True, "sampling_errors": [],
            "lifetime_peak_physical_footprint_bytes": 100 * 2**30,
        },
        "mlx_memory": {"peak_active_bytes": 2**30, "cached_after_fit_bytes": 2 * 2**30} if method == "metal" else None,
        "_path": Path(f"/repo/results/faiss_comparison/{method}-{seed}-{trial}.json"),
    }


def test_measured_ratio_and_quality():
    pairs, rejected = report.matched_pairs([row(), row("faiss-emeans")])
    assert not rejected
    assert pairs[0]["speedup"] == 5
    assert pairs[0]["mse_change_pct"] == pytest.approx(-1)


@pytest.mark.parametrize("field", report.RECIPE_FIELDS)
def test_resolved_recipe_mismatch_is_rejected(field):
    a, b = row(), row("faiss-emeans")
    value = b["recipe_resolved"][field]
    b["recipe_resolved"][field] = value + "x" if isinstance(value, str) else [999] if isinstance(value, list) else value + 1
    pairs, rejected = report.matched_pairs([a, b])
    assert not pairs
    assert rejected


def test_recipe_rounding_tolerance_and_implementation_metadata():
    a, b = row(), row("faiss-emeans")
    b["recipe_resolved"]["alpha0"] += 1e-15
    assert report.matched_pairs([a, b])[0]
    b["recipe_resolved"]["alpha0"] += 1e-8
    assert not report.matched_pairs([a, b])[0]


@pytest.mark.parametrize("field", report.PAIR_FIELDS)
def test_provenance_mismatch_is_rejected(field):
    a, b = row(), row("faiss-emeans")
    b[field] = "different"
    pairs, rejected = report.matched_pairs([a, b])
    assert not pairs
    assert rejected


@pytest.mark.parametrize("field,value", [("warmup", True), ("recipe", "paper"), ("epochs", 2),
                                         ("precision", "float16"), ("rows", 1000), ("k", 128),
                                         ("finalization", "ema")])
def test_primary_protocol_cannot_be_replaced(field, value):
    a, b = row(), row("faiss-emeans")
    a["config"][field] = b["config"][field] = value
    assert not report.matched_pairs([a, b])[0]


def test_ambiguous_duplicate_is_not_cherry_picked():
    a, b = row(), row("faiss-emeans")
    second = copy.deepcopy(b)
    second["train_seconds"] = 1
    pairs, rejected = report.matched_pairs([a, b, second])
    assert not pairs
    assert rejected[0]["reason"] == "ambiguous duplicate method records"


def test_cold_trials_are_distinct_from_seeds_and_pairs_not_reused():
    a, b, c, d = row(), row("faiss-emeans"), row(trial=1), row("faiss-emeans", trial=1)
    c["train_seconds"], d["train_seconds"] = 4, 12
    pairs, rejected = report.matched_pairs([a, b, c, d])
    group = report.grouped_pairs(pairs)[0]
    assert not rejected
    assert group["paired_trials"] == 2
    assert group["seeds"] == [10]
    assert group["metal_seconds"] == 3
    assert group["faiss_seconds"] == 11
    assert group["speedup"] == 4
    assert len(report.matched_pairs([a, b, c])[0]) == 1


def test_memory_uses_fit_peak_not_lifetime_or_sum():
    pairs, _ = report.matched_pairs([row(), row("faiss-emeans")])
    group = report.grouped_pairs(pairs)[0]
    assert group["metal_peak_physical_footprint_bytes"] == 3 * 2**30
    table = "\n".join(report.memory_table(pairs, Path("/repo/docs/report.md")))
    assert "3.000 | 2.000 | 4.000 | 1.000" in table
    assert "100.000" not in table


@pytest.mark.parametrize("change", ["incomplete", "error", "wrong_delta", "below_baseline", "no_samples"])
def test_bad_memory_suppresses_memory_but_not_runtime(change):
    a, b = row(), row("faiss-emeans")
    if change == "incomplete":
        a["memory"]["complete"] = False
    elif change == "error":
        a["memory"]["sampling_errors"] = ["denied"]
    elif change == "wrong_delta":
        a["memory"]["physical_footprint_peak_increase_bytes"] += 1
    elif change == "below_baseline":
        a["memory"]["peak_rss_bytes"] = 1
    else:
        a["memory"]["sample_count"] = 0
    pairs, rejected = report.matched_pairs([a, b])
    assert len(pairs) == 1 and not rejected
    group = report.grouped_pairs(pairs)[0]
    assert group["metal_peak_physical_footprint_bytes"] is None
    assert group["faiss_peak_physical_footprint_bytes"] == 3 * 2**30
    assert group["speedup"] == 5


def test_history_is_ignored_and_new_failure_visible(tmp_path):
    for name, item in (("old", {"status": "completed", "config": {"method": "faiss-emeans"}}),
                       ("new", {"kind": "faiss_direct_comparison", "status": "failed"})):
        (tmp_path / f"{name}.json").write_text(json.dumps(item))
    rows = report.evidence_rows(tmp_path)
    assert len(rows) == 1
    assert report.matched_pairs(rows)[1][0]["reason"] == "status=failed"


def test_deferred_ten_million_is_not_failed_or_in_headline():
    coverage = "\n".join(report.coverage_table([], [], []))
    assert coverage.count("deferred by requested scope; not run") == 5
    assert coverage.count("| primary | pending |") == 17
    text = report.build_report([], [], [], [], Path("/repo/docs/report.md"), Path("/repo/summary.csv"), Path("/repo/manifest.json"), [])
    assert "0/17" in text
    assert "three-epoch Metal GPU" in text
    assert "first fit in a fresh process" in text
    assert "not independent initialization seeds" in text


def test_zero_reference_mse_no_infinite_ratio():
    a, b = row(), row("faiss-emeans")
    b["test_mse"] = 0
    pairs, _ = report.matched_pairs([a, b])
    assert pairs[0]["mse_change_pct"] is None
    assert report.grouped_pairs(pairs)[0]["mse_change_pct"] is None


def test_lead_counts_and_memory_ratio_are_derived_from_pairs():
    a, b = row(), row("faiss-emeans")
    a["memory"]["peak_physical_footprint_bytes"] = 6 * 2**30
    a["memory"]["physical_footprint_peak_increase_bytes"] = 5 * 2**30
    pairs, rejected = report.matched_pairs([a, b])
    groups = report.grouped_pairs(pairs)
    text = report.build_report([a, b], pairs, groups, rejected, Path("/repo/docs/report.md"), Path("/repo/summary.csv"), Path("/repo/manifest.json"), [])
    assert "lower on **1**, equal on **0**, and higher on **0**" in text
    assert "ratio ranges from **2.00× to 2.00×**" in text
    assert "every five-batch check" in text
    assert "cache after fit" in text


def test_historical_new_kind_missing_provenance_is_not_admitted():
    a, b = row(), row("faiss-emeans")
    del a["source_sha256"]["benchmarks/e_means/memory.py"]
    pairs, rejected = report.matched_pairs([a, b])
    assert not pairs
    assert "incomplete comparison source hashes" in [r["reason"] for r in rejected]


def test_figure_context_uses_recorded_hardware_and_thread_count():
    pairs, _ = report.matched_pairs([row(), row("faiss-emeans")])
    assert report.figure_context(report.grouped_pairs(pairs)) == "Apple M4 Max · Faiss CPU: 8 OpenMP threads"


def test_supplement_absent_produces_no_appendix():
    assert report.memory_variability([], []) is None
    assert report.variability_table(None, Path("/repo/report.md")) == []


def test_additional_trials_validated_but_never_pooled_in_primary():
    pairs, rejected = report.matched_pairs([row(), row("faiss-emeans")])
    extras = [row(trial=i) for i in range(3)]
    for r in extras:
        r["_path"] = Path("/repo/results/faiss_comparison_memory_variability") / r["_path"].name
    extras[1]["memory"]["peak_physical_footprint_bytes"] += 2**30
    extras[1]["memory"]["physical_footprint_peak_increase_bytes"] += 2**30
    extras[1]["mlx_memory"]["cached_after_fit_bytes"] += 2**30
    supplement = report.memory_variability(extras, pairs)
    assert not supplement["rejected"]
    assert len(supplement["rows"]) == 3
    groups = report.grouped_pairs(pairs)
    text = report.build_report([row(), row("faiss-emeans")], pairs, groups, rejected, Path("/repo/report.md"), Path("/repo/summary.csv"), Path("/repo/manifest.json"), [], supplement)
    assert "1/17" in text
    assert groups[0]["paired_trials"] == 1
    assert "3 additional measured Metal-only first fits" in text
    assert "excluded from all primary paired tables" in text
    assert "4.000 | 1.000 | 3.000" in text


@pytest.mark.parametrize("change", ["source", "recipe", "seed", "method", "incomplete_memory", "missing_cache"])
def test_supplement_rejects_unmatched_or_incomplete_records(change):
    pairs, _ = report.matched_pairs([row(), row("faiss-emeans")])
    extra = row(trial=2)
    if change == "source":
        extra["source_sha256"]["benchmarks/e_means/memory.py"] = "changed"
        extra["config"]["expected_source_sha256"]["benchmarks/e_means/memory.py"] = "changed"
    elif change == "recipe":
        extra["recipe_resolved"]["split_epsilon"] = .5
    elif change == "seed":
        extra["config"]["seed"] = extra["recipe_resolved"]["seed"] = 11
    elif change == "method":
        extra["config"]["method"] = "faiss-emeans"
    elif change == "incomplete_memory":
        extra["memory"]["complete"] = False
    else:
        del extra["mlx_memory"]["cached_after_fit_bytes"]
    supplement = report.memory_variability([extra], pairs)
    assert not supplement["rows"]
    assert len(supplement["rejected"]) == 1


def test_supplement_needs_unambiguous_primary_and_unique_trial_numbers():
    assert report.memory_variability([row()], [])["rejected"]
    pairs, _ = report.matched_pairs([row(), row("faiss-emeans")])
    supplement = report.memory_variability([row(trial=1), row(trial=1)], pairs)
    assert not supplement["rows"]
    assert supplement["rejected"][0]["reason"] == "ambiguous duplicate supplemental trial"
