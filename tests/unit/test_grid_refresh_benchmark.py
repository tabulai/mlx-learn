"""Numerical and provenance checks for the generic grid; no timed fits."""

import copy
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import grid_refresh as grid


def test_grid_covers_original_cases_and_deduplicates_only_fit_query_labels():
    from benchmarks.run import PROFILES

    cases = grid.all_cases()
    originals = [c.label for c in PROFILES["full"] if c.algorithm in {"knn", "svc"}]
    assert len(cases) == 84
    assert sorted(label for c in cases for label in c["original_grid_labels"]) == sorted(originals)
    assert sum(c["algorithm"] == "knn" and c["operation"] == "fit" for c in cases) == 18
    assert sum(c["algorithm"] == "knn" and c["operation"] == "kneighbors" for c in cases) == 54
    assert sum(c["algorithm"] == "svc" for c in cases) == 12
    for case in cases:
        assert len(case["original_grid_labels"]) == (3 if case["algorithm"] == "knn" and case["operation"] == "fit" else 1)


def test_case_filters_reject_unknown_and_preserve_ids():
    selected = grid.select_cases(["svc"], ["svc/fit/32000x*"])
    assert [case["id"] for case in selected] == ["svc/fit/32000x16", "svc/fit/32000x128"]
    with pytest.raises(ValueError, match="matched nothing"):
        grid.select_cases(patterns=["nonexistent"])


def test_environment_policy_clears_route_overrides_and_explicitly_controls_threads(monkeypatch):
    monkeypatch.setenv("MLXLEARN_FORCE_BACKEND", "cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("PYTHONPATH", "/unexpected")
    single = grid.controlled_environment("1")
    default = grid.controlled_environment("default")
    assert all(single[key] == "1" for key in grid.THREAD_VARIABLES)
    assert all(key not in default for key in grid.THREAD_VARIABLES)
    assert not any(key.startswith("MLXLEARN_") for key in single)
    assert "/unexpected" not in single["PYTHONPATH"]


def test_fixture_is_float64_reproducible_and_independent_of_query_count():
    case = {"algorithm": "knn", "n_samples": 24, "n_features": 3, "n_queries": 17}
    train, _, query, _ = grid.make_data(case, 41)
    train2, _, query2, _ = grid.make_data({**case, "n_queries": 32}, 41)
    assert train.dtype == query.dtype == np.float64
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(query, query2[:17])
    assert not np.array_equal(train, grid.make_data(case, 42)[0])


def test_svc_prediction_workload_preserves_original_thousand_query_rows():
    case = {"algorithm": "svc", "operation": "predict", "n_samples": 2000, "n_features": 16}
    assert grid.make_data(case, 41)[2].shape == (1000, 16)
    assert grid.make_data({**case, "operation": "fit"}, 41)[2].shape == (257, 16)


@pytest.fixture
def neighbor_problem():
    rng = np.random.default_rng(12)
    train, query = rng.normal(size=(31, 4)), rng.normal(size=(11, 4))
    return train, query


def full_neighbors(train, query, canonical):
    if canonical:
        train, query = train.astype(np.float32).astype(float), query.astype(np.float32).astype(float)
    squared = np.sum((train[None] - query[:, None]) ** 2, axis=2)
    indices = np.argsort(squared, axis=1, kind="stable")[:, :10]
    return np.sqrt(np.take_along_axis(squared, indices, axis=1)), indices


@pytest.mark.parametrize("canonical", [False, True])
def test_chunked_reference_and_audit_match_independent_full_distance_matrix(neighbor_problem, canonical):
    train, query = neighbor_problem
    reference = grid.independent_neighbors(train, query, canonical_float32=canonical)
    value = full_neighbors(train, query, canonical)
    audit = grid.audit_neighbors(value, train, query, reference)
    assert audit["passed"]
    assert len(audit["audited_query_rows"]) == 9
    assert audit["reference_geometry"] == ("canonical_float32" if canonical else "caller_float64")


@pytest.mark.parametrize("corruption", ["duplicate", "wrong_member", "distance", "nan", "shape", "order"])
def test_neighbor_audit_rejects_incorrect_results(neighbor_problem, corruption):
    train, query = neighbor_problem
    reference = grid.independent_neighbors(train, query)
    distances, indices = full_neighbors(train, query, False)
    if corruption == "duplicate":
        indices[0, 1] = indices[0, 0]
    elif corruption == "wrong_member":
        indices[0, 0] = next(i for i in range(len(train)) if i not in indices[0])
    elif corruption == "distance":
        distances[0, 0] += 1
    elif corruption == "nan":
        distances[0, 0] = np.nan
    elif corruption == "shape":
        distances = distances[:, :-1]
    else:
        indices[0, :2] = indices[0, :2][::-1]
        distances[0, :2] = distances[0, :2][::-1]
    assert not grid.audit_neighbors((distances, indices), train, query, reference)["passed"]


def test_exact_distance_tie_alternative_remains_explicit():
    train = np.zeros((12, 2), dtype=np.float64)
    query = np.zeros((1, 2), dtype=np.float64)
    reference = grid.independent_neighbors(train, query)
    value = (np.zeros((1, 10)), np.arange(2, 12).reshape(1, 10))
    audit = grid.audit_neighbors(value, train, query, reference)
    assert audit["passed"] and audit["valid_exact_tie_alternative"]
    assert not reference["unique_rank"][0]


def test_float32_rank_changes_are_recorded_without_changing_sklearn_input():
    # Float32 rounds the final two distances equal, reversing the float64 rank.
    train = np.arange(12, dtype=np.float64).reshape(-1, 1)
    train[9, 0], train[10, 0] = 9.0 + 1e-8, 9.0
    query = np.zeros((1, 1))
    refs = {"canonical": grid.independent_neighbors(train, query, canonical_float32=True),
            "original": grid.independent_neighbors(train, query)}
    trials = {}
    for variant in ("direct_auto", "patched_auto", "sklearn_brute", "sklearn_auto"):
        canonical = variant.endswith("_auto") and not variant.startswith("sklearn")
        trials[variant] = {"audit": grid.audit_neighbors(full_neighbors(train, query, canonical), train, query,
                                                       refs["canonical" if canonical else "original"])}
    grid.compare_knn_trials(trials, refs)
    assert all(trial["audit"]["passed"] for trial in trials.values())
    assert all(trial["audit"]["precision_ambiguous_query_rows"] == [0] for trial in trials.values())
    assert train[9, 0] != np.float32(train[9, 0])


def svc_trial(prediction=(0, 1), margin=(-1.0, 1.0), passed=True):
    return {"audit": {"prediction": list(prediction), "margin": list(margin), "passed": passed}}


@pytest.mark.parametrize("order", [("direct_auto", "patched_auto", "sklearn"),
                                    ("sklearn", "patched_auto", "direct_auto")])
def test_svc_comparison_handles_sklearn_first_and_removes_raw_arrays(order):
    trials = {variant: svc_trial() for variant in order}
    assert grid.compare_svc_trials(trials)
    assert all("prediction" not in trial["audit"] and "margin" not in trial["audit"] for trial in trials.values())


@pytest.mark.parametrize("bad", [svc_trial(prediction=(1, 1)), svc_trial(margin=(-2, 1)), svc_trial(passed=False)])
def test_svc_comparison_does_not_qualify_different_predictions_or_margins(bad):
    trials = {"direct_auto": copy.deepcopy(bad), "patched_auto": svc_trial(), "sklearn": svc_trial()}
    assert not grid.compare_svc_trials(trials)


@pytest.mark.parametrize("status,gap,timed_prediction", [(1, 0.0, None), (0, 0.1, None), (0, 0.0, [1, 1])])
def test_svc_audit_checks_status_kkt_and_actual_timed_output(status, gap, timed_prediction):
    model = SimpleNamespace(predict=lambda x: np.array([0, 1]), decision_function=lambda x: np.array([-1., 1.]),
                            fit_status_=status, _pair_kkt_violation_=np.array([gap]), tol=1e-3)
    assert not grid.audit_svc(model, np.zeros((2, 3)), np.array([0, 1]), timed_prediction)["passed"]


def test_checkpoint_resumes_only_exact_identity_and_retains_failures(tmp_path):
    path = tmp_path / "result.json"
    identity = {"sources": {"solver.cpp": "hash"}, "seed": 41}
    report = grid.load_checkpoint(path, identity, False)
    report["errors"]["case"] = {"stderr": "failure"}
    grid.checkpoint(path, report)
    assert grid.load_checkpoint(path, identity, True)["errors"]["case"]["stderr"] == "failure"
    with pytest.raises(ValueError, match="already exists"):
        grid.load_checkpoint(path, identity, False)
    with pytest.raises(ValueError, match="identity differs"):
        grid.load_checkpoint(path, {**identity, "seed": 42}, True)
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("algorithm,operation", [("knn", "fit"), ("knn", "kneighbors"), ("svc", "fit"), ("svc", "predict")])
def test_small_worker_qualifies_real_models_without_collecting_timings(monkeypatch, algorithm, operation):
    # Execute the full fitting/audit plumbing, replacing the clock boundary.
    monkeypatch.setattr(grid, "timed", lambda call, mx: (1.0, call()))
    monkeypatch.setattr(grid, "source_identity", lambda: {"test": "source"})
    case = {"algorithm": algorithm, "operation": operation, "n_samples": 24,
            "n_features": 4, "n_queries": 11, "id": "tiny", "original_grid_labels": ["tiny"]}
    result = grid.worker(case, SimpleNamespace(seed=41, repeats=2, thread_policy="1"))
    assert result["quality_passed"]
    assert result["qualified_speedup"] is not None
    assert len(result["warmup_trials"]) == 1
    assert len(result["measured_trials"]) == 2
    assert all(timing["samples_s"] == [1.0, 1.0] for timing in result["timings"].values())


def test_module_import_does_not_import_numerical_libraries():
    # Inspecting executable imports avoids a timing subprocess in this unit test.
    import ast

    module = ast.parse(grid.Path(grid.__file__).read_text())
    top_level = [statement for statement in module.body if isinstance(statement, (ast.Import, ast.ImportFrom))]
    imported = {alias.name.split(".")[0] for statement in top_level if isinstance(statement, ast.Import)
                for alias in statement.names}
    imported |= {statement.module.split(".")[0] for statement in top_level if isinstance(statement, ast.ImportFrom)}
    assert not imported & {"numpy", "scipy", "sklearn", "mlx", "mlxlearn"}
    assert importlib.import_module("benchmarks.grid_refresh") is grid
