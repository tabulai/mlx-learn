"""Evidence integrity and numerical checks for the SVC acceleration benchmark."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import svc_acceleration as bench

from mlxlearn._solvers import native_smo
from mlxlearn._solvers.smo import SMOState


def test_frozen_loader_does_not_rebind_or_use_current_solver(tmp_path, monkeypatch):
    from mlxlearn._kernels import kernels
    from mlxlearn._solvers import smo
    from mlxlearn.svm import _classes

    def forbidden(*args, **kwargs):
        raise AssertionError("baseline reached current implementation")

    monkeypatch.setattr(smo, "solve_c_svc", forbidden)
    monkeypatch.setattr(kernels, "kernel_matrix", forbidden)
    original_bindings = (_classes.solve_c_svc, _classes.kernel_matrix)
    legacy, frozen = bench.load_baseline(tmp_path)
    assert (_classes.solve_c_svc, _classes.kernel_matrix) == original_bindings
    assert legacy._solve_pair.__globals__["solve_c_svc"] is frozen.solve_c_svc
    assert legacy._solve_pair.__globals__["kernel_matrix"] is not forbidden

    y, bounds, diagonal, gram = bench.solver_problem("rbf", 51, n=17, d=4)
    state = frozen.solve_c_svc(y, bounds, diagonal, gram.__getitem__, frozen.SMOConfig())
    assert state.converged
    assert bench.solver_diagnostics(state, y, bounds, gram)["independent_kkt_violation"] < 1e-3


def test_frozen_loader_rejects_archive_tampering(tmp_path, monkeypatch):
    archive = tmp_path / "changed.zip"
    archive.write_bytes(bench.REFERENCE.read_bytes() + b"unrecorded changes")
    archive.with_suffix(".manifest.json").write_text(
        bench.REFERENCE.with_suffix(".manifest.json").read_text()
    )
    monkeypatch.setattr(bench, "REFERENCE", archive)
    with pytest.raises(ValueError, match="archive hash"):
        bench.load_baseline(tmp_path)


def test_frozen_loader_rejects_source_manifest_tampering(tmp_path, monkeypatch):
    archive = tmp_path / "changed.zip"
    archive.write_bytes(bench.REFERENCE.read_bytes())
    manifest = json.loads(bench.REFERENCE.with_suffix(".manifest.json").read_text())
    manifest["files"]["src/mlxlearn/_solvers/smo.py"] = "0" * 64
    archive.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(bench, "REFERENCE", archive)
    with pytest.raises(ValueError, match="source hash"):
        bench.load_baseline(tmp_path)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_same_gram_probe_checks_numerics_without_timing(tmp_path, monkeypatch, kernel):
    if not native_smo.available():
        pytest.skip("native extension unavailable")
    _, frozen = bench.load_baseline(tmp_path)
    # Exercise real solver calls and diagnostics, with the clock removed:
    # correctness tests must never assert wall-clock performance.
    monkeypatch.setattr(bench, "timed", lambda call: (1.0, call()))
    result = bench.run_solver_probe(kernel, frozen, 2, 51, n=31, d=6)
    assert result["quality_passed"]
    assert result["alpha_exact_equal"]
    assert result["max_alpha_difference"] == 0
    assert result["objective_difference"] == 0
    for diagnostics in result["quality"].values():
        assert diagnostics["independent_kkt_violation"] < 1e-3
        assert diagnostics["equality_residual"] < 1e-12
        assert diagnostics["lower_box_violation"] == 0
        assert diagnostics["upper_box_violation"] == 0


def test_independent_diagnostics_detect_corrupt_returned_iterate():
    y = np.array([1.0, -1.0])
    bounds = np.ones(2)
    gram = np.eye(2, dtype=np.float32)
    # The reported status and residual deliberately lie. Evidence must derive
    # feasibility and KKT from alpha, not trust solver self-reporting.
    state = SMOState(np.array([1.5, -0.25]), 0.0, 1, True, 0.0)
    result = bench.solver_diagnostics(state, y, bounds, gram)
    assert result["equality_residual"] == 1.75
    assert result["lower_box_violation"] == 0.25
    assert result["upper_box_violation"] == 0.5
    assert result["dual_objective"] == pytest.approx(-0.09375)
    initial = SMOState(np.zeros(2), 0.0, 0, True, 0.0)
    assert bench.solver_diagnostics(initial, y, bounds, gram)["independent_kkt_violation"] == 2.0


def models_for_quality():
    prediction = np.arange(100) % 2
    decision = (prediction * 2 - 1).astype(np.float64)
    models = {}
    for name in ("legacy", "native", "sklearn"):
        model = SimpleNamespace(
            n_iter_=np.array([1]), n_support_=np.array([2, 2]), fit_status_=0,
            _pair_kkt_violation_=np.array([0.0003]),
            decision_function=lambda X, values=decision: values,
            predict=lambda X, values=prediction: values,
        )
        models[name] = model
    return models, np.empty((100, 1)), prediction


@pytest.mark.parametrize("variant", ["legacy", "native", "sklearn"])
def test_quality_rejects_any_nonconverged_comparator(variant):
    models, query, target = models_for_quality()
    assert bench.estimator_quality("rbf", models, query, target)[1]
    models[variant].fit_status_ = 1
    assert not bench.estimator_quality("rbf", models, query, target)[1]


def test_quality_rejects_failed_earlier_fit_and_reports_diagnostics():
    models, query, target = models_for_quality()
    history = {name: [0, 0, 0] for name in models}
    history["legacy"][0] = 1
    quality, passed = bench.estimator_quality("rbf", models, query, target, history)
    assert not passed
    assert quality["native"]["reported_pair_kkt_violation"] == [0.0003]
    assert quality["native"]["max_decision_difference_from_sklearn"] == 0


def test_quality_does_not_accept_agreement_with_an_incorrect_baseline():
    models, query, target = models_for_quality()
    # Native and legacy agree perfectly with each other; the independent
    # sklearn comparator catches a shared 4% classification regression.
    bad = target.copy()
    bad[:4] = 1 - bad[:4]
    for variant in ("native", "legacy"):
        models[variant].predict = lambda X: bad
    assert not bench.estimator_quality("rbf", models, query, target)[1]
    assert bench.estimator_quality("sigmoid", models, query, target)[1]
    bad[4:6] = 1 - bad[4:6]
    assert not bench.estimator_quality("sigmoid", models, query, target)[1]


def test_smoke_case_uses_real_models_with_clock_removed(tmp_path, monkeypatch):
    if not native_smo.available():
        pytest.skip("native extension unavailable")
    legacy = bench.legacy_estimator(tmp_path)
    monkeypatch.setattr(bench, "timed", lambda call: (1.0, call()))
    result = bench.run_case(("smoke", "rbf", 48, 8, 2, "cpu", 1.0), legacy, 1, 51)
    assert result["quality_passed"]
    assert result["quality"]["native"]["backend"] == "cpu"
    for quality in result["quality"].values():
        assert quality["fit_status_samples"] == [0, 0]
    assert result["quality"]["sklearn"]["reported_pair_kkt_violation"] is None
