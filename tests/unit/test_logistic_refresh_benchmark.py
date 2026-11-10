"""Protocol checks use tiny fake fits; they do not benchmark hardware."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import logistic_refresh as bench


def good_score():
    return {"objective_float64": .5, "gradient_max_abs_float64": 1e-6,
            "objective_relative_gap_to_sklearn": 0.0,
            "prediction_agreement_sklearn": 1.0}


def good_detail(variant):
    backend = {"direct_auto": "cpu", "patched_auto": "sklearn",
               "forced_gpu": "mlx", "sklearn": "sklearn"}[variant]
    return {"error": None, "warnings": [], "backend": backend,
            "solutions": [] if backend == "sklearn" else [{"converged": True}],
            "solver_calls": [] if backend == "sklearn" else [{"implementation": "observed"}]}


def test_full_grid_deduplicates_by_semantics_and_covers_auto_multiclass():
    signatures = [bench.case_signature(case) for case in bench.CASES]
    assert len(signatures) == len(set(signatures)) == 25
    for rows in (1_000, 10_000, 100_000, 500_000):
        for features in (16, 128, 1_024):
            assert any(case.n_samples == rows and case.n_features == features
                       and case.n_classes == 2 for case in bench.CASES)
    assert any(case.n_samples == 20_000 and case.n_features == 1_024
               and case.n_classes == 5 and case.weighted for case in bench.CASES)
    assert bench.selected_cases(["wide_20k_1024"])[0].name == "binary_wide"
    assert bench.selected_cases(["grid_500000_128"])[0].name == "binary_tall_wide"
    with pytest.raises(ValueError, match="unknown cases"):
        bench.selected_cases(["not_a_case"])


def test_rotated_order_balances_positions():
    orders = [bench.rotated(bench.VARIANTS, repeat) for repeat in range(4)]
    for position in range(4):
        assert {order[position] for order in orders} == set(bench.VARIANTS)


@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_float32_sklearn_coefficients_receive_full_float64_certificate(n_classes, weighted):
    rng = np.random.default_rng(392)
    X = rng.normal(0, 8, (257, 17)).astype(np.float32)
    codes = rng.integers(n_classes, size=len(X))
    classes = 10 * np.arange(1, n_classes + 1)
    y = classes[codes]
    weights = rng.uniform(.1, 5, len(X)).astype(np.float32) if weighted else None
    if weights is not None:
        weights[::7] = 0
    model = bench.SkLogisticRegression(C=1.0)
    targets = 1 if n_classes == 2 else n_classes
    model.coef_ = rng.normal(0, .15, (targets, X.shape[1])).astype(np.float32)
    model.intercept_ = rng.normal(0, .2, targets).astype(np.float32)
    model.classes_, model.n_features_in_ = classes, X.shape[1]
    original_predict = model.predict
    prediction_calls = []

    def predict(original_input):
        assert original_input is X
        assert original_input.dtype == np.float32
        prediction_calls.append(True)
        return original_predict(original_input)

    model.predict = predict
    # A separate row-wise float64 calculation checks every contribution,
    # including sample weighting, the unpenalized bias and the C=1 penalty.
    W = model.coef_.astype(np.float64)
    b = model.intercept_.astype(np.float64)
    sw = np.ones(len(X), dtype=np.float64) if weights is None else weights.astype(np.float64)
    total_weight = sw.sum(dtype=np.float64)
    gradient = np.zeros_like(W)
    bias_gradient = np.zeros_like(b)
    loss = np.float64(0)
    for values, target, weight in zip(X.astype(np.float64), codes, sw, strict=True):
        logits = W @ values + b
        if n_classes == 2:
            residual = 1 / (1 + np.exp(-logits)) - target
            row_loss = np.logaddexp(0, logits[0]) - target * logits[0]
        else:
            exponentials = np.exp(logits - logits.max())
            residual = exponentials / exponentials.sum(dtype=np.float64)
            residual[target] -= 1
            row_loss = logits.max() + np.log(exponentials.sum(dtype=np.float64)) - logits[target]
        gradient += np.outer(residual * weight, values)
        bias_gradient += residual * weight
        loss += weight * row_loss
    gradient = (gradient + W) / total_weight
    bias_gradient /= total_weight
    expected_gradient = max(np.abs(gradient).max(), np.abs(bias_gradient).max())
    expected_objective = (loss + .5 * np.square(W).sum(dtype=np.float64)) / total_weight
    result = bench.quality(model, model, X, y, weights)
    assert result["gradient_max_abs_float64"] == pytest.approx(expected_gradient, rel=1e-13, abs=1e-13)
    assert result["objective_float64"] == pytest.approx(expected_objective, rel=1e-13, abs=1e-13)
    assert result["objective_relative_gap_to_sklearn"] == 0
    assert result["prediction_agreement_sklearn"] == 1
    assert len(prediction_calls) == 2
    assert model.coef_.dtype == model.intercept_.dtype == X.dtype == np.float32
    if weights is not None:
        assert weights.dtype == np.float32
    json.dumps(result)


def test_bad_stock_reference_does_not_qualify_fast_candidate(monkeypatch):
    models = {v: SimpleNamespace(name=v) for v in bench.VARIANTS}

    def quality(model, reference, *args):
        return good_score() | {"gradient_max_abs_float64": .01 if model.name == "sklearn" else 1e-6}

    monkeypatch.setattr(bench, "quality", quality)
    audits = bench.audit_models(models, {v: good_detail(v) for v in bench.VARIANTS},
                                None, None, None, 1e-4)
    assert audits["forced_gpu"]["own_checks_pass"]
    assert not audits["sklearn"]["own_checks_pass"]
    assert all(not audit["passed"] and not audit["reference_valid"] for audit in audits.values())
    timings = {
        v: bench.timing_summary([.001 if v != "sklearn" else .01], 1)
        for v in bench.VARIANTS}
    ratios = bench.speedups(timings, {v: audits[v]["passed"] for v in bench.VARIANTS})
    assert ratios["forced_gpu"]["timing_ratio"] == 10
    assert ratios["forced_gpu"]["speedup_over_sklearn"] is None
    assert not ratios["forced_gpu"]["quality_qualified"]


@pytest.mark.parametrize("change,failed", [
    ({"gradient_max_abs_float64": np.nan}, "finite"),
    ({"objective_relative_gap_to_sklearn": 1.1e-5}, "objective"),
    ({"prediction_agreement_sklearn": .99}, "predictions"),
])
def test_quality_failures_are_not_filtered(change, failed):
    assert not bench.fit_checks("forced_gpu", good_detail("forced_gpu"),
                                good_score() | change, 1e-4)[failed]


def test_expected_fallback_warning_is_not_nonconvergence():
    detail = good_detail("patched_auto")
    detail["warnings"] = [{"category": "FallbackWarning", "message": "below crossover"}]
    assert bench.fit_checks("patched_auto", detail, good_score(), 1e-4)["convergence"]
    detail["warnings"].append({"category": "ConvergenceWarning", "message": "max_iter"})
    assert not bench.fit_checks("patched_auto", detail, good_score(), 1e-4)["convergence"]


def test_missing_or_wrong_execution_path_fails_audit():
    detail = good_detail("forced_gpu") | {"backend": "cpu"}
    assert not bench.fit_checks("forced_gpu", detail, good_score(), 1e-4)["backend"]
    detail = good_detail("forced_gpu") | {"solver_calls": []}
    assert not bench.fit_checks("forced_gpu", detail, good_score(), 1e-4)["solver_observed"]
    assert not all(bench.fit_checks("sklearn", good_detail("sklearn"), None, 1e-4).values())


def test_timing_failure_preserves_sample_and_invalidates_ratio():
    sample = bench.timing_summary([.001, None, .003], 3)
    assert sample["samples_s"] == [.001, None, .003]
    assert not sample["complete"]
    timings = dict.fromkeys(bench.VARIANTS, sample)
    assert not bench.speedups(timings, dict.fromkeys(bench.VARIANTS, True))["forced_gpu"]["quality_qualified"]


def test_probability_checks_reject_nan_shape_and_bad_normalization():
    reference = np.array([[.2, .8], [.7, .3]])
    assert bench.inference_check("predict_proba", reference, reference, 2, 2)["passed"]
    assert not bench.inference_check("predict_proba", reference[:, :1], reference, 2, 2)["passed"]
    assert not bench.inference_check("predict_proba", reference * 2, reference, 2, 2)["passed"]
    assert not bench.inference_check("predict_proba", reference * np.nan, reference, 2, 2)["passed"]
    distorted = np.array([[.01, .99], [.99, .01]])
    check = bench.inference_check("predict_proba", distorted, reference, 2, 2)
    assert check["prediction_agreement_sklearn"] == 1.0
    assert not check["probabilities_close"]
    assert not check["passed"]


def test_float32_probability_rounding_is_not_invalid_normalization():
    probabilities = np.full((2, 3), 1 / 3, dtype=np.float32)
    result = bench.inference_check("predict_proba", probabilities, probabilities, 2, 3)
    assert result["passed"]
    assert result["normalization_max_error"][0] > 1e-8
    assert result["normalization_max_error"][0] < result["normalization_tolerance"][0]
    json.dumps(result)


def test_run_case_audits_each_fit_and_separates_inference(monkeypatch):
    fit_calls, score_calls, prediction_calls = [], [], []

    class Model:
        def predict(self, query):
            prediction_calls.append(("predict", len(query)))
            return np.zeros(len(query), dtype=int)

        def predict_proba(self, query):
            prediction_calls.append(("predict_proba", len(query)))
            return np.tile([.7, .3], (len(query), 1))

    def fit(variant, X, y, weights, **kwargs):
        fit_calls.append(variant)
        return .001, Model(), good_detail(variant)

    def score(*args):
        score_calls.append(True)
        return good_score()

    monkeypatch.setattr(bench, "fit_once", fit)
    monkeypatch.setattr(bench, "quality", score)
    monkeypatch.setattr(bench, "timed_call", lambda fn: (.002, fn(), {"error": None, "warnings": []}))
    case = replace(bench.CASES[0], name="tiny", n_samples=20, n_features=3)
    row = bench.run_case(case, seed=41, dtype="float32", repeats=2, warmup=1,
                         tol=1e-4, max_iter=200, queries=[1, 5])
    assert len(fit_calls) == 4 * 3
    assert len(score_calls) == 4 * 2
    assert len(prediction_calls) == 4 * 3 * 2 * 2
    assert row["quality_audit_pass"]
    assert len(row["fit"]["timings"]["forced_gpu"]["samples_s"]) == 2
    assert len(row["fit"]["quality_each_fit"]["sklearn"]) == 2
    assert set(row["inference"]) == {"1", "5"}
    assert set(row["inference"]["1"]) == {"predict", "predict_proba"}
    json.dumps(row)


def test_checkpoint_roundtrip_and_resume_reject_changed_panel(tmp_path):
    parameters, hashes = {"tol": 1e-4, "requested_thread_limit": None}, {"source": "digest"}
    payload = {"schema_version": bench.SCHEMA_VERSION,
               "quality_certificate": bench.QUALITY_CERTIFICATE, "parameters": parameters,
               "source_integrity_before": hashes,
               "thread_environment_before_import": bench.THREAD_ENV_AT_IMPORT,
               "results": [{"quality_audit_pass": False}]}
    path = tmp_path / "output.json"
    bench.checkpoint(path, payload)
    restored = json.loads(path.read_text())
    bench.validate_resume(restored, parameters, hashes)
    assert restored["results"][0]["quality_audit_pass"] is False
    assert not path.with_name("output.json.tmp").exists()
    with pytest.raises(ValueError, match="float64 quality certificate schema"):
        bench.validate_resume(restored | {"schema_version": 1}, parameters, hashes)
    with pytest.raises(ValueError, match="float64 quality certificate schema"):
        bench.validate_resume(restored | {"quality_certificate": "old"}, parameters, hashes)
    with pytest.raises(ValueError, match="parameters differ"):
        bench.validate_resume(restored, parameters | {"tol": 1e-5}, hashes)
    with pytest.raises(ValueError, match="source or native"):
        bench.validate_resume(restored, parameters, {"source": "changed"})
    restored["thread_environment_before_import"] = {"VECLIB_MAXIMUM_THREADS": "changed"}
    with pytest.raises(ValueError, match="thread environment"):
        bench.validate_resume(restored, parameters, hashes)


@pytest.mark.parametrize("change", [{"numpy": "different"}, {"tuning": {"lbfgs_history": 11}}])
def test_resume_rejects_changed_runtime_or_tuning(change):
    environment = {"numpy": "version", "tuning": {"lbfgs_history": 10}}
    payload = {"sessions": [{"environment": environment}]}
    bench.validate_runtime(payload, environment)
    with pytest.raises(ValueError, match="runtime, package versions, tuning"):
        bench.validate_runtime(payload, environment | change)
