"""Original-coordinate convergence on production-sized shifted GPU inputs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning

from mlxlearn._common.config import config_context, tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import native_command
from mlxlearn.linear_model import LogisticRegression

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def production_data():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    rng = np.random.default_rng(461)
    n, d = 20_000, 1024
    X = rng.normal(size=(n, d)).astype(np.float32)
    score = X[:, 0] + rng.normal(scale=0.5, size=n)
    labels = {2: (score > 0).astype(int), 3: np.digitize(score, [-0.5, 0.5])}
    X += np.float32(1000.0)
    weights = rng.uniform(0.25, 3.0, n)
    return X, labels, weights


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_production_shifted_fit_certifies_original_gradient(
    production_data, monkeypatch, n_classes, weighted
):
    X, labels, weights = production_data
    y = labels[n_classes]
    sw = weights if weighted else None
    calls = []
    original = native_command.run_native_loop

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(native_command, "run_native_loop", observe)
    with (
        config_context(device="gpu"),
        tuning_context(force_backend="mlx"),
        warnings.catch_warnings(record=True) as seen,
    ):
        warnings.simplefilter("always")
        model = LogisticRegression(C=1.0, tol=1e-4, max_iter=200).fit(X, y, sample_weight=sw)

    assert model.execution_backend_ == "mlx"
    assert 0 < int(model.n_iter_[0]) <= 200
    convergence_warnings = [str(w.message) for w in seen if issubclass(w.category, ConvergenceWarning)]
    assert not convergence_warnings, convergence_warnings
    if n_classes == 2 and native_command._metal_loop is not None and native_command._supports_metal_dlpack():
        # The command loop may legitimately return None when a direct check
        # needs reanchoring; in that case the established GPU solver continues.
        assert calls

    data = X.astype(np.float64)
    raw = data @ model.coef_.T + model.intercept_
    if n_classes == 2:
        residual = (expit(raw[:, 0]) - y)[:, None]
        losses = np.logaddexp(0.0, (1 - 2 * y) * raw[:, 0])
    else:
        residual = softmax(raw, axis=1)
        residual[np.arange(len(y)), y] -= 1.0
        from scipy.special import logsumexp

        losses = logsumexp(raw, axis=1) - raw[np.arange(len(y)), y]
    denominator = float(len(y) if sw is None else sw.sum())
    if sw is not None:
        residual *= sw[:, None]
        losses *= sw
    gradient = residual.T @ data / denominator + model.coef_ / denominator
    bias_gradient = residual.sum(axis=0) / denominator
    assert max(np.max(np.abs(gradient)), np.max(np.abs(bias_gradient))) <= 1e-4
    objective = losses.sum() / denominator + np.square(model.coef_).sum() / (2 * denominator)
    assert np.isfinite(objective) and objective < np.log(n_classes)


def test_shifted_gpu_respects_exhausted_iteration_budget(production_data):
    X, labels, _ = production_data
    with (
        config_context(device="gpu"),
        tuning_context(force_backend="mlx"),
        pytest.warns(ConvergenceWarning),
    ):
        model = LogisticRegression(C=1.0, tol=1e-4, max_iter=1).fit(X, labels[2])
    assert int(model.n_iter_[0]) == 1
