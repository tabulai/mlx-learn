"""The opt-in Hessian sketch is a capability of the prepared native loop.

``MLXLEARN_LBFGS_SKETCH_SIZE`` (or ``tuning_context(lbfgs_sketch_size=...)``)
used to turn every GPU fit that cannot build the sketch into a ``ValueError``:
fewer than 512 or more than 4096 features, fewer than four million elements,
``fit_intercept=False``, multinomial targets, a non-default history, or a
missing extension. The estimator now runs such fits unsketched at the requested
tolerance; only a direct :func:`solve_logistic` request for a sketch is an error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from mlxlearn._common.config import get_tuning, reset_config, tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import logistic, native_command
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, True),
        ({"initial_parts": None}, False),
        ({"n_targets": 3}, False),
        ({"fit_intercept": False}, False),
        ({"n_features": 511}, False),
        ({"n_features": 4097}, False),
        ({"history_size": 8}, False),
        ({"max_line_search": 4}, False),
        ({"metal_loop": None}, False),
        ({"dlpack": False}, False),
    ],
)
def test_sketch_support_needs_every_native_loop_capability(monkeypatch, change, expected):
    monkeypatch.setattr(native_command, "_metal_loop", change.get("metal_loop", object()))
    monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: change.get("dlpack", True))
    config = logistic.LogisticConfig(
        fit_intercept=change.get("fit_intercept", True),
        history_size=change.get("history_size", 10),
        max_line_search=change.get("max_line_search", 8),
    )
    parts = change.get("initial_parts", object())
    supported = logistic.sketch_supported(
        config, change.get("n_targets", 1), change.get("n_features", 1024), parts
    )
    assert supported is expected


def _small_problem(n_classes, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1000, 8))
    signal = X[:, :n_classes] + 0.3 * rng.standard_normal((1000, n_classes))
    y = (signal[:, 0] > 0).astype(np.int64) if n_classes == 2 else np.argmax(signal, axis=1)
    return X, y


@pytest.mark.gpu
@pytest.mark.parametrize("n_classes", [2, 3])
def test_environment_sketch_opt_in_does_not_break_fits_it_cannot_seed(monkeypatch, n_classes):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    monkeypatch.setenv("MLXLEARN_LBFGS_SKETCH_SIZE", "4")
    reset_config()
    assert get_tuning().lbfgs_sketch_size == 4
    X, y = _small_problem(n_classes)
    observed = []
    original = estimator.solve_logistic

    def observe(*args, **kwargs):
        solution = original(*args, **kwargs)
        observed.append((kwargs["config"], solution))
        return solution

    monkeypatch.setattr(estimator, "solve_logistic", observe)
    with tuning_context(force_backend="mlx"), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=1.0, tol=1e-4, max_iter=100).fit(X, y)
    assert model._execution_backend_ == "mlx"
    (config, solution), = observed
    assert config.sketch_size == 0
    assert solution.converged and solution.required_tol == pytest.approx(1e-4)
    assert float(np.mean(model.predict(X) == y)) > 0.5


@pytest.mark.gpu
def test_supported_width_sketch_opt_in_falls_back_without_native_extension(monkeypatch):
    """A missing extension alone disables the optional sketch for a public fit."""
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    monkeypatch.setattr(native_command, "_metal_loop", None)
    monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
    rng = np.random.default_rng(71)
    X = rng.standard_normal((2048, 512)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=len(X)) > 0).astype(np.int64)
    observed = []
    original = estimator.solve_logistic

    def observe(*args, **kwargs):
        config, parts = kwargs["config"], kwargs["initial_parts"]
        assert parts is not None
        # Prove no other gate (width, prepared statistics, or MLX version)
        # explains the fallback before exercising the actual unseeded solve.
        with monkeypatch.context() as available:
            available.setattr(native_command, "_metal_loop", object())
            assert logistic.sketch_supported(config, 1, X.shape[1], parts)
        solution = original(*args, **kwargs)
        observed.append((config, solution))
        return solution

    monkeypatch.setattr(estimator, "solve_logistic", observe)
    with tuning_context(force_backend="mlx", lbfgs_sketch_size=4), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=1.0, tol=1e-4, max_iter=100).fit(X, y)
    assert model.execution_backend_ == "mlx"
    (config, solution), = observed
    assert config.sketch_size == 0
    assert solution.converged and solution.required_tol == pytest.approx(1e-4)
    assert np.isfinite(model.coef_).all()


@pytest.mark.gpu
@pytest.mark.parametrize("n_classes", [2, 3])
def test_direct_sketch_request_without_the_capability_is_still_an_error(n_classes):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    X, y = _small_problem(n_classes)
    with pytest.raises(ValueError, match="Hessian sketch requires"):
        logistic.solve_logistic(
            X.astype(np.float32), y, n_classes=n_classes,
            config=logistic.LogisticConfig(device="gpu", sketch_size=4),
        )
