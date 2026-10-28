"""The L-BFGS curvature floor is relative, and a frozen native loop stops early.

A feature scaled 1e3 to 1e5 times the others makes ``yᵀH₀y`` far below one.
The former guard ``sᵀy > min_curvature · max(yᵀy, 1)`` then became an absolute
1e-12 floor, rejected every valid pair, froze the history, and the native loop
accepted one-ulp penalty-only steps until ``max_iter`` while the model stayed
at the null model (accuracy 0.51 against scikit-learn's 0.86).
"""

from __future__ import annotations

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig, curvature_pair_accepted
from mlxlearn._solvers.logistic import reference_objective
from mlxlearn.linear_model import LogisticRegression


@pytest.mark.parametrize(
    ("ys", "yy", "accepted"),
    [
        (1e-13, 1e-14, True),  # Rejected by the former absolute 1e-12 floor.
        (2e-12, 1.0, True),
        (1e-12, 1.0, False),  # The floor itself is not accepted.
        (1e-13, 0.0, False),  # y == 0 never forms a pair.
        (0.0, 0.0, False),
        (-1.0, 1.0, False),
        (1.0, 1e30, False),  # Positive but far below the relative floor.
        (math.nan, 1.0, False),
        (1.0, math.nan, False),
    ],
)
def test_curvature_pair_floor_is_relative_to_the_pair(ys, yy, accepted):
    assert curvature_pair_accepted(ys, yy, 1e-12) is accepted


def test_zero_floor_accepts_any_positive_curvature():
    assert curvature_pair_accepted(1e-30, 1e-20, 0.0)
    assert not curvature_pair_accepted(0.0, 1e-20, 0.0)


def _scaled_column_design(n_samples, n_features, scale, seed=0):
    """Standard-normal features with one column scaled; labels from a linear signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    w = rng.standard_normal(n_features) / math.sqrt(n_features)
    y = (X @ w + 0.5 * rng.standard_normal(n_samples) > 0).astype(np.int64)
    X[:, 3] *= scale
    return X, y


@pytest.mark.gpu
def test_native_history_stores_pairs_whose_metric_norm_is_below_one():
    """On a 1e5-scaled column the device history must keep growing."""
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    n, d = 20_000, 16
    X, y = _scaled_column_design(n, d, 1e5)
    center = X.mean(axis=0)
    p = d + 1
    counts = []
    with use_device("gpu"):
        Xm = mx.array((X - center).astype(np.float32))
        ym = mx.array(y.astype(np.float32))
        sw = mx.ones(n, dtype=mx.float32)
        reg = mx.concatenate([mx.full(d, 1.0 / n, dtype=mx.float32), mx.zeros(1, dtype=mx.float32)])
        denom = mx.array(float(n), dtype=mx.float32)
        g, h0inv = native_logistic.initial(Xm, ym, sw, p, reg, denom)
        w = mx.zeros(p, dtype=mx.float32)
        history = mx.zeros((10, p), dtype=mx.float32)
        meta = mx.concatenate([mx.zeros(11), mx.array([math.log(2)])]).astype(mx.float32)
        state = (w, g, w, g, history, mx.zeros_like(history), mx.zeros((10, 2)), meta, mx.zeros(n))
        for _ in range(12):
            state = native_logistic.step(
                Xm, ym, sw, reg, denom, mx.array(1e-4, dtype=mx.float32), h0inv,
                mx.array(1e-4, dtype=mx.float32), mx.array(1e-12, dtype=mx.float32), 8,
                mx.array(1.0, dtype=mx.float32), mx.array(center.astype(np.float32)), *state,
            )
            mx.eval(state)
            host_meta = np.asarray(state[7])
            counts.append(int(host_meta[2]))
            assert int(host_meta[1]) == 0, host_meta
    # The absolute floor froze the ring at one or two entries.
    assert counts[-1] >= 8, counts
    assert counts == sorted(counts)


@pytest.mark.parametrize("logits_move", [False, True])
def test_accepted_steps_with_unchanged_logits_reset_history_once_then_stop(monkeypatch, logits_move):
    """A penalty-only step that leaves every cached logit unchanged is a stall."""
    with mx.stream(mx.cpu):
        X = mx.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], mx.float32)
        y = mx.array([0, 1, 0], mx.float32)
        # A gradient above the tolerance keeps the device status active, so
        # the chunk check reaches the stall test rather than a reanchor.
        monkeypatch.setattr(
            native_logistic, "initial", lambda *args: (mx.full((3,), 1e-3), mx.ones((3,)))
        )
        seen = []

        def accepted_step(*args, **kwargs):
            w, g, oldw, oldg, hs, hy, hc, meta, raw = args[12:]
            values = np.asarray(meta).copy()
            seen.append((int(values[2]), int(values[3]), bool(np.any(np.asarray(hs)))))
            values[0] += 1  # Accepted.
            values[2] = min(values[2] + 1, 10)
            values[3] = (values[3] + 1) % 10
            values[4] = values[10] = -1e-9
            values[8] = 1.0
            values[9] += 8
            values[11] = 0.2
            new_raw = raw + 1e-3 if logits_move else raw
            return w + 1e-3, g, w, g, hs + 1.0, hy + 1.0, hc + 1.0, mx.array(values), new_raw

        monkeypatch.setattr(native_logistic, "step", accepted_step)
        result = native_logistic.minimize_binary(
            X, y, None, 3.0, 0.1, True,
            LBFGSConfig(max_iter=12, tol=1e-4, max_line_search=8),
            lambda theta: (0.2, mx.full(theta.shape, 1e-3)),
        )
    assert seen[:4] == [(0, 0, False), (1, 1, True), (2, 2, True), (3, 3, True)]
    if logits_move:
        # Ordinary progress: the history is kept and the budget is spent.
        assert len(seen) == 12 and seen[4] == (4, 4, True)
        assert result.n_iter == 12 and result.stop_reason == "max_iter"
    else:
        # One reset to steepest descent, then a failed-search report after the
        # second stalled chunk: 8 accepted steps, not the 12 of the budget.
        assert len(seen) == 8 and seen[4] == (0, 0, False)
        assert seen[5:] == [(1, 1, True), (2, 2, True), (3, 3, True)]
        assert result.n_iter == 8 and result.stop_reason == "line-search-failure"
    assert not result.converged
    assert result.n_fev == 8 * len(seen) + 1


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("n_samples", "n_features"),
    [(20_000, 16), (20_000, 1_024)],
    ids=["narrow-mlx-loop", "wide-prepared-native-loop"],
)
def test_scaled_column_fit_matches_sklearn_instead_of_the_null_model(n_samples, n_features):
    """Both GPU routes must fit a 1e5-scaled column to scikit-learn's accuracy.

    The frozen history returned the null model (objective ``log 2``, accuracy
    0.51). A ``ConvergenceWarning`` can remain: float32 L-BFGS does not reach
    ``tol=1e-4`` on this design within the budget, and neither does scikit-learn
    on the wide case. The gate is the fitted model, not the warning.
    """
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    X, y = _scaled_column_design(n_samples, n_features, 1e5)
    params = {"C": 1.0, "tol": 1e-4, "max_iter": 100}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with tuning_context(force_backend="mlx"):
            model = LogisticRegression(**params).fit(X, y)
        reference = SkLogisticRegression(**params).fit(X, y)
    assert model._execution_backend_ == "mlx"
    accuracy = float(np.mean(model.predict(X) == y))
    reference_accuracy = float(np.mean(reference.predict(X) == y))
    assert accuracy >= reference_accuracy - 0.02, (accuracy, reference_accuracy)
    objective = reference_objective(model.coef_, model.intercept_, X, y, n_classes=2)
    reference_value = reference_objective(reference.coef_, reference.intercept_, X, y, n_classes=2)
    # The null model sits 1.3 relative above the optimum.
    assert (objective - reference_value) / reference_value <= 0.05, (objective, reference_value)
