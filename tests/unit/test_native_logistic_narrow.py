"""The fused one-pass step serves every feature count, including narrow designs."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig
from mlxlearn._solvers.logistic import _binary_objective, _native_binary_objective
from mlxlearn._solvers.native_geometry import speculative_geometry
from mlxlearn._solvers.native_logistic import initial, minimize_binary, step
from mlxlearn._solvers.native_logistic_fused import forward_gradient

WIDTHS = [1, 7, 20, 31, 32, 33, 37, 128, 129, 300, 511]
ROW_COUNTS = [67, 259, 16385, 16391]


def test_the_narrow_line_search_kernel_is_gone():
    """Every width now runs the fused pass; the old two-kernel path must not return."""
    assert not hasattr(native_logistic, "_forward_search")


def test_geometry_keeps_the_wide_specialization_and_rejects_unsupported_widths():
    """The native command loop compiles the wide tuples; they must not move."""
    wide = {
        (67, 512): (2, 1, 16), (67, 1024): (2, 1, 16), (67, 1025): (8, 4, 8),
        (67, 2048): (8, 4, 8), (67, 4096): (8, 4, 8),
        (16384, 512): (4, 1, 32), (16384, 1024): (4, 1, 32), (16384, 1025): (8, 4, 32),
        (16384, 2048): (8, 4, 32), (16384, 2049): (8, 4, 64), (16384, 4096): (8, 4, 64),
    }
    for (n, d), expected in wide.items():
        groups, feature_groups, rows, blocks = speculative_geometry(n, d)
        assert (groups, feature_groups, rows) == expected, (n, d)
        assert blocks == -(-n // ((groups // feature_groups) * rows))
    assert speculative_geometry(16383, 511)[:3] == (8, 1, 4)
    assert speculative_geometry(16384, 511)[:3] == (8, 1, 32)
    assert speculative_geometry(16384, 33)[:3] == (8, 1, 32)
    assert speculative_geometry(16384, 32)[:3] == (4, 1, 32)
    assert speculative_geometry(16384, 1)[:3] == (4, 1, 32)
    assert speculative_geometry(67, 1) == (8, 1, 4, 3)
    # The row-cluster kernel batches four rows per SIMD group.
    for n in (67, 16384):
        for d in (1, 32):
            assert speculative_geometry(n, d)[2] % 4 == 0
    for d in (0, 4097):
        with pytest.raises(ValueError, match="1 to 4096 features"):
            speculative_geometry(16384, d)


@pytest.fixture
def gpu():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with use_device("gpu"):
        yield


def _fixture(width, intercept, n, seed=301):
    rng = np.random.default_rng(seed)
    p = width + int(intercept)
    X = rng.normal(scale=0.2, size=(n, width)).astype(np.float32)
    direction = rng.normal(scale=0.05, size=p).astype(np.float32)
    raw = rng.normal(size=n).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    weights = np.geomspace(1e-3, 1e3, n).astype(np.float32)
    weights[::11] = 0
    return X, direction, raw, y, weights


def _reference(X, direction, raw, y, weights, alpha, candidates, intercept):
    width = X.shape[1]
    X64 = X.astype(np.float64)
    d64 = direction.astype(np.float64)
    denominator = float(weights.sum(dtype=np.float64))
    projection = X64 @ d64[:width] + (d64[width] if intercept else 0.0)
    sign = 1.0 - 2.0 * y
    u = sign * raw.astype(np.float64)
    changes = np.array([
        np.sum(weights * (np.logaddexp(0.0, u + sign * projection * alpha * 2.0**-k)
                          - np.logaddexp(0.0, u))) / denominator
        for k in range(candidates)
    ])
    candidate = raw.astype(np.float64) + alpha * projection
    residual = weights * sign * expit(sign * candidate)
    gradient = residual @ X64 / denominator
    if intercept:
        gradient = np.append(gradient, residual.sum() / denominator)
    return projection, changes, gradient, denominator


@pytest.mark.gpu
@pytest.mark.parametrize("alpha, candidates", [(1.0, 8), (2.0**-8, 24), (0.5, 32)])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("n", ROW_COUNTS)
@pytest.mark.parametrize("width", WIDTHS)
def test_fused_pass_matches_float64_on_narrow_designs(gpu, width, intercept, n, alpha, candidates):
    """Projections, every candidate's loss change and the first gradient, float64-checked.

    Row counts straddle the launch-geometry switch at 16384 and leave partial
    row batches; widths straddle every quad and SIMD-lane boundary below 512
    and both kernel switches (32/33 and 128/129). The solver issues at most
    24 candidates per batch; 32 fills every lane slot of the narrow kernels.
    """
    X, direction, raw, y, weights = _fixture(width, intercept, n)
    projection, changes, gradient, denominator = _reference(
        X, direction, raw, y, weights, alpha, candidates, intercept
    )
    stats = mx.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, alpha, 0.0])
    actual = forward_gradient(
        mx.array(X), mx.array(direction), mx.array(raw), mx.array(y), mx.array(weights),
        stats, mx.array(denominator), candidates,
    )
    mx.eval(actual)
    actual_projection, actual_changes, actual_gradient = (np.asarray(value, dtype=np.float64) for value in actual)
    # The worst observed projection error over this matrix is 1.3e-7 at 511
    # features; the bound leaves about an order of magnitude for other GPUs.
    np.testing.assert_allclose(actual_projection, projection, rtol=1e-5, atol=5e-8 * math.sqrt(width))
    # Loss changes are sums of per-row differences evaluated with the stable
    # small-argument series; their float32 block partials bound the error.
    scale = np.max(np.abs(changes))
    np.testing.assert_allclose(actual_changes, changes, rtol=2e-4, atol=1e-6 * scale + 1e-9)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=2e-5, atol=8e-8)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [1, 7, 32, 33, 128, 511])
def test_converged_status_makes_the_fused_step_a_no_op(gpu, width):
    """With the device status already final, nothing in the state may move."""
    rng = np.random.default_rng(11)
    n, p = 259, width + 1
    X = mx.array(rng.normal(size=(n, width)).astype(np.float32))
    y = mx.array(rng.integers(0, 2, size=n).astype(np.float32))
    weights = mx.ones(n)
    reg = mx.concatenate([mx.full(width, 0.1), mx.zeros(1)])
    denominator = mx.array(float(n))
    w = mx.array(rng.normal(size=p).astype(np.float32))
    g = mx.array(rng.normal(scale=1e-3, size=p).astype(np.float32))
    history = mx.array(rng.normal(size=(3, p)).astype(np.float32))
    # Accepted iterations, status 1 (converged), history count/head, loss
    # statistics, zero accepted step, evaluations, running objective.
    meta = mx.array([5.0, 1.0, 2.0, 2.0, -1e-6, 0.0, 0.0, 0.0, 0.0, 40.0, -2e-6, 0.4])
    raw = X @ w[:width] + w[width]
    before = (w, g, w - 0.1, g + 1e-4, history, history * 2, mx.ones((3, 2)), meta, raw)
    mx.eval(before)
    after = step(
        X, y, weights, reg, denominator, mx.array(1e-4), mx.ones(p), mx.array(1e-4),
        mx.array(1e-10), 8, mx.array(1.0), mx.zeros(width), *before,
    )
    mx.eval(after)
    for name, old, new in zip("w g oldw oldg hs hy hc meta raw".split(), before, after, strict=True):
        if name in ("oldw", "oldg"):
            # The step always shifts the current iterate into the previous slot.
            old = before[0] if name == "oldw" else before[1]
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old), err_msg=name)


def _single_step(X, y, inverse_scale, alpha_start=1.0, candidates=8):
    n, d = X.shape
    Xm, ym = mx.array(X), mx.array(y)
    weights, reg = mx.ones(n), mx.zeros(d)
    denominator = mx.array(float(n))
    gradient, _ = initial(Xm, ym, weights, d, reg, denominator)
    w = mx.zeros(d)
    history = mx.zeros((3, d))
    result = step(
        Xm, ym, weights, reg, denominator, mx.array(1e-8), mx.full(d, inverse_scale),
        mx.array(1e-4), mx.array(1e-10), candidates, mx.array(alpha_start), mx.zeros(d),
        w, gradient, w, gradient, history, mx.zeros_like(history), mx.zeros((3, 2)),
        mx.concatenate([mx.zeros(11), mx.array([np.log(2)])]), mx.zeros(n),
    )
    mx.eval(result)
    return result


@pytest.mark.gpu
@pytest.mark.parametrize("width", [32, 64, 300])
@pytest.mark.parametrize(
    "metric_product, alpha, candidates, first",
    [
        (1.0, 1.0, 8, True),
        (20.0, 1.0, 8, False),
        (1024.0, 2.0**-8, 24, True),
    ],
)
def test_narrow_step_uses_the_accepted_candidates_gradient(gpu, width, metric_product, alpha, candidates, first):
    """Narrow analogue of the wide contract: the speculative gradient is reused
    only for an accepted first candidate; any later candidate takes the
    separate backward pass, which must match a fresh native evaluation exactly.

    On the rank-one design the unit step moves every logit by
    ``-width * inverse_scale * gradient``; keeping that product fixed across
    widths keeps the same candidate accepted (the quadratic loss term
    rejects the first candidates once the product exceeds about 8).
    """
    X = np.ones((259, width), dtype=np.float32)
    y = (np.arange(len(X)) % 2).astype(np.float32)
    inverse_scale = metric_product / width
    state = _single_step(X, y, inverse_scale, alpha, candidates)
    w, gradient, metadata, raw = state[0], state[1], state[7], state[8]
    assert float(metadata[1]) == 0
    assert (float(metadata[8]) == alpha) is first
    assert float(metadata[8]) > 0
    residual = expit(np.asarray(raw, dtype=np.float64)) - y
    expected = residual @ X / len(X)
    if not first:
        direct = _native_binary_objective(mx.array(X), mx.array(y), None, float(len(X)), 0.0, False)
        _, direct_gradient = direct(w.reshape(1, -1))
        np.testing.assert_array_equal(np.asarray(gradient), np.asarray(direct_gradient)[0])
    np.testing.assert_allclose(
        np.asarray(gradient),
        expected,
        rtol=2e-4,
        atol=2 * float(np.spacing(np.float32(0.5))) if first else 8 * np.finfo(np.float32).eps,
    )
    if first and alpha == 2.0**-8:
        assert abs(float(gradient[0]) - (0.5 - y).mean()) > 1e-3
    if not first:
        wrong_raw = -inverse_scale * (0.5 - y).mean() * X.shape[1]
        assert abs(float(gradient[0]) - (expit(wrong_raw) - y.mean())) > 1e-3
    assert np.isfinite(np.asarray(w)).all()


def _problem(X, y, *, weights=None, C=1.0, fit_intercept=True):
    Xm, ym = mx.array(X), mx.array(y)
    sw = None if weights is None else mx.array(weights, dtype=mx.float32)
    sw_sum = float(len(X) if weights is None else np.sum(weights, dtype=np.float64))
    l2 = 0.0 if math.isinf(C) else 1.0 / (C * sw_sum)
    fun = _binary_objective(mx, Xm, ym, sw, sw_sum, l2, fit_intercept, X.shape[1])
    return Xm, ym, sw, sw_sum, l2, fit_intercept, fun


def _float64_gradient(theta, X, y, weights, l2, fit_intercept):
    coefficients = np.asarray(theta, dtype=np.float64)[0]
    w = coefficients[: X.shape[1]]
    logits = np.asarray(X, dtype=np.float64) @ w
    if fit_intercept:
        logits += coefficients[-1]
    residual = expit(logits) - y
    if weights is not None:
        residual *= weights
    sw_sum = len(X) if weights is None else np.sum(weights, dtype=np.float64)
    gradient = residual @ X / sw_sum + l2 * w
    if fit_intercept:
        gradient = np.append(gradient, residual.sum() / sw_sum)
    return gradient


@pytest.mark.gpu
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("width", [1, 20, 300])
def test_narrow_solver_preserves_float64_gradient_tolerance(gpu, width, weighted, fit_intercept):
    rng = np.random.default_rng(718 + width)
    n = 16391  # Past the geometry switch, with a partial final row batch.
    X = rng.normal(size=(n, width)).astype(np.float32)
    X *= np.geomspace(0.1, 10.0, width).astype(np.float32)
    logits = X[:, 0] * 0.5 + 0.2
    y = (rng.uniform(size=n) < expit(logits)).astype(np.float32)
    weights = None
    if weighted:
        weights = np.geomspace(1e-2, 1e2, n).astype(np.float32)
        weights[::13] = 0
    problem = _problem(X, y, weights=weights, C=1.0, fit_intercept=fit_intercept)
    # The scalar initial metric needs a few hundred iterations on this badly
    # scaled design (the same count as the previous narrow kernel took).
    state = minimize_binary(*problem[:-1], LBFGSConfig(max_iter=600, tol=1e-4), problem[-1])
    assert state.converged, state
    assert state.stop_reason == "gtol"
    gradient = _float64_gradient(state.x, X, y, weights, problem[4], fit_intercept)
    assert np.max(np.abs(gradient)) <= 1.01e-4
    value, _ = problem[-1](state.x)
    assert state.objective == pytest.approx(value, abs=1e-10)


@pytest.mark.gpu
def test_narrow_rank_one_design_continues_the_line_search_through_the_fallback(gpu):
    """A rank-one narrow design rejects the first eight candidates: the resumed
    24-candidate batch and the separate backward pass carry the fit.

    With the scalar initial metric the unit step overshoots a rank-one design
    by a factor of the width, so only widths above 256 exhaust the first
    eight candidates; 300 keeps the narrow launch geometry.
    """
    rng = np.random.default_rng(35)
    column = rng.normal(size=259).astype(np.float32)
    X = np.tile(column[:, None], (1, 300))
    y = rng.integers(0, 2, size=259).astype(np.float32)
    problem = _problem(X, y)

    limited = minimize_binary(*problem[:-1], LBFGSConfig(max_iter=1, max_line_search=8), problem[-1])
    continued = minimize_binary(*problem[:-1], LBFGSConfig(max_iter=1, max_line_search=32), problem[-1])

    assert limited.n_iter == 0
    assert limited.n_fev == 9
    assert limited.stop_reason == "line-search-failure"
    assert continued.n_iter == 1
    assert continued.n_fev == 33
    assert continued.objective < limited.objective
    assert np.isfinite(np.asarray(continued.x)).all()

    full = minimize_binary(*problem[:-1], LBFGSConfig(max_iter=200, max_line_search=32, tol=1e-4), problem[-1])
    assert full.converged, full
    gradient = _float64_gradient(full.x, X, y, None, problem[4], True)
    assert np.max(np.abs(gradient)) <= 1.01e-4


@pytest.mark.gpu
@pytest.mark.parametrize("width", [3, 300])
def test_overflowing_narrow_design_never_claims_convergence(gpu, width):
    """Projections overflow float32 on an extreme design; the search fails
    honestly and the finite starting point is kept."""
    rng = np.random.default_rng(5)
    X = (rng.normal(size=(259, width)) * 1e19).astype(np.float32)
    y = rng.integers(0, 2, size=259).astype(np.float32)
    problem = _problem(X, y, C=math.inf)
    state = minimize_binary(*problem[:-1], LBFGSConfig(max_iter=20, tol=1e-4), problem[-1])
    assert not state.converged
    assert state.stop_reason in ("line-search-failure", "non-finite-objective")
    assert np.isfinite(np.asarray(state.x)).all()
