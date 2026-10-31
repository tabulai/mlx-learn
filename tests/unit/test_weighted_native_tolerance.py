"""A weighted loop reserves rounding margin without relaxing certification."""

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available
from mlxlearn._solvers.native_command import _native_loop_tol
from mlxlearn_device import _logistic, fit_logistic


@pytest.mark.parametrize("tol", [1.0, 1e-4, 2.5e-5, float(np.finfo(np.float32).tiny)])
def test_weighted_loop_tolerance_never_exceeds_requested(tol):
    unweighted = _native_loop_tol(tol, False)
    weighted = _native_loop_tol(tol, True)
    assert 0 < weighted <= unweighted <= tol
    if tol * 0.95 >= np.finfo(np.float32).tiny:
        assert tol * 0.949 < weighted <= tol * 0.95
    assert math.isfinite(weighted)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tol", [1e40, float(np.finfo(np.float64).max)])
def test_large_finite_tolerance_clamps_without_overflow(tol, weighted):
    with np.errstate(over="raise"):
        actual = _native_loop_tol(tol, weighted)
    assert actual == float(np.finfo(np.float32).max)
    assert actual <= (0.95 * tol if weighted else tol)


@pytest.mark.parametrize("sketch_size", [0, 1])
@pytest.mark.gpu
def test_device_weighted_reserve_is_applied_once(monkeypatch, sketch_size):
    if not gpu_available() or _logistic.metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    calls = []

    def run(*args):
        calls.append(args)
        return None, 3, 1, 0, 0.0, 0.0

    def score(*args):
        return mx.array(0.2), mx.array(1e-6), mx.zeros(513), mx.zeros(513), mx.array(0.0)

    def sketch(*args, **kwargs):
        return mx.zeros((10, 513)), mx.zeros((10, 513)), mx.zeros((10, 2)), 1

    monkeypatch.setattr(_logistic.metal_loop, "run", run)
    monkeypatch.setattr(_logistic, "score_original", score)
    monkeypatch.setattr(_logistic, "build_sketch_history", sketch)
    with mx.stream(mx.gpu), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X = mx.reshape(mx.arange(4 * 512, dtype=mx.float32), (4, 512)) / 2048
        weights = mx.array([0.5, 1.5, 0.5, 1.5], dtype=mx.float32)
        model = fit_logistic(X, mx.array([0, 1, 0, 1]), sample_weight=weights,
                             max_iter=10, sketch_size=sketch_size)
    requested = _logistic._device_tol(2.5e-5)
    assert len(calls) == 1 and calls[0][10] == _logistic._seeded_native_tol(requested)
    assert requested * 0.949 < calls[0][10] <= requested * 0.95
    assert calls[0][16] == sketch_size and calls[0][-1] == 1
    assert not caught and model.converged and model.n_iter == 3
