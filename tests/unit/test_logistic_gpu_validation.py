"""GPU validation and temporary input ownership preserve the estimator contract."""

from __future__ import annotations

import contextlib
import gc
import weakref

import mlx.core as mx
import numpy as np
import pytest
from sklearn import config_context as sklearn_config_context

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._kernels.logistic_input import prepare_centered
from mlxlearn.exceptions import BackendExecutionError, UnsupportedInputError
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _require_gpu():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")


def _data(dtype=np.float32):
    rng = np.random.default_rng(842)
    X = rng.normal(size=(129, 7)).astype(dtype)
    y = (X[:, 0] + rng.normal(size=len(X)) > 0).astype(np.int32)
    return X, y


@contextlib.contextmanager
def _gpu_route(route):
    if route == "forced":
        with tuning_context(force_backend="mlx"):
            yield
    else:
        # Exercise ordinary device selection without allocating a huge fixture.
        with tuning_context(logreg_min_samples=1, logreg_min_work=1):
            yield


@pytest.mark.parametrize("route", ["forced", "default"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_a_validation_error(route, dtype, bad_value):
    X, y = _data(dtype)
    X[0, 0] = bad_value
    with _gpu_route(route), pytest.raises(ValueError) as caught:
        LogisticRegression().fit(X, y)
    assert not isinstance(caught.value, BackendExecutionError)
    assert not isinstance(caught.value, UnsupportedInputError)
    assert "NaN" in str(caught.value) or "infinity" in str(caught.value)


@pytest.mark.parametrize("route", ["forced", "default"])
def test_finite_float64_overflow_remains_a_capability_error(route):
    X, y = _data(np.float64)
    X[0, 0] = 1e300
    with _gpu_route(route), pytest.raises(UnsupportedInputError) as caught:
        LogisticRegression().fit(X, y)
    assert caught.value.reason == "float32-overflow"


@pytest.mark.parametrize("route", ["forced", "default"])
def test_negative_sample_weight_remains_a_capability_error(route):
    X, y = _data()
    weights = np.ones(len(y))
    weights[5] = -1.0
    with _gpu_route(route), pytest.raises(UnsupportedInputError) as caught:
        LogisticRegression().fit(X, y, sample_weight=weights)
    assert caught.value.reason == "negative-sample-weight"


@pytest.mark.parametrize("route", ["forced", "default"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prepared_input_is_private_and_fit_retains_no_input_alias(monkeypatch, route, dtype, fit_intercept):
    """Every GPU fit owns one prepared copy: centered with an intercept, raw without."""
    X, y = _data(dtype)
    original_X = X.copy()
    original = estimator_module._prepare_centered_logistic_input
    observations = []

    @contextlib.contextmanager
    def observe(array, labels, weights, *, native, center):
        with original(array, labels, weights, native=native, center=center) as prepared:
            owned, shift, finite, parts = prepared
            observations.append((
                finite, native, center, shift is None, parts.shape,
                np.shares_memory(np.asarray(owned), X),
            ))
            if center:
                np.testing.assert_array_equal(np.asarray(owned), array - np.asarray(shift))
            else:
                np.testing.assert_array_equal(np.asarray(owned), array)
            yield prepared

    monkeypatch.setattr(estimator_module, "_prepare_centered_logistic_input", observe)
    with _gpu_route(route):
        model = LogisticRegression(max_iter=200, fit_intercept=fit_intercept).fit(X, y)
    assert model.execution_backend_ == "mlx"
    blocks = (len(X) + 255) // 256
    assert observations == [
        (True, True, fit_intercept, not fit_intercept, (2, blocks, X.shape[1] + int(fit_intercept)), False)
    ]
    np.testing.assert_array_equal(X, original_X)
    query = original_X[:20].copy()
    expected = model.predict_proba(query)
    X[:] = 0
    np.testing.assert_array_equal(model.predict_proba(query), expected)


def test_private_converted_input_is_released_after_fit(monkeypatch):
    X, y = _data(np.float64)
    original = estimator_module._prepare_centered_logistic_input
    private_arrays = []

    @contextlib.contextmanager
    def observe(array, *args, **kwargs):
        assert array.dtype == np.float32 and not np.shares_memory(array, X)
        private_arrays.append(weakref.ref(array))
        with original(array, *args, **kwargs) as prepared:
            yield prepared

    monkeypatch.setattr(estimator_module, "_prepare_centered_logistic_input", observe)
    with _gpu_route("forced"):
        model = LogisticRegression(max_iter=200).fit(X, y)
    mx.synchronize()
    gc.collect()
    assert private_arrays
    assert all(reference() is None for reference in private_arrays)
    # The fitted model remains usable after its transient input owner is gone.
    assert np.isfinite(model.predict_proba(X[:10])).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dataframe_keeps_feature_names_and_input_values(dtype):
    pd = pytest.importorskip("pandas")
    X, y = _data(dtype)
    columns = [f"feature_{index}" for index in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    before = frame.copy(deep=True)
    with _gpu_route("forced"):
        model = LogisticRegression(max_iter=200).fit(frame, y)
    np.testing.assert_array_equal(model.feature_names_in_, columns)
    pd.testing.assert_frame_equal(frame, before)
    assert np.isfinite(model.predict_proba(frame.iloc[:10])).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_assume_finite_does_not_change_valid_fits(dtype):
    X, y = _data(dtype)
    with _gpu_route("forced"):
        checked = LogisticRegression(max_iter=200).fit(X, y)
        with sklearn_config_context(assume_finite=True):
            assumed = LogisticRegression(max_iter=200).fit(X, y)
    np.testing.assert_array_equal(assumed.coef_, checked.coef_)
    np.testing.assert_array_equal(assumed.intercept_, checked.intercept_)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_readonly_inputs_fit_without_changing_values_or_writeability(dtype):
    X, y = _data(dtype)
    original = X.copy()
    X.setflags(write=False)
    with _gpu_route("forced"):
        actual = LogisticRegression(max_iter=200).fit(X, y)
        reference = LogisticRegression(max_iter=200).fit(original, y)
    assert not X.flags.writeable
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(actual.coef_, reference.coef_)
    np.testing.assert_array_equal(actual.intercept_, reference.intercept_)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided", "negative-stride"])
def test_noncontiguous_inputs_match_the_same_contiguous_values(dtype, layout):
    X, y = _data(dtype)
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = np.repeat(X, 2, axis=1)[:, ::2]
    else:
        X, y = X[::-1], y[::-1]
    assert not X.flags.c_contiguous
    original = X.copy()
    with _gpu_route("forced"):
        actual = LogisticRegression(max_iter=200).fit(X, y)
        reference = LogisticRegression(max_iter=200).fit(original, y)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(actual.coef_, reference.coef_)
    np.testing.assert_array_equal(actual.intercept_, reference.intercept_)


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("size", [1, 31, 32, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049])
def test_gpu_copy_preserves_bits_and_owns_storage_at_tile_boundaries(size, labels):
    """Both uncentered copy kernels preserve every bit, including a negative zero."""
    X = np.arange(size, dtype=np.float32).reshape(1, size)
    X[0, 0] = -0.0
    expected_bits = X.view(np.uint32).copy()
    borrowed = mx.asarray(X) if hasattr(mx, "asarray") else mx.array(X)
    y = mx.zeros(1) if labels else None
    copied, center, finite, parts = prepare_centered(borrowed, y, center=False)
    mx.eval(copied, finite, *([parts] if labels else []))
    assert bool(finite)
    assert center is None and (parts is not None) is labels
    assert not np.shares_memory(np.asarray(copied), X)
    X.fill(-1)
    np.testing.assert_array_equal(np.asarray(copied).view(np.uint32), expected_bits)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("size", [257, 1025, 2049])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gpu_copy_checks_the_last_value_of_an_incomplete_tile(size, value, labels, center):
    X = np.zeros((3, size), dtype=np.float32)
    X[-1, -1] = value
    borrowed = mx.asarray(X) if hasattr(mx, "asarray") else mx.array(X)
    y = mx.zeros(3) if labels else None
    copied, shift, finite, parts = prepare_centered(borrowed, y, center=center)
    mx.eval(copied, finite)
    assert not bool(finite)
    if not center:
        np.testing.assert_array_equal(np.asarray(copied).view(np.uint32), X.view(np.uint32))
