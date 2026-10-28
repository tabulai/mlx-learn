"""Input preparation preserves NumPy layout, dtype and container contracts."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn.exceptions import UnsupportedInputError
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module


@pytest.mark.parametrize("dtype", [np.float64, np.dtype(">f8"), np.longdouble])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "negative-stride"])
def test_float64_preparation_owns_contiguous_storage_for_each_layout(dtype, layout):
    X = np.linspace(-2, 2, 60).reshape(10, 6).astype(dtype)
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = np.repeat(X, 2, axis=1)[:, ::2]
    elif layout == "negative-stride":
        X = X[::-1]
    X.setflags(write=False)
    before = X.copy()
    prepared, finite = estimator_module._prepare_logistic_input(X, return_finite=True)
    assert finite
    assert prepared.dtype == np.dtype(np.float32)
    assert prepared.flags.c_contiguous
    assert not np.shares_memory(prepared, X)
    assert not X.flags.writeable
    np.testing.assert_array_equal(prepared, X.astype(np.float32))
    np.testing.assert_array_equal(X, before)


def test_float32_preparation_leaves_readonly_strided_storage_for_validation():
    X = np.arange(120, dtype=np.float32).reshape(10, 12)[:, ::2]
    X.setflags(write=False)
    prepared, finite = estimator_module._prepare_logistic_input(X, return_finite=True)
    assert prepared is X
    assert not finite
    assert not prepared.flags.writeable
    assert not prepared.flags.c_contiguous


@pytest.mark.parametrize("dtype", [np.float64, np.dtype(">f8"), np.longdouble])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_preparation_preserves_nonfinite_values_for_ordinary_validation(dtype, value):
    X = np.array([[0, value]], dtype=dtype)
    prepared, finite = estimator_module._prepare_logistic_input(X, return_finite=True)
    assert not finite
    np.testing.assert_array_equal(prepared, X.astype(np.float32))


@pytest.mark.parametrize("dtype", [np.float64, np.dtype(">f8"), np.longdouble])
def test_preparation_reports_finite_overflow_beside_nan(dtype):
    X = np.array([[np.nan, 1e300]], dtype=dtype)
    with pytest.raises(UnsupportedInputError) as caught:
        estimator_module._prepare_logistic_input(X, return_finite=True)
    assert caught.value.reason == "float32-overflow"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matrix_subclass_is_rejected_before_upload(monkeypatch, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        X = np.matrix([[0, 1], [1, 0], [2, 1], [1, 2]], dtype=dtype)
    y = np.array([0, 1, 1, 0])

    def unexpected_upload(*args, **kwargs):
        pytest.fail("matrix validation must reject the input before GPU upload")

    monkeypatch.setattr(estimator_module, "_prepare_centered_logistic_input", unexpected_upload)
    assert estimator_module._prepare_logistic_input(X) is X
    with tuning_context(force_backend="mlx"), pytest.raises(TypeError, match="np.matrix"):
        LogisticRegression().fit(X, y)
