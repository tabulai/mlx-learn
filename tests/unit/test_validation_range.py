"""Float32 capability checks preserve invalid-input and overflow distinctions."""

from __future__ import annotations

import numpy as np
import pytest

from mlxlearn._common.validation import FLOAT32_MAX, check_float32_range
from mlxlearn.exceptions import UnsupportedInputError


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_finite_boundary_is_accepted_without_mutating_input(dtype):
    X = np.array([[-FLOAT32_MAX, -1.0], [0.0, FLOAT32_MAX]], dtype=dtype)
    before = X.copy()
    check_float32_range(X)
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("value", [1e300, -1e300])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_finite_overflow_is_a_capability_error_for_each_layout(value, layout):
    X = np.zeros((4, 4), dtype=np.float64)
    X[2, 2] = value
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = X[::2, ::2]
    with pytest.raises(UnsupportedInputError, match="float32") as caught:
        check_float32_range(X)
    assert caught.value.reason == "float32-overflow"


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf, -np.inf], [np.nan, np.inf, -1.0, 2.0]])
def test_nonfinite_values_are_left_for_input_validation(values):
    check_float32_range(np.array(values, dtype=np.float64))


@pytest.mark.parametrize("value", [1e300, -1e300])
def test_finite_overflow_beside_nonfinite_values_is_still_detected(value):
    # Cross the scan's block boundary and use a noncontiguous view.
    X = np.zeros(2 * 65_537, dtype=np.float64)
    X[0], X[2], X[-2] = np.nan, np.inf, value
    with pytest.raises(UnsupportedInputError, match="float32"):
        check_float32_range(X[::2])


def test_float32_nonfinite_and_large_integer_inputs_do_not_raise_capability_errors():
    check_float32_range(np.array([np.nan, np.inf], dtype=np.float32))
    check_float32_range(np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max]))


def test_array_like_float64_overflow_is_detected():
    with pytest.raises(UnsupportedInputError, match="float32"):
        check_float32_range([[0.0, 1e300]])


def test_foreign_arrays_with_non_numpy_dtypes_are_converted_first():
    """An mlx.core.array's dtype has no .kind; the helper must not crash on it."""
    mx = pytest.importorskip("mlx.core")
    check_float32_range(mx.array(np.ones((4, 3), dtype=np.float32)))
    with pytest.raises(UnsupportedInputError):

        class Foreign:
            dtype = "not a numpy dtype"

            def __array__(self, dtype=None, copy=None):
                return np.array([[1e300, 0.0]])

        check_float32_range(Foreign())


def test_sparse_subclasses_outside_scipy_are_sparse():
    sparse = __import__("pytest").importorskip("scipy.sparse")
    from mlxlearn._common.validation import is_sparse

    class UserCSR(sparse.csr_matrix):
        pass

    assert is_sparse(UserCSR(np.eye(3)))
    assert not is_sparse(np.eye(3))
