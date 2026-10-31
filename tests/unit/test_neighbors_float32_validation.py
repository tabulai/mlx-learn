"""Large float32 validation retains errors, layout support and input ownership."""

from __future__ import annotations

import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn.exceptions import BackendExecutionError, UnsupportedInputError
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.neighbors._base import _prepare_neighbors_input

_MIN_SIZE = 1 << 18
_SHAPE = (_MIN_SIZE // 32, 32)


@pytest.mark.parametrize("size, checked", [(_MIN_SIZE - 1, False), (_MIN_SIZE, True)])
def test_float32_finiteness_check_starts_at_size_boundary(size, checked):
    X = np.ones(size, dtype=np.float32)
    prepared, finite = _prepare_neighbors_input(X)
    assert prepared is X
    assert finite is checked


@pytest.mark.parametrize("kind", ["native", "non_native", "subclass"])
def test_only_plain_native_float32_arrays_claim_checked_finiteness(kind):
    class ArraySubclass(np.ndarray):
        pass

    X = np.ones(_SHAPE, dtype=np.float32)
    if kind == "non_native":
        X = X.astype(X.dtype.newbyteorder("S"))
    elif kind == "subclass":
        X = X.view(ArraySubclass)
    prepared, finite = _prepare_neighbors_input(X)
    assert prepared is X
    assert finite is (kind == "native")


def test_finite_extremes_and_cancellation_do_not_require_arithmetic_reduction():
    info = np.finfo(np.float32)
    X = np.empty(_MIN_SIZE, dtype=np.float32)
    X[: _MIN_SIZE // 2] = info.max
    X[_MIN_SIZE // 2 :] = -info.max
    X[-4:] = [info.tiny, np.nextafter(np.float32(0), np.float32(1)), 0.0, -0.0]
    before = X.view(np.uint32).copy()
    # Every element is finite even though summing same-sign blocks overflows.
    with np.errstate(over="raise", invalid="raise"):
        prepared, finite = _prepare_neighbors_input(X)
    assert prepared is X and finite is True
    np.testing.assert_array_equal(X.view(np.uint32), before)


@pytest.mark.parametrize(
    "bad_values", [(np.nan,), (np.inf,), (-np.inf,), (np.inf, -np.inf)],
    ids=["nan", "positive_infinity", "negative_infinity", "opposite_infinities"],
)
def test_nonfinite_tail_still_raises_validation_errors_at_fit_and_query(bad_values):
    X = np.ones(_SHAPE, dtype=np.float32)
    X.reshape(-1)[-len(bad_values) :] = bad_values
    with np.errstate(over="raise", invalid="raise"):
        prepared, finite = _prepare_neighbors_input(X)
    assert prepared is X and finite is False

    with tuning_context(force_backend="cpu"):
        model = NearestNeighbors(n_neighbors=1).fit(np.zeros((2, 32), dtype=np.float32))
        for operation in (model.kneighbors, model.fit):
            with pytest.raises(ValueError) as caught:
                operation(X)
            assert not isinstance(caught.value, (BackendExecutionError, UnsupportedInputError))


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly"])
def test_large_float32_layouts_are_checked_and_fit_keeps_owned_storage(layout):
    width = 64 if layout == "strided" else 32
    source = np.random.default_rng(319).normal(size=(_SHAPE[0], width)).astype(np.float32)
    if layout == "fortran":
        source = np.asfortranarray(source)
    X = source[:, ::2] if layout == "strided" else source.view()
    if layout == "readonly":
        X.setflags(write=False)
    expected = X.copy()
    prepared, finite = _prepare_neighbors_input(X)
    assert prepared is X and finite is True

    with tuning_context(force_backend="cpu"):
        model = NearestNeighbors(n_neighbors=1).fit(X)
    # The checked flag must not be confused with ownership of a private cast.
    source.fill(-100)
    assert not np.shares_memory(model._fit_X, source)
    np.testing.assert_array_equal(model._fit_X, expected)
    np.testing.assert_array_equal(model._index_.device_index.X, expected)
    np.testing.assert_array_equal(model.kneighbors(expected[3:5], return_distance=False), [[3], [4]])

    # Strides and read-only views must also expose invalid values to the check.
    source[-1, -2 if layout == "strided" else -1] = np.nan
    prepared, finite = _prepare_neighbors_input(X)
    assert prepared is X and finite is False
    assert X.flags.writeable is (layout != "readonly")


@pytest.mark.parametrize("estimator", [KNeighborsClassifier, KNeighborsRegressor])
@pytest.mark.parametrize("target_kind", ["nan", "infinity", "negative_infinity", "short", "missing"])
def test_checked_features_do_not_disable_target_validation(estimator, target_kind):
    X = np.ones(_SHAPE, dtype=np.float32)
    y = np.zeros(len(X))
    if target_kind == "short":
        y = y[:-1]
    elif target_kind == "missing":
        y = None
    else:
        y[-1] = {"nan": np.nan, "infinity": np.inf, "negative_infinity": -np.inf}[target_kind]
    with tuning_context(force_backend="cpu"), pytest.raises(ValueError):
        estimator(n_neighbors=1).fit(X, y)


def test_checked_features_retain_classification_and_shape_validation():
    X = np.ones(_SHAPE, dtype=np.float32)
    with tuning_context(force_backend="cpu"):
        with pytest.raises(ValueError, match="continuous"):
            KNeighborsClassifier(n_neighbors=1).fit(X, np.linspace(0.0, 1.0, len(X)))
        with pytest.raises(ValueError, match="2D"):
            NearestNeighbors(n_neighbors=1).fit(X.reshape(-1))
        model = NearestNeighbors(n_neighbors=1).fit(np.zeros((2, 32), dtype=np.float32))
        with pytest.raises(ValueError, match="features"):
            model.kneighbors(X.reshape(-1, 16))
