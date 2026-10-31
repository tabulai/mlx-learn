"""The small automatic fit shortcut preserves validation and owned state."""

from __future__ import annotations

import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, _base


@pytest.fixture(autouse=True)
def _cpu_only_auto(monkeypatch):
    from mlxlearn._common import base

    monkeypatch.setattr(base, "resolve_device", lambda requested: "cpu")


def _observe_validation(monkeypatch):
    calls = []
    original = _base.validate_data

    def observe(estimator, X, *args, **kwargs):
        calls.append((X, kwargs))
        return original(estimator, X, *args, **kwargs)

    monkeypatch.setattr(_base, "validate_data", observe)
    return calls


def _ordinary_fit(model, X):
    prepared, finite = _base._prepare_neighbors_input(X)
    return _base.validate_data(
        model, prepared, accept_sparse=False, dtype=np.float32, order="C",
        ensure_min_samples=1, ensure_all_finite=not finite, reset=True,
    )


@pytest.mark.parametrize("readonly", [False, True])
def test_small_canonical_fit_skips_redundant_validation_and_owns_source(monkeypatch, readonly):
    source = np.arange(64, dtype=np.float32).reshape(16, 4).copy()
    X = source.view()
    X.setflags(write=not readonly)
    expected = X.copy()
    calls = _observe_validation(monkeypatch)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    assert len(calls) == 1 and calls[0][0] is X
    assert calls[0][1] == {"skip_check_array": True, "reset": True}
    assert model.n_features_in_ == 4 and model.n_samples_fit_ == 16
    assert X.flags.writeable is (not readonly)
    assert not np.shares_memory(model._index_.training_data, source)
    source.fill(-100)
    model._fit_X.fill(1000)
    np.testing.assert_array_equal(model._index_.training_data, expected)
    np.testing.assert_array_equal(model.kneighbors(expected[[3]], return_distance=False), [[3]])


def test_small_canonical_refit_resets_dataframe_feature_names_and_feature_count(monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(np.zeros((8, 4), dtype=np.float32), columns=list("abcd"))
    model = NearestNeighbors(n_neighbors=1).fit(frame)
    np.testing.assert_array_equal(model.feature_names_in_, list("abcd"))
    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    calls = _observe_validation(monkeypatch)
    model.fit(X)
    assert calls[0][1] == {"skip_check_array": True, "reset": True}
    assert model.n_features_in_ == 3 and model.n_samples_fit_ == 10
    assert not hasattr(model, "feature_names_in_")
    with pytest.raises(ValueError, match="features"):
        model.kneighbors(np.zeros((1, 4), dtype=np.float32))


def test_small_fit_preserves_extremes_subnormals_and_signed_zero():
    info = np.finfo(np.float32)
    X = np.array([
        [info.max, -info.max, info.tiny, -info.tiny],
        [np.nextafter(np.float32(0), np.float32(1)), 0.0, -0.0, 1.0],
    ], dtype=np.float32)
    expected_bits = X.view(np.uint32).copy()
    with np.errstate(over="raise", invalid="raise"):
        model = NearestNeighbors(n_neighbors=1).fit(X)
    np.testing.assert_array_equal(model._index_.training_data.view(np.uint32), expected_bits)
    np.testing.assert_array_equal(X.view(np.uint32), expected_bits)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_small_fit_uses_ordinary_validation_errors(monkeypatch, bad):
    X = np.zeros((8, 4), dtype=np.float32)
    X[-1, -1] = bad
    with pytest.raises(ValueError) as expected:
        _ordinary_fit(NearestNeighbors(), X)
    calls = _observe_validation(monkeypatch)
    with pytest.raises(type(expected.value)) as actual:
        NearestNeighbors().fit(X)
    assert str(actual.value) == str(expected.value)
    assert len(calls) == 1 and not calls[0][1].get("skip_check_array", False)


@pytest.mark.parametrize("X", [
    np.empty((0, 4), dtype=np.float32),
    np.empty((3, 0), dtype=np.float32),
    np.zeros(4, dtype=np.float32),
    np.zeros((1, 2, 4), dtype=np.float32),
    np.array(2, dtype=np.float32),
], ids=["no_rows", "no_features", "one_dimension", "three_dimensions", "scalar"])
def test_malformed_small_fits_preserve_error_type_and_message(monkeypatch, X):
    with pytest.raises((ValueError, TypeError)) as expected:
        _ordinary_fit(NearestNeighbors(), X)
    calls = _observe_validation(monkeypatch)
    with pytest.raises(type(expected.value)) as actual:
        NearestNeighbors().fit(X)
    assert str(actual.value) == str(expected.value)
    assert len(calls) == 1 and not calls[0][1].get("skip_check_array", False)


@pytest.mark.parametrize("kind", [
    "fortran", "strided", "negative_stride", "integer", "list", "subclass",
    "non_native", "dtype_metadata",
])
def test_noncanonical_small_fits_keep_ordinary_conversion(monkeypatch, kind):
    class ArraySubclass(np.ndarray):
        pass

    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    variants = {
        "fortran": np.asfortranarray(values),
        "strided": values[::2],
        "negative_stride": values[::-1],
        "integer": values.astype(np.int64),
        "list": values.tolist(),
        "subclass": values.view(ArraySubclass),
        "non_native": values.astype(values.dtype.newbyteorder("S")),
        "dtype_metadata": values.view(np.dtype(np.float32, metadata={"unit": "meters"})),
    }
    X = variants[kind]
    expected = _ordinary_fit(NearestNeighbors(), X)
    calls = _observe_validation(monkeypatch)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    assert len(calls) == 1 and not calls[0][1].get("skip_check_array", False)
    stored = model._index_.training_data
    assert type(stored) is np.ndarray and stored.dtype == np.float32
    assert stored.dtype.metadata is None and stored.flags.c_contiguous
    np.testing.assert_array_equal(stored, expected)


def test_validated_float64_cast_remains_owned_after_small_fit():
    X = np.arange(64, dtype=np.float64).reshape(16, 4).copy()
    expected = X.astype(np.float32)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    assert not np.shares_memory(model._index_.training_data, X)
    X.fill(-100)
    np.testing.assert_array_equal(model._index_.training_data, expected)


@pytest.mark.parametrize("shape", [(2049, 4), (16, 65), (50000, 32), (4096, 128)])
def test_large_or_wide_fits_do_not_enter_the_extra_canonical_finiteness_scan(monkeypatch, shape):
    X = np.random.default_rng(594).normal(size=shape).astype(np.float32)

    def reject_extra_scan(values):
        raise AssertionError("large/wide fit entered the small canonical validation check")

    monkeypatch.setattr(_base, "_canonical_float32_matrix", reject_extra_scan)
    calls = _observe_validation(monkeypatch)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    assert len(calls) == 1 and not calls[0][1].get("skip_check_array", False)
    assert model.n_features_in_ == shape[1] and model.n_samples_fit_ == shape[0]


@pytest.mark.parametrize("estimator", [KNeighborsClassifier, KNeighborsRegressor])
def test_supervised_small_fits_keep_target_validation(monkeypatch, estimator):
    X = np.arange(64, dtype=np.float32).reshape(16, 4)
    y = np.arange(len(X)) % 2
    calls = _observe_validation(monkeypatch)
    estimator(n_neighbors=1).fit(X, y)
    assert len(calls) == 1 and not calls[0][1].get("skip_check_array", False)
    with pytest.raises(ValueError):
        estimator(n_neighbors=1).fit(X, y[:-1])


def test_explicit_cpu_fit_keeps_ordinary_validation(monkeypatch):
    X = np.arange(64, dtype=np.float32).reshape(16, 4)
    calls = _observe_validation(monkeypatch)
    with tuning_context(force_backend="cpu"):
        model = NearestNeighbors(n_neighbors=1).fit(X)
    assert model.execution_policy_ == "fixed"
    assert len(calls) == 1 and not calls[0][1].get("skip_check_array", False)
