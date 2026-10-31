"""The canonical-array query shortcut retains sklearn's input contract.

These tests exercise only host validation; they never fit or execute an index.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlxlearn.neighbors import NearestNeighbors, _base


def _model(features=4):
    model = NearestNeighbors(n_neighbors=3)
    model.n_features_in_ = features
    model.n_samples_fit_ = 20
    model._execution_backend_ = "mlx"
    return model


def _ordinary(model, X):
    prepared, finite = _base._prepare_neighbors_input(X)
    return _base.validate_data(
        model, prepared, accept_sparse=False, dtype=np.float32, order="C",
        ensure_all_finite=not finite, reset=False,
    )


def _observe_validation(monkeypatch):
    calls = []
    original = _base.validate_data

    def observe(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(_base, "validate_data", observe)
    return calls


@pytest.mark.parametrize("readonly", [False, True])
def test_canonical_queries_keep_values_identity_and_feature_checks(monkeypatch, readonly):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    values.setflags(write=not readonly)
    before = values.view(np.uint32).copy()
    calls = _observe_validation(monkeypatch)
    query, k, self_query = _model()._prepare_neighbor_query(values)
    assert query is values and k == 3 and not self_query
    assert calls == [{"skip_check_array": True, "reset": False}]
    np.testing.assert_array_equal(values.view(np.uint32), before)
    assert values.flags.writeable is (not readonly)


def test_extremes_subnormals_and_signed_zero_are_not_changed():
    info = np.finfo(np.float32)
    values = np.array([
        [info.max, -info.max, info.tiny, -info.tiny],
        [np.nextafter(np.float32(0), np.float32(1)), 0.0, -0.0, 1.0],
    ], dtype=np.float32)
    before = values.view(np.uint32).copy()
    with np.errstate(over="raise", invalid="raise"):
        result = _base._validate_neighbor_query(_model(), values)
    assert result is values
    np.testing.assert_array_equal(result.view(np.uint32), before)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_input_mutation_is_revalidated_and_nonfinite_errors_are_unchanged(monkeypatch, bad):
    values = np.zeros((3, 4), dtype=np.float32)
    model = _model()
    assert _base._validate_neighbor_query(model, values) is values
    values[-1, -1] = bad
    with pytest.raises(ValueError) as expected:
        _ordinary(model, values)
    calls = _observe_validation(monkeypatch)
    with pytest.raises(ValueError) as actual:
        model._prepare_neighbor_query(values)
    assert str(actual.value) == str(expected.value)
    assert not calls[0].get("skip_check_array", False)


@pytest.mark.parametrize("kind", [
    "fortran", "strided", "negative_stride", "float64", "integer", "list", "subclass",
    "non_native", "object", "dtype_metadata",
])
def test_other_valid_inputs_use_the_ordinary_conversion(monkeypatch, kind):
    class ArraySubclass(np.ndarray):
        pass

    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    variants = {
        "fortran": np.asfortranarray(values),
        "strided": values[::2],
        "negative_stride": values[::-1],
        "float64": values.astype(np.float64),
        "integer": values.astype(np.int64),
        "list": values.tolist(),
        "subclass": values.view(ArraySubclass),
        "non_native": values.astype(values.dtype.newbyteorder("S")),
        "object": values.astype(object),
        "dtype_metadata": values.view(np.dtype(np.float32, metadata={"unit": "meters"})),
    }
    source = variants[kind]
    expected = _ordinary(_model(), source)
    calls = _observe_validation(monkeypatch)
    actual, _, _ = _model()._prepare_neighbor_query(source)
    assert not calls[0].get("skip_check_array", False)
    assert type(actual) is np.ndarray and actual.dtype == np.float32
    assert actual.dtype.metadata is None
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("values", [
    np.empty((0, 4), dtype=np.float32),
    np.empty((3, 0), dtype=np.float32),
    np.zeros(4, dtype=np.float32),
    np.zeros((1, 2, 4), dtype=np.float32),
    np.array(2, dtype=np.float32),
    np.ones((3, 4), dtype=np.complex64) * (1 + 2j),
    np.full((3, 4), "invalid", dtype=object),
], ids=["no_rows", "no_features", "one_dimension", "three_dimensions", "scalar",
        "complex", "nonnumeric"])
def test_malformed_inputs_preserve_error_type_and_message(monkeypatch, values):
    with pytest.raises((ValueError, TypeError)) as expected:
        _ordinary(_model(), values)
    calls = _observe_validation(monkeypatch)
    with pytest.raises(type(expected.value)) as actual:
        _model()._prepare_neighbor_query(values)
    assert str(actual.value) == str(expected.value)
    assert not calls[0].get("skip_check_array", False)


def test_fast_path_keeps_the_fitted_feature_count_error():
    values = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ValueError) as expected:
        _ordinary(_model(), values)
    with pytest.raises(ValueError) as actual:
        _model()._prepare_neighbor_query(values)
    assert str(actual.value) == str(expected.value)


def test_dataframe_fit_metadata_still_warns_for_a_numpy_query():
    pd = pytest.importorskip("pandas")
    model = _model()
    _base.validate_data(model, pd.DataFrame(np.zeros((3, 4)), columns=list("abcd")), reset=True)
    query = np.zeros((2, 4), dtype=np.float32)
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        _ordinary(model, query)
    with warnings.catch_warnings(record=True) as actual:
        warnings.simplefilter("always")
        result = model._prepare_neighbor_query(query)[0]
    assert result is query
    assert len(actual) == len(expected) == 1
    assert actual[0].category is expected[0].category
    assert str(actual[0].message) == str(expected[0].message)


def test_dataframe_query_keeps_feature_name_validation():
    pd = pytest.importorskip("pandas")
    model = _model()
    _base.validate_data(model, pd.DataFrame(np.zeros((3, 4)), columns=list("abcd")), reset=True)
    query = pd.DataFrame(np.zeros((2, 4), dtype=np.float32), columns=list("abdc"))
    with pytest.raises(ValueError) as expected:
        _ordinary(model, query)
    with pytest.raises(ValueError) as actual:
        model._prepare_neighbor_query(query)
    assert str(actual.value) == str(expected.value)
