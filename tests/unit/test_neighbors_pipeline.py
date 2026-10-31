"""Public neighbor boundaries preserve validation, ownership and resident voting."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier as SkClassifier
from sklearn.neighbors import KNeighborsRegressor as SkRegressor

from mlxlearn._common.config import tuning_context
from mlxlearn.exceptions import BackendExecutionError, UnsupportedInputError
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors


def _data():
    rng = np.random.default_rng(286)
    X = rng.normal(size=(301, 6)).astype(np.float32)
    X[:3] = X[0]
    y = np.where(X[:, 0] > 0, "positive", "negative")
    y[1] = "positive"
    return X, y, np.vstack([X[:8], rng.normal(size=(19, 6)).astype(np.float32)])


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly"])
def test_wide_float_fit_transfers_private_cast_and_keeps_input_isolation(monkeypatch, backend, layout):
    from mlxlearn.neighbors import _base

    source = np.arange(200, dtype=np.float64).reshape(25, 8).copy()
    if layout == "fortran":
        source = np.asfortranarray(source)
    X = source[::2, ::2] if layout == "strided" else source.view()
    if layout == "readonly":
        X.setflags(write=False)
    expected = np.array(X, dtype=np.float32, copy=True)
    converted = []
    original_cast = _base.cast_float32_checked

    def observe_cast(values):
        result = original_cast(values)
        converted.append(result[0])
        return result

    monkeypatch.setattr(_base, "cast_float32_checked", observe_cast)
    with tuning_context(force_backend=backend):
        model = NearestNeighbors(n_neighbors=1).fit(X)
    # The newly allocated cast is the owned host training buffer, rather than
    # an additional full-size allocation. It never aliases the user's source.
    assert len(converted) == 1 and model._fit_X is converted[0]
    assert not np.shares_memory(model._fit_X, source)
    source.fill(-1)
    np.testing.assert_array_equal(model._fit_X, expected)
    np.testing.assert_array_equal(model._index_.device_index.X, expected)
    np.testing.assert_array_equal(model.kneighbors(expected[2:4], return_distance=False), [[2], [3]])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.kneighbors(expected[2:4], return_distance=False), [[2], [3]])


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("features", [8, 64])
def test_host_bridge_owns_training_before_first_query(backend, features):
    X = np.arange(20 * features, dtype=np.float32).reshape(20, features).copy()
    original = X.copy()
    Q = X[3:5].copy()
    with tuning_context(force_backend=backend):
        model = NearestNeighbors(n_neighbors=1).fit(X)
    X.fill(-10)
    # Even the retained private host copy is independent of the device copy.
    # This catches accidental use of a borrowing constructor at the bridge.
    model._fit_X.fill(999)
    np.testing.assert_array_equal(model._index_.device_index.X, original)
    np.testing.assert_array_equal(model.kneighbors(Q, return_distance=False), [[3], [4]])


@pytest.mark.parametrize("features", [8, 64])
def test_gpu_training_host_mirror_is_lazy_and_independent(features):
    X = np.arange(40 * features, dtype=np.float32).reshape(40, features).copy()
    original = X.copy()
    Q = X[3:5].copy()
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=1).fit(X)
    assert model._index_._host_X is None
    X.fill(-10)
    np.testing.assert_array_equal(model.kneighbors(Q, return_distance=False), [[3], [4]])
    assert model._index_._host_X is None
    host = model._fit_X
    assert host is model._index_.X and host.flags.writeable
    assert not np.shares_memory(host, X)
    np.testing.assert_array_equal(host, original)
    host.fill(999)
    np.testing.assert_array_equal(model._index_.device_index.X, original)
    np.testing.assert_array_equal(model.kneighbors(Q, return_distance=False), [[3], [4]])


def test_lazy_gpu_host_mirror_supports_radius_pickle_and_refit():
    from sklearn.neighbors import NearestNeighbors as SkNeighbors

    X, _, Q = _data()
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=3, radius=2.0).fit(X)
    assert model._index_._host_X is None
    restored = pickle.loads(pickle.dumps(model))
    # The established host-only pickle fields remain present, and device state
    # is rebuilt from them rather than serialized as an MLX buffer.
    assert "_fit_X" in restored.__dict__
    assert restored._index_._device_index is None
    expected = SkNeighbors(n_neighbors=3, radius=2.0, algorithm="brute").fit(X)
    actual_radius = restored.radius_neighbors(Q, sort_results=True)
    expected_radius = expected.radius_neighbors(Q, sort_results=True)
    for actual, reference in zip(actual_radius[0], expected_radius[0], strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
    for actual, reference in zip(actual_radius[1], expected_radius[1], strict=True):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(restored.kneighbors(Q, return_distance=False),
                                  model.kneighbors(Q, return_distance=False))
    # A prior explicit/sklearn assignment must not mask a subsequent MLX fit.
    restored._fit_X = np.zeros_like(X)
    with tuning_context(force_backend="mlx"):
        restored.fit(X + 10)
    assert "_fit_X" not in restored.__dict__ and restored._index_._host_X is None
    np.testing.assert_array_equal(restored._fit_X, X + 10)


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_search_and_voting_stay_on_device(monkeypatch, backend, weights):
    X, y, Q = _data()
    with tuning_context(force_backend=backend):
        model = KNeighborsClassifier(n_neighbors=7, weights=weights).fit(X, y)
    reference = SkClassifier(n_neighbors=7, weights=weights, algorithm="brute").fit(X, y)
    original_query = type(model._index_).query_device
    queries = []

    def observe_query(index, *args, **kwargs):
        queries.append(kwargs["return_distance"])
        return original_query(index, *args, **kwargs)

    def reject_host_query(*args, **kwargs):
        raise AssertionError("classifier prediction materialized neighbors on the host")

    monkeypatch.setattr(type(model._index_), "query_device", observe_query)
    monkeypatch.setattr(type(model._index_), "query", reject_host_query)
    targets = model._y_device_
    np.testing.assert_array_equal(model.predict(Q), reference.predict(Q))
    actual = model.predict_proba(Q)
    np.testing.assert_allclose(actual, reference.predict_proba(Q), atol=1e-6, rtol=1e-6)
    assert actual.dtype == np.float64
    assert model._y_device_ is targets
    assert queries == [weights == "distance", weights == "distance"]


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_self_query_reuses_owned_device_training_data(monkeypatch, backend):
    X, _, _ = _data()
    with tuning_context(force_backend=backend):
        model = NearestNeighbors(n_neighbors=3).fit(X)
    original = type(model._index_).query

    def observe(index, Q, *args, **kwargs):
        assert Q is index.device_index.X
        assert kwargs["exclude_self"]
        return original(index, Q, *args, **kwargs)

    monkeypatch.setattr(type(model._index_), "query", observe)
    indices = model.kneighbors(return_distance=False)
    assert not np.any(indices == np.arange(len(X))[:, None])


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("target_kind", ["wide_range", "fine_precision"])
def test_regression_retains_float64_targets_and_owns_them(monkeypatch, backend, weights, target_kind):
    X = np.arange(4, dtype=np.float32)[:, None]
    if target_kind == "wide_range":
        y = np.array([1e100, 1e100, 2e100, 2e100])
    else:
        y = 2.0**40 + np.array([0.25, 0.75, 1.5, 3.0])
    Q = np.array([[0.4], [2.6]], dtype=np.float32)
    reference = SkRegressor(n_neighbors=2, algorithm="brute", weights=weights).fit(X, y.copy())
    with tuning_context(force_backend=backend):
        model = KNeighborsRegressor(n_neighbors=2, weights=weights).fit(X, y)
    original = type(model._index_).query
    requested_distances = []

    def observe(index, *args, **kwargs):
        requested_distances.append(kwargs["return_distance"])
        return original(index, *args, **kwargs)

    monkeypatch.setattr(type(model._index_), "query", observe)
    before = model.predict(Q)
    np.testing.assert_allclose(before, reference.predict(Q), rtol=1e-7)
    if weights == "uniform":
        np.testing.assert_array_equal(before, reference.predict(Q))
    y.fill(0)
    np.testing.assert_array_equal(model.predict(Q), before)
    assert before.dtype == np.float64
    assert requested_distances == [weights == "distance"] * 2


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_classifier_target_cache_rebuilds_after_pickle_and_refit(backend):
    X, y, Q = _data()
    with tuning_context(force_backend=backend):
        model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    expected = model.predict_proba(Q)
    restored = pickle.loads(pickle.dumps(model))
    assert "_y_device_" not in restored.__dict__
    np.testing.assert_array_equal(restored.predict_proba(Q), expected)
    y2 = np.full(len(y), "new label")
    old_targets = restored._y_device_
    with tuning_context(force_backend=backend):
        restored.fit(X, y2)
    assert restored._y_device_ is not old_targets
    y2[:] = "mutated"
    np.testing.assert_array_equal(restored.predict(Q), np.full(len(Q), "new label"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_fit_and_query_nonfinite_values_remain_validation_errors(dtype, value):
    X, _, Q = _data()
    X, Q = X.astype(dtype), Q.astype(dtype)
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=3).fit(X)
        Q[-1, -1] = value
        with pytest.raises(ValueError) as caught:
            model.kneighbors(Q)
        assert not isinstance(caught.value, (BackendExecutionError, UnsupportedInputError))
        X[-1, -1] = value
        with pytest.raises(ValueError) as caught:
            model.fit(X)
        assert not isinstance(caught.value, (BackendExecutionError, UnsupportedInputError))


def test_float64_overflow_is_a_capability_error_at_fit_and_query():
    X, _, Q = _data()
    X, Q = X.astype(np.float64), Q.astype(np.float64)
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=3).fit(X)
        Q[-1, -1] = 1e300
        with pytest.raises(UnsupportedInputError, match="float32"):
            model.kneighbors(Q)
        X[-1, -1] = 1e300
        with pytest.raises(UnsupportedInputError, match="float32"):
            model.fit(X)


@pytest.mark.parametrize("layout", ["fortran", "negative_stride", "readonly"])
def test_checked_float64_input_preserves_layout_values_and_predictions(layout):
    X, y, Q = _data()
    X, Q = X.astype(np.float64), Q.astype(np.float64)
    if layout == "fortran":
        X, Q = np.asfortranarray(X), np.asfortranarray(Q)
    elif layout == "negative_stride":
        X, y, Q = X[::-1], y[::-1], Q[::-1]
    else:
        X.setflags(write=False)
        Q.setflags(write=False)
    before_X, before_Q = X.copy(), Q.copy()
    with tuning_context(force_backend="mlx"):
        actual = KNeighborsClassifier(n_neighbors=7).fit(X, y).predict_proba(Q)
        reference = KNeighborsClassifier(n_neighbors=7).fit(before_X, y).predict_proba(before_Q)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(X, before_X)
    np.testing.assert_array_equal(Q, before_Q)


def test_dataframe_feature_names_survive_preparation():
    pd = pytest.importorskip("pandas")
    X, y, Q = _data()
    names = [f"feature_{j}" for j in range(X.shape[1])]
    frame = pd.DataFrame(X.astype(np.float64), columns=names)
    queries = pd.DataFrame(Q.astype(np.float64), columns=names)
    with tuning_context(force_backend="mlx"):
        model = KNeighborsClassifier(n_neighbors=7).fit(frame, y)
    np.testing.assert_array_equal(model.feature_names_in_, names)
    assert np.isfinite(model.predict_proba(queries)).all()
    with pytest.raises(ValueError, match="feature names"):
        model.predict_proba(queries.rename(columns={names[0]: "different"}))


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_classifier_distance_vote_keeps_distinct_squared_distances(backend):
    X = np.array([[1, 2.0**-11.5], [1, 0]], dtype=np.float32)
    y, Q = np.array([0, 1]), np.array([[0, 0]], dtype=np.float32)
    with tuning_context(force_backend=backend):
        model = KNeighborsClassifier(n_neighbors=2, weights="distance").fit(X, y)
    assert model.predict(Q)[0] == 1
    # Public probabilities have float32 accuracy and must remain normalized.
    np.testing.assert_allclose(model.predict_proba(Q).sum(axis=1), 1.0, atol=1e-7)


def test_classifier_rejects_invalid_weights_changed_after_fit():
    X, y, Q = _data()
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y).set_params(weights="bogus")
    with pytest.raises(ValueError, match="weights not recognized"):
        model.predict(Q)
    with pytest.raises(ValueError, match="weights not recognized"):
        model.predict_proba(Q)


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_classifier_rejects_all_zero_distance_weights(backend):
    # Values fit float32, but every squared separation overflows to infinity.
    X = np.array([[1e30], [2e30]], np.float32)
    with tuning_context(force_backend=backend):
        model = KNeighborsClassifier(n_neighbors=2, weights="distance").fit(X, [0, 1])
    Q = np.zeros((1, 1), np.float32)
    with pytest.raises(ValueError, match="zero weights"):
        model.predict(Q)
    with pytest.raises(ValueError, match="zero weights"):
        model.predict_proba(Q)
