"""Automatic KNN dispatch preserves fitted policy, ownership and exact results."""

from __future__ import annotations

import pickle
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import mlxlearn
from mlxlearn._common.config import config_context, tuning_context
from mlxlearn.exceptions import BackendExecutionError
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors


@pytest.fixture
def cpu_only_auto(monkeypatch):
    """Exercise the automatic host route without initializing a GPU device."""
    from mlxlearn._common import base

    monkeypatch.setattr(base, "resolve_device", lambda requested: "cpu")


def _exact_neighbors(X, Q, k, *, exclude_self=False):
    # Integer/dyadic fixtures make these direct float64 sums exact. The oracle
    # does not use a sklearn solver or share the implementation's selection.
    train, queries = X.astype(np.float64), Q.astype(np.float64)
    squared = np.empty((len(Q), len(X)), dtype=np.float64)
    for start in range(0, len(Q), 16):
        delta = queries[start:start + 16, None] - train
        squared[start:start + 16] = np.sum(delta * delta, axis=2)
    if exclude_self:
        np.fill_diagonal(squared, np.inf)
    order = np.stack([np.lexsort((np.arange(len(X)), row))[:k] for row in squared])
    return np.sqrt(np.take_along_axis(squared, order, axis=1)), order


def _reject_device_search(monkeypatch, model):
    def reject(*args, **kwargs):
        raise AssertionError("automatic CPU query entered the MLX search path")

    monkeypatch.setattr(type(model._index_), "query_device", reject)


@pytest.mark.parametrize("estimator,query", [
    (NearestNeighbors, "kneighbors"),
    (KNeighborsClassifier, "predict"),
    (KNeighborsRegressor, "predict"),
])
@pytest.mark.parametrize("serialization", ["pickle", "deepcopy"])
def test_unfitted_serialization_does_not_create_fitted_state(estimator, query, serialization):
    model = estimator()
    original = vars(model).copy()
    assert not any(name.endswith("_") for name in original)
    restored = (pickle.loads(pickle.dumps(model)) if serialization == "pickle"
                else deepcopy(model))
    for current in (model, restored):
        assert vars(current) == original
        assert "_knn_query_state_" not in vars(current)
        for attribute in ("execution_backend_", "execution_policy_", "last_query_backend_"):
            assert not hasattr(current, attribute)
        with pytest.raises(NotFittedError):
            check_is_fitted(current)
        with pytest.raises(NotFittedError):
            getattr(current, query)([[0.0]])


def test_automatic_policy_and_query_diagnostics_are_distinct(cpu_only_auto):
    model = NearestNeighbors(n_neighbors=2)
    for attribute in ("execution_backend_", "execution_policy_", "last_query_backend_"):
        assert not hasattr(model, attribute)
    X = np.arange(64, dtype=np.float32).reshape(16, 4)
    model.fit(X)
    assert model.execution_policy_ == "auto"
    assert model.execution_backend_ == "cpu"
    assert not hasattr(model, "last_query_backend_")
    model.kneighbors(X[:1])
    assert model.last_query_backend_ == "cpu"
    event = mlxlearn.get_last_backend_event()
    assert event.backend == "cpu" and not event.fallback


def test_automatic_fit_policy_ignores_later_global_backend_changes(cpu_only_auto, monkeypatch):
    X = np.arange(64, dtype=np.float32).reshape(16, 4)
    Q = X[[2, 7]].copy()
    model = NearestNeighbors(n_neighbors=2).fit(X)
    _reject_device_search(monkeypatch, model)
    expected = _exact_neighbors(X, Q, 2)
    for device, forced in (("gpu", "mlx"), ("cpu", "cpu"), ("auto", "")):
        with config_context(device=device), tuning_context(force_backend=forced):
            actual = model.kneighbors(Q)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert model.execution_policy_ == "auto"
        assert model.execution_backend_ == "cpu"
        assert model.last_query_backend_ == "cpu"


@pytest.mark.parametrize("device,forced,backend", [
    ("gpu", "", "mlx"),
    ("cpu", "", "cpu"),
    ("auto", "mlx", "mlx"),
    ("auto", "cpu", "cpu"),
    ("cpu", "mlx", "mlx"),
    ("gpu", "cpu", "cpu"),
])
def test_explicit_backend_choices_are_fixed_even_for_tiny_fits(monkeypatch, device, forced, backend):
    from mlxlearn._common import base
    from mlxlearn.neighbors import _base

    # Inspect fit selection without constructing a CPU or GPU device index.
    monkeypatch.setattr(base, "resolve_device", lambda requested: requested)
    calls = []

    def capture_index(X, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(X=X.copy(), backend=kwargs["backend"])

    monkeypatch.setattr(_base, "build_index", capture_index)
    with config_context(device=device), tuning_context(force_backend=forced):
        model = NearestNeighbors(n_neighbors=1).fit(np.zeros((8, 2), dtype=np.float32))
    assert model.execution_backend_ == backend
    assert model.execution_policy_ == "fixed"
    assert len(calls) == 1 and calls[0]["backend"] == backend
    assert not calls[0].get("_auto", False)


@pytest.mark.parametrize("n,rows,features,k,expected", [
    (2048, 256, 64, 8, "cpu"),
    (8192, 1, 32, 5, "cpu"),
    (16384, 4, 32, 8, "mlx"),
    (8192, 256, 32, 5, "mlx"),
    (50000, 256, 32, 5, "mlx"),
    (4096, 128, 128, 5, "mlx"),
    (2048, 32, 1024, 5, "mlx"),
])
def test_automatic_route_preserves_established_large_batches(n, rows, features, k, expected):
    from mlxlearn._kernels.distance import build_index

    # The host fit can inspect the fitted policy without creating GPU storage.
    index = build_index(
        np.zeros((n, features), dtype=np.float32), backend="cpu", _auto=True,
        _gpu_available=True,
    )
    assert index.backend_for_query(rows, k) == expected
    assert index._device_index is None


def test_missing_gpu_keeps_large_automatic_queries_on_cpu():
    from mlxlearn._kernels.distance import build_index

    index = build_index(
        np.zeros((8192, 32), dtype=np.float32), backend="cpu", _auto=True,
        _gpu_available=False,
    )
    assert index.backend_for_query(256, 5) == "cpu"
    assert index._device_index is None


@pytest.mark.parametrize("n,rows,features,expected", [
    (256, 64, 32, "cpu"),
    (256, 128, 32, "mlx"),
    (512, 32, 32, "cpu"),
    (512, 64, 32, "mlx"),
    (1024, 32, 32, "mlx"),
    (1024, 64, 32, "mlx"),
    (2048, 32, 32, "mlx"),
    (4096, 4, 32, "cpu"),
    (4097, 4, 32, "mlx"),
    (16384, 1, 32, "cpu"),
])
def test_gaussian_batch_policy_keeps_large_work_on_gpu(n, rows, features, expected):
    from mlxlearn._kernels.distance import build_index

    rng = np.random.default_rng(184)
    train = rng.normal(size=(n, features)).astype(np.float32)
    queries = rng.normal(size=(rows, features)).astype(np.float32)
    index = build_index(train, backend="cpu", _auto=True, _gpu_available=True)
    assert index.backend_for_query(rows, 5, queries) == expected
    assert index._device_index is None


@pytest.mark.parametrize("kind,expected", [
    ("sparse", "cpu"), ("dense", "mlx"), ("below_sparse_threshold", "mlx"),
    ("fractional_train_tail", "mlx"), ("large_train_tail", "mlx"),
    ("fractional_queries", "mlx"), ("large_queries", "mlx"),
])
def test_wide_cpu_policy_requires_sparse_bounded_integers_in_train_and_queries(kind, expected):
    from mlxlearn._kernels.distance import build_index

    train = np.random.default_rng(651).integers(1, 17, size=(512, 64)).astype(np.float32)
    if kind != "dense":
        train[:, :16] = 0  # Exactly the required 25% zeros.
    if kind == "below_sparse_threshold":
        train[-1, 0] = 1
    elif kind == "fractional_train_tail":
        train[-1, -1] = 0.5
    elif kind == "large_train_tail":
        train[-1, -1] = 1024
    queries = train[:16].copy()
    if kind == "fractional_queries":
        queries[-1, -1] += 0.5
    elif kind == "large_queries":
        queries[-1, -1] = 1024
    index = build_index(train, backend="cpu", _auto=True, _gpu_available=True)
    assert index.backend_for_query(len(queries), 5, queries) == expected
    assert index._device_index is None


def test_dense_rejection_sample_keeps_globally_sparse_integer_data_on_gpu():
    from mlxlearn._kernels.distance import build_index

    train = np.zeros((512, 64), np.float32)
    train[:16] = 1
    queries = train[:16].copy()
    assert np.count_nonzero(train) < 0.75 * train.size
    index = build_index(train, backend="cpu", _auto=True, _gpu_available=True)
    # Sampling may conservatively decline the CPU route. It never certifies
    # sparsity or integer arithmetic without checking all training/query data.
    assert index.backend_for_query(len(queries), 5, queries) == "mlx"
    assert index._device_index is None


def test_concentrated_feature_ranges_are_a_bounded_cpu_policy_exception():
    from mlxlearn._kernels.distance import build_index

    train = np.random.default_rng(221).normal(size=(512, 32)).astype(np.float32)
    train[:, 24:] *= 1024
    index = build_index(train, backend="cpu", _auto=True, _gpu_available=True)
    queries = train[:128].copy()
    assert index.backend_for_query(128, 5, queries) == "cpu"
    assert index.backend_for_query(129, 5, np.vstack([queries, queries[:1]])) == "mlx"
    assert index._device_index is None


def test_cpu_search_owns_training_and_ignores_mutable_sklearn_mirror(cpu_only_auto, monkeypatch):
    X = np.arange(96, dtype=np.float32).reshape(24, 4)
    original = X.copy()
    Q = original[[3, 11]].copy()
    model = NearestNeighbors(n_neighbors=3).fit(X)
    _reject_device_search(monkeypatch, model)
    X.fill(-100)
    model._fit_X.fill(1000)
    expected = _exact_neighbors(original, Q, 3)
    for current in (model, pickle.loads(pickle.dumps(model))):
        actual = current.kneighbors(Q)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert current.execution_policy_ == "auto"
        assert current.last_query_backend_ == "cpu"


def test_self_queries_exclude_only_the_matching_row(cpu_only_auto, monkeypatch):
    X = np.array([[0, 0], [0, 0], [1, 0], [-1, 0]], dtype=np.float32)
    model = NearestNeighbors(n_neighbors=2).fit(X)
    _reject_device_search(monkeypatch, model)
    expected = _exact_neighbors(X, X, 2, exclude_self=True)
    actual = model.kneighbors()
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert model._index_._device_index is None


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("labels", [
    np.array(["zebra", "ant", "zebra", "ant"]),
    np.array([97, -11, 97, -11]),
])
def test_cpu_classifier_votes_without_device_neighbors_or_labels(
    cpu_only_auto, monkeypatch, weights, labels,
):
    X = np.array([[0, 0], [0, 0], [1, 0], [2, 0]], dtype=np.float32)
    Q = X[[0, 2]].copy()
    y = labels.copy()
    model = KNeighborsClassifier(n_neighbors=3, weights=weights).fit(X, y)
    _reject_device_search(monkeypatch, model)
    y[:] = y[-1]
    expected = (np.array([[1 / 3, 2 / 3], [1 / 3, 2 / 3]]) if weights == "uniform"
                else np.array([[0.5, 0.5], [0.0, 1.0]]))
    np.testing.assert_allclose(model.predict_proba(Q), expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(model.predict(Q), model.classes_[np.argmax(expected, axis=1)])
    assert model.last_query_backend_ == "cpu"
    assert getattr(model, "_y_device_", None) is None
    assert model._index_._device_index is None


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("values", [
    2.0**40 + np.array([0.25, 0.75, 1.5, 3.0]),
    np.array([1e100, 1e100, 2e100, 2e100]),
])
def test_cpu_regression_preserves_owned_float64_targets(cpu_only_auto, monkeypatch, weights, values):
    X = np.arange(4, dtype=np.float32)[:, None]
    Q = np.array([[0.5], [1.5], [2.5]], dtype=np.float32)
    y = values.copy()
    model = KNeighborsRegressor(n_neighbors=2, weights=weights).fit(X, y)
    _reject_device_search(monkeypatch, model)
    y.fill(-1)
    expected = (values[:-1] + values[1:]) / 2
    actual = model.predict(Q)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    assert model.last_query_backend_ == "cpu"


def test_refit_clears_policy_and_last_query_then_owns_the_new_data(cpu_only_auto):
    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    model.kneighbors(X[:1])
    model.set_params(n_neighbors=0)
    with pytest.raises(ValueError):
        model.fit(X)
    for attribute in ("execution_backend_", "execution_policy_", "last_query_backend_"):
        assert not hasattr(model, attribute)
    model.set_params(n_neighbors=1).fit(X + 100)
    assert model.execution_policy_ == "auto"
    assert not hasattr(model, "last_query_backend_")
    np.testing.assert_array_equal(model.kneighbors(X[[3]] + 100, return_distance=False), [[3]])


def test_automatic_cpu_failure_is_not_retried_by_another_backend(cpu_only_auto, monkeypatch):
    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    model = NearestNeighbors(n_neighbors=1).fit(X)
    _reject_device_search(monkeypatch, model)

    def fail(*args, **kwargs):
        raise RuntimeError("injected automatic CPU search failure")

    monkeypatch.setattr(type(model._index_), "query", fail)
    with pytest.raises(BackendExecutionError, match="injected automatic CPU search failure"):
        model.kneighbors(X[:1])
    assert model.execution_backend_ == "cpu" and model.execution_policy_ == "auto"


def test_patched_small_auto_queries_use_owned_search_not_sklearn(cpu_only_auto):
    import sklearn.neighbors

    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    mlxlearn.patch_sklearn()
    with config_context(fallback_policy="raise"):
        model = sklearn.neighbors.NearestNeighbors(n_neighbors=1).fit(X)
        np.testing.assert_array_equal(model.kneighbors(X[[3]], return_distance=False), [[3]])
    assert model.execution_policy_ == "auto"
    assert model.last_query_backend_ == "cpu"
    assert not any(event.fallback for event in mlxlearn.get_backend_diagnostics())


def test_capability_fallback_keeps_a_fixed_sklearn_policy(cpu_only_auto):
    import sklearn.neighbors

    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    mlxlearn.patch_sklearn()
    with config_context(fallback_policy="silent"):
        model = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(X)
    assert model.execution_policy_ == "fixed"
    assert model.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(model.kneighbors(X[[3]], return_distance=False), [[3]])
    assert model.last_query_backend_ == "sklearn"


def test_regressor_capability_fallback_reports_sklearn_prediction(cpu_only_auto):
    import sklearn.neighbors

    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    y = np.arange(8, dtype=np.float64)
    mlxlearn.patch_sklearn()
    with config_context(fallback_policy="silent"):
        model = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=1, algorithm="kd_tree",
        ).fit(X, y)
    assert model.execution_policy_ == "fixed" and model.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(model.predict(X[[3]]), [3.0])
    assert model.last_query_backend_ == "sklearn"
    assert mlxlearn.get_last_backend_event().backend == "sklearn"


@pytest.mark.gpu
def test_one_automatic_fit_can_switch_cpu_gpu_cpu_without_changing_policy():
    rng = np.random.default_rng(1254)
    X = rng.integers(-128, 128, size=(8192, 32)).astype(np.float32) / 16
    Q = rng.integers(-128, 128, size=(256, 32)).astype(np.float32) / 16
    with config_context(device="auto"):
        model = NearestNeighbors(n_neighbors=5).fit(X)
    initial_backend = model.execution_backend_
    for query, expected_backend in ((Q[:1], "cpu"), (Q, "mlx"), (Q[-1:], "cpu")):
        expected = _exact_neighbors(X, query, 5)
        actual = model.kneighbors(query)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert model.execution_backend_ == initial_backend
        assert model.execution_policy_ == "auto"
        assert model.last_query_backend_ == expected_backend
        event = mlxlearn.get_last_backend_event()
        assert event.backend == expected_backend and not event.fallback


@pytest.mark.gpu
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_cpu_fitted_classifier_keeps_estimator_attributes_and_rebuilds_gpu_caches(weights):
    rng = np.random.default_rng(985)
    X = rng.integers(-32, 32, size=(512, 32)).astype(np.float32) / 8
    Q = rng.integers(-32, 32, size=(257, 32)).astype(np.float32) / 8
    y = np.where(X[:, 0] >= 0, "positive", "negative")
    model = KNeighborsClassifier(n_neighbors=5, weights=weights).fit(X, y)
    assert model.execution_backend_ == "cpu" and model.execution_policy_ == "auto"
    assert model._y_device_ is None and model._index_._device_index is None
    before = vars(model).copy()

    expected_cpu = model.predict_proba(Q[:1])
    assert model.last_query_backend_ == "cpu" and model._y_device_ is None
    expected_gpu = model.predict_proba(Q)
    assert model.last_query_backend_ == "mlx" and model._y_device_ is not None
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[name] is value for name, value in before.items())
    np.testing.assert_allclose(expected_gpu[:1], expected_cpu, rtol=1e-6, atol=1e-7)

    # The pickle keeps portable host state and the fitted policy, without
    # embedding the uploaded target cache or a device search index.
    restored = pickle.loads(pickle.dumps(model))
    assert restored.execution_policy_ == "auto" and restored.execution_backend_ == "cpu"
    assert restored._y_device_ is None and restored._index_._device_index is None
    before_restored = vars(restored).copy()
    np.testing.assert_array_equal(restored.predict_proba(Q[:1]), expected_cpu)
    assert restored.last_query_backend_ == "cpu" and restored._y_device_ is None
    assert restored._index_._device_index is None
    np.testing.assert_array_equal(restored.predict_proba(Q), expected_gpu)
    targets = restored._y_device_
    assert restored.last_query_backend_ == "mlx" and targets is not None
    np.testing.assert_array_equal(restored.predict_proba(Q[:1]), expected_cpu)
    assert restored.last_query_backend_ == "cpu" and restored._y_device_ is targets
    assert vars(restored).keys() == before_restored.keys()
    assert all(vars(restored)[name] is value for name, value in before_restored.items())


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["neighbors", "classifier"])
def test_sparse_integer_queries_recheck_fractional_changes_under_sticky_auto_policy(kind):
    rng = np.random.default_rng(963)
    train = rng.integers(1, 17, size=(512, 64)).astype(np.float32)
    train[:, :16] = 0
    integer_queries = train[:16].copy()
    fractional_queries = integer_queries.copy()
    fractional_queries[-1, -1] += 0.5
    labels = np.arange(len(train)) % 3
    model = (NearestNeighbors(n_neighbors=5).fit(train) if kind == "neighbors" else
             KNeighborsClassifier(n_neighbors=5).fit(train, labels))
    assert model.execution_backend_ == "cpu" and model.execution_policy_ == "auto"
    for queries, expected_backend, device, forced in (
        (integer_queries, "cpu", "gpu", "mlx"),
        (fractional_queries, "mlx", "cpu", "cpu"),
        (integer_queries, "cpu", "gpu", "mlx"),
    ):
        distances, indices = _exact_neighbors(train, queries, 5)
        with config_context(device=device), tuning_context(force_backend=forced):
            if kind == "neighbors":
                actual = model.kneighbors(queries)
                np.testing.assert_array_equal(actual[0], distances)
                np.testing.assert_array_equal(actual[1], indices)
            else:
                expected = np.stack([np.bincount(labels[row], minlength=3) / 5 for row in indices])
                np.testing.assert_allclose(model.predict_proba(queries), expected, rtol=1e-6, atol=1e-7)
        assert model.last_query_backend_ == expected_backend
        assert model.execution_backend_ == "cpu" and model.execution_policy_ == "auto"


@pytest.mark.gpu
def test_first_gpu_query_prepares_from_private_cpu_training_after_caller_and_mirror_mutation():
    train = np.random.default_rng(145).integers(-16, 17, size=(512, 32)).astype(np.float32) / 8
    original = train.copy()
    queries = original[:64] + np.float32(0.0625)
    model = NearestNeighbors(n_neighbors=3).fit(train)
    assert model.execution_backend_ == "cpu" and model._index_._device_index is None
    mirror = model._fit_X
    assert not np.shares_memory(model._index_.training_data, train)
    assert not np.shares_memory(model._index_.training_data, mirror)
    train.fill(-999)
    mirror.fill(999)
    expected = _exact_neighbors(original, queries, 3)
    actual = model.kneighbors(queries)
    assert model.last_query_backend_ == "mlx"
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(np.asarray(model._index_.device_index.X), original)
    mirror.fill(50)
    np.testing.assert_array_equal(model.kneighbors(queries[:1])[1], expected[1][:1])
    assert model.last_query_backend_ == "cpu"
