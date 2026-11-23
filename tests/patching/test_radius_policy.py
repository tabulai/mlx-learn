"""Patched radius estimators: capability delegation and state-free stock policy."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse
from sklearn import neighbors as sk_neighbors

from mlxlearn._common.config import tuning_context
from mlxlearn.patching._estimators import (
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)


def _data(rng, n=600, d=5):
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(40, d)).astype(np.float32)
    return X, Q


def _assert_same_sets(ours, theirs):
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(np.sort(np.asarray(a)), np.sort(np.asarray(b)))


def _assert_ragged_equal(ours, theirs):
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_sparse_query_is_delegated_to_stock(rng):
    X, Q = _data(rng, d=20)
    model = NearestNeighbors(radius=5.0).fit(X)
    stock = sk_neighbors.NearestNeighbors(radius=5.0).fit(X.astype(np.float64))
    query = sparse.csr_matrix(Q.astype(np.float64))
    ours = model.radius_neighbors(query)
    theirs = stock.radius_neighbors(query)
    assert model.last_query_backend_ == "sklearn"
    _assert_ragged_equal(ours[1], theirs[1])
    _assert_ragged_equal(ours[0], theirs[0])


def test_per_query_radius_array_is_delegated_to_stock(rng):
    X, Q = _data(rng)
    radii = np.linspace(0.5, 2.0, len(Q))
    model = NearestNeighbors().fit(X)
    stock = sk_neighbors.NearestNeighbors().fit(X.astype(np.float64))
    ours = model.radius_neighbors(Q, radius=radii)
    theirs = stock.radius_neighbors(Q.astype(np.float64), radius=radii)
    assert model.last_query_backend_ == "sklearn"
    _assert_ragged_equal(ours[1], theirs[1])


@pytest.mark.parametrize("radius", [-1.0, None, float("nan")])
def test_unusual_scalar_radii_follow_stock(rng, radius):
    X, Q = _data(rng)
    model = NearestNeighbors(radius=1.0).fit(X)
    stock = sk_neighbors.NearestNeighbors(radius=1.0).fit(X.astype(np.float64))
    try:
        expected = stock.radius_neighbors(Q.astype(np.float64), radius=radius)
    except Exception as exc:  # stock raises: we must raise the same type
        with pytest.raises(type(exc)):
            model.radius_neighbors(Q, radius=radius)
        return
    ours = model.radius_neighbors(Q, radius=radius)
    _assert_same_sets(ours[1], expected[1])


def test_callable_weights_after_fit_are_delegated(rng):
    X, Q = _data(rng)
    y = (X[:, 0] > 0).astype(int)
    model = RadiusNeighborsClassifier(radius=1.5, outlier_label="most_frequent").fit(X, y)
    def decay(dist):
        # scikit-learn passes radius distances as an object array of rows.
        return np.array([np.exp(-np.asarray(row, dtype=np.float64)) for row in dist],
                        dtype=object)

    model.set_params(weights=decay)
    stock = sk_neighbors.RadiusNeighborsClassifier(
        radius=1.5, outlier_label="most_frequent", weights=decay,
    ).fit(X.astype(np.float64), y)
    np.testing.assert_array_equal(model.predict(Q), stock.predict(Q.astype(np.float64)))
    assert model.last_query_backend_ == "sklearn"


def test_values_beyond_float32_are_delegated(rng):
    X, Q = _data(rng)
    model = NearestNeighbors(radius=2.0).fit(X)
    Q64 = Q.astype(np.float64)
    Q64[0, 0] = 1e300
    ours = model.radius_neighbors(Q64)
    stock = sk_neighbors.NearestNeighbors(radius=2.0).fit(X.astype(np.float64))
    theirs = stock.radius_neighbors(Q64)
    assert model.last_query_backend_ == "sklearn"
    _assert_ragged_equal(ours[1], theirs[1])


def test_stock_delegation_returns_stock_output_unchanged(rng):
    X, Q = _data(rng, n=3000, d=12)
    with tuning_context(radius_route="stock"):
        model = NearestNeighbors(radius=3.0).fit(X)
        dist, idx = model.radius_neighbors(Q, sort_results=True)
        graph = model.radius_neighbors_graph(Q, mode="distance")
    stock = sk_neighbors.NearestNeighbors(radius=3.0).fit(X.astype(np.float64))
    ref_dist, ref_idx = stock.radius_neighbors(Q.astype(np.float64), sort_results=True)
    _assert_ragged_equal(idx, ref_idx)
    _assert_ragged_equal(dist, ref_dist)
    ref_graph = stock.radius_neighbors_graph(Q.astype(np.float64), mode="distance")
    assert (graph != ref_graph).nnz == 0


def test_stock_delegation_mirrors_the_users_algorithm(rng):
    X, Q = _data(rng, n=3000, d=4)
    with tuning_context(radius_route="stock"):
        model = NearestNeighbors(radius=0.5, leaf_size=7).fit(X)
        model.radius_neighbors(Q)
    stock = model._knn_query_state_["stock_radius"]
    assert stock.algorithm == "auto" and stock.leaf_size == 7
    assert stock._fit_method == "kd_tree"


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_patched_estimators_delegate_whole_predictions(rng, weights):
    X, Q = _data(rng, n=2000, d=6)
    y_class = (X[:, 0] > 0).astype(int)
    y_value = X[:, 1].astype(np.float64)
    with tuning_context(radius_route="stock"):
        clf = RadiusNeighborsClassifier(radius=1.2, weights=weights,
                                        outlier_label="most_frequent").fit(X, y_class)
        reg = RadiusNeighborsRegressor(radius=1.2, weights=weights).fit(X, y_value)
        proba = clf.predict_proba(Q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = reg.predict(Q)
    sk_clf = sk_neighbors.RadiusNeighborsClassifier(
        radius=1.2, weights=weights, outlier_label="most_frequent").fit(X.astype(np.float64), y_class)
    sk_reg = sk_neighbors.RadiusNeighborsRegressor(radius=1.2, weights=weights).fit(
        X.astype(np.float64), y_value)
    np.testing.assert_array_equal(proba, sk_clf.predict_proba(Q.astype(np.float64)))
    np.testing.assert_array_equal(pred, sk_reg.predict(Q.astype(np.float64)))


def test_owned_patched_calls_are_bitwise_reproducible(rng):
    X = rng.normal(size=(20_000, 3)).astype(np.float32)
    Q = rng.normal(size=(500, 3)).astype(np.float32)
    model = NearestNeighbors(radius=0.15).fit(X)
    runs = [model.radius_neighbors(Q, sort_results=True) for _ in range(3)]
    for other in runs[1:]:
        _assert_ragged_equal(runs[0][0], other[0])
        _assert_ragged_equal(runs[0][1], other[1])
