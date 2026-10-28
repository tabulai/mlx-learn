"""Radius-neighbor parity against stock scikit-learn.

Membership and distances follow ``docs/fp32_policy.md`` §3; neighborhood size is
ragged, so each query is compared on its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import neighbors as sk_neighbors

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels.distance import build_index
from mlxlearn.exceptions import SparseInputError, UnsupportedParameterError
from mlxlearn.neighbors import (
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from tests._helpers import assert_radius_parity

BACKENDS = ["mlx", "cpu"]


def sk_reference(X, *, radius):
    return sk_neighbors.NearestNeighbors(
        radius=radius, algorithm="brute", metric="euclidean"
    ).fit(np.asarray(X, dtype=np.float64))


# ----------------------------------------------------------------------------------
# kernel level
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "n_train, n_features, n_query, radius",
    [(80, 4, 11, 1.5), (400, 8, 25, 2.5), (40, 3, 40, 0.8)],
)
def test_kernel_radius_matches_sklearn(backend, n_train, n_features, n_query, radius, rng):
    X = rng.normal(size=(n_train, n_features)).astype(np.float32)
    Q = rng.normal(size=(n_query, n_features)).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query_radius(Q, radius)
    ref_dist, ref_idx = sk_reference(X, radius=radius).radius_neighbors(Q.astype(np.float64))
    assert_radius_parity(idx, ref_idx, dist, ref_dist)


@pytest.mark.parametrize("backend", BACKENDS)
def test_radius_ties_break_on_training_index(backend):
    X = np.array([[1.0], [1.0], [1.0], [4.0]], dtype=np.float32)
    dist, idx = build_index(X, backend=backend).query_radius(
        np.array([[0.0]], dtype=np.float32), radius=1.5, sort_results=True
    )
    np.testing.assert_array_equal(idx[0], [0, 1, 2])
    np.testing.assert_allclose(dist[0], [1.0, 1.0, 1.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_radius_blocking_is_invariant(backend, rng):
    X = rng.normal(size=(180, 5)).astype(np.float32)
    Q = rng.normal(size=(23, 5)).astype(np.float32)
    index = build_index(X, backend=backend)
    reference = index.query_radius(Q, 1.75)

    for query_block in (1, 5, 8, 23):
        for train_block in (3, 7, 180):
            dist, idx = index.query_radius(
                Q, 1.75, query_block_size=query_block, train_block_size=train_block
            )
            for row in range(len(idx)):
                np.testing.assert_array_equal(
                    np.sort(idx[row]),
                    np.sort(reference[1][row]),
                    err_msg=f"{query_block=} {train_block=} row={row}",
                )


@pytest.mark.parametrize("backend", BACKENDS)
def test_radius_self_query_drops_the_query_row(backend, rng):
    X = rng.normal(size=(60, 4)).astype(np.float32)
    dist, idx = build_index(X, backend=backend).query_radius(X, 2.0, exclude_self=True)
    for row, neighbors in enumerate(idx):
        assert row not in set(np.asarray(neighbors).tolist())
        assert np.all(np.asarray(dist[row]) <= 2.0 + 1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_radius_empty_neighborhoods_are_empty_arrays(backend):
    X = np.array([[0.0], [10.0]], dtype=np.float32)
    Q = np.array([[100.0]], dtype=np.float32)
    dist, idx = build_index(X, backend=backend).query_radius(Q, 0.5)
    assert idx[0].shape == (0,)
    assert dist[0].shape == (0,)
    assert idx[0].dtype == np.int64
    assert dist[0].dtype == np.float64


def test_negative_radius_raises():
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        build_index(X, backend="cpu").query_radius(X, -0.1)


# ----------------------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------------------


def test_nearest_neighbors_radius_matches_sklearn(classification_data, rng):
    X, _ = classification_data
    query = np.vstack([X[:15], rng.normal(size=(20, X.shape[1]))])
    radius = 8.0

    mlx = NearestNeighbors(radius=radius).fit(X)
    ref = sk_neighbors.NearestNeighbors(radius=radius, algorithm="brute").fit(X)
    mlx_dist, mlx_idx = mlx.radius_neighbors(query, sort_results=True)
    ref_dist, ref_idx = ref.radius_neighbors(query, sort_results=True)
    assert_radius_parity(mlx_idx, ref_idx, mlx_dist, ref_dist)
    assert mlx.execution_backend_ == "mlx"


def test_radius_self_query_matches_sklearn(classification_data):
    X, _ = classification_data
    radius = 6.0
    mlx = NearestNeighbors(radius=radius).fit(X)
    ref = sk_neighbors.NearestNeighbors(radius=radius, algorithm="brute").fit(X)
    mlx_dist, mlx_idx = mlx.radius_neighbors()
    ref_dist, ref_idx = ref.radius_neighbors()
    assert_radius_parity(mlx_idx, ref_idx, mlx_dist, ref_dist)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_parity(weights, classification_data, rng):
    X, y = classification_data
    query = np.vstack([X[:20], rng.normal(scale=0.5, size=(15, X.shape[1]))])
    radius = 12.0

    mlx = RadiusNeighborsClassifier(radius=radius, weights=weights).fit(X, y)
    ref = sk_neighbors.RadiusNeighborsClassifier(
        radius=radius, weights=weights, algorithm="brute"
    ).fit(X, y)

    np.testing.assert_array_equal(mlx.predict(query), ref.predict(query))
    np.testing.assert_allclose(mlx.predict_proba(query), ref.predict_proba(query), atol=1e-5)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_parity(weights, regression_data, rng):
    X, y = regression_data
    query = np.vstack([X[:20], rng.normal(size=(15, X.shape[1]))])
    radius = 10.0

    mlx = RadiusNeighborsRegressor(radius=radius, weights=weights).fit(X, y)
    ref = sk_neighbors.RadiusNeighborsRegressor(
        radius=radius, weights=weights, algorithm="brute"
    ).fit(X, y)

    np.testing.assert_allclose(mlx.predict(query), ref.predict(query), atol=1e-5, rtol=1e-5)


def test_outlier_label_none_raises_on_empty_neighborhood():
    X = np.array([[0.0], [0.1], [0.2]])
    y = np.array([0, 0, 1])
    clf = RadiusNeighborsClassifier(radius=0.05).fit(X, y)
    with pytest.raises(ValueError, match="No neighbors found"):
        clf.predict([[10.0]])


def test_outlier_label_is_applied():
    X = np.array([[0.0], [0.1], [0.2]])
    y = np.array([0, 0, 1])
    clf = RadiusNeighborsClassifier(radius=0.05, outlier_label=0).fit(X, y)
    np.testing.assert_array_equal(clf.predict([[10.0]]), [0])
    proba = clf.predict_proba([[10.0]])
    np.testing.assert_allclose(proba, [[1.0, 0.0]])


def test_outlier_label_most_frequent():
    X = np.array([[0.0], [0.1], [0.2], [0.3]])
    y = np.array([0, 0, 0, 1])
    clf = RadiusNeighborsClassifier(radius=0.05, outlier_label="most_frequent").fit(X, y)
    np.testing.assert_array_equal(clf.predict([[50.0]]), [0])


def test_regressor_empty_neighborhood_is_nan():
    X = np.array([[0.0], [0.1]])
    y = np.array([1.0, 2.0])
    reg = RadiusNeighborsRegressor(radius=0.01).fit(X, y)
    with pytest.warns(UserWarning, match="no neighbors"):
        pred = reg.predict([[10.0]])
    assert np.isnan(pred[0])


def test_sort_results_false_still_requires_distances_for_true_flag():
    X = np.array([[0.0], [1.0], [2.0]])
    nn = NearestNeighbors(radius=1.5).fit(X)
    with pytest.raises(ValueError, match="return_distance must be True"):
        nn.radius_neighbors([[0.5]], return_distance=False, sort_results=True)


def test_radius_neighbors_graph_matches_sklearn(rng):
    X = rng.normal(size=(80, 4))
    Q = rng.normal(size=(12, 4))
    radius = 2.0
    mlx = NearestNeighbors(radius=radius).fit(X)
    ref = sk_neighbors.NearestNeighbors(radius=radius, algorithm="brute").fit(X)
    mlx_graph = mlx.radius_neighbors_graph(Q, mode="connectivity")
    ref_graph = ref.radius_neighbors_graph(Q, mode="connectivity")
    np.testing.assert_array_equal(mlx_graph.toarray(), ref_graph.toarray())


def test_inference_reuses_the_fitted_backend(classification_data):
    from mlxlearn import get_backend_diagnostics

    X, y = classification_data
    model = RadiusNeighborsClassifier(radius=12.0).fit(X, y)
    fitted = model.execution_backend_
    model.predict(X[:8])

    inference = [
        e
        for e in get_backend_diagnostics()
        if e.operation in ("predict", "predict_proba", "radius_neighbors")
    ]
    assert inference
    assert all(e.backend == fitted for e in inference)
    assert all(e.reason == "sticky" for e in inference)


def test_small_problems_use_the_cpu_path(small_classification_data):
    X, y = small_classification_data
    model = RadiusNeighborsClassifier(radius=5.0).fit(X, y)
    assert model.execution_backend_ == "cpu"


def test_forced_backend_overrides_crossover(small_classification_data):
    X, y = small_classification_data
    with tuning_context(force_backend="mlx"):
        model = RadiusNeighborsClassifier(radius=5.0).fit(X, y)
    assert model.execution_backend_ == "mlx"


def test_unsupported_algorithm_raises(small_classification_data):
    X, y = small_classification_data
    with pytest.raises(UnsupportedParameterError) as excinfo:
        RadiusNeighborsClassifier(radius=1.0, algorithm="kd_tree").fit(X, y)
    assert excinfo.value.parameter == "algorithm"


def test_sparse_input_raises(small_classification_data):
    sparse = pytest.importorskip("scipy.sparse")
    X, y = small_classification_data
    with pytest.raises(SparseInputError) as excinfo:
        RadiusNeighborsClassifier(radius=1.0).fit(sparse.csr_matrix(X), y)
    assert "sparse" in str(excinfo.value).lower()
