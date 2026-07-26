"""Neighbor parity against stock scikit-learn.

Tolerances and the tie-ordering rule come from ``docs/fp32_policy.md``; nothing
here invents its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import neighbors as sk_neighbors

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels.distance import build_index
from mlxlearn.exceptions import SparseInputError, UnsupportedParameterError
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from tests._helpers import assert_neighbor_parity

BACKENDS = ["mlx", "cpu"]


def sk_reference(X, **kwargs):
    return sk_neighbors.NearestNeighbors(algorithm="brute", metric="euclidean", **kwargs).fit(
        np.asarray(X, dtype=np.float64)
    )


# ----------------------------------------------------------------------------------
# kernel level
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "n_train, n_features, n_query, k",
    [(512, 16, 31, 7), (5000, 24, 96, 9), (300, 12, 16, 80), (40, 3, 40, 1)],
)
def test_kernel_matches_sklearn(backend, n_train, n_features, n_query, k, rng):
    X = rng.normal(size=(n_train, n_features)).astype(np.float32)
    Q = rng.normal(size=(n_query, n_features)).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query(Q, k)
    ref_dist, ref_idx = sk_reference(X, n_neighbors=k).kneighbors(Q.astype(np.float64))

    assert_neighbor_parity(dist, idx, ref_dist, ref_idx)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ties_break_on_training_index(backend):
    """Three identical points must come back in index order, deterministically."""
    X = np.array([[1.0], [1.0], [1.0], [2.0]], dtype=np.float32)
    dist, idx = build_index(X, backend=backend).query(np.array([[0.0]], dtype=np.float32), 3)
    np.testing.assert_array_equal(idx, [[0, 1, 2]])
    np.testing.assert_allclose(dist, [[1.0, 1.0, 1.0]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_blocking_is_invariant(backend, rng):
    """Results must not depend on how the work was tiled.

    This is what pins the cross-tile merge and the total order together. A merge
    that is merely "usually right" shows up here as a block-size-dependent answer.
    """
    X = rng.normal(size=(300, 5)).astype(np.float32)
    Q = rng.normal(size=(37, 5)).astype(np.float32)
    index = build_index(X, backend=backend)

    reference = index.query(Q, 4)
    for query_block in (1, 2, 7, 8, 37):
        for train_block in (3, 4, 7, 300):
            dist, idx = index.query(
                Q, 4, query_block_size=query_block, train_block_size=train_block
            )
            np.testing.assert_array_equal(idx, reference[1], err_msg=f"{query_block=} {train_block=}")
            np.testing.assert_allclose(dist, reference[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_short_trailing_tile(backend, rng):
    """A trailing tile narrower than k must work.

    The ancestor's GPU path raised ``Converting -1 to uint32 would result in
    overflow`` here, and because its dispatcher swallowed every exception, users
    saw an unexplained fallback rather than an error.
    """
    X = rng.normal(size=(10, 3)).astype(np.float32)
    Q = rng.normal(size=(4, 3)).astype(np.float32)
    index = build_index(X, backend=backend)

    dist, idx = index.query(Q, 3, train_block_size=4)
    ref_dist, ref_idx = index.query(Q, 3, train_block_size=10)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_query(backend, rng):
    X = rng.normal(size=(20, 3)).astype(np.float32)
    dist, idx = build_index(X, backend=backend).query(np.empty((0, 3), np.float32), 4)
    assert idx.shape == (0, 4)
    assert dist.shape == (0, 4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_k_larger_than_training_set(backend, rng):
    X = rng.normal(size=(6, 3)).astype(np.float32)
    index = build_index(X, backend=backend)
    with pytest.raises(ValueError, match="n_neighbors must be <= 6"):
        index.query(X, 7)
    with pytest.raises(ValueError, match="n_neighbors must be <= 5"):
        index.query(X, 6, exclude_self=True)


# ----------------------------------------------------------------------------------
# exclude_self -- the ancestor's worst bug
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_train, query_block", [(1500, None), (60, 8), (2049, 512)])
def test_self_query_across_blocks(backend, n_train, query_block, rng):
    """A row must never be returned as its own nearest neighbor.

    The ancestor compared each row against its *block-local* index, so every query
    block after the first tested the wrong identity. Verified there at 1 wrong row
    in 1500 with default blocking and 4 in 60 with an 8-row block — and its tests
    only ever exercised 3 rows, so it never showed up.
    """
    X = rng.normal(size=(n_train, 4)).astype(np.float32)
    k = 3
    dist, idx = build_index(X, backend=backend).query(
        X, k, exclude_self=True, query_block_size=query_block
    )

    own = np.arange(n_train)[:, None]
    assert not np.any(idx == own), "a row was returned as its own neighbor"

    # kneighbors() with no argument is scikit-learn's self-query: each row's own
    # index is excluded, which is what exclude_self=True reproduces.
    ref_dist, ref_idx = sk_reference(X, n_neighbors=k).kneighbors()
    assert_neighbor_parity(dist, idx, ref_dist, ref_idx)


@pytest.mark.parametrize("backend", BACKENDS)
def test_self_query_with_duplicate_rows(backend):
    """With duplicates the self index need not be first; it must still be the one dropped."""
    X = np.array([[0.0], [0.0], [0.0], [0.0], [5.0]], dtype=np.float32)
    _, idx = build_index(X, backend=backend).query(X, 2, exclude_self=True)
    own = np.arange(5)[:, None]
    assert not np.any(idx == own)


def test_self_query_requires_matching_row_count(rng):
    X = rng.normal(size=(10, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="exclude_self requires"):
        build_index(X, backend="cpu").query(X[:5], 2, exclude_self=True)


# ----------------------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_parity(weights, classification_data, rng):
    X, y = classification_data
    query = np.vstack([X[:20], rng.normal(scale=2.0, size=(30, X.shape[1]))])

    mlx = KNeighborsClassifier(n_neighbors=7, weights=weights).fit(X, y)
    ref = sk_neighbors.KNeighborsClassifier(
        n_neighbors=7, weights=weights, algorithm="brute"
    ).fit(X, y)

    np.testing.assert_array_equal(mlx.predict(query), ref.predict(query))
    np.testing.assert_allclose(mlx.predict_proba(query), ref.predict_proba(query), atol=1e-5)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_parity(weights, regression_data, rng):
    X, y = regression_data
    query = np.vstack([X[:20], rng.normal(size=(30, X.shape[1]))])

    mlx = KNeighborsRegressor(n_neighbors=7, weights=weights).fit(X, y)
    ref = sk_neighbors.KNeighborsRegressor(
        n_neighbors=7, weights=weights, algorithm="brute"
    ).fit(X, y)

    np.testing.assert_allclose(mlx.predict(query), ref.predict(query), atol=1e-6, rtol=1e-6)


def test_distance_weights_on_coincident_points(classification_data):
    """A query sitting exactly on training points must not produce an infinity."""
    X, y = classification_data
    query = X[:25]

    mlx = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
    ref = sk_neighbors.KNeighborsClassifier(
        n_neighbors=5, weights="distance", algorithm="brute"
    ).fit(X, y)

    proba = mlx.predict_proba(query)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(mlx.predict(query), ref.predict(query))


def test_classifier_vote_tie_goes_to_smallest_class():
    X = np.array([[1.0], [1.0], [1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1, 1])
    query = np.array([[0.0]])

    mlx = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=3, algorithm="brute").fit(X, y)
    np.testing.assert_array_equal(mlx.predict(query), ref.predict(query))
    np.testing.assert_allclose(mlx.predict_proba(query), ref.predict_proba(query))


def test_kneighbors_self_query_matches_sklearn(classification_data):
    X, _ = classification_data
    mlx = NearestNeighbors(n_neighbors=6).fit(X)
    ref = sk_neighbors.NearestNeighbors(n_neighbors=6, algorithm="brute").fit(X)
    assert_neighbor_parity(*mlx.kneighbors(), *ref.kneighbors())


def test_proba_shape_and_dtype(classification_data):
    X, y = classification_data
    mlx = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    proba = mlx.predict_proba(X[:11])
    assert proba.shape == (11, len(np.unique(y)))
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


# ----------------------------------------------------------------------------------
# no hidden state
# ----------------------------------------------------------------------------------


def test_mutating_the_input_after_fit_does_not_change_predictions(classification_data):
    """The index owns a copy. The ancestor aliased the caller's buffer.

    This is the failure mode that matters most for the agent workloads mlxlearn
    targets: refitting a probe and getting the previous fit's answers back,
    because a device copy was keyed on the identity of an array that has since
    been overwritten.
    """
    X, y = classification_data
    X = np.array(X, dtype=np.float64)
    query = X[:50].copy()

    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    before = model.predict(query)

    X[:] = 0.0
    np.testing.assert_array_equal(before, model.predict(query))


def test_two_estimators_do_not_contaminate_each_other(rng):
    """Interleaved use of two fits must not share device state through a global."""
    X1 = rng.normal(size=(5000, 8))
    X2 = rng.normal(size=(5000, 8)) + 50.0
    q = rng.normal(size=(10, 8))

    a = NearestNeighbors(n_neighbors=3).fit(X1)
    b = NearestNeighbors(n_neighbors=3).fit(X2)

    solo_a = a.kneighbors(q)
    solo_b = b.kneighbors(q)

    for _ in range(3):
        np.testing.assert_array_equal(a.kneighbors(q)[1], solo_a[1])
        np.testing.assert_array_equal(b.kneighbors(q)[1], solo_b[1])


def test_repeated_calls_are_deterministic(classification_data):
    X, _ = classification_data
    model = NearestNeighbors(n_neighbors=5).fit(X)
    first = model.kneighbors(X[:100])
    for _ in range(3):
        again = model.kneighbors(X[:100])
        np.testing.assert_array_equal(again[1], first[1])
        np.testing.assert_array_equal(again[0], first[0])


@pytest.mark.parametrize(
    "transform",
    [
        lambda X: X.astype(np.float32),
        lambda X: np.asfortranarray(X),
        lambda X: X[::1].copy().astype(np.float64),
        lambda X: X.tolist(),
    ],
    ids=["float32", "fortran", "float64", "list"],
)
def test_input_layout_does_not_change_results(transform, rng):
    X = rng.normal(size=(5000, 6))
    q = rng.normal(size=(20, 6))
    reference = NearestNeighbors(n_neighbors=4).fit(X).kneighbors(q)
    result = NearestNeighbors(n_neighbors=4).fit(transform(X)).kneighbors(q)
    np.testing.assert_array_equal(result[1], reference[1])


# ----------------------------------------------------------------------------------
# capability limits -- Layer 1 raises rather than guessing
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, parameter",
    [
        ({"algorithm": "kd_tree"}, "algorithm"),
        ({"algorithm": "ball_tree"}, "algorithm"),
        ({"metric": "manhattan"}, "metric"),
        ({"metric": "cosine"}, "metric"),
        ({"metric": "minkowski", "p": 1}, "p"),
        ({"metric": "minkowski", "p": 3}, "p"),
        ({"metric_params": {"w": [1.0]}}, "metric_params"),
        ({"weights": lambda d: d}, "weights"),
    ],
)
def test_unsupported_parameters_raise_precisely(kwargs, parameter, small_classification_data):
    X, y = small_classification_data
    with pytest.raises(UnsupportedParameterError) as excinfo:
        KNeighborsClassifier(n_neighbors=3, **kwargs).fit(X, y)
    assert excinfo.value.parameter == parameter


def test_metric_params_is_never_silently_ignored(small_classification_data):
    """The ancestor accepted metric_params and computed unweighted Euclidean anyway.

    A wrong answer delivered quickly is the worst possible outcome, so this is a
    dedicated test rather than one row of a parameter table.
    """
    X, y = small_classification_data
    with pytest.raises(UnsupportedParameterError, match="metric_params"):
        KNeighborsClassifier(metric="minkowski", p=2, metric_params={"w": np.ones(5)}).fit(X, y)


def test_sparse_input_raises_type_error(small_classification_data):
    """TypeError, with 'sparse' in the message — what scikit-learn's checks expect."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y = small_classification_data
    with pytest.raises(SparseInputError) as excinfo:
        KNeighborsClassifier(n_neighbors=3).fit(sparse.csr_matrix(X), y)
    assert isinstance(excinfo.value, TypeError)
    assert "sparse" in str(excinfo.value).lower()


def test_multioutput_y_raises(rng):
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 2, size=(200, 2))
    with pytest.raises(UnsupportedParameterError, match="single-output"):
        KNeighborsClassifier(n_neighbors=3).fit(X, y)


def test_nan_input_raises(small_classification_data):
    X, y = small_classification_data
    X = np.array(X)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        KNeighborsClassifier(n_neighbors=3).fit(X, y)


# ----------------------------------------------------------------------------------
# backend selection and stickiness
# ----------------------------------------------------------------------------------


def test_small_problems_use_the_cpu_path(small_classification_data):
    """Layer 1 has its own CPU path; it never reaches for scikit-learn."""
    X, y = small_classification_data
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert model.execution_backend_ == "cpu"


def test_large_problems_use_mlx(classification_data):
    X, y = classification_data
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert model.execution_backend_ == "mlx"


def test_inference_reuses_the_fitted_backend(classification_data):
    """The sticky invariant, asserted through the diagnostics record."""
    from mlxlearn import get_backend_diagnostics

    X, y = classification_data
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    fitted = model.execution_backend_
    model.predict(X[:10])

    inference = [e for e in get_backend_diagnostics() if e.operation in ("predict", "kneighbors")]
    assert inference, "no inference event recorded"
    assert all(e.backend == fitted for e in inference)
    assert all(e.reason == "sticky" for e in inference)


def test_refit_may_change_backend(small_classification_data, classification_data):
    small_X, small_y = small_classification_data
    big_X, big_y = classification_data

    model = KNeighborsClassifier(n_neighbors=3).fit(small_X, small_y)
    assert model.execution_backend_ == "cpu"
    model.fit(big_X, big_y)
    assert model.execution_backend_ == "mlx"


def test_unfitted_backend_attribute_raises():
    with pytest.raises(AttributeError, match="not fitted"):
        KNeighborsClassifier().execution_backend_


def test_forced_backend_overrides_crossover(small_classification_data):
    X, y = small_classification_data
    with tuning_context(force_backend="mlx"):
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert model.execution_backend_ == "mlx"


def test_layer1_cannot_be_forced_to_sklearn(small_classification_data):
    from mlxlearn.exceptions import MLXLearnError

    X, y = small_classification_data
    with tuning_context(force_backend="sklearn"):
        with pytest.raises(MLXLearnError, match="Layer 1 estimators never dispatch"):
            KNeighborsClassifier(n_neighbors=3).fit(X, y)
