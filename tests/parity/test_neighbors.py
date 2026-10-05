"""Neighbor parity against stock scikit-learn.

Tolerances and the tie-ordering rule come from ``docs/fp32_policy.md``; nothing
here invents its own.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn import neighbors as sk_neighbors
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.estimator_checks import check_estimator, check_param_validation

import mlxlearn
from mlxlearn._common.config import tuning_context
from mlxlearn._kernels.distance import build_index
from mlxlearn.exceptions import (
    CapabilityError,
    SparseInputError,
    UnsupportedInputError,
    UnsupportedParameterError,
)
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.patching import _estimators as patched
from tests._helpers import assert_neighbor_parity

BACKENDS = ["mlx", "cpu"]

LAYER1 = [NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor]
LAYER2 = [patched.NearestNeighbors, patched.KNeighborsClassifier, patched.KNeighborsRegressor]
SUPERVISED = [
    (KNeighborsClassifier, sk_neighbors.KNeighborsClassifier),
    (KNeighborsRegressor, sk_neighbors.KNeighborsRegressor),
    (patched.KNeighborsClassifier, sk_neighbors.KNeighborsClassifier),
    (patched.KNeighborsRegressor, sk_neighbors.KNeighborsRegressor),
]


def _layer_id(cls):
    return ("L2-" if cls._mlxlearn_allow_fallback else "L1-") + cls.__name__


@pytest.fixture
def mlx_sized():
    """Every neighbor gate lifted for the test, so a small problem reaches the MLX path
    by the ordinary route in both layers. For tests about that path, not dispatch."""
    with tuning_context(
        knn_min_train_samples=0, knn_min_work=0, knn_min_features=0, knn_narrow_min_samples=0,
        knn_probe_max_uncertified=1.0, knn_wide_features=0, knn_max_neighbors=2**31,
        knn_wide_min_samples=0,
    ):
        yield


def _events(operation=None):
    events = mlxlearn.get_backend_diagnostics()
    return [e for e in events if operation is None or e.operation == operation]


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
        ({"metric": "minkowski", "p": 2, "metric_params": {"p": 3}}, "p"),
        ({"metric_params": {"w": [1.0]}}, "metric_params"),
        ({"metric_params": {"p": 2, "w": [1.0]}}, "metric_params"),
        ({"metric": "euclidean", "metric_params": {"p": 2}}, "metric_params"),
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


def _mlx_sized_data(rng):
    """About the smallest problem the default thresholds send to MLX: 4 096 rows and
    6 000 000 elements."""
    X = rng.normal(size=(4_096, 1_465))
    return X, (X[:, 0] > 0).astype(int)


@pytest.mark.gpu
def test_large_problems_use_mlx(rng):
    X, y = _mlx_sized_data(rng)
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert model.execution_backend_ == "mlx"
    assert KNeighborsClassifier(n_neighbors=3).fit(X[:-1], y[:-1]).execution_backend_ == "cpu"


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


@pytest.mark.gpu
def test_refit_may_change_backend(small_classification_data, rng):
    small_X, small_y = small_classification_data
    big_X, big_y = _mlx_sized_data(rng)

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


# ----------------------------------------------------------------------------------
# parameter validation -- scikit-learn's constraints, in both layers
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", LAYER1 + LAYER2, ids=_layer_id)
def test_check_param_validation(cls):
    """scikit-learn's own check, which ``check_estimator`` does not run.

    Overriding ``fit`` drops the ``_fit_context`` that validates constructor
    parameters, so without an explicit ``_validate_params()`` an invalid value
    was accepted and an invalid ``algorithm`` reported as a capability mismatch.
    """
    check_param_validation(cls.__name__, cls())


@pytest.mark.parametrize("cls", LAYER2, ids=_layer_id)
def test_check_param_validation_on_the_patched_mlx_path(cls):
    """The check's 20-row data is below the crossover, so patched mode hands the
    fit to scikit-learn, which validates on its own. Forcing MLX is what reaches
    mlxlearn's fit."""
    with tuning_context(force_backend="mlx"):
        check_param_validation(cls.__name__, cls())


@pytest.mark.parametrize(
    "cls",
    [KNeighborsClassifier, KNeighborsRegressor, patched.KNeighborsClassifier, patched.KNeighborsRegressor],
    ids=_layer_id,
)
def test_n_neighbors_zero_is_rejected_above_the_crossover(cls, rng):
    """It used to fit, then predict ``classes_[0]`` (or NaN) for every row."""
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 3, 600)
    with pytest.raises(ValueError, match="The 'n_neighbors' parameter of .* must be"):
        cls(n_neighbors=0).fit(X, y)


def test_n_neighbors_bound_message_matches_sklearn(rng):
    """Including the ``n_samples`` suffix, and the order of the checks."""
    X = rng.normal(size=(300, 16))
    Q = rng.normal(size=(2, 16))
    mine = NearestNeighbors().fit(X)
    theirs = sk_neighbors.NearestNeighbors(algorithm="brute").fit(X)

    for query, k in ((Q, 301), (None, 300)):
        with pytest.raises(ValueError) as expected:
            theirs.kneighbors(query, n_neighbors=k)
        with pytest.raises(ValueError) as got:
            mine.kneighbors(query, n_neighbors=k)
        assert str(got.value) == str(expected.value)

    # scikit-learn validates the query before the bound, so a wrong width wins.
    with pytest.raises(ValueError, match="X has 3 features"):
        mine.kneighbors(Q[:, :3], n_neighbors=500)
    with pytest.raises(ValueError, match="Expected n_neighbors > 0. Got -1$"):
        mine.kneighbors(Q, n_neighbors=-1.5)


# ----------------------------------------------------------------------------------
# targets -- column vectors, multi-output, and no aliasing
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("n_samples", [120, 600], ids=["below-crossover", "above-crossover"])
@pytest.mark.parametrize("cls", [KNeighborsRegressor, patched.KNeighborsRegressor], ids=_layer_id)
def test_regressor_column_vector_y_keeps_its_shape(cls, n_samples, weights, rng):
    """A ``(n, 1)`` target predicts ``(n_queries, 1)``, as scikit-learn's does, silently.

    The MLX path used to ravel it with a DataConversionWarning, so under
    ``patch_sklearn()`` the output shape depended on the size of the data, and
    ``pred - y_test`` quietly broadcast to a square matrix.
    """
    X = rng.normal(size=(n_samples, 16))
    y = rng.normal(size=(n_samples, 1))
    Q = rng.normal(size=(9, 16))
    ref = sk_neighbors.KNeighborsRegressor(n_neighbors=5, weights=weights).fit(X, y).predict(Q)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        pred = cls(n_neighbors=5, weights=weights).fit(X, y).predict(Q)

    assert pred.shape == ref.shape == (9, 1)
    np.testing.assert_allclose(pred, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cls", [KNeighborsClassifier, patched.KNeighborsClassifier], ids=_layer_id)
def test_classifier_column_vector_y_warns_and_ravels_like_sklearn(cls, rng):
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 3, size=(600, 1))
    Q = rng.normal(size=(9, 16))
    with pytest.warns(DataConversionWarning):
        ref = sk_neighbors.KNeighborsClassifier(n_neighbors=5).fit(X, y)

    with pytest.warns(DataConversionWarning, match="column-vector y"):
        model = cls(n_neighbors=5).fit(X, y)

    assert model.outputs_2d_ is False
    np.testing.assert_array_equal(model.classes_, ref.classes_)
    np.testing.assert_array_equal(model.predict(Q), ref.predict(Q))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("cls", [KNeighborsClassifier, patched.KNeighborsClassifier], ids=_layer_id)
def test_multi_output_classification_matches_sklearn(cls, weights, rng, mlx_sized):
    """Multi-output targets stay on the MLX path; only the vote runs per output."""
    X = rng.normal(size=(600, 16))
    Y = np.column_stack([rng.integers(0, 3, 600), 10 * rng.integers(0, 2, 600) + 5])
    Q = np.vstack([X[:10], rng.normal(size=(40, 16))])

    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=7, weights=weights).fit(X, Y)
    model = cls(n_neighbors=7, weights=weights).fit(X, Y)

    assert model.execution_backend_ != "sklearn"
    assert model.outputs_2d_ is True
    for mine, theirs in zip(model.classes_, ref.classes_, strict=True):
        np.testing.assert_array_equal(mine, theirs)
    np.testing.assert_array_equal(model.predict(Q), ref.predict(Q))
    for mine, theirs in zip(model.predict_proba(Q), ref.predict_proba(Q), strict=True):
        np.testing.assert_allclose(mine, theirs, atol=1e-5)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("cls", [KNeighborsRegressor, patched.KNeighborsRegressor], ids=_layer_id)
def test_multi_output_regression_matches_sklearn(cls, weights, rng, mlx_sized):
    X = rng.normal(size=(600, 16))
    Y = rng.normal(size=(600, 3))
    Q = np.vstack([X[:10], rng.normal(size=(40, 16))])

    ref = sk_neighbors.KNeighborsRegressor(n_neighbors=7, weights=weights).fit(X, Y).predict(Q)
    model = cls(n_neighbors=7, weights=weights).fit(X, Y)

    assert model.execution_backend_ != "sklearn"
    pred = model.predict(Q)
    assert pred.shape == ref.shape == (50, 3)
    np.testing.assert_allclose(pred, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutating_float64_y_after_fit_does_not_change_predictions(backend, regression_data):
    """``validate_data`` hands back the caller's own buffer for a float64 ``y``.

    The regressor used to keep it, so writing into ``y`` after fit changed its
    predictions. An int ``y`` is copied by conversion, which is why the invariant
    test that mutates a classification target never saw this.
    """
    X, y = regression_data
    y = np.array(y, dtype=np.float64)
    with tuning_context(force_backend=backend):
        model = KNeighborsRegressor(n_neighbors=5).fit(X, y)
    before = model.predict(X[:20])

    assert not np.shares_memory(model._y, y)
    assert not hasattr(model, "_y_checked_")
    y[:] = 1000.0
    np.testing.assert_array_equal(before, model.predict(X[:20]))


def test_vote_helpers_match_a_naive_accumulation(rng):
    """``np.bincount`` replaced ``np.add.at``; the sums must be bitwise the same."""
    from mlxlearn.neighbors._base import class_probabilities, weighted_mode

    codes = rng.integers(0, 4, size=(200, 7))
    for weights in (None, rng.random((200, 7))):
        naive = np.zeros((200, 4))
        rows = np.repeat(np.arange(200), 7)
        np.add.at(naive, (rows, codes.ravel()), 1.0 if weights is None else weights.ravel())
        np.testing.assert_array_equal(
            class_probabilities(codes, weights, 4), naive / naive.sum(axis=1, keepdims=True)
        )
        np.testing.assert_array_equal(weighted_mode(codes, weights, 4), naive.argmax(axis=1))


# ----------------------------------------------------------------------------------
# input handling -- strings, sparse queries, prebuilt indexes
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", LAYER1 + LAYER2, ids=_layer_id)
def test_string_data_is_rejected_like_sklearn(cls, rng):
    """An explicit float32 dtype made ``check_array`` parse ``"1.5"``; stock refuses it.

    Under ``patch_sklearn()`` that meant string data raised below the crossover
    and was accepted above it.
    """
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 2, 600)
    for strings in (X.astype(str), X.astype(str).tolist()):
        with pytest.raises(ValueError, match="bytes/strings"):
            cls().fit(strings, y)

    model = cls().fit(X, y)
    with pytest.raises(ValueError, match="bytes/strings"):
        model.kneighbors(X[:3].astype(str))


def test_object_arrays_of_numbers_are_still_accepted(rng):
    """scikit-learn converts object arrays, so mlxlearn must not start refusing them."""
    X = rng.normal(size=(600, 16))
    reference = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X[:5])
    result = NearestNeighbors(n_neighbors=3).fit(X.astype(object)).kneighbors(X[:5].astype(object))
    np.testing.assert_array_equal(result[1], reference[1])


def _sparse_query_problem(rng):
    X = rng.normal(size=(600, 32))
    y = rng.integers(0, 3, 600)
    Q = rng.normal(size=(40, 32))
    Q[Q < 0.5] = 0.0
    return X, y, Q


def test_patched_mlx_fit_answers_sparse_queries_like_sklearn(rng, mlx_sized):
    """Layer 2 advertises sparse input, and stock brute force serves this query.

    The MLX fit populates the state scikit-learn's brute-force search reads, so a
    sparse query is answered by it, recorded as one fallback event on the call.
    """
    sparse = pytest.importorskip("scipy.sparse")
    X, y, Q = _sparse_query_problem(rng)
    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X, y)
    model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert model.execution_backend_ != "sklearn"

    mlxlearn.clear_backend_diagnostics()
    with pytest.warns(mlxlearn.MLXLearnFallbackWarning, match="sparse"):
        pred = model.predict(sparse.csr_matrix(Q))
    np.testing.assert_array_equal(pred, ref.predict(Q))
    [event] = _events()
    assert (event.operation, event.backend, event.reason) == ("predict", "sklearn", "sparse-input")
    assert event.fallback

    dist, idx = model.kneighbors(sparse.csr_matrix(Q))
    assert_neighbor_parity(dist, idx, *ref.kneighbors(Q), atol=1e-5)
    np.testing.assert_allclose(model.predict_proba(sparse.csr_matrix(Q)), ref.predict_proba(Q))


def test_patched_sparse_query_honors_fallback_policy_raise(rng, mlx_sized):
    sparse = pytest.importorskip("scipy.sparse")
    X, y, Q = _sparse_query_problem(rng)
    model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    with mlxlearn.config_context(fallback_policy="raise"):
        with pytest.raises(CapabilityError, match="fallback_policy='raise'"):
            model.predict(sparse.csr_matrix(Q))


def test_layer1_mlx_fit_still_raises_on_sparse_queries(rng):
    sparse = pytest.importorskip("scipy.sparse")
    X, y, Q = _sparse_query_problem(rng)
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    with pytest.raises(SparseInputError, match="predict received sparse input"):
        model.predict(sparse.csr_matrix(Q))
    with pytest.raises(SparseInputError, match="kneighbors received sparse input"):
        model.kneighbors(sparse.csr_matrix(Q))


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_sparse_queries_reach_sklearns_search_in_the_training_dtype(metric, monkeypatch, rng, mlx_sized):
    """The sparse query used to reach scikit-learn as float64, against float32.

    scikit-learn's search needs one dtype on both sides. With two it fell to a
    pairwise path several times slower than stock, which cannot compute
    'sqeuclidean' on sparse input: predict, kneighbors and radius_neighbors all
    raised TypeError where stock scikit-learn answers.
    """
    sparse = pytest.importorskip("scipy.sparse")
    X, y, Q = _sparse_query_problem(rng)
    Qs = sparse.csr_matrix(Q)
    kwargs = {"n_neighbors": 5, "metric": metric}
    ref = sk_neighbors.KNeighborsClassifier(algorithm="brute", **kwargs).fit(X, y)
    model = patched.KNeighborsClassifier(**kwargs).fit(X, y)
    assert model.execution_backend_ != "sklearn"

    atol = 1e-4  # squared distances around 30, from float32 training data
    np.testing.assert_array_equal(model.predict(Qs), ref.predict(Q))
    np.testing.assert_allclose(model.predict_proba(Qs), ref.predict_proba(Q))
    assert_neighbor_parity(*model.kneighbors(Qs), *ref.kneighbors(Q), atol=atol)
    graph = model.kneighbors_graph(Qs, mode="distance")
    assert abs(graph - ref.kneighbors_graph(Q, mode="distance")).max() <= atol

    reg = patched.KNeighborsRegressor(**kwargs).fit(X, y.astype(float))
    ref_reg = sk_neighbors.KNeighborsRegressor(algorithm="brute", **kwargs).fit(X, y.astype(float))
    np.testing.assert_allclose(reg.predict(Qs), ref_reg.predict(Q))

    # Radius queries are scikit-learn's code in both layers, so Layer 1 serves a
    # sparse one too, as an ordinary call.
    radius = 30.0 if metric == "sqeuclidean" else 5.5
    ref_nn = sk_neighbors.NearestNeighbors(radius=radius, algorithm="brute", metric=metric).fit(X)
    for cls in (NearestNeighbors, patched.NearestNeighbors):
        nn = cls(radius=radius, metric=metric).fit(X)
        mlxlearn.clear_backend_diagnostics()
        _assert_radius_parity(nn.radius_neighbors(Qs), ref_nn.radius_neighbors(Q), radius, atol=atol)
        assert [(e.operation, e.reason, e.fallback) for e in _events()] == [
            ("radius_neighbors", "sticky", False)
        ]

    seen = []
    original = sk_neighbors.KNeighborsClassifier.kneighbors

    def spy(self, X=None, *args, **kwargs):
        seen.append(X.dtype)
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(sk_neighbors.KNeighborsClassifier, "kneighbors", spy)
    model.kneighbors(Qs)
    model.kneighbors(Qs.astype(np.float32))
    assert seen == [model._fit_X.dtype] * 2 == [np.dtype(np.float32)] * 2


def test_sparse_query_float32_cannot_hold_is_an_unsupported_input_error(rng, mlx_sized):
    """As a dense query's is; the cast would otherwise turn it into infinity."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y, Q = _sparse_query_problem(rng)
    Q[0, 0] = 1e300
    model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    with pytest.raises(UnsupportedInputError) as excinfo:
        model.predict(sparse.csr_matrix(Q))
    assert excinfo.value.reason == "float32-overflow"
    with pytest.raises(UnsupportedInputError):
        NearestNeighbors(radius=1.0).fit(X).radius_neighbors(sparse.csr_matrix(Q))


@pytest.mark.parametrize(
    "make_query",
    [lambda Q: Q, lambda Q: Q.T.copy().T, lambda Q: Q.astype(np.float32)],
    ids=["float64", "fortran", "float32"],
)
@pytest.mark.parametrize("cls", [NearestNeighbors, patched.NearestNeighbors], ids=_layer_id)
def test_radius_neighbors_keeps_feature_names_like_sklearn(cls, make_query, rng, mlx_sized):
    """The float32 cast used to turn a DataFrame query into a bare array.

    scikit-learn validates the query again, so a correctly named DataFrame warned
    that it had no feature names, and an array warned twice where stock warns once.
    """
    pd = pytest.importorskip("pandas")
    X = rng.normal(size=(2000, 16))
    Q = make_query(rng.normal(size=(20, 16)))
    columns = [f"f{i}" for i in range(16)]
    frame, query = pd.DataFrame(X, columns=columns), pd.DataFrame(Q, columns=columns)
    ref = sk_neighbors.NearestNeighbors(radius=4.5, algorithm="brute").fit(frame)
    model = cls(radius=4.5).fit(frame)
    assert model.execution_backend_ != "sklearn"

    # UserWarning only: NumPy's matmul on some BLAS builds raises spurious
    # RuntimeWarnings, which say nothing about the query's feature names.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _assert_radius_parity(model.radius_neighbors(query), ref.radius_neighbors(query), 4.5)
        model.radius_neighbors_graph(query)

    for fitted in (model, ref):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            fitted.radius_neighbors(Q)
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == [
            f"X does not have valid feature names, but {type(fitted).__name__} was "
            "fitted with feature names"
        ]
        with pytest.raises(ValueError, match="feature names should match"):
            fitted.radius_neighbors(query.rename(columns={"f0": "g0"}))


@pytest.mark.parametrize("cls", [KNeighborsRegressor, patched.KNeighborsRegressor], ids=_layer_id)
def test_regressor_refuses_a_string_target(cls, rng, mlx_sized):
    """A float64 copy of y parsed "1.5" as 1.5. Stock stores it and fails in predict,
    so under patch_sklearn() whether string targets worked depended on the size."""
    X = rng.normal(size=(600, 16))
    y = rng.normal(size=600)
    for strings in (y.astype(str), y.astype(str).tolist(), y.astype(bytes)):
        with pytest.raises(ValueError, match="bytes/strings"):
            cls().fit(X, strings)
    assert cls().fit(X, y.astype(object)).execution_backend_ != "sklearn"


class _CountedRow:
    """A row that counts how often NumPy reads it, which is once per conversion of its list."""

    def __init__(self, values, reads):
        self._values, self._reads = values, reads

    def __array__(self, dtype=None, copy=None):
        self._reads.append(None)
        return self._values if dtype is None else self._values.astype(dtype)

    def __getitem__(self, i):
        return self._values[i]

    def __len__(self):
        return len(self._values)


@pytest.mark.parametrize("backend", BACKENDS)
def test_list_input_is_converted_once(backend, rng):
    """The string check converted a list just to read its dtype, then validation
    converted it again, and fit's range check a third time."""
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 2, 600)
    reads = []
    rows = [_CountedRow(row, reads) for row in X]
    queries = rows[:10]

    with tuning_context(force_backend=backend):
        model = KNeighborsClassifier(n_neighbors=3).fit(rows, y)
    assert model.execution_backend_ == backend
    assert len(reads) == len(rows)
    for method in (model.kneighbors, model.predict):
        reads.clear()
        method(queries)
        assert len(reads) == len(queries)

    nn = NearestNeighbors(radius=4.0).fit(X)
    reads.clear()
    nn.radius_neighbors(queries)
    assert len(reads) == len(queries)

    reference = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    np.testing.assert_array_equal(model.kneighbors(queries)[1], reference.kneighbors(X[:10])[1])


@pytest.mark.parametrize(
    "estimator, method",
    [
        ("KNeighborsClassifier", "predict"),
        ("KNeighborsClassifier", "predict_proba"),
        ("KNeighborsRegressor", "predict"),
    ],
)
@pytest.mark.parametrize("layer", ["L1", "L2"])
def test_callable_weights_match_sklearn(estimator, method, layer, rng, mlx_sized):
    """Weighting is host code on the distances the search returns, so a callable is
    served on MLX. It used to be refused at fit, and set after an MLX fit it made
    predict raise "weights not recognized" where stock scikit-learn predicts."""
    X = rng.normal(size=(2000, 16))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    if estimator == "KNeighborsRegressor":
        y = X[:, 0] + rng.normal(scale=0.1, size=2000)
    Q = np.vstack([X[:5], rng.normal(size=(40, 16))])

    def weights(dist):
        return np.exp(-dist)

    cls = getattr(patched if layer == "L2" else mlxlearn.neighbors, estimator)
    ref = getattr(sk_neighbors, estimator)(n_neighbors=7, weights=weights).fit(X, y)
    model = cls(n_neighbors=7, weights=weights).fit(X, y)
    refit = cls(n_neighbors=7).fit(X, y).set_params(weights=weights)
    for fitted in (model, refit):
        assert fitted.execution_backend_ != "sklearn"
        np.testing.assert_allclose(getattr(fitted, method)(Q), getattr(ref, method)(Q), atol=1e-6)


def test_all_zero_callable_weights_behave_like_sklearn(rng):
    """The classifier has no vote to take and raises; the regressor returns NaN."""
    X = rng.normal(size=(600, 16))
    y = (X[:, 0] > 0).astype(int)

    def zero(dist):
        return np.zeros_like(dist)

    for cls in (KNeighborsClassifier, sk_neighbors.KNeighborsClassifier):
        clf = cls(weights=zero).fit(X, y)
        for method in (clf.predict, clf.predict_proba):
            with pytest.raises(ValueError, match="getting zero weights"):
                method(X[:3])
    for cls in (KNeighborsRegressor, sk_neighbors.KNeighborsRegressor):
        with np.errstate(invalid="ignore"):
            assert np.isnan(cls(weights=zero).fit(X, y.astype(float)).predict(X[:3])).all()


@pytest.mark.parametrize(
    "make_index",
    [
        lambda X: sk_neighbors.KDTree(X),
        lambda X: sk_neighbors.BallTree(X),
        lambda X: sk_neighbors.NearestNeighbors().fit(X),
    ],
    ids=["KDTree", "BallTree", "fitted-estimator"],
)
def test_prebuilt_index_is_a_capability_mismatch(make_index, rng):
    """scikit-learn's fit accepts a prebuilt index; mlxlearn used to fail with a TypeError."""
    X = rng.normal(size=(600, 16))
    Q = rng.normal(size=(5, 16))
    index = make_index(X)
    ref = sk_neighbors.NearestNeighbors(n_neighbors=3).fit(index)

    with pytest.warns(mlxlearn.MLXLearnFallbackWarning, match="prebuilt"):
        model = patched.NearestNeighbors(n_neighbors=3).fit(index)
    assert model.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(model.kneighbors(Q)[1], ref.kneighbors(Q)[1])

    with pytest.raises(UnsupportedInputError) as excinfo:
        NearestNeighbors(n_neighbors=3).fit(index)
    assert excinfo.value.reason == "prebuilt-index"


# ----------------------------------------------------------------------------------
# radius queries and fitted attributes
# ----------------------------------------------------------------------------------


def _assert_radius_parity(mine, theirs, radius, *, atol=1e-5):
    """Same neighbor sets, except for points sitting on the radius itself."""
    (m_dist, m_idx), (t_dist, t_idx) = mine, theirs
    for md, mi, td, ti in zip(m_dist, m_idx, t_dist, t_idx, strict=True):
        mine_d = dict(zip(mi.tolist(), md.tolist(), strict=True))
        theirs_d = dict(zip(ti.tolist(), td.tolist(), strict=True))
        for i in set(mine_d) ^ set(theirs_d):
            assert abs(mine_d.get(i, theirs_d.get(i)) - radius) <= atol
        for i in set(mine_d) & set(theirs_d):
            assert abs(mine_d[i] - theirs_d[i]) <= atol


def test_radius_neighbors_matches_sklearn_and_gets_a_float32_query(monkeypatch, rng):
    """The training copy is float32. A float64 query used to miss scikit-learn's
    Cython radius search, which requires matching dtypes, and ran about 4.5x
    slower than stock on its chunked fallback."""
    X = rng.normal(size=(2000, 16))
    Q = rng.normal(size=(50, 16))
    radius = 4.5
    model = NearestNeighbors(radius=radius).fit(X)
    ref = sk_neighbors.NearestNeighbors(radius=radius, algorithm="brute").fit(
        X.astype(np.float32).astype(np.float64)
    )

    _assert_radius_parity(model.radius_neighbors(Q), ref.radius_neighbors(Q), radius)
    _assert_radius_parity(model.radius_neighbors(), ref.radius_neighbors(), radius)
    sorted_dist, _ = model.radius_neighbors(Q, sort_results=True)
    assert all(np.all(np.diff(row) >= 0) for row in sorted_dist)
    for mode in ("connectivity", "distance"):
        graph = model.radius_neighbors_graph(Q, mode=mode)
        expected = ref.radius_neighbors_graph(Q, mode=mode)
        assert abs(graph - expected).max() <= 1e-5

    seen = []
    original = sk_neighbors.NearestNeighbors.radius_neighbors

    def spy(self, X=None, *args, **kwargs):
        seen.append(None if X is None else X.dtype)
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(sk_neighbors.NearestNeighbors, "radius_neighbors", spy)
    mlxlearn.clear_backend_diagnostics()
    model.radius_neighbors(Q)
    assert seen == [np.dtype(np.float32)] == [model._fit_X.dtype]
    assert [(e.operation, e.reason) for e in _events()] == [("radius_neighbors", "sticky")]


@pytest.mark.parametrize(
    "metric, expected",
    [("minkowski", "euclidean"), ("euclidean", "euclidean"), ("l2", "l2"), ("sqeuclidean", "sqeuclidean")],
)
def test_effective_metric_matches_sklearn(metric, expected, rng):
    X = rng.normal(size=(600, 16))
    model = NearestNeighbors(metric=metric).fit(X)
    ref = sk_neighbors.NearestNeighbors(metric=metric).fit(X)
    assert model.execution_backend_ != "sklearn"
    assert model.effective_metric_ == ref.effective_metric_ == expected
    assert model.effective_metric_params_ == ref.effective_metric_params_


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_sqeuclidean_matches_sklearn(weights, rng):
    """The kernel's own squared distances: same neighbors, distances squared, and
    ``weights='distance'`` weighting by ``1/d**2`` exactly as scikit-learn does."""
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 3, 600)
    Q = np.vstack([X[:10], rng.normal(size=(40, 16))])
    kwargs = {"n_neighbors": 7, "metric": "sqeuclidean", "weights": weights}

    ref = sk_neighbors.KNeighborsClassifier(**kwargs).fit(X, y)
    model = KNeighborsClassifier(**kwargs).fit(X, y)
    assert model.execution_backend_ != "sklearn"

    assert_neighbor_parity(*model.kneighbors(Q), *ref.kneighbors(Q), atol=1e-5)
    np.testing.assert_array_equal(model.predict(Q), ref.predict(Q))
    np.testing.assert_allclose(model.predict_proba(Q), ref.predict_proba(Q), atol=1e-5)


@pytest.mark.parametrize("p", [2, 1, None])
def test_minkowski_p_in_metric_params_is_resolved_like_sklearn(p, rng):
    """``metric_params={'p': 2}`` is Euclidean, whatever ``p`` says, as in scikit-learn.

    It used to be refused outright, and ``p=1`` alongside it was refused as
    Manhattan although scikit-learn ignores ``p`` then and computes Euclidean.
    """
    X = rng.normal(size=(600, 16))
    Q = rng.normal(size=(20, 16))
    kwargs = {"n_neighbors": 5, "metric": "minkowski", "p": p, "metric_params": {"p": 2}}

    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        ref = sk_neighbors.NearestNeighbors(**kwargs).fit(X)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        model = NearestNeighbors(**kwargs).fit(X)

    assert model.execution_backend_ != "sklearn"
    assert [w.category for w in got] == [w.category for w in expected]
    assert (SyntaxWarning in [w.category for w in got]) is (p is not None)
    assert model.effective_metric_ == ref.effective_metric_ == "euclidean"
    assert model.effective_metric_params_ == ref.effective_metric_params_ == {}
    assert_neighbor_parity(*model.kneighbors(Q), *ref.kneighbors(Q))


# ----------------------------------------------------------------------------------
# refit hygiene and diagnostics
# ----------------------------------------------------------------------------------


def test_refit_onto_sklearn_drops_the_previous_index(classification_data):
    """The old index — a float32 copy of the training set, plus device buffers —
    used to survive a refit that fell back, in memory and in every pickle."""
    sparse = pytest.importorskip("scipy.sparse")
    X, y = classification_data
    with tuning_context(force_backend="mlx"):
        model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert hasattr(model, "_index_")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mlxlearn.MLXLearnFallbackWarning)
        model.fit(sparse.csr_matrix(X[:500]), y[:500])

    assert model.execution_backend_ == "sklearn"
    assert not hasattr(model, "_index_")
    assert model._fit_X.shape == (500, X.shape[1])


def test_failed_refit_keeps_nothing_of_the_previous_fit(classification_data):
    from sklearn.exceptions import NotFittedError

    sparse = pytest.importorskip("scipy.sparse")
    X, y = classification_data
    model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    with pytest.raises(SparseInputError):
        model.fit(sparse.csr_matrix(X), y)

    assert not hasattr(model, "_index_")
    assert not hasattr(model, "_fit_X")
    # An incomplete fit reports the estimator unfitted (BackendMixin.__sklearn_is_fitted__):
    # scikit-learn's NotFittedError, which its callers catch, not a bare MLXLearnError.
    with pytest.raises(NotFittedError):
        model.predict(X[:3])


@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"], ids=["mlx-fitted", "sklearn-fitted"])
@pytest.mark.parametrize(
    "cls, method",
    [
        (patched.KNeighborsClassifier, "predict"),
        (patched.KNeighborsClassifier, "predict_proba"),
        (patched.KNeighborsClassifier, "kneighbors"),
        (patched.KNeighborsRegressor, "predict"),
    ],
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_one_inference_event_per_public_call(cls, method, algorithm, classification_data):
    """Named after the call, whichever backend fitted the model.

    The MLX path recorded ``predict`` as ``kneighbors``; the scikit-learn path
    recorded it twice, as ``predict`` and again as the ``kneighbors`` stock
    ``predict`` makes back into the class.
    """
    X, y = classification_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mlxlearn.MLXLearnFallbackWarning)
        model = cls(n_neighbors=5, algorithm=algorithm).fit(X, y)

    mlxlearn.clear_backend_diagnostics()
    getattr(model, method)(X[:10])

    events = _events()
    assert [(e.operation, e.reason) for e in events] == [(method, "sticky")]
    assert events[0].backend == model.execution_backend_


@pytest.mark.parametrize("cls", LAYER2, ids=_layer_id)
def test_patched_mlx_path_passes_sklearn_checks(cls):
    """At check_estimator's data sizes patched mode hands every fit to scikit-learn,
    so the Layer 2 compliance run never reaches mlxlearn's own fit or inference.
    Forced onto MLX, it does — multi-output targets included."""
    with tuning_context(force_backend="mlx"):
        results = check_estimator(cls(), on_fail=None)
    failures = [
        f"{r['check_name']}: {r['exception']}" for r in results if r["status"] == "failed"
    ]
    assert not failures, "\n".join(failures)


# ----------------------------------------------------------------------------------
# patched dispatch -- decided from the shape, before touching the data
# ----------------------------------------------------------------------------------


#: The size floors lifted, so the narrow-data handoff is what decides.
_NO_SIZE_FLOOR = {"knn_min_train_samples": 0, "knn_min_work": 0, "knn_narrow_min_samples": 0}


@pytest.mark.parametrize(
    "shape, tuning, reason",
    [
        ((120, 5), {}, "below-crossover"),
        ((5000, 3), _NO_SIZE_FLOOR, "tree-wins"),
        ((5000, 32), {"knn_min_train_samples": 0, "knn_min_work": 0}, "below-crossover"),
    ],
    ids=["small", "narrow", "short"],
)
def test_patched_handoff_does_not_scan_or_copy_the_data(shape, tuning, reason, monkeypatch, rng):
    """Patched mode decides these from ``X.shape`` alone and records one event.

    It used to scan for float32 overflow and make a validated float32 copy first —
    pure overhead for work scikit-learn validates again, which made the handoff
    slower than the stock fit it stands in for.
    """
    from mlxlearn.neighbors import _base

    def forbidden(*args, **kwargs):
        raise AssertionError("a fit handed to scikit-learn touched the data first")

    for name in ("check_float32_range", "validate_data", "build_index"):
        monkeypatch.setattr(_base, name, forbidden)

    X = rng.normal(size=shape)
    y = (X[:, 0] > 0).astype(int)
    with warnings.catch_warnings(), tuning_context(**tuning):
        warnings.simplefilter("error", mlxlearn.MLXLearnFallbackWarning)
        model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)

    assert model.execution_backend_ == "sklearn"
    assert [(e.backend, e.reason) for e in _events("fit")] == [("sklearn", reason)]
    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    np.testing.assert_array_equal(model.predict(X[:50]), ref.predict(X[:50]))


def test_patched_low_dimensional_auto_goes_to_sklearns_kd_tree(rng):
    """Stock ``algorithm='auto'`` answers this with a KD-tree, at any size.

    The tree answered single rows faster than an MLX call at every size measured up
    to 8 features, and at 12 features up to 65 536 rows; on data of low intrinsic
    dimension it won every batch (0.12x-0.68x for MLX at 8 and 12 features). The
    size floors are lifted so that it is this rule that decides.
    """
    X = rng.normal(size=(5000, 3))
    Q = rng.normal(size=(20, 3))
    with tuning_context(**_NO_SIZE_FLOOR):
        model = patched.NearestNeighbors(n_neighbors=5).fit(X)

    assert model.execution_backend_ == "sklearn"
    assert model._fit_method == "kd_tree"
    [event] = _events("fit")
    assert event.reason == "tree-wins" and event.fallback
    ref = sk_neighbors.NearestNeighbors(n_neighbors=5).fit(X)
    np.testing.assert_array_equal(model.kneighbors(Q)[1], ref.kneighbors(Q)[1])


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"algorithm": "brute"}, "below-crossover"),
        ({"metric": "sqeuclidean"}, "below-crossover"),
        ({"n_neighbors": 2500}, "below-crossover"),
        ({}, "tree-wins"),
    ],
    ids=["explicit-brute", "no-tree-metric", "k-over-half-n", "kd-tree"],
)
def test_narrow_data_goes_to_sklearn_whatever_it_would_build(kwargs, reason, rng):
    """Below ``knn_min_features``, to the KD-tree where scikit-learn would build one,
    and to its brute force otherwise: at 2-8 features that answers a 16-query batch
    faster than MLX up to 524 288 rows."""
    X = rng.normal(size=(5000, 3))
    with tuning_context(**_NO_SIZE_FLOOR):
        model = patched.NearestNeighbors(**kwargs).fit(X)
    assert model.execution_backend_ == "sklearn"
    [event] = _events("fit")
    assert event.reason == reason
    assert "features" in event.detail


@pytest.mark.parametrize(
    "cls, shape",
    [(NearestNeighbors, (5000, 3)), (patched.NearestNeighbors, (5000, 64))],
    ids=["layer1", "wide"],
)
def test_the_narrow_rule_leaves_everything_else_on_mlx(cls, shape, rng):
    """Layer 1 has no scikit-learn to hand the work to, and from 64 features patched
    mode keeps the search once the size floors are cleared."""
    with tuning_context(**_NO_SIZE_FLOOR):
        model = cls().fit(rng.normal(size=shape))
    assert model.execution_backend_ != "sklearn"


def test_the_narrow_rule_is_a_tuning_knob_that_force_backend_outranks(rng):
    X = rng.normal(size=(5000, 3))
    with tuning_context(**_NO_SIZE_FLOOR, knn_min_features=0):
        assert patched.NearestNeighbors().fit(X).execution_backend_ != "sklearn"
    with tuning_context(**_NO_SIZE_FLOOR, knn_min_features=3):
        assert patched.NearestNeighbors().fit(X).execution_backend_ != "sklearn"
    with tuning_context(**_NO_SIZE_FLOOR, knn_min_features=4):
        assert patched.NearestNeighbors().fit(X).execution_backend_ == "sklearn"
    with tuning_context(force_backend="mlx"):
        assert patched.NearestNeighbors().fit(X).execution_backend_ == "mlx"


def test_the_knn_size_floors_are_both_required(rng):
    """Rows and elements, in both layers: below 4 096 rows a 4096-query batch lost
    at every width measured (2 048 x 256-1 024: 0.44x-0.79x), and below 6 million
    elements batches of 1-32 queries lost at 4 096-8 192 rows and 256-1 024 features
    (0.70x-0.95x), and at 4 194 304 elements single rows only tied (0.90x-0.99x)."""
    for model in (patched.NearestNeighbors(), NearestNeighbors()):
        assert model._below_crossover(4095, 2048)[0]
        assert model._below_crossover(8_192, 512)[0]
        assert model._below_crossover(4_096, 1_024)[0]
        assert not model._below_crossover(16_384, 384)[0]
        assert not model._below_crossover(4_096, 1_465)[0]
        below, detail = model._below_crossover(256, 768)
        assert below and "both required" in detail


def test_patched_dispatch_needs_a_million_rows_below_256_features():
    """Measured with the GPU idle before every call, as the benchmark times it: at
    16-128 features a batch of 8-512 queries lost to stock brute force, on one
    version or both, at most sizes measured below 1 000 000 rows — 400 000 rows at 64
    and 96 features among them; from 1 000 000 rows at 64-128 features every batch
    won or tied on both; at 256 features every batch won or tied from 12 000 rows."""
    model = patched.NearestNeighbors()
    for n_samples, n_features in ((600_000, 64), (400_000, 96), (400_000, 128), (131_072, 64)):
        reason, detail = model._patched_handoff(n_samples, n_features)
        assert reason == "below-crossover", (n_samples, n_features)
        assert "256 features" in detail and "1000000 samples" in detail
        # The size floors alone would have sent every one of these to MLX.
        assert not model._below_crossover(n_samples, n_features)[0]
    for n_samples, n_features in ((1_000_000, 64), (1_000_000, 128), (32_768, 256), (8_192, 768)):
        assert model._patched_handoff(n_samples, n_features) is None
        assert not model._below_crossover(n_samples, n_features)[0]


def test_patched_dispatch_needs_8192_rows_from_256_features():
    """Right after a stock query, whose OpenMP threads are still spinning, an MLX call
    of some mid-size batches ran up to twice as slow as its neighbours, and on 4 096-7 500
    training rows at 2 048-4 096 features stock brute force answered those batches in
    about as long (table ``knn-batches``). So patched mode needs 8 192 rows from 256
    features, which the size floors alone did not ask for."""
    model = patched.NearestNeighbors()
    for n_samples, n_features in ((4_096, 2_048), (6_000, 4_096), (7_500, 4_096), (8_191, 1_024)):
        assert not model._below_crossover(n_samples, n_features)[0]
        reason, detail = model._patched_handoff(n_samples, n_features)
        assert reason == "below-crossover" and "8192 samples" in detail, (n_samples, n_features)
    for n_samples, n_features in ((8_192, 2_048), (8_192, 4_096), (12_000, 512), (32_768, 256)):
        assert model._patched_handoff(n_samples, n_features) is None


@pytest.mark.parametrize("cls", [NearestNeighbors, patched.NearestNeighbors])
def test_the_wide_row_rule_is_patched_only(cls, rng):
    X = rng.normal(size=(6_000, 1_024))
    model = cls().fit(X)
    assert (model.execution_backend_ == "sklearn") == cls._mlxlearn_allow_fallback


def test_patched_dispatch_reads_the_fitted_n_neighbors():
    """The size rules were measured at k = 10. At k = 100 a 1 024-query batch lost at
    1 000 000 x 64 on both versions, and at 2 048-4 096 features batches of 32-4 096
    queries; k = 50 won or tied everywhere from 150 rows per neighbor, which the row
    floors guarantee. So a patched fit needs at most 50 neighbors."""
    for k, n_samples, n_features, handed in (
        (10, 1_000_000, 64, False),
        (50, 1_000_000, 64, False),
        (51, 1_000_000, 64, True),
        (100, 1_000_000, 128, True),
        (50, 8_192, 4_096, False),
        (51, 8_192, 2_048, True),
        (200, 8_192, 1_024, True),
    ):
        model = patched.KNeighborsClassifier(n_neighbors=k)
        assert not model._below_crossover(n_samples, n_features)[0]
        handoff = model._patched_handoff(n_samples, n_features)
        assert (handoff is not None) == handed, (k, n_samples, n_features)
        if handed:
            assert handoff[0] == "below-crossover" and f"n_neighbors={k}" in handoff[1]


#: Every size gate lifted, so a small index is fitted on MLX by the ordinary route.
_LIFTED = {"knn_min_train_samples": 0, "knn_min_work": 0, "knn_min_features": 0,
           "knn_narrow_min_samples": 0, "knn_probe_max_uncertified": 1.0}


@pytest.mark.parametrize("cls", [NearestNeighbors, patched.NearestNeighbors])
def test_the_k_rule_is_patched_only_and_reads_the_query(cls, rng):
    """Layer 1 searches on MLX whatever k is. A patched fit decides by its own
    ``n_neighbors``, and a query asking for more than the rules keep on MLX is answered
    by scikit-learn's brute force, with stock scikit-learn's answer."""
    X = rng.normal(size=(3000, 16))
    Q = rng.normal(size=(7, 16))
    with tuning_context(**_LIFTED):
        many = cls(n_neighbors=60).fit(X)
        few = cls(n_neighbors=5).fit(X)
    ref = sk_neighbors.NearestNeighbors(n_neighbors=60).fit(X)
    if cls is NearestNeighbors:
        assert many.execution_backend_ != "sklearn"
        idx = few.kneighbors(Q, n_neighbors=60)[1]
        assert _events("kneighbors")[-1].backend == few.execution_backend_
        np.testing.assert_array_equal(idx, ref.kneighbors(Q)[1])
        return
    assert many.execution_backend_ == "sklearn"
    assert many.fit_event_.reason == "below-crossover"
    assert "n_neighbors=60" in many.fit_event_.detail
    assert few.execution_backend_ != "sklearn"

    mlxlearn.clear_backend_diagnostics()
    dist, idx = few.kneighbors(Q, n_neighbors=60)
    [event] = _events()
    assert (event.operation, event.backend, event.reason) == (
        "kneighbors", "sklearn", "below-crossover"
    )
    assert "n_neighbors=60" in event.detail and event.shape == (7, 16)
    ref_dist, ref_idx = ref.kneighbors(Q)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-12)
    # The training rows themselves, each excluded from its own neighbors, as stock does.
    np.testing.assert_array_equal(few.kneighbors(n_neighbors=60)[1], ref.kneighbors()[1])
    # Within the rules the query stays on MLX.
    few.kneighbors(Q, n_neighbors=50)
    assert _events("kneighbors")[-1].backend == few.execution_backend_
    assert few.execution_backend_ != "sklearn"


def test_a_handed_off_query_keeps_float64_precision_and_is_not_pickled(rng):
    """The handed-off query runs on the float32 rows plus their float64 residuals,
    translated by the index's mean, so timestamps within an hour — which float32 alone
    rounds to 128-second steps, and which stock scikit-learn's search, expanding squared
    distances around the origin, returns at distance 0 — keep their exact neighbors.
    That float64 copy is derived state, and not pickled."""
    import pickle

    X = np.hstack([
        1.7e9 + rng.uniform(0.0, 3600.0, size=(3000, 1)),
        rng.normal(size=(3000, 15)),
    ])
    Q = X[:9] + rng.normal(scale=0.5, size=(9, 16))
    with tuning_context(**_LIFTED):
        model = patched.NearestNeighbors(n_neighbors=5).fit(X)
    assert model.execution_backend_ != "sklearn"
    exact = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(exact, axis=1, kind="stable")[:, :80]
    dist, idx = model.kneighbors(Q, n_neighbors=80)
    np.testing.assert_array_equal(idx, order)
    np.testing.assert_allclose(dist, np.take_along_axis(exact, order, axis=1), atol=1e-5)
    assert model._index_._host64 is not None
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded._index_._host64 is None and "_host_brute_" not in vars(loaded)
    np.testing.assert_array_equal(loaded.kneighbors(Q, n_neighbors=80)[1], order)


def test_a_handed_off_query_checks_feature_names_like_stock(rng):
    """A frame is validated against the fitted names, and an array queried on a model
    fitted on a frame warns, as stock scikit-learn does; a float32 array is served as
    float64."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(rng.normal(size=(3000, 16)), columns=[f"f{i}" for i in range(16)])
    Q = X.iloc[:5] + 0.1
    with tuning_context(**_LIFTED):
        model = patched.NearestNeighbors(n_neighbors=5).fit(X)
        plain = patched.NearestNeighbors(n_neighbors=5).fit(X.to_numpy())
    ref = sk_neighbors.NearestNeighbors(n_neighbors=60).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(model.kneighbors(Q, n_neighbors=60)[1], ref.kneighbors(Q)[1])
        np.testing.assert_array_equal(
            plain.kneighbors(Q.to_numpy(np.float32), n_neighbors=60)[1],
            sk_neighbors.NearestNeighbors(n_neighbors=60).fit(X.to_numpy()).kneighbors(
                Q.to_numpy(np.float32).astype(np.float64)
            )[1],
        )
    with pytest.warns(UserWarning, match="feature names"):
        model.kneighbors(Q.to_numpy(), n_neighbors=60)
    with pytest.raises(ValueError, match="features"):
        plain.kneighbors(Q.to_numpy()[:, :8], n_neighbors=60)


def test_a_query_after_set_params_is_held_to_the_same_rule(rng):
    X = rng.normal(size=(3000, 16))
    y = (X[:, 0] > 0).astype(int)
    Q = rng.normal(size=(11, 16))
    with tuning_context(**_LIFTED):
        model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert model.execution_backend_ != "sklearn"
    model.set_params(n_neighbors=75)
    mlxlearn.clear_backend_diagnostics()
    expected = sk_neighbors.KNeighborsClassifier(n_neighbors=75).fit(X, y).predict_proba(Q)
    np.testing.assert_allclose(model.predict_proba(Q), expected)
    [event] = _events()
    assert (event.operation, event.backend) == ("predict_proba", "sklearn")


def test_a_forced_fit_keeps_its_queries_on_mlx(rng):
    X = rng.normal(size=(3000, 16))
    with tuning_context(force_backend="mlx"):
        model = patched.NearestNeighbors(n_neighbors=5).fit(X)
    mlxlearn.clear_backend_diagnostics()
    model.kneighbors(X[:4], n_neighbors=60)
    [event] = _events()
    assert event.backend == "mlx" and event.reason in ("sticky", "gpu-unavailable")


@pytest.mark.parametrize("cls", [NearestNeighbors, patched.NearestNeighbors])
def test_the_row_rule_below_256_features_is_patched_only(cls, rng):
    """Layer 1 has no scikit-learn to hand the work to, and its CPU path is slower
    than the GPU at every batch; patched mode records the handoff as a size decision."""
    X = rng.normal(size=(5000, 64))
    with tuning_context(knn_min_train_samples=0, knn_min_work=0):
        model = cls().fit(X)
    if cls is NearestNeighbors:
        assert model.execution_backend_ != "sklearn"
        return
    assert model.execution_backend_ == "sklearn"
    [event] = _events("fit")
    assert event.reason == "below-crossover" and "1000000 samples" in event.detail
    with tuning_context(knn_min_train_samples=0, knn_min_work=0, knn_narrow_min_samples=5000):
        assert cls().fit(X).execution_backend_ != "sklearn"
    with tuning_context(
        knn_min_train_samples=0, knn_min_work=0, knn_wide_features=64, knn_wide_min_samples=0
    ):
        assert cls().fit(X).execution_backend_ != "sklearn"


#: The size and width floors lifted, so that the data rule is what decides.
_ONLY_DATA = {
    "knn_min_train_samples": 0, "knn_min_work": 0, "knn_min_features": 0,
    "knn_narrow_min_samples": 0,
}


def _awkward(kind, rng, n=6000, d=32):
    if kind == "dups":  # every row about 20 times
        return rng.normal(size=(n // 20, d))[rng.integers(0, n // 20, size=n)]
    if kind == "unscaled":  # one column in raw units beside standardized ones
        X = rng.normal(size=(n, d))
        X[:, 0] = rng.uniform(0.0, 1e4, size=n)
        return X
    return rng.poisson(2.0, size=(n, d)).astype(np.float64)  # counts


@pytest.mark.parametrize("kind", ["dups", "unscaled", "counts"])
def test_patched_dispatch_hands_data_the_fast_search_cannot_serve_to_sklearn(kind, rng):
    """The size rules were calibrated on Gaussian data. On duplicated rows or a column
    on a far larger scale than the rest the first float32 pass cannot certify most
    rows, and every query is searched again; integer data narrower than 256 features
    takes an exact path that lost to stock brute force from 64-query batches. The fit
    probes its own rows, and patched mode hands such data to scikit-learn."""
    X = _awkward(kind, rng)
    y = (X[:, 1] > np.median(X[:, 1])).astype(int)
    with tuning_context(**_ONLY_DATA):
        model = patched.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert model.execution_backend_ == "sklearn"
    [event] = _events("fit")
    assert event.reason == "data-probe" and event.fallback
    assert event.extra["integer_path"] == (kind == "counts")
    if kind != "counts":
        assert event.extra["probe_uncertified"] > 0.5 * event.extra["probe_rows"]
    ref = sk_neighbors.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    np.testing.assert_array_equal(model.predict(X[:50]), ref.predict(X[:50]))
    np.testing.assert_array_equal(model.classes_, ref.classes_)


@pytest.mark.parametrize("kind", ["dups", "counts"])
def test_the_data_rule_is_patched_only_and_a_tuning_knob(kind, rng):
    """Layer 1 searches on MLX regardless, a forced backend is not second-guessed, and
    the rule has its two knobs: the uncertified fraction, and the width from which
    integer data stays on MLX."""
    X = _awkward(kind, rng)
    with tuning_context(**_ONLY_DATA):
        assert NearestNeighbors().fit(X).execution_backend_ != "sklearn"
    with tuning_context(force_backend="mlx"):
        assert patched.NearestNeighbors().fit(X).execution_backend_ == "mlx"
    knob = {"knn_probe_max_uncertified": 1.0} if kind == "dups" else {
        "knn_wide_features": 32, "knn_wide_min_samples": 0,
    }
    with tuning_context(**_ONLY_DATA, **knob):
        assert patched.NearestNeighbors().fit(X).execution_backend_ != "sklearn"


def test_gaussian_data_passes_the_probe(rng):
    """What the size rules were calibrated on stays on MLX; the probe's count goes on
    the fit's own event only when it hands the fit over."""
    with tuning_context(**_ONLY_DATA):
        model = patched.NearestNeighbors().fit(rng.normal(size=(6000, 32)))
    assert model.execution_backend_ != "sklearn"
    assert model.fit_event_.reason == "selected"


def test_capability_mismatch_outranks_the_tree_rule(rng):
    """A metric mlxlearn cannot serve is reported as that, with its warning."""
    X = rng.normal(size=(5000, 3))
    with pytest.warns(mlxlearn.MLXLearnFallbackWarning, match="metric"):
        model = patched.NearestNeighbors(metric="manhattan").fit(X)
    assert model.execution_backend_ == "sklearn"
    [event] = _events("fit")
    assert event.reason == "unsupported-parameter:metric"
