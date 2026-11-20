"""Independent Lloyd/nearest-center oracles and sklearn KMeans contracts."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans as SKKMeans
from sklearn.exceptions import ConvergenceWarning, NotFittedError


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


@pytest.fixture
def estimator():
    from mlxlearn.cluster import KMeans

    return KMeans


def direct_distances(x, centers):
    return np.square(x.astype(np.float64)[:, None, :] - centers.astype(np.float64)).sum(axis=2)


def separated_data():
    rng = np.random.default_rng(410)
    centers = np.array([[-6, -2, 1], [4, 4, -3], [1, -5, 5]], dtype=np.float32)
    x = np.concatenate([center + rng.normal(scale=.35, size=(47, 3)) for center in centers]).astype(np.float32)
    return x, x[[0, 47, 94]].copy()


@pytest.mark.gpu
@pytest.mark.parametrize("max_iter", [1, 2, 20])
@pytest.mark.parametrize("statistics_batch_size", [13, 65536])
def test_explicit_initialization_matches_sklearn_lloyd(metal, estimator, max_iter, statistics_batch_size):
    x, initial = separated_data()
    expected = SKKMeans(3, init=initial, n_init=1, max_iter=max_iter, tol=0, algorithm="lloyd").fit(x)
    actual = estimator(3, init=initial, n_init=1, max_iter=max_iter, tol=0,
                       statistics_batch_size=statistics_batch_size, assignment_batch_size=19).fit(x)
    np.testing.assert_allclose(actual.cluster_centers_, expected.cluster_centers_, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(actual.labels_, expected.labels_)
    assert actual.inertia_ == pytest.approx(expected.inertia_, rel=3e-6, abs=3e-6)
    assert actual.n_iter_ == expected.n_iter_
    assert actual.cluster_centers_.dtype == np.float32


@pytest.mark.gpu
def test_one_iteration_returns_fresh_labels_and_inertia_not_online_labels(metal, estimator):
    x = np.array([0, 2, 3, 8, 9, 10], dtype=np.float32)[:, None]
    model = estimator(2, init=x[[0, 2]].copy(), n_init=1, max_iter=1, tol=0).fit(x)
    np.testing.assert_allclose(model.cluster_centers_[:, 0], [0, 6.4], atol=1e-6)
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 1, 1, 1])
    distances = direct_distances(x, model.cluster_centers_)
    assert model.inertia_ == pytest.approx(distances.min(axis=1).sum(), rel=3e-6)
    np.testing.assert_array_equal(model.predict(x), model.labels_)


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [.125, 1, 8])
@pytest.mark.parametrize("tol", [0, .2, 100])
def test_variance_scaled_tolerance_matches_sklearn(metal, estimator, scale, tol):
    rng = np.random.default_rng(761)
    x = (rng.normal(size=(89, 5)) * scale).astype(np.float32)
    initial = x[[1, 11, 56, 68]].copy()
    expected = SKKMeans(4, init=initial, n_init=1, max_iter=30, tol=tol, algorithm="lloyd").fit(x)
    actual = estimator(4, init=initial, n_init=1, max_iter=30, tol=tol).fit(x)
    assert actual.n_iter_ == expected.n_iter_
    np.testing.assert_array_equal(actual.labels_, expected.labels_)
    np.testing.assert_allclose(actual.cluster_centers_, expected.cluster_centers_, rtol=8e-6, atol=4e-6)


@pytest.mark.gpu
def test_empty_clusters_relocate_farthest_rows_and_match_sklearn(metal, estimator):
    # Distinct farthest residuals avoid an unspecified tie-order dependency.
    x = np.array([-13, -11, -10, -8, -3, 2, 8, 10, 13], dtype=np.float32)[:, None]
    initial = np.full((3, 1), -13, dtype=np.float32)
    expected = SKKMeans(3, init=initial, n_init=1, max_iter=1, tol=0, algorithm="lloyd").fit(x)
    actual = estimator(3, init=initial, n_init=1, max_iter=1, tol=0, statistics_batch_size=4).fit(x)
    np.testing.assert_allclose(np.sort(actual.cluster_centers_[:, 0]), np.sort(expected.cluster_centers_[:, 0]), atol=2e-6)
    assert actual.inertia_ == pytest.approx(expected.inertia_, rel=3e-6)
    assert np.isfinite(actual.cluster_centers_).all()


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["identical", "fewer_distinct", "singleton"])
def test_duplicates_and_degenerate_inputs_terminate_without_invalid_centers(metal, estimator, kind):
    if kind == "identical":
        x = np.full((19, 3), 7, np.float32)
        k = 5
    elif kind == "fewer_distinct":
        x = np.array([[0, 0], [1, 1], [1, 1], [1, 0], [0, 1]], np.float32)
        k = len(x)
    else:
        x = np.array([[4, -7, 2]], np.float32)
        k = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = estimator(k, init=x[:k].copy(), n_init=1, max_iter=12, tol=0).fit(x)
    assert model.n_iter_ <= 3
    assert np.isfinite(model.cluster_centers_).all()
    assert model.inertia_ == pytest.approx(0, abs=1e-7)
    np.testing.assert_array_equal(model.labels_, direct_distances(x, model.cluster_centers_).argmin(axis=1))


@pytest.mark.gpu
def test_one_cluster_is_mean_and_transform_is_euclidean_distance(metal, estimator):
    x = np.array([[1, -3], [5, 4], [9, -1], [-7, 8]], np.float32)
    model = estimator(1, init=x[[0]].copy(), n_init=1).fit(x)
    np.testing.assert_allclose(model.cluster_centers_, x.mean(axis=0, keepdims=True), atol=1e-6)
    expected = np.sqrt(direct_distances(x, model.cluster_centers_))
    np.testing.assert_allclose(model.transform(x), expected, rtol=3e-6, atol=2e-6)
    assert model.score(x) == pytest.approx(-np.square(expected).sum(), rel=3e-6)
    assert model.n_features_in_ == 2


@pytest.mark.gpu
@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_restarts_are_deterministic_and_best_inertia_is_no_worse_than_first(metal, estimator, init):
    x = np.random.default_rng(16).normal(size=(113, 4)).astype(np.float32)
    one = estimator(5, init=init, n_init=1, max_iter=5, random_state=42).fit(x)
    many = estimator(5, init=init, n_init=4, max_iter=5, random_state=42).fit(x)
    repeated = estimator(5, init=init, n_init=4, max_iter=5, random_state=42).fit(x)
    assert many.inertia_ <= one.inertia_ * (1 + 2e-6)
    np.testing.assert_array_equal(many.labels_, repeated.labels_)
    np.testing.assert_allclose(many.cluster_centers_, repeated.cluster_centers_, rtol=0, atol=2e-6)
    assert many.inertia_ == pytest.approx(repeated.inertia_, rel=2e-6)


@pytest.mark.gpu
def test_noncontiguous_float64_input_is_copied_as_fp32_without_mutating_caller(metal, estimator):
    x, initial = separated_data()
    storage = np.column_stack([x, np.zeros_like(x)]).astype(np.float64)
    view = storage[:, :3]
    original, init_original = storage.copy(), initial.copy()
    model = estimator(3, init=initial, n_init=1, max_iter=4).fit(view)
    np.testing.assert_array_equal(storage, original)
    np.testing.assert_array_equal(initial, init_original)
    assert model.cluster_centers_.dtype == np.float32
    np.testing.assert_array_equal(model.predict(view), model.labels_)


@pytest.mark.gpu
def test_common_offset_does_not_break_cluster_assignments(metal, estimator):
    x, initial = separated_data()
    shifted = x + np.float32(8192)
    expected = SKKMeans(3, init=initial + np.float32(8192), n_init=1, max_iter=20, tol=0).fit(shifted)
    actual = estimator(3, init=initial + np.float32(8192), n_init=1, max_iter=20, tol=0).fit(shifted)
    np.testing.assert_array_equal(actual.labels_, expected.labels_)
    np.testing.assert_allclose(actual.cluster_centers_, expected.cluster_centers_, rtol=0, atol=.001)
    assert actual.inertia_ == pytest.approx(expected.inertia_, rel=3e-5)


@pytest.mark.gpu
def test_clone_pickle_and_fit_predict_preserve_public_estimator_contract(metal, estimator):
    x, initial = separated_data()
    original = estimator(3, init=initial, n_init=1, random_state=17, max_iter=5)
    copied = clone(original)
    assert copied.get_params()["random_state"] == 17
    assert not hasattr(copied, "cluster_centers_")
    labels = original.fit_predict(x)
    np.testing.assert_array_equal(labels, original.labels_)
    restored = pickle.loads(pickle.dumps(original))
    np.testing.assert_array_equal(restored.predict(x), original.predict(x))
    assert restored.score(x) == pytest.approx(original.score(x), rel=2e-6)
    np.testing.assert_allclose(copied.fit(x).cluster_centers_, original.cluster_centers_, atol=2e-6)


@pytest.mark.parametrize("kwargs", [
    {"n_clusters": 0}, {"n_clusters": True}, {"n_clusters": 1.5}, {"max_iter": 0},
    {"max_iter": 1.5}, {"tol": -1}, {"tol": float("nan")}, {"tol": float("inf")},
    {"n_init": 0}, {"n_init": "bad"}, {"n_init": True}, {"init": "bad"},
    {"random_state": -1}, {"random_state": 2**32}, {"statistics_batch_size": 0},
    {"assignment_batch_size": 0},
])
def test_invalid_parameters_fail_before_training(estimator, kwargs):
    options = {"n_clusters": 2, "n_init": 1, **kwargs}
    with pytest.raises((ValueError, TypeError)):
        estimator(**options).fit(np.zeros((5, 2), np.float32))


@pytest.mark.parametrize("x", [
    np.empty((0, 2), np.float32), np.empty((4, 0), np.float32), np.ones(4, np.float32),
    np.array([[1, np.nan], [2, 3]], np.float32), np.array([[1, np.inf], [2, 3]], np.float32),
    np.full((4, 2), 1e30, np.float32),
])
def test_invalid_data_rejected(estimator, x):
    with pytest.raises((ValueError, TypeError)):
        estimator(1, n_init=1).fit(x)


def test_unfitted_inference_and_unsupported_weights_fail_explicitly(estimator):
    x = np.ones((4, 2), np.float32)
    model = estimator(2)
    for method in (model.predict, model.transform, model.score):
        with pytest.raises(NotFittedError):
            method(x)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        model.fit(x, sample_weight=np.ones(len(x)))


@pytest.mark.gpu
def test_initialization_and_inference_shape_validation(metal, estimator):
    x, initial = separated_data()
    with pytest.raises(ValueError):
        estimator(len(x) + 1, n_init=1).fit(x)
    for bad in (initial[:2], np.zeros((3, 4)), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError):
            estimator(3, init=bad, n_init=1).fit(x)
    model = estimator(3, init=initial, n_init=1).fit(x)
    for method in (model.predict, model.transform, model.score):
        with pytest.raises(ValueError):
            method(np.ones((5, 2), np.float32))


@pytest.mark.gpu
def test_prepared_input_owns_snapshot_and_restarts_cleanly(metal, estimator):
    x, initial = separated_data()
    original_x, original_initial = x.copy(), initial.copy()
    params = {"n_clusters": 3, "init": initial, "n_init": 1, "max_iter": 5, "tol": 0, "random_state": 19}
    model = estimator(**params)
    prepared = model.prepare_device(x)
    assert isinstance(prepared.data, metal.array)
    np.testing.assert_array_equal(np.asarray(prepared.data), original_x)
    x.fill(1000)
    initial.fill(-1000)
    first = model.train_device(prepared)
    second = estimator(**params).train_device(prepared)
    assert isinstance(first.centers, metal.array) and isinstance(first.labels, metal.array)
    assert first.labels.dtype == metal.int64
    assert first.assignments is first.labels
    assert not hasattr(model, "cluster_centers_")
    expected = SKKMeans(3, init=original_initial, n_init=1, max_iter=5, tol=0).fit(original_x)
    np.testing.assert_allclose(np.asarray(first.centers), expected.cluster_centers_, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(np.asarray(first.labels), expected.labels_)
    np.testing.assert_array_equal(np.asarray(first.labels), np.asarray(second.labels))
    np.testing.assert_allclose(np.asarray(first.centers), np.asarray(second.centers), atol=2e-6)
    assert float(first.inertia.item()) == pytest.approx(expected.inertia_, rel=3e-6)
    assert first.n_iter == expected.n_iter_
    assert first.convergence_reason in {"labels", "tolerance", "max_iter"}
    assert len(first.history) == first.n_iter
    np.testing.assert_array_equal(np.asarray(prepared.data), original_x)
    with pytest.raises(ValueError):
        estimator(**{**params, "max_iter": 6}).train_device(prepared)


@pytest.mark.gpu
@pytest.mark.parametrize("init,expected_restarts", [("random", 10), ("k-means++", 1)])
def test_auto_n_init_resolves_expected_restart_count(metal, estimator, init, expected_restarts):
    x, _ = separated_data()
    prepared = estimator(3, init=init, n_init="auto", random_state=44, max_iter=2).prepare_device(x)
    assert len(prepared.initial) == expected_restarts


@pytest.mark.gpu
def test_explicit_centers_warn_and_use_one_restart(metal, estimator):
    x, initial = separated_data()
    with pytest.warns(RuntimeWarning, match="init|initial|n_init"):
        prepared = estimator(3, init=initial, n_init=5).prepare_device(x)
    assert len(prepared.initial) == 1


@pytest.mark.gpu
def test_copy_x_false_does_not_mutate_input(metal, estimator):
    x, initial = separated_data()
    saved = x.copy()
    estimator(3, init=initial, n_init=1, copy_x=False, algorithm="lloyd").fit(x)
    np.testing.assert_array_equal(x, saved)


def test_unsupported_algorithm_is_rejected(estimator):
    with pytest.raises((ValueError, NotImplementedError)):
        estimator(2, algorithm="elkan").fit(np.ones((5, 2), np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("stage", ["input", "initialization", "inference"])
def test_mlx_nan_is_rejected_even_if_minmax_reduction_ignores_nan(metal, estimator, stage):
    x, initial = separated_data()
    if stage == "input":
        x[3, 1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimator(3, init=initial, n_init=1).prepare_device(metal.array(x))
    elif stage == "initialization":
        initial[1, 2] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimator(3, init=metal.array(initial), n_init=1).prepare_device(x)
    else:
        model = estimator(3, init=initial, n_init=1).fit(x)
        x[3, 1] = np.nan
        for method in (model.predict, model.transform, model.score):
            with pytest.raises(ValueError, match="finite"):
                method(metal.array(x))


@pytest.mark.gpu
def test_prepared_mlx_input_is_an_owned_snapshot(metal, estimator):
    x, initial = separated_data()
    resident = metal.array(x)
    prepared = estimator(3, init=initial, n_init=1).prepare_device(resident)
    resident[0] = metal.array([123, 456, 789], dtype=metal.float32)
    np.testing.assert_array_equal(np.asarray(prepared.data), x)


def test_compacted_batches_use_four_row_counts_per_power_of_two():
    from mlxlearn.cluster._k_means import _compacted_rows

    assert [_compacted_rows(rows) for rows in (0, 1, 128, 129, 160, 161, 255, 256, 257, 3585, 4096, 4097)] == [
        128, 128, 128, 160, 160, 192, 256, 256, 320, 4096, 4096, 5120]
    sizes = [_compacted_rows(rows) for rows in range(1, 2**16 + 1)]
    assert sizes == sorted(sizes)
    assert all(rows <= size < max(1.25 * rows, 129) for rows, size in enumerate(sizes, 1))
    # 128 to 65,536 spans nine powers of two.
    assert len(set(sizes)) == 4 * 9 + 1
