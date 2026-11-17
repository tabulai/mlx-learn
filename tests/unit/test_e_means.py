"""Paper recipe, independent numerical oracles, and Metal clustering contracts.

The oracles deliberately use FP64 direct squared differences and host indexed
accumulation, independently of the Metal assignment and reduction algorithms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlxlearn.cluster._recipe import epoch_rows, resolve_recipe, scheduled_alpha


def _direct_assignment(x, centers):
    distance = ((x.astype(np.float64)[:, None] - centers.astype(np.float64)) ** 2).sum(2)
    labels = distance.argmin(1)
    return labels, distance[np.arange(len(x)), labels]


def _statistics(x, labels, k):
    sums = np.zeros((k, x.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, x.astype(np.float64))
    return sums, np.bincount(labels, minlength=k).astype(np.float64)


def _nearest_power_two(value):
    return 2 ** max(0, math.floor(math.log2(value) + 0.5))


@pytest.mark.parametrize("n,k,d,epochs", [(100_000, 256, 96, 3), (3_000_000, 8192, 128, 5)])
def test_paper_scaling_laws_match_table_two(n, k, d, epochs):
    recipe = resolve_recipe(n, k, d, epochs, recipe="paper")
    raw_batch = 0.035 * k**0.13 * n**0.88 * epochs**0.81
    if n > 2_000_000:
        raw_batch /= 8
    batch = min(n, _nearest_power_two(raw_batch))
    alpha = min(1, 0.824 * k**-0.46 * batch**0.90 * n**-0.45 * epochs**-0.11)
    assert recipe["batch_size"] == batch
    assert recipe["alpha0"] == pytest.approx(alpha, rel=1e-12)
    assert recipe["init_usage"] == pytest.approx(1e-10)
    assert recipe["split_epsilon"] == pytest.approx(2**-7)


def test_manual_batch_recomputes_coupled_step_law():
    recipe = resolve_recipe(10_000, 32, 64, 3, batch_size=37, recipe="paper")
    assert recipe["batch_size"] == 37
    assert recipe["alpha0"] == pytest.approx(
        min(1, 0.824 * 32**-0.46 * 37**0.9 * 10_000**-0.45 * 3**-0.11)
    )
    manual = resolve_recipe(10_000, 32, 64, 3, batch_size=37, alpha0=0.2, recipe="paper")
    assert manual["alpha0"] == 0.2


def test_statistics_batch_count_capacity_is_checked_without_allocating_data():
    with pytest.raises(ValueError, match="2\\*\\*24"):
        resolve_recipe(20_000_000, 1, 1, 3, batch_size=20_000_000)
    assert resolve_recipe(20_000_000, 1, 1, 3, batch_size=2**24)["batch_size"] == 2**24


def test_current_faiss_recipe_is_explicitly_distinct():
    paper = resolve_recipe(100_000, 256, 3, 3, recipe="paper")
    faiss = resolve_recipe(100_000, 256, 3, 3, recipe="faiss")
    raw_batch = 0.00439175 * 256**0.134749 * 100_000**0.876473 * 3**0.813344
    assert faiss["batch_size"] == min(_nearest_power_two(raw_batch), 50_000)
    assert faiss["init_usage"] == pytest.approx(2**-16)
    assert faiss["split_epsilon"] == pytest.approx(1e-4)
    assert paper["batch_size"] != faiss["batch_size"]


def test_schedule_has_paper_peak_floor_and_monotonicity():
    steps = np.array([scheduled_alpha(0.4, t, 17) for t in range(18)])
    assert steps[0] == pytest.approx(0.4)
    assert steps[-1] == pytest.approx(0.04)
    assert np.all(np.diff(steps) < 0)
    assert steps[8] == pytest.approx(0.4 * (0.1 + 0.9 * (1 + np.cos(np.pi * 8 / 17)) / 2))


@pytest.mark.parametrize(
    "n,epochs,expected",
    [(11, 3, [11, 11, 11]), (11, 2.5, [11, 6, 11]), (11, 0.5, [6]), (10, 1.5, [5, 10])],
)
def test_fractional_budget_finishes_with_full_pass_when_possible(n, epochs, expected):
    assert list(epoch_rows(n, epochs)) == expected


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


@pytest.mark.gpu
@pytest.mark.parametrize("n,k,d", [(7, 3, 1), (33, 17, 9), (129, 65, 31), (83, 11, 128)])
@pytest.mark.parametrize("method", ["gemm", "fused"])
def test_metal_assignment_matches_direct_double_oracle(metal, n, k, d, method):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(2481 + d)
    x = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    expected_labels, expected_distance = _direct_assignment(x, centers)
    labels, distance = assign(metal.array(x), metal.array(centers), method=method)
    np.testing.assert_array_equal(np.asarray(labels), expected_labels)
    np.testing.assert_allclose(np.asarray(distance), expected_distance, rtol=3e-6, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["gemm", "fused"])
def test_assignment_chunking_and_duplicate_centroid_ties(metal, method):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(747)
    x = rng.normal(size=(41, 17)).astype(np.float32)
    centers = np.tile(x[0], (67, 1))
    labels, distance = assign(metal.array(x), metal.array(centers), batch_size=7, method=method)
    np.testing.assert_array_equal(np.asarray(labels), np.zeros(len(x), dtype=np.int32))
    expected = ((x.astype(np.float64) - centers[0]) ** 2).sum(1)
    np.testing.assert_allclose(np.asarray(distance), expected, rtol=3e-6, atol=3e-6)
    assert float(np.asarray(distance)[0]) == 0


@pytest.mark.gpu
def test_spherical_assignment_is_max_inner_product(metal):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(1167)
    x = rng.normal(size=(39, 13)).astype(np.float32)
    centers = rng.normal(size=(9, 13)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels, distance = assign(metal.array(x), metal.array(centers), spherical=True)
    expected_labels = np.argmax(x.astype(np.float64) @ centers.astype(np.float64).T, axis=1)
    expected_distance = ((x.astype(np.float64) - centers[expected_labels]) ** 2).sum(1)
    np.testing.assert_array_equal(np.asarray(labels), expected_labels)
    np.testing.assert_allclose(np.asarray(distance), expected_distance, rtol=3e-6, atol=3e-6)


@pytest.mark.gpu
def test_float16_assignment_uses_represented_norms_and_float32_scores(metal):
    from mlxlearn._kernels.e_means import assign

    # Coordinates fit half, but dot products do not: this detects accidental
    # half GEMM score output, which would overflow before the argmin.
    rng = np.random.default_rng(1847)
    x = (rng.normal(size=(47, 19)) * 300).astype(np.float32)
    centers = (rng.normal(size=(71, 19)) * 300).astype(np.float32)
    represented_x = x.astype(np.float16).astype(np.float64)
    represented_centers = centers.astype(np.float16).astype(np.float64)
    expected_labels, _ = _direct_assignment(represented_x, represented_centers)
    labels, distance = assign(
        metal.array(x), metal.array(centers), method="gemm", precision="float16"
    )
    np.testing.assert_array_equal(np.asarray(labels), expected_labels)
    expected_distance = ((x.astype(np.float64) - centers[expected_labels]) ** 2).sum(1)
    np.testing.assert_allclose(np.asarray(distance), expected_distance, rtol=3e-6, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("n,k,d", [(3, 7, 5), (259, 13, 33), (513, 67, 128)])
def test_statistics_reduce_ragged_shapes_and_empty_clusters(metal, n, k, d):
    from mlxlearn._kernels.e_means import batch_stats

    rng = np.random.default_rng(441 + d)
    x = rng.normal(size=(n, d)).astype(np.float32)
    labels = rng.integers(0, max(1, k // 2), size=n, dtype=np.int32)
    expected_sums, expected_counts = _statistics(x, labels, k)
    sums, counts = batch_stats(metal.array(x), metal.array(labels), k)
    np.testing.assert_array_equal(np.asarray(counts), expected_counts)
    np.testing.assert_allclose(np.asarray(sums), expected_sums, rtol=2e-5, atol=2e-5)
    assert np.asarray(counts).sum() == n


@pytest.mark.gpu
@pytest.mark.parametrize("alpha", [0.03, 0.7, 1.0])
def test_ema_step_matches_independent_sum_and_mass_equations(metal, alpha):
    from mlxlearn._kernels.e_means import ema_update

    rng = np.random.default_rng(101)
    centers = rng.normal(size=(5, 7)).astype(np.float32)
    masses = np.array([1e-10, 0.1, 10, 17, 30], dtype=np.float32)
    x = rng.normal(size=(19, 7)).astype(np.float32)
    labels = np.array([0] * 4 + [1] * 6 + [2] * 9, dtype=np.int32)
    sums, counts = _statistics(x, labels, 5)
    updated, updated_mass = ema_update(
        metal.array(centers), metal.array(masses), metal.array(sums.astype(np.float32)),
        metal.array(counts.astype(np.float32)), alpha,
    )
    expected_mass = (1 - alpha) * masses.astype(np.float64) + alpha * counts
    expected_sum = (1 - alpha) * centers.astype(np.float64) * masses[:, None] + alpha * sums
    expected_centers = centers.astype(np.float64).copy()
    occupied = counts > 0
    expected_centers[occupied] = expected_sum[occupied] / expected_mass[occupied, None]
    np.testing.assert_allclose(np.asarray(updated)[occupied], expected_centers[occupied], rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(updated)[~occupied], centers[~occupied])
    np.testing.assert_allclose(np.asarray(updated_mass)[occupied], expected_mass[occupied], rtol=2e-6)
    assert np.isfinite(np.asarray(updated)).all()


def _numpy_train_without_revival(
    x, init, *, seed, passes, batch_size, peak, half_life,
    init_mass=1e-10, preserve_empty_at_one=False,
):
    """Independent scalar schedule and FP64 sum/count recurrence, no MLX."""
    x = x.astype(np.float64)
    centers = init.astype(np.float64).copy()
    mass = np.full(len(init), init_mass)
    slow_sum, slow_mass = centers * mass[:, None], mass.copy()
    sums = slow_sum.copy()
    slow_step = 1 - 2 ** (-1 / half_life)
    total_steps = sum(math.ceil(rows / batch_size) for rows in passes)
    step = 0
    last_sum = np.zeros_like(centers)
    last_count = np.zeros(len(centers))
    for epoch, rows in enumerate(passes):
        order = np.random.RandomState((seed * 1000003 + epoch) % 2**32).permutation(len(x))[:rows]
        for start in range(0, rows, batch_size):
            batch = x[order[start:start + batch_size]]
            labels, _ = _direct_assignment(batch, centers)
            batch_sum, count = _statistics(batch, labels, len(centers))
            if epoch == len(passes) - 1:
                last_sum += batch_sum
                last_count += count
            alpha = peak * (0.1 + 0.9 * (1 + math.cos(math.pi * step / total_steps)) / 2)
            previous_mass = mass.copy()
            sums = (1 - alpha) * sums + alpha * batch_sum
            mass = (1 - alpha) * mass + alpha * count
            if preserve_empty_at_one and alpha == 1:
                empty = count == 0
                mass[empty] = previous_mass[empty]
                sums[empty] = centers[empty] * mass[empty, None]
            occupied = mass > 0
            centers[occupied] = sums[occupied] / mass[occupied, None]
            slow_sum = (1 - slow_step) * slow_sum + slow_step * batch_sum
            slow_mass = (1 - slow_step) * slow_mass + slow_step * count
            step += 1
    closing = centers.copy()
    nonempty = last_count > 0
    closing[nonempty] = last_sum[nonempty] / last_count[nonempty, None]
    return centers, closing, slow_sum / slow_mass[:, None], total_steps


@pytest.mark.gpu
@pytest.mark.parametrize("finalization,index", [("ema", 0), ("last_epoch", 1), ("slow_ema", 2)])
def test_training_and_all_finalizers_match_independent_double_oracle(metal, finalization, index):
    from mlxlearn.cluster import EMeans

    rng = np.random.default_rng(772)
    x = rng.normal(size=(23, 7)).astype(np.float32)
    init = x[[0, 4, 9, 18]].copy()
    expected = _numpy_train_without_revival(
        x, init, seed=31, passes=[23, 12, 23], batch_size=7, peak=0.31, half_life=13,
    )
    model = EMeans(
        4, init=init, max_iter=2.5, batch_size=7, alpha0=0.31, random_state=31,
        unused_threshold=0, finalization=finalization, slow_ema_half_life=13,
    ).fit(x)
    np.testing.assert_allclose(model.ema_centers_, expected[0], rtol=8e-6, atol=3e-6)
    np.testing.assert_allclose(model.last_epoch_centers_, expected[1], rtol=8e-6, atol=3e-6)
    np.testing.assert_allclose(model.slow_ema_centers_, expected[2], rtol=8e-6, atol=3e-6)
    np.testing.assert_allclose(model.cluster_centers_, expected[index], rtol=8e-6, atol=3e-6)
    assert model.n_steps_ == expected[3]
    assert model.n_splits_ == 0
    labels, distances = _direct_assignment(x, model.cluster_centers_)
    np.testing.assert_array_equal(model.labels_, labels)
    np.testing.assert_array_equal(model.predict(x), labels)
    assert model.inertia_ == pytest.approx(distances.sum(), rel=3e-6)
    assert model.score(x) == pytest.approx(-distances.sum(), rel=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("recipe,init_mass,preserve_empty", [("paper", 1e-10, False), ("faiss", 2**-16, True)])
def test_alpha_one_empty_cluster_mass_matches_each_recipe(metal, recipe, init_mass, preserve_empty):
    from mlxlearn.cluster import EMeans

    # Seed4 first sees [-10,11], leaving the middle centroid empty. It gets
    # [0,1] next, where the retained Faiss pseudocount measurably changes its
    # update. Paper Eq2 instead erased that mass at the alpha=1 first step.
    x = np.array([[-10], [-9], [0], [1], [10], [11]], np.float32)
    init = np.array([[-10], [-1], [10]], np.float32)
    expected = _numpy_train_without_revival(
        x, init, seed=4, passes=[6], batch_size=2, peak=1, half_life=13,
        init_mass=init_mass, preserve_empty_at_one=preserve_empty,
    )[0]
    model = EMeans(
        3, max_iter=1, batch_size=2, alpha0=1, recipe=recipe, random_state=4,
        init=init, unused_threshold=0, finalization="ema", compute_labels=False,
    ).fit(x)
    np.testing.assert_allclose(model.cluster_centers_, expected, rtol=1e-7, atol=1e-7)
    if recipe == "faiss":
        assert model.cluster_centers_[1, 0] < 0.5 - 1e-6
    else:
        assert model.cluster_centers_[1, 0] == 0.5


@pytest.mark.gpu
def test_revival_splits_mass_and_first_moment_and_reuses_mapping_for_slow_ema(metal):
    from mlxlearn.cluster._e_means import revive

    centers = np.array([[2, 4, 6], [-3, 5, 7], [99, 99, 99], [-99, -99, -99]], np.float32)
    counts = np.array([16, 8, 0, 0], np.float32)
    slow_centers = centers / 3
    slow_counts = np.array([6, 4, 0, 0], np.float32)
    actual, mass, splits, companion = revive(
        metal.array(centers), metal.array(counts), metal.array([0.9, 0.2, 0.5, 0.7]),
        0.1, 2**-7, companion=(metal.array(slow_centers), metal.array(slow_counts)),
    )
    actual, mass = np.asarray(actual), np.asarray(mass)
    assert int(splits.item()) == 2
    np.testing.assert_array_equal(mass, [8, 4, 8, 4])
    np.testing.assert_allclose((actual * mass[:, None]).sum(0), (centers * counts[:, None]).sum(0))
    offset = np.array([1, -1, 1]) * 2**-7
    np.testing.assert_allclose(actual[:2], centers[:2] * (1 - offset))
    np.testing.assert_allclose(actual[2:], centers[:2] * (1 + offset))
    companion_centers, companion_counts = map(np.asarray, companion)
    np.testing.assert_array_equal(companion_counts, [3, 2, 3, 2])
    np.testing.assert_allclose(
        (companion_centers * companion_counts[:, None]).sum(0),
        (slow_centers * slow_counts[:, None]).sum(0), rtol=2e-7,
    )


@pytest.mark.gpu
def test_spherical_fit_normalizes_directions_and_preserves_zero_rows(metal):
    from mlxlearn.cluster import EMeans

    x = np.array([[0, 0, 0], [3, 0, 0], [5, 1, 0], [0, -4, 0], [0, -3, 1]], np.float32)
    model = EMeans(3, max_iter=2, batch_size=2, spherical=True, random_state=7).fit(x)
    norms = np.linalg.norm(model.cluster_centers_, axis=1)
    assert np.isfinite(model.cluster_centers_).all()
    assert np.all(np.isclose(norms, 1, atol=2e-6) | (norms == 0))
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1)
    expected_labels = np.argmax(normalized @ model.cluster_centers_.T, axis=1)
    np.testing.assert_array_equal(model.predict(x), expected_labels)


@pytest.mark.gpu
def test_seed_reproducibility_and_compute_labels_false(metal):
    from mlxlearn.cluster import EMeans

    x = np.random.default_rng(991).normal(size=(71, 5)).astype(np.float32)
    kwargs = {"n_clusters": 5, "max_iter": 2, "batch_size": 9, "random_state": 87, "compute_labels": False}
    first, second = EMeans(**kwargs).fit(x), EMeans(**kwargs).fit(x)
    assert first.labels_ is None and first.inertia_ is None
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.execution_backend_ == "metal"


@pytest.mark.gpu
def test_centered_assignment_and_transform_handle_common_coordinate_offset(metal):
    from mlxlearn.cluster import EMeans

    rng = np.random.default_rng(430)
    x = (10_000 + rng.normal(size=(64, 5)) * 3).astype(np.float32)
    model = EMeans(4, max_iter=1, batch_size=64, unused_threshold=0, random_state=13).fit(x)
    labels, distances = _direct_assignment(x, model.cluster_centers_)
    np.testing.assert_array_equal(model.labels_, labels)
    np.testing.assert_array_equal(model.predict(x), labels)
    expected_transform = np.sqrt(((x.astype(np.float64)[:, None] - model.cluster_centers_) ** 2).sum(2))
    np.testing.assert_allclose(model.transform(x), expected_transform, rtol=3e-6, atol=3e-6)
    assert model.inertia_ == pytest.approx(distances.sum(), rel=3e-6)


@pytest.mark.gpu
def test_half_precision_rejects_out_of_range_operands_before_kernel_dispatch(metal):
    from mlxlearn.cluster import EMeans

    x = np.array([[-100_000., 0], [100_000., 0], [0, 0]], np.float32)
    with pytest.raises(ValueError, match="float16 range"):
        EMeans(2, precision="float16", max_iter=1).fit(x)
    model = EMeans(2, precision="float16", max_iter=1).fit(x / 100_000)
    with pytest.raises(ValueError, match="float16 range"):
        model.predict(x)


@pytest.mark.gpu
def test_finite_coordinates_that_overflow_float32_arithmetic_are_rejected(metal):
    from mlxlearn.cluster import EMeans

    huge = np.array([[-1e20, 0], [1e20, 0], [0, 0]], np.float32)
    with pytest.raises(ValueError, match="float32"):
        EMeans(2, max_iter=1).fit(huge)
    small = huge / 1e20
    with pytest.raises(ValueError, match="float32"):
        EMeans(2, max_iter=1, init=huge[:2]).fit(small)
    model = EMeans(2, max_iter=1).fit(small)
    for method in (model.predict, model.score, model.transform):
        with pytest.raises(ValueError, match="float32"):
            method(huge)


def test_sklearn_clone_and_unfitted_predict_contract():
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError

    from mlxlearn.cluster import EMeans

    model = EMeans(3, max_iter=2.5, finalization="slow_ema", slow_ema_half_life=11)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        copied.predict(np.ones((4, 3), np.float32))


@pytest.mark.parametrize("parameters", [
    {"n_clusters": 0}, {"n_clusters": 9}, {"max_iter": 0}, {"max_iter": float("nan")},
    {"batch_size": 0}, {"batch_size": 1.5}, {"alpha0": 1.1}, {"alpha0": 0},
    {"recipe": "unknown"}, {"precision": "float64"}, {"assignment_batch_size": -1},
])
def test_invalid_parameters_fail_explicitly(parameters):
    from mlxlearn.cluster import EMeans

    kwargs = {"n_clusters": 2, **parameters}
    with pytest.raises(ValueError):
        EMeans(**kwargs).fit(np.ones((8, 3), np.float32))


@pytest.mark.parametrize("x", [
    np.ones(8), np.empty((0, 3)), np.array([[1., np.nan], [0., 1.]]),
    np.array([[1., np.inf], [0., 1.]]), np.ones((8, 3), np.complex64),
])
def test_invalid_inputs_fail_explicitly(x):
    from mlxlearn.cluster import EMeans

    with pytest.raises(ValueError):
        EMeans(2).fit(x)
