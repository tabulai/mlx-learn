"""Independent represented-BF16 oracles and optimized e-means contracts.

The host oracle rounds IEEE float32 bits to BF16 using round-to-nearest-even;
it does not use MLX casts, Metal kernels, or the implementation's schedules.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _bf16(x):
    """Finite FP32 -> represented BF16 -> FP32, independently of MLX."""
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _represented_assignment(x, centers, *, spherical=False):
    rows = _bf16(x).astype(np.float64)
    centroids = _bf16(centers).astype(np.float64)
    if spherical:
        distances = 2 - 2 * (rows @ centroids.T)
    else:
        distances = ((rows[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(rows)), labels]


def _train_bf16_without_revival(x, init, *, seed, passes, batch, peak, init_mass):
    """FP32 state recurrence with BF16 assignment and represented-row sums."""
    centers = np.asarray(init, np.float32).copy()
    mass = np.full(len(centers), init_mass, np.float32)
    last_sums = np.zeros_like(centers)
    last_counts = np.zeros_like(mass)
    captured = np.full(len(x), -1, np.int64)
    nsteps = sum(math.ceil(rows / batch) for rows in passes)
    step = 0
    for epoch, rows in enumerate(passes):
        order = np.random.RandomState((seed * 1000003 + epoch) % 2**32).permutation(len(x))
        for begin in range(0, rows, batch):
            values = _bf16(x[order[begin:min(begin + batch, rows)]])
            labels, _ = _represented_assignment(values, centers)
            sums = np.zeros_like(centers)
            np.add.at(sums, labels, values)
            counts = np.bincount(labels, minlength=len(centers)).astype(np.float32)
            if epoch == len(passes) - 1:
                last_sums += sums
                last_counts += counts
                captured[order[begin:min(begin + batch, rows)]] = labels
            alpha = peak * (0.1 + 0.9 * (1 + math.cos(math.pi * step / nsteps)) / 2)
            rate, decay = np.float32(alpha), np.float32(1 - alpha)
            old_weight = decay * mass
            updated_mass = old_weight + rate * counts
            occupied = (counts > 0) & (updated_mass > 0)
            centers[occupied] = (
                old_weight[occupied, None] * centers[occupied] + rate * sums[occupied]
            ) / updated_mass[occupied, None]
            mass = updated_mass
            step += 1
    closing = centers.copy()
    occupied = last_counts > 0
    closing[occupied] = last_sums[occupied] / last_counts[occupied, None]
    return centers, closing, captured


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


def test_bf16_oracle_rounds_halfway_values_to_even_and_keeps_exponent_range():
    values = np.array([0., -0., 1., 1 + 1 / 256, 1 + 3 / 256,
                       -1 - 1 / 256, -1 - 3 / 256, 65536., 2**80], np.float32)
    expected = np.array([0., -0., 1., 1., 1 + 1 / 64,
                         -1., -1 - 1 / 64, 65536., 2**80], np.float32)
    np.testing.assert_array_equal(_bf16(values).view(np.uint32), expected.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("n,k,d", [(1, 1, 1), (7, 3, 9), (33, 65, 31), (67, 129, 96)])
@pytest.mark.parametrize("method", ["auto", "gemm", "fused"])
def test_bf16_assignment_masks_and_represented_residuals_match_oracle(metal, n, k, d, method):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(n + k + d)
    x = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    expected_labels, expected_distances = _represented_assignment(x, centers)
    labels, distances = assign(metal.array(x), metal.array(centers), precision="bfloat16", method=method, batch_size=11)
    np.testing.assert_array_equal(np.asarray(labels), expected_labels)
    np.testing.assert_allclose(np.asarray(distances), expected_distances, rtol=3e-6, atol=2e-6)


@pytest.mark.gpu
def test_bf16_uses_rounded_centroid_norms_and_smallest_index_for_cast_ties(metal):
    from mlxlearn._kernels.e_means import assign

    # Both centroids cast exactly to 1. A norm computed before casting would
    # incorrectly prefer index1, while represented distances tie at zero.
    x = np.array([[1.0], [1 + 1 / 256]], np.float32)
    centers = np.array([[1.003], [1.001]], np.float32)
    labels, distances = assign(metal.array(x), metal.array(centers), precision="bfloat16")
    np.testing.assert_array_equal(np.asarray(labels), [0, 0])
    np.testing.assert_array_equal(np.asarray(distances), [0, 0])


@pytest.mark.gpu
def test_bf16_does_not_accidentally_use_fp16_operands_or_score_outputs(metal):
    from mlxlearn._kernels.e_means import assign

    x = np.array([[65536., 131072., -65536.], [300., 310., -290.]], np.float32)
    centers = np.array([[66048., 130048., -65536.], [301., 311., -289.]], np.float32)
    expected_labels, expected_distances = _represented_assignment(x, centers)
    labels, distances = assign(metal.array(x), metal.array(centers), precision="bfloat16")
    np.testing.assert_array_equal(np.asarray(labels), expected_labels)
    np.testing.assert_allclose(np.asarray(distances), expected_distances, rtol=3e-6)
    assert np.isfinite(np.asarray(distances)).all()


@pytest.mark.gpu
def test_bf16_spherical_assignment_uses_rounded_directions_without_renormalizing(metal):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(421)
    x, centers = [rng.normal(size=(size, 9)).astype(np.float32) for size in (13, 7)]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    expected_labels, expected_distances = _represented_assignment(x, centers, spherical=True)
    labels, distances = assign(metal.array(x), metal.array(centers), precision="bfloat16", spherical=True)
    np.testing.assert_array_equal(np.asarray(labels), expected_labels)
    np.testing.assert_allclose(np.asarray(distances), expected_distances, rtol=3e-6, atol=2e-7)


@pytest.mark.gpu
@pytest.mark.parametrize("recipe,init_mass", [("paper", 1e-10), ("faiss", 2**-16)])
@pytest.mark.parametrize("finalization,index", [("ema", 0), ("last_epoch", 1)])
def test_bf16_training_rounds_rows_for_statistics_and_matches_independent_recurrence(metal, recipe, init_mass, finalization, index):
    from mlxlearn.cluster import EMeans

    x = np.random.default_rng(772).normal(size=(53, 9)).astype(np.float32)
    init = x[[0, 4, 19, 28, 48]].copy()
    expected = _train_bf16_without_revival(
        x, init, seed=31, passes=[53, 26, 53], batch=11, peak=.37, init_mass=init_mass,
    )
    model = EMeans(
        5, init=init, max_iter=2.5, batch_size=11, alpha0=.37, random_state=31,
        recipe=recipe, unused_threshold=0, finalization=finalization,
        precision="bfloat16", compute_labels=False,
    ).fit(x)
    np.testing.assert_allclose(model.ema_centers_, expected[0], rtol=1e-5, atol=3e-6)
    np.testing.assert_allclose(model.last_epoch_centers_, expected[1], rtol=1e-5, atol=3e-6)
    np.testing.assert_allclose(model.cluster_centers_, expected[index], rtol=1e-5, atol=3e-6)
    assert model.cluster_centers_.dtype == np.float32
    assert model.recipe_["precision"] == "bfloat16"
    labels, distances = _represented_assignment(x, model.cluster_centers_)
    np.testing.assert_array_equal(model.predict(x), labels)
    assert model.score(x) == pytest.approx(-distances.sum(), rel=3e-6)


@pytest.mark.gpu
def test_bf16_casts_raw_coordinates_without_global_centering_and_keeps_empty_init(metal):
    from mlxlearn.cluster import EMeans

    # Every point and both initial centers round to 9984, so the assignment
    # ties choose index0. Centering before the BF16 cast changes this result.
    x = np.array([[10001.], [10002.], [10007.], [10012.]], np.float32)
    init = np.array([[10000.], [10010.]], np.float32)
    model = EMeans(
        2, init=init, max_iter=1, batch_size=4, alpha0=.4, random_state=19,
        unused_threshold=0, finalization="last_epoch", precision="bfloat16",
    ).fit(x)
    np.testing.assert_array_equal(model.cluster_centers_, [[9984.], [10010.]])
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 0])
    assert model.inertia_ == 0


@pytest.mark.gpu
@pytest.mark.parametrize("value,epsilon,expected_bits", [
    (1., 1e-4, 0x3F810000), (-1., 1e-4, 0xBF7F0000),
    (0., 1e-4, 0x00010000), (-0., 1e-4, 0x00010000),
    (np.array([0x7F7F0000], np.uint32).view(np.float32)[0], 0., 0x7F7E0000),
])
def test_bf16_revival_repairs_collapsed_casts_with_finite_next_representable_value(metal, value, epsilon, expected_bits):
    from mlxlearn.cluster._e_means import revive

    centers = np.array([[value], [17.]], np.float32)
    result, mass, splits, _ = revive(
        metal.array(centers), metal.array([8., 0.]), metal.array([.9, .2]),
        .1, epsilon, precision="bfloat16",
    )
    result = np.asarray(result)
    np.testing.assert_array_equal(np.asarray(mass), [4., 4.])
    assert int(splits.item()) == 1
    assert result[1, 0].view(np.uint32) == expected_bits
    assert result[0, 0] == np.float32(value) * (np.float32(1) - np.float32(epsilon))


@pytest.mark.gpu
def test_bf16_revival_reuses_donor_map_but_repairs_companion_own_positions(metal):
    from mlxlearn.cluster._e_means import revive

    centers = metal.array([[1., 2.], [17., 19.]])
    counts = metal.array([8., 0.])
    companion = (metal.array([[-1., -2.], [23., 29.]]), metal.array([6., 0.]))
    result, mass, _, slow = revive(
        centers, counts, metal.array([.9, .2]), .1, 1e-4,
        companion=companion, precision="bfloat16",
    )
    result, slow_centers = np.asarray(result), np.asarray(slow[0])
    assert result[1, 0] == np.float32(1 + 1 / 128)
    assert slow_centers[1, 0] == np.float32(-1 + 1 / 256)
    np.testing.assert_allclose(result[:, 1], [2 * 1.0001, 2 * .9999], rtol=1e-7)
    np.testing.assert_allclose(slow_centers[:, 1], [-2 * 1.0001, -2 * .9999], rtol=1e-7)
    np.testing.assert_array_equal(np.asarray(mass), [4., 4.])
    np.testing.assert_array_equal(np.asarray(slow[1]), [3., 3.])


@pytest.mark.gpu
def test_bf16_statistics_keep_float32_sums_and_exact_cluster_counts(metal):
    from mlxlearn._kernels.e_means import batch_stats

    x = np.random.default_rng(94).normal(size=(29, 9)).astype(np.float32)
    labels = (np.arange(len(x)) % 3).astype(np.uint32)
    sums, counts = batch_stats(metal.array(x).astype(metal.bfloat16), metal.array(labels), 5)
    expected = np.zeros((5, 9), np.float64)
    np.add.at(expected, labels, _bf16(x))
    assert sums.dtype == counts.dtype == metal.float32
    np.testing.assert_allclose(np.asarray(sums), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(counts), np.bincount(labels, minlength=5))


@pytest.mark.gpu
@pytest.mark.parametrize("n,d", [(1, 1), (1, 4), (17, 3), (17, 7), (17, 96), (17, 128), (17, 200)])
def test_fused_bf16_gather_matches_cast_bits_with_ragged_and_repeated_indices(metal, n, d):
    from mlxlearn._kernels.e_means import gather_bf16

    values = np.random.default_rng(d).normal(size=(n, d)).astype(np.float32)
    edge = np.array([0., -0., 1 + 1 / 256, 1 + 3 / 256, -1 - 3 / 256,
                     65536., 2**80, 2**-130, -2**-130], np.float32)
    values.flat[:min(values.size, edge.size)] = edge[:min(values.size, edge.size)]
    data = metal.array(values)
    order = metal.array(np.arange(23) % n, dtype=metal.int32)
    actual = gather_bf16(data, order)
    expected = data[order].astype(metal.bfloat16)
    np.testing.assert_array_equal(np.array(actual.view(metal.uint16)), np.array(expected.view(metal.uint16)))


@pytest.mark.gpu
def test_fused_bf16_gather_handles_strided_input_and_empty_order(metal):
    from mlxlearn._kernels.e_means import gather_bf16

    data = metal.arange(140, dtype=metal.float32).reshape(4, 35).T
    order = metal.array([34, 2, 8, 2], dtype=metal.uint32)
    np.testing.assert_array_equal(np.array(gather_bf16(data, order).astype(metal.float32)),
                                  np.array(data[order]))
    empty = gather_bf16(data, metal.array([], dtype=metal.int32))
    assert empty.shape == (0, 4) and empty.dtype == metal.bfloat16


@pytest.mark.gpu
@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("spherical", [False, True])
def test_coalesced_bf16_merge_masks_large_ragged_parts_and_preserves_ties(metal, d, spherical):
    from mlxlearn._kernels.e_means_bf16 import assign_bf16

    rng = np.random.default_rng(567 + d)
    x = rng.normal(size=(67, d)).astype(np.float32)
    centers = rng.normal(size=(1025, d)).astype(np.float32)
    centers[1] = centers[0]
    centers[-1] = centers[0]  # Equal winner in another partial tile/merge warp.
    x[0] = centers[0]
    xr, cr = _bf16(x).astype(np.float64), _bf16(centers).astype(np.float64)
    scores = -xr @ cr.T if spherical else np.sum(cr * cr, axis=1) - 2 * xr @ cr.T
    labels, actual_scores = assign_bf16(metal.array(x), metal.array(centers), spherical=spherical, batch_size=33)
    np.testing.assert_array_equal(np.array(labels), np.argmin(scores, axis=1))
    np.testing.assert_allclose(np.array(actual_scores), np.min(scores, axis=1), rtol=3e-6, atol=3e-5)


@pytest.mark.gpu
def test_bf16_update_uses_independent_decay_and_exact_mean_when_weight_is_one(metal):
    from mlxlearn._kernels.e_means import ema_update

    centers = np.array([[1e20, -1e20], [1e6, 1e6], [7., -8.]], np.float32)
    mass = np.array([0., 2**24, 3.], np.float32)
    sums = np.array([[1.25, -.75], [-1e6, 2e6], [0., 0.]], np.float32)
    hits = np.array([1., 2., 0.], np.float32)
    alpha, decay = .9, .1
    actual, actual_mass = ema_update(
        metal.array(centers), metal.array(mass), metal.array(sums), metal.array(hits), alpha, decay=decay,
    )
    actual, actual_mass = np.asarray(actual), np.asarray(actual_mass)
    expected_mass = np.float32(decay) * mass + np.float32(alpha) * hits
    # With zero prior mass, computing c + 1*(mean-c) loses the small mean.
    # The reference's explicit weight==1 selection must preserve it exactly.
    np.testing.assert_array_equal(actual[0], sums[0])
    np.testing.assert_array_equal(actual[2], centers[2])
    np.testing.assert_array_equal(actual_mass, expected_mass)
    assert actual.dtype == actual_mass.dtype == np.float32
    rounded_subtraction_mass = (np.float32(1) - np.float32(alpha)) * mass[1] + np.float32(alpha) * hits[1]
    assert actual_mass[1] != rounded_subtraction_mass


@pytest.mark.gpu
@pytest.mark.parametrize("epochs,passes", [(2.5, [53, 26, 53]), (.5, [26])])
def test_resident_fit_preserves_moving_label_order_and_partial_pass_sentinel(metal, epochs, passes):
    from mlxlearn.cluster import EMeans

    x = np.random.default_rng(772).normal(size=(53, 9)).astype(np.float32)
    init = x[[0, 4, 19, 28, 48]].copy()
    model = EMeans(
        5, init=init, max_iter=epochs, batch_size=11, alpha0=.37, random_state=31,
        recipe="faiss", unused_threshold=0, precision="bfloat16", compute_labels=False,
    )
    prepared = model.prepare_device(x)
    result = model.train_device(prepared, capture_assignments=True)
    expected = _train_bf16_without_revival(
        x, init, seed=31, passes=passes, batch=11, peak=.37, init_mass=2**-16,
    )
    assert isinstance(result.centers, metal.array)
    assert isinstance(result.assignments, metal.array)
    assert result.assignments.dtype == metal.int64
    np.testing.assert_array_equal(np.asarray(result.assignments), expected[2])
    np.testing.assert_allclose(np.asarray(result.centers), expected[1], rtol=1e-5, atol=3e-6)
    assert result.steps == sum(math.ceil(n / 11) for n in passes)
    assert result.epochs == len(passes) == len(result.history)
    assert [record["rows"] for record in result.history] == passes
    assert result.seconds >= 0
    assert not hasattr(model, "cluster_centers_")
    if epochs >= 1:
        # Captured labels describe the moving final pass, not another Lloyd
        # assignment against the closing means after optimization.
        nearest, _ = _represented_assignment(x, expected[1])
        assert np.any(expected[2] != nearest)
    else:
        assert np.count_nonzero(np.asarray(result.assignments) == -1) == 27


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_resident_preparation_is_reusable_and_public_fit_uses_same_training_path(metal, precision):
    from mlxlearn.cluster import EMeans

    x = np.random.default_rng(914).normal(size=(37, 5)).astype(np.float32)
    initial = x[[2, 8, 14, 26]].copy()
    model = EMeans(
        4, init=initial, max_iter=2, batch_size=9, alpha0=.3, random_state=12,
        precision=precision, unused_threshold=0, compute_labels=False,
    )
    prepared = model.prepare_device(metal.array(x))
    first, second = model.train_device(prepared), model.train_device(prepared)
    assert first.assignments is None and second.assignments is None
    np.testing.assert_array_equal(np.asarray(prepared.initial), initial)
    np.testing.assert_allclose(np.asarray(first.centers), np.asarray(second.centers), rtol=3e-6, atol=3e-6)
    model.fit(x)
    np.testing.assert_allclose(model.cluster_centers_, np.asarray(first.centers), rtol=3e-6, atol=3e-6)
    model.set_params(alpha0=.4)
    with pytest.raises(ValueError, match="configuration"):
        model.train_device(prepared)
