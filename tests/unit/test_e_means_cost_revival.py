"""Cost-based revival of e-means: kernels, plan, split, closing and training.

Every oracle below is NumPy float64 with direct squared differences, stable
sorts and indexed accumulation, independent of the Metal kernels. The
selection oracle is a port of the reference ``_choose_removal_splits``
(facebookresearch/faiss, contrib/e_means/base.py, accessed 2026-11-24).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlxlearn.cluster._e_means import REMOVAL_THRESHOLD, epoch_seed


def _bf16(x):
    """Finite FP32 -> represented BF16 -> FP32, independently of MLX."""
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _two_nearest(x, centers, *, spherical=False):
    x, centers = x.astype(np.float64), centers.astype(np.float64)
    if spherical:
        distance = 2 - 2 * (x @ centers.T)
    else:
        distance = ((x[:, None] - centers[None]) ** 2).sum(2)
    order = np.argsort(distance, axis=1, kind="stable")
    rows = np.arange(len(x))
    best = distance[rows, order[:, 0]]
    if centers.shape[0] == 1:
        return order[:, 0], best, np.full(len(x), np.inf)
    return order[:, 0], best, distance[rows, order[:, 1]] - best


def _reference_plan(removal, usage, lengths):
    """Donors and the centroids that move onto them, as the reference pairs them."""
    removal, usage = removal.astype(np.float64), usage.astype(np.float64)
    candidates = np.flatnonzero((removal <= 0) | (removal < REMOVAL_THRESHOLD * np.median(removal)))
    if len(candidates) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    kept = candidates[np.argsort(removal[candidates], kind="stable")]
    donors = usage > 0
    donors[kept] = False
    donors = np.flatnonzero(donors)
    donors = donors[np.argsort(-lengths[donors].astype(np.float32), kind="stable")]
    number = min(len(kept), len(donors))
    return donors[:number], kept[:number]


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


def test_pass_seed_avoids_the_initialization_draw_only_where_asked():
    for seed in (1, 7, 1234, 2**31 - 1, 2**32 - 1):
        for epoch in range(3):
            plain = (seed * 1000003 + epoch) % 2**32
            assert epoch_seed(seed, epoch) == epoch_seed(seed, epoch, distinct=True) == plain
    for seed in (0, 2**31):
        assert epoch_seed(seed, 0) == seed
        assert epoch_seed(seed, 0, distinct=True) == (seed + 0x9E3779B9) % 2**32
        assert epoch_seed(seed, 1, distinct=True) == (seed * 1000003 + 1) % 2**32


@pytest.mark.gpu
# The last shape reaches the coalesced BF16 tiles and, on an M4 Max, the
# streaming float32 tiles, with ragged rows, centroid tiles and merge parts.
@pytest.mark.parametrize("n,k,d", [(7, 3, 1), (33, 17, 9), (129, 65, 31), (83, 11, 128), (67, 200, 96),
                                   (300, 4500, 96)])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("precision,method", [
    ("float32", "gemm"), ("float32", "fused"), ("float32", "auto"), ("float16", "fused"),
    ("bfloat16", "auto"), ("bfloat16", "gemm"), ("bfloat16", "fused"),
])
def test_margin_matches_double_oracle_and_leaves_the_assignment_unchanged(
        metal, n, k, d, spherical, precision, method):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(1000 * d + k)
    x = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    if spherical:
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    options = {"spherical": spherical, "method": method, "precision": precision, "batch_size": 37}
    plain = assign(metal.array(x), metal.array(centers), **options)
    labels, distances, margins = assign(metal.array(x), metal.array(centers), second=True, **options)
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(plain[0]))
    np.testing.assert_array_equal(np.asarray(distances).view(np.uint32),
                                  np.asarray(plain[1]).view(np.uint32))
    if precision == "bfloat16":
        x, centers = _bf16(x), _bf16(centers)
    elif precision == "float16":
        x, centers = x.astype(np.float16).astype(np.float32), centers.astype(np.float16).astype(np.float32)
    _, _, expected = _two_nearest(x, centers, spherical=spherical)
    # Scores are float32 sums of d products plus a squared norm.
    scale = float((np.abs(x.astype(np.float64)) @ np.abs(centers.astype(np.float64)).T).max()
                  + (centers.astype(np.float64) ** 2).sum(1).max())
    margins = np.asarray(margins)
    assert margins.dtype == np.float32 and (margins >= 0).all()
    np.testing.assert_allclose(margins, expected, rtol=0, atol=3e-6 * scale)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("method", ["gemm", "fused", "auto"])
def test_margin_is_zero_between_duplicates_and_infinite_for_one_centroid(metal, precision, method):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(55)
    x = rng.normal(size=(70, 24)).astype(np.float32)
    centers = rng.normal(size=(140, 24)).astype(np.float32)
    centers[70:] = centers[:70]  # each copy lands in another 64-centroid tile
    labels, _, margins = assign(metal.array(x), metal.array(centers), method=method,
                                precision=precision, second=True)
    assert (np.asarray(labels) < 70).all()  # the lower index wins a tie
    np.testing.assert_array_equal(np.asarray(margins), np.zeros(len(x), np.float32))
    _, _, alone = assign(metal.array(x), metal.array(centers[:1]), method=method,
                         precision=precision, second=True)
    assert np.isposinf(np.asarray(alone)).all()
    empty = assign(metal.array(x[:0]), metal.array(centers), method=method, precision=precision,
                   second=True)
    assert [part.shape for part in empty] == [(0,), (0,), (0,)]


@pytest.mark.gpu
@pytest.mark.parametrize("spherical", [False, True])
def test_streaming_margin_matches_gemm_on_ragged_and_nearly_tied_tiles(metal, spherical):
    from mlxlearn._kernels import e_means as kernels

    if not kernels._streaming_fp32_device():
        pytest.skip("requires the calibrated M4 Max FP32 GEMM")
    rng = np.random.default_rng(812)
    x = (8192 + rng.standard_normal((259, 100))).astype(np.float32)
    centers = (8192 + rng.standard_normal((2049, 100))).astype(np.float32)
    centers = np.concatenate([centers, centers[:-1]])
    centers[2049::2, 1] = np.nextafter(centers[2049::2, 1], np.float32(np.inf))
    x, centers = metal.array(x), metal.array(centers)
    options = {"batch_size": 129, "spherical": spherical, "second": True}
    expected = kernels.assign(x, centers, method="gemm", **options)
    actual = kernels.assign(x, centers, **options)
    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    # Both routes accumulate the scores in the same order.
    np.testing.assert_array_equal(np.asarray(actual[2]).view(np.uint32),
                                  np.asarray(expected[2]).view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("n,k,d,half", [(3, 7, 5, False), (259, 13, 33, False), (513, 67, 128, True),
                                       (1000, 5, 1, False), (64, 9, 300, False)])
def test_removal_statistics_match_indexed_double_sums(metal, n, k, d, half):
    from mlxlearn._kernels.e_means import removal_stats

    rng = np.random.default_rng(900 + d)
    x = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    axes = rng.normal(size=(k, d)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    labels = rng.integers(0, max(1, k // 2), size=n).astype(np.uint32)
    margins = rng.random(n).astype(np.float32)
    margins[::7], margins[3::11] = np.inf, np.nan  # neither counts
    rows = metal.array(x).astype(metal.bfloat16) if half else metal.array(x)
    axis_sums, costs = removal_stats(rows, metal.array(labels), metal.array(margins),
                                     metal.array(centers), metal.array(axes))
    represented = _bf16(x) if half else x
    y = represented.astype(np.float64) - centers[labels]
    projection = (y * axes[labels]).sum(1, keepdims=True)
    expected = np.zeros((k, d))
    np.add.at(expected, labels, projection * y)
    expected_costs = np.zeros(k)
    np.add.at(expected_costs, labels, np.where(np.isfinite(margins), margins, 0).astype(np.float64))
    bound = np.zeros((k, d))
    np.add.at(bound, labels, np.abs(projection * y) + (np.abs(y) * np.abs(axes[labels])).sum(1, keepdims=True) * np.abs(y))
    np.testing.assert_array_less(np.abs(np.asarray(axis_sums) - expected), 4e-6 * (d + 8) * bound + 1e-30)
    np.testing.assert_allclose(np.asarray(costs), expected_costs, rtol=2e-5, atol=1e-6)
    assert not np.asarray(axis_sums)[max(1, k // 2):].any()


@pytest.mark.gpu
@pytest.mark.parametrize("alpha", [0.03, 0.7, 1.0])
def test_removal_update_decays_sums_and_keeps_the_axis_of_an_empty_sum(metal, alpha):
    from mlxlearn._kernels.e_means import removal_update

    rng = np.random.default_rng(31)
    k, d = 9, 37
    long_axis = rng.normal(size=(k, d)).astype(np.float32)
    batch = rng.normal(size=(k, d)).astype(np.float32)
    long_axis[[0, 2]] = 0
    batch[[0, 1]] = 0
    removal, costs = rng.random(k).astype(np.float32), rng.random(k).astype(np.float32)
    axes = rng.normal(size=(k, d)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    new_long, new_removal, new_axes, lengths = map(np.asarray, removal_update(
        metal.array(long_axis), metal.array(removal), metal.array(batch), metal.array(costs),
        metal.array(axes), alpha))
    expected = (1 - alpha) * long_axis.astype(np.float64) + alpha * batch
    norm = np.linalg.norm(expected, axis=1)
    np.testing.assert_allclose(new_long, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(lengths, norm, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(new_removal, (1 - alpha) * removal + alpha * costs.astype(np.float64), rtol=2e-6)
    moving = norm > 0
    assert not moving[0] and (alpha < 1 or not moving[1])
    np.testing.assert_allclose(new_axes[moving], expected[moving] / norm[moving, None], rtol=1e-5, atol=2e-6)
    np.testing.assert_array_equal(new_axes[~moving], axes[~moving])


@pytest.mark.gpu
def test_removal_update_length_does_not_overflow_where_it_is_representable(metal):
    from mlxlearn._kernels.e_means import removal_update

    # 64 components of 5e36: their squares overflow float32, the length 4e37 does not.
    long_axis = metal.array(np.full((2, 64), 1e37, np.float32))
    zeros = metal.zeros((2, 64), metal.float32)
    _, _, axes, lengths = removal_update(long_axis, metal.zeros((2,)), zeros, metal.zeros((2,)), zeros, 0.5)
    np.testing.assert_allclose(np.asarray(lengths), 4e37, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(axes), 0.125, rtol=1e-6)
    # A sum below the normal float32 range counts as zero: the axis is kept.
    previous = metal.array(np.eye(2, 64, dtype=np.float32))
    faint = metal.array(np.full((2, 64), 1e-41, np.float32))
    _, _, kept, lengths = removal_update(faint, metal.zeros((2,)), zeros, metal.zeros((2,)), previous, 0.5)
    np.testing.assert_array_equal(np.asarray(lengths), [0, 0])
    np.testing.assert_array_equal(np.asarray(kept), np.eye(2, 64, dtype=np.float32))


@pytest.mark.gpu
def test_plan_pairs_cheapest_centroids_with_longest_donors_as_the_reference(metal):
    from mlxlearn.cluster._e_means import removal_plan

    rng = np.random.default_rng(11)
    moved = 0
    for _ in range(120):
        k = int(rng.integers(1, 200))
        removal = rng.gamma(1.0, 1.0, k).astype(np.float32)
        removal[rng.random(k) < rng.random() * 0.6] = 0  # centroids without cost, in index order
        mass = rng.random(k).astype(np.float32)
        mass[rng.random(k) < 0.1] = 0
        lengths = rng.random(k).astype(np.float32)
        source, target = _reference_plan(removal, mass, lengths)
        is_child, is_donor, donor_of, child_of, number = map(np.asarray, removal_plan(
            metal.array(removal), metal.array(mass), metal.array(lengths)))
        assert int(number) == len(source)
        np.testing.assert_array_equal(np.flatnonzero(is_child), np.sort(target))
        np.testing.assert_array_equal(np.flatnonzero(is_donor), np.sort(source))
        np.testing.assert_array_equal(donor_of[target], source)
        np.testing.assert_array_equal(child_of[source], target)
        moved += len(source)
    assert moved > 1000


@pytest.mark.gpu
def test_plan_never_splits_a_centroid_whose_axis_length_is_not_a_number(metal):
    from mlxlearn.cluster._e_means import removal_plan

    removal = metal.array([0, 0, 1, 2, 3], dtype=metal.float32)
    lengths = metal.array([1, 2, np.nan, np.nan, 5], dtype=metal.float32)
    is_child, is_donor, donor_of, child_of, number = map(
        np.asarray, removal_plan(removal, metal.ones((5,)), lengths))
    assert int(number) == 1  # two free centroids, one usable donor
    np.testing.assert_array_equal(is_child, [True, False, False, False, False])
    np.testing.assert_array_equal(is_donor, [False, False, False, False, True])
    assert donor_of[0] == 4 and child_of[4] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("removal,mass", [
    ([0, 0, 0], [1, 1, 1]),      # every centroid is free: none is left to split
    ([1], [1]), ([0], [1]),      # a single centroid
    ([1, 2, 3], [0, 0, 0]),      # no centroid with mass
    ([4, 5, 6, 7], [1, 1, 1, 1]),  # none below 0.3 of the median
])
def test_plan_moves_nothing_without_a_candidate_and_a_donor(metal, removal, mass):
    from mlxlearn.cluster._e_means import removal_plan

    plan = removal_plan(metal.array(removal, dtype=metal.float32), metal.array(mass, dtype=metal.float32),
                        metal.ones((len(mass),)))
    assert int(plan[4].item()) == 0
    assert not np.asarray(plan[0]).any() and not np.asarray(plan[1]).any()


@pytest.mark.gpu
def test_split_places_gaussian_half_means_and_halves_every_statistic(metal):
    from mlxlearn.cluster._e_means import removal_plan, revive_by_cost

    rng = np.random.default_rng(4)
    k, d = 6, 4
    centers = rng.normal(size=(k, d)).astype(np.float32)
    mass = np.array([8, 4, 2, 6, 3, 5], np.float32)
    long_axis = rng.normal(size=(k, d)).astype(np.float32)
    long_axis[3] = 0  # a donor without a long axis yet
    lengths = np.linalg.norm(long_axis, axis=1).astype(np.float32)
    axes = np.where(lengths[:, None] > 0, long_axis / np.maximum(lengths[:, None], 1e-30),
                    np.eye(k, d, dtype=np.float32)).astype(np.float32)
    removal = np.array([5, 0, 0, 7, 0.1, 9], np.float32)
    source, target = _reference_plan(removal, mass, lengths)
    # Centroids 1 and 2 are free and 4 is cheap; the axis-less donor comes last.
    assert sorted(source[:2]) == [0, 5] and source[2] == 3
    np.testing.assert_array_equal(target, [1, 2, 4])
    plan = removal_plan(metal.array(removal), metal.array(mass), metal.array(lengths))
    bounds = (metal.array(np.full(d, -100, np.float32)), metal.array(np.full(d, 100, np.float32)))
    slow_centers, slow_mass = centers * 0.5, mass * 0.25
    new_centers, new_mass, new_long, new_removal, new_axes, number, companion = revive_by_cost(
        metal.array(centers), metal.array(mass), metal.array(long_axis), metal.array(removal),
        metal.array(axes), metal.array(lengths), plan, bounds,
        companion=(metal.array(slow_centers), metal.array(slow_mass)))
    new_centers, new_mass, new_long, new_removal, new_axes = map(
        np.asarray, (new_centers, new_mass, new_long, new_removal, new_axes))
    assert int(number.item()) == 3
    spread = np.sqrt(2 * lengths[source].astype(np.float64) / (math.pi * mass[source]))
    offset = spread[:, None] * axes[source]
    np.testing.assert_allclose(new_centers[source], centers[source] - offset, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(new_centers[target[:2]], centers[source[:2]] + offset[:2], rtol=2e-6, atol=2e-6)
    # The axis-less donor keeps its place; its copy differs by one float32 in coordinate zero.
    np.testing.assert_array_equal(new_centers[3], centers[3])
    assert new_centers[4, 0] == np.nextafter(centers[3, 0], np.float32(np.inf))
    np.testing.assert_array_equal(new_centers[4, 1:], centers[3, 1:])
    for values, before in ((new_mass, mass), (new_long, long_axis), (new_removal, removal)):
        np.testing.assert_allclose(values[source], before[source] / 2)
        np.testing.assert_allclose(values[target], before[source] / 2)
    np.testing.assert_array_equal(new_axes[target], axes[source])
    np.testing.assert_array_equal(new_axes[source], axes[source])
    slow_new, slow_new_mass = map(np.asarray, companion)
    np.testing.assert_allclose(slow_new[source], slow_centers[source] - offset, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(slow_new[target[:2]], slow_centers[source[:2]] + offset[:2], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(slow_new_mass[target], slow_mass[source] / 2)


@pytest.mark.gpu
def test_split_stays_inside_the_coordinate_range_and_separates_represented_copies(metal):
    from mlxlearn.cluster._e_means import removal_plan, revive_by_cost

    rng = np.random.default_rng(8)
    k, d = 8, 5
    centers = rng.normal(size=(k, d)).astype(np.float32)
    mass = np.full(k, 3, np.float32)
    long_axis = rng.normal(size=(k, d)).astype(np.float32)
    lengths = np.linalg.norm(long_axis, axis=1).astype(np.float32)
    axes = (long_axis / lengths[:, None]).astype(np.float32)
    removal = np.array([0, 0, 0, 5, 6, 7, 8, 9], np.float32)
    plan = removal_plan(metal.array(removal), metal.array(mass), metal.array(lengths))
    source, target = _reference_plan(removal, mass, lengths)
    lower, upper = centers.min(0), centers.max(0)
    far = np.asarray(revive_by_cost(
        metal.array(centers), metal.array(mass), metal.array(long_axis * 1e4), metal.array(removal),
        metal.array(axes), metal.array(lengths * 1e4), plan, (metal.array(lower), metal.array(upper)))[0])
    assert (far >= lower).all() and (far <= upper).all()
    assert (far[source] != far[target]).any(axis=1).all()
    # A spread below the BF16 spacing: the represented copies must still differ.
    wide = (metal.array(np.full(d, -100, np.float32)), metal.array(np.full(d, 100, np.float32)))
    tiny = np.asarray(revive_by_cost(
        metal.array(centers), metal.array(mass), metal.array(long_axis * 1e-12), metal.array(removal),
        metal.array(axes), metal.array(lengths * 1e-12), plan, wide, precision="bfloat16")[0])
    assert (_bf16(tiny[source]) != _bf16(tiny[target])).any(axis=1).all()
    # At the top of the range the copy moves down instead of leaving it.
    top = centers.copy()
    top[source, 0] = upper[0]
    still = (metal.zeros((k, d)), metal.array(removal), metal.array(axes), metal.zeros((k,)), plan)
    capped = np.asarray(revive_by_cost(
        metal.array(top), metal.array(mass), *still, (metal.array(lower), metal.array(upper)))[0])
    np.testing.assert_array_equal(capped[target, 0], np.full(len(target), np.nextafter(upper[0], np.float32(-np.inf))))
    np.testing.assert_array_equal(capped[source], top[source])
    # A feature without any range leaves no adjacent value: the copies stay coincident.
    flat_lower, flat_upper = lower.copy(), upper.copy()
    flat_lower[0] = flat_upper[0]
    for precision in ("float32", "bfloat16"):
        flat = np.asarray(revive_by_cost(
            metal.array(top), metal.array(mass), *still,
            (metal.array(flat_lower), metal.array(flat_upper)), precision=precision)[0])
        np.testing.assert_array_equal(flat[target], top[source])
        np.testing.assert_array_equal(flat[source], top[source])
    # BF16 copies at zero move to the smallest normal BF16 value, which MLX's cast keeps.
    zeroed = centers.copy()
    zeroed[source, 0] = 0
    apart = revive_by_cost(metal.array(zeroed), metal.array(mass), *still, wide, precision="bfloat16")[0]
    np.testing.assert_array_equal(np.asarray(apart)[target, 0], np.full(len(target), 2.0**-126, np.float32))
    represented = np.asarray(apart.astype(metal.bfloat16).astype(metal.float32))
    assert (represented[target, 0] != represented[source, 0]).all()


@pytest.mark.gpu
@pytest.mark.parametrize("n,k,d", [(1, 2, 1), (33, 5, 31), (259, 17, 96), (70, 9, 200)])
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "float16"])
def test_twin_route_keeps_the_nearer_of_a_centroid_and_its_twin(metal, n, k, d, precision):
    from mlxlearn._kernels.e_means import twin_route

    rng = np.random.default_rng(n + k + d)
    rows = rng.normal(size=(n, d)).astype(np.float32)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    centers[k - 1] = centers[0]  # equal distances: the lower index wins
    labels = rng.integers(0, k, size=n).astype(np.uint32)
    twins = rng.integers(-1, k, size=n).astype(np.int32)
    if n > 2:
        labels[:2], twins[:2] = (0, k - 1), (k - 1, 0)
    dtype = {"float32": metal.float32, "bfloat16": metal.bfloat16, "float16": metal.float16}[precision]
    operand = metal.array(rows).astype(dtype)
    routed = np.asarray(twin_route(operand, metal.array(labels), metal.array(twins), metal.array(centers)))
    seen = np.asarray(operand.astype(metal.float32)).astype(np.float64)
    other = np.where(twins < 0, labels, twins)
    own_distance = ((seen - centers[labels].astype(np.float64)) ** 2).sum(1)
    other_distance = ((seen - centers[other].astype(np.float64)) ** 2).sum(1)
    switch = (other_distance < own_distance) | ((other_distance == own_distance) & (other < labels))
    expected = np.where(switch, other, labels)
    # Float32 sums may order two distances that float64 holds a rounding apart.
    close = np.abs(other_distance - own_distance) <= 1e-5 * np.maximum(own_distance, 1e-30)
    decided = ~close | (centers[other] == centers[labels]).all(1)
    assert decided.sum() >= n - 1
    np.testing.assert_array_equal(routed[decided], expected[decided])
    assert np.isin(routed, np.stack((labels, other))).all() and routed.dtype == np.uint32
    assert (routed[twins < 0] == labels[twins < 0]).all()
    if n > 2:
        assert routed[0] == 0 and routed[1] == 0
    with pytest.raises(ValueError, match="uint32"):
        twin_route(operand, metal.array(labels.astype(np.int32)), metal.array(twins), metal.array(centers))
    with pytest.raises(ValueError, match="one entry per row"):
        twin_route(operand, metal.array(labels), metal.array(twins[:-1]), metal.array(centers))


def _closing_log_case(seed, k=9, steps=37):
    """Random closing pass: labels per batch, and at some steps the moves of a check."""
    rng = np.random.default_rng(seed)
    batches, checks = [], {}
    for step in range(steps):
        batches.append(rng.integers(0, k, size=int(rng.integers(1, 12))).astype(np.uint32))
        if step % 4 == 3:
            pairs = int(rng.integers(0, 4))
            chosen = rng.permutation(k)[:2 * pairs]
            checks[step] = (chosen[:pairs], chosen[pairs:])  # donors, the centroids moved onto them
    return batches, checks


def _closing_log_oracle(batches, checks, k):
    """Each row by itself: is it stale, and which centroid is its twin."""
    labels = np.concatenate(batches)
    events, moved_at, done = [], np.full(k, -1), 0
    for step, batch in enumerate(batches):
        done += len(batch)
        if step in checks:
            donors, children = checks[step]
            events.append((done, dict(zip(donors.tolist(), children.tolist(), strict=True))))
            moved_at[children] = done
    stale, twins = np.zeros(len(labels), bool), np.full(len(labels), -1)
    kinds = {"expired": 0, "second_split": 0}
    for rank, label in enumerate(labels):
        if rank < moved_at[label]:
            stale[rank] = True
            continue
        later = [(time, halves[label]) for time, halves in events if time > rank and label in halves]
        if later and moved_at[later[0][1]] == later[0][0]:
            twins[rank] = later[0][1]
        elif later:
            kinds["expired"] += 1
        kinds["second_split"] += len(later) > 1
    return labels, stale, twins, kinds


@pytest.mark.gpu
@pytest.mark.parametrize("segment", [None, 1, 2, 3])
def test_closing_log_finds_stale_rows_and_twins_as_a_row_by_row_search(metal, segment):
    from mlxlearn.cluster._e_means import ClosingLog

    k = 9
    seen = {"stale": 0, "twins": 0, "expired": 0, "second_split": 0}
    for seed in range(6):
        batches, checks = _closing_log_case(seed, k)
        # ``segment`` checks are kept between sweeps: four bytes per centroid each.
        log = ClosingLog(k, stash_bytes=None if segment is None else 4 * k * segment)
        for step, batch in enumerate(batches):
            log.rows(metal.array(batch))
            if step in checks:
                donors, children = checks[step]
                is_child, is_donor = np.zeros(k, bool), np.zeros(k, bool)
                target = np.zeros(k, np.uint32)
                is_child[children], is_donor[donors], target[donors] = True, True, children
                metal.eval(*log.check((metal.array(is_child), metal.array(is_donor), None,
                                       metal.array(target), metal.array(len(donors), dtype=metal.int32))))
        labels, stale, twins = (np.asarray(value) for value in log.finish())
        expected = _closing_log_oracle(batches, checks, k)
        np.testing.assert_array_equal(labels, expected[0])
        np.testing.assert_array_equal(stale, expected[1])
        np.testing.assert_array_equal(twins, expected[2])
        assert log.done == len(labels)
        assert int(log.moves.item()) == sum(len(donors) for donors, _ in checks.values())
        seen["stale"] += int(stale.sum())
        seen["twins"] += int((twins >= 0).sum())
        seen["expired"] += expected[3]["expired"]
        seen["second_split"] += expected[3]["second_split"]
    assert all(seen.values())


@pytest.mark.gpu
def test_closing_log_without_moves_marks_no_row(metal):
    from mlxlearn.cluster._e_means import ClosingLog

    k = 4
    log = ClosingLog(k)
    nothing = (metal.zeros((k,), metal.bool_), metal.zeros((k,), metal.bool_), None,
               metal.zeros((k,), metal.uint32), metal.array(0, dtype=metal.int32))
    log.rows(metal.array(np.array([0, 3, 1], np.uint32)))
    log.check(nothing)
    log.rows(metal.array(np.array([2, 2], np.uint32)))
    labels, stale, twins = (np.asarray(value) for value in log.finish())
    assert labels.tolist() == [0, 3, 1, 2, 2] and not stale.any() and (twins == -1).all()
    assert int(log.moves.item()) == 0
    empty = ClosingLog(k).finish()
    assert all(value.shape == (0,) for value in empty)


def test_rows_looked_up_again_are_assigned_in_recurring_call_sizes():
    from mlxlearn.cluster._e_means import _search_chunks

    assert _search_chunks(0, 4096) == []
    assert _search_chunks(1, 4096) == [(1, 128)]
    assert _search_chunks(4096, 4096) == [(4096, 4096)]
    assert _search_chunks(35000, 32768) == [(32768, 32768), (2048, 2048), (184, 1024)]
    # Just above a power of two: split, not padded to the next one.
    assert _search_chunks(35000, 65536) == [(32768, 32768), (2048, 2048), (184, 1024)]
    assert _search_chunks(50000, 65536) == [(50000, 65536)]
    for batch_size in (16, 200, 1100, 4096, 65536):
        for count in (1, 5, 127, 128, 129, 1023, 1024, 1025, 3000, 49152, 49153, 200_000):
            chunks = _search_chunks(count, batch_size)
            assert sum(rows for rows, _ in chunks) == count
            assert all(0 < rows <= size <= batch_size for rows, size in chunks)
            # A call takes a whole training batch or a power of two.
            assert all(size == batch_size or size & (size - 1) == 0 for _, size in chunks)
            padding = sum(size - rows for rows, size in chunks)
            assert padding < max(1024, count // 3 + 1)


def _oracle_fit(x, init, axes, *, seed, passes, batch_size, peak, slow_step, init_mass, bounds):
    """Float64 e-means with cost-based revival: assignment, statistics, EMA, moves, closing.

    The closing looks at every closing-pass row by itself. A row whose centroid
    was moved onto a donor afterwards is assigned again to the final centroids.
    A row whose centroid was split afterwards goes to the nearer of that
    centroid and the one that took the other half at the first such split,
    unless that one moved again. Returns the EMA centroids, the closing means,
    the slow EMA centroids, the number of moves, the counts of each kind of
    row, and the closing-pass rows with their final labels.
    """
    x = x.astype(np.float64)
    k, d = init.shape
    centers, mass = init.astype(np.float64).copy(), np.full(k, init_mass)
    slow, slow_mass = centers.copy(), mass.copy()
    long_axis, removal, axes = np.zeros((k, d)), np.zeros(k), axes.astype(np.float64).copy()
    lower, upper = bounds
    total_steps = sum(math.ceil(rows / batch_size) for rows in passes)
    last = len(passes) - 1
    seen, assigned, splits, moved_at = [], [], [], np.full(k, -1)
    step = moves = 0
    for epoch, rows in enumerate(passes):
        order = np.random.RandomState(epoch_seed(seed, epoch, distinct=True)).permutation(len(x))[:rows]
        for start in range(0, rows, batch_size):
            batch = x[order[start:start + batch_size]]
            labels, _, margins = _two_nearest(batch, centers)
            sums, counts = np.zeros((k, d)), np.bincount(labels, minlength=k).astype(np.float64)
            np.add.at(sums, labels, batch)
            y = batch - centers[labels]
            projection = (y * axes[labels]).sum(1, keepdims=True)
            axis_sums, costs = np.zeros((k, d)), np.zeros(k)
            np.add.at(axis_sums, labels, projection * y)
            np.add.at(costs, labels, margins)
            if epoch == last:
                seen.extend(order[start:start + batch_size])
                assigned.extend(labels)
            alpha = peak * (0.1 + 0.9 * (1 + math.cos(math.pi * step / total_steps)) / 2)
            for position, weight, rate in ((centers, mass, alpha), (slow, slow_mass, slow_step)):
                updated = (1 - rate) * weight + rate * counts
                hit = (counts > 0) & (updated > 0)
                position[hit] = ((1 - rate) * weight[hit, None] * position[hit] + rate * sums[hit]) / updated[hit, None]
                weight[:] = updated
            long_axis = (1 - alpha) * long_axis + alpha * axis_sums
            removal = (1 - alpha) * removal + alpha * costs
            lengths = np.linalg.norm(long_axis, axis=1)
            axes = np.where(lengths[:, None] > 0, long_axis / np.where(lengths > 0, lengths, 1)[:, None], axes)
            if (step + 1) % 5 == 0:
                source, target = _reference_plan(removal, mass, lengths)
                if len(source):
                    spread = np.sqrt(2 * lengths[source] / (math.pi * mass[source]))
                    offset = spread[:, None] * axes[source]
                    for position, weight in ((centers, mass), (slow, slow_mass)):
                        origin = position[source].copy()
                        position[source] = np.clip(origin - offset, lower, upper)
                        position[target] = np.clip(origin + offset, lower, upper)
                        weight[source] /= 2
                        weight[target] = weight[source]
                    long_axis[source] /= 2
                    long_axis[target] = long_axis[source]
                    removal[source] /= 2
                    removal[target] = removal[source]
                    axes[target] = axes[source]
                    moves += len(source)
                    if epoch == last:
                        splits.append((len(seen), dict(zip(source.tolist(), target.tolist(), strict=True))))
                        moved_at[target] = len(seen)
            step += 1
    final = np.array(assigned)
    kinds = {"stale": 0, "twins": 0, "switched": 0, "expired": 0}
    for rank, (row, label) in enumerate(zip(seen, assigned, strict=True)):
        if rank < moved_at[label]:
            final[rank] = np.argmin(((x[row] - centers) ** 2).sum(1))
            kinds["stale"] += 1
            continue
        later = [(time, halves[label]) for time, halves in splits if time > rank and label in halves]
        if not later:
            continue
        time, twin = later[0]
        if moved_at[twin] != time:
            kinds["expired"] += 1
            continue
        own, other = ((x[row] - centers[label]) ** 2).sum(), ((x[row] - centers[twin]) ** 2).sum()
        kinds["twins"] += 1
        if other < own or (other == own and twin < label):
            final[rank] = twin
            kinds["switched"] += 1
    rows_seen = np.array(seen)
    closing = centers.copy()
    for centroid in range(k):
        members = rows_seen[final == centroid]
        if len(members):
            closing[centroid] = x[members].mean(0)
    kinds["split_again"] = sum(sum(c in halves for _, halves in splits) > 1 for c in range(k))
    kinds["without_rows"] = k - len(np.unique(final))
    return centers, closing, slow, moves, kinds, rows_seen, final


def _blobs(seed, *, rows=320, blobs=8, d=6):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(blobs, d)) * 6
    member = rng.integers(0, blobs, size=rows)
    return (means[member] + rng.normal(size=(rows, d))).astype(np.float32), member


def _oracle_case():
    """Rows, initial centroids and options of the whole-fit oracle comparison."""
    x, member = _blobs(14)
    k, seed = 12, 23
    # Start every centroid in three of the eight blobs: revival has to move them.
    init = x[np.flatnonzero(member < 3)[:k]].copy()
    bounds = (np.minimum(x.min(0), init.min(0)), np.maximum(x.max(0), init.max(0)))
    half_life = 9
    # Half a pass, then a closing pass in which centroids still move.
    oracle = {"seed": seed, "passes": [160, 320], "batch_size": 16, "peak": 0.3,
              "slow_step": 1 - 2 ** (-1 / half_life), "init_mass": 1e-10, "bounds": bounds}
    options = {"n_clusters": k, "init": init, "max_iter": 1.5, "batch_size": 16, "alpha0": 0.3,
               "random_state": seed, "revival": "cost", "slow_ema_half_life": half_life}
    return x, init, oracle, options


def _starting_axes(metal, k, d, seed):
    axes = np.asarray(metal.random.normal((k, d), key=metal.random.key(seed))).astype(np.float64)
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)


@pytest.mark.gpu
@pytest.mark.parametrize("finalization,index", [("ema", 0), ("last_epoch", 1), ("slow_ema", 2)])
def test_training_with_cost_revival_matches_independent_double_oracle(metal, finalization, index):
    from mlxlearn.cluster import EMeans

    x, init, oracle, options = _oracle_case()
    expected = _oracle_fit(x, init, _starting_axes(metal, *init.shape, oracle["seed"]), **oracle)
    model = EMeans(finalization=finalization, **options).fit(x)
    # The closing pass has every kind of row: stale ones, rows that stay with a
    # split centroid and rows that go to its twin, rows whose twin moved again,
    # a centroid split twice, and centroids left without rows.
    kinds = expected[4]
    assert expected[3] >= 6 and kinds["stale"] and kinds["expired"] and kinds["split_again"]
    assert 0 < kinds["switched"] < kinds["twins"] and kinds["without_rows"]
    assert model.n_splits_ == expected[3]
    np.testing.assert_allclose(model.ema_centers_, expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.last_epoch_centers_, expected[1], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.slow_ema_centers_, expected[2], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.cluster_centers_, expected[index], rtol=3e-5, atol=3e-5)
    assert model.recipe_["revival"] == "cost"
    assert model.recipe_["closing"] == "last_epoch_mean_of_rerouted_rows"
    assert model.recipe_["unused_threshold"] is None and model.recipe_["split_epsilon"] is None
    assert model.recipe_["removal_threshold"] == REMOVAL_THRESHOLD


@pytest.mark.gpu
@pytest.mark.parametrize("stash_bytes", [None, 1])
def test_captured_assignments_are_the_rows_behind_the_closing_means(metal, monkeypatch, stash_bytes):
    import mlxlearn.cluster._e_means as module
    from mlxlearn.cluster import EMeans

    # One byte keeps a single revival check at a time, so that every check is
    # swept by itself and earlier rows take their twins from later sweeps.
    if stash_bytes is not None:
        monkeypatch.setattr(module, "_CLOSING_STASH_BYTES", stash_bytes)
    x, init, oracle, options = _oracle_case()
    expected = _oracle_fit(x, init, _starting_axes(metal, *init.shape, oracle["seed"]), **oracle)
    model = EMeans(**options)
    result = model.train_device(model.prepare_device(x), capture_assignments=True)
    captured = np.asarray(result.assignments)
    rows, final = expected[5], expected[6]
    np.testing.assert_array_equal(captured[rows], final)
    closing = np.asarray(result.last_epoch_centers)
    np.testing.assert_allclose(closing, expected[1], rtol=3e-5, atol=3e-5)
    for centroid in range(len(init)):
        members = x[rows[final == centroid]].astype(np.float64)
        if len(members):
            np.testing.assert_allclose(closing[centroid], members.mean(0), rtol=3e-5, atol=3e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("spherical", [False, True])
def test_cost_revival_separates_blobs_that_mass_revival_leaves_merged(metal, precision, spherical):
    from mlxlearn.cluster import EMeans

    # Forty separated blobs and forty centroids drawn from the rows: some blobs
    # start with several centroids and others with none. No centroid is short
    # of mass, so the mass rule has nothing to revive.
    rng = np.random.default_rng(0)
    means = rng.normal(size=(40, 16)).astype(np.float32) * 6
    x = means[rng.integers(0, 40, size=20000)] + rng.normal(size=(20000, 16)).astype(np.float32)
    scores = {}
    for revival in ("mass", "cost"):
        model = EMeans(40, max_iter=3, random_state=5, revival=revival, precision=precision,
                       spherical=spherical).fit(x)
        assert np.isfinite(model.cluster_centers_).all()
        scores[revival] = -model.score(x) / len(x)
        if revival == "cost":
            assert model.n_splits_ > 0
    assert scores["cost"] < 0.75 * scores["mass"]


@pytest.mark.gpu
def test_closing_means_repair_the_running_positions_when_most_centroids_move(metal):
    from mlxlearn.cluster import EMeans

    # More centroids than separated clusters: in a single pass most centroids
    # are moved or split, many of them shortly before the end, and the running
    # positions of those are poor. The closing means are taken over the rows
    # each centroid holds at the end and must not inherit that.
    rng = np.random.default_rng(1)
    means = rng.normal(size=(1500, 32)).astype(np.float32) * 3
    x = means[rng.integers(0, 1500, size=110_000)] + rng.normal(size=(110_000, 32)).astype(np.float32)
    train, test = x[:100_000], x[100_000:].astype(np.float64)

    def error(centers):
        centers = centers.astype(np.float64)
        distance = (test * test).sum(1)[:, None] - 2 * test @ centers.T + (centers * centers).sum(1)[None]
        return float(distance.min(1).mean())

    options = {"max_iter": 1, "recipe": "faiss", "random_state": 4, "compute_labels": False}
    model = EMeans(2048, revival="cost", **options).fit(train)
    assert model.n_splits_ > 2048
    assert error(model.last_epoch_centers_) < 0.95 * error(model.ema_centers_)
    assert error(model.last_epoch_centers_) < 0.97 * error(EMeans(2048, **options).fit(train).cluster_centers_)


@pytest.mark.gpu
def test_cost_revival_is_repeatable_and_keeps_centroids_inside_the_data_range(metal):
    from mlxlearn.cluster import EMeans

    x, _ = _blobs(3, rows=600, blobs=12, d=6)
    options = {"n_clusters": 24, "max_iter": 2, "batch_size": 30, "random_state": 0,
               "revival": "cost", "compute_labels": False}
    first, second = EMeans(**options).fit(x), EMeans(**options).fit(x)
    # Batch sums are float32 atomics, so repeated fits agree to rounding only.
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_, rtol=1e-4, atol=1e-4)
    assert first.n_splits_ == second.n_splits_ > 0
    for centers in (first.cluster_centers_, first.ema_centers_):
        assert (centers >= x.min(0) - 1e-6).all() and (centers <= x.max(0) + 1e-6).all()
    assert first.recipe_["epoch_seed"] == "distinct_from_initialization"


@pytest.mark.gpu
def test_cost_revival_with_one_centroid_and_with_captured_assignments(metal):
    from mlxlearn.cluster import EMeans

    x, _ = _blobs(5, rows=600, blobs=4, d=3)
    single = EMeans(1, max_iter=2, batch_size=50, revival="cost", random_state=2).fit(x)
    assert single.n_splits_ == 0
    np.testing.assert_allclose(single.cluster_centers_[0], x.mean(0), rtol=1e-4, atol=1e-4)
    model = EMeans(8, max_iter=1.5, batch_size=50, revival="cost", random_state=2)
    result = model.train_device(model.prepare_device(x), capture_assignments=True)
    captured = np.asarray(result.assignments)
    assert captured.shape == (600,) and captured.min() >= 0 and captured.max() < 8
    assert np.isfinite(np.asarray(result.centers)).all()


def test_revival_parameter_is_validated_and_cloned():
    from sklearn.base import clone

    from mlxlearn.cluster import EMeans, OMeans

    x = np.zeros((8, 2), np.float32)
    with pytest.raises(ValueError, match="revival must be 'mass' or 'cost'"):
        EMeans(2, revival="count").fit(x)
    with pytest.raises(ValueError, match="unused_threshold applies to revival='mass'"):
        EMeans(2, revival="cost", unused_threshold=0.1).fit(x)
    assert EMeans().get_params()["revival"] == "mass"
    assert clone(EMeans(3, revival="cost")).get_params()["revival"] == "cost"
    # o-means keeps mass-based revival and does not expose the option.
    assert "revival" not in OMeans().get_params() and OMeans.revival == "mass"


@pytest.mark.gpu
def test_mass_revival_records_its_rule_and_keeps_the_reference_pass_seed(metal):
    from mlxlearn.cluster import EMeans

    x, _ = _blobs(9, rows=400, blobs=5, d=4)
    assert EMeans(6, max_iter=1, random_state=3).fit(x).recipe_["revival"] == "mass"
    off = EMeans(6, max_iter=1, random_state=3, unused_threshold=0).fit(x)
    assert off.recipe_["revival"] == "none" and off.n_splits_ == 0
    assert "epoch_seed" not in off.recipe_
