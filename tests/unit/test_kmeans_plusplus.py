"""k-means++ seeding: reference centers, the Metal screen's proofs, and the sampling law."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


@pytest.fixture
def plusplus(monkeypatch):
    """The seeding module with the Metal screen enabled for small inputs."""
    from mlxlearn.cluster import _plusplus

    monkeypatch.setattr(_plusplus, "_MIN_SCREEN_WORK", 0)
    return _plusplus


def dataset(kind, n, d, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        x = rng.normal(size=(n, d))
    elif kind == "offset":
        # Norms far above the distances: the reference's float64 round-off is
        # then a visible fraction of a float32 unit.
        x = rng.normal(size=(n, d)) + 1000.0
    elif kind == "integers":
        x = rng.integers(0, 4, size=(n, d)).astype(float)
    elif kind == "duplicates":
        distinct = max(n // 8, 2)
        x = rng.normal(size=(distinct, d))[rng.integers(0, distinct, n)]
    elif kind == "blobs":
        x = rng.normal(size=(5, d))[rng.integers(0, 5, n)] * 10 + rng.normal(size=(n, d))
    elif kind == "tiny":
        x = rng.normal(size=(n, d)) * 1e-18
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def seed_indices(mx, module, x, k, seed, profile=None):
    data = mx.array(x)
    state = np.random.RandomState(seed)
    indices = module.plusplus_indices(data, np.asarray(data), k, state, profile=profile)
    return indices, state


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["normal", "offset", "integers", "duplicates", "blobs", "tiny"])
@pytest.mark.parametrize("n,d,k", [(9, 8, 3), (257, 16, 50), (1000, 5, 8), (1533, 7, 300),
                                   (3000, 130, 60), (4097, 33, 404)])
def test_screened_seeding_selects_the_reference_centers(metal, plusplus, kind, n, d, k):
    x = dataset(kind, n, d)
    for seed in (0, 1):
        profile = {}
        indices, state = seed_indices(metal, plusplus, x, k, seed, profile)
        expected_state = np.random.RandomState(seed)
        _, expected = kmeans_plusplus(x, k, random_state=expected_state)
        np.testing.assert_array_equal(indices, expected)
        # The same random numbers were consumed, so later restarts agree too.
        assert state.uniform() == expected_state.uniform()
        assert "certified" in profile


@pytest.mark.gpu
def test_screen_carries_the_seeding_on_ordinary_data(metal, plusplus):
    x = dataset("blobs", 6000, 48, seed=3)
    profile = {}
    indices, _ = seed_indices(metal, plusplus, x, 200, 7, profile)
    np.testing.assert_array_equal(indices, kmeans_plusplus(x, 200, random_state=7)[1])
    assert profile["certified"] and profile["steps"] == 199
    # After the first centers only a small share of rows reaches the host.
    assert profile["dense_steps"] < 20
    assert profile["rows"] < 0.2 * 199 * len(x)


@pytest.mark.gpu
def test_noise_only_potential_falls_back_to_the_reference_routine(metal, plusplus):
    # More centers than distinct rows: once every row coincides with a center
    # the reference samples its own float64 round-off, which cannot be certified.
    x = dataset("duplicates", 257, 16)
    profile = {}
    indices, state = seed_indices(metal, plusplus, x, 50, 0, profile)
    expected_state = np.random.RandomState(0)
    np.testing.assert_array_equal(indices, kmeans_plusplus(x, 50, random_state=expected_state)[1])
    assert profile["certified"] is False and 0 < profile["steps"] < 49
    assert state.uniform() == expected_state.uniform()


@pytest.mark.gpu
def test_default_gate_uses_the_screen_for_large_inputs_only(metal):
    from mlxlearn.cluster import _plusplus

    small = dataset("normal", 400, 6)
    profile = {}
    indices, _ = seed_indices(metal, _plusplus, small, 12, 5, profile)
    np.testing.assert_array_equal(indices, kmeans_plusplus(small, 12, random_state=5)[1])
    assert profile == {}
    large = dataset("blobs", 20000, 64, seed=1)
    indices, _ = seed_indices(metal, _plusplus, large, 512, 5, profile)
    np.testing.assert_array_equal(indices, kmeans_plusplus(large, 512, random_state=5)[1])
    assert profile["certified"]


@pytest.mark.parametrize("kind", ["blobs", "offset"])
@pytest.mark.parametrize("n,d,count", [(700, 9, 1), (700, 9, 6), (1129, 16, 5), (64, 200, 3), (2500, 40, 11)])
def test_reference_distances_reproduce_scikit_learn_bit_for_bit(kind, n, d, count):
    from mlxlearn.cluster._plusplus import _Reference

    x = dataset(kind, n, d, seed=2)
    reference = _Reference(x)
    candidates = x[np.random.default_rng(0).integers(0, n, count)]
    expected = euclidean_distances(candidates, x, squared=True)
    np.testing.assert_array_equal(reference.every(candidates), expected)
    # Products over selected rows may round differently: their bounds must
    # contain the reference value, and pin it unless the norms dwarf the distance.
    rows = np.sort(np.random.default_rng(1).choice(n, n // 3, replace=False))
    position, which = np.meshgrid(np.arange(len(rows)), np.arange(count))
    least, most = reference.bounds(candidates, rows, which.ravel(), position.ravel())
    target = expected[:, rows].ravel()
    assert np.all(least <= target) and np.all(target <= most)
    if kind == "blobs":
        # A row equal to a candidate has only round-off for a distance.
        distinct = target > 1e-6 * target.max()
        assert np.mean(least[distinct] == most[distinct]) > 0.99


def test_reference_chunk_layout_keeps_a_single_trailing_row():
    from mlxlearn.cluster._plusplus import _Reference

    # Choose n so the reference's last chunk holds exactly one row.
    d, count = 12, 4
    length = _Reference(dataset("normal", 64, d)).length(count)
    n = length + 1
    x = dataset("offset", n, d, seed=4)
    reference = _Reference(x)
    assert reference.length(count) == length
    candidates = x[[0, 5, n - 1, 17]]
    np.testing.assert_array_equal(reference.every(candidates),
                                  euclidean_distances(candidates, x, squared=True))


@pytest.mark.gpu
@pytest.mark.parametrize("n,d,trials", [(64, 4, 2), (1000, 7, 5), (333, 33, 8), (4099, 16, 11)])
def test_screen_only_proves_pairs_the_reference_leaves_unchanged(metal, n, d, trials):
    from mlxlearn._kernels.kmeans_plusplus import candidate_screen

    mx = metal
    rng = np.random.default_rng(n + d)
    x = dataset("blobs", n, d, seed=5)
    ids = rng.integers(0, n, trials)
    reference = euclidean_distances(x[ids], x, squared=True)
    # Limits within a few float32 steps of the true distance are the hard
    # cases; others are well above or below it.
    steps = rng.integers(-3, 4, size=n)
    pick = reference[rng.integers(0, trials, n), np.arange(n)]
    near = pick.copy()
    for _ in range(3):
        near = np.where(steps > 0, np.nextafter(near, np.float32(np.inf)),
                        np.where(steps < 0, np.nextafter(near, np.float32(0)), near))
    choice = rng.integers(0, 3, n)
    high = np.where(choice == 0, near, np.where(choice == 1, pick * 0.5, pick * 2)).astype(np.float32)
    low = np.where(rng.random(n) < 0.2, np.float32(0), high).astype(np.float32)
    norms = (np.square(x.astype(np.float64)).sum(axis=1) * (1 + 2.0**-20)).astype(np.float32)
    padded = np.zeros(max(trials, 8), np.float32)
    padded[:trials] = norms[ids]
    table, mask, touched = candidate_screen(mx.array(x), mx.array(x[ids]), mx.array(low),
                                            mx.array(high), mx.array(norms), mx.array(padded))
    table, mask, touched = np.asarray(table), np.asarray(mask), np.asarray(touched)
    opened = ((mask[None, :] >> np.arange(trials, dtype=np.uint32)[:, None]) & 1).astype(bool)
    np.testing.assert_array_equal(touched.astype(bool), mask != 0)
    assert np.all(table[0][opened] == -1) and np.all(table[1][opened] == -1)
    np.testing.assert_array_equal(table[0][~opened], np.broadcast_to(low, opened.shape)[~opened])
    np.testing.assert_array_equal(table[1][~opened], np.broadcast_to(high, opened.shape)[~opened])
    # Soundness: a proved pair's reference distance does not go below the limit.
    assert np.all(reference[~opened] >= np.broadcast_to(high, opened.shape)[~opened])
    # Usefulness: pairs well above the limit are proved.
    clear = reference > 1.01 * high[None, :] + 1e-6
    assert np.all(~opened[clear])


def second_center_law(x, trials):
    """Exact law of the second center for greedy k-means++ with ``trials`` candidates.

    The first center is uniform. Candidates are drawn independently with
    probability proportional to the squared distance to it, and the candidate
    leaving the smallest potential wins.
    """
    n = len(x)
    squared = np.square(x[:, None, :] - x[None, :, :]).sum(axis=2)
    law = np.zeros(n)
    for first in range(n):
        weight = squared[first] / squared[first].sum()
        potential = np.minimum(squared[first][None, :], squared).sum(axis=1)
        order = np.argsort(potential)
        # P(winner = j) = P(all draws at least as bad as j) - P(all strictly worse).
        tail = np.concatenate([np.cumsum(weight[order][::-1])[::-1], [0.0]])
        law[order] += (tail[:-1] ** trials - tail[1:] ** trials) / n
    return law


@pytest.mark.gpu
def test_second_center_follows_the_greedy_d2_sampling_law(metal, plusplus):
    mx = metal
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.normal(size=(8, 8)), rng.normal(size=(4, 8)) + 6]).astype(np.float32)
    data = mx.array(x)
    host = np.asarray(data)
    runs = 3000
    counts = np.zeros(len(x))
    for seed in range(runs):
        indices = plusplus.plusplus_indices(data, host, 2, np.random.RandomState(seed))
        counts[indices[1]] += 1
    law = second_center_law(x.astype(np.float64), trials=2 + int(np.log(2)))
    assert law.sum() == pytest.approx(1)
    expected = runs * law
    # Twelve outcomes: the 99.9% point of chi-square with eleven degrees of freedom is 31.3.
    statistic = (np.square(counts - expected) / expected).sum()
    assert statistic < 31.3
    # The test has power: uniform seeding of the second center is rejected.
    uniform = runs / len(x)
    assert (np.square(counts - uniform) / uniform).sum() > 10 * 31.3


@pytest.mark.gpu
def test_k_means_initial_centers_are_the_reference_centers(metal, plusplus):
    from mlxlearn.cluster import KMeans

    x = dataset("blobs", 1500, 24, seed=6)
    model = KMeans(40, random_state=9, n_init=2)
    prepared = model.prepare_device(x)
    state = np.random.RandomState(9)
    for initial in prepared.initial:
        expected, _ = kmeans_plusplus(x, 40, random_state=state)
        np.testing.assert_array_equal(np.asarray(initial), expected)
