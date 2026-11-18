"""Numerical certificates for skipping exhaustive Lloyd assignments."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._kernels import lloyd_bounds
from mlxlearn._kernels.e_means import _streaming_fp32_device, assign, selected_distances
from mlxlearn.cluster import _k_means

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")]


def test_tiled_separations_are_lower_bounds_and_exclude_only_the_same_id(monkeypatch):
    rng = np.random.default_rng(9403)
    coords = rng.normal(size=(19, 41)).astype(np.float32)
    coords[11] = coords[4]
    centers = mx.array(coords)
    norms = mx.sum(centers * centers, axis=1)
    ids = np.array([7, 4, 11, 3, 18, 7], np.uint32)
    monkeypatch.setattr(lloyd_bounds, "_CENTER_TILE_BYTES", 2 * len(coords) * 4)
    result = np.asarray(lloyd_bounds.separation_lower(centers, mx.array(ids), norms, mx.max(norms)))
    full = np.linalg.norm(coords[ids, None].astype(np.float64) - coords[None, :], axis=2)
    full[np.arange(len(ids)), ids] = np.inf
    exact = full.min(axis=1)
    assert np.all(result <= exact)
    assert np.all(result[exact > 0] >= 0.99 * exact[exact > 0])
    assert np.all(result[exact == 0] == 0.0)


@pytest.mark.parametrize("d", [1, 96, 1024])
def test_only_proven_incumbents_are_kept_and_return_original_residual(d):
    rng = np.random.default_rng(9404)
    coords = rng.normal(size=(32, d)).astype(np.float32)
    coords[1] = coords[0]
    labels = np.arange(len(coords), dtype=np.uint32)
    nearby = coords + rng.normal(scale=1e-5, size=coords.shape).astype(np.float32)
    # First batch has good incumbents except the duplicate. Every other
    # query in the second batch uses a deliberately wrong incumbent.
    queries = np.concatenate([nearby, nearby])
    incumbents = np.concatenate([labels, np.roll(labels, 1)])
    centers, x, incumbent = mx.array(coords), mx.array(queries), mx.array(incumbents)
    norms = mx.sum(centers * centers, axis=1)
    largest = mx.max(norms)
    sep = lloyd_bounds.separation_lower(centers, mx.array(labels), norms, largest)
    certain, residual = lloyd_bounds.reusable_labels(x, centers, incumbent, sep[incumbent], largest)
    expected, _ = assign(x, centers, method="gemm")
    accepted = np.asarray(certain)
    assert np.count_nonzero(accepted[:32]) >= 20
    assert not accepted[:2].any()
    np.testing.assert_array_equal(incumbents[accepted], np.asarray(expected)[accepted])
    np.testing.assert_array_equal(np.asarray(residual).view(np.uint32),
                                  np.asarray(selected_distances(x, centers, incumbent)).view(np.uint32))


@pytest.mark.parametrize("offset,delta", [(0.0, 1.0), (2.0**58, 2.0**36), (0.0, 2.0**-70)])
def test_roundoff_and_boundaries_fall_back_to_exhaustive_assignment(offset, delta):
    # In the large-offset row, subtracting two nearly equal Gram terms
    # destroys the distinction in distance. The tiny row exercises flushing.
    coords = np.full((2, 32), offset, dtype=np.float32)
    coords[1, 0] += delta
    midpoint = coords[0].copy()
    midpoint[0] += delta * 0.5
    x, centers = mx.array(np.stack([midpoint, midpoint])), mx.array(coords)
    ids = mx.arange(2, dtype=mx.uint32)
    norms = mx.sum(centers * centers, axis=1)
    maximum = mx.max(norms)
    sep = lloyd_bounds.separation_lower(centers, ids, norms, maximum)
    certain, _ = lloyd_bounds.reusable_labels(x, centers, ids, sep, maximum)
    assert not np.asarray(certain).any()


@pytest.mark.parametrize("uncertain_count,rows", [(1, 128), (139, 160)])
def test_pruned_tile_scatter_preserves_full_results_with_tied_uncertain_rows(
    uncertain_count, rows, monkeypatch,
):
    rng = np.random.default_rng(1446)
    c = rng.normal(size=(4096, 96)).astype(np.float32)
    c[1] = c[0]
    ids = np.full(319, 2, np.uint32)
    ambiguous = rng.choice(len(ids), uncertain_count, replace=False)
    ids[ambiguous] = 1
    q = c[ids] + rng.normal(scale=0.01, size=(len(ids), 96)).astype(np.float32)
    if uncertain_count > 1:
        # Mix exact ties and queries very near a two-center boundary.
        q[ambiguous[1::2]] = np.nextafter((c[0] + c[3]) * 0.5, c[3])
    x, centers, incumbent = mx.array(q), mx.array(c), mx.array(ids)
    norms = mx.sum(centers * centers, axis=1)
    maximum = mx.max(norms)
    radii = mx.zeros((len(c),), mx.float32)
    requested = mx.arange(3)
    radii = radii.at[requested].add(lloyd_bounds.separation_lower(centers, requested, norms, maximum))
    expected, expected_residual = assign(x, centers, method="gemm")
    calls = []

    def trace_assign(points, centers, **kwargs):
        calls.append(len(points))
        return assign(points, centers, **kwargs)

    monkeypatch.setattr(_k_means, "assign", trace_assign)
    labels, residual, saved = _k_means._pruned_tile(x, centers, incumbent, radii, maximum)
    assert saved == len(x) - uncertain_count
    assert calls == [rows]
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(residual).view(np.uint32),
                                  np.asarray(expected_residual).view(np.uint32))


def test_final_pruned_assignment_scores_restored_coordinates(monkeypatch):
    # The offset rounds away the small displacement from the negative center.
    # The restored-coordinate residual must retain that displacement.
    raw = np.zeros((256, 96), dtype=np.float32)
    raw[:128, 0] = -1.001
    raw[128:, 0] = 2**21
    original = np.zeros((2, 96), dtype=np.float32)
    original[:, 0] = [-1, 2**21]
    offset = mx.array(original.mean(axis=0))
    centers = mx.array(original)
    data = mx.array(raw)
    old = mx.array(np.repeat([0, 1], 128).astype(np.uint32))
    model = _k_means.KMeans(n_clusters=2)

    def enable_small_fixture(data, centered_centers, offset, previous):
        norms = mx.sum(centered_centers * centered_centers, axis=1)
        maximum = mx.max(norms)
        return lloyd_bounds.separation_lower(centered_centers, mx.arange(2), norms, maximum), maximum

    monkeypatch.setattr(model, "_exclusion_radii", enable_small_fixture)
    labels, residual, saved = model._assignment(
        data, centers - offset, offset, original_centers=centers, previous=old,
    )
    expected, centered_residual = assign(data - offset, centers - offset, method="gemm")
    direct = selected_distances(data, centers, expected)
    assert saved == len(data)
    assert float(direct[0].item()) > float(centered_residual[0].item())
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(residual).view(np.uint32), np.asarray(direct).view(np.uint32))


def test_mixed_reuse_matches_dense_assignment_on_a_bounded_set_of_batch_sizes(monkeypatch):
    if not _streaming_fp32_device():
        pytest.skip("label reuse is calibrated for the M4 Max streaming kernel")
    rng = np.random.default_rng(2217)
    k, d, tile = 4096, 64, 2048
    n = 8 * k
    truth = rng.normal(0, 4, (k, d))
    # Rows of the wide clusters sit too close to a boundary to be certified,
    # so every tile mixes reused and uncertain rows in a different proportion.
    spread = np.where(rng.random(k) < 0.1, 2.6, 0.25)
    owner = rng.permutation(n) % k
    data = mx.array((truth[owner] + rng.normal(0, 1, (n, d)) * spread[owner][:, None]).astype(np.float32))
    before = mx.array((truth + rng.normal(0, 0.3, (k, d))).astype(np.float32))
    centers = before + mx.array(rng.normal(0, 0.02, (k, d)).astype(np.float32))
    offset = mx.mean(data, axis=0)
    previous, _ = assign(data - offset, before - offset)
    expected, expected_residual = assign(data - offset, centers - offset)
    calls = []

    def trace_assign(points, centers, **kwargs):
        calls.append(len(points))
        return assign(points, centers, **kwargs)

    monkeypatch.setattr(_k_means, "assign", trace_assign)
    monkeypatch.setattr(_k_means, "_scratch_budget", lambda: 4 * k * tile)
    labels, residual, saved = _k_means.KMeans(n_clusters=k)._assignment(
        data, centers - offset, offset, previous=previous,
    )
    assert len(calls) == n // tile
    assert 0 < saved < n
    # Tiles differ in their uncertain count. Their batches are padded to a few sizes.
    assert all(128 <= rows < tile and rows == _k_means._compacted_rows(rows) for rows in calls)
    assert n - saved < sum(calls) < 1.25 * (n - saved)
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(residual).view(np.uint32),
                                  np.asarray(expected_residual).view(np.uint32))
