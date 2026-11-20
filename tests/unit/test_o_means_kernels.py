"""Two-level assignment kernels against independent float64 oracles.

The oracles search exactly the cells a row probes, by direct float64 scores
and host indexing, independently of the Metal kernels. BF16 oracles round on
the host.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


def _bf16(x):
    """Finite FP32 -> represented BF16 -> FP32, independently of MLX."""
    values = np.ascontiguousarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _mixture(rng, n, k, d, modes=None):
    """Rows around ``modes`` centres; centroids drawn near the same centres."""
    modes = modes or max(4, k // 4)
    means = rng.normal(size=(modes, d)).astype(np.float32) * 2
    x = means[rng.integers(0, modes, n)] + rng.normal(size=(n, d)).astype(np.float32) * 0.4
    c = means[rng.integers(0, modes, k)] + rng.normal(size=(k, d)).astype(np.float32) * 0.4
    return x.astype(np.float32), c.astype(np.float32)


def _nearest(points, centres):
    a, b = points.astype(np.float64), centres.astype(np.float64)
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(2).argmin(1)


def _lloyd_cells(centers, centres, iterations):
    """Float64 k-means over the centroids; an empty cell keeps its centre."""
    centres = centres.astype(np.float64).copy()
    for _ in range(iterations):
        cell = _nearest(centers, centres)
        for g in range(len(centres)):
            if np.any(cell == g):
                centres[g] = centers[cell == g].astype(np.float64).mean(0)
    return centres, _nearest(centers, centres)


def _cell_of(cells, k):
    """Cell of every centroid id, read back from the device layout."""
    order, offsets = np.asarray(cells.order), np.asarray(cells.offsets).astype(np.int64)
    cell = np.empty(k, dtype=np.int64)
    for g in range(len(offsets) - 1):
        cell[order[offsets[g]:offsets[g + 1]]] = g
    return cell


def _reference(x, centers, cells, n_probe, child=None):
    """Float64 search of each row's probed cells, then its winner's links."""
    k = len(centers)
    rows, table = x.astype(np.float64), centers.astype(np.float64)
    centres = np.asarray(cells.centres).astype(np.float64)
    sizes = np.asarray(cells.sizes)
    cell = _cell_of(cells, k)
    coarse = (centres ** 2).sum(1)[None, :] - 2 * rows @ centres.T
    coarse[:, sizes == 0] = np.inf
    probes = np.argsort(coarse, axis=1, kind="stable")[:, :n_probe]
    scores = (table ** 2).sum(1)[None, :] - 2 * rows @ table.T
    labels = np.empty(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        allowed = np.isin(cell, probes[i])
        winner = int(np.where(allowed, scores[i], np.inf).argmin())
        if child is not None:
            first = child[winner]
            second = child[first] if first >= 0 else -1
            for candidate in (first, second):
                if candidate >= 0 and scores[i, candidate] < scores[i, winner]:
                    winner = int(candidate)
        labels[i] = winner
    scale = (rows ** 2).sum(1) + (table ** 2).sum(1).max()
    return labels, scores, scale, probes


def _device(metal, x, centers, cells, n_probe, *, child=None, precision="float32",
            scan="tiled", tile=None):
    from mlxlearn._kernels.o_means import sorted_mirror, two_level_assign

    k = len(centers)
    dtype = metal.bfloat16 if precision == "bfloat16" else metal.float32
    links = metal.full((k,), -1, dtype=metal.int32) if child is None else metal.array(child.astype(np.int32))
    mirror, norms = sorted_mirror(metal.array(centers), cells, precision=precision)
    labels, best, evaluated = two_level_assign(
        metal.array(x).astype(dtype), mirror, norms, cells, n_probe, links, scan=scan, tile=tile)
    metal.eval(labels, best, evaluated)
    return np.asarray(labels).astype(np.int64), np.asarray(best), float(evaluated.item())


def _cells(metal, centers, n_cells, iterations=4, seed=5):
    from mlxlearn._kernels.o_means import build_cells

    cells = build_cells(metal.array(centers), n_cells, iterations, rng=np.random.RandomState(seed))
    metal.eval(*cells.arrays())
    return cells


@pytest.mark.gpu
@pytest.mark.parametrize("count,bins,dtype", [(1, 1, np.int32), (5, 3, np.uint32), (4099, 37, np.int32)])
def test_histogram_counts_integer_values(metal, count, bins, dtype):
    from mlxlearn._kernels.o_means import histogram

    values = np.random.default_rng(count).integers(0, bins, size=count).astype(dtype)
    counts = histogram(metal.array(values), bins)
    assert counts.dtype == metal.uint32
    np.testing.assert_array_equal(np.asarray(counts), np.bincount(values, minlength=bins))
    np.testing.assert_array_equal(np.asarray(histogram(metal.zeros((0,), metal.int32), 4)), [0, 0, 0, 0])


@pytest.mark.gpu
@pytest.mark.parametrize("k,d,n_cells", [(300, 17, 8), (2304, 96, 32), (65, 3, 65), (9, 2, 1)])
def test_cell_build_matches_double_lloyd_and_sorts_the_table(metal, k, d, n_cells):
    from mlxlearn._kernels.o_means import build_cells

    rng = np.random.default_rng(100 + k)
    _, centers = _mixture(rng, 8, k, d)
    picks = np.random.RandomState(5).permutation(k)[:n_cells]
    expected_centres, expected_cell = _lloyd_cells(centers, centers[picks], 4)
    cells = build_cells(metal.array(centers), n_cells, 4, rng=np.random.RandomState(5))
    np.testing.assert_allclose(np.asarray(cells.centres), expected_centres, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(_cell_of(cells, k), expected_cell)
    order, rank = np.asarray(cells.order), np.asarray(cells.rank)
    # Centroids keep their id order inside a cell.
    np.testing.assert_array_equal(order, np.argsort(expected_cell, kind="stable"))
    np.testing.assert_array_equal(rank[order], np.arange(k))
    sizes = np.bincount(expected_cell, minlength=n_cells)
    np.testing.assert_array_equal(np.asarray(cells.sizes), sizes)
    np.testing.assert_array_equal(np.asarray(cells.offsets), np.concatenate([[0], np.cumsum(sizes)]))
    assert cells.order.dtype == metal.int32 and cells.offsets.dtype == metal.uint32


@pytest.mark.gpu
def test_refile_keeps_cell_centres_and_marks_empty_cells(metal):
    from mlxlearn._kernels.o_means import EMPTY_NORM, build_cells

    rng = np.random.default_rng(31)
    _, centers = _mixture(rng, 8, 200, 12)
    first = _cells(metal, centers, 8)
    moved = centers.copy()
    # Move every centroid of one cell onto another cell's centre.
    target = np.asarray(first.centres)[1]
    members = np.asarray(first.order)[int(first.offsets[0].item()):int(first.offsets[1].item())]
    moved[members] = target + rng.normal(size=(len(members), 12)).astype(np.float32) * 1e-3
    refiled = build_cells(metal.array(moved), 8, 0, previous=first)
    np.testing.assert_array_equal(np.asarray(refiled.centres), np.asarray(first.centres))
    sizes = np.asarray(refiled.sizes)
    assert sizes[0] == 0 and sizes.sum() == 200
    assert float(refiled.norms[0].item()) == np.float32(EMPTY_NORM)
    np.testing.assert_array_equal(_cell_of(refiled, 200), _nearest(moved, np.asarray(first.centres)))
    rebuilt = build_cells(metal.array(moved), 8, 2, previous=refiled)
    expected_centres, expected_cell = _lloyd_cells(moved, np.asarray(first.centres), 2)
    np.testing.assert_allclose(np.asarray(rebuilt.centres), expected_centres, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(_cell_of(rebuilt, 200), expected_cell)


@pytest.mark.gpu
@pytest.mark.parametrize("scan", ["tiled", "scalar"])
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("n,k,d,n_cells,n_probe", [
    (4097, 4096, 96, 32, 8),
    (1500, 2304, 100, 48, 16),
    (777, 1000, 131, 16, 4),
    (257, 300, 33, 16, 1),
    (33, 70, 7, 5, 2),
])
def test_two_level_assignment_matches_brute_force_of_probed_cells(
        metal, scan, precision, n, k, d, n_cells, n_probe):
    rng = np.random.default_rng(n + k)
    x, centers = _mixture(rng, n, k, d)
    cells = _cells(metal, centers, n_cells)
    labels, best, evaluated = _device(metal, x, centers, cells, n_probe, precision=precision, scan=scan)
    represented = (_bf16(x), _bf16(centers)) if precision == "bfloat16" else (x, centers)
    expected, scores, scale, probes = _reference(*represented, cells, n_probe)
    rows = np.arange(n)
    assert labels.min() >= 0 and labels.max() < k
    assert np.mean(labels == expected) >= 0.995
    gap = (scores[rows, labels] - scores[rows, expected]) / scale
    assert np.quantile(np.abs(gap), 0.99) <= 2e-6
    np.testing.assert_allclose(best, scores[rows, labels], rtol=2e-5, atol=2e-5 * scale.max())
    assert evaluated == pytest.approx(n * n_cells + np.asarray(cells.sizes)[probes].sum(), rel=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("rows,cell_count,probes", [
    (1, 1, 1), (5, 3, 1), (5, 3, 3), (4099, 37, 3), (4099, 37, 24), (513, 200, 24), (64, 2, 2),
])
def test_pairs_are_bucketed_by_cell_exactly_once(metal, rows, cell_count, probes):
    from mlxlearn._kernels.o_means import Cells, bucket_pairs

    rng = np.random.default_rng(rows + probes)
    # Skewed popularity: low cells are probed far more often.
    keys = rng.random((rows, cell_count)) * np.arange(1, cell_count + 1)[None, :] ** 2
    host = np.argsort(keys, axis=1)[:, :probes].astype(np.int32)
    sizes = rng.integers(0, 9, cell_count).astype(np.uint32)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.uint32)
    cells = Cells(None, None, None, None, metal.array(offsets), metal.array(sizes))
    pairs, plan = bucket_pairs(metal.array(host), cells, 64)
    pairs, plan = np.asarray(pairs), np.asarray(plan).astype(np.int64)
    assert plan.dtype == np.int64 and len(plan) == 3 * (cell_count + 1)
    pair_offsets, tile_offsets, cell_offsets = np.split(plan, 3)
    counts = np.bincount(host.reshape(-1), minlength=cell_count)
    np.testing.assert_array_equal(np.sort(pairs), np.arange(rows * probes))
    np.testing.assert_array_equal(pair_offsets, np.concatenate([[0], np.cumsum(counts)]))
    np.testing.assert_array_equal(tile_offsets, np.concatenate([[0], np.cumsum(-(-counts // 64))]))
    np.testing.assert_array_equal(cell_offsets, offsets)
    for cell in range(cell_count):
        members = pairs[pair_offsets[cell]:pair_offsets[cell + 1]]
        assert np.all(host.reshape(-1)[members] == cell)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("tile", [(32, 16, 8, 64), (64, 16, 4, 64), (64, 32, 8, 64), (64, 16, 4, 32)])
@pytest.mark.parametrize("n,k,d,n_cells,n_probe", [
    (1000, 3000, 96, 16, 6),     # cells of several 64-centroid blocks
    (331, 700, 45, 40, 9),       # ragged rows, features and cells
    (65, 9, 8, 9, 9),            # one centroid per cell
    (3, 130, 1, 2, 2),           # a single feature
])
def test_tiled_scan_matches_scalar_scan(metal, precision, tile, n, k, d, n_cells, n_probe):
    from mlxlearn._kernels.o_means import probe_cells, scalar_scan, sorted_mirror, tiled_scan

    rng = np.random.default_rng(n * 7 + d)
    x, centers = _mixture(rng, n, k, d)
    cells = _cells(metal, centers, n_cells)
    dtype = metal.bfloat16 if precision == "bfloat16" else metal.float32
    batch = metal.array(x).astype(dtype)
    mirror, norms = sorted_mirror(metal.array(centers), cells, precision=precision)
    probes = probe_cells(batch, cells, n_probe)
    expected_scores, expected_positions = (np.asarray(a) for a in scalar_scan(batch, mirror, norms, probes, cells))
    scores, positions = (np.asarray(a) for a in tiled_scan(batch, mirror, norms, probes, cells, tile=tile))
    assert scores.shape == positions.shape == (n, n_probe)
    scale = (x.astype(np.float64) ** 2).sum(1).max() + (centers.astype(np.float64) ** 2).sum(1).max()
    finite = np.isfinite(expected_scores)
    np.testing.assert_array_equal(np.isfinite(scores), finite)
    np.testing.assert_allclose(scores[finite], expected_scores[finite], rtol=0, atol=4e-6 * scale)
    assert np.mean(positions == expected_positions) >= 0.999
    assert positions.max() < k
    # Every reported position lies inside the probed cell.
    offsets, probed = np.asarray(cells.offsets).astype(np.int64), np.asarray(probes)
    inside = (positions >= offsets[probed]) & (positions < offsets[probed + 1])
    assert np.all(inside | ~finite)


@pytest.mark.gpu
def test_tiled_scan_takes_new_batch_sizes_without_new_kernels(metal, monkeypatch):
    from mlxlearn._kernels import o_means

    rng = np.random.default_rng(41)
    x, centers = _mixture(rng, 300, 500, 24)
    cells = _cells(metal, centers, 8)
    kernel, templates = o_means._tiled_scan_kernel(), []

    def recording(**arguments):
        templates.append(tuple(arguments["template"]))
        return kernel(**arguments)

    monkeypatch.setattr(o_means, "_tiled_scan_kernel", lambda: recording)
    for rows in (300, 299, 65, 1):
        labels, _, _ = _device(metal, x[:rows], centers, cells, 4)
        assert len(labels) == rows
    # Row counts, cell counts and offsets are data, not template values, so a
    # tail batch or a re-filed table compiles nothing new.
    assert len(templates) == 4 and len(set(templates)) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_probing_every_cell_is_a_full_search(metal, precision):
    from mlxlearn._kernels.e_means import assign

    rng = np.random.default_rng(77)
    n, k, d, n_cells = 3001, 2000, 64, 24
    x, centers = _mixture(rng, n, k, d)
    cells = _cells(metal, centers, n_cells)
    labels, _, evaluated = _device(metal, x, centers, cells, n_cells, precision=precision)
    dense, _ = assign(metal.array(x), metal.array(centers), precision=precision)
    dense = np.asarray(dense).astype(np.int64)
    represented = (_bf16(x), _bf16(centers)) if precision == "bfloat16" else (x, centers)
    rows, table = represented[0].astype(np.float64), represented[1].astype(np.float64)
    distance = ((rows[:, None, :] - table[None, :, :]) ** 2).sum(2)
    scale = (rows ** 2).sum(1) + (table ** 2).sum(1).max()
    assert np.mean(labels == dense) >= 0.999
    assert np.max((distance[np.arange(n), labels] - distance.min(1)) / scale) <= 2e-6
    assert evaluated == n * n_cells + n * k


@pytest.mark.gpu
def test_split_children_are_reached_through_links(metal):
    rng = np.random.default_rng(5)
    n, k, d, n_cells, n_probe = 6000, 4096, 96, 32, 4
    x, centers = _mixture(rng, n, k, d)
    cells = _cells(metal, centers, n_cells)
    # Give some centroids a child placed next to them in the data but filed
    # elsewhere in the table, as a split since the last cell build would.
    parent, children = rng.permutation(k)[:300], rng.permutation(k)[:300]
    keep = ~np.isin(parent, children) & ~np.isin(children, parent)
    parent, children = parent[keep], children[keep]
    moved = centers.copy()
    moved[children] = centers[parent] + rng.normal(size=(len(parent), d)).astype(np.float32) * 0.05
    child = np.full(k, -1, dtype=np.int64)
    child[parent] = children
    linked, _, _ = _device(metal, x, moved, cells, n_probe, child=child)
    unlinked, _, _ = _device(metal, x, moved, cells, n_probe)
    expected, scores, scale, _ = _reference(x, moved, cells, n_probe, child=child)
    assert np.mean(linked == expected) >= 0.995
    reached = np.isin(linked, children) & ~np.isin(unlinked, children)
    assert reached.sum() > 0
    rows = np.arange(n)
    # Following a link never selects a worse centroid.
    assert np.all(scores[rows, linked] <= scores[rows, unlinked] + 2e-6 * scale)


@pytest.mark.gpu
def test_grandchild_link_is_followed_but_not_further(metal):
    from mlxlearn._kernels.o_means import Cells

    # One cell holds centroid 0; centroids 1..3 are filed in a cell no row
    # probes and are reachable only through the chain 0 -> 1 -> 2 -> 3.
    centers = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], np.float32)
    x = np.array([[0.1, 0], [1.2, 0], [2.1, 0], [3.2, 0]], np.float32)
    cells = Cells(
        centres=metal.array(np.array([[0, 0], [50, 50]], np.float32)),
        norms=metal.array(np.array([0, 5000], np.float32)),
        order=metal.array(np.arange(4, dtype=np.int32)),
        rank=metal.array(np.arange(4, dtype=np.int32)),
        offsets=metal.array(np.array([0, 1, 4], np.uint32)),
        sizes=metal.array(np.array([1, 3], np.uint32)),
    )
    child = np.array([1, 2, 3, -1])
    labels, _, _ = _device(metal, x, centers, cells, 1, child=child)
    # Row 3 would need a third hop.
    np.testing.assert_array_equal(labels, [0, 1, 2, 2])


@pytest.mark.gpu
def test_empty_cells_never_win(metal):
    from mlxlearn._kernels.o_means import EMPTY_NORM, Cells

    rng = np.random.default_rng(8)
    n, k, d, n_cells, n_probe = 2000, 512, 24, 32, 12
    x, centers = _mixture(rng, n, k, d, modes=4)
    built = _cells(metal, centers, 3)
    # All centroids stay in cells 0-2. The other cells are empty and their
    # centres sit on rows, where they would win the coarse ranking unmarked.
    centres = np.concatenate([np.asarray(built.centres), x[:n_cells - 3]])
    norms = np.concatenate([np.asarray(built.norms), np.full(n_cells - 3, EMPTY_NORM, np.float32)])
    sizes = np.concatenate([np.asarray(built.sizes), np.zeros(n_cells - 3, np.uint32)])
    cells = Cells(metal.array(centres), metal.array(norms), built.order, built.rank,
                  metal.array(np.concatenate([[0], np.cumsum(sizes)]).astype(np.uint32)),
                  metal.array(sizes.astype(np.uint32)))
    labels, best, _ = _device(metal, x, centers, cells, n_probe)
    expected, scores, scale, _ = _reference(x, centers, cells, n_probe)
    assert labels.min() >= 0 and labels.max() < k and np.isfinite(best).all()
    assert np.mean(labels == expected) >= 0.995
    assert np.max(np.abs(scores[np.arange(n), labels] - scores[np.arange(n), expected]) / scale) <= 2e-6
    # A row that probes only empty cells still receives a valid centroid.
    only_empty = Cells(
        metal.array(centres[3:5]), metal.array(norms[3:5]), built.order, built.rank,
        metal.array(np.array([0, 0, 0], np.uint32)), metal.array(np.array([0, 0], np.uint32)))
    far, _, _ = _device(metal, x[:5], centers, only_empty, 2)
    assert far.min() >= 0 and far.max() < k


@pytest.mark.gpu
def test_equal_centroids_and_tiny_inputs(metal):
    rng = np.random.default_rng(12)
    # Duplicated centroids tie exactly; either copy is a correct answer.
    base = rng.normal(size=(6, 4)).astype(np.float32)
    centers = np.concatenate([base, base])
    x = (base[rng.integers(0, 6, 40)] + rng.normal(size=(40, 4)).astype(np.float32) * 0.01)
    cells = _cells(metal, centers, 3)
    labels, _, _ = _device(metal, x, centers, cells, 3)
    np.testing.assert_array_equal(labels % 6, _nearest(x, base))
    # Fewer than eight elements in every kernel input.
    centers = np.array([[0.0], [1.0], [5.0]], np.float32)
    x = np.array([[0.2], [4.0]], np.float32)
    cells = _cells(metal, centers, 2)
    labels, _, _ = _device(metal, x, centers, cells, 2)
    np.testing.assert_array_equal(labels, [0, 2])
    single = _cells(metal, centers[:1], 1)
    labels, _, _ = _device(metal, x, centers[:1], single, 1)
    np.testing.assert_array_equal(labels, [0, 0])


@pytest.mark.gpu
def test_two_level_assignment_rejects_inconsistent_inputs(metal):
    from mlxlearn._kernels.o_means import build_cells, sorted_mirror, two_level_assign

    rng = np.random.default_rng(3)
    x, centers = _mixture(rng, 20, 40, 5)
    cells = _cells(metal, centers, 4)
    mirror, norms = sorted_mirror(metal.array(centers), cells)
    child = metal.full((40,), -1, dtype=metal.int32)
    with pytest.raises(ValueError, match="n_probe"):
        two_level_assign(metal.array(x), mirror, norms, cells, 5, child)
    with pytest.raises(ValueError, match="n_probe"):
        two_level_assign(metal.array(x), mirror, norms, cells, 0, child)
    with pytest.raises(ValueError, match="feature dimension"):
        two_level_assign(metal.array(x[:, :4]), mirror, norms, cells, 2, child)
    with pytest.raises(ValueError, match="scan"):
        two_level_assign(metal.array(x), mirror, norms, cells, 2, child, scan="unknown")
    with pytest.raises(ValueError, match="n_cells"):
        build_cells(metal.array(centers), 41, 1, rng=np.random.RandomState(0))


@pytest.mark.gpu
def test_split_links_follow_the_reference_convention(metal):
    from mlxlearn._kernels.o_means import record_splits
    from mlxlearn.cluster._e_means import split_plan

    # Donor 2 already had child 7. Splits 2 -> 4 and 5 -> 6: the new child 4
    # takes over 2's earlier child, giving the chain 2 -> 4 -> 7.
    child = np.full(10, -1, np.int32)
    child[2] = 7
    is_child, is_donor = np.zeros(10, bool), np.zeros(10, bool)
    is_child[[4, 6]], is_donor[[2, 5]] = True, True
    source, target = np.zeros(10, np.int32), np.zeros(10, np.int32)
    source[[4, 6]], target[[2, 5]] = [2, 5], [4, 6]
    plan = tuple(metal.array(a) for a in (is_child, is_donor, source, target)) + (metal.array(2),)
    np.testing.assert_array_equal(
        np.asarray(record_splits(metal.array(child), plan)), [-1, -1, 4, -1, 7, 6, -1, -1, -1, -1])
    # The plan of a real revival check pairs each dead slot with one donor.
    counts = np.array([16, 0, 8, 0, 4, 0.001], np.float32)
    plan = split_plan(metal.array(counts), metal.array(np.full(6, 0.5, np.float32)), 0.1)
    is_child, is_donor, source, target, number = (np.asarray(a) for a in plan)
    assert int(number) == 3
    np.testing.assert_array_equal(np.flatnonzero(is_child), [1, 3, 5])
    np.testing.assert_array_equal(np.flatnonzero(is_donor), [0, 2, 4])
    np.testing.assert_array_equal(target[source[is_child]], np.flatnonzero(is_child))
    links = np.asarray(record_splits(metal.full((6,), -1, dtype=metal.int32), plan))
    np.testing.assert_array_equal(links[is_donor], target[is_donor])
    np.testing.assert_array_equal(links[is_child], [-1, -1, -1])
