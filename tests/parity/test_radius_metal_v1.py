"""Exact membership of the interim Metal radius route (``radius_metal_v1``).

Every result is compared with a self-contained exact checker: float64 squared
distances from direct differences, with every pair within 1e-12 (relative) of
``R2`` decided by integer arithmetic on the float32 bit patterns.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
import threading
from fractions import Fraction

import numpy as np
import pytest

from mlxlearn._common.device import gpu_available
from mlxlearn._kernels import radius_metal_v1 as metal
from mlxlearn._kernels.radius_contract import thresholds
from mlxlearn._kernels.radius_host import canonical_distances as host_distances

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not gpu_available(), reason="requires a working MLX GPU"),
]

_BAND = 1e-12


# ----------------------------------------------------------------------------------
# exact checker
# ----------------------------------------------------------------------------------


def _scaled(row):
    # float32 values are integer multiples of 2**-149; the product is exact.
    return [int(v) for v in (np.asarray(row, np.float32).astype(np.float64) * 2.0**149).tolist()]


def _exact_le(q, x, R2):
    if np.isinf(R2):
        return True
    total = sum((a - b) ** 2 for a, b in zip(_scaled(q), _scaled(x), strict=True))
    numerator, denominator = Fraction(R2).as_integer_ratio()
    return total * denominator <= numerator << 298


def exact_neighbors(X, Q, R2, *, exclude_self=False, q_offset=0):
    """Per-query ascending member indices of ``S(q, x) <= R2``."""
    X = np.asarray(X, np.float32)
    Q = np.asarray(Q, np.float32)
    n, d = X.shape
    X64 = X.astype(np.float64)
    block = max(1, (1 << 22) // max(1, n * d))
    rows = []
    for start in range(0, len(Q), block):
        diff = Q[start:start + block, None, :].astype(np.float64) - X64[None, :, :]
        s = np.einsum("ijk,ijk->ij", diff, diff)
        for i, values in enumerate(s):
            if R2 == 0:
                member = values == 0
            else:
                member = values <= R2 * (1 - _BAND)
                for j in np.flatnonzero(np.abs(values - R2) <= _BAND * R2):
                    member[j] = _exact_le(Q[start + i], X[j], R2)
            if exclude_self:
                member[start + i + q_offset] = False
            rows.append(np.flatnonzero(member))
    return rows


def assert_exact(result, X, Q, R2, *, exclude_self=False, q_offset=0, need_distance=True):
    expected = exact_neighbors(X, Q, R2, exclude_self=exclude_self, q_offset=q_offset)
    assert result.route == "metal"
    assert result.indptr.dtype == np.int64 and result.indices.dtype == np.int64
    assert len(result.indptr) == len(Q) + 1 and result.indptr[0] == 0
    lengths = np.diff(result.indptr)
    for row, members in enumerate(expected):
        got = result.indices[result.indptr[row]:result.indptr[row + 1]]
        assert np.array_equal(got, members), (
            f"row {row}: missing {np.setdiff1d(members, got)[:8]}, "
            f"extra {np.setdiff1d(got, members)[:8]}")
    assert lengths.sum() == result.indices.size
    if need_distance:
        rows = result.row_ids()
        squared = host_distances(np.asarray(X, np.float32), np.asarray(Q, np.float32),
                                 rows, result.indices)
        np.testing.assert_array_equal(result.distances, np.sqrt(np.minimum(squared, R2)))
        assert np.all(result.distances <= np.sqrt(R2))
    else:
        assert result.distances is None
    return expected


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------


def _index(X):
    import mlx.core as mx

    from mlxlearn_device import DeviceNeighborIndex

    return DeviceNeighborIndex(mx.array(np.asarray(X, np.float32)), device="gpu")


def _query(X, Q, R2, *, index=None, **options):
    X = np.asarray(X, np.float32)
    Q = np.asarray(Q, np.float32)
    thr = thresholds(X.shape[1], R2)
    return metal.metal_query(index if index is not None else _index(X), X, Q, thr, **options)


def _radius_for(X, Q, neighbors, rng):
    """A radius giving about ``neighbors`` members per query (sampled)."""
    sample = Q[rng.choice(len(Q), min(len(Q), 64), replace=False)].astype(np.float64)
    X64 = X.astype(np.float64)
    k = min(neighbors, len(X) - 1)
    kth = []
    for q in sample:
        diff = X64 - q
        kth.append(np.partition(np.einsum("ij,ij->i", diff, diff), k)[k])
    return float(np.sqrt(np.median(kth)))


def _gaussian(rng, n, nq, d, offset=0.0):
    X = (rng.normal(size=(n, d)) + offset).astype(np.float32)
    Q = (rng.normal(size=(nq, d)) + offset).astype(np.float32)
    return X, Q


def _blobs(rng, n, nq, d, centers=12):
    middle = rng.normal(scale=6.0, size=(centers, d))
    X = (middle[rng.integers(0, centers, n)] + rng.normal(size=(n, d))).astype(np.float32)
    Q = (middle[rng.integers(0, centers, nq)] + rng.normal(size=(nq, d))).astype(np.float32)
    return X, Q


def _lattice(rng, n, nq, d):
    """Integer points: a small lattice in three coordinates plus a sparse +-1.

    Squared distances are small integers, so many pairs tie with integral R2.
    """
    def make(rows):
        A = np.zeros((rows, d), np.float32)
        k = min(d, 3)
        A[:, :k] = rng.integers(-3, 4, size=(rows, k))
        if d > 3:
            A[np.arange(rows), rng.integers(3, d, size=rows)] = rng.integers(-1, 2, size=rows)
        return A

    return make(n), make(nq)


DIMENSIONS = [1, 2, 3, 8, 16, 64, 300]


# ----------------------------------------------------------------------------------
# membership
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("kind", ["gaussian", "blobs"])
def test_membership_is_exact(kind, d):
    rng = np.random.default_rng(100 + d)
    n, nq = (3000, 200) if d < 300 else (1200, 60)
    X, Q = (_gaussian if kind == "gaussian" else _blobs)(rng, n, nq, d)
    radius = _radius_for(X, Q, 10, rng)
    result = _query(X, Q, radius * radius, cap_hint=10)
    assert_exact(result, X, Q, radius * radius)
    assert result.nnz > 0


@pytest.mark.parametrize("offset", [1e3, 1e6])
@pytest.mark.parametrize("d", [3, 16, 64])
def test_offset_data_is_screened_tightly(offset, d):
    # PR #3's uncentred screen admitted nearly every pair at these offsets.
    rng = np.random.default_rng(int(offset) % 97 + d)
    X, Q = _gaussian(rng, 4000, 150, d, offset=offset)
    radius = _radius_for(X, Q, 10, rng)
    metal.debug.reset()
    result = _query(X, Q, radius * radius, cap_hint=10)
    expected = assert_exact(result, X, Q, radius * radius)
    members = sum(len(row) for row in expected)
    assert members > 0
    assert metal.debug.candidates <= 2 * members + 2 * len(Q)


@pytest.mark.parametrize("d", DIMENSIONS)
@pytest.mark.parametrize("radius", [1.0, 2.0, 3.0])
def test_integer_lattice_ties_are_exact(d, radius):
    rng = np.random.default_rng(int(radius) * 31 + d)
    X, Q = _lattice(rng, 2500, 120, d)
    R2 = radius * radius
    result = _query(X, Q, R2, cap_hint=50)
    expected = assert_exact(result, X, Q, R2)
    # Many members sit exactly on the boundary.
    assert np.count_nonzero(result.distances == radius) > len(Q)
    assert sum(len(row) for row in expected) == result.nnz


def test_non_integral_squared_radius_on_a_lattice():
    rng = np.random.default_rng(7)
    X, Q = _lattice(rng, 2000, 80, 5)
    radius = np.sqrt(5.0)
    R2 = radius * radius  # 5.000000000000001: pairs at S == 5 are members
    assert R2 != 5.0
    assert_exact(_query(X, Q, R2, cap_hint=100), X, Q, R2)


def test_rows_ascend_and_distances_can_be_omitted():
    rng = np.random.default_rng(3)
    X, Q = _gaussian(rng, 5000, 300, 12)
    radius = _radius_for(X, Q, 25, rng)
    result = _query(X, Q, radius * radius, cap_hint=25, need_distance=False)
    assert_exact(result, X, Q, radius * radius, need_distance=False)
    rows = result.row_ids()
    same_row = rows[1:] == rows[:-1]
    assert np.all(np.diff(result.indices)[same_row] > 0)


def test_empty_results():
    rng = np.random.default_rng(4)
    X, Q = _gaussian(rng, 3000, 100, 8)
    result = _query(X, Q, 1e-6, cap_hint=0)
    assert result.nnz == 0 and np.all(result.indptr == 0)
    assert result.distances.size == 0
    empty = _query(X, Q[:0], 1.0)
    assert empty.n_query == 0 and empty.nnz == 0


def test_zero_radius_keeps_exact_duplicates():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(400, 6)).astype(np.float32)
    X = np.concatenate([base, base[:50], base[:10]])
    Q = np.concatenate([base[:30], rng.normal(size=(20, 6)).astype(np.float32)])
    result = _query(X, Q, 0.0)
    expected = assert_exact(result, X, Q, 0.0)
    assert [len(row) for row in expected[:10]] == [3] * 10
    assert all(len(row) == 0 for row in expected[30:])
    assert np.all(result.distances == 0)


@pytest.mark.parametrize("q_offset", [0, 150])
def test_exclude_self_keeps_duplicates(q_offset):
    rng = np.random.default_rng(6)
    base = rng.normal(size=(600, 10)).astype(np.float32)
    X = np.concatenate([base, base[:100], base[150:250]])
    Q = X[q_offset:q_offset + 300]
    radius = _radius_for(X, Q, 8, rng)
    result = _query(X, Q, radius * radius, exclude_self=True, q_offset=q_offset, cap_hint=8)
    assert_exact(result, X, Q, radius * radius, exclude_self=True, q_offset=q_offset)
    rows = result.row_ids()
    assert not np.any(result.indices == rows + q_offset)
    # Duplicated rows still find their copies at distance zero.
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    duplicated = np.flatnonzero(counts[inverse.ravel()[q_offset:q_offset + 300]] > 1)
    assert duplicated.size >= 100
    for row in duplicated:
        segment = slice(result.indptr[row], result.indptr[row + 1])
        assert np.any(result.distances[segment] == 0)


def test_tiny_shapes_are_padded():
    # Every kernel input has fewer than eight elements before padding.
    X = np.array([[0.0], [1.0], [2.5]], np.float32)
    Q = np.array([[1.2]], np.float32)
    result = _query(X, Q, 1.44, cap_hint=0)
    assert_exact(result, X, Q, 1.44)
    # float32(1.2) is slightly above 1.2, so x = 0 lies just outside.
    assert result.indices.tolist() == [1]
    X5 = np.arange(10, dtype=np.float32).reshape(5, 2)
    assert_exact(_query(X5, X5[:2], 8.0), X5, X5[:2], 8.0)


def test_extreme_magnitudes_stay_exact():
    # Products overflow or norms exceed 2**108: such entries are unsafe and
    # always candidates, so membership stays exact.
    rng = np.random.default_rng(8)
    X = (rng.normal(size=(500, 4)) * 1e19).astype(np.float32)
    Q = X[:40] + np.float32(1e18)
    R2 = 4e37
    assert_exact(_query(X, Q, R2, cap_hint=4), X, Q, R2)


def test_uncertified_dimension_returns_none():
    X = np.zeros((10, 3), np.float32)
    thr = dataclasses.replace(thresholds(3, 1.0), gram_cut=None)
    assert metal.metal_query(_index(X), X, X, thr) is None


def test_training_rows_must_match_the_index():
    X = np.zeros((10, 3), np.float32)
    with pytest.raises(ValueError, match="training rows"):
        metal.metal_query(_index(X), X[:9], X, thresholds(3, 1.0))


def test_repeated_runs_are_bitwise_identical():
    rng = np.random.default_rng(17)
    X, Q = _blobs(rng, 5000, 600, 20, centers=6)
    radius = _radius_for(X, Q, 30, rng)
    index = _index(X)
    runs = [_query(X, Q, radius * radius, index=index, cap_hint=4) for _ in range(5)]
    for other in runs[1:]:
        np.testing.assert_array_equal(other.indptr, runs[0].indptr)
        np.testing.assert_array_equal(other.indices, runs[0].indices)
        np.testing.assert_array_equal(other.distances, runs[0].distances)


def test_the_default_device_does_not_matter():
    # Every operation names the GPU stream, so a CPU default (which another
    # thread may set at any time; MLX's default is process-global) changes
    # nothing, including the lazy preparation of a fresh index.
    import mlx.core as mx

    rng = np.random.default_rng(18)
    X, Q = _gaussian(rng, 3000, 300, 24)
    radius = _radius_for(X, Q, 10, rng)
    reference = _query(X, Q, radius * radius, cap_hint=10)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        result = _query(X, Q, radius * radius, cap_hint=10)
    finally:
        mx.set_default_device(previous)
    np.testing.assert_array_equal(result.indices, reference.indices)
    np.testing.assert_array_equal(result.distances, reference.distances)


def test_kernel_inputs_are_padded_to_eight_elements(monkeypatch):
    # Inputs shorter than eight elements bind as constant memory, which changes
    # the generated source (and before MLX 0.32 rebuilds it unsafely).
    from mlxlearn_device import _radius

    kernel = _radius._compact_kernel()
    sizes = []

    def counting(**kwargs):
        sizes.append([int(array.size) for array in kwargs["inputs"]])
        return kernel(**kwargs)

    monkeypatch.setattr(_radius, "_compact_kernel", lambda: counting)
    X = np.array([[0.0], [1.0], [2.5]], np.float32)
    _query(X, X[:1], 1.0)
    _query(X, X, 4.0, exclude_self=True)
    assert sizes and all(min(call) >= 8 for call in sizes)


# ----------------------------------------------------------------------------------
# overflow, tiling and pipelining
# ----------------------------------------------------------------------------------


def test_minimum_capacity_runs_the_overflow_loop(monkeypatch):
    rng = np.random.default_rng(9)
    X, Q = _blobs(rng, 4000, 300, 16, centers=4)
    radius = _radius_for(X, Q, 40, rng)
    monkeypatch.setattr(metal, "_FORCE_CAPACITY", 1)
    metal.debug.reset()
    result = _query(X, Q, radius * radius, cap_hint=40)
    assert metal.debug.overflow_launches >= 1 and metal.debug.overflow_rows > 0
    assert_exact(result, X, Q, radius * radius)


@pytest.mark.parametrize("skewed", [False, True])
def test_row_sorting_restores_ascending_columns(skewed):
    rng = np.random.default_rng(20)
    rows, capacity = 300, 4096
    counts = rng.poisson(5 if skewed else 300, rows)
    if skewed:
        counts[:4] = [4000, 5000, 1500, 0]  # 5000 overflows the capacity
    packed = np.zeros((rows, capacity), np.uint32)
    for row, count in enumerate(np.minimum(counts, capacity)):
        packed[row, :count] = rng.choice(10**6, count, replace=False)
    lengths, fits, cols = metal._sorted_rows(counts.astype(np.int64), packed, capacity)
    np.testing.assert_array_equal(fits, counts <= capacity)
    np.testing.assert_array_equal(lengths, np.where(fits, counts, 0))
    expected = [np.sort(packed[row, :count]) for row, count in enumerate(counts) if fits[row]]
    np.testing.assert_array_equal(cols, np.concatenate(expected))


def test_dense_hub_rows_are_exact(monkeypatch):
    # A tight cluster gives some queries thousands of members and the rest a
    # handful: rows overflow, and long rows sort apart from short ones. (The
    # generous small-batch capacity is switched off to exercise the overflow.)
    monkeypatch.setattr(metal, "_GENEROUS_SLOTS", 0)
    rng = np.random.default_rng(21)
    spread = rng.normal(scale=4.0, size=(6000, 12))
    hub = rng.normal(scale=0.05, size=(2500, 12))
    X = np.concatenate([spread, hub]).astype(np.float32)
    Q = np.concatenate([rng.normal(scale=4.0, size=(150, 12)),
                        rng.normal(scale=0.05, size=(30, 12))]).astype(np.float32)
    R2 = 0.25
    metal.debug.reset()
    result = _query(X, Q, R2, cap_hint=20)
    assert metal.debug.overflow_rows >= 30
    assert result.lengths().max() > 1000
    assert_exact(result, X, Q, R2)


def test_capacity_follows_the_hint():
    assert metal._capacity(0, 10**6) == 256
    assert metal._capacity(10, 10**6) == 256
    assert metal._capacity(100, 10**6) == 512
    assert metal._capacity(5000, 10**6) == 16384
    assert metal._capacity(1e9, 10**6) == 65536
    assert metal._capacity(10, 100) == 100
    assert metal._capacity(None, 3) == 3


def test_query_block_rows_follow_the_tile_budget():
    assert metal._plan(200_000, 2000)[:2] == (200_000, 320)
    assert metal._plan(100_000, 10_000)[:2] == (100_000, 640)
    assert metal._plan(1000, 10_000)[:2] == (1000, 4096)
    assert metal._plan(3_000_000, 100)[:2] == (1 << 20, 64)


@pytest.mark.parametrize("width", [1000, 1024])
def test_column_tiles_merge_in_column_order(monkeypatch, width):
    rng = np.random.default_rng(10)
    X, Q = _blobs(rng, 4500, 260, 20, centers=5)
    radius = _radius_for(X, Q, 30, rng)
    monkeypatch.setattr(metal, "_COLUMN_TILE", width)
    monkeypatch.setattr(metal, "_TILE_ELEMENTS", 64 * 4500)
    metal.debug.reset()
    result = _query(X, Q, radius * radius, cap_hint=30, exclude_self=False)
    assert metal.debug.tiles == 5 * 5 and metal.debug.waves > 2
    assert_exact(result, X, Q, radius * radius)


def test_column_tiles_with_overflow_and_self_exclusion(monkeypatch):
    rng = np.random.default_rng(11)
    X, _ = _blobs(rng, 3000, 1, 9, centers=3)
    X = np.concatenate([X, X[:200]])
    radius = _radius_for(X, X, 20, rng)
    monkeypatch.setattr(metal, "_COLUMN_TILE", 700)
    monkeypatch.setattr(metal, "_TILE_ELEMENTS", 64 * 3200)
    monkeypatch.setattr(metal, "_FORCE_CAPACITY", 3)
    metal.debug.reset()
    Q = X[1000:1700]
    result = _query(X, Q, radius * radius, exclude_self=True, q_offset=1000)
    assert metal.debug.overflow_launches > 0
    assert_exact(result, X, Q, radius * radius, exclude_self=True, q_offset=1000)


def test_results_do_not_depend_on_tiling(monkeypatch):
    rng = np.random.default_rng(12)
    X, Q = _gaussian(rng, 6000, 700, 24)
    radius = _radius_for(X, Q, 15, rng)
    index = _index(X)
    reference = _query(X, Q, radius * radius, index=index, cap_hint=15)
    monkeypatch.setattr(metal, "_TILE_ELEMENTS", 64 * 6000)
    monkeypatch.setattr(metal, "_WAVE_ELEMENTS", 2 * 64 * 6000)
    monkeypatch.setattr(metal, "_COLUMN_TILE", 2500)
    tiled = _query(X, Q, radius * radius, index=index, cap_hint=0)
    np.testing.assert_array_equal(tiled.indptr, reference.indptr)
    np.testing.assert_array_equal(tiled.indices, reference.indices)
    np.testing.assert_array_equal(tiled.distances, reference.distances)


def test_binary32_kernel_route_is_exact_on_sliced_tiles(monkeypatch):
    # GPUs that may round float32 GEMM to TF32 (M5) take mlxlearn's binary32
    # kernel. Force it here, with query blocks and column tiles that are
    # offset views of the centred arrays.
    from mlxlearn_device import _binary32

    calls = []
    kernel_product = _binary32.float32_matmul

    def counted(*args, **kwargs):
        calls.append(1)
        return kernel_product(*args, **kwargs)

    monkeypatch.setattr(_binary32, "matmul_may_round_float32", lambda: True)
    monkeypatch.setattr(_binary32, "float32_matmul", counted)
    monkeypatch.setattr(metal, "_COLUMN_TILE", 1500)
    monkeypatch.setattr(metal, "_TILE_ELEMENTS", 64 * 4000)
    rng = np.random.default_rng(22)
    X, Q = _blobs(rng, 4000, 300, 40, centers=6)
    radius = _radius_for(X, Q, 20, rng)
    metal.debug.reset()
    result = _query(X, Q, radius * radius, cap_hint=20)
    assert metal.debug.tiles == 5 * 3 and len(calls) >= 15
    assert_exact(result, X, Q, radius * radius)


def test_concurrent_threads_give_identical_results():
    import mlx.core as mx

    version = tuple(int(part) for part in mx.__version__.split(".")[:2])
    if version < (0, 32):
        pytest.skip("concurrent GPU evaluation needs MLX 0.32")
    rng = np.random.default_rng(13)
    X, Q = _gaussian(rng, 8000, 900, 32)
    index = _index(X)
    radii = [_radius_for(X, Q, k, rng) for k in (5, 10, 20)]
    expected = [_query(X, Q, r * r, index=index, cap_hint=10) for r in radii]
    results, errors = {}, []

    def worker(slot):
        try:
            for repeat in range(3):
                results[slot, repeat] = _query(X, Q, radii[slot] ** 2, index=index, cap_hint=10)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    stop = threading.Event()

    def cpu_default_toggler():
        # Another thread switching MLX's process-global default to the CPU.
        ones = mx.ones((1000,), mx.float32, stream=mx.cpu)
        while not stop.is_set():
            with mx.stream(mx.cpu):
                mx.eval(mx.sum(ones * 2))

    toggler = threading.Thread(target=cpu_default_toggler)
    toggler.start()
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    toggler.join()
    assert not errors, errors
    for (slot, _), result in results.items():
        np.testing.assert_array_equal(result.indptr, expected[slot].indptr)
        np.testing.assert_array_equal(result.indices, expected[slot].indices)
        np.testing.assert_array_equal(result.distances, expected[slot].distances)


def test_threads_are_serialised_before_mlx_032():
    # Below MLX 0.32 concurrent GPU evaluation can abort the process; the
    # module lock serialises whole calls. Run in a subprocess with a watchdog.
    program = textwrap.dedent("""
        import faulthandler, threading
        import numpy as np, mlx.core as mx
        faulthandler.dump_traceback_later(120, exit=True)
        from mlxlearn._kernels import radius_metal_v1 as metal
        from mlxlearn._kernels.radius_contract import thresholds
        from mlxlearn_device import DeviceNeighborIndex
        version = tuple(int(p) for p in mx.__version__.split(".")[:2])
        assert metal._serialize_gpu() == (version < (0, 32))
        rng = np.random.default_rng(13)
        X = rng.normal(size=(6000, 32)).astype(np.float32)
        Q = rng.normal(size=(700, 32)).astype(np.float32)
        index = DeviceNeighborIndex(mx.array(X), device="gpu")
        thr = thresholds(32, 30.0)
        reference = metal.metal_query(index, X, Q, thr, cap_hint=10)
        results, errors = [], []
        def worker():
            try:
                for _ in range(3):
                    results.append(metal.metal_query(index, X, Q, thr, cap_hint=10))
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, errors
        assert len(results) == 9
        for r in results:
            assert np.array_equal(r.indptr, reference.indptr)
            assert np.array_equal(r.indices, reference.indices)
            assert np.array_equal(r.distances, reference.distances)
        print("ok")
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            timeout=180)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout + result.stderr


def test_binary32_products_follow_the_named_stream(monkeypatch):
    # Both MLX's own GEMM and the binary32 kernel route (forced here, as on M5)
    # run on the stream named by the caller, whatever the default device is.
    import mlx.core as mx

    from mlxlearn_device import _binary32

    rng = np.random.default_rng(19)
    a = rng.normal(size=(70, 33)).astype(np.float32)
    b = rng.normal(size=(65, 33)).astype(np.float32)
    # einsum, not BLAS: NumPy 2.2's Accelerate GEMM raises spurious FP flags.
    exact = np.einsum("ik,jk->ij", a.astype(np.float64), b.astype(np.float64))
    scale = np.einsum("ik,jk->ij", np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    for may_round in (False, True):
        monkeypatch.setattr(_binary32, "matmul_may_round_float32",
                            lambda may_round=may_round: may_round)
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            out = _binary32.matmul(mx.array(a), mx.array(b), transpose_b=True, stream=mx.gpu)
            out = np.array(out, dtype=np.float64)
        finally:
            mx.set_default_device(previous)
        assert np.all(np.abs(out - exact) <= 33 * 2.0**-23 * scale + 2.0**-126)


def test_peak_memory_holds_two_waves_of_products(monkeypatch):
    import mlx.core as mx

    # The generous small-batch capacity adds up to 16 MiB of packed slots
    # per tile; this test pins the product-tile bound without it.
    monkeypatch.setattr(metal, "_GENEROUS_SLOTS", 0)

    rng = np.random.default_rng(14)
    n, nq, d = 30_000, 4000, 32
    X, Q = _gaussian(rng, n, nq, d)
    radius = _radius_for(X, Q, 10, rng)
    index = _index(X)
    tile_elements = 1 << 22
    monkeypatch.setattr(metal, "_TILE_ELEMENTS", tile_elements)
    monkeypatch.setattr(metal, "_WAVE_ELEMENTS", tile_elements)
    _query(X, Q[:64], radius * radius, index=index, cap_hint=10)  # prepare the index
    mx.eval(index._train, index._train_sq)
    metal.debug.reset()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    result = _query(X, Q, radius * radius, index=index, cap_hint=10)
    peak = mx.get_peak_memory() - base
    assert metal.debug.waves >= 16
    tile_bytes = 128 * n * 4  # 128 query rows of float32 products
    small = 4 * nq * d * 4 + 8 * 1024 * 1024  # queries, centred copy, norms, packed
    assert peak <= 2 * tile_bytes + small, (peak, tile_bytes)
    assert result.nnz > 0


# ----------------------------------------------------------------------------------
# the compaction kernel
# ----------------------------------------------------------------------------------


def _tile(X, Q):
    import mlx.core as mx

    from mlxlearn_device._binary32 import matmul

    index = _index(X)
    with mx.stream(mx.gpu):
        index._prepare_gram()
    centred = mx.subtract(mx.array(Q), index._center, stream=mx.gpu)
    norms = mx.sum(centred * centred, axis=1, stream=mx.gpu)
    products = matmul(centred, index._train, transpose_b=True, stream=mx.gpu)
    return index, centred, norms, products


@pytest.mark.parametrize("d,offset", [(16, 0.0), (64, 1e3), (8, 1e6), (300, 0.0)])
def test_kernel_evaluates_the_knn_gram_bound(d, offset):
    # The screen must be exactly the certified kNN lower bound, in float32.
    import mlx.core as mx

    from mlxlearn_device._neighbors_general import _gram_lower_from_products
    from mlxlearn_device._radius import compact_gram

    rng = np.random.default_rng(15 + d)
    X, Q = _gaussian(rng, 3000, 100, d, offset=offset)
    R2 = _radius_for(X, Q, 30, rng) ** 2
    thr = thresholds(d, R2)
    index, centred, norms, products = _tile(X, Q)
    with mx.stream(mx.gpu):
        lower = _gram_lower_from_products(centred, products, index._train_sq,
                                          thr.gram_factor, thr.gram_floor)
        expected = np.array(lower <= np.float32(thr.gram_cut))
    counts, packed = compact_gram(products, norms, index._train_sq, capacity=len(X),
                                  cut=thr.gram_cut, factor=thr.gram_factor,
                                  floor=thr.gram_floor)
    counts, packed = np.array(counts), np.array(packed)
    found = np.zeros_like(expected)
    for row in range(len(Q)):
        found[row, packed[row, :counts[row]]] = True
    np.testing.assert_array_equal(found, expected)
    np.testing.assert_array_equal(counts, expected.sum(axis=1))


def test_kernel_counts_past_capacity():
    from mlxlearn_device._radius import compact_gram

    rng = np.random.default_rng(16)
    X, Q = _blobs(rng, 2000, 40, 6, centers=2)
    thr = thresholds(6, 50.0)
    index, _, norms, products = _tile(X, Q)
    full_counts, full = compact_gram(products, norms, index._train_sq, capacity=len(X),
                                     cut=thr.gram_cut, factor=thr.gram_factor,
                                     floor=thr.gram_floor)
    counts, packed = compact_gram(products, norms, index._train_sq, capacity=3,
                                  cut=thr.gram_cut, factor=thr.gram_factor,
                                  floor=thr.gram_floor, col_offset=0)
    full_counts, full = np.array(full_counts), np.array(full)
    counts, packed = np.array(counts), np.array(packed)
    np.testing.assert_array_equal(counts, full_counts)
    assert counts.max() > 3
    for row in range(len(Q)):
        stored = packed[row, :min(3, counts[row])]
        assert np.unique(stored).size == stored.size
        assert np.isin(stored, full[row, :full_counts[row]]).all()
