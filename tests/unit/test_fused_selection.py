"""The fused first-pass selection kernel (``_kernels/_topk.py``) and its gate (``_metal.py``).

The kernel may change nothing but time. On every case here the neighbor search answers
with the same indices, the same float64 distance bits and the same diagnostics counts
with it as with MLX's built-in selection — ``Tuning.metal_kernels`` off, which is the
search as it was before the kernel — on hard data that sends rows to the second pass
and the exact search, at every bucket and bucket boundary, at row counts on and around
the block boundaries, and on both devices. The kernel itself is checked against a host
oracle, the gate against every way it closes.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import Tuning, _tuning_from_env, tuning_context
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._kernels import _metal, _topk, distance
from mlxlearn._kernels.distance import build_index
from mlxlearn.neighbors import NearestNeighbors
from tests._helpers import join_or_fail

MiB = 1024 * 1024

#: Whether the kernel can run here: the GPU, an MLX release it was validated on, Metal.
FUSED = (
    gpu_available()
    and _metal.mlx_version()[:2] >= _metal.MIN_MLX
    and _metal.mlx_version() in _topk.VALIDATED_MLX
)


def needs_kernel(test):
    """A test that runs the kernel: the GPU, and an MLX release the kernel serves."""
    skip = pytest.mark.skipif(
        not FUSED, reason="the fused kernel needs the GPU and an MLX release it was validated on"
    )
    return pytest.mark.gpu(skip(test))


@pytest.fixture
def launches(monkeypatch):
    """Counts of the kernel's launches, by mode, through a spy on :class:`Selection`."""
    counts = {"scan": 0, "merge": 0}
    for mode in counts:
        real = getattr(_topk.Selection, mode)

        def spy(self, *args, _real=real, _mode=mode, **kwargs):
            counts[_mode] += 1
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(_topk.Selection, mode, spy)
    return counts


@pytest.fixture
def any_size(monkeypatch):
    """Let the kernel take a first pass of any size, so small data reaches it."""
    monkeypatch.setattr(distance, "_FUSED_MIN_PAIRS", 0)
    monkeypatch.setattr(distance, "_FUSED_MIN_TRAIN_ROWS", 0)


@pytest.fixture
def fresh_first_use():
    """Forget which kernels passed their first-use check, before and after the test."""
    _topk._checked.cache_clear()
    yield
    _topk._checked.cache_clear()


def _bits_equal(got, want):
    got, want = np.ascontiguousarray(got), np.ascontiguousarray(want)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))


def _same_as_builtin(index, Q, k, launches, *, fused=True, **kwargs):
    """Query with the kernel and without it; assert bitwise-equal answers and counts."""
    before = launches["scan"]
    on_stats, off_stats = {}, {}
    on = index.query(Q, k, stats=on_stats, **kwargs)
    ran = launches["scan"] - before
    with tuning_context(metal_kernels=False):
        off = index.query(Q, k, stats=off_stats, **kwargs)
    assert launches["scan"] - before == ran, "the switch did not keep the kernel off"
    _bits_equal(on[1], off[1])
    if on[0] is not None:
        _bits_equal(on[0], off[0])
    assert on_stats == off_stats
    assert (ran > 0) == fused, f"the kernel ran {ran} times"
    return on_stats


def _data(kind: str, n: int, d: int, rng) -> np.ndarray:
    if kind == "gauss":
        return rng.standard_normal((n, d))
    if kind == "dup20":  # every row about 20 times: ties at the k-th distance
        base = rng.standard_normal((n // 20, d))
        return base[rng.integers(0, n // 20, n)]
    if kind == "blobs":  # tight clusters: the Gram noise outgrows the gaps
        centers = np.random.default_rng(99).normal(scale=3.0, size=(100, d))
        return centers[rng.integers(0, 100, n)] + 0.01 * rng.standard_normal((n, d))
    if kind == "far":  # two clusters 1000 apart: the exact search, the count's skip
        X = rng.standard_normal((n, d)) * 0.01
        X[n // 2 :] += 1000.0
        return X
    if kind == "raw":  # one column in raw units, 0-1e4
        X = rng.standard_normal((n, d))
        X[:, 0] = rng.uniform(0.0, 1e4, n)
        return X
    if kind == "unscaled":  # one column on 100 times the scale of the rest
        X = rng.standard_normal((n, d))
        X[:, 0] *= 100.0
        return X
    if kind == "float32":
        return rng.standard_normal((n, d)).astype(np.float32)
    if kind == "offset":  # float64 far from the origin: a float32 residual
        return rng.standard_normal((n, d)) + 1e4
    if kind == "huge":  # squares overflow float32: the span is scaled down
        return rng.standard_normal((n, d)) * 1e19
    if kind == "tiny":  # squares underflow float32: the span is scaled up
        return rng.standard_normal((n, d)) * 1e-20
    raise ValueError(kind)


def _queries(X, q, rng):
    """Half beside training rows, half fresh: the beside ones trip the certificate on hard data."""
    near = q // 2
    scale = float(np.std(X.astype(np.float64))) or 1.0
    A = X[rng.integers(0, len(X), near)].astype(np.float64)
    A = A + 1e-3 * scale * rng.standard_normal(A.shape)
    B = rng.standard_normal((q - near, X.shape[1])) * scale + float(np.mean(X))
    Q = np.vstack([A, B])
    return Q.astype(X.dtype)


# ----------------------------------------------------------------------------------
# the switch and the gate
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, value", [("0", False), ("off", False), ("1", True), ("TRUE", True)])
def test_the_switch_reads_the_environment(monkeypatch, raw, value):
    monkeypatch.setenv("MLXLEARN_METAL_KERNELS", raw)
    assert _tuning_from_env(Tuning()).metal_kernels is value


def test_a_malformed_switch_is_an_error(monkeypatch):
    monkeypatch.setenv("MLXLEARN_METAL_KERNELS", "sometimes")
    with pytest.raises(ValueError, match="MLXLEARN_METAL_KERNELS"):
        _tuning_from_env(Tuning())


@pytest.mark.parametrize("value", [1, "yes", None])
def test_the_switch_is_a_bool(value):
    with pytest.raises(TypeError, match="metal_kernels must be a bool"):
        Tuning(metal_kernels=value)


def test_the_switch_is_on_by_default():
    assert Tuning().metal_kernels is True
    assert Tuning(metal_kernels=np.bool_(False)).metal_kernels is False


def test_no_kernel_runs_for_the_cpu_device():
    assert not _metal.kernels_enabled("cpu")
    assert _topk.selection("cpu", 14) is None


@pytest.mark.parametrize(
    "version", [(0, 31, 2), (0, 32, 0), (0, 32, 1), (0, 32, 3), (0, 33, 0), (1, 0, 0)]
)
def test_mlx_releases_the_kernel_was_not_validated_on_keep_the_builtin_path(version):
    """0.31 for its unlocked library map; any other release, a patch release of 0.32
    included, until it has been checked: which GEMM kernel MLX picks can change in one."""
    real = _metal.mlx_version
    _metal.mlx_version = lambda: version
    _metal._runtime_ready.cache_clear()
    try:
        assert _topk.selection("gpu", 14) is None
        if version[:2] < _metal.MIN_MLX:
            assert not _metal.kernels_enabled("gpu")
    finally:
        _metal.mlx_version = real
        _metal._runtime_ready.cache_clear()


@pytest.mark.parametrize("text, version", [
    ("0.32.2", (0, 32, 2)), ("0.32.2.dev20261001+g1234", (0, 32, 2)), ("0.33", (0, 33, 0)),
    ("1.0.0rc1", (1, 0, 0)),
])
def test_the_mlx_version_is_read_from_its_leading_digits(monkeypatch, text, version):
    monkeypatch.setattr(mx, "__version__", text)
    _metal.mlx_version.cache_clear()
    try:
        assert _metal.mlx_version() == version
    finally:
        monkeypatch.undo()
        _metal.mlx_version.cache_clear()


def test_buckets():
    assert [_topk.bucket(k) for k in (1, 16, 17, 32, 33, 64, 65)] == [16, 16, 32, 32, 64, 64, None]


def test_the_kernel_is_built_under_the_metal_rules(monkeypatch):
    """contract(off) first, ASCII, safe math, row-contiguous inputs, stable prefixed names."""
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: built.append(kwargs) or kwargs)
    for length in (16, 32):
        _topk._kernel.__wrapped__(length)
        _topk._kernel.__wrapped__(length)
    names = [b["name"] for b in built]
    assert names[0] == names[1] and names[2] == names[3] and names[0] != names[2]
    for kwargs in built:
        assert kwargs["name"].startswith(_metal.NAME_PREFIX)
        assert kwargs["header"].startswith("#pragma METAL fp contract(off)\n")
        assert kwargs["compile_options"] == {"math_mode": "safe"}
        assert kwargs["ensure_row_contiguous"] is True
        source = kwargs["header"] + kwargs["source"]
        source.encode("ascii")
        # Fast-math functions would round differently from MLX's own operations.
        for fn in ("exp(", "sqrt(", "rsqrt(", "log(", "fast::", "fma("):
            assert fn not in source


def test_inputs_shorter_than_eight_elements_are_padded():
    a = mx.array([1.0, 2.0, 3.0])
    padded = _metal.pad_min8(mx, a, np.inf)
    assert padded.shape == (8,) and np.isinf(np.asarray(padded)[3:]).all()
    assert _metal.pad_min8(mx, mx.zeros(9), 0.0).shape == (9,)
    params = _metal.int_params(mx, 0, 3, 7)
    assert params.dtype == mx.int32 and np.asarray(params).tolist() == [0, 3, 7, 0, 0, 0, 0, 0]


# ----------------------------------------------------------------------------------
# the kernel, against a host oracle
# ----------------------------------------------------------------------------------


def _host_tile(G, q_sq, t_sq):
    x = (q_sq.reshape(-1, 1) + t_sq.reshape(1, -1)) - np.float32(2.0) * G
    return np.maximum(x, np.float32(0.0)).astype(np.float32)


def _host_first(T, k):
    cols = np.broadcast_to(np.arange(T.shape[1]), T.shape)
    order = np.lexsort((cols, T), axis=1)[:, :k]
    return np.take_along_axis(T, order, axis=1), order


def _selection(k_carry):
    with use_device("gpu"):
        select = _topk.selection("gpu", k_carry)
    assert select is not None, "the kernel did not pass its first-use check"
    return select


@pytest.mark.parametrize(
    "rows, width, k_carry, ties",
    [
        (1, 1000, 5, 0),
        (3, 7001, 14, 5),  # a ragged width: single loads
        (8, 65536, 16, 3),
        (2, 130, 16, 2),  # one chunk narrower than the list: inf padding
        (5, 4099, 17, 0),
        (130, 20000, 32, 0),
        (6, 9000, 33, 4),
        (64, 30000, 54, 7),
        (7, 1024, 64, 4),
        (1, 70, 64, 0),
    ],
)
@needs_kernel
def test_the_kernel_keeps_the_first_k_by_value_then_column(rows, width, k_carry, ties, rng):
    if ties:
        G = rng.integers(-ties, ties + 1, size=(rows, width)).astype(np.float32)
        t_sq = rng.integers(0, ties, size=width).astype(np.float32)
        q_sq = rng.integers(0, ties, size=(rows, 1)).astype(np.float32)
    else:
        G = rng.standard_normal((rows, width)).astype(np.float32)
        t_sq = (1.0 + rng.random(width)).astype(np.float32)
        q_sq = (1.0 + rng.random((rows, 1))).astype(np.float32)
    T = _host_tile(G, q_sq, t_sq)
    want_vals, want_cols = _host_first(T, k_carry)
    select = _selection(k_carry)
    with use_device("gpu"):
        g, q, t = mx.array(G), mx.array(q_sq), mx.array(t_sq)
        vals, cols, flags = select.smallest(mx, g, q, t, col0=0, t_offset=0)
        # MLX's own selection over the same tile, as the search runs it.
        tile = distance._kernels("gpu").tile(q, t[None, :], g)
        b_cols, b_vals = distance._select_smallest(mx, tile, k_carry, segmented=True)
        mx.eval(vals, cols, tile, b_cols, b_vals)
    np.testing.assert_array_equal(np.asarray(tile).view(np.uint32), T.view(np.uint32))
    assert flags is None or not np.asarray(flags).any()
    np.testing.assert_array_equal(np.asarray(cols), want_cols)
    np.testing.assert_array_equal(np.asarray(vals).view(np.uint32), want_vals.view(np.uint32))
    # MLX 0.32's GPU argpartition keeps equal values in column order: the same list.
    np.testing.assert_array_equal(np.asarray(b_cols), want_cols)


@pytest.mark.parametrize("train_cols", [7001, 16384, 50003])
@pytest.mark.parametrize("offset", [0, 3])
@needs_kernel
def test_the_kernel_carries_its_list_across_train_tiles(train_cols, offset, rng):
    """Tile after tile, from unaligned slices of the column norms, as the search runs it."""
    rows, n, k_carry = 9, 50003, 14
    G = rng.integers(-6, 7, size=(rows, n)).astype(np.float32)
    t_sq = np.concatenate([np.zeros(offset), rng.integers(0, 6, size=n)]).astype(np.float32)
    q_sq = rng.integers(0, 6, size=(rows, 1)).astype(np.float32)
    want_vals, want_cols = _host_first(_host_tile(G, q_sq, t_sq[offset:]), k_carry)
    select = _selection(k_carry)
    with use_device("gpu"):
        t_all = mx.array(t_sq)
        best = None
        for t0 in range(0, n, train_cols):
            t1 = min(t0 + train_cols, n)
            vals, cols, _ = select.smallest(
                mx, mx.array(np.ascontiguousarray(G[:, t0:t1])), mx.array(q_sq),
                t_all[offset + t0 : offset + t1], col0=t0, t_offset=offset + t0, prev=best,
            )
            best = (vals, cols)
        mx.eval(*best)
    np.testing.assert_array_equal(np.asarray(best[1]), want_cols)
    np.testing.assert_array_equal(np.asarray(best[0]).view(np.uint32), want_vals.view(np.uint32))


@needs_kernel
def test_a_list_that_can_overflow_is_flagged(monkeypatch):
    """The 64 bucket keeps lists of 32: more than 32 of a row's first 64 in one lane is
    flagged, never returned. One chunk, and every winning column read by lane 0."""
    select = _selection(54)
    monkeypatch.setattr(_topk, "_CHUNK_MIN", 65536)
    rows, width = 2, 65536
    G = np.full((rows, width), -1.0, dtype=np.float32)
    G[:, ::128] = 10.0
    with use_device("gpu"):
        _, _, flags = select.smallest(
            mx, mx.array(G), mx.array(np.full((rows, 1), 30.0, np.float32)),
            mx.array(np.full(width, 30.0, np.float32)), col0=0, t_offset=0,
        )
        mx.eval(flags)
    assert np.asarray(flags).all()
    assert not _selection(32).may_overflow and select.may_overflow


# ----------------------------------------------------------------------------------
# the first pass: the built-in candidates, bit for bit
# ----------------------------------------------------------------------------------


def _search(index, k, *, fused, device="gpu"):
    k_carry = distance._first_pass_carry(k, index.n_features, index.n_samples)
    return distance._Search(
        index=index,
        state=index._ensure_device(device),
        mx=mx,
        k_search=k,
        k_carry=k_carry,
        segmented=True,
        fns=distance._kernels(device),
        plan_backend="mlx",
        train_block_size=None,
        counts={"rows": 0, "widened_rows": 0, "widen_skipped_rows": 0, "exact_rows": 0},
        fused=_selection(k_carry) if fused else None,
        group_cols=index.n_samples,  # where a flagged row is selected again
    )


@pytest.mark.parametrize("kind", ["gauss", "dup20", "blobs", "unscaled", "offset", "huge"])
@pytest.mark.parametrize("k", [1, 12, 13, 28, 29, 60])  # k_carry 5, 16, 17, 32, 33, 64
@needs_kernel
def test_the_fused_first_pass_keeps_the_builtin_candidates(kind, k, rng):
    """The set, its order and the largest kept Gram value: what the certificate reads."""
    X = _data(kind, 12_000, 16, rng)
    Q = _queries(X, 96, rng)
    index = build_index(X, backend="mlx")
    builtin, fused = _search(index, k, fused=False), _search(index, k, fused=True)
    with use_device("gpu"):
        qb = builtin.query_block(Q, 0, len(Q), self_query=False)
        b_idx, b_gmax = builtin.candidates(qb, builtin.k_carry, index.n_samples)
        f_idx, f_gmax = fused.first_pass(qb, 5000)  # three train tiles, the last ragged
        mx.eval(b_idx, b_gmax, f_idx, f_gmax)
    np.testing.assert_array_equal(np.asarray(f_idx), np.asarray(b_idx))
    _bits_equal(np.asarray(f_gmax), np.asarray(b_gmax))


def _lane_zero_neighbors(rng, n=131072, d=16):
    """Training rows whose near ones are every 128th: one lane of one chunk reads them all."""
    X = rng.standard_normal((n, d)) * 10
    X[::128] = rng.standard_normal((n // 128, d)) * 0.01
    return X, rng.standard_normal((5, d)) * 0.01


@pytest.mark.parametrize("k", [50, 60])
@needs_kernel
def test_an_overflowed_row_is_searched_again_by_the_builtin_selection(
    k, rng, monkeypatch, launches, any_size
):
    X, Q = _lane_zero_neighbors(rng)
    index = build_index(X, backend="mlx")
    _selection(64)  # its first-use check, at the real chunk widths
    monkeypatch.setattr(_topk, "_CHUNK_MIN", 131072)
    redone = []
    real = distance._Search.candidates

    def candidates(self, qb, k_carry, train_cols):
        if self.fused is not None and k_carry == self.k_carry:
            redone.append(qb.n)
        return real(self, qb, k_carry, train_cols)

    monkeypatch.setattr(distance._Search, "candidates", candidates)
    _same_as_builtin(index, Q, k, launches)
    assert redone, "no row overflowed: the test lost its premise"


@pytest.mark.parametrize("d", [16, 256])
@pytest.mark.usefixtures("any_size")
@needs_kernel
def test_one_overflowed_row_of_a_block_is_redone_beside_another(d, rng, monkeypatch, launches):
    """Alone it would be a one-row product, which rounds differently; a partner keeps the
    bits, and so do the built-in path's tiles, at 256 features too."""
    X = _data("blobs", 20_000, d, rng)
    Q = _queries(X, 40, rng)
    index = build_index(X, backend="mlx")
    _selection(14)  # its first-use check, before the flags are forged
    real_smallest = _topk.Selection.smallest

    def flag_row_one(self, mx_, *args, **kwargs):
        vals, cols, _ = real_smallest(self, mx_, *args, **kwargs)
        flags = np.zeros(int(vals.shape[0]), dtype=np.int32)
        flags[1] = 1
        return vals, cols, mx_.array(flags)

    redone = []
    real = distance._Search.candidates

    def candidates(self, qb, k_carry, train_cols):
        if self.fused is not None and k_carry == self.k_carry:
            redone.append(qb.n)
        return real(self, qb, k_carry, train_cols)

    monkeypatch.setattr(_topk.Selection, "smallest", flag_row_one)
    monkeypatch.setattr(distance._Search, "candidates", candidates)
    _same_as_builtin(index, Q, 10, launches)
    assert redone == [2]


# ----------------------------------------------------------------------------------
# the query: same answers, same counts
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gauss", "dup20", "blobs", "unscaled", "float32", "offset",
                                  "huge", "tiny"])
@pytest.mark.parametrize("k", [1, 12, 13, 28, 29, 60, 61])  # k_carry 5 ... 64, then 65
@pytest.mark.usefixtures("any_size")
@needs_kernel
def test_fused_queries_answer_as_the_builtin_ones(kind, k, rng, launches):
    X = _data(kind, 12_000, 16, rng)
    Q = _queries(X, 150, rng)
    index = build_index(X, backend="mlx")
    k_carry = distance._first_pass_carry(k, 16, 12_000)
    _same_as_builtin(index, Q, k, launches, fused=_topk.bucket(k_carry) is not None)
    _same_as_builtin(index, Q, k, launches, fused=_topk.bucket(k_carry) is not None,
                     query_block_size=32, train_block_size=5001)


@pytest.mark.parametrize("kind", ["dup20", "blobs", "far", "raw"])
@needs_kernel
def test_hard_data_resolves_the_same_rows_the_same_way(kind, rng, launches, any_size):
    """Rows the certificate rejects take the second pass, the count's skip and the exact
    search in the built-in blocks' groups, so the counts — and each wider product — are
    the same."""
    X = _data(kind, 50_000, 16, rng)
    Q = _queries(X, 64, rng)
    budget = 9 * MiB  # built-in blocks of 6 rows here, fused ones of 18
    seen = {"widened_rows": 0, "widen_skipped_rows": 0, "exact_rows": 0}
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        plan = distance.resolve_blocks(64, 50_000, backend="mlx", k_search=14)
        wide = distance.resolve_blocks(64, 50_000, backend="mlx", k_search=14,
                                       workset_factor=distance._TILE_WORKSET_FUSED)
        assert (plan.query_rows, wide.query_rows) == (6, 23), "the test lost its premise"
        for k in (10, 50):
            stats = _same_as_builtin(index, Q, k, launches)
            for key in seen:
                seen[key] += stats[key]
    assert seen["widened_rows"] > 0, "nothing reached the second pass"
    if kind in ("blobs", "far"):
        assert seen["exact_rows"] > 0, "nothing reached the exact search"
    if kind == "far":
        assert seen["widen_skipped_rows"] > 0, "the count skipped nothing"


@pytest.mark.parametrize("k", [5, 10])
@needs_kernel
def test_a_fused_block_makes_each_builtin_blocks_choices(k, rng, launches, any_size):
    """Integer training rows, queries on them, one row off the integer grid: the built-in
    path searches that row's block on the float path and every other block exactly. A
    fused block of several built-in blocks must not move them all to the float path,
    where ties on integer data fail the certificate and the counts grow."""
    X = rng.integers(0, 4, size=(50_000, 16)).astype(np.float64)
    Q = X[rng.integers(0, len(X), 90)].copy()
    Q[7, 0] += 0.3
    budget = 9 * MiB  # built-in blocks of 6 rows, fused ones of 18
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        stats = _same_as_builtin(index, Q, k, launches)
    assert stats["widened_rows"] > 0, "the float path's block was certified: no premise"


@needs_kernel
def test_query_blocks_split_where_the_builtin_blocks_choose_differently(rng):
    """Runs of whole built-in blocks that choose alike; an integer-path block alone."""
    X = rng.integers(0, 4, size=(2_000, 16)).astype(np.float64)
    index = build_index(X, backend="mlx")
    search = _search(index, 5, fused=True)
    search.group_rows = 6
    Q = X[:30].copy()
    Q[7, 0] += 0.3  # a float64 residual in the second block
    Q[13, 1] += 0.25  # a quarter, no residual: the integer path, shifted by 2
    Q[19, 2] += 1 / 3  # in the fourth block
    with use_device("gpu"):
        runs = search.query_blocks(Q, 0, 30, self_query=False)
        whole = search.query_blocks(Q, 0, 6, self_query=False)
    shape = [(a, b, blk.lo is not None, blk.int_shift) for a, b, blk in runs]
    assert shape == [
        (0, 6, False, 0), (6, 12, True, None), (12, 18, False, 2), (18, 24, True, None),
        (24, 30, False, 0),
    ]
    assert len(whole) == 1
    Y = rng.standard_normal((2_000, 16))
    gauss = _search(build_index(Y, backend="mlx"), 5, fused=True)
    gauss.group_rows = 6
    with use_device("gpu"):
        assert len(gauss.query_blocks(Y[:30] + 0.1, 0, 30, self_query=False)) == 1


@needs_kernel
def test_a_query_whose_rows_overflow_gives_the_kernel_up(rng, launches, monkeypatch, any_size):
    """A set stacked 40 times with a period of 128 rows: every query row's copies reach
    one lane, so the 64 bucket flags every row of the first block, and the rest of the
    query takes the built-in path."""
    monkeypatch.setattr(_topk, "_CHUNK_MIN", 131072)  # one chunk per row: one lane
    X = np.tile(rng.standard_normal((128, 16)), (40, 1))
    Q = X[rng.integers(0, 128, 600)]
    budget = 9 * MiB  # built-in blocks of 65 rows, fused ones of 195
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        _selection(64)  # its first-use check, at the real chunk widths
        before = launches["scan"]
        _same_as_builtin(index, Q, 30, launches)
    assert launches["scan"] - before == 1, "the kernel ran past the first block"


@pytest.mark.parametrize("d", [16, 256])
@pytest.mark.parametrize("n_query", [1, 2, 3, 5, 6, 7, 11, 12, 13, 18, 19, 31, 36, 37])
@needs_kernel
def test_query_row_counts_around_the_blocks(n_query, d, rng, launches, any_size):
    """One row, two, and either side of whole built-in blocks (6 rows here), including a
    one-row last block, which stays a block of its own — and at 256 features, where MLX
    multiplies a product no wider than deep with a kernel whose bits depend on its rows."""
    n_train = 50_000 if d == 16 else 3_200
    X = _data("dup20", n_train, d, rng)
    Q = _queries(X, n_query, rng)
    budget = 9 * MiB if d == 16 else int(9 * MiB * 3_200 / 50_000)
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        k_carry = distance._first_pass_carry(10, d, n_train)
        assert distance.resolve_blocks(37, n_train, backend="mlx", k_search=k_carry).query_rows == 6
        _same_as_builtin(index, Q, 10, launches)
        # Tiles of 1 000 columns: the last, 50 or 200 wide, is no wider than deep at 256
        # features, and every query keeps the built-in path.
        _same_as_builtin(index, Q, 10, launches, fused=d == 16, train_block_size=1_000)
        _same_as_builtin(index, Q, 10, launches, train_block_size=1_200)


def test_the_bounds_keep_whole_blocks_and_a_last_single_row_apart():
    assert distance._fused_bounds(37, 6, 18) == [(0, 18), (18, 36), (36, 37)]
    assert distance._fused_bounds(36, 6, 18) == [(0, 18), (18, 36)]
    assert distance._fused_bounds(5, 6, 18) == [(0, 5)]
    assert distance._fused_bounds(7, 6, 6) == [(0, 6), (6, 7)]
    assert distance._fused_bounds(1, 1, 20) == [(0, 1)]


@pytest.mark.parametrize(
    "budget, n_train, n_query",
    [
        # Too long for the built-in path to search a block in one tile, short enough
        # that the fused budget holds a few full-width rows: 4 * n_train in
        # (budget / 7, budget / 2]. The fused plan held 2-3 rows, and its blocks of a
        # built-in block's rows took the whole training set as one tile.
        (2 * MiB, 100_000, 300),
        (9 * MiB, 400_000, 2_048),
        (256 * MiB, 10_000_000, 4_096),
        (1024 * MiB, 40_000_000, 4_096),
        # And where the fused budget holds several built-in blocks.
        (9 * MiB, 50_000, 90),
        (1024 * MiB, 1_000_000, 4_096),
    ],
)
def test_a_fused_block_is_tiled_for_the_rows_it_holds(budget, n_train, n_query, monkeypatch):
    """Every fused tile within the fused budget, at least one whole built-in block."""
    monkeypatch.setattr(_topk, "selection", lambda device, k_carry: object())
    k_carry = 14
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        plan = distance.resolve_blocks(n_query, n_train, backend="mlx", k_search=k_carry)
        first = distance._first_pass_blocks(
            n_query, n_train, 2, k_carry, plan, device="gpu", plan_backend="mlx"
        )
    assert first.fused is not None
    assert (first.group_rows, first.group_cols) == (plan.query_rows, plan.train_cols)
    rows = max(end - start for start, end in first.bounds)
    assert rows % plan.query_rows == 0 or rows == n_query
    assert rows * first.train_cols * 4 <= budget // distance._TILE_WORKSET_FUSED
    assert rows * first.train_cols < 2**31


def test_a_tile_no_wider_than_deep_keeps_the_builtin_path(monkeypatch):
    """From 128 features MLX multiplies such a product with a kernel whose bits depend
    on its row count, so neither path's tiles, the ragged last one included, may be one."""
    monkeypatch.setattr(_topk, "selection", lambda device, k_carry: object())

    def fused(n_train, d, **kwargs):
        plan = distance.resolve_blocks(4_096, n_train, backend="mlx", k_search=14, **kwargs)
        return distance._first_pass_blocks(
            4_096, n_train, d, 14, plan, device="gpu", plan_backend="mlx", **kwargs
        ).fused is not None

    assert fused(1_025, 1_024) and not fused(1_024, 1_024) and not fused(1_500, 2_048)
    assert not fused(10_100, 256, train_block_size=1_000)  # a last tile of 100 columns
    assert fused(10_300, 256, train_block_size=1_000)
    assert distance._tiles_wider(10_300, 1_000, 256)
    assert not distance._tiles_wider(10_300, 1_000, 300)
    assert distance._tiles_wider(500, 10_000, 499) and not distance._tiles_wider(500, 10_000, 500)


@needs_kernel
def test_single_row_blocks_keep_the_builtin_path(rng, launches, any_size):
    """Every built-in block a one-row product: a fused block would round differently."""
    X = _data("gauss", 5_000, 16, rng)
    index = build_index(X, backend="mlx")
    _same_as_builtin(index, _queries(X, 9, rng), 10, launches, fused=False, query_block_size=1)
    _same_as_builtin(index, _queries(X, 1, rng), 10, launches, query_block_size=1)


@pytest.mark.parametrize("n_query, fused", [(255, False), (256, True)])
@needs_kernel
def test_the_pair_threshold(n_query, fused, rng, launches):
    """The first pass is fused from exactly _FUSED_MIN_PAIRS (query row, training row) pairs."""
    assert distance._FUSED_MIN_PAIRS == 256 * 16_384
    X = _data("gauss", 16_384, 8, rng)
    index = build_index(X, backend="mlx")
    _same_as_builtin(index, _queries(X, n_query, rng), 5, launches, fused=fused)


@pytest.mark.parametrize("kind", ["dup20", "offset"])
@needs_kernel
def test_self_queries_answer_as_the_builtin_ones(kind, rng, launches, any_size):
    """``kneighbors()`` with no query: the index's own rows, each one left out."""
    X = _data(kind, 6_000, 16, rng)
    index = build_index(X, backend="mlx")
    for k in (5, 32):
        _same_as_builtin(index, index.X, k, launches, exclude_self=True)
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=7).fit(X)
        before = launches["scan"]
        dist, idx = model.kneighbors()
        assert launches["scan"] > before
        with tuning_context(metal_kernels=False):
            dist_off, idx_off = model.kneighbors()
    _bits_equal(idx, idx_off)
    _bits_equal(dist, dist_off)


@pytest.mark.slow
@needs_kernel
def test_a_long_training_set_is_searched_in_tiles_its_budget_holds(rng, launches):
    """10 million rows at a 256 MiB budget, an 8 GB Mac's: the fused budget holds three
    full-width rows, a built-in block 256. Its block once took the whole training set
    as one tile, 2.56e9 elements, and MLX refused the reshape."""
    X = rng.standard_normal((10_000_000, 2)).astype(np.float32)
    Q = rng.standard_normal((256, 2)).astype(np.float32)
    budget = 256 * MiB
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        _same_as_builtin(index, Q, 5, launches)
        found = index.probe(5, rows=256)
        with tuning_context(metal_kernels=False):
            assert index.probe(5, rows=256) == found


@needs_kernel
def test_the_fused_working_set_stays_near_the_budget(rng):
    budget = 16 * MiB
    X = rng.normal(size=(200_000, 16)).astype(np.float32)
    Q = rng.normal(size=(512, 16)).astype(np.float32)
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        index.query(Q[:64], 10)
        mx.eval(mx.zeros(1))
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        index.query(Q, 10)
        peak = mx.get_peak_memory() - base
    assert peak <= 2 * budget, f"peak {peak / MiB:.0f} MiB against a {budget / MiB:.0f} MiB budget"


@needs_kernel
def test_a_block_searched_in_many_tiles_stays_near_the_budget(rng, launches):
    """64 rows against 200 000 at 4 MiB: the fused pass takes 25 train tiles, and the
    second pass, where the duplicates send every row, 86 more. Left lazy, every tile was
    alive at once: 153 MiB here, and 7.0 GiB above the index for 64 rows against
    40 000 000 x 2 at a 1 GiB budget."""
    budget = 4 * MiB
    X = _data("dup20", 200_000, 4, rng).astype(np.float32)
    Q = _queries(X, 64, rng)
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
        index.query(Q, 5)
        mx.eval(mx.zeros(1))
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        before = launches["scan"]
        stats: dict = {}
        index.query(Q, 5, stats=stats)
        peak = mx.get_peak_memory() - base
    assert launches["scan"] - before > 10, "the fused pass did not take many tiles"
    assert stats["widened_rows"] > 0, "nothing reached the second pass"
    assert peak <= 2 * budget, f"peak {peak / MiB:.0f} MiB against a {budget / MiB:.0f} MiB budget"


# ----------------------------------------------------------------------------------
# devices, the probe and the estimators
# ----------------------------------------------------------------------------------


def test_the_cpu_device_never_launches_a_kernel(rng, launches, any_size, monkeypatch):
    built = []
    monkeypatch.setattr(_topk, "_kernel", lambda *a: built.append(a))
    X = _data("dup20", 4_000, 16, rng)
    Q = _queries(X, 40, rng)
    cpu_index = build_index(X, backend="cpu")
    _same_as_builtin(cpu_index, Q, 10, launches, fused=False)
    if gpu_available():
        gpu_index = build_index(X, backend="mlx")
        _same_as_builtin(gpu_index, Q, 10, launches, fused=False, device="cpu")
    assert built == []


@pytest.mark.parametrize("kind", ["gauss", "blobs"])
@needs_kernel
def test_the_probe_answers_as_the_builtin_one(kind, rng, launches, any_size):
    index = build_index(_data(kind, 12_000, 64, rng), backend="mlx")
    before = launches["scan"]
    found = index.probe(10, rows=256)
    assert launches["scan"] > before
    with tuning_context(metal_kernels=False):
        assert index.probe(10, rows=256) == found


@needs_kernel
def test_a_patched_fit_probes_through_the_kernel(rng, launches):
    """512 probe rows against 30 000 training rows are past the pair threshold."""
    from mlxlearn.patching import _estimators as patched

    X = _data("gauss", 30_000, 256, rng)
    Q = _queries(X, 300, rng)
    model = patched.NearestNeighbors(n_neighbors=10).fit(X)
    assert launches["scan"] > 0 and model.execution_backend_ == "mlx"
    found = model._index_.probe(10, rows=256)
    fused = model.kneighbors(Q)
    with tuning_context(metal_kernels=False):
        builtin = patched.NearestNeighbors(n_neighbors=10).fit(X)
        assert builtin._index_.probe(10, rows=256) == found
        for a, b in zip(fused, builtin.kneighbors(Q), strict=True):
            _bits_equal(a, b)


# ----------------------------------------------------------------------------------
# first use: lazy, checked, once
# ----------------------------------------------------------------------------------


@pytest.mark.usefixtures("fresh_first_use")
@needs_kernel
def test_the_kernel_compiles_on_the_first_query_that_takes_it(rng, launches):
    X = _data("gauss", 16_384, 8, rng)
    index = build_index(X, backend="mlx")
    index.query(_queries(X, 255, rng), 5)
    assert _topk._checked.cache_info().currsize == 0, "compiled for a query below the threshold"
    index.query(_queries(X, 256, rng), 5)
    assert _topk._checked.cache_info().currsize == 1


@pytest.mark.usefixtures("fresh_first_use", "any_size")
@needs_kernel
def test_a_kernel_that_fails_to_build_keeps_the_builtin_path(rng, launches, monkeypatch):
    """What MLX raises for a library Metal cannot build; the query still answers."""

    def unbuildable(length):
        raise RuntimeError("[metal::Device] Unable to build metal library from source")

    monkeypatch.setattr(_topk, "_kernel", unbuildable)
    X = _data("gauss", 6_000, 16, rng)
    index = build_index(X, backend="mlx")
    _same_as_builtin(index, _queries(X, 40, rng), 10, launches, fused=False)
    assert _topk._checked.cache_info().currsize == 1, "the failure was not remembered"


@pytest.mark.usefixtures("fresh_first_use", "any_size")
@needs_kernel
def test_a_kernel_that_answers_wrongly_is_not_used(rng, launches, monkeypatch):
    real = _topk.Selection.smallest

    def off_by_one(self, mx_, *args, **kwargs):
        vals, cols, flags = real(self, mx_, *args, **kwargs)
        return vals, cols + 1, flags

    monkeypatch.setattr(_topk.Selection, "smallest", off_by_one)
    with use_device("gpu"):
        assert _topk.selection("gpu", 14) is None


@pytest.mark.usefixtures("fresh_first_use", "any_size")
@needs_kernel
def test_concurrent_fused_queries_and_the_first_compile(rng, launches, monkeypatch):
    """Eight threads at once on a process's first fused query: one compiles and checks
    the kernel, the others wait for it, and every answer is the built-in one."""
    checks = []
    real_check = _topk._known_answer

    def counting_check(select):
        checks.append(threading.get_ident())
        return real_check(select)

    monkeypatch.setattr(_topk, "_known_answer", counting_check)
    X = _data("blobs", 30_000, 32, rng)
    Q = _queries(X, 256, rng)
    index = build_index(X, backend="mlx")
    with tuning_context(metal_kernels=False):
        want = [index.query(Q[i::8], 10) for i in range(8)]

    n_threads = 8
    barrier = threading.Barrier(n_threads, timeout=60)
    results: list = [None] * n_threads
    errors: list[BaseException] = []

    def worker(slot):
        try:
            barrier.wait()
            results[slot] = [index.query(Q[slot::8], 10) for _ in range(3)]
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    join_or_fail(threads)
    assert not errors, errors
    assert len(checks) == 1, f"{len(checks)} first-use checks"
    assert launches["scan"] >= n_threads * 3
    for slot in range(n_threads):
        for dist, idx in results[slot]:
            _bits_equal(idx, want[slot][1])
            _bits_equal(dist, want[slot][0])


# ----------------------------------------------------------------------------------
# MLX 0.31
# ----------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(
    _metal.mlx_version()[:2] >= _metal.MIN_MLX, reason="the floor's gate, on MLX 0.31"
)
def test_mlx_031_never_builds_a_kernel(rng, monkeypatch, any_size):
    """0.31's unlocked library map crashed under two threads: not one kernel object."""
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kwargs: built.append(kwargs))
    X = _data("dup20", 6_000, 16, rng)
    index = build_index(X, backend="mlx")
    index.query(_queries(X, 40, rng), 10)
    index.probe(10)
    assert not _metal.kernels_enabled("gpu")
    assert built == []
