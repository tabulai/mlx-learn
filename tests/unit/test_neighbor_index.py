"""``NeighborIndex`` mechanics: publication, memory, devices, pickling, selection.

The numerical claims live in ``tests/parity/test_neighbors_fp32.py``. These pin the
machinery around them — the parts that fail as crashes, blow-ups and silent
aliasing rather than as a wrong distance.
"""

from __future__ import annotations

import copy
import copyreg
import dataclasses
import pickle
import sys
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._kernels import distance
from mlxlearn._kernels.distance import NeighborIndex, build_index, resolve_blocks
from mlxlearn.exceptions import UnsupportedInputError
from mlxlearn.neighbors import KNeighborsRegressor, NearestNeighbors
from tests._helpers import join_or_fail

MiB = 1024 * 1024

DEVICES = ["gpu", "cpu"] if gpu_available() else ["cpu"]


def _search(index, Q, k, *, device, train_block_size=None):
    """A query's search machinery over ``index`` and its one block of rows ``Q``.

    For driving one stage of the search directly; ``query`` builds the same pair.
    """
    search = distance._Search(
        index=index,
        state=index._ensure_device(device),
        mx=mx,
        k_search=k,
        k_carry=distance._first_pass_carry(k, index.n_features, index.n_samples),
        segmented=device == "gpu",
        fns=distance._kernels(device),
        plan_backend="mlx" if device == "gpu" else "cpu",
        train_block_size=train_block_size,
        counts={"rows": len(Q), "widened_rows": 0, "widen_skipped_rows": 0, "exact_rows": 0},
    )
    with use_device(device):
        block = search.query_block(Q, 0, len(Q), self_query=False)
    return search, block


def _exact(search, block, *, integer=False):
    with use_device("gpu" if search.segmented else "cpu"):
        return search.exact(block, integer=integer)


def _spy(monkeypatch, owner, name):
    """Record each call of ``owner.name``, which still runs."""
    calls = []
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


def _bits_equal(got, want):
    for a, b in zip(got, want, strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


# ----------------------------------------------------------------------------------
# publication
# ----------------------------------------------------------------------------------


def test_concurrent_first_queries_on_an_unpickled_index(rng, monkeypatch):
    """A burst of threads right after unpickling — joblib.load, then a thread pool.

    The device arrays are rebuilt lazily on the first query. They used to be
    published one at a time and still unevaluated, so a second thread could pass
    the "already built" check and read ``None``, or evaluate a lazy graph that
    belonged to another thread's stream: a segfault or "There is no Stream(gpu, N)
    in current thread". The build is slowed down here to hold that window open.
    """
    X = rng.normal(size=(3000, 16)).astype(np.float32)
    Q = rng.normal(size=(64, 16)).astype(np.float32)
    expected_dist, expected_idx = build_index(X, backend="mlx").query(Q, 5)
    restored = pickle.loads(pickle.dumps(build_index(X, backend="mlx")))

    builds = []
    real_build = distance._build_state

    def slow_build(*args):
        builds.append(threading.get_ident())
        time.sleep(0.05)
        return real_build(*args)

    monkeypatch.setattr(distance, "_build_state", slow_build)

    n_threads = 12
    barrier = threading.Barrier(n_threads, timeout=60)
    results: list = [None] * n_threads
    errors: list[BaseException] = []

    def worker(slot):
        try:
            barrier.wait()
            results[slot] = restored.query(Q, 5)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        join_or_fail(threads)
    finally:
        sys.setswitchinterval(interval)

    assert not errors, errors
    assert len(builds) == 1, "every thread uploaded its own copy of the training set"
    for dist, idx in results:
        np.testing.assert_array_equal(idx, expected_idx)
        np.testing.assert_array_equal(dist, expected_dist)


def test_the_lock_is_not_pickled_and_is_fresh_after_loading(rng):
    index = build_index(rng.normal(size=(50, 3)).astype(np.float32), backend="cpu")
    restored = pickle.loads(pickle.dumps(index))
    assert restored._lock is not index._lock
    assert restored._lock.acquire(blocking=False)
    restored._lock.release()


# ----------------------------------------------------------------------------------
# no aliasing, one copy
# ----------------------------------------------------------------------------------


def test_the_index_owns_a_read_only_copy(rng):
    """Never the caller's buffer; and the host view is the device buffer itself."""
    X = rng.normal(size=(300, 4)).astype(np.float32)
    Q = rng.normal(size=(5, 4)).astype(np.float32)
    index = build_index(X, backend="cpu")
    before = index.query(Q, 3)

    assert not np.shares_memory(index.X, X)
    assert not index.X.flags.writeable
    X[:] = 0.0
    after = index.query(Q, 3)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[0], after[0])


def test_querying_with_the_index_own_rows_matches_a_copy_of_them(rng):
    """The self-query path reads the device rows directly; it must agree with the
    ordinary path given the same values."""
    X = rng.normal(size=(700, 6))
    index = build_index(X, backend="mlx")
    direct = index.query(index.X, 4, exclude_self=True)
    copied = index.query(np.array(X), 4, exclude_self=True)
    np.testing.assert_array_equal(direct[1], copied[1])
    np.testing.assert_allclose(direct[0], copied[0], rtol=1e-6)


# ----------------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------------


def test_the_distance_tile_is_budgeted_as_a_working_set():
    """The budget bounds the whole working set, not the tile alone."""
    budget = 64 * MiB
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        plan = resolve_blocks(4096, 262_144, backend="mlx", k_search=14)
    assert plan.query_rows * plan.train_cols * 4 * distance._TILE_WORKSET_FACTOR <= budget


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
def test_query_peak_memory_stays_near_the_budget(backend, rng):
    """Large ``k`` on wide data: the refinement gather used to be unbudgeted.

    Before, the peak was about 7× the budget from the tile alone, and the
    ``(rows, k, d)`` gather came on top — 7.4 GB at k=1000, d=768. Here it was
    20× the budget.
    """
    budget = 16 * MiB
    X = rng.normal(size=(8000, 256)).astype(np.float32)
    Q = rng.normal(size=(512, 256)).astype(np.float32)
    with tuning_context(
        distance_block_bytes=budget,
        distance_block_bytes_max=budget,
        cpu_distance_block_bytes=budget,
    ):
        index = build_index(X, backend=backend)
        index.query(Q[:8], 200)
        mx.eval(mx.zeros(1))
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        index.query(Q, 200)
        peak = mx.get_peak_memory() - base
    assert peak <= 2 * budget, f"peak {peak / MiB:.0f} MiB against a {budget / MiB:.0f} MiB budget"


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
def test_a_block_searched_in_many_tiles_stays_near_the_budget(backend, rng):
    """64 rows against 200 000 at 4 MiB, by MLX's own selection: the first pass takes 86
    train tiles, and the second pass, where the duplicated rows send 63 of them, 85 more.
    Left lazy, every tile was alive at once: 167 MiB here on the GPU and 39 MiB on MLX's
    CPU, and 7.0 GiB above the index for 64 rows against 40 000 000 x 2 at a 1 GiB budget.
    ``test_fused_selection.py`` holds the fused kernel's pass to the same budget.
    """
    budget = 4 * MiB
    distinct = rng.normal(size=(10_000, 4)).astype(np.float32)
    X = distinct[rng.integers(0, len(distinct), 200_000)]  # every row about 20 times
    Q = X[rng.integers(0, len(X), 64)] + np.float32(1e-3) * rng.normal(size=(64, 4)).astype(
        np.float32
    )
    with tuning_context(
        distance_block_bytes=budget,
        distance_block_bytes_max=budget,
        cpu_distance_block_bytes=budget,
        metal_kernels=False,
    ):
        k_carry = distance._first_pass_carry(5, 4, len(X))
        plan = resolve_blocks(len(Q), len(X), backend=backend, k_search=k_carry)
        index = build_index(X, backend=backend)
        index.query(Q, 5)
        mx.eval(mx.zeros(1))
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        stats: dict = {}
        index.query(Q, 5, stats=stats)
        peak = mx.get_peak_memory() - base
    assert plan.query_rows == len(Q) and plan.train_cols * 50 < len(X), "not many tiles"
    assert stats["widened_rows"] > 0, "nothing reached the second pass"
    assert peak <= 2 * budget, f"peak {peak / MiB:.0f} MiB against a {budget / MiB:.0f} MiB budget"


# ----------------------------------------------------------------------------------
# the exact search's tiles
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 3, 21, 256, 1024])
@pytest.mark.parametrize("pair_bytes", [248, 3128])
def test_exact_tiles_are_as_wide_as_the_budget_allows(rows, pair_bytes):
    """Power-of-two multiples of the old fixed width, and never narrower than it."""
    budget = 256 * MiB
    cols = distance._exact_cols(budget, rows, pair_bytes)
    multiple = cols // distance._EXACT_MIN_COLS
    assert cols % distance._EXACT_MIN_COLS == 0 and multiple & (multiple - 1) == 0
    assert budget < 2 * rows * cols * pair_bytes, "a tile twice as wide still fits"
    if cols > distance._EXACT_MIN_COLS:
        assert rows * cols * pair_bytes <= budget


@pytest.mark.parametrize("device", DEVICES)
def test_the_exact_search_takes_few_rows_in_one_tile(device, rng, monkeypatch):
    """A handful of rows the certificate sent to the exact search, over 100 000 rows.

    With a fixed 4096-row tile, evaluated one at a time, that was 25 tiles and 25
    round trips to the device, however few rows there were. The budget holds all of
    it; and the tile's width changes no bit of the answer.
    """
    X = rng.normal(size=(100_000, 8)).astype(np.float32)
    Q = rng.normal(size=(3, 8)).astype(np.float32)
    index = build_index(X, backend="mlx" if device == "gpu" else "cpu", device=device)

    tiles = _spy(monkeypatch, distance, "_select_smallest")
    search, block = _search(index, Q, 10, device=device)
    wide = _exact(search, block)
    assert len(tiles) == 1

    del tiles[:]
    narrow_search, block = _search(index, Q, 10, device=device, train_block_size=4096)
    _bits_equal(_exact(narrow_search, block), wide)
    assert len(tiles) == 25


@pytest.mark.parametrize("device", DEVICES)
def test_narrow_exact_tiles_are_evaluated_together(device, rng, monkeypatch):
    """Tiles are evaluated once one more would take those in flight past the budget,
    rather than one at a time."""
    X = rng.normal(size=(50_000, 8)).astype(np.float32)
    Q = rng.normal(size=(3, 8)).astype(np.float32)
    budget = 8 * MiB
    with tuning_context(
        distance_block_bytes=budget, distance_block_bytes_max=budget, cpu_distance_block_bytes=budget
    ):
        index = build_index(X, backend="mlx" if device == "gpu" else "cpu", device=device)
        search, block = _search(index, Q, 10, device=device, train_block_size=1000)
        evaluations = _spy(monkeypatch, mx, "eval")
        _exact(search, block)
    # 50 tiles of 3 x 1000 pairs at 152 bytes: evaluated after the 18th and the 36th,
    # and the last 14 when the result is read. One at a time, that was 50.
    assert len(evaluations) == 2


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
@pytest.mark.parametrize("n_query", [2, 64])
def test_exact_search_peak_memory_stays_near_the_budget(backend, n_query, rng):
    """Rows the certificate cannot help, all searched exactly, against 200 000 rows.

    Two rows get tiles eight times the old fixed width here, 64 rows the old width.
    A tile as wide as the training set would be ``n_query`` x 200 000 differences of
    16 floats: 26 MB for two rows and 820 MB for 64, three times that unfused on the
    CPU.
    """
    budget = 16 * MiB
    X = np.vstack([1000.0 + 1e-4 * rng.normal(size=(2000, 16)), rng.normal(size=(198_000, 16))])
    Q = 1000.0 + 1e-4 * rng.normal(size=(n_query, 16))
    with tuning_context(
        distance_block_bytes=budget,
        distance_block_bytes_max=budget,
        cpu_distance_block_bytes=budget,
    ):
        index = build_index(X, backend=backend)
        index.query(Q, 5)
        mx.eval(mx.zeros(1))
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        stats: dict = {}
        index.query(Q, 5, stats=stats)
        peak = mx.get_peak_memory() - base
    assert stats["exact_rows"] == n_query, "precondition: every row took the exact search"
    assert peak <= 2 * budget, f"peak {peak / MiB:.0f} MiB against a {budget / MiB:.0f} MiB budget"


# ----------------------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------------------


def test_a_gpu_built_index_runs_where_it_is_told(rng, monkeypatch):
    """A model fitted on a Mac, unpickled in a process without Metal.

    MLX's CPU runs the same float32 code. Whether to use it is the estimator's
    decision, from the configured device (``inference_device``); the index runs on
    the device it is handed and reads no configuration. It used to read it itself,
    below the layer allowed to.
    """
    X = rng.normal(size=(2000, 8)).astype(np.float32)
    Q = rng.normal(size=(40, 8)).astype(np.float32)
    expected_dist, expected_idx = build_index(X, backend="cpu").query(Q, 5)

    def no_configuration():
        raise AssertionError("the index read the user's configuration")

    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    monkeypatch.setattr("mlxlearn._common.device.get_config", no_configuration)
    fitted = build_index(X, backend="mlx", device="cpu")
    for index in (fitted, pickle.loads(pickle.dumps(fitted))):
        dist, idx = index.query(Q, 5, device="cpu")
        np.testing.assert_array_equal(idx, expected_idx)
        np.testing.assert_array_equal(dist, expected_dist)


def test_the_cpu_device_does_not_compile(rng, monkeypatch):
    """Only the GPU fuses the elementwise pieces.

    On MLX's CPU each compiled graph shells out to a C compiler, once per MLX
    version and ``$TMPDIR``, and runs no faster: a 200-row fit and predict — the
    Layer 1 path for every problem below the crossover — took 3-4 s on a fresh
    machine, against 0.1 s for 0.1.0a2.
    """
    compiled = []
    real_compile = mx.compile

    def recording_compile(fn, *args, **kwargs):
        compiled.append(fn.__name__)
        return real_compile(fn, *args, **kwargs)

    monkeypatch.setattr(mx, "compile", recording_compile)
    distance._kernels.cache_clear()
    try:
        X = rng.normal(size=(200, 5))
        for data in (X.astype(np.float32), X):  # the float64 residual path too
            index = build_index(data, backend="cpu")
            index.query(data[:7] + 0.1, 3)
            index.query(index.X, 3, exclude_self=True)
    finally:
        distance._kernels.cache_clear()
    assert compiled == []


# ----------------------------------------------------------------------------------
# pickling
# ----------------------------------------------------------------------------------


def test_a_float64_index_round_trips_exactly(rng):
    """The residual is part of the state; without it a reloaded index would answer
    differently from the one that was saved."""
    X = 1.7e9 + rng.uniform(0, 3600, size=(3000, 2))
    Q = 1.7e9 + rng.uniform(0, 3600, size=(40, 2))
    index = build_index(X, backend="mlx")
    assert index.X_lo is not None

    restored = pickle.loads(pickle.dumps(index))
    for got, want in zip(restored.query(Q, 5), index.query(Q, 5), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "reload",
    [lambda model: pickle.loads(pickle.dumps(model)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
@pytest.mark.parametrize("estimator", [NearestNeighbors, KNeighborsRegressor])
def test_a_reloaded_model_keeps_one_copy_of_its_training_rows(estimator, reload, rng):
    """A fitted model's ``_fit_X`` is the index's own host view, and stays that way.

    Reloaded, the index rebuilds its device copy on the first query and points its
    ``X`` at it; ``_fit_X`` stayed on the loaded array, so the model held one more
    full copy of the training set than a fresh fit, and saving it again doubled the
    file.
    """
    X = rng.normal(size=(3000, 16)).astype(np.float32)
    model = estimator(n_neighbors=3).fit(X, X[:, 0])
    assert model.execution_backend_ in ("mlx", "cpu")
    fresh = len(pickle.dumps(model))

    loaded = reload(model)
    loaded.kneighbors(X[:5])
    assert loaded._fit_X is loaded._index_.X
    assert len(pickle.dumps(loaded)) < 1.05 * fresh


def test_a_pickle_from_before_the_residual_still_loads(rng):
    X = rng.normal(size=(500, 4)).astype(np.float32)
    Q = rng.normal(size=(10, 4)).astype(np.float32)
    index = build_index(X, backend="cpu")

    old = NeighborIndex.__new__(NeighborIndex)
    old.__setstate__({"X": np.array(index.X), "center": index.center, "backend": "cpu"})
    for got, want in zip(old.query(Q, 3), index.query(Q, 3), strict=True):
        np.testing.assert_array_equal(got, want)


class _PickledBeforeTheColumnExtremes:
    """Pickles to the stream a ``NeighborIndex`` made before the index kept its training
    columns' extremes: the same reduction, and the state dict it had then. (The
    pickler checks that ``__newobj__``'s class is the object's own.)"""

    __class__ = property(lambda self: NeighborIndex)  # type: ignore[assignment]

    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        return copyreg.__newobj__, (NeighborIndex,), self.state


@pytest.mark.parametrize("device", DEVICES)
def test_an_index_pickled_before_the_column_extremes_answers_the_same(device, rng, monkeypatch):
    """The extremes decide whether the integer path's keys fit in 32 bits. They are
    derived state, so an old pickle rebuilds them on its first query; and a state
    without them — which nothing builds now — keys in 64 bits, to the same answer."""
    backend = "mlx" if device == "gpu" else "cpu"
    X = rng.integers(0, 4, size=(3000, 6)).astype(np.float64)
    Q = rng.integers(0, 4, size=(40, 6)).astype(np.float64)
    index = build_index(X, backend=backend, device=device)
    want = index.query(Q, 7, device=device)
    assert set(index.__getstate__()) == {"X", "center", "backend"}, "the pickle changed shape"

    old = pickle.loads(pickle.dumps(_PickledBeforeTheColumnExtremes(
        {"X": np.array(index.X), "center": index.center, "backend": backend}
    )))
    assert isinstance(old, NeighborIndex) and old._state is None
    wide_keys = _spy(monkeypatch, distance, "_composite_keys")
    _bits_equal(old.query(Q, 7, device=device), want)
    assert wide_keys == [], "the rebuilt index keys in 32 bits"

    stripped = pickle.loads(pickle.dumps(index))
    stripped._state = dataclasses.replace(
        stripped._ensure_device(device), col_min=None, col_max=None
    )
    _bits_equal(stripped.query(Q, 7, device=device), want)
    assert wide_keys, "without the extremes the keys fall back to 64 bits"


# ----------------------------------------------------------------------------------
# row limits
# ----------------------------------------------------------------------------------


def test_exact_search_keys_address_rows_past_int32():
    """The exact search packs each training index into the low 32 bits of its key.

    The indices were an ``arange`` starting at the tile's first row, which MLX 0.31
    — the supported floor — rejects with a TypeError from row ``2**31`` on.
    """
    vals = mx.array([[1.0, 0.5, 2.0]], dtype=mx.float32)
    for t_start in (0, 2**31 + 5, 2**32 - 3):
        keys = np.asarray(distance._composite_keys(mx, vals, t_start))
        np.testing.assert_array_equal(keys & 0xFFFFFFFF, [t_start + np.arange(3)])
        values = (keys >> np.uint64(32)).astype(np.uint32).view(np.float32)
        np.testing.assert_array_equal(values, [[1.0, 0.5, 2.0]])


@pytest.mark.parametrize(
    ("max_sq", "n_train", "index_bits"),
    [
        (2**15 - 1, 2**17, 17),  # 15 + 17 bits: exactly 32
        (2**15, 2**17, None),  # 16 + 17: 33
        (2**15 - 1, 2**17 + 1, None),  # 15 + 18: 33
        (2**24, 2**7, 7),  # the largest distance the integer path reaches, 25 bits
        (0, 2**31, 31),
        (0, 2**31 + 1, None),  # an index of 32 bits leaves nothing to shift by
        (5, 1, 0),
        (None, 10, None),  # no column extremes to bound the distances with
    ],
)
def test_integer_keys_are_32_bits_exactly_when_both_parts_fit(max_sq, n_train, index_bits):
    assert distance._int_key_bits(max_sq, n_train) == index_bits


def _lexsorted_neighbors(X, Q, k):
    """The first ``k`` under ``(distance, index)``, in float64: exact for small integers."""
    d2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    order = np.arange(X.shape[0])
    idx = np.array([np.lexsort((order, row))[:k] for row in d2])
    return np.sqrt(np.take_along_axis(d2, idx, axis=1)), idx


@pytest.mark.parametrize("device", DEVICES)
def test_integer_keys_at_the_32_bit_boundary(device, rng, monkeypatch):
    """4097 training rows need 13 bits of index. Queries whose farthest possible
    training row is 512² + 511² away (19 bits) are keyed in 32 bits; one more query
    with a zero in the second column makes it 2·512² = 2**19 (20 bits), and 64.

    Heavy ties — 81 distinct points among 4097 rows — so the key has to order the
    index exactly: both widths must return the exact ``(distance, index)`` order.
    """
    X = (rng.integers(0, 9, size=(4097, 2)) * 64).astype(np.float64)
    X[:2] = [[0, 0], [512, 512]]  # the columns' extremes
    Q32 = np.column_stack([rng.integers(0, 513, 30), rng.integers(1, 512, 30)]).astype(np.float64)
    Q32[0] = [0, 1]  # 512 from the farthest training row in the first column, 511 in the second
    Q64 = Q32.copy()
    Q64[0, 1] = 0
    index = build_index(X, backend="mlx" if device == "gpu" else "cpu", device=device)
    wide_keys = _spy(monkeypatch, distance, "_composite_keys")

    answers = []
    for Q, bound, width in ((Q32, 512**2 + 511**2, 32), (Q64, 2**19, 64)):
        del wide_keys[:]
        dist, idx = index.query(Q, 10, device=device)
        assert bool(wide_keys) == (width == 64)
        _, block = _search(index, Q, 10, device=device)
        assert (block.int_shift, block.int_max_sq) == (0, bound)

        want_dist, want_idx = _lexsorted_neighbors(X, Q, 10)
        np.testing.assert_array_equal(idx, want_idx)
        np.testing.assert_array_equal(dist, want_dist)
        answers.append((dist[1:], idx[1:]))
    _bits_equal(*answers)  # the rows both blocks share


def test_more_rows_than_a_key_can_address_are_refused():
    """Past ``2**32`` rows the index would come back truncated, and the distance
    corrupted by its high bits: refused at fit instead, before anything is copied."""
    X = np.lib.stride_tricks.as_strided(
        np.zeros(1, np.float32), shape=(2**32 + 1, 1), strides=(0, 4)
    )
    with pytest.raises(UnsupportedInputError, match="rows"):
        build_index(X, backend="cpu")


# ----------------------------------------------------------------------------------
# the count before the second pass
# ----------------------------------------------------------------------------------


def _band_rows(n_train, stride):
    """The rows ``band_count`` samples, as the host computes them."""
    window = np.arange(distance._band_windows(n_train, stride), dtype=np.uint32)
    return window * np.uint32(stride) + distance._band_offset(window, np.uint32(stride))


@pytest.mark.parametrize(
    ("n_train", "stride"), [(40_320, 16), (1_000_003, 16), (5_000, 17), (544, 17), (545, 17)]
)
def test_the_count_samples_one_row_in_each_window(n_train, stride):
    """One row in each window of ``stride``, inside it; the last window, when short,
    keeps its row only where the row falls inside the index."""
    rows = _band_rows(n_train, stride).astype(np.int64)
    np.testing.assert_array_equal(rows // stride, np.arange(rows.size))
    assert rows.max() < n_train
    last = -(-n_train // stride) - 1
    offset = int(distance._band_offset(np.array([last], np.uint32), np.uint32(stride))[0])
    assert rows.size == last + int(last * stride + offset < n_train)


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_samples_the_rows_the_host_says(device):
    """The hash is ``uint32`` arithmetic, which wraps alike in NumPy and in MLX."""
    n_train, stride = 100_003, 17
    fns = distance._kernels(device)
    with use_device(device):
        got = fns.band_rows(
            mx.arange(distance._band_windows(n_train, stride), dtype=mx.uint32),
            mx.array(stride, dtype=mx.uint32),
        )
        got = np.array(got)
    assert got.dtype == np.uint32
    np.testing.assert_array_equal(got, _band_rows(n_train, stride))


@pytest.mark.parametrize("period", [16, 1008, 1024, 16_384])
def test_the_count_does_not_line_up_with_rows_that_repeat(period):
    """Rows repeating every ``period`` rows, a multiple of the stride, 48 times over.

    Every 16th row, which the count used to take, holds all 48 copies of a sixteenth
    of the rows and none of the others', so the rows at multiples of 16 counted 16
    times their copies. Sampled at a hashed place in each window, every row has
    three copies in the sample on average, and none has half the 34 that the call
    at twice the second pass's carry needs at k=10.
    """
    stride, copies = 16, 48
    per_row = np.bincount(_band_rows(period * copies, stride) % period, minlength=period)
    assert per_row.mean() == copies / stride
    assert per_row.max() < 17


# ----------------------------------------------------------------------------------
# two-stage selection
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2048, 2049, 2048 + 511, 5003, 9001])
@pytest.mark.parametrize("take", [1, 14, 128, 129, 270])
def test_segmented_selection_is_exact(width, take, rng):
    """Ragged tails, takes either side of a segment-width step, and heavy ties."""
    vals = rng.random((7, width)).astype(np.float32)
    vals[:, ::3] = vals[:, :1]
    device = mx.gpu if gpu_available() else mx.cpu
    with mx.stream(device):
        idx, got = distance._select_smallest(mx, mx.array(vals), take, segmented=True)
        mx.eval(idx, got)
    idx = np.asarray(idx).astype(np.int64)
    got = np.asarray(got)

    np.testing.assert_array_equal(np.sort(got, axis=1), np.sort(vals, axis=1)[:, :take])
    np.testing.assert_array_equal(np.take_along_axis(vals, idx, axis=1), got)
    assert np.all(idx < width)
    assert all(len(set(row)) == take for row in idx.tolist())


# ----------------------------------------------------------------------------------
# the fit-time probe
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["mlx", "cpu"] if gpu_available() else ["cpu"])
def test_the_probe_counts_what_the_first_pass_cannot_certify(backend, rng):
    """Gaussian rows certify on the first pass; rows repeated 20 times cannot, since
    their k-th neighbor ties with the copies left out; integer rows take the exact
    path and are reported as such, unsearched."""
    X = rng.normal(size=(6000, 32))
    found = build_index(X, backend=backend).probe(5)
    assert found == {"rows": 64, "uncertified": 0, "integer": False}

    duplicated = rng.normal(size=(300, 32))[rng.integers(0, 300, size=6000)]
    found = build_index(duplicated, backend=backend).probe(5)
    assert found["rows"] == 64 and found["uncertified"] > 48 and not found["integer"]

    counts = rng.poisson(2.0, size=(6000, 32)).astype(np.float64)
    assert build_index(counts, backend=backend).probe(5) == {
        "rows": 0, "uncertified": 0, "integer": True,
    }


def test_the_probe_agrees_with_the_query_it_predicts(rng):
    """The probe is the query's own first pass: the rows it counts are the rows a query
    of the same training rows searches again."""
    X = rng.normal(size=(5000, 24))
    X[:, 0] = rng.uniform(0.0, 1e4, size=5000)
    X = X.astype(np.float32)
    index = build_index(X, backend="cpu")
    found = index.probe(10)
    rows = np.unique(np.linspace(0, 4999, 64).astype(np.intp))
    stats: dict = {}
    index.query(X[rows], 10, stats=stats)
    assert found["uncertified"] == stats["widened_rows"] > 0


def test_the_probe_searches_as_many_rows_as_it_is_asked_to(rng):
    """Patched dispatch probes ``knn_probe_rows`` rows (512), enough to see a rate of
    uncertified rows near 1% all but one time in 170, which 64 rows miss about half the
    time and 256 one time in thirteen; fewer when the index has fewer rows."""
    X = rng.normal(size=(3000, 16))
    X[:, 0] *= 100.0
    index = build_index(X, backend="cpu")
    wide = index.probe(10, rows=256)
    assert wide["rows"] == 256 and not wide["integer"]
    rows = np.unique(np.linspace(0, 2999, 256).astype(np.intp))
    stats: dict = {}
    index.query(X[rows], 10, stats=stats)
    assert wide["uncertified"] == stats["widened_rows"]
    assert build_index(X[:100], backend="cpu").probe(10, rows=256)["rows"] == 100


def test_the_probe_of_a_tiny_index_is_the_exact_answer(rng):
    """With every row a candidate there is nothing to certify."""
    X = rng.normal(size=(12, 3))
    assert build_index(X, backend="cpu").probe(10) == {
        "rows": 12, "uncertified": 0, "integer": False,
    }
