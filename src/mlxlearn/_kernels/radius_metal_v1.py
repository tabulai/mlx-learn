"""Interim Metal radius route: a certified binary32 Gram screen with host refinement.

The GPU screens every (query, training row) pair with the k-nearest-neighbour
Gram lower bound on centred coordinates (see ``mlxlearn_device._radius``):
pairs whose certified lower bound exceeds the float32 squared-radius cut are
provably outside the radius. The remaining candidates are compacted on the GPU
with exact per-row counts and decided on the host by
:func:`radius_host.refine`, so membership is exact and identical to every
other route.

Work is tiled and pipelined:

* Query blocks of ``clamp(floor64(2**26 / n), 64, 4096)`` rows; full-width
  product tiles while ``n <= 2**20``, column tiles of ``2**20`` beyond.
* Waves of at most two product tiles (and at most ``2**26`` product elements)
  get one evaluation each. The next wave is queued with ``mx.async_eval``
  before the host finishes the current one, so at most two waves of products
  are alive and the GPU stays busy while the host sorts and refines.
* Each row stores up to ``capacity`` candidates while still counting all of
  them. Rows that overflow are rescanned with exactly the capacity they need.
* Per-row atomic arrival order is unspecified; the host restores ascending
  columns with a row-wise sort of the small packed block (sentinel padded), so
  results are deterministic. Rows stay grouped; column tiles merge per row
  without any global sort.

Every MLX operation names the GPU stream: the process-global default device
may be changed by another thread at any time.
"""

from __future__ import annotations

import math
import re
import threading
from collections import deque
from contextlib import nullcontext
from functools import lru_cache

import numpy as np

from .._common.arrays import INDEX_DTYPE
from .radius_contract import Thresholds
from .radius_host import refine
from .radius_result import RadiusCSR, empty_result, from_grouped

__all__ = ["ROUTE", "debug", "metal_query"]

ROUTE = "metal"

#: Product elements per tile (256 MiB of float32 products).
_TILE_ELEMENTS = 1 << 26
#: Product elements per wave; a wave holds one or two tiles.
_WAVE_ELEMENTS = 1 << 26
_MAX_WAVE_TILES = 2
#: Waves queued on the GPU before the host finishes the oldest one.
_PIPELINE_WAVES = 2
#: Training rows per product tile; wider indexes are split into column tiles.
_COLUMN_TILE = 1 << 20
_ROW_ALIGN = 64
_MIN_QUERY_ROWS = 64
_MAX_QUERY_ROWS = 4096
_MIN_CAPACITY = 256
_MAX_CAPACITY = 65536
#: Tests only: replaces the first-pass capacity to exercise the overflow loop.
_FORCE_CAPACITY: int | None = None
#: Overflow rescans size themselves from the previous counts; after this many
#: rounds a rescan takes the full tile width, which cannot overflow.
_MAX_OVERFLOW_ROUNDS = 3
_SENTINEL = np.uint32(0xFFFFFFFF)


class _Counters:
    """Totals for tests and diagnostics. Never holds arrays."""

    _FIELDS = ("calls", "screened_pairs", "candidates", "waves", "tiles",
               "overflow_launches", "overflow_rows")

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            for name in self._FIELDS:
                setattr(self, name, 0)

    def add(self, **amounts) -> None:
        with self._lock:
            for name, amount in amounts.items():
                setattr(self, name, getattr(self, name) + int(amount))

    def snapshot(self) -> dict:
        with self._lock:
            return {name: getattr(self, name) for name in self._FIELDS}


#: Module-level counters: ``candidates`` is the number of screened pairs that
#: reached host refinement (self pairs included), ``screened_pairs`` is
#: ``n_query * n_train`` summed over calls.
debug = _Counters()

_PREPARE_LOCK = threading.Lock()
_SERIAL_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _serialize_gpu() -> bool:
    """MLX before 0.32 aborts or rebuilds kernels unsafely under threaded GPU work."""
    import mlx.core as mx

    match = re.match(r"(\d+)\.(\d+)", str(getattr(mx, "__version__", "")))
    return match is None or (int(match.group(1)), int(match.group(2))) < (0, 32)


def _plan(n: int, n_query: int) -> tuple[int, int, int]:
    """Column tile width, query block rows and product tiles per wave."""
    width = n if n <= _COLUMN_TILE else _COLUMN_TILE
    rows = _TILE_ELEMENTS // max(n, 1) // _ROW_ALIGN * _ROW_ALIGN
    rows = min(max(rows, _MIN_QUERY_ROWS), _MAX_QUERY_ROWS)
    largest = min(rows, n_query) * width
    per_wave = max(1, min(_MAX_WAVE_TILES, _WAVE_ELEMENTS // max(largest, 1)))
    return width, rows, per_wave


#: A tile's packed candidate buffer may use this many slots regardless of the
#: density hint (16 MiB): small batches then never pay an overflow rescan,
#: which costs a GPU round trip, for a few skewed rows.
_GENEROUS_SLOTS = 1 << 22


def _capacity(cap_hint, width: int, rows: int = 0) -> int:
    """``clamp(next_pow2(2 * hint + 64), 256, min(width, 65536))``, or the
    widest capacity when ``rows`` such slots fit in :data:`_GENEROUS_SLOTS`."""
    if _FORCE_CAPACITY is not None:
        return max(1, min(int(_FORCE_CAPACITY), width))
    widest = max(1, min(width, _MAX_CAPACITY))
    if rows and rows * widest <= _GENEROUS_SLOTS:
        return widest
    hint = 0.0 if cap_hint is None else float(cap_hint)
    if math.isnan(hint) or hint < 0:
        hint = 0.0
    if hint > _MAX_CAPACITY:
        capacity = _MAX_CAPACITY
    else:
        target = 2 * math.ceil(hint) + 64
        capacity = 1 << (target - 1).bit_length()
    return max(1, min(max(capacity, _MIN_CAPACITY), width, _MAX_CAPACITY))


def _prepared(device_index):
    """The index's centred training rows, centre and centred norms, evaluated."""
    import mlx.core as mx

    with _PREPARE_LOCK:
        if device_index._train is None:
            # The preparation is an mx.compile'd graph without stream
            # arguments; build it under the GPU stream.
            with mx.stream(mx.gpu):
                device_index._prepare_gram()
        arrays = (device_index._center, device_index._train, device_index._train_sq)
        mx.eval(*arrays)
    return arrays


def _sort_block(packed: np.ndarray, lengths: np.ndarray, width: int,
                sort: bool = True) -> np.ndarray:
    """Pad each row past its length with the sentinel, sort rows, compress.

    With ``sort=False`` rows are only compressed (grouped, unordered); the
    native finisher orders every row by column itself.
    """
    valid = np.arange(width, dtype=INDEX_DTYPE) < lengths[:, None]
    if not sort:
        return packed[:, :width][valid].astype(INDEX_DTYPE)
    block = np.where(valid, packed[:, :width], _SENTINEL)
    block.sort(axis=1)
    return block[valid].astype(INDEX_DTYPE)


def _sorted_rows(counts: np.ndarray, packed, capacity: int, sort: bool = True):
    """Ascending candidates of every row whose count fits in ``capacity``.

    Returns ``(lengths, fits, cols)``: overflowed rows get length zero and
    ``fits[row] = False``; ``cols`` is grouped by row. Rows are sorted in one
    sentinel-padded block, or, when a few long rows would make that block
    mostly padding, in blocks of rows whose lengths share a power of two.
    """
    fits = counts <= capacity
    lengths = np.where(fits, counts, 0)
    needed = int(lengths.max(initial=0))
    if needed == 0:
        return lengths, fits, np.empty(0, dtype=INDEX_DTYPE)
    packed = np.asarray(packed)
    total = int(lengths.sum())
    if not sort or len(lengths) * needed <= 4 * total + 4096:
        return lengths, fits, _sort_block(packed, lengths, needed, sort)
    rows = np.flatnonzero(lengths)
    classes = np.zeros(len(rows), dtype=INDEX_DTYPE)
    longer = lengths[rows] > 1
    classes[longer] = np.ceil(np.log2(lengths[rows][longer])).astype(INDEX_DTYPE)
    parts = []
    for level in np.unique(classes).tolist():
        members = rows[classes == level]
        part = lengths[members]
        parts.append((members, part, _sort_block(packed[members], part, int(part.max()))))
    _, cols = _merge_rows(len(lengths), parts)
    return lengths, fits, cols


def _merge_rows(n_rows: int, parts) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate, per row, segments from several row-grouped parts.

    ``parts`` holds ``(positions, lengths, values)`` in the order their
    segments should follow each other within a row. ``positions`` are the
    ascending row positions the part covers (``None`` for every row) and
    ``values`` is grouped in that row order. No sort is involved.
    """
    total = np.zeros(n_rows, dtype=INDEX_DTYPE)
    for positions, lengths, _ in parts:
        if positions is None:
            total += lengths
        else:
            total[positions] += lengths
    starts = np.zeros(n_rows, dtype=INDEX_DTYPE)
    np.cumsum(total[:-1], out=starts[1:])
    out = np.empty(int(total.sum()), dtype=INDEX_DTYPE)
    for positions, lengths, values in parts:
        if values.size == 0:
            continue
        rows = slice(None) if positions is None else positions
        begin = starts[rows]
        source = np.cumsum(lengths) - lengths
        out[np.repeat(begin - source, lengths) + np.arange(values.size)] = values
        starts[rows] = begin + lengths
    return total, out


class _Rescan:
    """Overflowed rows of one product tile, rescanned until every row fits."""

    __slots__ = ("q0", "tile", "over", "base", "rows", "pending", "counts", "capacity",
                 "rounds", "parts", "outputs")

    def __init__(self, q0, tile, over, base, rows, counts):
        self.q0, self.tile, self.over, self.base = q0, tile, over, base
        self.rows = rows  # query rows (indices into the call's queries)
        self.pending = np.arange(len(rows))
        self.counts = counts
        self.capacity = 0
        self.rounds = 0
        self.parts = []
        self.outputs = None


class _Query:
    """One metal_query call: tiles, waves, overflow rescans and assembly.

    A wave's results are read only after the next wave is queued. Rows that
    overflow their capacity are rescanned asynchronously and resolved while
    finishing the following wave, so overflow never drains the GPU queue.
    """

    def __init__(self, device_index, train32, queries32, thr, *, exclude_self, q_offset,
                 need_distance, cap_hint, threads, sort=False):
        import mlx.core as mx

        from mlxlearn_device._binary32 import matmul
        from mlxlearn_device._radius import compact_gram

        self.mx = mx
        self.gpu = mx.gpu
        self.matmul = matmul
        self.compact_gram = compact_gram
        # The native finisher orders every row by column itself; the NumPy
        # refinement needs rows that are already ascending.
        self.sort_rows = _native_finisher() is None
        self.train32 = train32
        self.queries32 = queries32
        self.thr = thr
        self.exclude_self = bool(exclude_self)
        self.q_offset = int(q_offset)
        self.need_distance = bool(need_distance)
        self.cap_hint = cap_hint
        self.threads = threads
        # With the native finisher, rows can be ordered by (distance, index)
        # while finishing; the NumPy path leaves that to the caller.
        self.sort = bool(sort)
        self.cut = float(thr.gram_cut)
        self.factor = float(thr.gram_factor)
        self.floor = float(thr.gram_floor)

        self.center, train, self.train_sq = _prepared(device_index)
        self.n = n = int(train.shape[0])
        self.n_query = len(queries32)
        self.width, self.block_rows, self.per_wave = _plan(n, self.n_query)
        if n <= self.width:
            self.train_tiles = [train]
        else:
            self.train_tiles = mx.split(train, list(range(self.width, n, self.width)),
                                        axis=0, stream=self.gpu)
        self.n_tiles = len(self.train_tiles)

        # q0 -> [q1, per-tile (lengths, cols) or None, tiles still missing]
        self.blocks: dict[int, list] = {}
        self.rescans: list[_Rescan] = []
        self.ready: list = []
        self.done: dict[int, tuple] = {}

    # -- device work ------------------------------------------------------------

    def _centred(self, host_rows):
        mx, gpu = self.mx, self.gpu
        queries = mx.array(host_rows, dtype=mx.float32)
        centred = mx.subtract(queries, self.center, stream=gpu)
        norms = mx.sum(mx.multiply(centred, centred, stream=gpu), axis=1, stream=gpu)
        return centred, norms

    def _screen(self, centred, norms, tile, capacity, row_offset=0):
        products = self.matmul(centred, self.train_tiles[tile], transpose_b=True,
                               stream=self.gpu)
        return self.compact_gram(
            products, norms, self.train_sq, capacity=capacity, cut=self.cut,
            factor=self.factor, floor=self.floor, row_offset=row_offset,
            col_offset=tile * self.width, stream=self.gpu,
        )

    def _tile_width(self, tile: int) -> int:
        return min(self.n, (tile + 1) * self.width) - tile * self.width

    # -- driver -----------------------------------------------------------------

    def run(self) -> RadiusCSR:
        mx, gpu = self.mx, self.gpu
        nq, rows = self.n_query, self.block_rows
        centred_all, norms_all = self._centred(self.queries32)
        starts = list(range(0, nq, rows))
        blocks = (mx.split(centred_all, starts[1:], axis=0, stream=gpu)
                  if len(starts) > 1 else [centred_all])
        del centred_all
        tiles = [(index, start, min(nq, start + rows), tile)
                 for index, start in enumerate(starts) for tile in range(self.n_tiles)]
        waves = [tiles[i:i + self.per_wave] for i in range(0, len(tiles), self.per_wave)]
        queued: deque = deque()
        for wave in waves:
            launched = []
            for index, q0, q1, tile in wave:
                if tile == 0:
                    self.blocks[q0] = [q1, [None] * self.n_tiles, self.n_tiles]
                capacity = _capacity(self.cap_hint, self._tile_width(tile), rows=q1 - q0)
                counts, packed = self._screen(blocks[index], norms_all, tile, capacity,
                                              row_offset=q0)
                launched.append((q0, tile, capacity, counts, packed))
                if tile == self.n_tiles - 1:
                    blocks[index] = None
            mx.async_eval(*[array for item in launched for array in item[3:]])
            queued.append(launched)
            if len(queued) >= _PIPELINE_WAVES:
                self._finish(queued.popleft())
        while queued:
            self._finish(queued.popleft())
        while self.rescans:
            self._resolve_rescans()
            self._refine_ready()
        debug.add(calls=1, screened_pairs=nq * self.n, waves=len(waves), tiles=len(tiles))
        return self._result()

    def _finish(self, launched) -> None:
        self.mx.eval(*[array for item in launched for array in item[3:]])
        # Rescans queued while finishing the previous wave ran right behind it.
        self._resolve_rescans()
        for q0, tile, capacity, counts, packed in launched:
            counts = np.array(counts, dtype=INDEX_DTYPE)
            lengths, fits, cols = _sorted_rows(counts, packed, capacity, self.sort_rows)
            if fits.all():
                self._complete_tile(q0, tile, lengths, cols)
            else:
                over = np.flatnonzero(~fits)
                rescan = _Rescan(q0, tile, over, (lengths, cols), q0 + over, counts[over])
                self._launch_rescan(rescan)
                self.rescans.append(rescan)
        self._refine_ready()

    # -- overflow ---------------------------------------------------------------

    def _launch_rescan(self, rescan: _Rescan) -> None:
        """Queue a rescan of the still-overflowing rows with the capacity they need."""
        width = self._tile_width(rescan.tile)
        rescan.rounds += 1
        needed = int(rescan.counts.max())
        # Powers of two keep the set of device buffer shapes small, so MLX's
        # buffer cache can reuse them across calls.
        rescan.capacity = (width if rescan.rounds > _MAX_OVERFLOW_ROUNDS
                           else max(1, min(width, 1 << max(0, needed - 1).bit_length())))
        centred, norms = self._centred(self.queries32[rescan.rows[rescan.pending]])
        rescan.outputs = self._screen(centred, norms, rescan.tile, rescan.capacity)
        self.mx.async_eval(*rescan.outputs)
        debug.add(overflow_launches=1, overflow_rows=rescan.pending.size)

    def _resolve_rescans(self) -> None:
        if not self.rescans:
            return
        self.mx.eval(*[array for rescan in self.rescans for array in rescan.outputs])
        still = []
        for rescan in self.rescans:
            counts, packed = rescan.outputs
            rescan.outputs = None
            counts = np.array(counts, dtype=INDEX_DTYPE)
            lengths, fits, cols = _sorted_rows(counts, packed, rescan.capacity,
                                               self.sort_rows)
            rescan.parts.append((rescan.pending[fits], lengths[fits], cols))
            rescan.pending, rescan.counts = rescan.pending[~fits], counts[~fits]
            if rescan.pending.size:
                # The rows' products changed shape, so their counts may too.
                self._launch_rescan(rescan)
                still.append(rescan)
                continue
            if len(rescan.parts) == 1:
                positions, lengths, cols = rescan.parts[0]
            else:
                positions = np.sort(np.concatenate([p for p, _, _ in rescan.parts]))
                lengths, cols = _merge_rows(len(positions), [
                    (np.searchsorted(positions, p), part_lengths, values)
                    for p, part_lengths, values in rescan.parts])
            base_lengths, base_cols = rescan.base
            q1 = self.blocks[rescan.q0][0]
            lengths, cols = _merge_rows(q1 - rescan.q0, [
                (None, base_lengths, base_cols), (rescan.over[positions], lengths, cols)])
            self._complete_tile(rescan.q0, rescan.tile, lengths, cols)
        self.rescans = still

    # -- assembly ---------------------------------------------------------------

    def _complete_tile(self, q0: int, tile: int, lengths, cols) -> None:
        block = self.blocks[q0]
        block[1][tile] = (lengths, cols)
        block[2] -= 1
        if block[2]:
            return
        q1, parts, _ = self.blocks.pop(q0)
        if len(parts) == 1:
            lengths, cols = parts[0]
        else:
            lengths, cols = _merge_rows(q1 - q0, [(None, part, values)
                                                  for part, values in parts])
        debug.add(candidates=cols.size)
        rows = np.repeat(np.arange(q0, q1, dtype=INDEX_DTYPE), lengths)
        if self.exclude_self and cols.size:
            own = cols == rows + self.q_offset
            if own.any():
                keep = ~own
                rows, cols = rows[keep], cols[keep]
        self.ready.append((q0, q1, rows, cols))

    def _refine_ready(self) -> None:
        if not self.ready:
            return
        ready, self.ready = self.ready, []
        if len(ready) == 1:
            rows, cols = ready[0][2], ready[0][3]
        else:
            rows = np.concatenate([item[2] for item in ready])
            cols = np.concatenate([item[3] for item in ready])
        native = _native_finisher()
        if native is not None:
            # Rows arrive grouped in query order (columns unordered within a
            # row; the finisher sorts them) with self pairs already dropped.
            # Only the flushed span of queries is passed, so a call's host work
            # stays linear in its query count.
            lo = min(item[0] for item in ready)
            hi = max(item[1] for item in ready)
            csr = native.finish_pairs(
                self.train32, self.queries32[lo:hi], rows - lo, cols, self.thr,
                presorted=True, need_distance=self.need_distance, sort=self.sort,
                threads=self.threads,
            ).to_csr(need_distance=self.need_distance, route=ROUTE)
            for q0, q1, _, _ in ready:
                a, b = int(csr.indptr[q0 - lo]), int(csr.indptr[q1 - lo])
                self.done[q0] = (
                    np.diff(csr.indptr[q0 - lo:q1 - lo + 1]), csr.indices[a:b],
                    None if csr.distances is None else csr.distances[a:b],
                )
            return
        keep, squared = refine(self.train32, self.queries32, rows, cols, self.thr,
                               need_distance=self.need_distance, threads=self.threads)
        distances = None if squared is None else np.sqrt(squared)
        start = 0
        for q0, q1, block_rows, _ in ready:
            stop = start + len(block_rows)
            kept = keep[start:stop]
            self.done[q0] = (
                np.bincount(block_rows[kept] - q0, minlength=q1 - q0),
                cols[start:stop][kept],
                None if distances is None else distances[start:stop][kept],
            )
            start = stop

    def _result(self) -> RadiusCSR:
        parts = [self.done[q0] for q0 in sorted(self.done)]
        lengths = np.concatenate([p[0] for p in parts]).astype(INDEX_DTYPE, copy=False)
        cols = np.concatenate([p[1] for p in parts]).astype(INDEX_DTYPE, copy=False)
        distances = (np.concatenate([p[2] for p in parts]) if self.need_distance else None)
        return from_grouped(lengths, cols, distances, route=ROUTE)


def _native_finisher():
    from .radius import native_engine

    return native_engine()


def metal_query(device_index, train32: np.ndarray, queries32: np.ndarray, thr: Thresholds, *,
                exclude_self: bool = False, q_offset: int = 0, need_distance: bool = True,
                cap_hint=0, threads: int | None = 1, sort: bool = False) -> RadiusCSR | None:
    """Radius neighbours of ``queries32`` through the interim Metal screen.

    Parameters
    ----------
    device_index : mlxlearn_device.DeviceNeighborIndex
        Owns the float32 training rows on the GPU. Its centred Gram arrays
        (``_center``, ``_train``, ``_train_sq``) are prepared on first use and
        shared with k-nearest-neighbour search; no other device copy is made.
    train32 : float32 (n, d)
        The same training rows on the host, for exact refinement.
    queries32 : float32 (n_query, d)
        Queries, owned float32 coordinates.
    thr : Thresholds
        From :func:`radius_contract.thresholds` for this ``d`` and ``R2``.
    exclude_self : bool
        Drop the pair ``(row, row + q_offset)`` (queries are training rows
        ``q_offset ...``); duplicates elsewhere stay.
    need_distance : bool
        Return float64 distances ``sqrt(min(s, R2))``.
    cap_hint : float
        Expected (upper estimate of) neighbours per query; sizes the first-pass
        capacity. Too small only costs a rescan of the rows that overflow.
    threads : int or None
        Host threads for refinement.

    Returns
    -------
    RadiusCSR with ``route="metal"``, rows in query order and ascending
    training indices within each row; or ``None`` when the Gram screen cannot
    be certified for this dimension (``thr.gram_cut is None``), in which case
    the caller must use another route.

    The caller routes rows outside the float32-safe range elsewhere; such rows
    stay exact here, but the screen admits all of their pairs.
    """
    if thr.gram_cut is None:
        return None
    queries32 = np.ascontiguousarray(queries32, dtype=np.float32)
    train32 = np.ascontiguousarray(train32, dtype=np.float32)
    if queries32.ndim != 2 or train32.ndim != 2 or queries32.shape[1] != train32.shape[1]:
        raise ValueError("queries and training rows must be matrices with equal widths")
    if tuple(device_index.X.shape) != train32.shape:
        raise ValueError("train32 must hold the device index's training rows")
    if len(queries32) == 0:
        return empty_result(0, need_distance=need_distance, route=ROUTE)
    with _SERIAL_LOCK if _serialize_gpu() else nullcontext():
        try:
            return _Query(device_index, train32, queries32, thr, exclude_self=exclude_self,
                          q_offset=q_offset, need_distance=need_distance, cap_hint=cap_hint,
                          threads=threads, sort=sort).run()
        finally:
            _bound_buffer_cache()


#: MLX keeps freed device buffers for reuse, but only for identical sizes; the
#: partial last query block's product tile differs between calls of different
#: sizes. Past this cache size the cache is released after a call.
_CACHE_LIMIT = 2 << 30


def _bound_buffer_cache() -> None:
    import mlx.core as mx

    get = getattr(mx, "get_cache_memory", None) or getattr(mx.metal, "get_cache_memory", None)
    clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if get is not None and clear is not None and get() > _CACHE_LIMIT:
        clear()
