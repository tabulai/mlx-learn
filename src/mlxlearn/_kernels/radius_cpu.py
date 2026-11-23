"""Optional native radius engines: KD-tree, brute force, candidate finisher, reductions.

The extension ``_radius_cpu_native`` is built when a C++ compiler is available.
When it is missing, :data:`AVAILABLE` is False and callers keep using the NumPy
engines in :mod:`radius_host`; nothing else here may be called.

Every engine honours the contract in :mod:`radius_contract`:

* membership is exactly ``S(q, x) <= R2`` for the float32 coordinates: the
  canonical float64 value (a strictly sequential sum over features of
  ``(double(q_k) - double(x_k))**2``) is classified against the thresholds and
  the uncertain band is resolved with exact integer arithmetic;
* distances are ``sqrt(min(canonical d2, R2))`` in float64, so the tree, brute
  and finisher routes return bitwise-identical values for any thread count;
* rows ascend by training index, or by ``(distance, index)`` with ``sort``;
* ``exclude_self`` drops only the query's own index ``q_offset + row``.

Engines return a :class:`NativeResult` that owns its data; it converts to a
:class:`~.radius_result.RadiusCSR` or feeds fused vote and mean reductions that
match :mod:`radius_reduce` bitwise, without building the CSR.
"""

from __future__ import annotations

import importlib
import os

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .radius_contract import Thresholds
from .radius_result import RadiusCSR

try:
    # import_module (not "from . import") raises ModuleNotFoundError naming the
    # extension when it is absent, so AVAILABLE becomes False instead of the
    # whole wrapper failing to import.
    _native = importlib.import_module(f"{__package__}._radius_cpu_native")
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the build
    if exc.name != f"{__package__}._radius_cpu_native":
        raise
    _native = None

__all__ = [
    "AVAILABLE",
    "NativeResult",
    "NativeTree",
    "brute_query",
    "build_tree",
    "canonical_distances",
    "cpu_count",
    "default_leaf_size",
    "finish_pairs",
    "resolve_threads",
    "tree_pilot",
    "tree_query",
    "tree_query_csr",
]

AVAILABLE = _native is not None

#: Mode bits of the native entry points.
NEED_DISTANCE = 1
SORT = 2
INTEGER_EXACT = 4
COUNT_ONLY = 8
TWO_PASS = 16
_FAIL_TEST = 1 << 20  # test hook: the last task raises
_NO_SCREEN = 1 << 21  # disable the float32 pre-screen (same output; for A/B runs)


# ----------------------------------------------------------------------------------
# thread policy
# ----------------------------------------------------------------------------------


def cpu_count() -> int:
    """Logical CPUs of this machine (P and E cores; GCD balances them)."""
    return os.cpu_count() or 1


def resolve_threads(n_jobs=None) -> int:
    """Worker threads for the native radius engines.

    ``Tuning.radius_threads`` when it is positive, else ``n_jobs`` when it is at
    least 1, else every core: ``None`` and ``-1`` both mean all cores, and
    ``n_jobs <= -2`` follows joblib (``cpu_count() + 1 + n_jobs``, at least 1).
    Workers pull tasks from a shared counter, so P and E cores stay balanced;
    small calls run on the calling thread regardless.
    """
    from .._common.config import get_tuning

    configured = int(getattr(get_tuning(), "radius_threads", 0) or 0)
    if configured > 0:
        return configured
    if n_jobs is not None:
        n_jobs = int(n_jobs)
        if n_jobs >= 1:
            return n_jobs
        if n_jobs <= -2:
            return max(1, cpu_count() + 1 + n_jobs)
    return cpu_count()


def _threads(threads) -> int:
    return resolve_threads(None) if threads is None else max(1, int(threads))


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------


def _require():
    if _native is None:
        raise RuntimeError("the native radius extension is not built")
    return _native


def _f32(values) -> np.ndarray:
    # The extension requires C-contiguous *and* aligned buffers; np.asarray
    # alone keeps misaligned views (np.frombuffer offsets, memmaps, records).
    array = np.require(values, dtype=NUMPY_FLOAT, requirements=("C", "A"))
    if array.ndim != 2:
        raise ValueError("expected a 2-d array")
    return array


def _index(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.int32:
        array = np.asarray(array, dtype=INDEX_DTYPE)
    return np.require(array, requirements=("C", "A"))


def threshold_tuple(thr: Thresholds) -> tuple[float, float, float, float, float]:
    """The ``(R2, f64_in, f64_out, prune_above, bulk_below)`` the extension takes."""
    return (float(thr.R2), float(thr.f64_in), float(thr.f64_out), float(thr.prune_above),
            float(thr.bulk_below))


def _mode(thr: Thresholds, *, need_distance: bool, sort: bool) -> int:
    mode = 0
    if need_distance or sort:
        mode |= NEED_DISTANCE
    if sort:
        mode |= SORT
    if thr.integer_exact:
        mode |= INTEGER_EXACT
    return mode


# ----------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------


class NativeResult:
    """Neighborhoods of one query batch held by the extension.

    The result owns its data (per-task int32 columns and float64 squared
    distances), so the arrays it was computed from may be freed. It is
    immutable and may be read from several threads.
    """

    __slots__ = ("_capsule", "n_query", "n_train", "nnz", "mode", "R2", "_threads")

    def __init__(self, capsule, threads: int):
        self._capsule = capsule
        n_query, n_train, nnz, mode, R2 = _native.result_info(capsule)
        self.n_query = n_query
        self.n_train = n_train
        self.nnz = nnz
        self.mode = mode
        self.R2 = R2
        self._threads = threads

    @property
    def has_distances(self) -> bool:
        return bool(self.mode & NEED_DISTANCE) and not self.mode & COUNT_ONLY

    @property
    def sorted_by_distance(self) -> bool:
        return bool(self.mode & SORT)

    @property
    def count_only(self) -> bool:
        return bool(self.mode & COUNT_ONLY)

    def sizes(self) -> tuple[int, int]:
        """``(n_query, nnz)``."""
        return _native.result_sizes(self._capsule)

    def memory_bytes(self) -> int:
        """Bytes the extension holds for this result (buffers at capacity)."""
        return _native.result_memory(self._capsule)

    def counts(self) -> np.ndarray:
        """Neighbors per query row (int64)."""
        counts = np.empty(self.n_query, dtype=INDEX_DTYPE)
        _native.result_counts(self._capsule, counts)
        return counts

    def to_csr(self, need_distance: bool | None = None, route: str = "") -> RadiusCSR:
        """Copy into a :class:`RadiusCSR` (int64 indices, float64 distances)."""
        if need_distance is None:
            need_distance = self.has_distances
        if self.count_only:
            raise ValueError("a count-only result has no neighbor columns")
        if need_distance and not self.has_distances:
            raise ValueError("the result was computed without distances")
        indptr = np.empty(self.n_query + 1, dtype=INDEX_DTYPE)
        indices = np.empty(self.nnz, dtype=INDEX_DTYPE)
        distances = np.empty(self.nnz, dtype=np.float64) if need_distance else None
        _native.result_fill_csr(self._capsule, indptr, indices, distances, self._threads)
        return RadiusCSR(indptr, indices, distances, route)

    def votes(self, codes, n_classes: int, weighted: bool) -> tuple[np.ndarray, np.ndarray]:
        """Class votes ``float64[n_query, n_classes]`` and neighbor counts.

        Bitwise equal to :func:`radius_reduce.reduce_votes_csr` on ``to_csr()``.
        """
        codes = _index(codes)
        votes = np.empty((self.n_query, int(n_classes)), dtype=np.float64)
        counts = np.empty(self.n_query, dtype=INDEX_DTYPE)
        _native.result_fill_votes(self._capsule, codes, int(n_classes), bool(weighted), votes,
                                  counts, self._threads)
        return votes, counts

    def mean(self, targets, weighted: bool) -> tuple[np.ndarray, np.ndarray]:
        """Per-query (weighted) mean of single-output targets; empty rows are NaN.

        Bitwise equal to :func:`radius_reduce.reduce_mean_csr` on ``to_csr()``.
        """
        targets = np.ascontiguousarray(targets, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError("targets must be one-dimensional")
        pred = np.empty(self.n_query, dtype=np.float64)
        counts = np.empty(self.n_query, dtype=INDEX_DTYPE)
        _native.result_fill_mean(self._capsule, targets, bool(weighted), pred, counts,
                                 self._threads)
        return pred, counts


# ----------------------------------------------------------------------------------
# tree
# ----------------------------------------------------------------------------------


def default_leaf_size(d: int) -> int:
    """Points per leaf: 16 up to four features, 32 above."""
    return 16 if d <= 4 else 32


class NativeTree:
    """An immutable native KD-tree over float32 rows (safe to query concurrently).

    It keeps its own copy of the points in tree order, so the training array
    may be released. It is never pickled; rebuild it after unpickling.
    """

    __slots__ = ("_capsule", "n", "d", "leaf_size", "depth", "n_nodes", "__weakref__")

    def __init__(self, capsule):
        self._capsule = capsule
        self.n, self.d, self.leaf_size, self.depth, self.n_nodes = _native.tree_info(capsule)

    def __reduce__(self):
        raise TypeError("native radius trees cannot be pickled; rebuild them after loading")

    def export(self) -> dict[str, np.ndarray]:
        """Copies of the permutation, the points in tree order and the node boxes."""
        perm = np.empty(self.n, dtype=np.int32)
        points = np.empty((self.n, self.d), dtype=np.float32)
        lo = np.empty((self.n_nodes, self.d), dtype=np.float32)
        hi = np.empty((self.n_nodes, self.d), dtype=np.float32)
        _native.tree_export(self._capsule, perm, points, lo, hi)
        return {"perm": perm, "points": points, "lo": lo, "hi": hi}


def build_tree(train32, leaf_size: int = 0, threads: int | None = None) -> NativeTree:
    """Build the native KD-tree (deterministic: independent of ``threads``)."""
    native = _require()
    train32 = _f32(train32)
    if leaf_size <= 0:
        leaf_size = default_leaf_size(train32.shape[1])
    return NativeTree(native.build_tree(train32, int(leaf_size), _threads(threads)))


def _check_tree(tree: NativeTree, train32, queries32) -> np.ndarray:
    queries32 = _f32(queries32)
    if train32 is not None:
        shape = np.shape(train32)
        if tuple(shape) != (tree.n, tree.d):
            raise ValueError(f"the tree was built on {(tree.n, tree.d)} rows, got {tuple(shape)}")
    return queries32


def tree_query(tree: NativeTree, train32, queries32, thr: Thresholds, *,
               exclude_self: bool = False, q_offset: int = 0, need_distance: bool = True,
               sort: bool = False, threads: int | None = None,
               count_only: bool = False, screen: bool = True) -> NativeResult:
    """Exact radius neighbors of ``queries32`` from the tree.

    ``train32`` is only checked against the tree's shape (the tree holds its own
    copy). Self-queries (``exclude_self``) run in tree order for locality.
    """
    native = _require()
    queries32 = _check_tree(tree, train32, queries32)
    workers = _threads(threads)
    mode = _mode(thr, need_distance=need_distance, sort=sort)
    if count_only:
        mode = (mode & ~(NEED_DISTANCE | SORT)) | COUNT_ONLY
    if not screen:
        mode |= _NO_SCREEN
    capsule = native.tree_query(tree._capsule, queries32, threshold_tuple(thr), int(q_offset),
                                bool(exclude_self), mode, workers)
    return NativeResult(capsule, workers)


def tree_query_csr(tree: NativeTree, train32, queries32, thr: Thresholds, *,
                   exclude_self: bool = False, q_offset: int = 0, need_distance: bool = True,
                   sort: bool = False, threads: int | None = None, route: str = "tree",
                   screen: bool = True) -> RadiusCSR:
    """Two-pass mode for dense outputs: count, then fill the exact-size CSR directly.

    Peak memory is the CSR itself plus one row of scratch per task; the tree is
    traversed twice.
    """
    native = _require()
    queries32 = _check_tree(tree, train32, queries32)
    workers = _threads(threads)
    counted = tree_query(tree, None, queries32, thr, exclude_self=exclude_self,
                         q_offset=q_offset, threads=workers, count_only=True, screen=screen)
    counts = counted.counts()
    del counted
    indptr = np.zeros(len(queries32) + 1, dtype=INDEX_DTYPE)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=INDEX_DTYPE)
    need = need_distance or sort
    distances = np.empty(nnz, dtype=np.float64) if need else None
    mode = _mode(thr, need_distance=need, sort=sort) | TWO_PASS | (0 if screen else _NO_SCREEN)
    native.tree_fill(tree._capsule, queries32, threshold_tuple(thr), int(q_offset),
                     bool(exclude_self), mode, workers, indptr, indices, distances)
    return RadiusCSR(indptr, indices, distances, route)


def tree_pilot(tree: NativeTree, queries32, R2: float, visit_cap: int,
               threads: int | None = 1) -> np.ndarray:
    """Per query: visited nodes, visited points and hits (``s <= R2``), for routing.

    A query stops after ``visit_cap`` points; bulk-included nodes count all
    their points as visited and as hits. Deterministic.
    """
    native = _require()
    queries32 = _check_tree(tree, None, queries32)
    out = np.empty((len(queries32), 3), dtype=INDEX_DTYPE)
    native.tree_pilot(tree._capsule, queries32, float(R2), int(visit_cap), out, _threads(threads))
    return out


# ----------------------------------------------------------------------------------
# brute force and the candidate finisher
# ----------------------------------------------------------------------------------


def brute_query(train32, queries32, thr: Thresholds, *, exclude_self: bool = False,
                q_offset: int = 0, need_distance: bool = True, sort: bool = False,
                threads: int | None = None, screen: bool = True) -> NativeResult:
    """All pairs, canonical distances and exact classification (small and latency calls).

    Threads are used only above ``n * m * d > 2**20``; with fewer than
    ``4 * threads`` queries the training rows are split into blocks instead.
    """
    native = _require()
    train32 = _f32(train32)
    queries32 = _f32(queries32)
    workers = _threads(threads)
    mode = _mode(thr, need_distance=need_distance, sort=sort) | (0 if screen else _NO_SCREEN)
    capsule = native.brute_query(train32, queries32, threshold_tuple(thr), int(q_offset),
                                 bool(exclude_self), mode, workers)
    return NativeResult(capsule, workers)


def finish_pairs(train32, queries32, rows, cols, thr: Thresholds, *, exclude_self: bool = False,
                 q_offset: int = 0, presorted: bool = False, need_distance: bool = True,
                 sort: bool = False, threads: int | None = None) -> NativeResult:
    """Decide candidate pairs ``(queries32[rows[i]], train32[cols[i]])`` exactly.

    Pairs may come in any order (a parallel counting sort groups them by row)
    or, with ``presorted``, grouped by non-decreasing row. Each pair must be
    listed once; non-members and the self pair are dropped.
    """
    native = _require()
    train32 = _f32(train32)
    queries32 = _f32(queries32)
    rows = _index(rows)
    cols = _index(cols)
    if rows.dtype != cols.dtype:
        rows = rows.astype(INDEX_DTYPE)
        cols = cols.astype(INDEX_DTYPE)
    workers = _threads(threads)
    capsule = native.finish_pairs(train32, queries32, rows, cols, threshold_tuple(thr),
                                  int(q_offset), bool(exclude_self),
                                  _mode(thr, need_distance=need_distance, sort=sort), workers,
                                  bool(presorted))
    return NativeResult(capsule, workers)


def canonical_distances(train32, queries32, rows, cols, *, threads: int | None = None) -> np.ndarray:
    """Canonical float64 squared distances of the given pairs (unclamped)."""
    native = _require()
    train32 = _f32(train32)
    queries32 = _f32(queries32)
    rows = _index(rows)
    cols = _index(cols)
    if rows.dtype != cols.dtype:
        rows = rows.astype(INDEX_DTYPE)
        cols = cols.astype(INDEX_DTYPE)
    out = np.empty(len(rows), dtype=np.float64)
    native.canonical_distances(train32, queries32, rows, cols, out, _threads(threads))
    return out
