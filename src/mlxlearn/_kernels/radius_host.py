"""NumPy/SciPy radius engines: canonical host distances, a tree and brute force.

``host_distances`` is the single float64 distance function of a NumPy
installation: gather, subtract and reduce with ``einsum`` over C-contiguous
blocks, which gives the same per-pair bits at every block size. Every host
route classifies its candidates with :mod:`radius_contract`, so membership is
exact and identical across routes.
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .radius_contract import Thresholds, classify_f64, resolve_band
from .radius_result import RadiusCSR, from_grouped

__all__ = ["brute_host", "canonical_distances", "host_distances", "refine", "tree_query"]

_BLOCK_ELEMENTS = 1 << 20
_THREAD_ELEMENTS = 1 << 22
#: Full-width screening tiles hold at most this many float64 products.
_SCREEN_ELEMENTS = 1 << 23
_U64 = 2.0**-53


def _host_threads(threads: int | None) -> int:
    if threads is None or threads <= 0:
        import os

        threads = os.cpu_count() or 1
    return max(1, min(int(threads), 8))


def host_distances(train: np.ndarray, queries: np.ndarray, rows: np.ndarray,
                   cols: np.ndarray, *, threads: int | None = 1) -> np.ndarray:
    """Float64 squared distances of the float32 rows ``queries[rows]`` and ``train[cols]``."""
    count = len(rows)
    out = np.empty(count, dtype=np.float64)
    if count == 0:
        return out
    d = train.shape[1]
    block = max(1, _BLOCK_ELEMENTS // max(d, 1))

    def work(start):
        stop = min(count, start + block)
        diff = train[cols[start:stop]].astype(np.float64)
        diff -= queries[rows[start:stop]]
        out[start:stop] = np.einsum("ij,ij->i", diff, diff)

    starts = range(0, count, block)
    workers = _host_threads(threads)
    if workers > 1 and count * d >= _THREAD_ELEMENTS and len(starts) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(starts))) as pool:
            list(pool.map(work, starts))
    else:
        for start in starts:
            work(start)
    return out


def canonical_distances(train: np.ndarray, queries: np.ndarray, rows: np.ndarray,
                        cols: np.ndarray, *, threads: int | None = 1) -> np.ndarray:
    """The installation's canonical float64 squared distances.

    Native installations sum each pair's squared differences strictly in
    feature order; NumPy-only installations use :func:`host_distances`. Every
    route of one installation returns distances from the same function, so
    they agree bitwise; across installations they can differ in the last bit.
    """
    from .radius import native_engine

    native = native_engine()
    if native is not None:
        return native.canonical_distances(train, queries, np.asarray(rows, np.int64),
                                          np.asarray(cols, np.int64), threads=threads)
    return host_distances(train, queries, rows, cols, threads=threads)


def refine(train: np.ndarray, queries: np.ndarray, rows: np.ndarray, cols: np.ndarray,
           thr: Thresholds, *, need_distance: bool = True,
           threads: int | None = 1) -> tuple[np.ndarray, np.ndarray | None]:
    """Decide candidate pairs exactly; return the keep mask and clamped squared distances."""
    if thr.everything:
        keep = np.ones(len(rows), dtype=bool)
        if not need_distance:
            return keep, None
        return keep, canonical_distances(train, queries, rows, cols, threads=threads)
    # The installation's canonical function, so every route of one
    # installation returns bitwise-identical distances.
    squared = canonical_distances(train, queries, rows, cols, threads=threads)
    keep, uncertain = classify_f64(squared, thr)
    if uncertain.any():
        band = np.flatnonzero(uncertain)
        keep[band] = resolve_band(train, queries, rows[band], cols[band], thr.R2)
    if not need_distance:
        return keep, None
    np.minimum(squared, thr.R2, out=squared)
    return keep, squared


def _finish_grouped(train, queries, rows, cols, lengths, thr, *, need_distance, threads,
                    route):
    """Refine candidates already grouped by row with ascending columns."""
    keep, squared = refine(train, queries, rows, cols, thr,
                           need_distance=need_distance, threads=threads)
    if keep.all():
        kept_lengths = lengths
    else:
        kept_lengths = np.bincount(rows[keep], minlength=len(lengths)).astype(INDEX_DTYPE)
        cols = cols[keep]
        if squared is not None:
            squared = squared[keep]
    distances = None if squared is None else np.sqrt(squared)
    return from_grouped(kept_lengths, cols, distances, route=route)


# ----------------------------------------------------------------------------------
# tree (SciPy cKDTree; the native tree replaces it when the extension is present)
# ----------------------------------------------------------------------------------


def build_tree(train: np.ndarray):
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(train, dtype=np.float64), leafsize=16)


def tree_query(tree, train: np.ndarray, queries: np.ndarray, thr: Thresholds, *,
               exclude_self: bool, q_offset: int = 0, need_distance: bool,
               threads: int | None = None) -> RadiusCSR:
    """Candidates from the tree with a widened radius, then exact refinement."""
    nq, d = queries.shape
    if thr.everything:
        extended = math.inf
    else:
        # The tree compares float64 sums of squares with its own rounding;
        # widening by the prune margin and a further relative slack makes it
        # return a superset of the members.
        extended = math.nextafter(
            math.sqrt(thr.prune_above) * (1.0 + 8.0 * (d + 1) * _U64), math.inf)
    workers = -1 if threads is None or threads <= 0 else int(threads)
    matches = tree.query_ball_point(
        np.asarray(queries, dtype=np.float64), extended,
        workers=workers if nq >= 64 else 1, return_sorted=True,
    )
    lengths = np.fromiter(map(len, matches), dtype=INDEX_DTYPE, count=nq)
    total = int(lengths.sum())
    cols = np.fromiter(itertools.chain.from_iterable(matches), dtype=INDEX_DTYPE, count=total)
    del matches
    rows = np.repeat(np.arange(nq, dtype=INDEX_DTYPE), lengths)
    if exclude_self:
        own = rows + q_offset != cols
        if not own.all():
            rows, cols = rows[own], cols[own]
            lengths = np.bincount(rows, minlength=nq).astype(INDEX_DTYPE)
    return _finish_grouped(train, queries, rows, cols, lengths, thr,
                           need_distance=need_distance, threads=threads, route="tree")


# ----------------------------------------------------------------------------------
# brute force (centred augmented float64 GEMM screen)
# ----------------------------------------------------------------------------------


class BruteState:
    """The training mean for the host screen, computed once per index.

    No float64 copy of the training rows is kept: each call centres training
    tiles on the fly, which costs O(n d) next to the screen's O(n m d).
    """

    def __init__(self, train: np.ndarray):
        X = np.asarray(train)
        d = X.shape[1]
        self.center = (X.mean(axis=0, dtype=np.float64) if len(X)
                       else np.zeros(d, dtype=np.float64))
        # The relative allowance covers the augmented GEMM (d + 1 terms), the
        # norm sums, the centring of both operands and the final subtractions,
        # with a wide margin; the floor covers subnormal products.
        self.factor = (64 * (d + 1) + 16) * _U64
        self.floor = d * 2.0**-1060


def brute_host(state: BruteState, train: np.ndarray, queries: np.ndarray, thr: Thresholds, *,
               exclude_self: bool, q_offset: int = 0, need_distance: bool,
               sort: bool = False, threads: int | None = 1) -> RadiusCSR:
    """Screen all pairs with a certified float64 Gram lower bound, then refine.

    Candidates are finished by the native engine when it is built (which can
    also order rows by distance); otherwise by :func:`refine`, and ``sort`` is
    left to the caller.
    """
    nq, d = queries.shape
    n = len(train)
    if thr.everything:
        rows = np.repeat(np.arange(nq, dtype=INDEX_DTYPE), n)
        cols = np.tile(np.arange(n, dtype=INDEX_DTYPE), nq)
        lengths = np.full(nq, n, dtype=INDEX_DTYPE)
        if exclude_self:
            own = rows + q_offset != cols
            rows, cols = rows[own], cols[own]
            lengths = np.bincount(rows, minlength=nq).astype(INDEX_DTYPE)
        return _finish_grouped(train, queries, rows, cols, lengths, thr,
                               need_distance=need_distance, threads=threads, route="brute")
    centred = np.asarray(queries, dtype=np.float64) - state.center
    q_norms = np.einsum("ij,ij->i", centred, centred)
    query_aug = np.ones((nq, d + 1), dtype=np.float64)
    query_aug[:, :d] = centred
    # Candidate iff (|q|^2 + |x|^2)(1 - f) - 2 q.x - floor <= R2, i.e.
    # q.x - |x|^2 (1 - f) / 2 >= (|q|^2 (1 - f) - R2 - floor) / 2. Lower the
    # per-row cut by more than its own rounding error, which is relative to
    # the operands rather than to the (possibly cancelled) result.
    cut = 0.5 * (q_norms * (1.0 - state.factor) - thr.R2 - state.floor)
    cut -= 4 * _U64 * (q_norms + thr.R2 + state.floor)
    # Tiles over training rows (centred on the fly) and query rows, each
    # product tile holding at most _SCREEN_ELEMENTS float64 values.
    train_rows = max(1, min(n, (1 << 22) // max(d, 1), _SCREEN_ELEMENTS // max(1, min(nq, 4096))))
    query_rows = max(1, _SCREEN_ELEMENTS // train_rows)
    row_parts, col_parts = [], []
    for t0 in range(0, n, train_rows):
        t1 = min(n, t0 + train_rows)
        block = np.asarray(train[t0:t1], dtype=np.float64) - state.center
        augmented = np.empty((t1 - t0, d + 1), dtype=np.float64)
        augmented[:, :d] = block
        augmented[:, d] = -0.5 * (1.0 - state.factor) * np.einsum("ij,ij->i", block, block)
        width = t1 - t0
        for q0 in range(0, nq, query_rows):
            q1 = min(nq, q0 + query_rows)
            gram = query_aug[q0:q1] @ augmented.T
            flat = np.flatnonzero(gram >= cut[q0:q1, None])
            if flat.size:
                local_rows, local_cols = np.divmod(flat, width)
                row_parts.append(local_rows + q0)
                col_parts.append(local_cols + t0)
    if row_parts:
        rows = np.concatenate(row_parts).astype(INDEX_DTYPE, copy=False)
        cols = np.concatenate(col_parts).astype(INDEX_DTYPE, copy=False)
        if len(row_parts) > 1 and n > train_rows:
            # Tiles over training rows interleave each query's columns; order
            # rows by (query, column) once.
            order = np.argsort(rows * np.int64(n) + cols, kind="stable")
            rows, cols = rows[order], cols[order]
    else:
        rows = cols = np.empty(0, dtype=INDEX_DTYPE)
    if exclude_self and rows.size:
        own = rows + q_offset != cols
        rows, cols = rows[own], cols[own]
    from .radius import native_engine

    native = native_engine()
    if native is not None:
        return native.finish_pairs(train, queries, rows, cols, thr, presorted=True,
                                   need_distance=need_distance, sort=sort,
                                   threads=threads).to_csr(need_distance=need_distance,
                                                           route="brute")
    lengths = np.bincount(rows, minlength=nq).astype(INDEX_DTYPE)
    return _finish_grouped(train, queries, rows, cols, lengths, thr,
                           need_distance=need_distance, threads=threads, route="brute")


def as_float32(values) -> np.ndarray:
    return np.asarray(values, dtype=NUMPY_FLOAT, order="C")
