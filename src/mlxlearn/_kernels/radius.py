"""Exact radius search with a host tree, bounded screening, and stock dispatch."""

from __future__ import annotations

import itertools
import math
from functools import cached_property

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.device import use_device

_F32 = np.finfo(NUMPY_FLOAT)
_F64 = np.finfo(np.float64)
_MAX_SCREEN_PAIRS = 8 * 1024 * 1024
_REFINE_ELEMENTS = 1 << 20


def _row_squares(X):
    return np.einsum("ij,ij->i", X, X, dtype=np.float64)


def pack_radius(rows, cols, squared, n_query, *, sort_results, return_distance):
    """Group pairs by query and impose a deterministic order within each row."""
    indices = np.empty(n_query, dtype=object)
    distances = np.empty(n_query, dtype=object) if return_distance else None
    if rows.size:
        keys = (cols, squared, rows) if sort_results else (cols, rows)
        order = np.lexsort(keys)
        cols = cols[order]
        if return_distance:
            sorted_dist = np.sqrt(squared[order])
    else:
        sorted_dist = np.empty(0, dtype=np.float64)
    ends = np.cumsum(np.bincount(rows, minlength=n_query))
    start = 0
    for row, stop in enumerate(ends):
        indices[row] = cols[start:stop]
        if return_distance:
            distances[row] = sorted_dist[start:stop]
        start = stop
    return (distances, indices) if return_distance else indices


class RadiusSearch:
    """Lazy radius-specific caches over the index's owned float32 coordinates."""

    def __init__(self, train):
        self.X = np.asarray(train, dtype=NUMPY_FLOAT, order="C")
        self._stock = None
        self._device_norms = None
        self._device_train = None
        self._seen = set()

    @cached_property
    def _host64(self):
        return self.X.astype(np.float64)

    @cached_property
    def _norms(self):
        return _row_squares(self.X)

    @cached_property
    def _max_norm(self):
        # A cheap bound is sufficient for dispatch and keeps the per-row
        # norm cache lazy until a screening query needs it.
        largest = max(abs(float(self.X.min())), abs(float(self.X.max())))
        return self.X.shape[1] * largest * largest

    @cached_property
    def _tree(self):
        from scipy.spatial import cKDTree

        return cKDTree(self._host64, leafsize=16)

    def _sample(self, Q, radius_sq):
        # The sample only chooses an implementation. Every route still evaluates
        # membership against the original coordinates.
        train = self.X[:: math.ceil(len(self.X) / 128)].astype(np.float64)
        query = Q[:: math.ceil(len(Q) / 8)].astype(np.float64)
        norm = _row_squares(train)
        approx = query @ train.T
        approx *= -2
        approx += norm[None, :]
        approx += _row_squares(query)[:, None]
        hits = len(self.X) * np.count_nonzero(approx <= radius_sq) / approx.size
        spread = np.sum(np.var(train, axis=0))
        return hits, spread

    def route(self, Q, radius_sq, *, fixed, tree_allowed, stock_factory, has_gpu):
        n, d = self.X.shape
        if fixed == "mlx":
            return "mlx"
        if fixed == "cpu":
            return "tree" if tree_allowed and d <= 8 else "cpu"
        if tree_allowed and d <= 8 and n <= 2048:
            return "tree"

        candidate = "cpu"
        tree_size = tree_allowed and d <= 8 and n >= 2048
        metal_size = 16 <= d <= 4096 and n >= 16_000 and n * len(Q) >= 2_000_000
        if len(Q) >= 64 and (tree_size or metal_size):
            hits, spread = self._sample(Q, radius_sq)
            if hits <= 256:
                if tree_size and (d <= 4 or radius_sq < 0.09 * spread):
                    candidate = "tree"
                elif metal_size:
                    candidate = "mlx"

        # Let the first eligible automatic query use host search. Tree building
        # and Metal setup paid for themselves on repeated batches in the grid.
        if candidate != "cpu":
            seen = candidate in self._seen
            self._seen.add(candidate)
            if (seen or stock_factory is None) and (candidate == "tree" or has_gpu()):
                return candidate
        if stock_factory is not None and self.stock_error(Q, radius_sq) is not None:
            return "sklearn"
        return "cpu"

    def stock_error(self, Q, radius_sq):
        """Bound cancellation in the stock float64 brute-force distance tiles."""
        if not 0 < radius_sq < math.inf:
            return None
        scale = self._max_norm + float(_row_squares(Q).max(initial=0))
        error = 64 * (self.X.shape[1] + 1) * _F64.eps * scale
        if error <= 1e-7 * radius_sq:
            return error
        return None

    def stock(self, Q, radius_sq, *, factory, exclude_self, sort_results, return_distance):
        error = self.stock_error(Q, radius_sq)
        assert error is not None
        if self._stock is None:
            self._stock = factory().fit(self._host64)
        # Include borderline points in the stock search, then verify them from
        # the coordinates. Near-zero distances and sort ties also need refinement.
        expanded = np.nextafter(math.sqrt(radius_sq + error), math.inf)
        distances, indices = self._stock.radius_neighbors(
            None if exclude_self else Q.astype(np.float64, copy=False),
            radius=expanded, sort_results=sort_results,
        )
        lengths = np.fromiter(map(len, distances), dtype=INDEX_DTYPE, count=len(Q))
        ends = np.cumsum(lengths)
        squared = np.concatenate(distances)
        np.square(squared, out=squared)
        refine = (np.abs(squared - radius_sq) <= 2 * error) | (squared <= 1e6 * error)
        if sort_results and squared.size > 1:
            adjacent = np.diff(squared) <= 4 * error
            boundaries = ends[:-1] - 1
            adjacent[boundaries[(boundaries >= 0) & (boundaries < adjacent.size)]] = False
            refine[:-1] |= adjacent
            refine[1:] |= adjacent
        # Most rows need no edits. Locate the affected rows in one pass over
        # the packed distances and keep the stock arrays for all other rows.
        affected = np.unique(np.searchsorted(ends, np.flatnonzero(refine), side="right"))
        for i in affected:
            row = slice(ends[i] - lengths[i], ends[i])
            ids, values, mask = indices[i], squared[row], refine[row]
            diff = self._host64[ids[mask]] - Q[i]
            values[mask] = _row_squares(diff)
            keep = values <= radius_sq
            ids, values = ids[keep], values[keep]
            if sort_results:
                order = np.lexsort((ids, values))
                ids, values = ids[order], values[order]
            indices[i], distances[i] = ids, np.sqrt(values)
        return (distances, indices) if return_distance else indices

    def tree(self, Q, radius_sq, *, exclude_self, sort_results, return_distance):
        radius = math.sqrt(radius_sq)
        extended = radius * (1 + 8 * _F64.eps * (Q.shape[1] + 1))
        matches = self._tree.query_ball_point(
            Q, extended, workers=4 if len(Q) >= 64 else 1, return_sorted=False,
        )
        lengths = np.fromiter(map(len, matches), dtype=INDEX_DTYPE, count=len(Q))
        rows = np.repeat(np.arange(len(Q), dtype=INDEX_DTYPE), lengths)
        cols = np.fromiter(itertools.chain.from_iterable(matches), dtype=INDEX_DTYPE,
                           count=int(lengths.sum()))
        return self._finish(Q, rows, cols, radius_sq, exclude_self=exclude_self,
                            sort_results=sort_results, return_distance=return_distance)

    def metal_supported(self, Q, radius_sq):
        return (self.X.shape[1] <= 4096
                and max(self._max_norm, float(_row_squares(Q).max(initial=0)), radius_sq) < float(_F32.max) / 16)

    def screen(self, Q, radius_sq, plan, *, backend, train_device, exclude_self,
               sort_results, return_distance):
        parts = (self._metal_candidates(Q, radius_sq, plan, train_device)
                 if backend == "mlx" else self._host_candidates(Q, radius_sq, plan))
        row_parts, col_parts = [], []
        for rows, cols in parts:
            if rows.size:
                row_parts.append(rows)
                col_parts.append(cols)
        empty = np.empty(0, dtype=INDEX_DTYPE)
        rows = np.concatenate(row_parts) if row_parts else empty
        cols = np.concatenate(col_parts) if col_parts else empty
        return self._finish(Q, rows, cols, radius_sq, exclude_self=exclude_self,
                            sort_results=sort_results, return_distance=return_distance)

    def _tiles(self, Q, plan):
        n = len(self.X)
        query_rows = min(plan.query_rows, 256)
        for q0 in range(0, len(Q), query_rows):
            q1 = min(len(Q), q0 + query_rows)
            train_cols = min(plan.train_cols, _MAX_SCREEN_PAIRS // (q1 - q0))
            for t0 in range(0, n, train_cols):
                yield q0, q1, t0, min(n, t0 + train_cols)

    def _host_candidates(self, Q, radius_sq, plan):
        query = Q.astype(np.float64, copy=False)
        scale = 1 - 64 * (self.X.shape[1] + 1) * _F64.eps
        query_norms = scale * _row_squares(query)
        train_norms = scale * self._norms
        for q0, q1, t0, t1 in self._tiles(query, plan):
            lower = query[q0:q1] @ self._host64[t0:t1].T
            lower *= -2
            lower += train_norms[None, t0:t1]
            lower += query_norms[q0:q1, None]
            rows, cols = np.nonzero(lower <= np.nextafter(radius_sq, math.inf))
            yield rows + q0, cols + t0

    def _metal_candidates(self, Q, radius_sq, plan, train_device):
        import mlx.core as mx

        from mlxlearn_device._binary32 import matmul
        from mlxlearn_device._radius import compact_radius

        with use_device("gpu"):
            if self._device_train is None:
                self._device_train = train_device if train_device is not None else mx.array(self.X)
            train = self._device_train
            queries = mx.array(Q)
            if self._device_norms is None:
                self._device_norms = mx.array(self._norms.astype(NUMPY_FLOAT))
            query_norms = mx.array(_row_squares(Q).astype(NUMPY_FLOAT))
            # Bound GEMM and norm error by a multiple of the sum of norms.
            # The absolute term covers binary32 flushing at very small scales.
            factor = 64 * (self.X.shape[1] + 1)
            settings = mx.array([
                float(np.nextafter(np.float32(radius_sq), np.float32(np.inf))),
                float(1 - factor * _F32.eps),
                float(factor * _F32.tiny),
            ], mx.float32)
            for q0, q1, t0, t1 in self._tiles(Q, plan):
                products = matmul(queries[q0:q1], train[t0:t1], transpose_b=True)
                capacity = min(256, t1 - t0)
                while True:
                    counts, packed = compact_radius(
                        products, self._device_norms[t0:t1], query_norms[q0:q1],
                        settings, capacity,
                    )
                    mx.eval(counts, packed)
                    lengths = np.asarray(counts, dtype=INDEX_DTYPE)
                    needed = int(lengths.max(initial=0))
                    if needed <= capacity:
                        break
                    # Reuse the evaluated products. Capacity never exceeds the
                    # width of this already memory-bounded product tile.
                    capacity = needed
                if needed:
                    present = np.arange(needed) < lengths[:, None]
                    cols = np.asarray(packed)[:, :needed][present].astype(INDEX_DTYPE) + t0
                    yield np.repeat(np.arange(q0, q1, dtype=INDEX_DTYPE), lengths), cols

    def _finish(self, Q, rows, cols, radius_sq, *, exclude_self, sort_results, return_distance):
        if exclude_self:
            keep = rows != cols
            rows, cols = rows[keep], cols[keep]
        squared = np.empty(rows.size, dtype=np.float64)
        block_size = max(1, _REFINE_ELEMENTS // self.X.shape[1])
        for start in range(0, rows.size, block_size):
            block = slice(start, start + block_size)
            diff = self.X[cols[block]].astype(np.float64)
            diff -= Q[rows[block]]
            squared[block] = _row_squares(diff)
        keep = squared <= radius_sq
        return pack_radius(rows[keep], cols[keep], squared[keep], len(Q),
                           sort_results=sort_results, return_distance=return_distance)
