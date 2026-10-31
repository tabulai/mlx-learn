"""Certified CPU nearest neighbors on owned original float32 coordinates.

Gram computations only propose candidates: direct original-coordinate bounds
certify membership, and exact dyadic arithmetic resolves ambiguous ordering.

Float32 inputs, including subnormals, convert exactly to float64. Their squared
differences lie inside float64's normal range whenever nonzero. The direct
calculation needs at most D additions plus a subtraction and multiplication per
term. Its relative error is below 16*(D+1)*eps, including rounding the interval
endpoints. The Gram calculation instead uses an absolute bound of
64*(D+1)*eps*(||q||^2 + ||x||^2): Cauchy-Schwarz bounds the absolute dot-product
sum by half the norm sum. Both constants generously cover the norm, dot, final
arithmetic, and bound construction. The supported array dimensions are far
below the D*eps regime where these first-order bounds cease to dominate gamma_D.
"""

import numpy as np
from scipy.spatial.distance import cdist

try:
    from ._neighbors_cpu_native import query as _native_query
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}._neighbors_cpu_native":
        raise
    # Source checkouts and installations without a compiler retain the same
    # certified search contract through the portable implementation below.
    _native_query = None
    _native_query_concentrated = None
    _native_integer_select = None
    _native_feature_ranges = None
else:
    from . import _neighbors_cpu_native as _native_module

    _native_integer_select = getattr(_native_module, "select_integer_gram", None)
    _native_feature_ranges = getattr(_native_module, "feature_ranges", None)
    _native_query_concentrated = getattr(_native_module, "query_concentrated", None)


def _integer_row(row):
    values = []
    for raw in row.view(np.uint32):
        raw = int(raw)
        exponent = (raw >> 23) & 255
        mantissa = (raw & 0x7fffff) | (0x800000 if exponent else 0)
        value = mantissa << max(0, exponent - 1)
        values.append(-value if raw >> 31 else value)
    return values


def _extract(work, take):
    """Stable row minima; work columns use ascending original training IDs."""
    rows = np.arange(len(work))
    ids = np.empty((len(work), take), np.intp)
    values = np.empty((len(work), take), work.dtype)
    for slot in range(take):
        current = np.argmin(work, axis=1)
        ids[:, slot] = current
        values[:, slot] = work[rows, current]
        work[rows, current] = np.inf
    return values, ids


def _extract_partition(work, take):
    """Small float64 prefixes; overlapping nonzero cutoffs are repaired later."""
    rows = np.arange(len(work))[:, None]
    ids = np.argpartition(work, take - 1, axis=1)[:, :take]
    values = work[rows, ids]
    order = np.lexsort((ids, values), axis=1)
    values, ids = values[rows, order], ids[rows, order]
    zero_rows = np.flatnonzero(values[:, -1] == 0)
    if len(zero_rows):
        # Partition may choose arbitrary references across a zero boundary.
        # Restore the original-index prefix before the direct zero shortcut.
        zero_values, zero_ids = _extract(work[zero_rows].copy(), take)
        values[zero_rows], ids[zero_rows] = zero_values, zero_ids
    work[rows, ids] = np.inf
    return values, ids


class CPUIndex:
    """Accept an independently owned, validated C-contiguous float32 matrix."""

    def __init__(self, train_owned_f32):
        self.X = train_owned_f32 if train_owned_f32.flags.aligned else train_owned_f32.copy()
        self._train64 = None
        self._norm64 = None
        self._integral = None
        self._integer_max = None
        self._norm32 = None
        self._dyadic_train = None
        self._dyadic_scale = None
        self._proposal_features = None
        self._proposal_tree = None
        self._native_order = None
        self._native_train = None
        self._native_concentrated = False
        self._sparse_integer_batch = None

    def _prepare_native_train(self, query_count):
        if self._native_order is None and query_count < 16:
            return self.X, None
        if self._native_order is None:
            self._native_order = False
            features = self.X.shape[1]
            if 256 <= len(self.X) <= 2048 and 8 < features <= 32:
                if _native_feature_ranges is None:
                    ranges = (self.X.max(axis=0).astype(np.float64)
                              - self.X.min(axis=0).astype(np.float64))
                else:
                    ranges = np.empty(features, np.float64)
                    _native_feature_ranges(self.X, ranges.reshape(1, -1))
                order = np.argsort(ranges)[::-1]
                energy = ranges * ranges
                self._native_concentrated = bool(
                    energy.sum() and energy[order[:8]].sum() >= 0.99 * energy.sum()
                )
                if self._native_concentrated and np.any(order[:8] >= 8):
                    # Put the most selective original coordinates first for
                    # the native partial-distance bound. This is a permutation,
                    # with no rounding, scaling or change to the metric; exact
                    # dyadic keys sum the same terms in either coordinate order.
                    self._native_order = order
                    self._native_train = np.ascontiguousarray(self.X[:, order])
        # A concurrent first query may still be preparing the permutation.
        # Return a matching snapshot so a caller cannot combine original train
        # columns with a permutation that another thread publishes afterward.
        train = self._native_train
        return (self.X, None) if train is None else (train, self._native_order)

    def sparse_integer_batch(self, Q=None):
        """Measured wide CPU regime; integer checks also certify exact Gram arithmetic."""
        if self._sparse_integer_batch is None:
            # A dense sample can cheaply decline the CPU optimization. A sparse
            # sample never certifies it: successful proposals still check all
            # training coordinates, the full sparsity and every query value.
            sample = self.X[:min(len(self.X), 16)]
            integral = (np.count_nonzero(sample) <= 0.75 * sample.size
                        and self._integer_query(self.X[:1]))
            self._sparse_integer_batch = bool(
                integral and np.count_nonzero(self.X) <= 0.75 * self.X.size
            )
        return self._sparse_integer_batch and (Q is None or self._integer_query(Q))

    @property
    def train64(self):
        if self._train64 is None:
            self._train64 = self.X.astype(np.float64)
        return self._train64

    def _integer_query(self, Q):
        n, features = self.X.shape
        if n > 2048 or features > 64:
            return False
        if self._integral is None:
            # A cheap rejection sample is not a certificate: successful rows
            # still require a complete training check before the exact path.
            sample = self.X[:min(n, 16)]
            # Integer bit equality cannot mistake a nonzero subnormal for a
            # truncated zero even under a caller's float32 flush-to-zero mode.
            if not np.equal(sample.view(np.uint32), np.trunc(sample).view(np.uint32)).all():
                self._integral = False
            else:
                maximum = float(np.max(np.abs(self.X)))
                self._integer_max = maximum
                self._integral = (4 * features * maximum * maximum <= 2**24
                                  and bool(np.equal(self.X.view(np.uint32),
                                                    np.trunc(self.X).view(np.uint32)).all()))
        if not self._integral:
            return False
        maximum = max(self._integer_max, float(np.max(np.abs(Q))))
        return (4 * features * maximum * maximum <= 2**24
                and bool(np.equal(Q.view(np.uint32), np.trunc(Q).view(np.uint32)).all()))

    def _exact_order(self, query, candidates, k):
        q = _integer_row(query)
        keys = []
        for index in candidates:
            point = _integer_row(self.X[index])
            squared = sum((left - right)**2 for left, right in zip(q, point, strict=True))
            keys.append((squared, int(index)))
        return np.asarray([index for _, index in sorted(keys)[:k]], np.intp)

    def _native_exact_order(self, queries, candidates, k):
        """Batch exact dyadic comparisons when signed integer sums cannot overflow."""
        if self._dyadic_scale is None:
            absolute = np.abs(self.train64)
            minimum = np.min(absolute, where=absolute != 0, initial=np.inf)
            exponent = max(-149, int(np.frexp(minimum)[1]) - 24) if np.isfinite(minimum) else 0
            scale = np.ldexp(1.0, -exponent)
            scaled = self.train64 * scale
            # Keep subtraction strictly inside int64 as well as checking the
            # squared-distance sum separately below.
            if np.max(np.abs(scaled)) <= 2**60:
                self._dyadic_train = scaled.astype(np.int64)
                self._dyadic_scale = scale
            else:
                self._dyadic_scale = False
        if self._dyadic_train is None:
            return None
        scaled = queries.astype(np.float64) * self._dyadic_scale
        if np.max(np.abs(scaled)) > 2**60 or not np.equal(scaled, np.trunc(scaled)).all():
            return None
        delta = self._dyadic_train[candidates] - scaled.astype(np.int64)[:, None, :]
        maximum = float(np.max(np.abs(delta)))
        # The half-range bound leaves ample room for rounding this certificate
        # in float64. Products and the complete positive sum are exact int64.
        if self.X.shape[1] * maximum * maximum > 2**62:
            return None
        squared = np.sum(delta * delta, axis=2, dtype=np.int64)
        order = np.lexsort((candidates, squared), axis=1)[:, :k]
        return candidates[np.arange(len(queries))[:, None], order]

    def _direct(self, Q, k, exclude_self=False, offset=0, exact=False):
        train, query = self.train64, Q.astype(np.float64, copy=False)
        work = cdist(query, train, "sqeuclidean")
        if exclude_self:
            work[np.arange(len(Q)), np.arange(offset, offset + len(Q))] = np.inf
        take = min(k + 1, len(train) - int(exclude_self))
        values, ids = (_extract_partition(work, take) if not exact and len(train) <= 256
                       else _extract(work, take))
        if not exact:
            # Original FP32 coordinates are represented exactly in float64.
            # Subtraction, squaring and D-term accumulation have a standard
            # gamma bound. Float32's entire finite range has neither overflow
            # nor subnormal squared differences in float64.
            error = 16 * (self.X.shape[1] + 1) * np.finfo(np.float64).eps
            lower = values * (1 - error)
            upper = values * (1 + error)
            ambiguous = np.any(lower[:, 1:] <= upper[:, :-1], axis=1)
            # A zero kth value certifies k identical original coordinates.
            # No nonzero float32 difference can underflow when squared in
            # float64, and _extract already chose their ascending row IDs.
            ambiguous &= values[:, k - 1] != 0
            repair = np.flatnonzero(ambiguous)
            # Most overlaps are contained in the retained prefix. Certify
            # that fact before batching their exact integer comparisons.
            if len(repair):
                cutoff = np.nextafter(upper[repair, k - 1] / (1 - error), np.inf)
                extra = np.any(work[repair] <= cutoff[:, None], axis=1)
                native_rows = repair[~extra]
                if len(native_rows):
                    selected = self._native_exact_order(Q[native_rows], ids[native_rows], k)
                    if selected is not None:
                        ids[native_rows, :k] = selected
                        delta = train[selected] - query[native_rows, None, :]
                        values[native_rows, :k] = np.sum(delta * delta, axis=2)
                        ambiguous[native_rows] = False
            for row in np.flatnonzero(ambiguous):
                if take > k and lower[row, k] <= upper[row, k - 1]:
                    # Chosen entries are infinity in work. Keep all selected
                    # IDs plus every remaining lower interval below the kth
                    # upper bound; no omitted candidate can beat this set.
                    cutoff = np.nextafter(upper[row, k - 1] / (1 - error), np.inf)
                    others = np.flatnonzero(work[row] <= cutoff)
                    candidates = np.concatenate((ids[row], others))
                else:
                    candidates = ids[row, :k]
                selected = self._exact_order(Q[row], candidates, k)
                ids[row, :k] = selected
                delta = train[selected] - query[row]
                values[row, :k] = np.sum(delta * delta, axis=1)
        return values[:, :k], ids[:, :k]

    def _integer_gram(self, Q, k, exclude_self=False, offset=0):
        if self._norm32 is None:
            self._norm32 = np.einsum("ij,ij->i", self.X, self.X)
        work = Q @ self.X.T
        query_norm = np.einsum("ij,ij->i", Q, Q)
        if _native_integer_select is not None:
            values = np.empty((len(Q), k), np.float64)
            ids = np.empty((len(Q), k), np.int64)
            _native_integer_select(
                work, self._norm32.reshape(1, -1), query_norm.reshape(-1, 1),
                k, exclude_self, offset, values, ids,
            )
            return values, ids
        work *= -2
        work += self._norm32[None, :]
        work += query_norm[:, None]
        if exclude_self:
            work[np.arange(len(Q)), np.arange(offset, offset + len(Q))] = np.inf
        values, ids = _extract(work, k)
        # Every input/product/partial sum is an integer, and the complete
        # absolute Gram expression is bounded by 4*D*max_abs**2 <=2**24.
        # Every arithmetic intermediate and the final cancellation is exact
        # in float32, including all mathematically equal distances.
        return values.astype(np.float64), ids

    def _tree_candidates(self, query, take):
        """Propose candidates for concentrated feature scales, never certify them."""
        if self._proposal_features is None:
            features = self.X.shape[1]
            self._proposal_features = False
            if 8 < features <= 64 and len(self.X) <= 2048:
                ranges = np.ptp(self.train64, axis=0)
                dimensions = np.argpartition(ranges, features - 8)[-8:]
                total = np.sum(ranges * ranges)
                if total and np.sum(ranges[dimensions] ** 2) >= 0.99 * total:
                    from scipy.spatial import cKDTree

                    # The tree only provides a small proposal. The subsequent
                    # full Gram lower bound examines every omitted reference,
                    # so tree arithmetic and projection cannot affect accuracy.
                    self._proposal_features = dimensions
                    self._proposal_tree = cKDTree(
                        np.ascontiguousarray(self.train64[:, dimensions]), leafsize=8,
                    )
        if self._proposal_tree is None:
            return None
        return self._proposal_tree.query(
            np.ascontiguousarray(query[:, self._proposal_features]), k=take, workers=1,
        )[1]

    def _gram(self, Q, k, exclude_self=False, offset=0):
        train, query = self.train64, Q.astype(np.float64, copy=False)
        if self._norm64 is None:
            self._norm64 = np.einsum("ij,ij->i", train, train)
        qnorm = np.einsum("ij,ij->i", query, query)
        lower = query @ train.T
        lower *= -2
        # The deliberately generous dot/norm/final-add envelope bounds every
        # original-coordinate squared distance. It is a full omitted-point
        # certificate, not a heuristic assertion about candidate membership.
        factor = 64 * (self.X.shape[1] + 1) * np.finfo(np.float64).eps
        # Scale the two norm vectors before broadcasting; this has the same
        # conservative envelope while avoiding full-size norm temporaries.
        lower += ((1 - factor) * self._norm64)[None, :]
        lower += ((1 - factor) * qnorm)[:, None]
        np.maximum(lower, 0, out=lower)
        if exclude_self:
            lower[np.arange(len(Q)), np.arange(offset, offset + len(Q))] = np.inf
        take = min(k + 2, len(train) - int(exclude_self))
        candidates = self._tree_candidates(query, take) if not exclude_self and take < len(train) else None
        if candidates is not None:
            lower[np.arange(len(Q))[:, None], candidates] = np.inf
            barrier = np.min(lower, axis=1)
        elif take < len(train):
            partition = np.argpartition(lower, take, axis=1)[:, :take + 1]
            candidates = partition[:, :take]
            barrier = lower[np.arange(len(Q)), partition[:, take]]
        else:
            candidates = np.broadcast_to(np.arange(len(train)), lower.shape)
            barrier = np.full(len(Q), np.inf)
        delta = train[candidates] - query[:, None, :]
        values = np.sum(delta * delta, axis=2)
        order = np.lexsort((candidates, values), axis=1)
        rows = np.arange(len(Q))[:, None]
        values, candidates = values[rows, order], candidates[rows, order]
        error = 16 * (self.X.shape[1] + 1) * np.finfo(np.float64).eps
        upper = values[:, k - 1] * (1 + error)
        uncertain = ~(barrier > upper)
        # When all references were selected, no omitted candidate exists.
        if take == len(train) - int(exclude_self):
            uncertain[:] = False
        ambiguous = np.any(values[:, 1:] * (1 - error) <= values[:, :-1] * (1 + error), axis=1)
        for row in np.flatnonzero(ambiguous & ~uncertain):
            selected = self._exact_order(Q[row], candidates[row], k)
            candidates[row, :k] = selected
            delta = train[selected] - query[row]
            values[row, :k] = np.sum(delta * delta, axis=1)
        for row in np.flatnonzero(uncertain):
            sq, ids = self._direct(Q[row:row + 1], k, exclude_self, offset + row)
            candidates[row, :k], values[row, :k] = ids[0], sq[0]
        return values[:, :k], candidates[:, :k]

    def _tiled(self, Q, k, train_block_size, exclude_self=False, offset=0):
        """Honor explicit reference tiling with certified local top-k merges."""
        values = np.empty((len(Q), 0), np.float64)
        candidates = np.empty((len(Q), 0), np.intp)
        rows = np.arange(len(Q))[:, None]
        error = 16 * (self.X.shape[1] + 1) * np.finfo(np.float64).eps
        for start in range(0, len(self.X), train_block_size):
            tile = CPUIndex(self.X[start:start + train_block_size])
            # One extra local neighbor suffices if the identity is removed.
            count = min(k + int(exclude_self), len(tile.X))
            local_values, local_ids = tile._direct(Q, count)
            local_ids += start
            if exclude_self:
                local_values[local_ids == np.arange(offset, offset + len(Q))[:, None]] = np.inf
            values = np.concatenate((values, local_values), axis=1)
            candidates = np.concatenate((candidates, local_ids), axis=1)
            order = np.lexsort((candidates, values), axis=1)
            values, candidates = values[rows, order], candidates[rows, order]
            take = min(k, values.shape[1])
            checked = values[:, :min(take + 1, values.shape[1])]
            ambiguous = np.any(
                checked[:, 1:] * (1 - error) <= checked[:, :-1] * (1 + error), axis=1,
            )
            for row in np.flatnonzero(ambiguous):
                finite = np.isfinite(values[row])
                selected = self._exact_order(Q[row], candidates[row, finite], take)
                count = len(selected)
                candidates[row, :count] = selected
                delta = self.train64[selected] - Q[row].astype(np.float64)
                values[row, :count] = np.sum(delta * delta, axis=1)
                # A tile containing only the excluded identity can leave an
                # empty slot until later references arrive.
                values[row, count:take] = np.inf
            values, candidates = values[:, :take], candidates[:, :take]
        return values, candidates

    def query(self, Q, k, *, return_distance=True, exclude_self=False,
              query_block_size=None, train_block_size=None, method="auto", _integer_exact=None):
        """Return float64 distances and int64 IDs; host inputs already validated."""
        Q = np.asarray(Q, dtype=np.float32, order="C")
        if not Q.flags.aligned:
            Q = Q.copy()
        if (_native_query is not None and method in {"auto", "direct"}
                and k <= 32 and self.X.shape[1] <= 32 and len(self.X) <= 16384
                and (len(self.X) <= 2048 or len(Q) <= 4)
                and (train_block_size is None or train_block_size >= len(self.X))):
            distances = np.empty((len(Q), k), np.float64) if return_distance else None
            indices = np.empty((len(Q), k), np.int64)
            rows = min(len(Q), query_block_size or 256)
            train, order = self._prepare_native_train(len(Q))
            # Keep the measured cutoff optimization on completed concentrated
            # snapshots. Older extensions retain their seven-argument query.
            native_query = (_native_query_concentrated
                            if order is not None and _native_query_concentrated is not None
                            else _native_query)
            for start in range(0, len(Q), rows or 1):
                block = Q[start:start + rows]
                if order is not None:
                    block = np.ascontiguousarray(block[:, order])
                native_query(
                    train, block, k, exclude_self, start,
                    None if distances is None else distances[start:start + rows],
                    indices[start:start + rows],
                )
            return distances, indices
        exact = (self._integer_query(Q) if _integer_exact is None else _integer_exact) if len(Q) else False
        rows = min(len(Q), query_block_size or 256)
        values, indices = [], []
        for start in range(0, len(Q), rows or 1):
            block = Q[start:start + rows]
            if train_block_size is not None and train_block_size < len(self.X):
                result = self._tiled(block, k, train_block_size, exclude_self, start)
            elif exact and len(Q) >= 16 and self.X.shape[1] >= 32 and len(self.X) >= 512:
                result = self._integer_gram(block, k, exclude_self, start)
            elif method == "gram" or (method == "auto" and len(Q) >= 16 and len(self.X) >= 256):
                result = self._gram(block, k, exclude_self, start)
            else:
                result = self._direct(block, k, exclude_self, start, exact=exact)
            values.append(result[0])
            indices.append(result[1])
        squared = (values[0] if len(values) == 1 else np.concatenate(values)) if values else np.empty((0, k), np.float64)
        ids = (indices[0] if len(indices) == 1 else np.concatenate(indices)) if indices else np.empty((0, k), np.intp)
        return (np.sqrt(squared) if return_distance else None), ids
