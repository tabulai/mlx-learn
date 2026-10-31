"""Experimental certified CPU search on owned original float32 coordinates.

No GPU work or sklearn estimators are used. This module is a performance and
correctness prototype, not production dispatch.
"""

import numpy as np
from scipy.spatial.distance import cdist


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


class CPUIndex:
    """Accept an independently owned, validated C-contiguous float32 matrix."""

    def __init__(self, train_owned_f32):
        self.X = train_owned_f32
        self._train64 = None
        self._norm64 = None
        self._integral = None
        self._integer_max = None
        self._norm32 = None

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
            if not np.equal(sample, np.trunc(sample)).all():
                self._integral = False
            else:
                maximum = float(np.max(np.abs(self.X)))
                self._integral = (4 * features * maximum * maximum <= 2**24
                                  and bool(np.equal(self.X, np.trunc(self.X)).all()))
                self._integer_max = maximum
        if not self._integral:
            return False
        maximum = max(self._integer_max, float(np.max(np.abs(Q))))
        return (4 * features * maximum * maximum <= 2**24
                and bool(np.equal(Q, np.trunc(Q)).all()))

    def _exact_order(self, query, candidates, k):
        q = _integer_row(query)
        keys = []
        for index in candidates:
            point = _integer_row(self.X[index])
            squared = sum((left - right)**2 for left, right in zip(q, point, strict=True))
            keys.append((squared, int(index)))
        return np.asarray([index for _, index in sorted(keys)[:k]], np.intp)

    def _direct(self, Q, k, exclude_self=False, offset=0, exact=False):
        train, query = self.train64, Q.astype(np.float64, copy=False)
        work = cdist(query, train, "sqeuclidean")
        if exclude_self:
            work[np.arange(len(Q)), np.arange(offset, offset + len(Q))] = np.inf
        take = min(k + 1, len(train) - int(exclude_self))
        values, ids = _extract(work, take)
        if not exact:
            # Original FP32 coordinates are represented exactly in float64.
            # Subtraction, squaring and D-term accumulation have a standard
            # gamma bound. Float32's entire finite range has neither overflow
            # nor subnormal squared differences in float64.
            error = 16 * (self.X.shape[1] + 1) * np.finfo(np.float64).eps
            lower = values * (1 - error)
            upper = values * (1 + error)
            ambiguous = np.any(lower[:, 1:] <= upper[:, :-1], axis=1)
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
        work *= -2
        work += self._norm32[None, :]
        work += np.einsum("ij,ij->i", Q, Q)[:, None]
        if exclude_self:
            work[np.arange(len(Q)), np.arange(offset, offset + len(Q))] = np.inf
        values, ids = _extract(work, k)
        # Every input/product/partial sum is an integer, and the complete
        # absolute Gram expression is bounded by 4*D*max_abs**2 <=2**24.
        # Every arithmetic intermediate and the final cancellation is exact
        # in float32, including all mathematically equal distances.
        return values.astype(np.float64), ids

    def _gram(self, Q, k, exclude_self=False, offset=0):
        train, query = self.train64, Q.astype(np.float64, copy=False)
        if self._norm64 is None:
            self._norm64 = np.einsum("ij,ij->i", train, train)
        qnorm = np.einsum("ij,ij->i", query, query)
        norms = qnorm[:, None] + self._norm64[None, :]
        lower = query @ train.T
        lower *= -2
        lower += norms
        # The deliberately generous dot/norm/final-add envelope bounds every
        # original-coordinate squared distance. It is a full omitted-point
        # certificate, not a heuristic assertion about candidate membership.
        factor = 64 * (self.X.shape[1] + 1) * np.finfo(np.float64).eps
        lower -= factor * norms
        np.maximum(lower, 0, out=lower)
        if exclude_self:
            lower[np.arange(len(Q)), np.arange(offset, offset + len(Q))] = np.inf
        take = min(k + 2, len(train) - int(exclude_self))
        _, candidates = _extract(lower, take)
        barrier = np.min(lower, axis=1)
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

    def query(self, Q, k, *, return_distance=True, exclude_self=False,
              query_block_size=None, train_block_size=None, method="auto"):
        """Return float64 distances and int64 IDs; host inputs already validated."""
        Q = np.asarray(Q, dtype=np.float32, order="C")
        exact = self._integer_query(Q) if len(Q) else False
        rows = min(len(Q), query_block_size or 256)
        values, indices = [], []
        for start in range(0, len(Q), rows or 1):
            block = Q[start:start + rows]
            if exact and len(Q) >= 16 and self.X.shape[1] >= 32 and len(self.X) >= 512:
                result = self._integer_gram(block, k, exclude_self, start)
            elif method == "gram" or (method == "auto" and len(Q) >= 16 and len(self.X) >= 256):
                result = self._gram(block, k, exclude_self, start)
            else:
                result = self._direct(block, k, exclude_self, start, exact=exact)
            values.append(result[0])
            indices.append(result[1])
        squared = np.concatenate(values) if values else np.empty((0, k), np.float64)
        ids = np.concatenate(indices) if indices else np.empty((0, k), np.intp)
        return (np.sqrt(squared) if return_distance else None), ids
