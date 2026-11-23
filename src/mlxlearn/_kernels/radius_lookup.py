"""Exact duplicate-row lookup for batched zero-radius searches."""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .._common.arrays import INDEX_DTYPE
from .radius_result import RadiusCSR


def _row_keys(rows):
    # Binary32 has a unique encoding for every finite value except signed zero.
    bits = rows.view(np.uint32)
    negative_zero = bits == 0x80000000
    if negative_zero.any():
        bits = bits.copy()
        bits[negative_zero] = 0
    return bits.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)


class RowLookup:
    def __init__(self, train):
        self.keys = _row_keys(train)
        self.order = np.argsort(self.keys, kind="stable")
        self.inverse = np.empty_like(self.order)
        self.inverse[self.order] = np.arange(len(train), dtype=INDEX_DTYPE)

    @cached_property
    def _groups(self):
        # Compute group boundaries once for self queries. Bound the temporary
        # gathered rows by their byte size, even for wide training matrices.
        n = len(self.keys)
        new_group = np.ones(n, dtype=bool)
        step = max(1, (1 << 20) // self.keys.dtype.itemsize)
        for start in range(1, n, step):
            stop = min(n, start + step)
            block = self.keys[self.order[start - 1:stop]]
            new_group[start:stop] = block[1:] != block[:-1]
        groups = np.empty(n, dtype=INDEX_DTYPE)
        groups[self.order] = np.cumsum(new_group) - 1
        return groups, np.append(np.flatnonzero(new_group), n)

    def query(self, queries, *, exclude_self, q_offset, need_distance):
        keys = _row_keys(queries)
        if exclude_self and np.array_equal(keys, self.keys[q_offset:q_offset + len(keys)]):
            groups, boundaries = self._groups
            group = groups[q_offset:q_offset + len(keys)]
            lo, hi = boundaries[group], boundaries[group + 1]
        else:
            lo = np.searchsorted(self.keys, keys, side="left", sorter=self.order)
            hi = np.searchsorted(self.keys, keys, side="right", sorter=self.order)
        counts = hi - lo
        if exclude_self:
            own = self.inverse[q_offset:q_offset + len(keys)]
            remove = (own >= lo) & (own < hi)
            counts -= remove
        indptr = np.zeros(len(keys) + 1, dtype=INDEX_DTYPE)
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), dtype=INDEX_DTYPE)
        # Bound temporary storage even when many training rows are identical.
        for start in range(0, len(indices), 1 << 17):
            stop = min(start + (1 << 17), len(indices))
            offsets = np.arange(start, stop, dtype=INDEX_DTYPE)
            rows = np.searchsorted(indptr[1:], offsets, side="right")
            pos = lo[rows] + (offsets - indptr[rows])
            if exclude_self:
                pos += remove[rows] & (pos >= own[rows])
            indices[start:stop] = self.order[pos]
        distances = np.zeros(len(indices), np.float64) if need_distance else None
        return RadiusCSR(indptr, indices, distances, "row-lookup")
