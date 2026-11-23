"""CSR-first radius results and the conversions scikit-learn's API needs.

Every radius engine produces one :class:`RadiusCSR` per query batch: exact
per-row counts, rows in query order, training indices ascending within each row
(or ordered by ``(distance, index)`` when sorting was requested), and float64
distances when they were requested. Nothing here sorts globally or loops over
rows in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._common.arrays import INDEX_DTYPE

_FLT_MAX = float(np.finfo(np.float32).max)

__all__ = [
    "RadiusCSR",
    "all_indices",
    "concatenate",
    "csr_to_objects",
    "empty_result",
    "pairs_to_csr",
    "sort_rows",
    "to_graph",
    "to_ragged",
]


@dataclass(frozen=True, slots=True)
class RadiusCSR:
    """One query batch's neighborhoods in compressed sparse row form."""

    indptr: np.ndarray  # int64[n_query + 1]
    indices: np.ndarray  # int64[nnz]
    distances: np.ndarray | None  # float64[nnz] or None
    route: str = ""

    @property
    def n_query(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1]) if len(self.indptr) else 0

    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def row_ids(self) -> np.ndarray:
        """The query row of every stored entry."""
        return np.repeat(np.arange(self.n_query, dtype=INDEX_DTYPE), self.lengths())


def all_indices(*, n_train: int, n_query: int, q_offset: int, exclude_self: bool) -> RadiusCSR:
    """Build the index-only result directly when the squared radius is infinite."""
    width = n_train - int(exclude_self)
    indptr = np.arange(n_query + 1, dtype=INDEX_DTYPE) * width
    source = np.arange(n_train, dtype=INDEX_DTYPE)
    if exclude_self:
        rows = np.empty((n_query, width), dtype=INDEX_DTYPE)
        for i, row in enumerate(rows, start=q_offset):
            row[:i], row[i:] = source[:i], source[i + 1:]
        indices = rows.ravel()
    else:
        indices = np.tile(source, n_query)
    return RadiusCSR(indptr, indices, None, "brute")


def empty_result(n_query: int, *, need_distance: bool, route: str = "") -> RadiusCSR:
    return RadiusCSR(
        np.zeros(n_query + 1, dtype=INDEX_DTYPE),
        np.empty(0, dtype=INDEX_DTYPE),
        np.empty(0, dtype=np.float64) if need_distance else None,
        route,
    )


def csr_to_objects(indptr: np.ndarray, values: np.ndarray) -> np.ndarray:
    """An object array of 1-d row views of ``values``, one per CSR row."""
    bounds = np.asarray(indptr).tolist()
    count = len(bounds) - 1
    if count <= 0:
        return np.empty(0, dtype=object)
    rows = map(values.__getitem__, map(slice, bounds[:-1], bounds[1:]))
    return np.fromiter(rows, dtype=object, count=count)


def concatenate(parts: list[RadiusCSR]) -> RadiusCSR:
    """Join batch results in query order."""
    if len(parts) == 1:
        return parts[0]
    if not parts:
        raise ValueError("no results to concatenate")
    offsets = np.cumsum([0] + [p.nnz for p in parts[:-1]])
    indptr = np.concatenate([parts[0].indptr[:1]] + [
        p.indptr[1:] + offset for p, offset in zip(parts, offsets, strict=True)
    ])
    indices = np.concatenate([p.indices for p in parts])
    distances = (None if parts[0].distances is None
                 else np.concatenate([p.distances for p in parts]))
    route = parts[0].route if len({p.route for p in parts}) == 1 else "mixed"
    return RadiusCSR(indptr.astype(INDEX_DTYPE, copy=False), indices, distances, route)


def to_ragged(parts: list[RadiusCSR], *, return_distance: bool):
    """scikit-learn's ragged output: object arrays of 1-d int64/float64 rows.

    Rows of one batch are views of that batch's buffers.
    """
    if not parts:
        empty = np.empty(0, dtype=object)
        return (empty, np.empty(0, dtype=object)) if return_distance else empty
    if len(parts) == 1:
        part = parts[0]
        indices = csr_to_objects(part.indptr, part.indices)
        if not return_distance:
            return indices
        return csr_to_objects(part.indptr, part.distances), indices
    indices = np.concatenate([csr_to_objects(p.indptr, p.indices) for p in parts])
    if not return_distance:
        return indices
    distances = np.concatenate([csr_to_objects(p.indptr, p.distances) for p in parts])
    return distances, indices


def to_graph(parts: list[RadiusCSR], n_samples_fit: int, *, mode: str):
    """Build scikit-learn's CSR neighbor graph without object arrays."""
    from scipy import sparse
    from sklearn import get_config as get_sklearn_config

    joined = concatenate(parts)
    if mode == "connectivity":
        data = np.ones(joined.nnz)
    else:
        data = joined.distances
    # Mirror sklearn: build a csr_array, then align with the configured
    # sparse interface (csr_matrix by default).
    graph = sparse.csr_array(
        (data, joined.indices, joined.indptr), shape=(joined.n_query, int(n_samples_fit)),
    )
    if get_sklearn_config().get("sparse_interface", "spmatrix") == "spmatrix":
        graph = sparse.csr_matrix(graph)
    return graph


def sort_rows(result: RadiusCSR) -> RadiusCSR:
    """Order every row by ``(distance, index)``.

    Input rows must already be in ascending index order. A stable sort on the
    float32 image of the float64 distance keeps equal keys in index order;
    rows where two entries share a float32 key but their float64 distances are
    out of order are then re-sorted exactly.
    """
    if result.distances is None:
        raise ValueError("sorting requires distances")
    nnz = result.nnz
    if nnz <= 1:
        return result
    rows = result.row_ids().astype(np.uint64)
    dist = result.distances
    # float32 rounding is monotone; clipping at FLT_MAX keeps it monotone and
    # avoids overflow warnings (the exact fix-up below orders any collisions).
    image = np.minimum(dist, _FLT_MAX).astype(np.float32)
    key = (rows << np.uint64(32)) | image.view(np.uint32).astype(np.uint64)
    order = np.argsort(key, kind="stable")
    indices = result.indices[order]
    dist = dist[order]
    key = key[order]
    tie = key[1:] == key[:-1]
    bad = np.flatnonzero(tie & (dist[1:] < dist[:-1]))
    if bad.size:
        bad_rows = np.unique(rows[order][bad])
        for row in bad_rows.tolist():
            start, stop = int(result.indptr[row]), int(result.indptr[row + 1])
            fix = np.lexsort((indices[start:stop], dist[start:stop]))
            indices[start:stop] = indices[start:stop][fix]
            dist[start:stop] = dist[start:stop][fix]
    return RadiusCSR(result.indptr, indices, dist, result.route)


def pairs_to_csr(rows: np.ndarray, cols: np.ndarray, values: np.ndarray | None,
                 n_query: int, n_train: int, *, route: str = "") -> RadiusCSR:
    """Group unordered (row, col) pairs into rows with ascending columns."""
    rows = np.asarray(rows, dtype=INDEX_DTYPE)
    cols = np.asarray(cols, dtype=INDEX_DTYPE)
    if rows.size:
        key = rows * np.int64(n_train) + cols
        order = np.argsort(key)
        rows, cols = rows[order], cols[order]
        if values is not None:
            values = np.asarray(values)[order]
    counts = np.bincount(rows, minlength=n_query) if rows.size else np.zeros(n_query, INDEX_DTYPE)
    indptr = np.zeros(n_query + 1, dtype=INDEX_DTYPE)
    np.cumsum(counts, out=indptr[1:])
    return RadiusCSR(indptr, cols, None if values is None else np.asarray(values, np.float64), route)


def from_grouped(row_lengths: np.ndarray, cols: np.ndarray, distances: np.ndarray | None,
                 *, route: str = "") -> RadiusCSR:
    """Wrap entries that are already grouped by row, in row order."""
    indptr = np.zeros(len(row_lengths) + 1, dtype=INDEX_DTYPE)
    np.cumsum(row_lengths, out=indptr[1:])
    return RadiusCSR(indptr, np.asarray(cols, dtype=INDEX_DTYPE), distances, route)
