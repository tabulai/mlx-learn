"""Vectorized radius votes and means from CSR neighborhoods.

Each row is accumulated in ascending training-index order (``np.bincount``
scans its input in array order and the rows are stored that way), so results
do not depend on how queries were batched.
"""

from __future__ import annotations

import numpy as np

from .._common.arrays import INDEX_DTYPE
from .radius_result import RadiusCSR

__all__ = ["distance_weights", "mean_of_parts", "reduce_mean_csr", "reduce_votes_csr",
           "votes_of_parts"]


def distance_weights(result: RadiusCSR, rows: np.ndarray) -> np.ndarray:
    """``1/d`` per entry, except rows containing a zero distance use its indicator.

    Matches scikit-learn's ``_get_weights`` for ragged neighborhoods.
    """
    dist = result.distances
    if dist is None:
        raise ValueError("distance weights require distances")
    zero = dist == 0.0
    with np.errstate(divide="ignore"):
        weights = 1.0 / dist
    if zero.any():
        coincident = np.bincount(rows[zero], minlength=result.n_query) > 0
        on_row = coincident[rows]
        weights[on_row] = zero[on_row]
    return weights


def reduce_votes_csr(result: RadiusCSR, codes: np.ndarray, n_classes: int, *,
                     weighted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-query class votes (float64 ``[n_query, n_classes]``) and neighbor counts."""
    nq = result.n_query
    lengths = result.lengths().astype(INDEX_DTYPE, copy=False)
    rows = result.row_ids()
    keys = rows * n_classes + np.asarray(codes)[result.indices]
    if weighted:
        weights = distance_weights(result, rows)
        # np.bincount returns int64 zeros for empty input; votes are float64.
        votes = np.bincount(keys, weights=weights, minlength=nq * n_classes).astype(
            np.float64, copy=False)
    else:
        votes = np.bincount(keys, minlength=nq * n_classes).astype(np.float64)
    return votes.reshape(nq, n_classes), lengths


def reduce_mean_csr(result: RadiusCSR, targets: np.ndarray, *,
                    weighted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-query (weighted) mean of single-output targets; empty rows are NaN."""
    nq = result.n_query
    lengths = result.lengths().astype(INDEX_DTYPE, copy=False)
    rows = result.row_ids()
    values = np.asarray(targets, dtype=np.float64)[result.indices]
    if weighted:
        weights = distance_weights(result, rows)
        numerator = np.bincount(rows, weights=values * weights, minlength=nq)
        denominator = np.bincount(rows, weights=weights, minlength=nq)
    else:
        numerator = np.bincount(rows, weights=values, minlength=nq)
        denominator = lengths.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        prediction = numerator / denominator
    prediction[lengths == 0] = np.nan
    return prediction, lengths


def votes_of_parts(parts, codes: np.ndarray, n_classes: int, *,
                   weighted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Votes over result batches (CSR or native results), in query order."""
    pieces = [part.votes(codes, n_classes, weighted) if not isinstance(part, RadiusCSR)
              else reduce_votes_csr(part, codes, n_classes, weighted=weighted)
              for part in parts]
    if len(pieces) == 1:
        return pieces[0]
    return (np.concatenate([p[0] for p in pieces]),
            np.concatenate([p[1] for p in pieces]))


def mean_of_parts(parts, targets: np.ndarray, *, weighted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Means over result batches (CSR or native results), in query order."""
    pieces = [part.mean(targets, weighted) if not isinstance(part, RadiusCSR)
              else reduce_mean_csr(part, targets, weighted=weighted)
              for part in parts]
    if len(pieces) == 1:
        return pieces[0]
    return (np.concatenate([p[0] for p in pieces]),
            np.concatenate([p[1] for p in pieces]))
