"""Optional native SVC arithmetic; NumPy only supplies boundary buffers.

Returning ``None`` when the extension is unavailable lets callers retain their
portable implementation. Numerical reductions and multiclass aggregation run
in C++, without the NumPy C API or Python iteration over samples.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np

try:
    _native = importlib.import_module(f"{__package__}._svc_cpu_native")
except ModuleNotFoundError as exc:
    # A missing optional module is expected in source-only installations. Do
    # not hide a broken binary or a dependency import failure.
    if exc.name != f"{__package__}._svc_cpu_native":
        raise
    _native = None


def _matrix(X, dtype=None):
    result = np.require(X, dtype=dtype, requirements=["C", "A"])
    if result.ndim != 2:
        raise ValueError("SVC arithmetic requires a two-dimensional matrix")
    return result


def contract(tile, weights, bias=None):
    """Return ``tile @ weights - bias`` in float64, or ``None``.

    Conversion scratch is bounded to four MiB, including for a large tile.
    Small calls use compensated sums; larger calls use native float64 BLAS.
    """
    if _native is None:
        return None
    tile = _matrix(tile, np.float32)
    weights = _matrix(weights, np.float64)
    if tile.shape[1] != weights.shape[0]:
        raise ValueError("kernel tile and weights must agree in support-vector count")
    if bias is not None:
        bias = np.require(bias, dtype=np.float64, requirements=["C", "A"])
    result = np.empty((tile.shape[0], weights.shape[1]), dtype=np.float64)
    _native.contract(tile, weights, bias, result)
    return result


def variance(X):
    """Population variance over every entry, accumulated in float64, or ``None``."""
    if _native is None:
        return None
    return _native.variance(_matrix(X))


def column_mean(X):
    """Column means accumulated in float64, or ``None``."""
    if _native is None:
        return None
    X = _matrix(X)
    result = np.empty(X.shape[1], dtype=np.float64)
    _native.column_mean(X, result)
    return result


def diagonal(X, spec):
    """Analytic kernel diagonal on float32 coordinates in float64, or ``None``."""
    if _native is None:
        return None
    X = _matrix(X)
    if X.dtype != np.float32 or int(spec.degree) > sys.maxsize:
        return None
    kinds = {"linear": 0, "rbf": 1, "poly": 2, "sigmoid": 3}
    result = np.empty(X.shape[0], dtype=np.float64)
    _native.diagonal(X, kinds[spec.kernel], float(spec.gamma), float(spec.coef0),
                     int(spec.degree), result)
    return result


def ovo_vote(decisions, n_classes, *, break_ties=False):
    """LIBSVM votes, or the OVR winner for ``break_ties``, or ``None``."""
    if _native is None:
        return None
    decisions = _matrix(decisions, np.float64)
    result = np.empty(decisions.shape[0], dtype=np.int64)
    _native.aggregate(decisions, int(n_classes), 2 if break_ties else 0, result)
    return result


def ovr_decision(decisions, n_classes):
    """OVR votes plus bounded confidence corrections, or ``None``.

    A pairwise zero votes for its first class here, as in sklearn's OVR
    conversion. The LIBSVM prediction rule instead gives zero to the second.
    """
    if _native is None:
        return None
    decisions = _matrix(decisions, np.float64)
    result = np.empty((decisions.shape[0], int(n_classes)), dtype=np.float64)
    _native.aggregate(decisions, int(n_classes), 1, result)
    return result


def matches_float32(source, snapshot):
    """Compare the source's float32 representation to an owned snapshot.

    Contiguous float64 input is checked without allocating a converted matrix.
    Bit comparisons include signed zero and NaN payloads; no hash can conceal
    an in-place mutation. Unsupported native input types return ``None``.
    """
    if _native is None or not hasattr(_native, "matches_float32"):
        return None
    source = _matrix(source)
    snapshot = _matrix(snapshot)
    if source.dtype not in (np.float32, np.float64) or snapshot.dtype != np.float32:
        return None
    return _native.matches_float32(source, snapshot)
