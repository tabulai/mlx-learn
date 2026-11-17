"""Build a row-space-preserving L-BFGS history from a GPU Hessian sketch.

This module deliberately imports neither NumPy nor the sklearn-compatible
``mlxlearn`` package. The sketch changes only the initial inverse-Hessian
approximation; the optimizer must still use its ordinary line search and
independent stopping checks. It is not an objective-error certificate.
"""

from __future__ import annotations

import math

import mlx.core as mx

# Absolute, so the module still loads by path outside its package.
from mlxlearn_device._binary32 import matmul

__all__ = ["build_sketch_history", "initial_metric_from_parts"]


def initial_metric_from_parts(parts, reg, denom):
    """Normalize initial statistics identically for both sketch entry points.

    This intentionally is not ``mx.compile``: fusion can change the float32
    reduction by an ULP, which can alter a seeded optimizer's trajectory. The
    ordinary unsketched solver retains its compiled calculation.
    """
    initial = mx.sum(parts, axis=1) / denom
    diagonal = mx.mean(initial[1] + reg)
    scale = mx.where(mx.isfinite(diagonal) & (diagonal > 1e-12), 1 / diagonal, 1.0)
    return initial[0], mx.ones_like(initial[0]) * scale


def build_sketch_history(
    X,
    sample_weight,
    sw_sum,
    l2,
    fit_intercept,
    h0inv,
    *,
    history_size=10,
    sketch_size=4,
    seed=0,
    min_curvature=1e-12,
):
    """Construct GPU-resident curvature pairs for centered binary logistic X.

    The Hessian at zero logits is ``X.T @ diag(sw / 4) @ X / sw_sum + l2 I``.
    With sample-space Rademacher probes ``Omega``, all feature components of
    ``S = X.T @ Omega`` and ``Y = H @ S`` lie in ``range(X.T)`` in exact
    arithmetic.  Each probe has zero initial intercept component; ``Y`` also
    includes the corresponding Hessian cross term for a fitted intercept.

    Returns contiguous MLX float32 arrays ``(s, y, coefficients, count)``
    compatible with the native loop's history ring. Invalid, nonpositive, or
    numerically tiny curvature pairs are omitted. The returned arrays are
    caller-owned and need to remain alive while a submitted Metal command uses
    them. ``count == 0`` indicates the unseeded optimizer should be used.

    This is a preconditioner, not a loss approximation: the supplied design is
    the same float32 centered design that will be used by the optimizer. The
    caller remains responsible for validating input, weights, objective,
    original-coordinate gradient and floating-point precision.
    """
    if X.ndim != 2 or X.dtype != mx.float32:
        raise ValueError("X must be a two-dimensional float32 MLX array")
    n, d = X.shape
    p = d + int(fit_intercept)
    if n < 1 or d < 1:
        raise ValueError("X must have at least one row and column")
    if h0inv.shape != (p,) or h0inv.dtype != mx.float32:
        raise ValueError("h0inv must be a float32 vector matching parameter size")
    if sample_weight is not None and (
        sample_weight.shape != (n,) or sample_weight.dtype != mx.float32
    ):
        raise ValueError("sample_weight must be a row-aligned float32 MLX vector")
    if not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    if not isinstance(sketch_size, int) or sketch_size < 0:
        raise ValueError("sketch_size must be a nonnegative integer")
    if not math.isfinite(float(sw_sum)) or sw_sum <= 0:
        raise ValueError("sw_sum must be finite and positive")
    if not math.isfinite(float(l2)) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    if not math.isfinite(float(min_curvature)) or min_curvature < 0:
        raise ValueError("min_curvature must be finite and nonnegative")
    k = min(sketch_size, history_size, n, d)
    if k == 0:
        return (
            mx.zeros((history_size, p), dtype=mx.float32),
            mx.zeros((history_size, p), dtype=mx.float32),
            mx.zeros((history_size, 2), dtype=mx.float32),
            0,
        )

    key = mx.random.key(seed)
    uniform = mx.random.uniform(shape=(n, k), key=key, dtype=mx.float32)
    probes = mx.where(uniform < 0.5, -1.0, 1.0).astype(mx.float32)
    # Scaling by sqrt(n) before the multiplication limits accumulation range;
    # a nonfinite result is rejected below rather than stored in the history.
    probes = probes / math.sqrt(n)
    features = matmul(X, probes, transpose_a=True)
    # Normalize each projected probe before the Hessian products. A max-norm
    # pass avoids overflowing a squared norm for large, but finite, columns.
    maximum = mx.max(mx.abs(features), axis=0)
    finite_scale = mx.isfinite(maximum) & (maximum > 0)
    scaled = features / mx.where(finite_scale, maximum, 1.0)
    length = mx.sqrt(mx.sum(scaled * scaled, axis=0))
    valid_length = finite_scale & mx.isfinite(length) & (length > 0)
    features = scaled / mx.where(valid_length, length, 1.0)

    projection = matmul(X, features)
    weighted = projection * (0.25 / float(sw_sum))
    if sample_weight is not None:
        weighted = weighted * sample_weight[:, None]
    y_features = matmul(X, weighted, transpose_a=True) + float(l2) * features
    if fit_intercept:
        s = mx.concatenate([features.T, mx.zeros((k, 1), dtype=mx.float32)], axis=1)
        bias_y = mx.sum(weighted, axis=0)
        y = mx.concatenate([y_features.T, bias_y[:, None]], axis=1)
    else:
        s, y = features.T, y_features.T
    sy = mx.sum(s * y, axis=1)
    yy = mx.sum(y * y * h0inv[None, :], axis=1)
    valid = (
        valid_length
        & mx.all(mx.isfinite(s), axis=1)
        & mx.all(mx.isfinite(y), axis=1)
        & mx.isfinite(sy)
        & mx.isfinite(yy)
        & (yy > 0)
        & (sy > float(min_curvature) * mx.maximum(yy, 1.0))
    )
    # A few scalar decisions cross to the host; no feature, training, or
    # optimizer vectors are read by the CPU or through NumPy.
    mx.eval(s, y, sy, yy, valid)
    selected = [i for i, ok in enumerate(valid.tolist()) if ok]
    count = len(selected)
    if count:
        indices = mx.array(selected, dtype=mx.int32)
        selected_s = mx.take(s, indices, axis=0)
        selected_y = mx.take(y, indices, axis=0)
        selected_sy = mx.take(sy, indices, axis=0)
        selected_yy = mx.take(yy, indices, axis=0)
        coefficients = mx.stack([1.0 / selected_sy, selected_sy / selected_yy], axis=1)
    else:
        selected_s = mx.zeros((0, p), dtype=mx.float32)
        selected_y = mx.zeros((0, p), dtype=mx.float32)
        coefficients = mx.zeros((0, 2), dtype=mx.float32)
    padding = history_size - count
    if padding:
        selected_s = mx.concatenate([selected_s, mx.zeros((padding, p), dtype=mx.float32)])
        selected_y = mx.concatenate([selected_y, mx.zeros((padding, p), dtype=mx.float32)])
        coefficients = mx.concatenate([coefficients, mx.zeros((padding, 2), dtype=mx.float32)])
    return selected_s, selected_y, coefficients, count
