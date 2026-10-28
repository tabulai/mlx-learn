"""Private device-only inexact Newton solver for wide binary logistic models.

Each Newton system uses a bounded native Metal CG loop. Relative residual
stopping avoids over-solving unresolved float32 null directions. The outer
solver keeps the original logistic objective, checks represented steps with a
stable Armijo difference, and requires both a numerical dual-gap check and the
compensated original-coordinate gradient before reporting convergence.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mlx.core as mx

from ._compensated import score_original
from ._duality import logistic_duality_gap
from ._native_cg import solve_cg


def use_newton_for_curvature(parts, n, d, denominator, l2):
    """Select scale-skewed regimes that benefit from Newton steps.

    Small regularization magnifies float32 dual-gradient roundoff. Keep those
    problems on scalar-metric L-BFGS, which also preserves its rank behavior.
    Below 2048 features, reserve Newton for diagonal spreads beyond float32's
    reciprocal epsilon: L-BFGS history differences can lose the slow directions
    at that scale even with a relative curvature floor. Newton's matrix-vector
    products avoid those gradient subtractions without changing the objective
    or introducing a diagonal metric that could alter rank-deficient solutions.
    This heuristic selects an algorithm; it never relaxes stopping checks.
    """
    from ._native_cg import _workspace_supported, metal_loop

    version = getattr(mx, "__version__", None)
    if (parts is None or n < 4096 or not 512 <= d <= 4096
        or not math.isfinite(l2) or l2 < 1e-7
        or not _workspace_supported(n, d)
        or not isinstance(version, str) or re.fullmatch(r"0\.32\.(?:0|[1-9]\d{0,2})", version) is None
        or metal_loop is None or not hasattr(metal_loop, "run_cg")):
        return False
    diagonal = mx.sum(parts[1, :, :d], axis=0) / denominator + l2
    extremes = mx.stack((mx.min(diagonal), mx.max(diagonal)))
    mx.eval(extremes)
    minimum, maximum = extremes.tolist()
    spread_threshold = 100.0 if d >= 2048 else 1.0 / 1.1920928955078125e-7
    return (math.isfinite(minimum) and math.isfinite(maximum)
            and minimum > 0 and maximum > spread_threshold * minimum)


@dataclass(frozen=True)
class NewtonResult:
    """A bounded attempt; only successful checks set ``converged``."""

    theta: mx.array
    n_iter: int
    n_fev: int
    converged: bool
    stop_reason: str
    objective: float
    gradient_max_abs: float
    relative_gap: float
    tail_bound: float
    hvp_count: int
    used_gradient_fallback: bool


@mx.compile
def _evaluate_fallback(X, y, sw, theta, denom, reg):
    d = X.shape[1]
    raw = X @ theta[:d] + theta[d]
    active = sw > 0
    logits = mx.where(active, raw, 0.0)
    signed = (1.0 - 2.0 * y) * logits
    weights = sw / denom
    residual = (1.0 - 2.0 * y) * weights * mx.sigmoid(signed)
    gradient = mx.concatenate((X.T @ residual, mx.sum(residual).reshape(1))) + reg * theta
    curvature = weights * mx.sigmoid(logits) * mx.sigmoid(-logits)
    loss = mx.sum(weights * mx.logaddexp(0.0, signed)) + 0.5 * mx.sum(reg * theta * theta)
    return loss, gradient, curvature, logits


@mx.compile
def _represented_difference(X, y, sw, old, new, old_logits, denom, reg):
    """Loss change of the represented parameter step without total subtraction."""
    step = new - old
    shift = X @ step[:-1] + step[-1]
    shift = mx.where(sw > 0, shift, 0.0)
    sign = 1.0 - 2.0 * y
    u, change = sign * old_logits, sign * shift
    # expm1 remains bounded on the branch used for small changes. Larger
    # changes use finite softplus differences; neither branch may overflow.
    small = mx.clip(change, -1.0, 1.0)
    near = mx.log1p(mx.sigmoid(u) * mx.expm1(small))
    far = mx.logaddexp(0.0, u + change) - mx.logaddexp(0.0, u)
    delta = mx.sum((sw / denom) * mx.where(mx.abs(change) <= 1.0, near, far))
    delta = delta + mx.sum(reg * step * (old + 0.5 * step))
    return delta, step


def _validate(original, centered, center, y, sw, denominator, l2, tol, max_iter, max_cg, relative_gap_tol):
    if (not isinstance(original, mx.array) or original.ndim != 2
        or original.dtype != mx.float32 or original.shape[0] <= 0
        or not 1 <= original.shape[1] <= 4096):
        raise ValueError("original must be a nonempty float32 matrix with at most 4096 features")
    n, d = original.shape
    for value, shape in ((centered, (n, d)), (center, (d,)), (y, (n,)), (sw, (n,))):
        if not isinstance(value, mx.array) or value.shape != shape or value.dtype != mx.float32:
            raise ValueError("Newton inputs must be matching float32 MLX arrays")
    for name, value in (("denominator", denominator), ("l2", l2), ("tol", tol),
                        ("relative_gap_tol", relative_gap_tol)):
        if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not 1.1754943508222875e-38 <= value <= 3.4028234663852886e38):
            raise ValueError(f"{name} must be a normal positive float32 value")
    if type(max_iter) is not int or max_iter < 1 or type(max_cg) is not int or not 1 <= max_cg <= 512:
        raise ValueError("invalid Newton or CG iteration budget")


def minimize_newton(
    original, centered, center, y, sample_weight, denominator, l2, *,
    tol, max_iter, max_cg=128, relative_gap_tol=1e-5, initial=None,
):
    """Attempt a GPU Newton solve on already validated, centered float32 data.

    The caller validates finite training data, both labels, nonnegative weights
    with positive total mass, and retains ``original``. Only binary models with
    a fitted, unregularized intercept are supported. Every accepted parameter
    update consumes one unit of ``max_iter``; rejected line-search trials and
    CG products are separately counted. Failure returns the last accepted
    parameters and cannot be promoted by a later gradient-only check.

    The caller may resume another method with ``max_iter - n_iter`` accepted
    updates. Training data, parameters, and gradients remain on the GPU.
    """
    _validate(original, centered, center, y, sample_weight, denominator, l2,
              tol, max_iter, max_cg, relative_gap_tol)
    n, d = original.shape
    if initial is not None and (
        not isinstance(initial, mx.array) or initial.shape != (d + 1,) or initial.dtype != mx.float32
    ):
        raise ValueError("initial must be a matching float32 parameter vector")
    with mx.stream(mx.gpu):
        centered = mx.contiguous(centered)
        sw = mx.contiguous(sample_weight)
        y = mx.contiguous(y)
        reg = mx.concatenate((mx.full((d,), l2, mx.float32), mx.zeros((1,), mx.float32)))
        denom = mx.array(denominator, mx.float32)
        theta = mx.zeros((d + 1,), mx.float32) if initial is None else mx.contiguous(initial)
        gradient_fallback = True
        iterations = evaluations = products = 0
        relative_gap = objective_value = norm_value = tail_value = float("inf")
        reason, converged = "max_iter", False

        def evaluate(value):
            nonlocal evaluations
            # Use MLX's supported primitives; a custom gradient specialization
            # needs its own pipeline capability checks before being enabled.
            result = _evaluate_fallback(centered, y, sw, value, denom, reg)
            mx.eval(*result)
            evaluations += 1
            return result

        loss, gradient, curvature, logits = evaluate(theta)
        # Include a final check after the last accepted budgeted update.
        for outer in range(max_iter + 1):
            objective_value = float(loss)
            centered_norm = float(mx.max(mx.abs(gradient)))
            if not (math.isfinite(objective_value) and math.isfinite(centered_norm)):
                reason = "non-finite-objective"
                break
            if centered_norm <= tol:
                gap = logistic_duality_gap(centered, y, theta, sw, denominator, l2,
                                           device=mx.gpu, logits=logits)
                mx.eval(gap.valid, gap.relative_gap)
                relative_gap = float(gap.relative_gap)
                if bool(gap.valid) and relative_gap <= relative_gap_tol:
                    objective, norm, _, _, tail = score_original(
                        original, center, theta, y, sw, denominator, l2,
                    )
                    mx.eval(objective, norm, tail)
                    objective_value, norm_value, tail_value = float(objective), float(norm), float(tail)
                    if (math.isfinite(objective_value) and math.isfinite(norm_value)
                        and math.isfinite(tail_value) and norm_value + tail_value <= tol * 0.95):
                        converged, reason = True, "duality-gap"
                        break
            if outer == max_iter:
                reason = "max_iter"
                break
            forcing = 0.1 if centered_norm > 1e-3 else 0.01
            try:
                direction, meta = solve_cg(centered, mx.contiguous(curvature),
                                           mx.contiguous(gradient), reg,
                                           forcing=forcing, max_iter=max_cg)
            except (RuntimeError, ValueError):
                reason = "native-cg-unavailable"
                break
            # Native metadata is (product_count, status, residual2, initial2,
            # max_rayleigh). Check the status before using a direction.
            count, status = int(meta[0]), int(meta[1])
            products += count + int(status == 2)
            if status == 3 or not bool(mx.all(mx.isfinite(direction))):
                reason = "non-finite-direction"
                break
            slope = float(mx.sum(gradient * direction))
            if not math.isfinite(slope) or slope >= 0:
                reason = "non-descent-direction"
                break
            accepted = False
            for search in range(12):
                candidate = theta + (2.0 ** -search) * direction
                difference, step = _represented_difference(
                    centered, y, sw, theta, candidate, logits, denom, reg,
                )
                actual_slope = mx.sum(gradient * step)
                mx.eval(difference, actual_slope)
                change, represented_slope = float(difference), float(actual_slope)
                evaluations += 1
                if (math.isfinite(change) and math.isfinite(represented_slope)
                    and represented_slope < 0 and change <= 1e-4 * represented_slope):
                    candidate_loss, candidate_gradient, candidate_curvature, candidate_logits = evaluate(candidate)
                    if (math.isfinite(float(candidate_loss))
                        and bool(mx.all(mx.isfinite(candidate_gradient)))):
                        theta = candidate
                        loss, gradient = candidate_loss, candidate_gradient
                        curvature, logits = candidate_curvature, candidate_logits
                        iterations += 1
                        accepted = True
                        break
            if not accepted:
                reason = "line-search-failure"
                break
        if not converged:
            objective, norm, _, _, tail = score_original(original, center, theta, y, sw, denominator, l2)
            mx.eval(objective, norm, tail)
            objective_value, norm_value, tail_value = float(objective), float(norm), float(tail)
        return NewtonResult(theta, iterations, evaluations, converged, reason,
                            objective_value, norm_value, relative_gap, tail_value,
                            products, gradient_fallback)
