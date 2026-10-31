"""L-BFGS with fused MLX vector arithmetic and device-resident history.

Coefficient vectors, gradients, curvature history and the two-loop recurrence
stay on the selected MLX device. ``mx.compile`` fuses array operations between
control decisions. A small statistics vector crosses to Python once per
direction/update, rather than one scalar per history dot product. Adaptive
strong-Wolfe control and the caller's high-precision objective accumulation
remain on the host; this is not a CPU optimizer or a synchronization-free loop.

The original solver remains in ``lbfgs`` as the benchmark reference. The two
share configuration, state, and line-search control, not mutable workspaces.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import mlx.core as mx
import numpy as np

from .lbfgs import LBFGSConfig, LBFGSState, ObjectiveGradient, _LineSearch, curvature_pair_accepted

__all__ = ["minimize"]


@mx.compile
def _direction(grad, s_history, y_history, ys_history, yy_history):
    q = grad
    alphas = []
    for s, y, ys in zip(
        reversed(s_history), reversed(y_history), reversed(ys_history), strict=True
    ):
        alpha = mx.sum(s * q) / ys
        alphas.append(alpha)
        q = q - alpha * y
    if s_history:
        r = (ys_history[-1] / mx.maximum(yy_history[-1], 1e-30)) * q
        for index, (s, y, ys) in enumerate(zip(s_history, y_history, ys_history, strict=True)):
            beta = mx.sum(y * r) / ys
            r = r + s * (alphas[-1 - index] - beta)
        direction = -r
    else:
        direction = -grad
    return direction, mx.stack(
        [mx.sum(grad * direction), mx.sum(mx.abs(grad)), mx.sum(grad * grad)]
    )


@mx.compile
def _trial(x, direction, step):
    return x + step * direction


def _convergence_gradient_max(grad, center):
    """Return the infinity norm in the caller's original coordinates.

    Centering changes the intercept parameterization: if ``c = b + w @
    center``, then ``grad_w_original = grad_w_centered + center * grad_c``.
    The optimizer must still use centered gradients for directions and line
    searches; only its stopping statistic is transformed.
    """
    if center is None:
        return mx.max(mx.abs(grad))
    feature_gradient = grad[..., :-1] + grad[..., -1:] * center
    return mx.maximum(mx.max(mx.abs(feature_gradient)), mx.max(mx.abs(grad[..., -1:])))


@mx.compile
def _curvature(x_new, x, g_new, grad, convergence_center=None):
    s, y = x_new - x, g_new - grad
    return s, y, mx.stack([
        mx.sum(s * y), mx.sum(y * y), _convergence_gradient_max(g_new, convergence_center)
    ])


@mx.compile
def _slope(grad, direction):
    return mx.sum(grad * direction)


def minimize(
    fun: ObjectiveGradient,
    x0: Any,
    config: LBFGSConfig,
    *,
    convergence_center: Any = None,
) -> LBFGSState:
    """Minimize the same objective as the reference solver using fused MLX work.

    A callback may implement ``value_gradient_slope(x, direction)`` to evaluate
    its objective, gradient and directional derivative at one evaluation boundary.
    Plain ``fun(x) -> (host_value, device_gradient)`` callbacks remain supported.
    Compilation specializes on shape and history length; functions receive array
    arguments explicitly and never capture a training dataset in a global cache.
    For centered designs, ``convergence_center`` supplies the feature shift so
    the stopping norm is evaluated in the original feature coordinates. It
    does not alter the optimized objective, gradients, or search directions.
    """
    if convergence_center is not None and (
        convergence_center.ndim != 1 or convergence_center.shape != (x0.shape[-1] - 1,)
    ):
        raise ValueError("convergence_center must match the parameter feature dimension")
    x = x0
    objective, grad = fun(x)
    grad_max = float(_convergence_gradient_max(grad, convergence_center))
    n_fev, n_iter = 1, 0
    converged = False
    stop_reason = "max_iter"

    def result():
        return LBFGSState(
            x=x, objective=objective, grad_max_abs=grad_max, n_iter=n_iter,
            n_fev=n_fev, converged=converged, stop_reason=stop_reason,
        )

    if not math.isfinite(objective) or not math.isfinite(grad_max):
        stop_reason = "non-finite-objective"
        return result()
    if grad_max <= config.tol:
        converged, stop_reason = True, "gtol"
        return result()

    s_history, y_history, ys_history, yy_history = [], [], [], []
    directional = getattr(fun, "value_gradient_slope", None)

    for _ in range(config.max_iter):
        direction, stats = _direction(
            grad, tuple(s_history), tuple(y_history), tuple(ys_history), tuple(yy_history)
        )
        mx.eval(direction, stats)
        gtd, g_l1, gg = np.asarray(stats).astype(np.float64)
        if gtd >= 0.0 or not math.isfinite(gtd):
            s_history.clear()
            y_history.clear()
            ys_history.clear()
            yy_history.clear()
            direction, gtd = -grad, -gg
        t_init = 1.0 if s_history or g_l1 <= 0.0 else min(1.0, 1.0 / g_l1)

        def make_probe(_d, _x=x):
            def probe(t):
                candidate = _trial(_x, _d, mx.array(t, dtype=_x.dtype))
                if directional is not None:
                    value, gradient, slope = directional(candidate, _d)
                else:
                    value, gradient = fun(candidate)
                    slope = float(_slope(gradient, _d))
                return candidate, value, gradient, slope

            return probe

        search = _LineSearch(make_probe(direction), objective, float(gtd), config)
        accepted, _, x_new, f_new, g_new, _ = search.run(float(t_init))
        n_fev += search.n_fev
        remaining = config.max_line_search - search.n_fev
        if (not accepted or not f_new < objective) and s_history and remaining > 0:
            # A failed search along a quasi-Newton direction usually indicts a
            # stale history rather than the point. Retry once along the
            # gradient from a cleared history using only this iteration's
            # remaining evaluations. A failed retry remains a line-search
            # failure; it does not prove the point is optimal.
            s_history.clear()
            y_history.clear()
            ys_history.clear()
            yy_history.clear()
            direction, gtd = -grad, -gg
            t_init = min(1.0, 1.0 / g_l1) if g_l1 > 0.0 else 1.0
            search = _LineSearch(
                make_probe(direction), objective, float(gtd),
                replace(config, max_line_search=remaining),
            )
            accepted, _, x_new, f_new, g_new, _ = search.run(float(t_init))
            n_fev += search.n_fev
        if not accepted or not f_new < objective:
            stop_reason = "line-search-failure"
            break

        s, y, stats = _curvature(x_new, x, g_new, grad, convergence_center)
        mx.eval(s, y, stats)
        ys, yy, grad_max = np.asarray(stats).astype(np.float64)
        if curvature_pair_accepted(float(ys), float(yy), config.min_curvature):
            s_history.append(s)
            y_history.append(y)
            # Keep denominators on device; computing a host reciprocal and
            # rounding it to float32 would introduce an extra rounding step.
            ys_history.append(stats[0])
            yy_history.append(stats[1])
            if len(s_history) > config.history_size:
                for history in (s_history, y_history, ys_history, yy_history):
                    history.pop(0)

        x, objective, grad = x_new, f_new, g_new
        n_iter += 1
        grad_max = float(grad_max)
        if not math.isfinite(objective) or not math.isfinite(grad_max):
            stop_reason = "non-finite-objective"
            break
        if grad_max <= config.tol:
            converged, stop_reason = True, "gtol"
            break

    return result()
