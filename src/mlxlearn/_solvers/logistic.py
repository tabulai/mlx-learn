"""The penalized logistic objective and its gradient, in MLX.

Same contract as every solver here: arrays and a frozen :class:`LogisticConfig`
in, a frozen :class:`LogisticSolution` out. No estimator, no scikit-learn import,
no global configuration.

The objective is scikit-learn's, exactly
------------------------------------------
For weights ``W`` (the intercept excluded) and per-sample weights ``sw`` summing to
``sw_sum``::

    J(W, b) = (1 / sw_sum) * Σᵢ swᵢ · lossᵢ  +  0.5 · (1 / (C · sw_sum)) · ‖W‖²

That is scikit-learn's ``LinearModelLoss`` with ``l2_reg_strength = 1/(C·sw_sum)``,
where ``sw_sum`` is ``n_samples`` when unweighted and the sum of the sample weights
*after* class weights have been folded in otherwise. The intercept is never
penalized. Matching this expression exactly is what makes objective-value parity a
meaningful assertion: two solvers minimizing the same function must agree on the
value at their respective solutions even when they disagree on the argument.

``lossᵢ`` is the negative log-likelihood — ``HalfBinomialLoss`` and
``HalfMultinomialLoss`` in scikit-learn's vocabulary::

    binary       log(1 + exp(raw)) − y·raw          y ∈ {0, 1}
    multinomial  logsumexp(raw) − raw[y]

Numerical stability is not optional here
----------------------------------------
Both forms above are written with ``logaddexp``/``logsumexp`` and never exponentiate
a raw score. The ancestor computed ``sigmoid(clip(score, -80, 80))`` and then
``log(clip(p, 1e-12, 1))``; in float32 a sigmoid saturates to exactly 1.0 by
``raw ≈ 17``, so every confidently classified sample contributed the same floored
``−log(1e-12) = 27.63`` and the loss surface acquired flat plateaus that no line
search can descend. Worse, its *gradient* was the exact derivative of the
*unclipped* objective, so the value and the slope described different functions —
which is what produced line-search failures reported as ordinary non-convergence.

Here the value and the gradient are derived from the same expression, and
``p − y`` is obtained from ``mx.sigmoid`` / ``exp(raw − logsumexp(raw))``, both of
which are the exact derivatives of the stable loss forms.

Layout
------
The objective parameter is a single ``(n_targets, n_features + fit_intercept)``
MLX array. Binary GPU fits use native Metal kernels with batched Armijo search
and device-resident optimizer history; the generic fused optimizer handles the
remaining cases with host strong-Wolfe control. Both verify the direct objective
and gradient at the returned parameters. ``n_targets`` is 1 for binary
— scikit-learn fits a single weight vector for the positive class ``classes_[1]``,
not a two-row softmax — and ``n_classes`` otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.device import use_device
from .fused_lbfgs import _convergence_gradient_max, minimize
from .lbfgs import LBFGSConfig, LBFGSState

__all__ = [
    "LogisticConfig",
    "LogisticSolution",
    "solve_logistic",
    "reference_objective",
]


@dataclass(frozen=True, slots=True)
class LogisticConfig:
    """Problem-independent settings for one logistic fit.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength, scikit-learn's meaning. ``math.inf``
        expresses an unpenalized fit, which is how scikit-learn itself encodes
        ``penalty=None``.
    fit_intercept : bool, default=True
        The intercept column is appended to the parameter array and excluded from
        the penalty.
    tol : float, default=1e-4
        Tolerance on ``max|gradient|`` in the original feature coordinates.
    max_iter : int, default=100
    history_size : int, default=10
    max_line_search : int, default=32
    sketch_size : int, default=0
        Number of GPU Hessian sketch pairs to seed. Zero disables sketching.
    device : {"auto", "gpu", "cpu"}, default="auto"
        Passed straight to :func:`~mlxlearn._common.device.use_device`. Explicit,
        because a solver that reads global configuration cannot be tested in
        isolation.
    """

    C: float = 1.0
    fit_intercept: bool = True
    tol: float = 1e-4
    max_iter: int = 100
    history_size: int = 10
    max_line_search: int = 32
    device: str = "auto"
    sketch_size: int = 0

    def __post_init__(self) -> None:
        if not (self.C > 0.0):
            raise ValueError(f"C must be strictly positive, got {self.C!r}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be non-negative, got {self.max_iter}")
        if self.tol < 0 or not math.isfinite(self.tol):
            raise ValueError(f"tol must be finite and non-negative, got {self.tol}")
        if not isinstance(self.sketch_size, int) or not 0 <= self.sketch_size <= self.history_size:
            raise ValueError("sketch_size must be a nonnegative integer no greater than history_size")


@dataclass(frozen=True, slots=True)
class LogisticSolution:
    """Fitted coefficients and everything needed to explain the run.

    Attributes
    ----------
    coef : ndarray of shape (n_targets, n_features), float64
    intercept : ndarray of shape (n_targets,), float64
        Zeros when ``fit_intercept=False``, matching scikit-learn.
    n_iter : int
        Accepted L-BFGS steps.
    objective : float
        Penalized objective at the solution, as computed in float32.
    grad_max_abs : float
        ``max|gradient|`` in the original feature coordinates — the quantity
        ``tol`` bounds.
    converged : bool
    stop_reason : str
        See :class:`~mlxlearn._solvers.lbfgs.LBFGSState`.
    required_tol : float or None
        Internal gradient threshold actually used for convergence. It can be
        stricter than the requested tolerance for scale-skewed inputs.
    """

    coef: np.ndarray
    intercept: np.ndarray
    n_iter: int
    objective: float
    grad_max_abs: float
    converged: bool
    stop_reason: str
    required_tol: float | None = None


def _l2_reg_strength(C: float, sw_sum: float) -> float:
    """``1/(C·sw_sum)``, with ``C = inf`` meaning "no penalty".

    scikit-learn's own ``penalty=None`` path sets ``C_ = np.inf`` and reuses the
    same expression, so this reproduces it rather than branching on a separate
    flag that could disagree with it.
    """
    if math.isinf(C):
        return 0.0
    return 1.0 / (C * sw_sum)


# --------------------------------------------------------------------- objectives

#: Partial sums brought back per objective evaluation instead of one total.
#:
#: The line search compares objective values, so the *resolution* of the returned
#: value sets how far the solver can descend before every candidate step looks
#: like a tie. A float32 total of ``n`` log-loss terms has an absolute error of
#: order ``eps·Σloss``; divided by ``sw_sum`` that leaves roughly ``5e-8`` of
#: resolution on a mean loss of ``0.54``, and the last few L-BFGS iterations
#: — the ones that take ``max|g|`` from ``2e-4`` down past ``1e-4`` — move the
#: objective by about ``1e-8``. Measured: the solver stalled at ``max|g| = 1.9e-4``
#: after 20 iterations and then spent 80 more taking null steps.
#:
#: Reducing to this many partial sums on the device and finishing the addition in
#: float64 on the host cuts that error by ``√n_blocks``, to about ``1e-9``, which
#: is enough to reach scikit-learn's own ``gtol``. It costs one 4 KiB transfer per
#: evaluation — the same synchronization that reading the scalar already required.
_SUM_BLOCKS = 1024


def _blocked_sum(mx, values):
    """Reduce ``values`` to at most ``_SUM_BLOCKS`` partial sums, on the device."""
    n = values.shape[0]
    if n <= _SUM_BLOCKS:
        return values
    chunk = -(-n // _SUM_BLOCKS)
    padding = chunk * _SUM_BLOCKS - n
    if padding:
        # Zero padding is exact and cannot shift the total.
        values = mx.pad(values, (0, padding))
    return mx.sum(mx.reshape(values, (_SUM_BLOCKS, chunk)), axis=1)


def _total(parts, penalty, sw_sum: float) -> float:
    """Finish the reduction in float64 on the host. Assumes both are evaluated."""
    value = float(np.asarray(parts).astype(np.float64).sum() / sw_sum)
    return value if penalty is None else value + float(penalty)


@mx.compile
def _binary_arrays(theta, X, y, sw, sw_sum, l2, fit_intercept, penalized, direction):
    """Array-only loss, gradient and slope, with fusible regions compiled by MLX."""
    n_features = X.shape[1]
    w = theta[0, :n_features]
    raw = X @ w
    if fit_intercept:
        raw = raw + theta[0, n_features]

    # The signed-logit form avoids subtracting raw from logaddexp(raw) or
    # subtracting 1 from a saturated sigmoid for positive-class examples.
    sign = 1.0 - 2.0 * y
    signed_raw = sign * raw
    loss = mx.logaddexp(mx.zeros_like(signed_raw), signed_raw)
    residual = sign * mx.sigmoid(signed_raw)
    if sw is not None:
        loss = loss * sw
        residual = residual * sw

    grad_w = (residual @ X) / sw_sum
    if penalized:
        grad_w = grad_w + l2 * w
    if fit_intercept:
        grad_b = mx.sum(residual) / sw_sum
        grad_w = mx.concatenate([grad_w, mx.reshape(grad_b, (1,))])
    grad = mx.reshape(grad_w, theta.shape)
    parts = _blocked_sum(mx, loss)
    # The absent penalty must not square large finite weights: inf * 0 is NaN.
    # l2 is an explicit zero array input in the unpenalized graph.
    penalty = mx.sum(w * w) * (0.5 * l2) if penalized else l2
    slope = mx.sum(grad * direction)
    return parts, penalty, grad, slope


@mx.compile
def _multinomial_arrays(
    theta, X, y_index, y_onehot, sw, sw_sum, l2, fit_intercept, penalized, direction
):
    n_features = X.shape[1]
    W = theta[:, :n_features]
    raw = X @ W.T
    if fit_intercept:
        raw = raw + theta[:, n_features]

    # Keep logits shifted through the subtraction. Re-adding the row maximum
    # before subtracting the correct logit loses small losses in float32.
    shifted = raw - mx.max(raw, axis=1, keepdims=True)
    log_norm = mx.logsumexp(shifted, axis=1)
    correct = mx.take_along_axis(shifted, y_index, axis=1)[:, 0]
    loss = log_norm - correct
    residual = mx.exp(shifted - mx.reshape(log_norm, (-1, 1))) - y_onehot
    if sw is not None:
        loss = loss * sw
        residual = residual * mx.reshape(sw, (-1, 1))

    grad = (residual.T @ X) / sw_sum
    if penalized:
        grad = grad + l2 * W
    if fit_intercept:
        grad_b = mx.sum(residual, axis=0) / sw_sum
        grad = mx.concatenate([grad, mx.reshape(grad_b, (-1, 1))], axis=1)
    parts = _blocked_sum(mx, loss)
    penalty = mx.sum(W * W) * (0.5 * l2) if penalized else l2
    slope = mx.sum(grad * direction)
    return parts, penalty, grad, slope


def _evaluated_objective(compute, sw_sum):
    """Host precision boundary shared by ordinary and line-search evaluations."""

    def evaluate(theta, direction):
        if direction is None:
            # MLX 0.32 cannot compile a standalone constant output. Supplying a
            # zero direction also keeps ordinary/probe evaluations in one graph.
            direction = mx.zeros_like(theta)
        parts, penalty, grad, slope = compute(theta, direction)
        # One evaluation boundary for the loss, gradient and directional derivative.
        mx.eval(parts, penalty, grad, slope)
        return _total(parts, penalty, sw_sum), grad, float(slope)

    def fun(theta):
        value, gradient, _ = evaluate(theta, None)
        return value, gradient

    fun.value_gradient_slope = evaluate
    return fun


def _binary_objective(mx, X, y, sw, sw_sum, l2, fit_intercept, n_features):
    scale, penalty = mx.array(sw_sum, dtype=X.dtype), mx.array(l2, dtype=X.dtype)
    return _evaluated_objective(
        lambda theta, direction: _binary_arrays(
            theta, X, y, sw, scale, penalty, fit_intercept, bool(l2), direction
        ),
        sw_sum,
    )


@mx.compile
def _native_binary_arrays(theta, X, y, sw, denominator, l2, fit_intercept):
    from .native_logistic import direct_arrays

    loss, penalty, gradient = direct_arrays(
        theta, X, y, sw, denominator, l2, fit_intercept
    )
    return _blocked_sum(mx, loss), penalty, gradient


def _native_binary_objective(X, y, sw, sw_sum, l2, fit_intercept):
    """Direct native evaluation, independent of the optimizer's cached logits."""
    denominator = mx.array(sw_sum, dtype=X.dtype)

    def fun(theta):
        parts, penalty, gradient = _native_binary_arrays(
            theta, X, y, sw, denominator, l2, fit_intercept
        )
        mx.eval(parts, penalty, gradient)
        return _total(parts, penalty, sw_sum), gradient

    return fun


@mx.compile
def _bias_statistics(linear, bias, y, sw):
    """Block sums for an intercept-only Newton update at fixed coefficients."""
    raw = linear + bias
    sign = 1.0 - 2.0 * y
    gradient = sign * mx.sigmoid(sign * raw)
    # Multiplying the two stable sigmoid sides avoids 1 - sigmoid(raw),
    # which rounds to zero prematurely at large positive logits.
    curvature = mx.sigmoid(raw) * mx.sigmoid(-raw)
    if sw is not None:
        gradient = gradient * sw
        curvature = curvature * sw
    return _blocked_sum(mx, gradient), _blocked_sum(mx, curvature)


def _small_expm1(value):
    """Evaluate small represented steps without rounding them away."""
    # Bias updates are capped at 0.5, so class-relative changes lie in
    # [-1, 1]. The next Taylor term at |x| = 1 is < 1.7e-10, and the
    # polynomial retains tiny changes when the backend's exp(x)-1 does not.
    series = 1.0 / 479001600.0  # 1 / 12!
    for factorial in (39916800.0, 3628800.0, 362880.0, 40320.0, 5040.0,
                      720.0, 120.0, 24.0, 6.0, 2.0, 1.0):
        series = 1.0 / factorial + value * series
    return mx.where(mx.abs(value) <= 1.0, value * series, mx.expm1(value))


def _small_log1p(value):
    """Keep signed tiny differences when 1 + value rounds to 1."""
    # For bounded class-relative steps, the argument lies in
    # [exp(-1)-1, exp(1)-1]. Over [-0.7, 2], |r| <= 7/13; the first
    # omitted atanh term is < 2e-10. Near -1, use the general primitive.
    r = value / (2.0 + value)
    square = r * r
    series = 1.0 / 31.0
    for denominator in (29.0, 27.0, 25.0, 23.0, 21.0, 19.0, 17.0, 15.0,
                        13.0, 11.0, 9.0, 7.0, 5.0, 3.0):
        series = 1.0 / denominator + square * series
    approximation = 2.0 * r * (1.0 + square * series)
    return mx.where((value >= -0.7) & (value <= 2.0), approximation, mx.log1p(value))


@mx.compile
def _bias_loss_difference(linear, old_bias, new_bias, y, sw):
    """Stable loss change for the represented intercept update."""
    old_raw = linear + old_bias
    sign = 1.0 - 2.0 * y
    signed_old = sign * old_raw
    # Both intercepts are represented float32 values. Their small difference is
    # exact (Sterbenz's lemma); subtracting separately rounded logits would
    # instead quantize away individual updates and swamp the tiny loss change.
    delta = sign * (new_bias - old_bias)
    # The update is capped to |new_bias - old_bias| <= 0.5. Use the signed
    # difference directly so losses do not need to resolve a ~1e-12 change.
    difference = _small_log1p(mx.sigmoid(signed_old) * _small_expm1(delta))
    if sw is not None:
        difference = difference * sw
    return _blocked_sum(mx, difference)


def _polish_binary_intercept(X, y, sw, sw_sum, center, config, fun, state, *, require_progress=False):
    """Finish a stalled centered binary fit with certified intercept steps.

    When a large feature shift magnifies the intercept gradient, a valid
    Newton step can improve the mean loss by less than float32 loss sums can
    resolve. Evaluate its loss *difference* directly, then certify the actual
    objective and original-coordinate gradient before declaring convergence.
    Normal converged fits take no extra matrix pass.
    """
    if (
        center is None
        or state.converged
        or state.stop_reason != "line-search-failure"
        or state.n_iter >= config.max_iter
        or require_progress
        # Wide native fits also require an accepted-loss progress check. A
        # bias-only polish cannot certify that separate condition.
        or X.shape[1] >= 2048
    ):
        return state

    linear = X @ state.x[0, :-1]
    mx.eval(linear)
    n_fev = state.n_fev
    for _ in range(min(4, config.max_iter - state.n_iter)):
        old_bias = float(np.asarray(state.x)[0, -1])
        old_device = mx.array(old_bias, dtype=X.dtype)
        g_parts, h_parts = _bias_statistics(linear, old_device, y, sw)
        mx.eval(g_parts, h_parts)
        gradient = float(np.asarray(g_parts).astype(np.float64).sum()) / sw_sum
        curvature = float(np.asarray(h_parts).astype(np.float64).sum()) / sw_sum
        if not math.isfinite(gradient) or not math.isfinite(curvature) or curvature <= 0:
            break
        proposed = max(-0.5, min(0.5, -gradient / curvature))
        if not math.isfinite(proposed) or proposed == 0:
            break

        accepted = False
        for _ in range(min(8, config.max_line_search)):
            new_bias = np.float32(old_bias + proposed)
            if not math.isfinite(float(new_bias)) or float(new_bias) == old_bias:
                break
            new_device = mx.array(float(new_bias), dtype=X.dtype)
            delta_parts = _bias_loss_difference(linear, old_device, new_device, y, sw)
            mx.eval(delta_parts)
            n_fev += 1
            difference = float(np.asarray(delta_parts).astype(np.float64).sum()) / sw_sum
            if math.isfinite(difference) and difference < 0:
                candidate = mx.concatenate([state.x[:, :-1], new_device.reshape(1, 1)], axis=1)
                objective, gradient_array = fun(candidate)
                n_fev += 1
                norm = float(_convergence_gradient_max(gradient_array, center))
                if math.isfinite(objective) and math.isfinite(norm):
                    converged = norm <= config.tol
                    state = replace(
                        state,
                        x=candidate,
                        objective=objective,
                        grad_max_abs=norm,
                        n_iter=state.n_iter + 1,
                        n_fev=n_fev,
                        converged=converged,
                        stop_reason="gtol" if converged else "line-search-failure",
                    )
                    accepted = True
                    break
            proposed *= 0.5
        if not accepted or state.converged:
            break
    if state.n_fev != n_fev:
        state = replace(state, n_fev=n_fev)
    if not state.converged and state.n_iter >= config.max_iter:
        state = replace(state, stop_reason="max_iter")
    return state


def _blocked_rows_sum(mx, values):
    """Reduce rows to partial sums while preserving each class column."""
    n, n_classes = values.shape
    if n <= _SUM_BLOCKS:
        return values
    chunk = -(-n // _SUM_BLOCKS)
    padding = chunk * _SUM_BLOCKS - n
    if padding:
        values = mx.pad(values, ((0, padding), (0, 0)))
    return mx.sum(mx.reshape(values, (_SUM_BLOCKS, chunk, n_classes)), axis=1)


@mx.compile
def _multinomial_bias_statistics(linear, bias, y_onehot, sw):
    raw = linear + bias
    shifted = raw - mx.max(raw, axis=1, keepdims=True)
    exponentials = mx.exp(shifted)
    probabilities = exponentials / mx.sum(exponentials, axis=1, keepdims=True)
    residual = probabilities - y_onehot
    weighted_probabilities = probabilities
    if sw is not None:
        residual = residual * sw.reshape(-1, 1)
        weighted_probabilities = probabilities * sw.reshape(-1, 1)
    sums = mx.sum(weighted_probabilities, axis=0)
    hessian = mx.diag(sums) - probabilities.T @ weighted_probabilities
    return _blocked_rows_sum(mx, residual), hessian


@mx.compile
def _multinomial_bias_loss_difference(linear, old_bias, new_bias, y_index, sw):
    raw = linear + old_bias
    shifted = raw - mx.max(raw, axis=1, keepdims=True)
    exponentials = mx.exp(shifted)
    probabilities = exponentials / mx.sum(exponentials, axis=1, keepdims=True)
    delta = new_bias - old_bias
    selected = mx.take_along_axis(mx.broadcast_to(delta, probabilities.shape), y_index, axis=1)
    # Subtract the selected class before exponentiation, so the loss delta
    # never subtracts two nearly equal O(delta) quantities.
    difference = _small_log1p(mx.sum(probabilities * _small_expm1(delta - selected), axis=1))
    if sw is not None:
        difference = difference * sw
    return _blocked_sum(mx, difference)


def _polish_multinomial_intercepts(X, y_index, y_onehot, sw, sw_sum, center, config, fun, state):
    """Refine stalled softmax biases without changing centered weight vectors."""
    n_classes = state.x.shape[0]
    if (
        center is None
        or state.converged
        or state.stop_reason != "line-search-failure"
        or state.n_iter >= config.max_iter
        # The small dense Hessian is only worthwhile for a modest class count.
        or not 2 <= n_classes <= 64
    ):
        return state

    linear = X @ state.x[:, :-1].T
    mx.eval(linear)
    n_fev = state.n_fev
    for _ in range(min(4, config.max_iter - state.n_iter)):
        old_bias = np.asarray(state.x)[:, -1].astype(np.float32)
        old_device = mx.array(old_bias.reshape(1, -1), dtype=X.dtype)
        g_parts, hessian = _multinomial_bias_statistics(linear, old_device, y_onehot, sw)
        mx.eval(g_parts, hessian)
        gradient = np.asarray(g_parts).astype(np.float64).sum(axis=0) / sw_sum
        h = np.asarray(hessian).astype(np.float64) / sw_sum
        if not np.isfinite(gradient).all() or not np.isfinite(h).all():
            break
        # A common shift of all logits is unidentifiable. Fix the final bias
        # and solve the positive-definite principal block instead.
        try:
            direction = np.append(np.linalg.solve(h[:-1, :-1], -gradient[:-1]), 0.0)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(direction).all() or float(gradient @ direction) >= 0:
            break
        scale = min(1.0, 0.5 / max(np.max(np.abs(direction)), 1e-30))
        accepted = False
        for _ in range(min(8, config.max_line_search)):
            new_bias = (old_bias.astype(np.float64) + scale * direction).astype(np.float32)
            if not np.isfinite(new_bias).all() or np.array_equal(new_bias, old_bias):
                break
            new_device = mx.array(new_bias.reshape(1, -1), dtype=X.dtype)
            delta_parts = _multinomial_bias_loss_difference(
                linear, old_device, new_device, y_index, sw
            )
            mx.eval(delta_parts)
            n_fev += 1
            difference = float(np.asarray(delta_parts).astype(np.float64).sum()) / sw_sum
            if math.isfinite(difference) and difference < 0:
                candidate = mx.concatenate([state.x[:, :-1], new_device.reshape(-1, 1)], axis=1)
                objective, gradient_array = fun(candidate)
                n_fev += 1
                norm = float(_convergence_gradient_max(gradient_array, center))
                if math.isfinite(objective) and math.isfinite(norm):
                    converged = norm <= config.tol
                    state = replace(
                        state,
                        x=candidate,
                        objective=objective,
                        grad_max_abs=norm,
                        n_iter=state.n_iter + 1,
                        n_fev=n_fev,
                        converged=converged,
                        stop_reason="gtol" if converged else "line-search-failure",
                    )
                    accepted = True
                    break
            scale *= 0.5
        if not accepted or state.converged:
            break
    if state.n_fev != n_fev:
        state = replace(state, n_fev=n_fev)
    if not state.converged and state.n_iter >= config.max_iter:
        state = replace(state, stop_reason="max_iter")
    return state


def _needs_float64_certificate(state, center, tol, original_X, represented_X, coef):
    """Select numerically risky fits for a direct original-input check.

    The eight-epsilon margin is a conservative *heuristic*, not a global error
    bound: it catches shifts that amplify float32 intercept-gradient error and
    near-tolerance cases without scanning or scoring ordinary well-centered
    designs on the CPU. A small strided sample also detects substantial changes
    when a float64 user input was converted to the represented float32 design.
    """
    if center is None:
        return False
    scale = float(np.max(np.abs(center)))
    margin = 8.0 * float(np.finfo(np.float32).eps) * (1.0 + scale)
    if tol == 0.0 or not math.isfinite(state.grad_max_abs) or state.grad_max_abs + margin >= tol:
        return True
    if (
        isinstance(original_X, np.ndarray)
        and np.issubdtype(original_X.dtype, np.floating)
        and original_X.dtype.itemsize > np.dtype(np.float32).itemsize
        and original_X.shape == represented_X.shape
        and original_X.size
    ):
        # Advanced indexing samples arbitrary strides without flattening or
        # copying an entire Fortran/strided caller array.
        indices = np.linspace(0, original_X.size - 1, min(128, original_X.size), dtype=np.int64)
        rows, cols = np.divmod(indices, original_X.shape[1])
        difference = np.max(np.abs(
            original_X[rows, cols].astype(np.float64)
            - represented_X[rows, cols].astype(np.float64)
        ))
        coefficient_scale = float(np.max(np.sum(np.abs(coef), axis=1)))
        if not math.isfinite(float(difference)) or difference * (1.0 + coefficient_scale) >= tol * 0.25:
            return True
    return False


def _original_float64_certificate(X, y, sw, sw_sum, l2, coef, intercept, n_classes):
    """Independently score the actual exported model on the user's input.

    Float64 conversion is bounded by row blocks. This is a correctness check,
    not a CPU optimizer; it runs only when the float32 risk heuristic requests
    it. Signed binary residuals and class-relative softmax loss avoid
    cancellation for confident predictions.
    """
    data = np.asarray(X)
    if data.shape != (len(y), coef.shape[1]):
        raise ValueError("certification_X must match the design shape")
    n_samples, n_features = data.shape
    chunk_rows = max(1, 4_194_304 // max(n_features, 1))
    gradient = np.zeros_like(coef, dtype=np.float64)
    bias_gradient = np.zeros(coef.shape[0], dtype=np.float64)
    total_loss = 0.0
    codes = np.asarray(y, dtype=np.int64)
    weights = None if sw is None else np.asarray(sw, dtype=np.float64)
    for first in range(0, n_samples, chunk_rows):
        last = min(n_samples, first + chunk_rows)
        block = np.asarray(data[first:last], dtype=np.float64, order="C")
        raw = block @ coef.T + intercept
        code = codes[first:last]
        if n_classes == 2:
            sign = 1.0 - 2.0 * code
            signed_raw = sign * raw[:, 0]
            exponential = np.exp(-np.abs(signed_raw))
            probability = np.where(
                signed_raw >= 0, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential)
            )
            residual = (sign * probability)[:, None]
            loss = np.logaddexp(0.0, signed_raw)
        else:
            shifted = raw - raw.max(axis=1, keepdims=True)
            exponential = np.exp(shifted)
            probabilities = exponential / exponential.sum(axis=1, keepdims=True)
            residual = probabilities.copy()
            # Avoid subtracting one from a probability rounded to one.
            rows = np.arange(last - first)
            residual[rows, code] = 0.0
            residual[rows, code] = -residual.sum(axis=1)
            relative = raw - raw[rows, code, None]
            loss = np.logaddexp.reduce(relative, axis=1)
        if weights is not None:
            block_weights = weights[first:last]
            residual *= block_weights[:, None]
            loss *= block_weights
        gradient += residual.T @ block
        bias_gradient += residual.sum(axis=0, dtype=np.float64)
        total_loss += float(loss.sum(dtype=np.float64))
    gradient = gradient / sw_sum + l2 * coef
    bias_gradient /= sw_sum
    objective = total_loss / sw_sum + 0.5 * l2 * float(np.sum(coef * coef))
    norm = float(max(np.max(np.abs(gradient)), np.max(np.abs(bias_gradient))))
    return objective, norm


def _multinomial_objective(
    mx, X, y_index, y_onehot, sw, sw_sum, l2, fit_intercept, n_features
):
    scale, penalty = mx.array(sw_sum, dtype=X.dtype), mx.array(l2, dtype=X.dtype)
    return _evaluated_objective(
        lambda theta, direction: _multinomial_arrays(
            theta, X, y_index, y_onehot, sw, scale, penalty, fit_intercept, bool(l2), direction
        ),
        sw_sum,
    )


# ------------------------------------------------------------------------- solve


def sketch_supported(config, n_targets, n_features, initial_parts):
    """Whether a Hessian-sketch warm start can seed this fit.

    The sketch seeds the history of the native command loop, so it needs
    everything that loop needs: prepared native initial statistics, a binary
    fit with an intercept, the loop's width band, its fixed history size and
    minimum search batch, and the Metal extension on an MLX release with
    verified buffer export. The estimator uses this to drop an opted-in
    sketch that cannot be built; a direct caller asking for one is an error.
    """
    from . import native_command

    return (
        initial_parts is not None
        and n_targets == 1
        and config.fit_intercept
        and 512 <= n_features <= 4096
        and config.history_size == 10
        and config.max_line_search >= 8
        and native_command._metal_loop is not None
        and native_command._supports_metal_dlpack()
    )


def _use_native_solver(config, n_targets, n_features):
    """Bound the register workspace of the binary Metal optimizer.

    This is a numerical implementation choice within the GPU backend. Device
    availability is resolved by ``use_device`` before this predicate is used.
    The generic fused solver handles multinomial, CPU, and larger workspaces.
    """
    return (
        config.device != "cpu"
        and n_targets == 1
        and n_features <= 4096
        and 1 <= config.history_size <= 32
        and 1 <= config.max_line_search <= 32
        # The device status vector uses float32 counters, exact through 2**24.
        and (config.max_iter + 1) * config.max_line_search <= 2**24
    )


def solve_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    sample_weight: np.ndarray | None = None,
    config: LogisticConfig,
    X_device=None,
    X_center=None,
    initial_parts=None,
    certification_X=None,
) -> LogisticSolution:
    """Fit penalized logistic regression by native or fused MLX L-BFGS.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense and finite. Converted to float32 for computation.
    y : ndarray of shape (n_samples,)
        Integer class codes in ``[0, n_classes)``. Encoding labels is the caller's
        job; this function never sees the original labels.
    n_classes : int
        Two selects the binary objective — one weight vector for code 1 — and more
        selects the unconstrained multinomial objective.
    sample_weight : ndarray of shape (n_samples,), optional
        float64, non-negative, with any class weights already folded in. ``None``
        means unweighted, and takes a code path that never multiplies by ones.
    config : LogisticConfig
    X_device : mx.array, optional
        An already validated, privately owned float32 copy of ``X``. The
        estimator supplies this to reuse GPU validation and input preparation.
    X_center : mx.array, optional
        When supplied, ``X_device`` already contains ``X - X_center``. The
        exact applied shift is reused to restore the fitted intercept.
    initial_parts : mx.array, optional
        Native binary gradient/curvature partials accumulated while preparing
        ``X_device``. They use the same row blocks as the native initialization.
    certification_X : array-like, optional
        Original caller data for a risk-triggered float64 convergence check.
        Defaults to ``X``. It is never retained after this call.

    Returns
    -------
    LogisticSolution

    Notes
    -----
    The starting point is always zero. There is no warm start and no cache: the
    ancestor's continuation cache was keyed on the *address* of the training array
    and enabled by default, which made ``fit`` depend on what had been fitted
    earlier in the same process. A fit here is a pure function of its arguments.
    """
    import mlx.core as mx

    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    X_host = np.ascontiguousarray(X, dtype=NUMPY_FLOAT)
    n_samples, n_features = X_host.shape
    y_host = np.asarray(y)
    if certification_X is not None:
        original_shape = getattr(certification_X, "shape", None)
        if original_shape is not None and tuple(original_shape) != X_host.shape:
            raise ValueError("certification_X must match the design shape")

    # Centering. With an intercept this is exact, not an approximation:
    #     w·(x − μ) + b′  ≡  w·x + (b′ − w·μ)
    # so the model is unchanged and the intercept is corrected afterwards.
    #
    # Without it, features carrying a large constant offset destroy the fit in
    # float32. Measured on 2000 x 8 with an offset of 1e4 added to every feature:
    # the solver stopped after 3 iterations with an all-zero coefficient vector,
    # an objective of exactly log 2 — the null model — and 0.50 accuracy, where
    # scikit-learn reached 0.62. The gradient of the offset-dominated design is
    # numerically parallel to the constant direction, so the first step goes
    # almost entirely into the intercept and the line search then cannot find a
    # decrease. The offset also costs precision directly: at 1e4, float32 spacing
    # is ~1e-3, so the informative part of each feature is quantized away.
    #
    # Centering cannot be applied when fit_intercept is False: there is no
    # intercept to absorb the shift, so the shifted problem is a *different*
    # problem, and silently solving it would be wrong. That case keeps the raw
    # design and the docstring says so.
    center = None
    if y_host.shape != (n_samples,):
        raise ValueError(
            f"y must have shape ({n_samples},) to match X, got {y_host.shape}"
        )

    # Exact unit weights are the unweighted problem. Keep the same compiled
    # graph so fusion cannot introduce different rounding through a multiply
    # by one; class weights have already been folded in by the estimator.
    if sample_weight is not None and np.all(np.asarray(sample_weight) == 1.0):
        sample_weight = None
    if sample_weight is None:
        sw_sum = float(n_samples)
    else:
        sw_sum = float(np.sum(np.asarray(sample_weight, dtype=np.float64)))
        if sw_sum <= 0.0:
            # The wording matters: scikit-learn 1.9's
            # check_all_zero_sample_weights_error matches r"weight.*zero|zero.*weight",
            # and every estimator is expected to say so recognizably.
            raise ValueError(
                "Sample weights sum to zero: every sample weight is zero, so there "
                "is nothing to fit."
            )

    l2 = _l2_reg_strength(config.C, sw_sum)
    n_targets = 1 if n_classes == 2 else n_classes
    n_columns = n_features + int(config.fit_intercept)

    with use_device(config.device) as device:
        if X_device is not None and (
            X_device.shape != X_host.shape or X_device.dtype != mx.float32
        ):
            raise ValueError("X_device must match X's shape and have float32 dtype")
        Xm = mx.array(X_host, dtype=mx.float32) if X_device is None else X_device
        swm = (
            None
            if sample_weight is None
            else mx.array(np.asarray(sample_weight, dtype=NUMPY_FLOAT), dtype=mx.float32)
        )

        native = device == "gpu" and _use_native_solver(config, n_targets, n_features)
        if initial_parts is not None and not native:
            raise ValueError("initial_parts requires the native binary GPU solver")
        ym = mx.array(y_host, dtype=mx.float32) if n_targets == 1 else None
        if X_center is not None:
            if (
                X_device is None
                or not config.fit_intercept
                or X_center.shape != (n_features,)
                or X_center.dtype != mx.float32
            ):
                raise ValueError("X_center requires centered X_device and a float32 feature vector")
            center = X_center
        elif config.fit_intercept and n_samples > 0 and device == "gpu":
            # The estimator normally prepares the design. A direct caller gets
            # the same centered copy and, for the native binary solver, initial
            # statistics from one pass over the design. Without an intercept
            # the owned design is used as is; the native solver then collects
            # its own statistics.
            from .._kernels.logistic_input import prepare_centered

            Xm, center, _, parts = prepare_centered(Xm, ym if native else None, swm)
            if native and initial_parts is None:
                initial_parts = parts
        elif config.fit_intercept and n_samples > 0:
            center = mx.array(X_host.mean(axis=0, dtype=np.float64).astype(NUMPY_FLOAT))
            Xm = Xm - center
        if native:
            fun = _native_binary_objective(
                Xm, ym, swm, sw_sum, l2, config.fit_intercept
            )
        elif n_targets == 1:
            fun = _binary_objective(
                mx,
                Xm,
                ym,
                swm,
                sw_sum,
                l2,
                config.fit_intercept,
                n_features,
            )
        else:
            codes = y_host.astype(np.int32)
            y_index = mx.array(codes.reshape(-1, 1), dtype=mx.int32)
            y_onehot = mx.array(np.eye(n_classes, dtype=NUMPY_FLOAT)[codes], dtype=mx.float32)
            fun = _multinomial_objective(
                mx,
                Xm,
                y_index,
                y_onehot,
                swm,
                sw_sum,
                l2,
                config.fit_intercept,
                n_features,
            )

        optimizer_config = LBFGSConfig(
            max_iter=config.max_iter,
            tol=config.tol,
            history_size=config.history_size,
            max_line_search=config.max_line_search,
        )
        effective_config = config
        used_newton = False
        if native:
            from .native_logistic import (
                _conservative_config_for_skewed_curvature,
                minimize_binary,
            )

            # The scale-skew check reads the curvature diagonal of the centered
            # design; fits without an intercept keep their previous threshold.
            optimizer_config = _conservative_config_for_skewed_curvature(
                initial_parts if config.fit_intercept else None,
                n_features, sw_sum, l2, optimizer_config,
            )
            # Sketch directions can land on a different gradient-tolerance
            # contour than the scalar history. Require a tighter gradient to
            # reduce objective regressions while retaining the user's budget.
            # Do not compound this with the scale-skew adjustment above.
            if config.sketch_size and config.max_iter > 0:
                optimizer_config = replace(
                    optimizer_config, tol=min(optimizer_config.tol, 0.5 * config.tol)
                )
            if optimizer_config.tol != config.tol:
                effective_config = replace(config, tol=optimizer_config.tol)
            initial_history = None
            if config.sketch_size and config.max_iter > 0:
                from mlxlearn_device.hessian_sketch import (
                    build_sketch_history,
                    initial_metric_from_parts,
                )

                if not sketch_supported(config, n_targets, n_features, initial_parts):
                    raise ValueError("Hessian sketch requires the native GPU binary solver with prepared inputs")
                reg = mx.concatenate([mx.full(n_features, l2, dtype=Xm.dtype), mx.zeros(1, dtype=Xm.dtype)])
                _, h0inv = initial_metric_from_parts(initial_parts, reg, mx.array(sw_sum, dtype=Xm.dtype))
                initial_history = build_sketch_history(
                    Xm, swm, sw_sum, l2, config.fit_intercept, h0inv,
                    history_size=optimizer_config.history_size, sketch_size=config.sketch_size,
                    min_curvature=optimizer_config.min_curvature,
                )
            native_options = {"initial_parts": initial_parts, "convergence_center": center}
            if initial_history is not None:
                native_options["initial_history"] = initial_history
            from mlxlearn_device._newton import minimize_newton, use_newton_for_curvature

            used_newton = bool(
                config.fit_intercept and config.max_iter > 0
                # minimize_newton validates its tolerance as a normal positive
                # float32; L-BFGS accepts a zero or subnormal tolerance.
                and optimizer_config.tol >= float(np.finfo(np.float32).tiny)
                and use_newton_for_curvature(initial_parts, n_samples, n_features, sw_sum, l2)
            )
            # On very wide, scale-skewed inputs, the strict unseeded native
            # command loop can exhaust the complete budget while the MLX GPU
            # loop reaches the same stricter check from zero. Select the latter
            # up front so there is no failed full-budget attempt or hidden
            # restart. This is a workload-specific routing heuristic, not a
            # proof about general conditioning; explicit sketches still run
            # through the requested native command loop.
            if (
                initial_parts is not None and n_features >= 2048
                and config.sketch_size == 0 and optimizer_config.tol < config.tol
                and not used_newton
            ):
                native_options["use_native_loop"] = False
            state = minimize_binary(
                Xm,
                ym,
                swm,
                sw_sum,
                l2,
                config.fit_intercept,
                replace(optimizer_config, max_iter=min(32, config.max_iter)) if used_newton else optimizer_config,
                fun,
                **native_options,
            )
            if used_newton:
                remaining = config.max_iter - state.n_iter
                warm_finite = (
                    state.stop_reason != "non-finite-objective"
                    and math.isfinite(state.objective) and math.isfinite(state.grad_max_abs)
                )
                if remaining > 0 and warm_finite:
                    refinement = minimize_newton(
                        mx.array(X_host, dtype=mx.float32), Xm, center, ym,
                        mx.ones(n_samples, dtype=mx.float32) if swm is None else swm,
                        sw_sum, l2, tol=optimizer_config.tol, max_iter=remaining,
                        initial=state.x.reshape(-1),
                    )
                    state = LBFGSState(
                        x=refinement.theta.reshape(1, n_columns),
                        objective=refinement.objective, grad_max_abs=refinement.gradient_max_abs,
                        n_iter=state.n_iter + refinement.n_iter,
                        n_fev=state.n_fev + refinement.n_fev,
                        converged=refinement.converged, stop_reason=refinement.stop_reason,
                    )
                else:
                    # The warm start's gradient/progress tests do not replace
                    # the Newton route's independent objective-gap check.
                    state = replace(state, converged=False,
                                    stop_reason="max_iter" if warm_finite else "non-finite-objective")
        else:
            if config.sketch_size and config.max_iter > 0:
                raise ValueError("Hessian sketch requires the native GPU binary solver")
            if device == "gpu" and n_targets == 1 and sample_weight is not None:
                # The generic solver does not expose accepted-loss progress.
                # Use a stricter gradient safeguard for weighted binary fits.
                optimizer_config = replace(optimizer_config, tol=0.01 * config.tol)
                effective_config = replace(config, tol=optimizer_config.tol)
            theta0 = mx.zeros((n_targets, n_columns), dtype=mx.float32)
            state = minimize(fun, theta0, optimizer_config, convergence_center=center)
            if (device == "gpu" and n_targets == 1 and sample_weight is not None and config.max_iter == 0
                    and state.grad_max_abs != 0):
                state = replace(state, converged=False, stop_reason="max_iter")
        if n_targets == 1:
            polish_options = {"require_progress": True} if native and sample_weight is not None else {}
            if not used_newton:
                state = _polish_binary_intercept(
                    Xm, ym, swm, sw_sum, center, effective_config, fun, state, **polish_options,
                )
        else:
            state = _polish_multinomial_intercepts(
                Xm, y_index, y_onehot, swm, sw_sum, center, effective_config, fun, state
            )
        mx.eval(state.x)
        center_host = None if center is None else np.asarray(center).astype(np.float64)

        def exported_parameters(result):
            theta = np.asarray(result.x, dtype=NUMPY_FLOAT).astype(np.float64)
            coefficients = np.ascontiguousarray(theta[:, :n_features])
            bias = (
                np.ascontiguousarray(theta[:, n_features])
                if config.fit_intercept
                else np.zeros(n_targets, dtype=np.float64)
            )
            if center_host is not None:
                # Undo the actual applied float32 shift in float64.
                bias = bias - coefficients @ center_host
            return coefficients, bias

        coef, intercept = exported_parameters(state)
        original_X = X if certification_X is None else certification_X
        if _needs_float64_certificate(state, center_host, effective_config.tol, original_X, X_host, coef):
            objective, norm = _original_float64_certificate(
                original_X, y_host, sample_weight, sw_sum, l2, coef, intercept, n_classes
            )
            n_fev = state.n_fev + 1
            finite = math.isfinite(objective) and math.isfinite(norm)
            optimizer_failed = state.stop_reason == "non-finite-objective"
            # If float32 rounding hid a violation, try a GPU bias-only step
            # within the remaining budget. The exact certificate below still
            # decides whether that candidate is acceptable.
            if (finite and not optimizer_failed and not used_newton
                and norm > effective_config.tol and state.n_iter < config.max_iter):
                attempted = replace(
                    state, n_fev=n_fev, converged=False, stop_reason="line-search-failure"
                )
                if n_targets == 1:
                    candidate = _polish_binary_intercept(
                        Xm, ym, swm, sw_sum, center, effective_config, fun, attempted, **polish_options,
                    )
                else:
                    candidate = _polish_multinomial_intercepts(
                        Xm, y_index, y_onehot, swm, sw_sum, center, effective_config, fun, attempted
                    )
                n_fev = candidate.n_fev
                if candidate.n_iter > state.n_iter:
                    candidate_coef, candidate_intercept = exported_parameters(candidate)
                    candidate_objective, candidate_norm = _original_float64_certificate(
                        original_X, y_host, sample_weight, sw_sum, l2,
                        candidate_coef, candidate_intercept, n_classes,
                    )
                    n_fev += 1
                    if (
                        math.isfinite(candidate_objective)
                        and math.isfinite(candidate_norm)
                        and candidate_objective <= objective
                        and candidate_norm < norm
                    ):
                        state, coef, intercept = candidate, candidate_coef, candidate_intercept
                        objective, norm = candidate_objective, candidate_norm
            # On wide fits, the native solver's accepted-loss progress guard
            # is independent of the gradient; a certificate cannot bypass it.
            converged = (
                finite and not optimizer_failed and norm <= effective_config.tol
                and (state.converged or (not native and n_features < 2048
                                         and (sample_weight is None or device != "gpu")))
            )
            if not finite or optimizer_failed:
                reason = "non-finite-objective"
            elif converged:
                reason = "gtol"
            elif state.n_iter >= config.max_iter:
                reason = "max_iter"
            else:
                reason = "line-search-failure"
            state = replace(state, objective=objective, grad_max_abs=norm, n_fev=n_fev,
                            converged=converged, stop_reason=reason)
    return LogisticSolution(
        coef=coef,
        intercept=intercept,
        n_iter=state.n_iter,
        objective=state.objective,
        grad_max_abs=state.grad_max_abs,
        converged=state.converged,
        stop_reason=state.stop_reason,
        required_tol=effective_config.tol,
    )


# --------------------------------------------------------------------- reference


def reference_objective(
    coef: np.ndarray,
    intercept: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    C: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> float:
    """The same objective in float64 NumPy, for parity assertions.

    ``docs/fp32_policy.md`` §3 makes the objective value the *strong* parity
    assertion and the coefficients the weak one, because two solvers can reach
    materially different coefficients that are equally optimal on an
    ill-conditioned problem. That assertion is only meaningful if both coefficient
    sets are scored by one function at one precision — this one.

    Parameters mirror :func:`solve_logistic`; ``coef`` is ``(n_targets, n_features)``
    with ``n_targets == 1`` for binary.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._solvers.logistic import reference_objective
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> float(round(reference_objective(np.zeros((1, 1)), np.zeros(1), X, y,
    ...                                 n_classes=2, C=1.0), 12))
    0.69314718056
    """
    Xd = np.asarray(X, dtype=np.float64)
    codes = np.asarray(y).astype(np.int64)
    W = np.atleast_2d(np.asarray(coef, dtype=np.float64))
    b = np.asarray(intercept, dtype=np.float64).ravel()

    n_samples = Xd.shape[0]
    if sample_weight is None:
        sw = None
        sw_sum = float(n_samples)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw_sum = float(sw.sum())

    raw = Xd @ W.T + b
    if n_classes == 2:
        score = raw[:, 0]
        target = codes.astype(np.float64)
        loss = np.logaddexp(0.0, score) - target * score
    else:
        shift = raw.max(axis=1, keepdims=True)
        log_norm = shift[:, 0] + np.log(np.exp(raw - shift).sum(axis=1))
        loss = log_norm - raw[np.arange(n_samples), codes]

    weighted = loss if sw is None else loss * sw
    penalty = 0.0 if math.isinf(C) else 0.5 * float((W * W).sum()) / (C * sw_sum)
    return float(weighted.sum() / sw_sum + penalty)
