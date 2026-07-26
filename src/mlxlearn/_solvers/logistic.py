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
The parameter is a single ``(n_targets, n_features + fit_intercept)`` MLX array;
:func:`~mlxlearn._solvers.lbfgs.minimize` is shape-agnostic, so there is no ravel
and no chance of the feature-major/class-major transposition hazard that made the
ancestor's warm-start vectors silently meaningless. ``n_targets`` is 1 for binary
— scikit-learn fits a single weight vector for the positive class ``classes_[1]``,
not a two-row softmax — and ``n_classes`` otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.device import use_device
from .lbfgs import LBFGSConfig, minimize

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
        Forwarded to L-BFGS as the tolerance on ``max|gradient|``.
    max_iter : int, default=100
    history_size : int, default=10
    max_line_search : int, default=32
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

    def __post_init__(self) -> None:
        if not (self.C > 0.0):
            raise ValueError(f"C must be strictly positive, got {self.C!r}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be non-negative, got {self.max_iter}")
        if self.tol < 0 or not math.isfinite(self.tol):
            raise ValueError(f"tol must be finite and non-negative, got {self.tol}")


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
        ``max|gradient|`` at the solution — the quantity ``tol`` bounds.
    converged : bool
    stop_reason : str
        See :class:`~mlxlearn._solvers.lbfgs.LBFGSState`.
    """

    coef: np.ndarray
    intercept: np.ndarray
    n_iter: int
    objective: float
    grad_max_abs: float
    converged: bool
    stop_reason: str


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


def _binary_objective(mx, X, y, sw, sw_sum, l2, fit_intercept, n_features):
    """``fun(theta) -> (value, grad)`` for the two-class log-loss.

    ``theta`` is ``(1, n_features + fit_intercept)``.
    """
    zero = mx.zeros((1,), dtype=X.dtype)

    def fun(theta):
        w = theta[0, :n_features]
        raw = X @ w
        if fit_intercept:
            raw = raw + theta[0, n_features]

        # log(1 + exp(raw)) − y·raw, evaluated without ever forming exp(raw).
        loss = mx.logaddexp(zero, raw) - y * raw
        residual = mx.sigmoid(raw) - y
        if sw is not None:
            loss = loss * sw
            residual = residual * sw

        grad_w = (residual @ X) / sw_sum
        if l2:
            grad_w = grad_w + l2 * w
        if fit_intercept:
            grad_b = mx.sum(residual) / sw_sum
            grad = mx.concatenate([grad_w, mx.reshape(grad_b, (1,))])
        else:
            grad = grad_w
        grad = mx.reshape(grad, (1, theta.shape[1]))

        parts = _blocked_sum(mx, loss)
        penalty = mx.sum(w * w) * (0.5 * l2) if l2 else None
        # One synchronization for the value and the gradient together.
        if penalty is None:
            mx.eval(parts, grad)
        else:
            mx.eval(parts, penalty, grad)
        return _total(parts, penalty, sw_sum), grad

    return fun


def _multinomial_objective(mx, X, y_index, y_onehot, sw, sw_sum, l2, fit_intercept, n_features):
    """``fun(theta) -> (value, grad)`` for the softmax cross-entropy.

    ``theta`` is ``(n_classes, n_features + fit_intercept)``.
    """

    def fun(theta):
        W = theta[:, :n_features]
        raw = X @ W.T
        if fit_intercept:
            raw = raw + theta[:, n_features]

        # logsumexp(raw) − raw[y]: the shifted-softmax identity, done by MLX, so
        # the per-sample loss stays exact for logits of any magnitude instead of
        # flooring once a probability underflows.
        log_norm = mx.logsumexp(raw, axis=1)
        correct = mx.take_along_axis(raw, y_index, axis=1)[:, 0]
        loss = log_norm - correct
        residual = mx.exp(raw - mx.reshape(log_norm, (-1, 1))) - y_onehot
        if sw is not None:
            loss = loss * sw
            residual = residual * mx.reshape(sw, (-1, 1))

        grad_W = (residual.T @ X) / sw_sum
        if l2:
            grad_W = grad_W + l2 * W
        if fit_intercept:
            grad_b = mx.sum(residual, axis=0) / sw_sum
            grad = mx.concatenate([grad_W, mx.reshape(grad_b, (-1, 1))], axis=1)
        else:
            grad = grad_W

        parts = _blocked_sum(mx, loss)
        penalty = mx.sum(W * W) * (0.5 * l2) if l2 else None
        if penalty is None:
            mx.eval(parts, grad)
        else:
            mx.eval(parts, penalty, grad)
        return _total(parts, penalty, sw_sum), grad

    return fun


# ------------------------------------------------------------------------- solve


def solve_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    sample_weight: np.ndarray | None = None,
    config: LogisticConfig,
) -> LogisticSolution:
    """Fit penalized logistic regression by L-BFGS.

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
    if y_host.shape != (n_samples,):
        raise ValueError(
            f"y must have shape ({n_samples},) to match X, got {y_host.shape}"
        )

    if sample_weight is None:
        sw_sum = float(n_samples)
    else:
        sw_sum = float(np.sum(np.asarray(sample_weight, dtype=np.float64)))
        if sw_sum <= 0.0:
            raise ValueError("sample_weight must sum to a strictly positive value.")

    l2 = _l2_reg_strength(config.C, sw_sum)
    n_targets = 1 if n_classes == 2 else n_classes
    n_columns = n_features + int(config.fit_intercept)

    with use_device(config.device):
        Xm = mx.array(X_host, dtype=mx.float32)
        swm = (
            None
            if sample_weight is None
            else mx.array(np.asarray(sample_weight, dtype=NUMPY_FLOAT), dtype=mx.float32)
        )

        if n_targets == 1:
            fun = _binary_objective(
                mx,
                Xm,
                mx.array(y_host.astype(NUMPY_FLOAT), dtype=mx.float32),
                swm,
                sw_sum,
                l2,
                config.fit_intercept,
                n_features,
            )
        else:
            codes = y_host.astype(np.int32)
            fun = _multinomial_objective(
                mx,
                Xm,
                mx.array(codes.reshape(-1, 1), dtype=mx.int32),
                mx.array(
                    np.eye(n_classes, dtype=NUMPY_FLOAT)[codes], dtype=mx.float32
                ),
                swm,
                sw_sum,
                l2,
                config.fit_intercept,
                n_features,
            )

        theta0 = mx.zeros((n_targets, n_columns), dtype=mx.float32)
        state = minimize(
            fun,
            theta0,
            LBFGSConfig(
                max_iter=config.max_iter,
                tol=config.tol,
                history_size=config.history_size,
                max_line_search=config.max_line_search,
            ),
        )
        mx.eval(state.x)
        theta = np.asarray(state.x, dtype=NUMPY_FLOAT).astype(np.float64)

    coef = np.ascontiguousarray(theta[:, :n_features])
    intercept = (
        np.ascontiguousarray(theta[:, n_features])
        if config.fit_intercept
        else np.zeros(n_targets, dtype=np.float64)
    )
    return LogisticSolution(
        coef=coef,
        intercept=intercept,
        n_iter=state.n_iter,
        objective=state.objective,
        grad_max_abs=state.grad_max_abs,
        converged=state.converged,
        stop_reason=state.stop_reason,
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
