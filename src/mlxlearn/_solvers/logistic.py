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
which are the exact derivatives of the stable loss forms. On the GPU the multinomial
``logsumexp`` is spelled out — ``max``, ``exp(raw − max)``, ``sum``, ``log`` — the
same shifted identity, with the value and ``p`` taken from the same exponentials;
:func:`_multinomial_objective_gpu` says why.

Layout
------
The parameter is a single ``(n_targets, n_features + fit_intercept)`` MLX array;
:func:`~mlxlearn._solvers.lbfgs.minimize` is shape-agnostic, so there is no ravel
and no chance of the feature-major/class-major transposition hazard that made the
ancestor's warm-start vectors silently meaningless. ``n_targets`` is 1 for binary
— scikit-learn fits a single weight vector for the positive class ``classes_[1]``,
not a two-row softmax — and ``n_classes`` otherwise.

The design is row-major on both devices. On the GPU, MLX's vector-matrix product
``(n,) @ (n, d)`` — the binary gradient ``residualᵀX`` — costs up to 38x the
forward ``X @ w`` on tall, narrow data: its kernel gives each threadgroup a slice
of the ``d`` output columns and has it walk all ``n`` rows, so at 16 features a
handful of threadgroups do the whole reduction while the rest of the GPU waits.
So the rows are split into ``B`` blocks of ``m`` (:func:`_row_blocks`, a function
of ``n`` alone) and the gradient is one *batched* product,
``(B, 1, m) @ (B, m, d)``: the same kernel with ``B`` times the threadgroups, each
walking ``m`` rows, whose ``B`` partial gradients the host adds in float64. Both
operands are views of the row-major arrays, so nothing is copied — the contiguous
``Xᵀ`` this replaces had to be built on the device, with a synchronization at setup
and, briefly, two copies of the design. The rows past ``B·m``, fewer than ``B``,
are one small product of their own. The multinomial gradient is a matrix-matrix
product already, except at one feature, where it is batched the same way. On the
CPU device a transposed design cost more than it saved, and these forms were
measured on the GPU only, so the CPU keeps the plain products and the float32
finish: :func:`_binary_objective_cpu`, :func:`_multinomial_objective_cpu`.

Exact rescalings
----------------
Three transformations are applied before any float32 arithmetic, all of which are
exact — they change how the problem is represented, never which problem is solved:

* **Centering, before the rounding to float32.** See :func:`_prepare_design`.
* **Sample weights divided by a power of two**, so the largest is in ``[1, 2)``.
  Weights of 1e36 are legal and scikit-learn fits them, but the float32 reductions
  of ``Σ swᵢ·xᵢ·rᵢ`` overflow long before the weights themselves do — from about
  1e36 at 2000 rows. The data term is normalized by the equally scaled
  ``sw_sum``, so ``(1/sw_sum)·Σ sw·loss`` is unchanged, and the penalty is computed
  from the *unscaled* sum, so ``l2`` is unchanged too.
* **A diagonal preconditioner from the column scales**, handed to L-BFGS as its
  initial inverse Hessian. It rescales nothing in the data; it tells the solver that
  a feature measured in units of 1e8 has curvature 1e16 times the intercept's.

A dense preconditioner where the features are correlated
--------------------------------------------------------
A diagonal cannot describe correlated features, and ten curvature pairs learn them
slowly: 21-451 evaluations where the curvature at the start of the fit, used as a
fixed model, took 6-51. So where a gate on a sample of the rows says the
features are correlated and no two classes are close to separable, L-BFGS starts
every iteration from the inverse of that curvature instead of the diagonal. See
:class:`_DensePlan`. It changes the search direction only — the objective, its
evaluation on the device and every stopping rule are the same — and a fit the gate
turns away runs the diagonal path bit for bit. A fit the model would leave short of
convergence, above the float32 floor, falls back to the diagonal path from where it
stands.
"""

from __future__ import annotations

import contextlib
import functools
import math
import threading
from dataclasses import dataclass

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.device import use_device
from .._common.host import column_mean, map_row_blocks
from ..exceptions import InvalidInputError
from .lbfgs import GRADIENT_RESOLUTION, DensePreconditioner, LBFGSConfig, minimize

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
    dense_preconditioner : bool, default=True
        Whether L-BFGS may start from the dense zero-start curvature where
        :class:`_DensePlan`'s gate says it pays. ``False`` is the diagonal path, bit
        for bit, with no sample taken.
    """

    C: float = 1.0
    fit_intercept: bool = True
    tol: float = 1e-4
    max_iter: int = 100
    history_size: int = 10
    max_line_search: int = 32
    device: str = "auto"
    dense_preconditioner: bool = True

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
        Penalized objective at the solution, as the solver evaluated it from
        float32 terms.
    grad_max_abs : float
        ``max|gradient|`` at the solution — the quantity ``tol`` bounds.
    converged : bool
    stop_reason : str
        See :class:`~mlxlearn._solvers.lbfgs.LBFGSState`.
    n_fev : int
        Objective/gradient evaluations, including the one at the start.
    initial_grad_max_abs : float
        ``max|gradient|`` at the zero start, the scale the float32 gradient floor is
        relative to.
    preconditioner : {"diagonal", "dense"}
        The initial inverse Hessian the L-BFGS iterations ended on.
    preconditioner_reason : str
        Why: ``"correlated"`` for the dense one; otherwise ``"disabled"``,
        ``"width"``, ``"uncorrelated"``, ``"separated"``, ``"non-finite"``,
        ``"not-positive-definite"`` or ``"no-step"`` (:class:`_DensePlan`), or
        ``"fallback"``: the run started on the dense one and left it for the
        diagonal where it would have stopped above the float32 floor without
        converging (:attr:`~mlxlearn._solvers.lbfgs.LBFGSState.fallback`).
    """

    coef: np.ndarray
    intercept: np.ndarray
    n_iter: int
    objective: float
    grad_max_abs: float
    converged: bool
    stop_reason: str
    n_fev: int = 0
    initial_grad_max_abs: float = math.nan
    preconditioner: str = "diagonal"
    preconditioner_reason: str = "disabled"

    @property
    def at_resolution_floor(self) -> bool:
        """Whether ``max|g|`` is below what float32 can resolve (fp32_policy §2.3)."""
        return self.grad_max_abs <= GRADIENT_RESOLUTION * self.initial_grad_max_abs


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


class _FirstEvaluation:
    """Host work that runs while the device computes a fit's first evaluation.

    The first evaluation is submitted (``mx.async_eval``), ``host`` runs, and only
    then does the host wait for the device. The device graph is the one every other
    evaluation runs, so nothing the device computes depends on it; it changes when
    the host's own work happens, not what either side computes.
    """

    __slots__ = ("host",)

    def __init__(self, host) -> None:
        self.host = host


def _evaluate(mx, outputs, first: _FirstEvaluation | None) -> None:
    """``mx.eval(*outputs)``, with ``first``'s host work overlapped the first time."""
    if first is None or first.host is None:
        mx.eval(*outputs)
        return
    host, first.host = first.host, None
    mx.async_eval(*outputs)
    host()
    mx.eval(*outputs)


#: The batched GPU gradient's row blocks: one per ``_BLOCK_ROWS`` rows, and never
#: fewer than ``_MIN_BLOCKS`` or more than ``_MAX_BLOCKS`` of them.
_BLOCK_ROWS = 4096
_MIN_BLOCKS = 16
_MAX_BLOCKS = 256


def _row_blocks(n: int) -> tuple[int, int, int]:
    """``(blocks, rows, tail)``: ``blocks`` blocks of ``rows`` rows, then ``tail`` more.

    ``tail < blocks``, and those rows get a product of their own. A function of
    ``n`` alone, so the order the gradient is summed in — and with it every bit of
    the result — never depends on the machine. Below two rows per minimum block it
    is one block: the unbatched product, which is fast at that size.
    """
    if n < 2 * _MIN_BLOCKS:
        return 1, n, 0
    blocks = min(max(n // _BLOCK_ROWS, _MIN_BLOCKS), _MAX_BLOCKS)
    rows = n // blocks
    return blocks, rows, n - blocks * rows


def _host_finish(theta, grad_w, loss_parts, grad_b, sw_sum: float, l2: float, n_features: int):
    """``(value, grad)`` in float64 from the device's evaluated partial sums.

    ``grad_w`` is ``Σ swᵢ·rᵢ·xᵢ``, ``(n_targets, n_features)``, and ``grad_b`` is
    ``Σ swᵢ·rᵢ`` per target or ``None``, both already float64 on the host. The
    penalty and its gradient are taken from ``theta`` itself, which is float32 and
    so exact in float64. ``grad`` is a NumPy array shaped like ``theta``, which
    :func:`~mlxlearn._solvers.lbfgs.minimize` reads as it reads an MLX one.
    """
    W = np.asarray(theta, dtype=np.float32).astype(np.float64)[:, :n_features]
    value = _total(loss_parts, None, sw_sum)
    grad_w = grad_w / sw_sum
    if l2:
        value += 0.5 * l2 * float(np.sum(W * W))
        grad_w = grad_w + l2 * W
    if grad_b is None:
        return value, grad_w
    return value, np.concatenate([grad_w, (grad_b / sw_sum)[:, None]], axis=1)


def _binary_terms(mx, X, y, sw, theta, fit_intercept, n_features, zero):
    """``(w, loss, residual)``: the per-sample terms both binary forms reduce."""
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
    return w, loss, residual


def _binary_objective_cpu(mx, X, y, sw, sw_sum, l2, fit_intercept, n_features, first=None):
    """``fun(theta) -> (value, grad)`` for the two-class log-loss, as the CPU runs it.

    ``theta`` is ``(1, n_features + fit_intercept)`` and ``X`` the row-major
    design. ``sw_sum`` is the normalizer of the data term — the sum of the weights
    as they are passed here. The gradient is one ``residual @ X``, finished on the
    device in float32. ``first``, when given, is :class:`_FirstEvaluation`'s host work.
    """
    zero = mx.zeros((1,), dtype=X.dtype)

    def fun(theta):
        w, loss, residual = _binary_terms(mx, X, y, sw, theta, fit_intercept, n_features, zero)

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
            _evaluate(mx, (parts, grad), first)
        else:
            _evaluate(mx, (parts, penalty, grad), first)
        return _total(parts, penalty, sw_sum), grad

    return fun


def _binary_objective_gpu(mx, X, y, sw, sw_sum, l2, fit_intercept, n_features, first=None):
    """The function :func:`_binary_objective_cpu` computes, in the form the GPU runs fast.

    The gradient is the batched product of the module docstring over the blocks of
    :func:`_row_blocks`. Everything after the device's reductions is the host's, in
    float64 and in a fixed order: the gradient partials, the ``_SUM_BLOCKS`` loss
    and residual partials (the residual's are the intercept's gradient), the
    division by ``sw_sum``, the penalty and ``l2·w``. That is more accurate than a
    float32 finish, and it replaces about six small dispatches per evaluation with
    arithmetic on ``n_features`` numbers. Either device can run it.
    """
    n = X.shape[0]
    blocks, rows, tail = _row_blocks(n)
    head = blocks * rows
    # Views of the row-major design, not copies.
    X_blocks = mx.reshape(X if head == n else X[:head], (blocks, rows, n_features))
    X_tail = X[head:] if tail else None
    zero = mx.zeros((1,), dtype=X.dtype)

    def fun(theta):
        _, loss, residual = _binary_terms(mx, X, y, sw, theta, fit_intercept, n_features, zero)

        # (B, 1, m) @ (B, m, d): one partial gradient per block.
        head_residual = residual if head == n else residual[:head]
        partials = mx.reshape(head_residual, (blocks, 1, rows)) @ X_blocks
        outputs = [partials]
        if tail:
            tail_partial = residual[head:] @ X_tail
            outputs.append(tail_partial)
        loss_parts = _blocked_sum(mx, loss)
        outputs.append(loss_parts)
        if fit_intercept:
            residual_parts = _blocked_sum(mx, residual)
            outputs.append(residual_parts)
        # One synchronization for everything the host finishes.
        _evaluate(mx, outputs, first)

        grad_w = np.asarray(partials).astype(np.float64).reshape(blocks, n_features).sum(axis=0)
        if tail:
            grad_w = grad_w + np.asarray(tail_partial).astype(np.float64)
        grad_b = None
        if fit_intercept:
            grad_b = np.asarray(residual_parts).astype(np.float64).sum(keepdims=True)
        return _host_finish(theta, grad_w[None, :], loss_parts, grad_b, sw_sum, l2, n_features)

    return fun


def _multinomial_objective_cpu(
    mx, X, y_index, y_onehot, sw, sw_sum, l2, fit_intercept, n_features, first=None
):
    """``fun(theta) -> (value, grad)`` for the softmax cross-entropy, as the CPU runs it.

    ``theta`` is ``(n_classes, n_features + fit_intercept)``; ``sw_sum`` as for
    :func:`_binary_objective_cpu`.
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
            _evaluate(mx, (parts, grad), first)
        else:
            _evaluate(mx, (parts, penalty, grad), first)
        return _total(parts, penalty, sw_sum), grad

    return fun


def _multinomial_objective_gpu(
    mx, X, y_index, y_onehot, sw, sw_sum, l2, fit_intercept, n_features, first=None
):
    """The function :func:`_multinomial_objective_cpu` computes, in the form the GPU runs fast.

    ``mx.logsumexp`` gives every row a threadgroup of its own, which a handful of
    classes leaves mostly idle, and takes its exponentials with ``fast::exp`` while
    the residual needs precise ones. Spelled out as ``max``, ``exp(raw − max)``,
    ``sum`` and ``log``, the reductions pack many short rows into a threadgroup, and
    the value and the probabilities come from the same exponentials. At one feature
    ``residualᵀX`` is the slow product of the module docstring again — the design
    against the ``(n, K)`` residuals, all ``n`` rows walked by one threadgroup — so
    it is batched over :func:`_row_blocks` as the binary gradient is; wider, it is a
    matrix-matrix product already. The finish is the host's, in float64, as in
    :func:`_binary_objective_gpu`. Either device can run it.
    """
    n = X.shape[0]
    batched = n_features == 1
    if batched:
        blocks, rows, tail = _row_blocks(n)
        head = blocks * rows
        # The design is the vector here: (B, 1, m) @ (B, m, K), views again.
        X_blocks = mx.reshape(X if head == n else X[:head], (blocks, 1, rows))
        X_tail = X[head:] if tail else None

    def fun(theta):
        n_classes = theta.shape[0]
        W = theta[:, :n_features]
        raw = X @ W.T
        if fit_intercept:
            raw = raw + theta[:, n_features]

        # logsumexp(raw) − raw[y] by the same shifted identity: exp(raw − max) is at
        # most 1, so the loss stays exact for logits of any magnitude.
        shift = mx.max(raw, axis=1, keepdims=True)
        shifted_exp = mx.exp(raw - shift)
        norm = mx.sum(shifted_exp, axis=1, keepdims=True)
        correct = mx.take_along_axis(raw, y_index, axis=1)
        loss = ((mx.log(norm) + shift) - correct)[:, 0]
        residual = shifted_exp / norm - y_onehot
        if sw is not None:
            loss = loss * sw
            residual = residual * mx.reshape(sw, (-1, 1))

        if batched:
            head_residual = residual if head == n else residual[:head]
            partials = X_blocks @ mx.reshape(head_residual, (blocks, rows, n_classes))
            outputs = [partials]
            if tail:
                tail_partial = residual[head:].T @ X_tail
                outputs.append(tail_partial)
        else:
            partials = residual.T @ X
            outputs = [partials]
        loss_parts = _blocked_sum(mx, loss)
        outputs.append(loss_parts)
        if fit_intercept:
            residual_sum = mx.sum(residual, axis=0)
            outputs.append(residual_sum)
        _evaluate(mx, outputs, first)

        grad_W = np.asarray(partials).astype(np.float64)
        if batched:
            grad_W = grad_W.reshape(blocks, n_classes).sum(axis=0).reshape(n_classes, 1)
            if tail:
                grad_W = grad_W + np.asarray(tail_partial).astype(np.float64)
        grad_b = None
        if fit_intercept:
            grad_b = np.asarray(residual_sum).astype(np.float64)
        return _host_finish(theta, grad_W, loss_parts, grad_b, sw_sum, l2, n_features)

    return fun


# ------------------------------------------------------------------------- solve


#: Rows sampled to estimate the column scales for the preconditioner. It is rounded
#: to a power of four, so an estimate is all it needs; the answer never depends on it.
_SCALE_SAMPLE_ROWS = 4096


def _prepare_design(X, fit_intercept: bool):
    """Center before rounding to float32. Returns ``(X32, center, scale)``.

    Centering with an intercept is exact, not an approximation:
        w·(x − μ) + b′  ≡  w·x + (b′ − w·μ)
    so the model is unchanged and the intercept is corrected afterwards, with the
    ``μ`` that was actually subtracted.

    Without it, features carrying a large constant offset destroy the fit in
    float32. Measured on 2000 x 8 with an offset of 1e4 added to every feature: the
    solver stopped after 3 iterations with an all-zero coefficient vector, an
    objective of exactly log 2 — the null model — and 0.50 accuracy, where
    scikit-learn reached 0.62. The gradient of the offset-dominated design is
    numerically parallel to the constant direction, so the first step goes almost
    entirely into the intercept and the line search then cannot find a decrease.

    The order matters as much as the centering. Rounding float64 input to float32
    first quantizes the *offset* — 0.0625 at 1e6, 128 for a Unix timestamp — and the
    informative part of each feature with it, before the subtraction meant to
    protect it: coefficients came out 4x attenuated on two timestamp columns. So
    float64 input is centered in float64 and only the centered values are rounded.
    NumPy buffers the mixed-precision subtraction internally, so no float64
    temporary the size of ``X`` exists. Float32 input is centered by the float32
    mean instead, in float32: for offset-dominated data the subtraction is then
    exact (Sterbenz), and the rounding of the mean itself is a constant shift per
    column that the intercept absorbs — provided the un-centering uses that same
    rounded mean, which it does.

    Centering cannot be applied when fit_intercept is False: there is no intercept
    to absorb the shift, so the shifted problem is a *different* problem, and
    silently solving it would be wrong. That case keeps the raw design.

    ``scale`` estimates each column's root-mean-square after centering, the input
    to :func:`_preconditioner`.
    """
    source = np.asarray(X)
    if source.dtype not in (np.float32, np.float64):
        source = source.astype(np.float64)
    source = np.ascontiguousarray(source)
    n_samples = source.shape[0]

    center = None
    if fit_intercept and n_samples > 0:
        # Both passes over X are split by rows across threads (_common/host.py): for
        # a fit that converges in an iteration or two they are most of its cost.
        center = column_mean(source)
        if source.dtype == np.float32:
            center = center.astype(np.float32).astype(np.float64)
        if not np.any(center != 0.0):
            center = None

    if center is None and source.dtype == NUMPY_FLOAT:
        design = source
    else:
        design = np.empty(source.shape, dtype=NUMPY_FLOAT)
        shift = None
        if center is not None:
            shift = center.astype(np.float32) if source.dtype == np.float32 else center

        def fill(start: int, stop: int) -> None:
            if shift is None:
                np.copyto(design[start:stop], source[start:stop], casting="same_kind")
            else:
                np.subtract(
                    source[start:stop], shift, out=design[start:stop], casting="same_kind"
                )

        map_row_blocks(fill, n_samples, source.shape[1] if source.ndim == 2 else 1)

    sample = design[:: max(1, n_samples // _SCALE_SAMPLE_ROWS)].astype(np.float64)
    scale = np.sqrt(np.einsum("ij,ij->j", sample, sample) / max(sample.shape[0], 1))
    return design, center, scale


#: Powers of four of curvature mismatch left to the quasi-Newton update; see
#: :func:`_preconditioner`.
_PRECONDITION_DEADBAND = 4


def _preconditioner(scale: np.ndarray, l2: float, n_targets: int, fit_intercept: bool) -> np.ndarray:
    """The diagonal initial inverse Hessian L-BFGS starts each iteration from.

    The curvature along a feature's coefficient is about ``p(1-p)·scale² + l2``,
    against ``p(1-p)`` for the intercept, so a feature in units of 1e8 beside an
    intercept of order one is a condition number of 1e16 that the quasi-Newton
    update has to discover one direction at a time. Measured on 3000 x 6 with every
    feature scaled by 1e5 or more: the coefficients converged, the intercept never
    moved, and the objective stopped 1.4e-3 above the optimum. The ratio of the two
    curvatures at the zero start, ``1 / (scale² + 4·l2)``, removes that; the
    ``l2`` term keeps a tiny, penalized column from being given a huge step the
    penalty's own curvature would not allow.

    Rounded to a power of four and shrunk toward the identity by
    ``_PRECONDITION_DEADBAND`` of them: a column within a factor of 4⁴ of the
    intercept's curvature gets exactly 1, so standardized and ordinarily scaled
    data are untouched, and one beyond it is corrected only for the excess. The
    shrinkage rather than a cutoff matters: a cutoff gives two columns of nearly the
    same scale entries 4⁴ apart, and on a six-class problem with features around 20
    that took convergence past 100 iterations where the shrunk version needs 76
    (scikit-learn: 251). On breast_cancer, raw units, it converges in 72 where
    scikit-learn needs 3267. The intercept's own entry is always 1.
    """
    curvature = scale * scale + 4.0 * l2
    exponents = np.zeros_like(curvature)
    # A column that overflowed float32 when centered has no usable scale; its
    # objective is not finite either, and the solver reports that.
    usable = np.isfinite(curvature) & (curvature > 0.0)
    exponents[usable] = np.round(0.5 * np.log2(curvature[usable]))
    exponents = np.sign(exponents) * np.maximum(np.abs(exponents) - _PRECONDITION_DEADBAND, 0.0)
    weights = np.ldexp(1.0, (-2 * exponents).astype(np.int64))
    if fit_intercept:
        weights = np.append(weights, 1.0)
    return np.tile(weights, (n_targets, 1))


def _weight_scale(sample_weight: np.ndarray) -> float:
    """The power of two that brings ``max(sample_weight)`` into ``[1, 2)``.

    Dividing by a power of two is exact, so unit and ordinary weights are left
    bitwise identical and only the representation of extreme ones changes.
    """
    largest = float(np.max(sample_weight))
    return math.ldexp(1.0, math.frexp(largest)[1] - 1)


# ----------------------------------------------------------- dense preconditioner
#
# The fit always starts from zero, where every predicted probability is 1/K. The
# Hessian of the loss there is its Fisher information, which for the logistic loss
# is also its Gauss-Newton matrix: c·XᵀWX/Σw + l2·I on the coefficients, c on the
# intercept, with c = p(1 − p) = 1/4 for two classes. For K classes it is
# (1/K)(I − 11ᵀ/K) ⊗ XᵀWX/Σw, and the model gives every class's block the same
# (1/K)(1 − 1/K) — the exact matrix on the directions the softmax gradient moves in,
# up to a scale the iteration's γₖ absorbs, and wrong only along the sum over the
# classes, in which the data term's gradient has no component. This is the matrix a
# Newton or trust-region Newton method for logistic regression works with (Lin, Weng
# & Keerthi, 2008), taken once, at the start, and held fixed: its inverse becomes
# L-BFGS's initial inverse Hessian (lbfgs.py, after Nocedal & Wright §7.2). For two
# classes it also bounds the curvature everywhere (p(1 − p) ≤ 1/4; Böhning, 1992), so
# the first step, a Newton step for it, is a conservative one.
#
# It pays where the curvature is correlated across features: ten curvature pairs
# learn it slowly, and on AR(0.99) features, low-rank data, one-hot blocks or columns
# on scales the diagonal leaves apart, fits of 21-451 evaluations took 6-51. It costs
# where classes are nearly separable: the fit ends confident, the curvature it ends with is
# a small and differently shaped fraction of the zero-start one, and there the dense
# path took more evaluations than the diagonal and stopped at a higher objective
# (digits upsampled to 20 000-200 000 rows, ten classes: 35-42 evaluations became
# 65-116). Hence a gate, in two stages.
#
# **The gate** reads a small sample spread over all the rows, or over their weight
# (_sample_rows) — at most 1 024 rows of the first 256 features — and asks whether
# the diagonal leaves curvature that the pairs learn slowly: features correlated
# within the classes (mean squared within-class correlation above _GATE_CORRELATION,
# or a Cholesky-pivot condition estimate above _GATE_CONDITION), or scales the
# diagonal's dead band leaves apart (a spread above _GATE_SPREAD). Most fits stop
# here, after 0.1-0.7 ms of host work.
#
# **The model** is then built from 8 192 rows (or four per feature) spread the same
# way, each feature's curvature floored at the whole design's (one more pass over the
# rows: what is rare, the sample misses), factorized, and asked one more question in
# its own metric: how separable are the classes? The largest eigenvalue λ of W⁻¹B —
# between-class over within-class scatter along the most separating direction, from
# the eigenvalues λ/(1 + λ) of the class means' scatter against the model's curvature
# — is at most _SEPARATION_LIMIT for the model to be used. λ = 5 is two classes whose
# means stand 2√5 within-class deviations apart, which overlap by about 1 %. The
# penalty enters through the model, so a strongly regularized fit, which cannot become
# confident, reads as less separable.
#
# Calibrated on evaluation counts, which do not depend on the machine, over 582 fits:
# 23 data laws, 2-20 classes, 569-200 000 rows, 16-256 features, C from 1e-4 to inf,
# tol 1e-4 and 1e-6, weights and fit_intercept=False; and checked on 168 more whose
# rows a sample reads wrong — rare one-hot levels, rows ordered by source or label,
# one column in units of 1e6-1e8, a few rows carrying most of the weight, a constant
# column (docs/fp32_policy.md §2.3).
#
# A model can still be wrong where no sample shows it. A run on it that would stop
# above the float32 floor without converging falls back to the diagonal instead
# (lbfgs.py), and reports "fallback".
#
# All of it runs on the host while the device computes the first evaluation
# (_FirstEvaluation), the model's factorizations on one BLAS thread
# (_one_blas_thread), and the device graph is the one every fit runs. That hides the
# gate on the fits patched dispatch routes, but not the model: its 2.2-7.7 ms at
# 60 000-200 000 rows, most of it the floor's pass over every row, outlast the first
# evaluation, so a fit the separation limit turns away pays nearly all of it (nearly
# separable blobs, ten-class digits: 0.69-0.92x the diagonal path's speed, and still
# 1.4-17x stock's; docs/fp32_policy.md §2.3).

#: Narrowest and widest designs the model is built for. At 1 024 features it is a
#: 1 025 x 1 025 factorization and its inverse.
_DENSE_MIN_FEATURES = 2
_DENSE_MAX_FEATURES = 1024
#: The model's sample: ``min(n, max(_MODEL_MIN_ROWS, _MODEL_ROWS_PER_FEATURE · d))``
#: rows (:func:`_sample_rows`).
_MODEL_MIN_ROWS = 8192
_MODEL_ROWS_PER_FEATURE = 4
#: The gate's sample: the first ``_GATE_MAX_COLUMNS`` features of at most
#: ``_GATE_ROWS`` rows (:func:`_sample_rows`), fewer on a wide block so that
#: ``rows · (columns + classes)²`` stays within ``_GATE_WORK`` — never fewer than
#: ``_GATE_MIN_ROWS``. The correlation statistic's sampling floor is about 1/rows,
#: against a gate of 0.01.
_GATE_ROWS = 1024
_GATE_MIN_ROWS = 512
_GATE_MAX_COLUMNS = 256
_GATE_WORK = 1 << 25
#: The condition estimate factorizes the leading block this wide: banded and low-rank
#: structure shows in it, and a gate that turns the fit away pays tens of
#: microseconds for it rather than the hundreds the whole block costs.
_GATE_CONDITION_COLUMNS = 64
#: The gate's thresholds, and the model's separation limit (the comment above).
_GATE_CORRELATION = 0.01
_GATE_CONDITION = 1.5
_GATE_SPREAD = 20.0
_SEPARATION_LIMIT = 5.0
#: Rows of either sample whose norm exceeds this multiple of the median row norm
#: count as if they had it. Near the optimum such rows sit far from the decision
#: boundary and carry almost no curvature, but the zero-start curvature weighs them by
#: ‖x‖², and on heavy-tailed features a handful of them would be the whole model.
#: Gaussian, column-scaled and one-hot rows do not reach it, and keep the exact
#: zero-start curvature.
_ROW_CLIP = 3.0
#: Up to this many features the model's sample is accumulated in float64; wider, in
#: float32, whose rounding the jitter below is sized for.
_FLOAT64_MODEL_MAX_FEATURES = 256
#: Ridges tried on the model's data block: what keeps the factorization defined for
#: ``C = inf`` or a rank-deficient design. Each feature's is relative to its own
#: curvature, never less than to the geometric mean of them all (:func:`_ridge`). A
#: float32 sample of one-hot rows beside an intercept has eigenvalues down to -5e-6
#: of the mean, so its ridge starts larger, and is raised if the factorization still
#: fails.
_JITTER = {"float64": (1e-10,), "float32": (1e-6, 1e-4)}
#: A feature whose variance left unexplained by the features before it is below this
#: fraction of its own variance (the factorization's pivot) is a null direction of the
#: design: one-hot blocks beside an intercept, duplicated or derived columns. The
#: model's curvature there is the ridge, and its inverse would multiply the
#: gradient's float32 rounding by 1/ridge: measured on one-hot data at ``C = inf``,
#: the same objective and predictions, and ‖coef‖ = 1 260 against 16.7. Those
#: directions get the model's mean curvature instead, as the diagonal path treats
#: them. Ten times the ridge or more; real ill-conditioning sits far above it
#: (AR(0.99): ~0.02, five latent factors under 0.3 noise: ~0.01).
_NULL_PIVOT = {"float64": 1e-6, "float32": 1e-3}
#: A feature whose curvature in the model's sample is below the whole design's by more
#: than this factor gets the whole design's (:meth:`_DensePlan._model`).
_FLOOR_SLACK = 2.0


def _zero_start_curvature(n_classes: int) -> float:
    """``p(1 − p)`` at the zero start: 1/4 for two classes, ``(1/K)(1 − 1/K)`` for K."""
    if n_classes == 2:
        return 0.25
    return (1.0 / n_classes) * (1.0 - 1.0 / n_classes)


def _gate_rows(rows: int, width: int) -> int:
    """Rows of the gate's sample for a block ``width`` columns wide (``_GATE_WORK``)."""
    return min(rows, max(_GATE_MIN_ROWS, min(_GATE_ROWS, _GATE_WORK // (width * width))))


def _sample_rows(n: int, rows: int, weights=None) -> np.ndarray:
    """``rows`` row indices spread evenly over all ``n`` rows, or over their weight.

    Unweighted, ``⌊i·n/rows⌋``: every stretch of the data is represented in proportion
    to its length. A strided prefix, ``X[:stride·rows:stride]``, never saw the last
    ``n − stride·rows`` rows — up to half of them just below twice the sample's size —
    and on data ordered by source, time or label the model and the separation limit
    then described only the head of it. At 16 383 rows, a label-sorted, nearly
    separable binary problem read λ = 0.04 where its rows in random order read 34, and
    was admitted; AR(0.99) rows followed by as many independent ones took 48
    evaluations, and 9 in random order. Spread, both read the same either way.

    Weighted, the same spacing along the cumulative weight — systematic sampling with
    probability proportional to weight — so each sampled row stands for the same weight
    and is counted once, and a row heavier than ``Σw/rows`` is drawn every time.
    Spread over rows instead, the sample held about one of ten rows carrying 90% of
    the weight in 60 000, and the model missed most of the curvature: 164 evaluations
    to the diagonal's 30, and a stop 1.9e-4 above the optimum. Unit weights give the
    unweighted indices.
    """
    if weights is None:
        return (np.arange(rows, dtype=np.int64) * n) // rows
    cumulative = np.cumsum(weights, dtype=np.float64)
    targets = np.arange(rows, dtype=np.float64) * cumulative[-1] / rows
    return np.minimum(np.searchsorted(cumulative, targets, side="right"), n - 1)


def _sample_moments(X, codes, weights, n_classes: int, rows: int, columns: int, dtype):
    """Weighted moments of ``rows`` rows (:func:`_sample_rows`) of the first ``columns``
    features.

    Returns ``(gram, sums, totals, cap)``: in float64, ``Σ w·xxᵀ``, each class's
    ``Σ w·x`` as a row, and each class's ``Σ w``, with ``w`` the row clip's shrink
    (``_ROW_CLIP``) — the sample weights are in which rows were drawn, each drawn row
    standing for the same weight; and the row norm the clip caps at, from this
    sample's median. One matrix product of ``[x | labels]`` with itself, accumulated
    in ``dtype``; the label columns are the one-hot classes, or ``[y, 1]`` for two.
    """
    n, d = X.shape
    b = min(columns, d)
    index = _sample_rows(n, rows, weights)
    k = 2 if n_classes == 2 else n_classes
    A = np.empty((rows, b + k), dtype=dtype)
    A[:, :b] = X[index, :b]
    code = codes[index]
    if n_classes == 2:
        A[:, b] = code
        A[:, b + 1] = 1.0
    else:
        A[:, b:] = code[:, None] == np.arange(n_classes)[None, :]
    norms = np.sqrt(np.einsum("ij,ij->i", A[:, :b], A[:, :b]))
    cap = _ROW_CLIP * float(np.median(norms))
    left = A
    if cap > 0.0 and float(norms.max()) > cap:
        shrink = np.minimum(1.0, (cap / np.maximum(norms, 1e-30)) ** 2).astype(dtype)
        left = A * shrink[:, None]
    M = (left.T @ A).astype(np.float64)
    gram = M[:b, :b]
    if n_classes == 2:
        positive, every = M[:b, b], M[:b, b + 1]
        sums = np.stack([every - positive, positive])
        positives, total = M[b, b + 1], M[b + 1, b + 1]
        totals = np.array([total - positives, positives])
    else:
        sums = M[b:, :b]
        totals = np.diag(M[b:, b:]).copy()
    return 0.5 * (gram + gram.T), sums, totals, cap


def _column_moments(X, weights, cap: float) -> np.ndarray:
    """Every row's weighted ``x²`` per column over the whole design, in float64.

    ``Σ w·s·x² / Σ w·s`` with ``s`` the row clip's shrink at ``cap``, as
    :func:`_sample_moments` applies it — the diagonal the model's sample estimates,
    computed exactly. One pass, split by rows across the host pool in a fixed order
    (``_common/host.py``) and with no BLAS call, so it neither depends on the thread
    count nor competes with the plan's own BLAS for threads.
    """
    n, d = X.shape

    def part(start: int, stop: int):
        # The plan's errstate (_DensePlan.run) is its own thread's, and these blocks run
        # on the pool's: an all-zero row beside rows of norm ~1e5 overflows cap² over its
        # norm here, to the shrink of 1 it gets anyway, and a fit without an intercept
        # warned — raised, under an error filter — where the diagonal path says nothing.
        with np.errstate(all="ignore"):
            block = X[start:stop]
            w = np.ones(stop - start) if weights is None else weights[start:stop]
            if cap > 0.0:
                norms = np.einsum("ij,ij->i", block, block, dtype=np.float64)
                w = w * np.minimum(1.0, cap * cap / np.maximum(norms, 1e-300))
            return np.einsum("i,ij,ij->j", w, block, block, dtype=np.float64), float(w.sum())

    parts = map_row_blocks(part, n, d)
    moments = parts[0][0].copy()
    total = parts[0][1]
    for column, weight in parts[1:]:
        moments += column
        total += weight
    return moments / max(total, 1e-300)


def _ridge(diagonal: np.ndarray, jitter: float) -> np.ndarray:
    """The ridge each feature gets: ``jitter`` times its own curvature, or times the
    geometric mean of all of them where that is larger.

    Relative to each feature's own curvature, a ridge leaves the problem's shape
    alone whatever the columns' scales. Relative to their mean, as it once was, one
    column of sixteen in units of 1e8 put it at 6e4 times a unit column's curvature:
    the model under-stepped every other feature, the separation statistic read 0.005
    where it is 1.05, and at the default ``max_iter`` the dense path stopped on it 1.4%
    to 4.6% above the optimum (60 000-200 000 rows, two to five classes), which the
    diagonal path reached in 110-180 evaluations and this ridge's in 18-38. The
    geometric mean, which no single column dominates, is the floor: centering a
    constant column in float64 leaves a residue of order 1e-13 collinear with the
    intercept, and a ridge relative to that column's curvature alone would make the
    model's step along it some 1e26 times the gradient there.
    """
    positive = diagonal[diagonal > 0.0]
    typical = float(np.exp(np.mean(np.log(positive)))) if positive.size else 1.0
    return jitter * np.maximum(diagonal, typical)


def _within(gram, sums, totals):
    """The within-class scatter: the Gram less every class's mean outer product."""
    return gram - (sums.T / np.maximum(totals, 1e-300)) @ sums


def _mean_square_correlation(M, reference) -> float:
    """Mean squared off-diagonal of ``D^-½ M D^-½``, ``D = diag(M)``, in float64.

    A column whose diagonal is below 1e-6 of ``reference``'s — a feature that encodes
    the label, whose within-class variance is only rounding — is left out rather than
    read as a correlation.
    """
    diag = np.diag(M)
    valid = (diag > 1e-6 * np.diag(reference)) & (diag > 0.0)
    count = int(valid.sum())
    if count < 2:
        return 0.0
    scale = 1.0 / diag[valid]
    block = M[np.ix_(valid, valid)]
    energy = float(scale @ ((block * block) @ scale))
    return (energy - count) / (count * (count - 1))


def _condition_estimate(within, gram) -> float:
    """``max/min`` of the Cholesky pivots of the within-class correlation matrix.

    Each pivot is the variance of a feature left unexplained by the ones before it: 1
    for independent features, ``1 − ρ²`` for AR(ρ). A 1e-4 ridge bounds an exact
    dependency's — the gate's sample is accumulated in float32 — and a matrix that
    still does not factorize counts as ill-conditioned. Columns left out as in
    :func:`_mean_square_correlation`.
    """
    diag = np.diag(within)
    valid = (diag > 1e-6 * np.diag(gram)) & (diag > 0.0)
    if int(valid.sum()) < 2:
        return 1.0
    inv = 1.0 / np.sqrt(diag[valid])
    R = within[np.ix_(valid, valid)] * inv[:, None] * inv[None, :]
    R[np.diag_indices_from(R)] += 1e-4
    try:
        pivots = np.diag(np.linalg.cholesky(R)) ** 2
    except np.linalg.LinAlgError:
        return math.inf
    return float(pivots.max() / pivots.min())


#: Serializes the one-thread BLAS limit between plans: it is process-wide state of the
#: BLAS library, which two plans on two threads would otherwise save and restore out of
#: order, leaving the process on one thread. It cannot order anyone else's limit; see
#: :func:`_one_blas_thread`.
_BLAS_LIMIT_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _blas_pools():
    """threadpoolctl's controller for the BLAS pools the plan can reach, or ``None``.

    Built once per process — a scan of the loaded libraries, about 10 ms — and keyed
    on nothing: a description of the process's libraries, never of data, as the
    kernel factories in ``_kernels`` cache compiled kernels. SciPy is imported first
    so that a SciPy shipping its own BLAS is in it. Accelerate has no pool to
    control, and then this is ``None``. threadpoolctl is scikit-learn's dependency.
    """
    import scipy.linalg  # noqa: F401
    from threadpoolctl import ThreadpoolController

    pools = ThreadpoolController().select(user_api="blas")
    return pools if pools.lib_controllers else None


@contextlib.contextmanager
def _one_blas_thread():
    """The model's BLAS and LAPACK calls on one thread.

    Its products and factorizations are at most 8 192 x 1 044 and 1 025 x 1 025, and
    run while the device computes. On a stack whose NumPy and SciPy each bring an
    OpenBLAS (NumPy 1.23 and SciPy 1.10 wheels: two pools of 16 threads), a 65 x 65
    Cholesky factorization took up to 50 ms instead of 0.04, and a plan for 60 000 x 64
    rows of three classes a median 0.28 s instead of 4 ms.

    The limit is process-wide, and threadpoolctl restores on exit what it found on
    entry, so another thread's limit that starts inside this one and ends after it
    restores this one's single thread, for good: scikit-learn's ``KMeans`` sets one
    around its iterations. The lock orders mlxlearn's plans only. So the limit is held
    as briefly as the plan allows — around the model, which only a fit the gate admits
    builds, never around the gate, whose one product and 64 x 64 factorization call
    NumPy's BLAS alone — and not at all where every pool already runs one thread, or
    where there is no pool to limit (Accelerate: the NumPy 2.2-2.5 and SciPy 1.15-1.18
    wheels for macOS).
    """
    pools = _blas_pools()
    if pools is None:
        yield
        return
    with _BLAS_LIMIT_LOCK:
        if all(pool.num_threads <= 1 for pool in pools.lib_controllers):
            yield
            return
        with pools.limit(limits=1):
            yield


class _DensePlan:
    """The dense initial inverse Hessian of one fit: the gate, the model, the operator.

    :meth:`run` is the host's work, done while the device computes the first
    evaluation; :meth:`resolve` hands its result to L-BFGS afterwards — a
    :class:`~mlxlearn._solvers.lbfgs.DensePreconditioner`, or ``None`` to keep the
    diagonal. ``reason`` says which and why: ``"correlated"`` (the operator),
    ``"uncorrelated"`` (the gate), ``"separated"`` (the model's separation limit),
    ``"non-finite"`` or ``"not-positive-definite"``. ``stats`` holds what the
    decision was made from.

    ``X`` is the float32 design the device holds (centered when there is an
    intercept), ``codes`` the class codes, ``weights`` the sample weights as the
    objective takes them, and ``diagonal`` the diagonal preconditioner the fit would
    otherwise use, whose dead band the gate's spread is measured after.
    """

    def __init__(self, X, codes, weights, *, n_classes, fit_intercept, l2, diagonal):
        self.X, self.codes, self.weights = X, codes, weights
        self.n_classes, self.fit_intercept, self.l2 = n_classes, fit_intercept, l2
        self.diagonal = diagonal
        self.reason = "pending"
        self.stats: dict[str, float] = {}
        self._operator: DensePreconditioner | None = None

    def run(self) -> None:
        # Floating-point flags are not warnings here: every result the plan uses is
        # checked for finiteness — a row that overflowed float32 makes the sample
        # non-finite, and the fit's own objective reports it — and NumPy 2.2 on
        # Accelerate raises spurious ones from matrix products.
        with np.errstate(all="ignore"):
            if self._gate():
                with _one_blas_thread():
                    self._operator = self._model()

    def resolve(self) -> DensePreconditioner | None:
        return self._operator

    def _gate(self) -> bool:
        n, d = self.X.shape
        rows = min(n, max(_MODEL_MIN_ROWS, _MODEL_ROWS_PER_FEATURE * d))
        k = 2 if self.n_classes == 2 else self.n_classes
        rows = _gate_rows(rows, min(d, _GATE_MAX_COLUMNS) + k)
        gram, sums, totals, _ = _sample_moments(
            self.X, self.codes, self.weights, self.n_classes, rows, _GATE_MAX_COLUMNS,
            np.float32,
        )
        if not (np.all(np.isfinite(gram)) and np.all(np.isfinite(sums))):
            self.reason = "non-finite"
            return False
        within = _within(gram, sums, totals)
        correlation = _mean_square_correlation(within, gram)
        b = min(gram.shape[0], _GATE_CONDITION_COLUMNS)
        # b independent columns from m rows give an estimate of about m/(m − b): the
        # last pivot keeps (m − b)/m of its variance by chance alone.
        condition = _condition_estimate(within[:b, :b], gram[:b, :b]) / (
            rows / max(rows - min(b, rows - 1), 1)
        )
        spread = self._spread(np.diag(gram) / max(float(totals.sum()), 1e-300))
        self.stats.update(
            gate_rows=rows, correlation=correlation, condition=condition, spread=spread
        )
        if (
            correlation > _GATE_CORRELATION
            or condition > _GATE_CONDITION
            or spread > _GATE_SPREAD
        ):
            return True
        self.reason = "uncorrelated"
        return False

    def _spread(self, variance) -> float:
        """``max/min`` of the curvature each coordinate presents to L-BFGS after the
        diagonal preconditioner — what its power-of-four dead band leaves, up to 16x in
        scale — over the gate's columns and the intercept. ``nan`` when undefined."""
        c = _zero_start_curvature(self.n_classes)
        seen = self.diagonal[0, : variance.shape[0]] * (c * variance + self.l2)
        seen = seen[np.isfinite(seen) & (variance > 0.0)]
        if self.fit_intercept:
            seen = np.append(seen, c)
        if not seen.size or not seen.min() > 0.0:
            return math.nan
        return float(seen.max() / seen.min())

    def _model(self) -> DensePreconditioner | None:
        from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular

        n, d = self.X.shape
        rows = min(n, max(_MODEL_MIN_ROWS, _MODEL_ROWS_PER_FEATURE * d))
        precision = "float64" if d <= _FLOAT64_MODEL_MAX_FEATURES else "float32"
        gram, sums, totals, cap = _sample_moments(
            self.X, self.codes, self.weights, self.n_classes, rows, d,
            np.float64 if precision == "float64" else np.float32,
        )
        total = float(totals.sum())
        c = _zero_start_curvature(self.n_classes)
        p = d + int(self.fit_intercept)
        features = np.arange(d)
        H = np.zeros((p, p))
        H[:d, :d] = c * gram / max(total, 1e-300)
        floored = 0
        if rows < n:
            # The sample cannot see what is rare in the data: a one-hot level of a few
            # rows in 60 000. Its curvature along such a feature is then c·p² (the
            # centering's) or nothing, where the data's is about c·p, and the model's
            # step along it orders of magnitude too long. So a feature the sample reads
            # at less than 1/_FLOOR_SLACK of the whole design's curvature gets the whole
            # design's, whose diagonal one pass computes exactly. Sampling noise stays
            # well inside that factor, and so do the exact dependencies the null
            # directions below are found from.
            floor = c * _column_moments(self.X, self.weights, cap)
            low = np.flatnonzero(np.diag(H)[:d] * _FLOOR_SLACK < floor)
            H[low, low] = floor[low]
            floored = int(low.size)
        mean = float(np.trace(H[:d, :d])) / d
        if self.fit_intercept:
            H[d, d] = c
            if self.weights is not None:
                # The design is centered by its unweighted mean, so under weights the
                # intercept's curvature is coupled to the coefficients'.
                H[:d, d] = H[d, :d] = c * sums.sum(axis=0) / max(total, 1e-300)
        if not (np.all(np.isfinite(H)) and mean > 0.0):
            self.reason = "non-finite"
            return None

        factor = threshold = None
        for jitter in _JITTER[precision]:
            # The data's curvature alone first: its pivots find the null directions.
            trial = H.copy()
            trial[features, features] += _ridge(np.diag(H)[:d], jitter)
            try:
                factor = cholesky(trial, lower=True, check_finite=False)
            except LinAlgError:
                continue
            H = trial
            threshold = _NULL_PIVOT[precision] * jitter / _JITTER[precision][0]
            break
        if factor is None:
            self.reason = "not-positive-definite"
            return None
        pivots = np.diag(factor)
        null = np.flatnonzero(pivots[:d] ** 2 < threshold * np.diag(H)[:d])
        H[features, features] += self.l2
        if null.size:
            # v = L₁⁻ᵀeᵢ, with L₁ the unit lower factor, is the null direction behind
            # pivot i: H·v = Dᵢᵢ²·L₁eᵢ ≈ 0.
            E = np.zeros((p, null.size))
            E[null, np.arange(null.size)] = 1.0
            V = solve_triangular(
                (factor / pivots[None, :]).T, E, lower=False, unit_diagonal=True,
                check_finite=False,
            )
            U, _ = np.linalg.qr(V)
            H += mean * (U @ U.T)
        try:
            factor = cholesky(H, lower=True, check_finite=False)
        except LinAlgError:
            self.reason = "not-positive-definite"
            return None

        # The classes' separation in the model's metric. With D the class means'
        # deviations, weighted by √(share), the eigenvalues of c·DᵀH⁻¹D are
        # λ/(1 + λ) for the eigenvalues λ of W⁻¹B (H ≈ c·(W + B)).
        shares = totals / max(total, 1e-300)
        means = sums / np.maximum(totals, 1e-300)[:, None]
        deviations = np.zeros((p, shares.shape[0]))
        deviations[:d] = ((means - shares @ means) * np.sqrt(shares)[:, None]).T
        Z = solve_triangular(factor, deviations, lower=True, check_finite=False)
        ratio = float(np.linalg.eigvalsh(c * (Z.T @ Z))[-1])
        separation = ratio / (1.0 - ratio) if ratio < 1.0 else math.inf
        self.stats.update(
            rows=rows, floored=floored, null_directions=int(null.size), separation=separation
        )
        if not separation < _SEPARATION_LIMIT:
            self.reason = "separated"
            return None

        inverse = cho_solve((factor, True), np.eye(p), check_finite=False)
        inverse = 0.5 * (inverse + inverse.T)
        n_targets = 1 if self.n_classes == 2 else self.n_classes

        def apply(v: np.ndarray) -> np.ndarray:
            # L-BFGS calls this outside run()'s errstate, on every iteration, and
            # under the one its two-loop recursion holds (lbfgs._direction). The
            # inverse is finite — factorized from a finite, positive-definite model —
            # and so is v when the gradient is, so a flag NumPy's BLAS raises here is
            # spurious (lbfgs._dot). A direction that really is not finite is found
            # without it: its slope resets the search, and a run it stalls above the
            # float32 floor leaves this operator for the diagonal.
            return (v.reshape(n_targets, p) @ inverse).reshape(-1)

        self.reason = "correlated"
        return DensePreconditioner(apply)


def solve_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    sample_weight: np.ndarray | None = None,
    config: LogisticConfig,
    callback=None,
) -> LogisticSolution:
    """Fit penalized logistic regression by L-BFGS.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense and finite, float64 or float32. Centered in float64 when there is an
        intercept, then rounded to float32 for computation.
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
    callback : callable, optional
        Called as ``callback(coef, intercept)`` after every accepted L-BFGS step,
        with float64 arrays in the caller's units — the same shapes and meaning as
        :attr:`LogisticSolution.coef` and ``.intercept``.

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

    y_host = np.asarray(y)
    n_samples = np.shape(X)[0]
    if y_host.shape != (n_samples,):
        raise ValueError(
            f"y must have shape ({n_samples},) to match X, got {y_host.shape}"
        )

    X_host, center, scale = _prepare_design(X, config.fit_intercept)
    n_features = X_host.shape[1]

    if sample_weight is None:
        sw_sum = float(n_samples)
        weights = None
        normalizer = sw_sum
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        sw_sum = float(np.sum(weights))
        if sw_sum <= 0.0:
            # The wording matters: scikit-learn 1.9's
            # check_all_zero_sample_weights_error matches r"weight.*zero|zero.*weight",
            # and every estimator is expected to say so recognizably.
            raise InvalidInputError(
                "Sample weights sum to zero: every sample weight is zero, so there "
                "is nothing to fit."
            )
        weight_scale = _weight_scale(weights)
        weights = weights / weight_scale
        normalizer = sw_sum / weight_scale

    # From the unscaled sum: the penalty is 1/(C·sw_sum) whatever the weights'
    # representation.
    l2 = _l2_reg_strength(config.C, sw_sum)
    n_targets = 1 if n_classes == 2 else n_classes
    n_columns = n_features + int(config.fit_intercept)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coef = np.array(theta[:, :n_features], dtype=np.float64, order="C")
        if config.fit_intercept:
            intercept = np.asarray(theta[:, n_features], dtype=np.float64)
            if center is not None:
                # Undo the centering exactly: b = b′ − w·μ, in float64, with the μ
                # that was actually subtracted. Both lie in float32's range — an
                # iterate the objective was finite at, means of range-checked
                # columns — so a flag NumPy's BLAS raises on w·μ is spurious
                # (lbfgs._dot).
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    shift = coef @ center
                intercept = intercept - shift
        else:
            intercept = np.zeros(n_targets, dtype=np.float64)
        return coef, np.ascontiguousarray(intercept)

    iteration_callback = None
    if callback is not None:

        def iteration_callback(theta):
            callback(*unpack(theta))

    diagonal = _preconditioner(scale, l2, n_targets, config.fit_intercept)
    plan = first = None
    reason = "disabled"
    if config.dense_preconditioner:
        if _DENSE_MIN_FEATURES <= n_features <= _DENSE_MAX_FEATURES:
            plan = _DensePlan(
                X_host, y_host, weights, n_classes=n_classes,
                fit_intercept=config.fit_intercept, l2=l2, diagonal=diagonal,
            )
            first = _FirstEvaluation(plan.run)
        else:
            reason = "width"

    with use_device(config.device) as stream:
        # use_device yields the block's stream, not the device's name.
        on_gpu = stream.device == mx.gpu
        Xm = mx.array(X_host, dtype=mx.float32)
        swm = None if weights is None else mx.array(weights.astype(NUMPY_FLOAT))

        if n_targets == 1:
            objective = _binary_objective_gpu if on_gpu else _binary_objective_cpu
            fun = objective(
                mx,
                Xm,
                mx.array(y_host.astype(NUMPY_FLOAT), dtype=mx.float32),
                swm,
                normalizer,
                l2,
                config.fit_intercept,
                n_features,
                first,
            )
        else:
            codes = y_host.astype(np.int32)
            objective = _multinomial_objective_gpu if on_gpu else _multinomial_objective_cpu
            fun = objective(
                mx,
                Xm,
                mx.array(codes.reshape(-1, 1), dtype=mx.int32),
                mx.array(
                    np.eye(n_classes, dtype=NUMPY_FLOAT)[codes], dtype=mx.float32
                ),
                swm,
                normalizer,
                l2,
                config.fit_intercept,
                n_features,
                first,
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
            callback=iteration_callback,
            preconditioner=diagonal,
            dense_preconditioner=None if plan is None else plan.resolve,
        )
        theta = np.asarray(state.x, dtype=NUMPY_FLOAT).astype(np.float64)

    if plan is not None:
        reason = plan.reason
        if state.fallback:
            reason = "fallback"
        elif reason == "correlated" and state.preconditioner != "dense":
            reason = "no-step"
    coef, intercept = unpack(theta)
    return LogisticSolution(
        coef=coef,
        intercept=intercept,
        n_iter=state.n_iter,
        objective=state.objective,
        grad_max_abs=state.grad_max_abs,
        converged=state.converged,
        stop_reason=state.stop_reason,
        n_fev=state.n_fev,
        initial_grad_max_abs=state.initial_grad_max_abs,
        preconditioner="dense" if state.preconditioner == "dense" else "diagonal",
        preconditioner_reason=reason,
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

    # A score that really overflowed makes the objective returned here non-finite,
    # which is what a caller compares; a flag NumPy's BLAS raises on finite scores
    # is spurious (lbfgs._dot).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        raw = Xd @ W.T
    raw += b
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
