"""Limited-memory BFGS with a strong-Wolfe line search, on MLX arrays.

The contract is the one every solver in this package keeps: a frozen
:class:`LBFGSConfig` and a callable in, a frozen :class:`LBFGSState` out. The
solver never sees an estimator, never imports scikit-learn, and never reads global
configuration. It does not know it is fitting a logistic regression.

Three decisions here are worth stating, because the ancestor made the opposite
choice on each and each one changed the answer rather than only the runtime.

**Strong Wolfe, not Armijo-with-halving.** The ancestor's hand-written L-BFGS
accepted any step meeting the sufficient-decrease test alone. Armijo says nothing
about the curvature ``sᵀy``, so a large fraction of accepted steps produce a
curvature pair the update rule then has to discard — and a discarded pair means
the inverse-Hessian approximation stops improving while the iteration count keeps
rising. The bracket-and-zoom search below enforces ``|∇f(x+td)ᵀd| ≤ c₂|∇f(x)ᵀd|``,
which is exactly the condition that makes ``sᵀy > 0`` automatic for a convex
objective, so the history stays full and the iteration count stays near
scikit-learn's.

**A line-search failure is a stop reason, not an exception.** It is a legitimate
outcome — near the float32 floor of a well-converged problem the objective simply
stops decreasing along any direction. Raising would make an accurate answer look
like a crash; silently returning ``converged=True`` would hide a real stall. So it
returns, with ``stop_reason="line-search-failure"``, and the caller decides
whether that deserves a warning.

**The convergence test is on ``max|g|``.** That is scikit-learn's ``gtol`` for its
lbfgs path, so the same ``tol`` means the same thing here as it does there. It is
checked *after* an accepted step, and ``n_iter`` counts accepted steps only — the
ancestor incremented at the top of the loop before testing, which reported
``n_iter=1`` for a problem already converged at the starting point and counted a
failed line search as work performed.

Notes
-----
Everything the caller passes and everything returned is an ``mx.array``; only
scalars (objective values, step lengths, inner products) cross to the host, where
they are Python floats and therefore float64. That split is deliberate: the
expensive work stays on the device in float32, while the scalar bookkeeping that
decides *whether to stop* is done at full precision, so a float32 rounding wobble
in one inner product cannot masquerade as convergence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["LBFGSConfig", "LBFGSState", "minimize"]

#: ``fun(x) -> (objective, gradient)``: a Python float and an ``mx.array`` shaped
#: like ``x``.
ObjectiveGradient = Callable[[Any], "tuple[float, Any]"]


@dataclass(frozen=True, slots=True)
class LBFGSConfig:
    """Everything that shapes the iteration, and nothing about the problem.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of *accepted* steps. ``0`` evaluates the objective once and
        returns, which is how a caller asks "is my starting point already
        optimal?".
    tol : float, default=1e-4
        Stop when ``max|gradient| <= tol``. Same convention and same scale as
        scikit-learn's ``gtol`` for L-BFGS-B, so an estimator can forward its own
        ``tol`` unchanged.
    history_size : int, default=10
        Number of curvature pairs kept. scikit-learn's L-BFGS-B default is 10; a
        larger history costs memory and inner products per iteration and does not
        reliably reduce iteration count on these objectives.
    max_line_search : int, default=32
        Objective evaluations the line search may spend per iteration, across both
        the bracketing and the zoom phase.
    c1 : float, default=1e-4
        Sufficient-decrease (Armijo) constant.
    c2 : float, default=0.9
        Curvature constant. Must satisfy ``0 < c1 < c2 < 1``.
    min_curvature : float, default=1e-12
        Relative floor on ``sᵀy`` below which the curvature pair is skipped rather
        than stored. Storing a pair with ``sᵀy ≤ 0`` makes the two-loop recursion
        produce an ascent direction.
    """

    max_iter: int = 100
    tol: float = 1e-4
    history_size: int = 10
    max_line_search: int = 32
    c1: float = 1e-4
    c2: float = 0.9
    min_curvature: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be non-negative, got {self.max_iter}")
        if self.tol < 0 or not math.isfinite(self.tol):
            raise ValueError(f"tol must be finite and non-negative, got {self.tol}")
        if self.history_size < 1:
            raise ValueError(f"history_size must be at least 1, got {self.history_size}")
        if self.max_line_search < 1:
            raise ValueError(
                f"max_line_search must be at least 1, got {self.max_line_search}"
            )
        if not 0.0 < self.c1 < self.c2 < 1.0:
            raise ValueError(
                "the Wolfe constants must satisfy 0 < c1 < c2 < 1; got "
                f"c1={self.c1}, c2={self.c2}"
            )
        if self.min_curvature < 0:
            raise ValueError(
                f"min_curvature must be non-negative, got {self.min_curvature}"
            )


@dataclass(frozen=True, slots=True)
class LBFGSState:
    """The result of a run. Immutable, and it always says why it stopped.

    Attributes
    ----------
    x : mx.array
        The best point found, shaped like ``x0``.
    objective : float
        Objective value at ``x``.
    grad_max_abs : float
        ``max|gradient|`` at ``x``, the quantity ``tol`` is compared against.
    n_iter : int
        Accepted steps taken. Zero when the starting point already satisfied
        ``tol`` or when the very first line search failed.
    n_fev : int
        Objective/gradient evaluations, including the one at ``x0``.
    converged : bool
        True only when ``grad_max_abs <= tol``. A run that stopped for any other
        reason reports False, whatever its objective value looks like.
    stop_reason : str
        One of ``"gtol"``, ``"max_iter"``, ``"line-search-failure"``,
        ``"non-finite-objective"``.
    """

    x: Any
    objective: float
    grad_max_abs: float
    n_iter: int
    n_fev: int
    converged: bool
    stop_reason: str


# ------------------------------------------------------------------ device scalars


def _dot(a: Any, b: Any) -> float:
    """Inner product of two like-shaped MLX arrays, as a host float."""
    import mlx.core as mx

    return float(mx.sum(a * b))


def _max_abs(g: Any) -> float:
    import mlx.core as mx

    return float(mx.max(mx.abs(g)))


# ------------------------------------------------------------------- line search


def _cubic_minimum(
    a: float, fa: float, ga: float, b: float, fb: float, gb: float
) -> float | None:
    """Minimizer of the cubic through ``(a, fa, ga)`` and ``(b, fb, gb)``.

    Returns ``None`` when the cubic has no interior minimum, which is the signal
    for the caller to bisect instead. Written out rather than delegating to SciPy
    because ``_solvers`` deliberately has no SciPy dependency: the ancestor's
    default path was SciPy's ``fmin_l_bfgs_b`` with a bare-``except`` fallback to a
    materially different hand-written optimizer, so which algorithm ran was not
    knowable from the outside.
    """
    if a == b:
        return None
    d1 = ga + gb - 3.0 * (fa - fb) / (a - b)
    disc = d1 * d1 - ga * gb
    if disc <= 0.0 or not math.isfinite(disc):
        return None
    d2 = math.sqrt(disc) * (1.0 if b > a else -1.0)
    denom = gb - ga + 2.0 * d2
    if denom == 0.0 or not math.isfinite(denom):
        return None
    candidate = b - (b - a) * ((gb + d2 - d1) / denom)
    return candidate if math.isfinite(candidate) else None


def _safeguard(candidate: float | None, lo: float, hi: float) -> float:
    """Keep a trial step strictly inside ``[lo, hi]``, bisecting when it is not."""
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    width = high - low
    if width <= 0.0:
        return low
    margin = 0.1 * width
    if candidate is None or not (low + margin <= candidate <= high - margin):
        return low + 0.5 * width
    return candidate


class _LineSearch:
    """Bracket-then-zoom search for a step satisfying the strong Wolfe conditions.

    ``probe(t)`` must return ``(x_t, f_t, g_t, gtd_t)``. The search tracks the best
    point it saw that met sufficient decrease, so that exhausting the evaluation
    budget still yields progress instead of an abandoned iteration — the difference
    between "we could not certify this step" and "we could not move".
    """

    def __init__(self, probe, f0: float, gtd0: float, config: LBFGSConfig) -> None:
        self._probe = probe
        self._f0 = f0
        self._gtd0 = gtd0
        self._c1 = config.c1
        self._c2 = config.c2
        self._budget = config.max_line_search
        self.n_fev = 0
        #: (t, x, f, g, gtd) for the best Armijo-acceptable point seen.
        self._best: tuple[float, Any, float, Any, float] | None = None

    def _armijo(self, t: float, f: float) -> bool:
        return f <= self._f0 + self._c1 * t * self._gtd0

    def _curvature(self, gtd: float) -> bool:
        return abs(gtd) <= -self._c2 * self._gtd0

    def _evaluate(self, t: float):
        x_t, f_t, g_t, gtd_t = self._probe(t)
        self.n_fev += 1
        if math.isfinite(f_t) and self._armijo(t, f_t):
            if self._best is None or f_t < self._best[2]:
                self._best = (t, x_t, f_t, g_t, gtd_t)
        return x_t, f_t, g_t, gtd_t

    def run(self, t_init: float):
        """Return ``(accepted, t, x, f, g, strong_wolfe)``."""
        lo = None  # (t, f, gtd, x, g) — the end with the lower objective
        hi = None

        t_prev, f_prev, gtd_prev = 0.0, self._f0, self._gtd0
        x_prev = g_prev = None
        t = t_init
        first = True

        while self.n_fev < self._budget:
            x_t, f_t, g_t, gtd_t = self._evaluate(t)

            if not math.isfinite(f_t):
                # The trial point overflowed. Contracting toward the last finite
                # point is the only safe move: an infinite objective carries no
                # slope information to interpolate on.
                t = 0.5 * (t_prev + t)
                if t <= t_prev * (1.0 + 1e-12) or t <= 0.0:
                    break
                continue

            if not self._armijo(t, f_t) or (not first and f_t >= f_prev):
                lo = (t_prev, f_prev, gtd_prev, x_prev, g_prev)
                hi = (t, f_t, gtd_t, x_t, g_t)
                break
            if self._curvature(gtd_t):
                return True, t, x_t, f_t, g_t, True
            if gtd_t >= 0.0:
                # The slope turned uphill: the minimum lies between the previous
                # point and this one, with this end now the *upper* bracket.
                lo = (t, f_t, gtd_t, x_t, g_t)
                hi = (t_prev, f_prev, gtd_prev, x_prev, g_prev)
                break

            t_prev, f_prev, gtd_prev, x_prev, g_prev = t, f_t, gtd_t, x_t, g_t
            t = t * 2.5
            first = False

        if lo is None or hi is None:
            return self._give_up()

        while self.n_fev < self._budget:
            t_lo, f_lo, gtd_lo, _, _ = lo
            t_hi, f_hi, gtd_hi, _, _ = hi
            if abs(t_hi - t_lo) <= 1e-12 * max(1.0, abs(t_lo)):
                break
            t = _safeguard(
                _cubic_minimum(t_lo, f_lo, gtd_lo, t_hi, f_hi, gtd_hi), t_lo, t_hi
            )
            x_t, f_t, g_t, gtd_t = self._evaluate(t)

            if not math.isfinite(f_t) or not self._armijo(t, f_t) or f_t >= f_lo:
                hi = (t, f_t, gtd_t, x_t, g_t)
                continue
            if self._curvature(gtd_t):
                return True, t, x_t, f_t, g_t, True
            if gtd_t * (t_hi - t_lo) >= 0.0:
                hi = lo
            lo = (t, f_t, gtd_t, x_t, g_t)

        return self._give_up()

    def _give_up(self):
        """Fall back to the best sufficient-decrease point, or report failure.

        Accepting a point that met Armijo but not the curvature condition is safe:
        the caller's ``sᵀy`` test discards the curvature pair when it is not
        positive, so a weak step degrades the Hessian approximation without ever
        corrupting it.
        """
        if self._best is None:
            return False, 0.0, None, self._f0, None, False
        t, x_t, f_t, g_t, _ = self._best
        return True, t, x_t, f_t, g_t, False


# ------------------------------------------------------------------------ solver


def minimize(fun: ObjectiveGradient, x0: Any, config: LBFGSConfig) -> LBFGSState:
    """Minimize ``fun`` from ``x0`` by limited-memory BFGS.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> (objective, gradient)``. ``x`` and ``gradient`` are
        ``mx.array`` of identical shape; ``objective`` is a Python float. The pair
        must be *consistent* — the gradient must be the derivative of the value
        actually returned. A clipped objective paired with the unclipped analytic
        gradient makes the line search chase a decrease the derivative does not
        predict, which is how the ancestor produced ``ABNORMAL_TERMINATION_IN_LNSRCH``
        and then reported it as ordinary non-convergence.
    x0 : mx.array
        Starting point. Any shape; the solver treats it as a vector space and never
        reshapes it, so a caller may keep a natural ``(n_targets, n_features)``
        layout.
    config : LBFGSConfig

    Returns
    -------
    LBFGSState

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlxlearn._solvers.lbfgs import LBFGSConfig, minimize
    >>> def quadratic(x):
    ...     return float(mx.sum(x * x)), 2.0 * x
    >>> state = minimize(quadratic, mx.array([3.0, -4.0]), LBFGSConfig(tol=1e-6))
    >>> state.converged, state.stop_reason
    (True, 'gtol')
    """
    import mlx.core as mx

    x = x0
    objective, grad = fun(x)
    mx.eval(grad)
    n_fev = 1
    grad_max = _max_abs(grad)

    if not math.isfinite(objective):
        return LBFGSState(
            x=x,
            objective=objective,
            grad_max_abs=grad_max,
            n_iter=0,
            n_fev=n_fev,
            converged=False,
            stop_reason="non-finite-objective",
        )
    if grad_max <= config.tol:
        return LBFGSState(
            x=x,
            objective=objective,
            grad_max_abs=grad_max,
            n_iter=0,
            n_fev=n_fev,
            converged=True,
            stop_reason="gtol",
        )

    s_history: list[Any] = []
    y_history: list[Any] = []
    rho_history: list[float] = []

    n_iter = 0
    converged = False
    stop_reason = "max_iter"

    for _ in range(config.max_iter):
        direction = _two_loop(mx, grad, s_history, y_history, rho_history)
        gtd = _dot(grad, direction)
        if gtd >= 0.0 or not math.isfinite(gtd):
            # The stored curvature no longer describes this objective — reset to
            # steepest descent rather than taking a step that increases f.
            s_history.clear()
            y_history.clear()
            rho_history.clear()
            direction = -grad
            gtd = -_dot(grad, grad)

        # Nocedal & Wright's initial step: unit for a quasi-Newton direction,
        # scaled by 1/‖g‖₁ on the first (steepest-descent) iteration where the
        # direction carries no curvature information and unit length is arbitrary.
        if s_history:
            t_init = 1.0
        else:
            g_l1 = float(mx.sum(mx.abs(grad)))
            t_init = min(1.0, 1.0 / g_l1) if g_l1 > 0.0 else 1.0

        def probe(t: float, _x=x, _d=direction):
            x_t = _x + t * _d
            f_t, g_t = fun(x_t)
            mx.eval(g_t)
            return x_t, f_t, g_t, _dot(g_t, _d)

        search = _LineSearch(probe, objective, gtd, config)
        accepted, step, x_new, f_new, g_new, _strong = search.run(t_init)
        n_fev += search.n_fev

        if not accepted or not (f_new < objective):
            # A step that does not strictly lower the objective is a stall, not
            # progress. In float32 this is the ordinary way a well-converged run
            # ends: the remaining descent is smaller than the objective's own
            # resolution. Continuing would take null steps until max_iter and
            # report the iteration count as if work had been done.
            stop_reason = "line-search-failure"
            break

        s = x_new - x
        y = g_new - grad
        mx.eval(s, y)
        ys = _dot(y, s)
        yy = _dot(y, y)
        if ys > config.min_curvature * max(yy, 1.0):
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / ys)
            if len(s_history) > config.history_size:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

        x, objective, grad = x_new, f_new, g_new
        n_iter += 1
        grad_max = _max_abs(grad)
        if grad_max <= config.tol:
            converged = True
            stop_reason = "gtol"
            break
        if not math.isfinite(objective):
            stop_reason = "non-finite-objective"
            break

    return LBFGSState(
        x=x,
        objective=objective,
        grad_max_abs=grad_max,
        n_iter=n_iter,
        n_fev=n_fev,
        converged=converged,
        stop_reason=stop_reason,
    )


def _two_loop(mx, grad, s_history, y_history, rho_history):
    """Nocedal's two-loop recursion: ``-H∇f`` without forming ``H``.

    Returns the *descent* direction directly, i.e. the negation is folded in.
    ``gamma = sᵀy / yᵀy`` from the newest pair scales the initial Hessian, which is
    what makes the first step of each iteration well-scaled.
    """
    if not s_history:
        return -grad

    q = grad
    alphas: list[float] = []
    for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history), strict=True):
        alpha = rho * _dot(s, q)
        alphas.append(alpha)
        q = q - alpha * y

    y_last = y_history[-1]
    gamma = 1.0 / (rho_history[-1] * max(_dot(y_last, y_last), 1e-30))
    r = gamma * q

    for index, (s, y, rho) in enumerate(zip(s_history, y_history, rho_history, strict=True)):
        beta = rho * _dot(y, r)
        r = r + s * (alphas[len(alphas) - 1 - index] - beta)

    mx.eval(r)
    return -r
