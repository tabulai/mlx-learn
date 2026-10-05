"""Limited-memory BFGS with a strong-Wolfe line search, for objectives evaluated in MLX.

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
outcome. Raising would make an accurate answer look like a crash; silently
returning ``converged=True`` would hide a real stall. So it returns, with a
``stop_reason``, and the caller decides whether that deserves a warning.

**The convergence test is on ``max|g|``.** That is scikit-learn's ``gtol`` for its
lbfgs path, so the same ``tol`` means the same thing here as it does there. It is
checked *after* an accepted step, and ``n_iter`` counts accepted steps only — the
ancestor incremented at the top of the loop before testing, which reported
``n_iter=1`` for a problem already converged at the starting point and counted a
failed line search as work performed.

Where the work happens
----------------------
The objective and its gradient — the ``O(n·d)`` part — are MLX, in float32, on
whatever device the caller chose. **Everything else is NumPy float64 on the host**:
the two-loop recursion, every inner product, the curvature test, ``max|g|``. The
parameter vector is small next to the data — ``n_targets × (n_features + 1)``
floats for a logistic regression — so moving it costs nothing, while keeping that
bookkeeping in MLX cost a blocking device round trip
(~120 µs on the GPU) for *every* inner product: ``2m + 1`` of them in the two-loop
recursion alone, and 12 to 33 per iteration in total. The line search averages
about 1.1 evaluations per iteration, so those scalar reads, not the line search,
were what made an iteration cost milliseconds. Now each evaluation of ``fun``
performs exactly one synchronization, the one that brings back its value and
gradient.

Doing the bookkeeping in float64 is also the more accurate choice: the scalar
decisions about *whether to stop* are made at full precision, so a float32
rounding wobble in one inner product cannot masquerade as convergence. Trial points
are rounded to float32 on the host *before* they are evaluated, so the curvature
pair ``(s, y)`` describes exactly the points whose objective was computed.

The float32 floor
-----------------
Near the optimum the decrease a step would buy can be smaller than the resolution
of a float32-accumulated objective, and a search that trusts only the objective
then rejects every step while the gradient is still perfectly informative. Two
mechanisms keep the solver honest there:

* **Approximate Wolfe acceptance** (Hager & Zhang, 2005). A trial point whose value
  is within ``objective_noise·|f₀|`` of the starting value — indistinguishable from
  it at the objective's own resolution — is accepted when its *slope* meets the
  strong-Wolfe curvature condition. With ``|gₜᵀd| ≤ c₂|g₀ᵀd|`` the trapezoid rule
  predicts a decrease of at least ``t(1 − c₂)|g₀ᵀd|/2``, so the step is a descent
  step even though the value alone cannot certify it.
* **A floor stop.** When the objective can no longer resolve the progress the
  gradient predicts — a failed search whose predicted decrease was already below
  the noise level, or ``stall_iterations`` accepted steps in a row that neither
  lowered the objective resolvably nor reached a new low of ``max|g|`` — the run
  stops with ``stop_reason="float32-floor"``. Without it, a ``tol`` below what
  float32 can resolve (``tol=0``, say) would wander at noise level until
  ``max_iter``.

A dense initial inverse Hessian
-------------------------------
The two-loop recursion needs *some* symmetric positive-definite ``H₀`` to start each
iteration from, and nothing else about it (Nocedal & Wright, *Numerical
Optimization*, 2nd ed., §7.2). A diagonal is enough when the curvature lives along
the coordinates; when the features are strongly correlated, ten curvature pairs
cannot describe it, and the iteration count climbs with the condition number. A
caller that knows a fixed curvature model can hand its inverse over as a
:class:`DensePreconditioner`, an operator ``v ↦ Pv``. Each iteration then starts from
``γₖP`` with ``γₖ = sₖᵀyₖ / yₖᵀPyₖ`` — the scaling §7.2 gives the identity, in ``P``'s
metric — and an iteration with no curvature pairs takes ``−Pg``, a Newton step for
the model, whose first trial is the unit step rather than a unit move. The line
search, the curvature test, the stopping rules and the float32 floor are the same
code; only the search direction differs.

The operator may be *deferred*: a zero-argument callable, resolved once, after the
first evaluation of ``fun``, and only if the run is going to take a step. A caller
whose model is built from the data can then build it while the device computes that
first evaluation, instead of before it.

A model can be wrong where its caller could not see it. So a run on the operator
that would stop without converging — a failed line search, or the stall that ends a
run on the float32 floor — while ``max|g|`` is still above what float32 resolves
(:data:`GRADIENT_RESOLUTION`) does not stop: it leaves the operator, clears the
curvature pairs gathered under it, and searches again from the same point as a run
given only ``preconditioner`` would start. It *falls back*, once, and says so
(:attr:`LBFGSState.fallback`). A run that converges, reaches the floor, or uses up
``max_iter`` on the operator is unchanged by it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "GRADIENT_RESOLUTION",
    "DensePreconditioner",
    "LBFGSConfig",
    "LBFGSState",
    "minimize",
]

#: ``fun(x) -> (objective, gradient)``: a Python float, and an ``mx.array`` or a NumPy
#: array shaped like ``x``, float32 or float64, which the solver reads through
#: ``np.asarray``.
ObjectiveGradient = Callable[[Any], "tuple[float, Any]"]

#: ``max|g|`` below this fraction of its starting value is not resolvable in float32
#: (``docs/fp32_policy.md`` §2.3). A run that stops on the float32 floor *below* it
#: has done everything the arithmetic allows; one that stops above it has stalled.
GRADIENT_RESOLUTION = 1e-6


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
        Floor on the cosine between ``s`` and ``y`` below which a curvature pair is
        skipped rather than stored. Storing a pair with ``sᵀy ≤ 0`` makes the
        two-loop recursion produce an ascent direction. The test is
        ``sᵀy > min_curvature·‖s‖‖y‖``, which is invariant to the scale of the
        features; a floor on ``sᵀy`` relative to ``yᵀy`` alone discarded every pair
        once features reached ~1e6, and the run degenerated to steepest descent.
    objective_noise : float, default=1e-7
        Relative resolution of the objective values ``fun`` returns: two values
        closer than ``objective_noise·|f|`` are treated as indistinguishable. Sets
        the band for approximate-Wolfe acceptance and for the float32-floor stop.
        ``0`` disables both, leaving the exact strong-Wolfe search.
    stall_iterations : int, default=10
        Accepted steps in a row that neither lower the objective resolvably nor
        reach a new low of ``max|g|`` before the run stops on the float32 floor.
    """

    max_iter: int = 100
    tol: float = 1e-4
    history_size: int = 10
    max_line_search: int = 32
    c1: float = 1e-4
    c2: float = 0.9
    min_curvature: float = 1e-12
    objective_noise: float = 1e-7
    stall_iterations: int = 10

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
        if not 0.0 <= self.objective_noise < 1.0:
            raise ValueError(
                f"objective_noise must be in [0, 1), got {self.objective_noise}"
            )
        if self.stall_iterations < 1:
            raise ValueError(
                f"stall_iterations must be at least 1, got {self.stall_iterations}"
            )


@dataclass(frozen=True, slots=True)
class LBFGSState:
    """The result of a run. Immutable, and it always says why it stopped.

    Attributes
    ----------
    x : mx.array
        The final iterate, shaped like ``x0``, in float32.
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
        ``"float32-floor"``, ``"non-finite-objective"``.
    initial_grad_max_abs : float
        ``max|gradient|`` at ``x0``, the scale :data:`GRADIENT_RESOLUTION` is
        relative to.
    preconditioner : str
        The initial inverse Hessian the iterations ended on: ``"identity"``,
        ``"diagonal"`` or ``"dense"``. A run that took no step reports the one it
        was given before any deferred operator was resolved, and one that left a
        dense operator (``fallback``) the one it left it for.
    fallback : bool
        Whether the run left a dense operator for ``preconditioner``: it would have
        stopped above the float32 floor without converging, and went on from the
        same point without the operator instead (module docstring).
    """

    x: Any
    objective: float
    grad_max_abs: float
    n_iter: int
    n_fev: int
    converged: bool
    stop_reason: str
    initial_grad_max_abs: float = math.nan
    preconditioner: str = "identity"
    fallback: bool = False

    @property
    def at_resolution_floor(self) -> bool:
        """Whether ``max|g|`` is below what float32 can resolve for this problem."""
        return self.grad_max_abs <= GRADIENT_RESOLUTION * self.initial_grad_max_abs


@dataclass(frozen=True, slots=True)
class DensePreconditioner:
    """A dense symmetric positive-definite initial inverse Hessian ``P``, as ``v ↦ Pv``.

    ``apply`` takes and returns a flat float64 vector the size of the parameter. The
    solver never forms ``P``; it multiplies by it, twice per iteration, under the
    errstate of the two-loop recursion (``_direction``), so floating-point flags
    ``apply`` raises are not reported. ``−Pg`` is taken to be scaled like a Newton
    step for the caller's model, so an iteration without curvature pairs tries the
    unit step along it.
    """

    apply: Callable[[np.ndarray], np.ndarray]


# ------------------------------------------------------------------- line search


def _cubic_minimum(
    a: float, fa: float, ga: float, b: float, fb: float, gb: float
) -> float | None:
    """Minimizer of the cubic through ``(a, fa, ga)`` and ``(b, fb, gb)``.

    Returns ``None`` when the cubic has no interior minimum, which is the signal
    for the caller to bisect instead. Written out rather than delegating to SciPy
    because the solver deliberately has no SciPy dependency: the ancestor's
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


#: How close to either end of the bracket, as a fraction of its width, an interpolated
#: trial step may fall before the zoom bisects instead. The first trial of a fit is a
#: unit move (L-BFGS-B's first step); where the minimum lies far inside it — labels
#: that carry no signal, whose optimum is a coefficient vector of norm ~1e-2 — the
#: cubic lands at 1-2% of the bracket, and a 10% margin bisected three times before
#: accepting it: six objective evaluations for one iteration, against scikit-learn's
#: three or four.
_ZOOM_MARGIN = 0.01


def _safeguard(candidate: float | None, lo: float, hi: float) -> float:
    """Keep a trial step strictly inside ``[lo, hi]``, bisecting when it is not."""
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    width = high - low
    if width <= 0.0:
        return low
    margin = _ZOOM_MARGIN * width
    if candidate is None or not (low + margin <= candidate <= high - margin):
        return low + 0.5 * width
    return candidate


class _Step(NamedTuple):
    """What a line search hands back. ``kind`` says how the step was certified."""

    accepted: bool
    #: ``"wolfe"`` (strong Wolfe), ``"approximate"`` (approximate Wolfe, inside the
    #: objective's noise band), ``"armijo"`` (budget exhausted; best
    #: sufficient-decrease point) or ``"none"``.
    kind: str
    t: float
    x: Any
    f: float
    g: Any


class _LineSearch:
    """Bracket-then-zoom search for a step satisfying the strong Wolfe conditions.

    ``probe(t)`` must return ``(x_t, f_t, g_t, gtd_t)``, or ``None`` when the trial
    point is indistinguishable from the start in float32 — at which point no smaller
    step can be evaluated either, and the search ends. The search tracks the best
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
        self._noise = config.objective_noise * abs(f0)
        self._budget = config.max_line_search
        self.n_fev = 0
        #: (t, x, f, g) for the best Armijo-acceptable point seen.
        self._best: tuple[float, Any, float, Any] | None = None

    def _armijo(self, t: float, f: float) -> bool:
        return f <= self._f0 + self._c1 * t * self._gtd0

    def _curvature(self, gtd: float) -> bool:
        return abs(gtd) <= -self._c2 * self._gtd0

    def _approximate_wolfe(self, f: float, gtd: float) -> bool:
        """A value indistinguishable from ``f0``, with a slope that proves descent."""
        return self._noise > 0.0 and f <= self._f0 + self._noise and self._curvature(gtd)

    def _evaluate(self, t: float):
        result = self._probe(t)
        if result is None:
            return None
        x_t, f_t, g_t, gtd_t = result
        self.n_fev += 1
        if math.isfinite(f_t) and self._armijo(t, f_t):
            if self._best is None or f_t < self._best[2]:
                self._best = (t, x_t, f_t, g_t)
        return x_t, f_t, g_t, gtd_t

    def run(self, t_init: float) -> _Step:
        lo = None  # (t, f, gtd, x, g) — the end with the lower objective
        hi = None

        t_prev, f_prev, gtd_prev = 0.0, self._f0, self._gtd0
        x_prev = g_prev = None
        t = t_init
        first = True

        while self.n_fev < self._budget:
            evaluated = self._evaluate(t)
            if evaluated is None:
                return self._give_up()
            x_t, f_t, g_t, gtd_t = evaluated

            if not math.isfinite(f_t):
                # The trial point overflowed. Contracting toward the last finite
                # point is the only safe move: an infinite objective carries no
                # slope information to interpolate on.
                t = 0.5 * (t_prev + t)
                if t <= t_prev * (1.0 + 1e-12) or t <= 0.0:
                    break
                continue

            if not self._armijo(t, f_t) or (not first and f_t >= f_prev):
                if not self._armijo(t, f_t) and self._approximate_wolfe(f_t, gtd_t):
                    return _Step(True, "approximate", t, x_t, f_t, g_t)
                lo = (t_prev, f_prev, gtd_prev, x_prev, g_prev)
                hi = (t, f_t, gtd_t, x_t, g_t)
                break
            if self._curvature(gtd_t):
                return _Step(True, "wolfe", t, x_t, f_t, g_t)
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
            # Relative, not absolute. With ``t_lo = 0`` an absolute floor of 1e-12
            # on the step is a floor on how large the features may be: steps scale
            # like 1/scale², so at features ~1e8 the search gave up before reaching
            # the first acceptable point and returned the null model. The
            # evaluation budget and the "indistinguishable in float32" probe bound
            # the search instead.
            if abs(t_hi - t_lo) <= 1e-12 * max(abs(t_lo), abs(t_hi)):
                break
            t = _safeguard(
                _cubic_minimum(t_lo, f_lo, gtd_lo, t_hi, f_hi, gtd_hi), t_lo, t_hi
            )
            evaluated = self._evaluate(t)
            if evaluated is None:
                break
            x_t, f_t, g_t, gtd_t = evaluated

            if not math.isfinite(f_t) or not self._armijo(t, f_t) or f_t >= f_lo:
                if (
                    math.isfinite(f_t)
                    and not self._armijo(t, f_t)
                    and self._approximate_wolfe(f_t, gtd_t)
                ):
                    return _Step(True, "approximate", t, x_t, f_t, g_t)
                hi = (t, f_t, gtd_t, x_t, g_t)
                continue
            if self._curvature(gtd_t):
                return _Step(True, "wolfe", t, x_t, f_t, g_t)
            if gtd_t * (t_hi - t_lo) >= 0.0:
                hi = lo
            lo = (t, f_t, gtd_t, x_t, g_t)

        return self._give_up()

    def _give_up(self) -> _Step:
        """Fall back to the best sufficient-decrease point, or report failure.

        Accepting a point that met Armijo but not the curvature condition is safe:
        the caller's ``sᵀy`` test discards the curvature pair when it is not
        positive, so a weak step degrades the Hessian approximation without ever
        corrupting it.
        """
        if self._best is None:
            return _Step(False, "none", 0.0, None, self._f0, None)
        t, x_t, f_t, g_t = self._best
        return _Step(True, "armijo", t, x_t, f_t, g_t)


# ------------------------------------------------------------------------ solver


def minimize(
    fun: ObjectiveGradient,
    x0: Any,
    config: LBFGSConfig,
    *,
    callback: Callable[[np.ndarray], None] | None = None,
    preconditioner: np.ndarray | None = None,
    dense_preconditioner: (
        DensePreconditioner | Callable[[], DensePreconditioner | None] | None
    ) = None,
) -> LBFGSState:
    """Minimize ``fun`` from ``x0`` by limited-memory BFGS.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> (objective, gradient)``. ``x`` is a float32 ``mx.array``;
        ``gradient`` has its shape, and is an ``mx.array`` or a NumPy array, float32
        or float64 — the solver reads it through ``np.asarray`` in float64, so an
        objective that finishes on the host can return its float64 gradient as it
        is; ``objective`` is a Python float. ``fun`` should evaluate its gradient
        before returning — one synchronization for the value and the gradient
        together is the whole per-evaluation cost the solver adds. The pair must be
        *consistent* — the gradient must be the derivative of the value actually
        returned. A clipped objective paired with the unclipped analytic gradient
        makes the line search chase a decrease the derivative does not predict, which
        is how the ancestor produced ``ABNORMAL_TERMINATION_IN_LNSRCH`` and then
        reported it as ordinary non-convergence.
    x0 : mx.array
        Starting point. Any shape; the solver treats it as a vector space and hands
        ``fun`` the same shape back, so a caller may keep a natural
        ``(n_targets, n_features)`` layout.
    config : LBFGSConfig
    callback : callable, optional
        Called as ``callback(x)`` after every accepted step, with the new iterate as
        a read-only float64 NumPy array shaped like ``x0``. It observes; it cannot
        steer the run.
    preconditioner : ndarray, optional
        Positive, finite, shaped like ``x0``: a diagonal initial inverse-Hessian
        approximation used in place of the identity. The quasi-Newton update
        learns curvature only along the directions the iterates have explored, so
        a problem whose coordinates live on scales decades apart — a feature
        measured in units of 1e8 beside an intercept of order one — otherwise
        takes steps sized for the largest curvature in every direction and never
        moves the others. ``None`` is the identity. Neither ``tol`` nor the
        reported gradient is rescaled by it.
    dense_preconditioner : DensePreconditioner or callable, optional
        A dense initial inverse Hessian (module docstring), used in place of
        ``preconditioner`` when there is one. A zero-argument callable is resolved
        once, after the first evaluation of ``fun`` and only if the run takes a
        step, and returns a :class:`DensePreconditioner` or ``None``; ``None`` keeps
        ``preconditioner``. A run that fails on the operator above the float32 floor
        falls back to ``preconditioner`` (module docstring).

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

    shape = tuple(x0.shape)

    def evaluate(point: np.ndarray) -> tuple[float, np.ndarray]:
        # ``point`` is already float32-representable (see ``probe``), so the upload
        # is exact and the value returned belongs to exactly this point.
        value, grad = fun(mx.array(point.reshape(shape).astype(np.float32)))
        return float(value), np.asarray(grad, dtype=np.float64).reshape(-1)

    def finish(stop_reason: str, *, converged: bool = False) -> LBFGSState:
        return LBFGSState(
            x=mx.array(x.reshape(shape).astype(np.float32)),
            objective=objective,
            grad_max_abs=grad_max,
            n_iter=n_iter,
            n_fev=n_fev,
            converged=converged,
            stop_reason=stop_reason,
            initial_grad_max_abs=initial_grad_max,
            preconditioner=(
                "dense" if dense is not None
                else "diagonal" if diagonal is not None else "identity"
            ),
            fallback=fallback,
        )

    x = np.asarray(x0, dtype=np.float32).astype(np.float64).reshape(-1)
    diagonal = None
    if preconditioner is not None:
        diagonal = np.asarray(preconditioner, dtype=np.float64).reshape(-1)
        if diagonal.shape != x.shape or not np.all(np.isfinite(diagonal) & (diagonal > 0)):
            raise ValueError(
                "preconditioner must be positive and finite, shaped like x0 "
                f"{shape}; got shape {np.shape(preconditioner)}"
            )
        if np.all(diagonal == 1.0):
            diagonal = None
    dense = None
    deferred = None
    fallback = False
    if isinstance(dense_preconditioner, DensePreconditioner):
        dense = dense_preconditioner
    elif dense_preconditioner is not None:
        deferred = dense_preconditioner

    objective, grad = evaluate(x)
    n_fev = 1
    n_iter = 0
    grad_max = initial_grad_max = float(np.max(np.abs(grad))) if grad.size else 0.0

    if not math.isfinite(objective):
        return finish("non-finite-objective")
    if grad_max <= config.tol:
        return finish("gtol", converged=True)
    if deferred is not None and config.max_iter > 0:
        dense = deferred()
        if dense is not None and not isinstance(dense, DensePreconditioner):
            raise TypeError(
                "a deferred dense_preconditioner must return a DensePreconditioner or "
                f"None, got {type(dense).__name__}"
            )

    s_history: deque[np.ndarray] = deque(maxlen=config.history_size)
    y_history: deque[np.ndarray] = deque(maxlen=config.history_size)
    rho_history: deque[float] = deque(maxlen=config.history_size)

    # Float32-floor bookkeeping: the best max|g| reached, and the objective at the
    # last iteration that made verifiable progress.
    best_grad_max = grad_max
    reference_objective = objective
    stalled = 0

    def unresolved() -> bool:
        """Whether ``max|g|`` is still above what float32 resolves (``at_resolution_floor``)."""
        return grad_max > GRADIENT_RESOLUTION * initial_grad_max

    def leave_dense() -> None:
        """Fall back (module docstring): the curvature pairs gathered under the dense
        operator go with it, and the next search starts as the diagonal path's first."""
        s_history.clear()
        y_history.clear()
        rho_history.clear()

    # Every pass either accepts a step, returns, or — once at most — leaves the dense
    # operator and tries again from the same point, so this counts accepted steps.
    while n_iter < config.max_iter:
        direction, gtd = _direction(grad, s_history, y_history, rho_history, diagonal, dense)
        if not gtd < 0.0 or not math.isfinite(gtd):
            # The stored curvature no longer describes this objective — reset to
            # (preconditioned) steepest descent rather than taking a step that
            # increases f.
            s_history.clear()
            y_history.clear()
            rho_history.clear()
            direction, gtd = _direction(
                grad, s_history, y_history, rho_history, diagonal, dense
            )

        # Unit for a quasi-Newton direction. With no curvature pairs — the first
        # (steepest-descent) iteration — the direction's length is arbitrary, and the
        # first trial moves the parameters a unit distance: 1/‖d‖₂, L-BFGS-B's own
        # first step, which scikit-learn's lbfgs takes. The 1/‖d‖₁ this replaced,
        # capped at 1, started up to √p times shorter: on well-separated classes,
        # where that one long step already meets scikit-learn's tolerance, the fit
        # took 11-15 iterations to get where scikit-learn's single one did (60 000 x
        # 256, three to ten classes), while on the pinned problems of
        # tests/unit/test_lbfgs.py it saved two or three of 16-47. A dense operator's
        # −Pg is a Newton step for the caller's model, not a direction of arbitrary
        # length, so its natural trial is the unit step itself.
        if s_history or dense is not None:
            t_init = 1.0
        else:
            d_l2 = float(np.sqrt(_dot(direction, direction)))
            t_init = 1.0 / d_l2 if d_l2 > 0.0 and math.isfinite(1.0 / d_l2) else 1.0

        def probe(t: float, _x=x, _d=direction):
            # Rounded to float32 here, not by the upload, so that ``s`` and ``y``
            # below describe the points that were actually evaluated.
            x_t = (_x + t * _d).astype(np.float32).astype(np.float64)
            if np.array_equal(x_t, _x):
                return None
            f_t, g_t = evaluate(x_t)
            return x_t, f_t, g_t, _dot(g_t, _d)

        search = _LineSearch(probe, objective, gtd, config)
        step = search.run(t_init)
        n_fev += search.n_fev

        if not step.accepted or (step.kind != "approximate" and not step.f < objective):
            # A step that does not strictly lower the objective is a stall, not
            # progress; continuing would take null steps until max_iter and report
            # the iteration count as if work had been done. Which kind of stall it
            # is depends on what the slope predicted: a decrease already below the
            # objective's resolution is the float32 floor, anything larger is a
            # genuine failure — typically a gradient inconsistent with the value. On a
            # dense operator, above the floor, it is the operator that failed: fall back.
            if dense is not None and unresolved():
                leave_dense()
                dense, fallback, stalled = None, True, 0
                continue
            predicted = abs(gtd) * t_init
            floor = predicted <= config.objective_noise * abs(objective)
            return finish("float32-floor" if floor else "line-search-failure")

        s = step.x - x
        y = step.g - grad
        with np.errstate(**_QUIET):  # three products, one errstate (_dot)
            ys = float(s @ y)
            curved = ys > config.min_curvature * math.sqrt(float(s @ s)) * math.sqrt(
                float(y @ y)
            )
        if curved:
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / ys)

        x, objective, grad = step.x, step.f, step.g
        n_iter += 1
        grad_max = float(np.max(np.abs(grad)))
        if callback is not None:
            view = x.reshape(shape)
            view.flags.writeable = False
            callback(view)
        if grad_max <= config.tol:
            return finish("gtol", converged=True)

        if config.objective_noise > 0.0:
            # The reference stays put while the run is stalled, so a sequence of
            # individually unresolvable decreases still counts once it adds up.
            resolved = objective < reference_objective - config.objective_noise * abs(
                reference_objective
            )
            if grad_max < best_grad_max or resolved:
                best_grad_max = min(best_grad_max, grad_max)
                reference_objective = min(reference_objective, objective)
                stalled = 0
            else:
                stalled += 1
                if stalled >= config.stall_iterations:
                    if dense is not None and unresolved():
                        leave_dense()
                        dense, fallback, stalled = None, True, 0
                        continue
                    return finish("float32-floor")

    return finish("max_iter")


def _direction(grad, s_history, y_history, rho_history, diagonal=None, dense=None):
    """The two-loop recursion's direction and its slope ``gᵀd``, under one errstate.

    The ``2m + 2`` inner products of the recursion and its slope, and the dense
    operator's two applications, report no floating-point flags (:func:`_dot` says
    why that is safe). One ``np.errstate`` around all of them, rather than one per
    product: entering one costs about half a microsecond on NumPy 2 and two on NumPy
    1.23, and one per product added 8-17 µs to every iteration — 2-11% of a
    directly imported fit of 3-10 ms.
    """
    with np.errstate(**_QUIET):
        direction = _two_loop(grad, s_history, y_history, rho_history, diagonal, dense)
        return direction, float(grad @ direction)


def _two_loop(grad, s_history, y_history, rho_history, diagonal=None, dense=None):
    """Nocedal's two-loop recursion: ``-H∇f`` without forming ``H``.

    Returns the *descent* direction directly, i.e. the negation is folded in. The
    initial inverse Hessian is ``gamma·D``, with ``D`` the dense operator ``P`` when
    there is one, else the diagonal preconditioner (the identity when ``None``), and
    ``gamma = sᵀy / yᵀDy`` from the newest pair, which is what makes the first step
    of each iteration well-scaled. Host NumPy float64 throughout: ``2m + 1`` inner
    products of vectors a few hundred elements long, plus two applications of a
    dense operator. Called under :func:`_direction`'s errstate.
    """
    if not s_history:
        if dense is not None:
            return -dense.apply(grad)
        return -grad if diagonal is None else -(diagonal * grad)

    q = grad.copy()
    alphas: list[float] = []
    for s, y, rho in zip(
        reversed(s_history), reversed(y_history), reversed(rho_history), strict=True
    ):
        alpha = rho * float(s @ q)
        alphas.append(alpha)
        q -= alpha * y

    y_last = y_history[-1]
    if dense is not None:
        gamma = 1.0 / (rho_history[-1] * max(float(y_last @ dense.apply(y_last)), 1e-300))
        r = gamma * dense.apply(q)
    elif diagonal is None:
        gamma = 1.0 / (rho_history[-1] * max(float(y_last @ y_last), 1e-300))
        r = gamma * q
    else:
        y_scaled = diagonal * y_last
        gamma = 1.0 / (rho_history[-1] * max(float(y_last @ y_scaled), 1e-300))
        r = gamma * (diagonal * q)

    for s, y, rho, alpha in zip(
        s_history, y_history, rho_history, reversed(alphas), strict=True
    ):
        beta = rho * float(y @ r)
        r += s * (alpha - beta)

    return -r


#: The floating-point flags the solver's own products do not report (:func:`_dot`).
_QUIET = {"divide": "ignore", "over": "ignore", "invalid": "ignore"}


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """``a @ b`` as a Python float, its floating-point flags not reported.

    NumPy 2.2 on macOS raised divide-by-zero, overflow and invalid-value warnings
    from these products once the vectors reached about 900 entries — on every
    iteration of a wide fit — with finite operands and a finite result: flags its
    BLAS left set. A product that really is not finite — a gradient from ``fun``
    that was not, or a direction that ran away — is found without the flag: a slope
    that is not finite resets the direction, a curvature pair that is not finite is
    skipped, and a ``max|g|`` that is not finite never meets ``tol``, so the run
    stops unconverged with it in ``grad_max_abs``. A NaN raises no flag anyway.

    This is for a product on its own: a line-search slope, the first step's length.
    Products that come together share one errstate: the two-loop recursion with its
    slope (:func:`_direction`), and the curvature test's three.
    """
    with np.errstate(**_QUIET):
        return float(a @ b)
