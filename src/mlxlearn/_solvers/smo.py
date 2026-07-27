"""Exact sequential minimal optimization for C-support-vector classification.

This solves the LIBSVM dual

    min_α  ½ αᵀQα − eᵀα      subject to  yᵀα = 0,  0 ≤ α_t ≤ C_t,

with ``Q_ts = y_t y_s K_ts``. Nothing here is approximate: no subsampling, no
random-feature surrogate, no ridge stand-in. Those produce a different classifier,
and a different classifier does not get to ship under an estimator name that
promises this one.

The formulation is LIBSVM's, and the details that look like trivia are not:

* **Second-order working-set selection** (Fan, Chen & Lin, JMLR 6:1889–1918, 2005).
  ``i`` is the maximal violator of the upper set; ``j`` maximizes the *predicted
  decrease in the objective*, ``(∇_i − ∇_j)² / (K_ii + K_jj − 2K_ij)``, over the
  lower set, rather than being the second maximal violator. That is the difference
  between converging in thousands of iterations and tens of thousands.
* **Tie-breaking on the later index.** Both the ``argmax`` for ``i`` and the
  ``argmin`` for ``j`` keep the *last* extremum, because LIBSVM's scans use ``>=``
  and ``<=``. At the first iteration every candidate is tied, so a first-wins
  ``argmax`` starts the solver down a different path; the optimum is the same but
  the support-vector set at the margin need not be.
* **The alpha status is one enum, not two predicates.** A sample with ``C_t = 0``
  satisfies both ``α ≥ C`` and ``α ≤ 0``. LIBSVM classifies it as *at the upper
  bound*, and the lower-bound test is therefore ``not upper and α ≤ 0``. Testing
  the two independently puts such a sample in the wrong constraint set and changes
  the stopping criterion. It stays in ``I_low`` either way and can never move, so
  a caller that leaves zero-box samples in the problem will watch the solver run
  to ``max_iter`` reporting a violation it has no move available to fix — which is
  why the estimator drops them before calling here.

**Memory.** Kernel rows are float32, exactly as LIBSVM caches them, and are held in
an LRU whose capacity the caller sizes. The full ``n × n`` matrix is never required:
the solver asks for one row at a time through ``fetch_row`` and never keeps more
than ``cache_rows`` of them. Gradients, alphas and the working-set arithmetic are
float64, again as LIBSVM does it, so the low precision is confined to the kernel.

This module imports numpy and nothing else. It never sees an estimator, never
reads global configuration, and never imports scikit-learn.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["SMOConfig", "SMOState", "solve_c_svc"]

#: LIBSVM's floor for a degenerate working-set quadratic coefficient. Duplicate
#: rows, and the indefinite sigmoid and polynomial kernels, can make
#: ``K_ii + K_jj − 2K_ij`` zero or negative; substituting this produces a huge
#: step that the box clipping immediately saturates, which is the intended
#: behavior and not an error.
_TAU = 1e-12


@dataclass(frozen=True, slots=True)
class SMOConfig:
    """Everything the solver needs that is not an array.

    Attributes
    ----------
    tol : float
        Stop when the maximal KKT violation ``max_{I_up}(−yg) + max_{I_low}(yg)``
        drops below this. LIBSVM's ``eps``; scikit-learn's ``tol``.
    max_iter : int
        Hard iteration cap, always positive. The caller resolves
        ``max_iter=-1`` into a concrete budget; a solver that read a sentinel
        would have to know what "unlimited" costs, which is a policy question and
        not a numerical one.
    cache_rows : int
        Kernel rows the LRU may hold. At least 2 — the working set needs both of
        its rows alive within one iteration.
    tau : float
        Floor for a non-positive quadratic coefficient.
    """

    tol: float = 1e-3
    max_iter: int = 10_000_000
    cache_rows: int = 256
    tau: float = _TAU

    def __post_init__(self) -> None:
        if not self.tol > 0.0:
            raise ValueError(f"tol must be strictly positive, got {self.tol!r}")
        # Zero is a legal budget, not a sentinel: scikit-learn accepts max_iter=0
        # and LIBSVM honors it by returning the initial iterate. The caller
        # resolves -1 to a concrete number before constructing this, so a
        # negative value here is a bug in that resolution.
        if self.max_iter < 0:
            raise ValueError(
                f"max_iter must be a resolved non-negative budget, got {self.max_iter!r}"
            )
        if self.cache_rows < 2:
            raise ValueError(f"cache_rows must be at least 2, got {self.cache_rows!r}")


@dataclass(frozen=True, slots=True)
class SMOState:
    """The solution of one binary subproblem.

    Attributes
    ----------
    alpha : ndarray of shape (n_samples,)
        Dual variables, float64, satisfying ``0 ≤ α_t ≤ C_t`` exactly — the box
        branches clip to the bounds rather than near them.
    rho : float
        The bias in LIBSVM's convention, so the decision value of a sample is
        ``Σ_t α_t y_t K(x_t, ·) − rho``. scikit-learn's ``intercept_`` for the
        corresponding one-vs-one classifier is ``−rho``.
    n_iter : int
        Working-set updates actually performed. Equal to ``max_iter`` exactly when
        the budget was exhausted.
    converged : bool
        Whether the loop ended on the KKT test rather than on the budget.
    max_kkt_violation : float
        The violation measured at the final selection, i.e. what the stopping test
        saw. Reported rather than discarded because it is the only honest evidence
        of how well an unconverged fit did.
    """

    alpha: np.ndarray
    rho: float
    n_iter: int
    converged: bool
    max_kkt_violation: float


class _RowCache:
    """Least-recently-used cache over ``K[i, :]``.

    Pure performance: which rows happen to be resident cannot change the solution,
    only how often ``fetch_row`` is called. The eviction is least-recently-*used*,
    so a hit moves its key to the end.
    """

    __slots__ = ("_fetch", "_capacity", "_rows", "_expected")

    def __init__(self, fetch_row: Callable[[int], np.ndarray], capacity: int, n: int) -> None:
        self._fetch = fetch_row
        self._capacity = max(2, int(capacity))
        self._rows: OrderedDict[int, np.ndarray] = OrderedDict()
        self._expected = int(n)

    def get(self, index: int) -> np.ndarray:
        row = self._rows.get(index)
        if row is not None:
            self._rows.move_to_end(index)
            return row
        row = np.asarray(self._fetch(index))
        if row.ndim != 1 or row.shape[0] != self._expected:
            raise ValueError(
                f"kernel row {index} has shape {row.shape}, expected "
                f"({self._expected},)."
            )
        self._rows[index] = row
        while len(self._rows) > self._capacity:
            self._rows.popitem(last=False)
        return row


def _bound_status(alpha: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LIBSVM's three-way alpha status, as two masks.

    The two are not independent predicates. A sample with ``C_t = 0`` — which a
    zero ``sample_weight`` produces — satisfies ``α ≥ C`` and ``α ≤ 0`` at once;
    LIBSVM's single enum resolves that as *upper bound*, so the lower test is
    conditioned on the upper one having failed. Anything free is neither.
    """
    upper = alpha >= C
    lower = ~upper & (alpha <= 0.0)
    return upper, lower


def _last_argmax(values: np.ndarray) -> int:
    """Index of the maximum, keeping the *last* one on a tie.

    LIBSVM's selection scans forward with ``>=``, so the last maximum wins.
    ``np.argmax`` keeps the first, and at iteration zero every candidate is tied.
    """
    return values.shape[0] - 1 - int(np.argmax(values[::-1]))


def _last_argmin(values: np.ndarray) -> int:
    """Index of the minimum, keeping the *last* one on a tie."""
    return values.shape[0] - 1 - int(np.argmin(values[::-1]))


def solve_c_svc(
    y: np.ndarray,
    C: np.ndarray,
    k_diag: np.ndarray,
    fetch_row: Callable[[int], np.ndarray],
    config: SMOConfig,
) -> SMOState:
    """Solve one binary C-SVC subproblem exactly.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Labels in ``{+1, -1}``, float64.
    C : ndarray of shape (n_samples,)
        Per-sample upper bounds, float64, strictly positive. This is where class
        weights and sample weights live:
        ``C_t = C · class_weight[class(t)] · sample_weight[t]``. A zero entry is
        handled correctly but never converges — see the module docstring — so the
        caller is expected to have dropped those rows.
    k_diag : ndarray of shape (n_samples,)
        ``K(x_t, x_t)``, float64, computed in closed form.
    fetch_row : callable
        ``fetch_row(i)`` returns ``K[i, :]`` as a 1-D float32 array of length
        ``n_samples``. The solver never asks for more than one row at a time and
        never retains more than ``config.cache_rows`` of them, so this is what
        keeps a full Gram matrix from being required.
    config : SMOConfig

    Returns
    -------
    SMOState

    Notes
    -----
    The gradient is maintained as ``yg = y ⊙ ∇``, not as ``∇`` itself. The update
    ``∇_k += Q_ki Δα_i + Q_kj Δα_j`` carries a factor ``y_k`` that cancels against
    the one in ``yg``, leaving ``yg += y_i Δα_i K[:, i] + y_j Δα_j K[:, j]`` — plain
    kernel rows, no per-iteration multiplication by the label vector. The raw
    gradients the two-variable update needs are recovered as ``∇_i = y_i yg_i``.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._solvers.smo import SMOConfig, solve_c_svc
    >>> X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    >>> y = np.array([-1.0, -1.0, 1.0, 1.0])
    >>> K = X @ X.T
    >>> state = solve_c_svc(
    ...     y, np.full(4, 1.0), np.diag(K).copy(),
    ...     lambda i: K[i].astype(np.float32), SMOConfig(cache_rows=4)
    ... )
    >>> state.converged
    True
    >>> bool(abs(state.alpha @ y) < 1e-9)
    True
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    k_diag = np.ascontiguousarray(k_diag, dtype=np.float64)
    n = int(y.shape[0])
    if C.shape[0] != n or k_diag.shape[0] != n:
        raise ValueError(
            f"y, C and k_diag must agree in length; got {n}, {C.shape[0]}, "
            f"{k_diag.shape[0]}."
        )

    alpha = np.zeros(n, dtype=np.float64)
    # ∇ = −e at α = 0, so yg = y ⊙ ∇ = −y.
    yg = -y.copy()
    positive = y > 0.0
    cache = _RowCache(fetch_row, min(config.cache_rows, max(n, 2)), n)
    tau = float(config.tau)
    tol = float(config.tol)

    # I_up and I_low as LIBSVM derives them from the single alpha status. They are
    # maintained incrementally rather than rebuilt: an SMO update moves exactly two
    # coordinates, so recomputing n memberships every iteration does n/2 times the
    # work the algorithm needs, and at n = 2000 that alone dominated the loop.
    upper, lower = _bound_status(alpha, C)
    up_set = np.where(positive, ~upper, ~lower)
    low_set = np.where(positive, ~lower, ~upper)

    n_iter = 0
    converged = False
    violation = 0.0

    while n_iter < config.max_iter:
        if not up_set.any() or not low_set.any():
            # No violating pair exists at all; LIBSVM reports this as convergence.
            converged = True
            violation = 0.0
            break

        candidates_i = np.where(up_set, -yg, -np.inf)
        i = _last_argmax(candidates_i)
        gmax = float(candidates_i[i])

        row_i = cache.get(i)
        gmax2 = float(np.where(low_set, yg, -np.inf).max())
        violation = gmax + gmax2

        grad_diff = gmax + yg
        quad = k_diag[i] + k_diag - 2.0 * row_i
        # LIBSVM substitutes TAU only when the coefficient is non-positive; a
        # clamp with maximum() would also rewrite small positive coefficients and
        # silently change which j gets selected.
        quad = np.where(quad > 0.0, quad, tau)
        eligible = low_set & (grad_diff > 0.0)
        if not eligible.any():
            converged = True
            break
        objective_gain = -(grad_diff * grad_diff) / quad
        j = _last_argmin(np.where(eligible, objective_gain, np.inf))

        if violation < tol:
            converged = True
            break

        n_iter += 1

        row_j = cache.get(j)
        y_i = y[i]
        y_j = y[j]
        C_i = C[i]
        C_j = C[j]
        old_i = alpha[i]
        old_j = alpha[j]
        grad_i = y_i * yg[i]
        grad_j = y_j * yg[j]

        quad_ij = k_diag[i] + k_diag[j] - 2.0 * float(row_i[j])
        if quad_ij <= 0.0:
            quad_ij = tau

        if y_i != y_j:
            delta = (-grad_i - grad_j) / quad_ij
            diff = old_i - old_j
            new_i = old_i + delta
            new_j = old_j + delta
            if diff > 0.0:
                if new_j < 0.0:
                    new_j = 0.0
                    new_i = diff
            else:
                if new_i < 0.0:
                    new_i = 0.0
                    new_j = -diff
            if diff > C_i - C_j:
                if new_i > C_i:
                    new_i = C_i
                    new_j = C_i - diff
            else:
                if new_j > C_j:
                    new_j = C_j
                    new_i = C_j + diff
        else:
            delta = (grad_i - grad_j) / quad_ij
            total = old_i + old_j
            new_i = old_i - delta
            new_j = old_j + delta
            if total > C_i:
                if new_i > C_i:
                    new_i = C_i
                    new_j = total - C_i
            else:
                if new_j < 0.0:
                    new_j = 0.0
                    new_i = total
            if total > C_j:
                if new_j > C_j:
                    new_j = C_j
                    new_i = total - C_j
            else:
                if new_i < 0.0:
                    new_i = 0.0
                    new_j = total

        delta_i = new_i - old_i
        delta_j = new_j - old_j
        alpha[i] = new_i
        alpha[j] = new_j
        yg += (delta_i * y_i) * row_i
        yg += (delta_j * y_j) * row_j

        for t in (i, j):
            is_upper = alpha[t] >= C[t]
            is_lower = (not is_upper) and alpha[t] <= 0.0
            if positive[t]:
                up_set[t] = not is_upper
                low_set[t] = not is_lower
            else:
                up_set[t] = not is_lower
                low_set[t] = not is_upper

    rho = _calculate_rho(alpha, C, yg, positive)
    return SMOState(
        alpha=alpha,
        rho=rho,
        n_iter=n_iter,
        converged=converged,
        max_kkt_violation=float(violation),
    )


def _calculate_rho(
    alpha: np.ndarray,
    C: np.ndarray,
    yg: np.ndarray,
    positive: np.ndarray,
) -> float:
    """LIBSVM's ``calculate_rho``: the free set's mean, with a bounded fallback.

    At the optimum every free sample sits exactly on the margin, so their ``yg``
    values agree and averaging them is the numerically stable way to read the
    bias off. When no sample is free — every one of them pinned at a bound, which
    happens for a tiny ``C`` or a perfectly separable problem solved to a vertex —
    the bias is only constrained to an interval, and LIBSVM takes its midpoint.
    """
    upper, lower = _bound_status(alpha, C)
    free = ~upper & ~lower

    n_free = int(free.sum())
    if n_free > 0:
        return float(yg[free].sum() / n_free)

    upper_side = (upper & ~positive) | (lower & positive)
    lower_side = (upper & positive) | (lower & ~positive)
    ub = float(yg[upper_side].min()) if upper_side.any() else np.inf
    lb = float(yg[lower_side].max()) if lower_side.any() else -np.inf
    return (ub + lb) / 2.0
