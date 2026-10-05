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
the solver asks for rows through ``fetch_row`` (or, when the caller can batch them,
``fetch_rows``) and never keeps more than ``cache_rows`` of them. Gradients, alphas
and the working-set arithmetic are float64, again as LIBSVM does it, so the low
precision is confined to the kernel.

**Shrinking**, when enabled, is LIBSVM's (Joachims 1999; Chang & Lin 2011 §5.1):
samples pinned at a bound that cannot form a violating pair are set aside every
``min(n, 1000)`` iterations, so the per-iteration arithmetic runs over the active
set only, and the full gradient is reconstructed before the solver is allowed to
stop. Nothing about the optimum changes; the stopping test is always evaluated on
every sample. Two departures from LIBSVM, both deliberate: the active set is kept in
ascending index order rather than compacted by swapping, so the last-index
tie-breaking above means the same thing with and without shrinking; and the
reconstruction reads the rows it needs in batches, outside the LRU, so it neither
costs one round trip per row nor evicts the rows the next iterations need. It
still adds them up one row at a time in index order, as LIBSVM does, so neither
the batch size nor the BLAS build decides how the rebuilt gradient rounds.

**Finishing a solution found elsewhere.** :func:`finish_c_svc` takes the dual
variables another solver reached — the one-vs-one GPU kernel's, whose state is
double-float rather than float64 (``_kernels/_ovo.py``) — and holds them to this
module's contract: the gradient is recomputed from them in float64, the full-set KKT
stop is tested on it, and a solution that fails the test is finished by the loop
below, warm-started where the other solver stopped. What it returns is therefore a
solution this module would have accepted as its own.

This module imports numpy and nothing else. It never sees an estimator, never
reads global configuration, and never imports scikit-learn.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

__all__ = ["SMOConfig", "SMOState", "finish_c_svc", "solve_c_svc"]

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
    prefetch_rows : int
        On a cache miss, how many further rows to fetch in the same call, taken
        from the samples currently closest to violating the KKT conditions; see
        ``_RowCache``. Only used when the caller passes ``fetch_rows``. Off by
        default, because whether it pays is a property of the row source — a
        fixed cost per call that a batch amortizes — which the solver cannot see.
    shrinking : bool
        LIBSVM's shrinking heuristic; see the module docstring. Off by default,
        which never sets a sample aside.
    """

    tol: float = 1e-3
    max_iter: int = 10_000_000
    cache_rows: int = 256
    tau: float = _TAU
    prefetch_rows: int = 0
    shrinking: bool = False

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
        if self.prefetch_rows < 0:
            raise ValueError(
                f"prefetch_rows must be non-negative, got {self.prefetch_rows!r}"
            )


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
        Whether the loop ended on the KKT test, with every gradient finite.
    max_kkt_violation : float
        The violation measured at the final selection, i.e. what the stopping test
        saw — at the initial iterate when ``max_iter=0`` left no selection to run.
        Reported rather than discarded because it is the only honest evidence
        of how well an unconverged fit did. With shrinking, a converged fit
        measured it over every sample; one that ran out of budget while samples
        were set aside measured it over the active ones. NaN when the gradient
        was not finite.
    stop_reason : {"converged", "max_iter", "non-finite"}
        Why the loop ended. ``"non-finite"`` means a kernel row, a diagonal entry
        or the gradient built from them held an infinity or a NaN — kernel values
        that overflowed float32, typically — so there was no optimum to converge
        to. Before this was checked, NaN gradients left no eligible working pair
        and the solver reported *convergence* after one iteration, with a NaN
        bias; ``converged`` is now ``False`` there.
    """

    alpha: np.ndarray
    rho: float
    n_iter: int
    converged: bool
    max_kkt_violation: float
    stop_reason: str


#: Once the row cache is full, a prefetching miss may evict at most this fraction
#: of it (as ``capacity // _PREFETCH_TURNOVER`` rows). It is what keeps prefetch
#: from thrashing a small cache; a large one is barely touched per miss.
_PREFETCH_TURNOVER = 8


class _RowCache:
    """Least-recently-used cache over ``K[i, :]``, with batched prefetch.

    Pure performance: which rows happen to be resident cannot change the solution,
    only how often the row source is called — provided the source returns the same
    values for a row however it was batched, which is the contract ``fetch_rows``
    carries. The eviction is least-recently-*used*, so a hit moves its key to the
    end.

    **Prefetch.** When a row source costs a fixed round trip per call — an MLX
    evaluation is about 200 µs whether it returns one row or thirty-three — a
    miss is the moment to ask for more than the row that missed. With
    ``fetch_rows`` and a ``priority`` supplied, a miss also fetches up to
    ``prefetch`` of the highest-priority rows not yet resident, in the same call.
    While the cache has free room they go there. Once it is full they displace
    the least-recently-used rows, at most ``capacity // 8`` of them per miss.

    That second half is not obviously a good idea, and was measured rather than
    assumed: at 12 000–20 000 samples, where the default cache fills early, it cut
    single-row misses about fourfold and sped fits up 1.2–1.9×, and *re*-fetches of
    evicted rows fell (900 → 141 at 16 000 × 64) rather than rose — the rows
    about to be needed are the near-violators, and the rows evicted for them had
    gone longest unused.
    """

    __slots__ = (
        "_fetch",
        "_fetch_many",
        "_prefetch",
        "_capacity",
        "_rows",
        "_expected",
        "_resident",
    )

    def __init__(
        self,
        fetch_row: Callable[[int], np.ndarray],
        capacity: int,
        n: int,
        fetch_rows: Callable[[np.ndarray], np.ndarray] | None = None,
        prefetch: int = 0,
    ) -> None:
        self._fetch = fetch_row
        self._fetch_many = fetch_rows
        self._prefetch = max(0, int(prefetch)) if fetch_rows is not None else 0
        self._capacity = max(2, int(capacity))
        self._rows: OrderedDict[int, np.ndarray] = OrderedDict()
        self._expected = int(n)
        self._resident = np.zeros(self._expected, dtype=bool) if self._prefetch else None

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def width(self) -> int:
        """Length of every row, i.e. the number of samples."""
        return self._expected

    def peek(self, index: int) -> np.ndarray | None:
        """The row if it is resident, without touching the recency order."""
        return self._rows.get(index)

    def get(
        self, index: int, priority: Callable[[], np.ndarray] | None = None
    ) -> np.ndarray:
        """``K[index, :]``, fetching it — and possibly others — on a miss.

        ``priority`` is called only on a miss that will prefetch, and returns one
        score per sample: higher is fetched first, ``-inf`` never.
        """
        row = self._rows.get(index)
        if row is not None:
            self._rows.move_to_end(index)
            return row

        room = self._capacity - len(self._rows) - 1
        limit = min(self._prefetch, max(room, self._capacity // _PREFETCH_TURNOVER))
        if priority is None or limit <= 0:
            row = self._check(index, np.asarray(self._fetch(index)))
            self._store(index, row)
            return row

        extra = self._pick(priority(), index, limit)
        batch = np.asarray(self._fetch_many(np.concatenate(([index], extra))))
        if batch.ndim != 2 or batch.shape != (extra.shape[0] + 1, self._expected):
            raise ValueError(
                f"fetch_rows returned shape {batch.shape} for {extra.shape[0] + 1} rows, "
                f"expected ({extra.shape[0] + 1}, {self._expected})."
            )
        # Prefetched rows go in *behind* the one asked for in recency order, and
        # each is copied out of the batch: a row kept as a view would pin the
        # whole batch after its siblings were evicted, and the cache's memory
        # would no longer be what its capacity says.
        for position, prefetched in enumerate(extra.tolist(), start=1):
            self._store(prefetched, batch[position].copy())
        row = batch[0] if extra.shape[0] == 0 else batch[0].copy()
        self._store(index, row)
        return row

    def fetch_uncached(self, indices: np.ndarray) -> np.ndarray:
        """Rows for ``indices`` without inserting them, for one-off bulk reads.

        Gradient reconstruction reads rows it will not need again; routing them
        through the LRU would evict the rows the next iterations do need.
        """
        if self._fetch_many is not None:
            return np.asarray(self._fetch_many(indices))
        return np.stack([np.asarray(self._fetch(int(t))) for t in indices])

    def _pick(self, scores: np.ndarray, index: int, limit: int) -> np.ndarray:
        scores = np.array(scores, dtype=np.float64)
        scores[self._resident] = -np.inf
        scores[index] = -np.inf
        wanted = int(min(limit, np.count_nonzero(scores > -np.inf)))
        if wanted <= 0:
            return np.empty(0, dtype=np.intp)
        top = np.argpartition(scores, scores.shape[0] - wanted)[-wanted:]
        return np.sort(top).astype(np.intp)

    def _check(self, index: int, row: np.ndarray) -> np.ndarray:
        if row.ndim != 1 or row.shape[0] != self._expected:
            raise ValueError(
                f"kernel row {index} has shape {row.shape}, expected "
                f"({self._expected},)."
            )
        return row

    def _store(self, index: int, row: np.ndarray) -> None:
        self._rows[index] = row
        if self._resident is not None:
            self._resident[index] = True
        while len(self._rows) > self._capacity:
            evicted, _ = self._rows.popitem(last=False)
            if self._resident is not None:
                self._resident[evicted] = False


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
    *,
    fetch_rows: Callable[[np.ndarray], np.ndarray] | None = None,
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
        ``n_samples``. The solver never retains more than ``config.cache_rows``
        rows, so this is what keeps a full Gram matrix from being required.
    config : SMOConfig
    fetch_rows : callable, optional
        ``fetch_rows(indices)`` returns ``K[indices, :]`` as a 2-D float32 array,
        in one call. Worth passing whenever a call has a fixed cost that a batch
        amortizes — an MLX evaluation — because it lets a miss prefetch further
        rows (``config.prefetch_rows``) and lets shrinking rebuild the gradient in
        bulk. Prefetched rows are used interchangeably with ``fetch_row``'s, so
        with prefetch on it should return the very same values: if the two can
        differ in their last bits, the path depends on which rows arrived
        together — still deterministic, still optimal, but no longer independent
        of the cache size.

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

    if not np.isfinite(k_diag).all():
        # An infinite K_ii makes every step through i zero-length, so the loop
        # would spin on the same pair until the budget ran out, reporting a
        # violation it can never reduce. Nothing is fetched.
        positive = y > 0.0
        return _finish(np.zeros(n), C, -y, positive, 0, False, math.nan)

    cache = _RowCache(
        fetch_row,
        min(config.cache_rows, max(n, 2)),
        n,
        fetch_rows=fetch_rows,
        prefetch=config.prefetch_rows,
    )
    return _solve(y, C, k_diag, cache, config)


#: Rows of the Gram read per matrix product when :func:`finish_c_svc` rebuilds a
#: gradient: their float64 copy stays within 32 MiB, at 4 096 columns. 256 rows at a
#: time measured 1.13x slower at 120 pairs of 1 000 rows.
_GRADIENT_ROWS = 1024


def finish_c_svc(
    y: np.ndarray,
    C: np.ndarray,
    k_diag: np.ndarray,
    gram: np.ndarray,
    alpha: np.ndarray,
    n_iter: int,
    config: SMOConfig,
) -> SMOState:
    """Hold dual variables found elsewhere to this solver's stop, finishing them if needed.

    ``alpha`` is a candidate solution of the subproblem :func:`solve_c_svc` solves —
    the one-vs-one GPU kernel's, carried there in double-float — reached in
    ``n_iter`` working-set updates. It is clipped into the box, and the gradient is
    recomputed from it in float64, ``yg = −y + Σ_s α_s y_s K[s, :]`` over the nonzero
    ``α_s``, from ``gram``'s float32 entries widened. Then:

    * a candidate that already spent ``config.max_iter`` updates is reported as
      exhausted, as :func:`solve_c_svc` reports a budget stop — its loop never tests
      the iterate its last update produced, and neither does this;
    * a finite gradient whose full-set KKT violation is below ``config.tol`` — the
      stop :func:`solve_c_svc` applies — is a converged solution;
    * anything else is finished by :func:`solve_c_svc`'s loop, warm-started at
      ``alpha`` and the recomputed gradient, with ``n_iter`` already spent.

    Either way the bias is read off the recomputed gradient by LIBSVM's rule.

    A gradient that is not finite is returned as ``stop_reason="non-finite"``, for
    the caller to solve again from scratch.

    Parameters
    ----------
    y, C, k_diag
        As for :func:`solve_c_svc`.
    gram : ndarray of shape (n_samples, n_samples), float32
        The subproblem's whole kernel matrix, every self-entry the closed-form
        diagonal: the values the candidate was computed against. Rows are read
        through it as views; nothing is copied whole.
    alpha : ndarray of shape (n_samples,)
    n_iter : int
        Updates already spent; the finish counts on from here.
    config : SMOConfig
        ``cache_rows`` is ignored: every row is resident in ``gram``.

    Returns
    -------
    SMOState

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._solvers.smo import SMOConfig, finish_c_svc, solve_c_svc
    >>> X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    >>> y = np.array([-1.0, -1.0, 1.0, 1.0])
    >>> K = (X @ X.T).astype(np.float32)
    >>> C, k_diag = np.full(4, 1.0), np.diag(K).astype(np.float64)
    >>> done = solve_c_svc(y, C, k_diag, K.__getitem__, SMOConfig(cache_rows=4))
    >>> again = finish_c_svc(y, C, k_diag, K, done.alpha, done.n_iter, SMOConfig())
    >>> again.converged, again.n_iter == done.n_iter
    (True, True)
    >>> finish_c_svc(y, C, k_diag, K, np.zeros(4), 0, SMOConfig()).converged
    True
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    k_diag = np.ascontiguousarray(k_diag, dtype=np.float64)
    n = int(y.shape[0])
    if gram.shape != (n, n) or C.shape[0] != n or k_diag.shape[0] != n:
        raise ValueError(
            f"y, C, k_diag and gram must agree in size; got {n}, {C.shape[0]}, "
            f"{k_diag.shape[0]} and {gram.shape}."
        )
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, C)
    n_iter = int(n_iter)
    positive = y > 0.0
    yg = _gradient(y, alpha, gram)
    if not np.isfinite(yg).all():
        return _finish(alpha, C, yg, positive, n_iter, False, math.nan)
    upper, lower = _bound_status(alpha, C)
    up_set = np.where(positive, ~upper, ~lower)
    low_set = np.where(positive, ~lower, ~upper)
    violation = 0.0
    if up_set.any() and low_set.any():
        violation = float((-yg[up_set]).max() + yg[low_set].max())
    if n_iter >= config.max_iter:
        return _finish(alpha, C, yg, positive, n_iter, False, violation)
    if violation < float(config.tol):
        return _finish(alpha, C, yg, positive, n_iter, True, violation)
    # Every row is resident and read as a view, so the cache adds no memory.
    cache = _RowCache(gram.__getitem__, max(n, 2), n)
    return _solve(y, C, k_diag, cache, config, start=(alpha, yg, n_iter))


def _gradient(y: np.ndarray, alpha: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """``yg = −y + Σ_{α_s ≠ 0} α_s y_s K[s, :]`` in float64, the rows widened.

    The rows are read :data:`_GRADIENT_ROWS` at a time and contracted by NumPy's
    matrix product, chunk after chunk in index order, so the result depends on the
    data and the BLAS alone — not on threads, timing or which rows were read
    together before.

    Floating-point flags are not reported: NumPy 2.2 on macOS raised divide-by-zero,
    overflow and invalid-value warnings from this product on every fit, finite data
    and a finite result, flags its BLAS left set. A gradient that is really not
    finite is the caller's to find, with ``np.isfinite``.
    """
    yg = -y.copy()
    support = np.flatnonzero(alpha)
    weights = alpha[support] * y[support]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for start in range(0, support.shape[0], _GRADIENT_ROWS):
            rows = gram[support[start : start + _GRADIENT_ROWS]].astype(np.float64)
            yg += weights[start : start + _GRADIENT_ROWS] @ rows
    return yg


def _finish(
    alpha: np.ndarray,
    C: np.ndarray,
    yg: np.ndarray,
    positive: np.ndarray,
    n_iter: int,
    converged: bool,
    violation: float,
) -> SMOState:
    """Package the solution, with an honest ``stop_reason``.

    The finiteness test runs once, at the end, over everything the state reports:
    a NaN introduced by the very last update, or an infinite gradient on a sample
    the selection could not see, would otherwise ship as a converged solution.
    """
    finite = bool(np.isfinite(yg).all() and np.isfinite(alpha).all())
    if not finite or not math.isfinite(violation):
        converged = False
        stop_reason = "non-finite"
        violation = math.nan
    else:
        stop_reason = "converged" if converged else "max_iter"
    return SMOState(
        alpha=alpha,
        rho=_calculate_rho(alpha, C, yg, positive),
        n_iter=n_iter,
        converged=converged,
        max_kkt_violation=float(violation),
        stop_reason=stop_reason,
    )


def _widened(row: np.ndarray, scale: float) -> np.ndarray:
    """``scale * row`` computed in float64, for adding into a float64 gradient.

    ``row`` is a float32 kernel row. NumPy 2 promotes a float64 scalar times a
    float32 array to float64 by itself; NumPy 1.x's value-based casting keeps the
    product in float32 and only then widens it into the gradient, which rounds
    every update to float32 and cost ``test_binary_parity[1.0-sigmoid]`` its
    parity on NumPy 1.23. Bit-identical to the plain product on NumPy 2.
    """
    return np.multiply(row, scale, dtype=np.float64)


def _violation_rank(
    candidates_i: np.ndarray,
    gmax: float,
    candidates_j: np.ndarray,
    gmax2: float,
    active: np.ndarray | None = None,
    n: int = 0,
) -> np.ndarray:
    """Prefetch priority: how close each sample is to being the next working pair.

    ``i`` is the maximum of ``−yg`` over ``I_up`` and ``j`` is chosen, by its
    second-order gain, from the large values of ``yg`` over ``I_low``. The two
    sides sit on different scales — near ``−ρ`` and ``+ρ`` respectively — so each
    is measured from its own extreme and the better of the two is kept: zero for
    the current maximal violators, more negative for samples further from being
    picked, ``-inf`` for samples in neither set.

    With ``active``, the candidates are over the active set only, and the ranks
    are scattered back to all ``n`` samples, inactive ones ranked ``-inf``.
    """
    rank = np.maximum(candidates_i - gmax, candidates_j - gmax2)
    if active is None:
        return rank
    full = np.full(n, -np.inf)
    full[active] = rank
    return full


def _update_sets(
    t: int,
    is_upper: bool,
    alpha_t: float,
    positive: np.ndarray,
    up_set: np.ndarray,
    low_set: np.ndarray,
) -> None:
    """Refresh ``t``'s membership of ``I_up``/``I_low`` from its alpha status."""
    is_lower = (not is_upper) and alpha_t <= 0.0
    if positive[t]:
        up_set[t] = not is_upper
        low_set[t] = not is_lower
    else:
        up_set[t] = not is_lower
        low_set[t] = not is_upper


def _pair_update(
    y_i: float,
    y_j: float,
    C_i: float,
    C_j: float,
    old_i: float,
    old_j: float,
    grad_i: float,
    grad_j: float,
    quad_ij: float,
    tau: float,
) -> tuple[float, float]:
    """LIBSVM's analytic two-variable step, clipped to the box.

    ``grad_i`` and ``grad_j`` are the raw gradients ``∇_i``, ``∇_j``; ``quad_ij``
    is ``K_ii + K_jj − 2K_ij`` before the ``tau`` floor. Every branch clips *to*
    a bound, never near one, which is what makes ``α == C`` and ``α == 0`` exact
    tests downstream.
    """
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
    return new_i, new_j


def _solve(
    y: np.ndarray,
    C: np.ndarray,
    k_diag: np.ndarray,
    cache: _RowCache,
    config: SMOConfig,
    start: tuple[np.ndarray, np.ndarray, int] | None = None,
) -> SMOState:
    """The SMO loop behind :func:`solve_c_svc`, shrinking when the config asks.

    ``start``, from :func:`finish_c_svc`, is ``(alpha, yg, n_iter)``: a feasible
    point, its float64 gradient and the updates already spent on it. Without it the
    loop starts at ``α = 0``.

    There is one loop, whether or not samples are ever set aside, so the numerics
    that decide LIBSVM parity exist once. It runs over *compact copies* of the
    active samples' arrays, indexed by position in ``active`` — an ascending array
    of sample indices, so "last index wins" still means the last sample. With every
    sample active, which is always the case without shrinking, the copies are the
    full arrays themselves. The full-length arrays are the record of truth and are
    written back whenever the active set changes. Once shrinking has run, one more
    vector is kept for every sample:

    ``ygbar_t = Σ_{s at upper bound} C_s y_s K_ts`` — LIBSVM's ``G_bar`` in the
    ``yg`` convention, updated with a full row whenever a sample enters or leaves
    its upper bound. It is what lets the gradient of a set-aside sample be rebuilt
    from the free samples alone:
    ``yg_t = −y_t + ygbar_t + Σ_{s free} α_s y_s K_ts``. Unlike LIBSVM it is only
    built at the first shrinking pass, from the rows of the samples at their upper
    bound then — mostly still resident — and maintained from there: a fit that
    converges before its first shrinking pass never needs it, and maintaining it
    from the start measured 5–10% slower on such fits.

    The stopping test is only ever passed on the full set: when the active set
    looks optimal, the gradient is reconstructed, every sample is reactivated and
    the selection is repeated before the solver may stop.
    """
    n = int(y.shape[0])
    tau = float(config.tau)
    tol = float(config.tol)
    shrinking = bool(config.shrinking)

    if start is None:
        alpha = np.zeros(n, dtype=np.float64)
        # ∇ = −e at α = 0, so yg = y ⊙ ∇ = −y.
        yg = -y.copy()
        n_iter = 0
    else:
        alpha = np.array(start[0], dtype=np.float64)
        yg = np.array(start[1], dtype=np.float64)
        n_iter = int(start[2])
    ygbar: np.ndarray | None = None  # built at the first shrinking pass
    positive = y > 0.0
    # I_up and I_low as LIBSVM derives them from the single alpha status. They are
    # maintained incrementally rather than rebuilt: an SMO update moves exactly two
    # coordinates, so recomputing n memberships every iteration does n/2 times the
    # work the algorithm needs, and at n = 2000 that alone dominated the loop.
    upper, lower = _bound_status(alpha, C)
    up_set = np.where(positive, ~upper, ~lower)
    low_set = np.where(positive, ~lower, ~upper)

    everyone = np.arange(n)
    active = everyone
    period = min(n, 1000)
    counter = period + 1
    unshrunk = False
    converged = False
    # Measured at α = 0 too, so a zero budget reports the violation it stopped at
    # rather than an unmeasured 0.0 that reads as optimal.
    violation = 0.0
    if up_set.any() and low_set.any():
        violation = float((-yg[up_set]).max() + yg[low_set].max())

    while True:
        # Compact working copies for this active set. With every sample active
        # they are the full arrays themselves and rows need no gather.
        whole = active.shape[0] == n
        if whole:
            a_y, a_C, a_kd, a_pos = y, C, k_diag, positive
            a_alpha, a_yg, a_up, a_low = alpha, yg, up_set, low_set
        else:
            a_y, a_C, a_kd, a_pos = y[active], C[active], k_diag[active], positive[active]
            a_alpha, a_yg = alpha[active], yg[active]
            a_up, a_low = up_set[active], low_set[active]

        event = "budget"
        while n_iter < config.max_iter:
            # LIBSVM's cadence: shrink on every ``period``-th pass, starting with
            # pass ``period + 1``. A pass here that shrinks does not also select,
            # so the counter restarts one higher to keep ``period`` selections
            # between shrinks, as LIBSVM has.
            if shrinking:
                counter -= 1
                if counter == 0:
                    counter = period + 1
                    event = "shrink"
                    break

            if not a_up.any() or not a_low.any():
                # No violating pair exists at all; LIBSVM reports this as convergence.
                violation = 0.0
                event = "optimal"
                break
            candidates_i = np.where(a_up, -a_yg, -np.inf)
            ii = _last_argmax(candidates_i)
            gmax = float(candidates_i[ii])
            candidates_j = np.where(a_low, a_yg, -np.inf)
            gmax2 = float(candidates_j.max())
            violation = gmax + gmax2
            if not math.isfinite(violation):
                # Every sample is in I_up or I_low, so a NaN or infinite gradient
                # anywhere surfaces here. Without this test the NaN comparisons below
                # found no eligible j and reported convergence.
                event = "non-finite"
                break

            # Only evaluated on a miss that prefetches, and bound to this
            # iteration's extremes so the two fetches below rank the same way.
            priority = partial(
                _violation_rank, candidates_i, gmax, candidates_j, gmax2,
                None if whole else active, n,
            )
            i = ii if whole else int(active[ii])
            full_i = cache.get(i, priority)
            row_i = full_i if whole else full_i[active]
            grad_diff = gmax + a_yg
            quad = a_kd[ii] + a_kd - 2.0 * row_i
            # LIBSVM substitutes TAU only when the coefficient is non-positive; a
            # clamp with maximum() would also rewrite small positive coefficients and
            # silently change which j gets selected.
            quad = np.where(quad > 0.0, quad, tau)
            eligible = a_low & (grad_diff > 0.0)
            if not eligible.any():
                event = "optimal"
                break
            objective_gain = -(grad_diff * grad_diff) / quad
            jj = _last_argmin(np.where(eligible, objective_gain, np.inf))
            if violation < tol:
                event = "optimal"
                break

            n_iter += 1

            j = jj if whole else int(active[jj])
            full_j = cache.get(j, priority)
            row_j = full_j if whole else full_j[active]
            y_i = a_y[ii]
            y_j = a_y[jj]
            C_i = a_C[ii]
            C_j = a_C[jj]
            old_i = a_alpha[ii]
            old_j = a_alpha[jj]
            new_i, new_j = _pair_update(
                y_i, y_j, C_i, C_j, old_i, old_j, y_i * a_yg[ii], y_j * a_yg[jj],
                a_kd[ii] + a_kd[jj] - 2.0 * float(row_i[jj]), tau,
            )
            a_alpha[ii] = new_i
            a_alpha[jj] = new_j
            # The float32 kernel row is widened explicitly (see _widened), as in
            # every gradient product in this module.
            a_yg += _widened(row_i, (new_i - old_i) * y_i)
            a_yg += _widened(row_j, (new_j - old_j) * y_j)

            upper_i = new_i >= C_i
            upper_j = new_j >= C_j
            _update_sets(ii, upper_i, new_i, a_pos, a_up, a_low)
            _update_sets(jj, upper_j, new_j, a_pos, a_up, a_low)
            # G_bar is updated with the *full* row: a set-aside sample's gradient
            # is rebuilt from it later, so it must hear about every change.
            if ygbar is not None:
                if upper_i != (old_i >= C_i):
                    ygbar += _widened(full_i, (C_i if upper_i else -C_i) * y_i)
                if upper_j != (old_j >= C_j):
                    ygbar += _widened(full_j, (C_j if upper_j else -C_j) * y_j)

        if not whole:
            alpha[active] = a_alpha
            yg[active] = a_yg
            up_set[active] = a_up
            low_set[active] = a_low

        if event in ("budget", "non-finite"):
            break

        if event == "optimal":
            if whole:
                converged = True
                break
            # Optimal on the active set only. Rebuild the gradient of everything
            # that was set aside and look again, over every sample; if that finds
            # more work, do it and shrink on the pass after (LIBSVM's counter = 1).
            _reconstruct_gradient(y, C, alpha, yg, ygbar, active, cache)
            active = everyone
            counter = 2
            continue

        # event == "shrink": LIBSVM's do_shrinking, on the active set's extremes.
        if ygbar is None:
            ygbar = _upper_bound_part(y, C, alpha, cache)
        a_up_now = up_set[active]
        a_low_now = low_set[active]
        a_yg_now = yg[active]
        gmax1 = float(np.where(a_up_now, -a_yg_now, -np.inf).max(initial=-np.inf))
        gmax2 = float(np.where(a_low_now, a_yg_now, -np.inf).max(initial=-np.inf))
        if not unshrunk and gmax1 + gmax2 <= 10.0 * tol:
            # Close to the end: reactivate everything once, so the final phase
            # shrinks against an accurate gradient rather than one set aside
            # early, when the extremes were far apart.
            unshrunk = True
            _reconstruct_gradient(y, C, alpha, yg, ygbar, active, cache)
            active = everyone
        active = active[~_be_shrunk(up_set[active], low_set[active], yg[active], gmax1, gmax2)]

    if active.shape[0] < n:
        # Out of budget while shrunk. The bias reads every sample's gradient when
        # no sample is free, so it must be exact for all of them.
        _reconstruct_gradient(y, C, alpha, yg, ygbar, active, cache)

    return _finish(alpha, C, yg, positive, n_iter, converged, violation)


def _be_shrunk(
    up: np.ndarray, low: np.ndarray, yg: np.ndarray, gmax1: float, gmax2: float
) -> np.ndarray:
    """LIBSVM's ``be_shrunk``, for a whole active set at once.

    A sample in ``I_up`` only can serve as ``i`` alone, and pairs into a violation
    only with a ``j`` whose ``yg`` exceeds its own; if even the largest ``yg`` over
    ``I_low`` (``gmax2``) does not, it cannot move. Symmetrically for ``I_low``
    only, against ``gmax1``. Free samples belong to both sets and are never
    shrunk. The inequalities are strict, as in LIBSVM.
    """
    only_up = up & ~low
    only_low = low & ~up
    return (only_up & (yg > gmax2)) | (only_low & (-yg > gmax1))


#: Rows read per call while reconstructing the gradient: enough to amortize an
#: MLX round trip, few enough that the transient block stays a few tens of MiB.
_RECONSTRUCT_ROWS = 256


def _upper_bound_part(
    y: np.ndarray, C: np.ndarray, alpha: np.ndarray, cache: _RowCache
) -> np.ndarray:
    """``ygbar`` from scratch: ``Σ_{s at upper bound} C_s y_s K[s, :]``, in index order."""
    upper, _ = _bound_status(alpha, C)
    # A zero box is "at its upper bound" too, and contributes nothing.
    bounded = np.flatnonzero(upper & (C > 0.0))
    total = np.zeros(y.shape[0], dtype=np.float64)
    weights = C[bounded] * y[bounded]
    chunk = max(2, min(_RECONSTRUCT_ROWS, cache.capacity))
    for start in range(0, bounded.shape[0], chunk):
        block = _read_rows(cache, bounded[start : start + chunk])
        _add_rows(total, block, weights[start : start + chunk])
    return total


def _reconstruct_gradient(
    y: np.ndarray,
    C: np.ndarray,
    alpha: np.ndarray,
    yg: np.ndarray,
    ygbar: np.ndarray,
    active: np.ndarray,
    cache: _RowCache,
) -> None:
    """Rebuild ``yg`` for every sample outside ``active``, in place.

    ``yg_t = −y_t + ygbar_t + Σ_{s free} α_s y_s K_ts``: the bounded samples'
    contribution is in ``ygbar`` already, and free samples are never inactive, so
    only ``K[inactive, free]`` has to be read. Like LIBSVM it reads whichever side
    of that block has fewer rows — the free rows, or the inactive ones — but in
    batches of ``_RECONSTRUCT_ROWS`` through ``fetch_rows`` and outside the LRU,
    using resident rows where there are any. Reading them one ``fetch_row`` at a
    time through the cache measured slower than not shrinking at all.

    Either way the terms are added in LIBSVM's order — to ``ygbar_t − y_t``, one
    free sample at a time, in index order — so the batching decides only how rows
    are read. A matrix product per batch rounded the sum by the batch size, which
    follows the cache capacity, and by the BLAS build; near a tie that rounding
    picks the working pair, so ``cache_size`` moved shrinking fits.
    """
    inactive = np.setdiff1d(np.arange(y.shape[0]), active, assume_unique=True)
    if inactive.shape[0] == 0:
        return
    rebuilt = ygbar[inactive] - y[inactive]
    upper, lower = _bound_status(alpha, C)
    free = np.flatnonzero(~upper & ~lower)
    if free.shape[0]:
        weights = alpha[free] * y[free]
        by_free = free.shape[0] <= inactive.shape[0]
        sources = free if by_free else inactive
        chunk = max(2, min(_RECONSTRUCT_ROWS, cache.capacity))
        for start in range(0, sources.shape[0], chunk):
            block = _read_rows(cache, sources[start : start + chunk])
            if by_free:
                _add_rows(rebuilt, block[:, inactive], weights[start : start + chunk])
            else:
                for k, row in enumerate(block[:, free], start=start):
                    # A running sum from rebuilt[k], in the by-free branch's order:
                    # accumulate is sequential by definition, where sum() may
                    # reassociate.
                    terms = np.multiply(row, weights, dtype=np.float64)
                    terms[0] += rebuilt[k]
                    rebuilt[k] = np.add.accumulate(terms)[-1]
    yg[inactive] = rebuilt


def _add_rows(total: np.ndarray, block: np.ndarray, weights: np.ndarray) -> None:
    """``total += weights @ block``, one row at a time in row order, in float64."""
    for row, weight in zip(block, weights, strict=True):
        total += _widened(row, weight)


def _read_rows(cache: _RowCache, indices: np.ndarray) -> np.ndarray:
    """``K[indices, :]``, resident rows from the cache and the rest in one call."""
    resident = [cache.peek(int(t)) for t in indices]
    missing = [k for k, row in enumerate(resident) if row is None]
    block = np.empty((indices.shape[0], cache.width), dtype=np.float32)
    if missing:
        fetched = cache.fetch_uncached(indices[missing])
        block[missing] = fetched
    for k, row in enumerate(resident):
        if row is not None:
            block[k] = row
    return block


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
