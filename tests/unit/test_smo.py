"""Direct tests of the SMO solver, without an estimator in the way.

The solver is shaped — arrays and a frozen config in, a frozen state out — so its
numerics can be checked on their own, and this is where that happens:

* against **LIBSVM itself**, through ``sklearn.svm.SVC(kernel="precomputed")`` fed
  the very float32 kernel values the solver sees, on the dual objective;
* against an **independent float64 KKT check** that appeals to nobody;
* against **itself**: the performance machinery — the row cache, batched
  prefetch and shrinking — must not change the answer, and where it cannot change
  the arithmetic either, not even the bits.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.svm import SVC as SkSVC

from mlxlearn._common.device import gpu_available
from mlxlearn._kernels.kernels import KernelSpec, build_kernel_rows, kernel_diagonal, kernel_matrix
from mlxlearn._solvers import smo
from mlxlearn._solvers.smo import SMOConfig, SMOState, _RowCache, solve_c_svc

TOL = 1e-3


def _problem(n=600, d=8, seed=0, kernel="rbf", flip_y=0.05):
    X, labels = make_classification(
        n_samples=n, n_features=d, n_informative=max(2, d // 2), n_redundant=1,
        flip_y=flip_y, random_state=seed,
    )
    y = np.where(labels == 1, 1.0, -1.0)
    if kernel == "rbf":
        K = rbf_kernel(X, gamma=1.0 / (d * X.var()))
    elif kernel == "poly":
        K = polynomial_kernel(X, degree=3, gamma=1.0 / (d * X.var()), coef0=1.0)
    else:
        K = linear_kernel(X)
    # The solver sees float32 rows, as LIBSVM's cache holds them. Every reference
    # below is computed on these same values, so any disagreement is the solver's.
    K32 = K.astype(np.float32)
    return X, y, K32


class _Source:
    """A Gram-backed row source that counts how it is called."""

    def __init__(self, K32):
        self.K = K32
        self.single = 0
        self.batches: list[int] = []

    def row(self, i):
        self.single += 1
        return self.K[i]

    def rows(self, indices):
        self.batches.append(len(indices))
        return self.K[np.asarray(indices)]


def _solve(y, K32, C, *, fetch_rows=True, **config):
    source = _Source(K32)
    state = solve_c_svc(
        y, C, np.diag(K32).astype(np.float64), source.row,
        SMOConfig(tol=TOL, cache_rows=config.pop("cache_rows", K32.shape[0]), **config),
        fetch_rows=source.rows if fetch_rows else None,
    )
    return state, source


def _integer_problem(n, d, seed, kernel):
    """Integer features, so the kernel is exact however it is computed.

    ``RandomState`` is NumPy's frozen legacy stream and an integer Gram leaves no
    rounding for a BLAS build to decide, so this problem is the same bits on every
    NumPy — which is what lets a test pin the solve itself, not just its optimum.
    """
    rs = np.random.RandomState(seed)
    X = rs.randint(-4, 5, size=(n, d), dtype=np.int64)
    y = np.where(X @ rs.randint(-3, 4, size=d, dtype=np.int64) > 0, 1.0, -1.0)
    flip = rs.random_sample(n) < 0.1
    y[flip] = -y[flip]
    G = (X @ X.T).astype(np.float64)
    K = G if kernel == "linear" else (G / 4.0 + 1.0) ** 3
    return y, K.astype(np.float32)


def _pin(state: SMOState):
    digest = hashlib.sha256(np.ascontiguousarray(state.alpha, dtype="<f8").tobytes())
    return state.n_iter, state.stop_reason, state.rho.hex(), digest.hexdigest()[:16]


def _dual(coef, K64):
    """LIBSVM's dual objective from signed coefficients ``α ⊙ y``."""
    return 0.5 * coef @ K64 @ coef - np.abs(coef).sum()


def _kkt_violation(alpha, y, C, K64):
    """``max_{I_up}(−yg) + max_{I_low}(yg)`` recomputed from scratch in float64."""
    yg = y * (y * (K64 @ (alpha * y)) - 1.0)
    up = np.where(y > 0, alpha < C, alpha > 0)
    low = np.where(y > 0, alpha > 0, alpha < C)
    return float(np.max(-yg[up]) + np.max(yg[low]))


def _assert_same_state(a: SMOState, b: SMOState):
    np.testing.assert_array_equal(a.alpha, b.alpha)
    assert a.rho == b.rho
    assert a.n_iter == b.n_iter
    assert a.converged == b.converged


# ----------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0.0},
        {"tol": -1e-3},
        {"max_iter": -1},
        {"cache_rows": 1},
        {"prefetch_rows": -1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SMOConfig(**kwargs)


def test_inputs_of_different_lengths_are_rejected():
    with pytest.raises(ValueError, match="agree in length"):
        solve_c_svc(np.ones(3), np.ones(4), np.ones(3), lambda i: np.ones(3), SMOConfig())


def test_a_wrongly_shaped_row_is_an_error_not_a_model():
    y = np.array([1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="kernel row"):
        solve_c_svc(y, np.ones(3), np.ones(3), lambda i: np.ones(2, np.float32), SMOConfig())
    cache = _RowCache(lambda i: np.ones(3), 3, 3, fetch_rows=lambda idx: np.ones((1, 3)), prefetch=2)
    with pytest.raises(ValueError, match="fetch_rows returned"):
        cache.get(0, lambda: np.zeros(3))


# ------------------------------------------------------------- against LIBSVM


@pytest.mark.parametrize("kernel", ["rbf", "poly", "linear"])
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_matches_libsvm_on_the_dual_objective(kernel, C):
    """Same float32 kernel values, same stopping rule, same optimum as LIBSVM.

    Both stop within ``tol`` of the KKT conditions, so the α themselves need not
    agree; the dual objective must, and the independent KKT check must pass.
    """
    _, y, K32 = _problem(n=400, kernel=kernel, seed=3)
    K64 = K32.astype(np.float64)
    Cs = np.full(y.shape[0], C)
    state, _ = _solve(y, K32, Cs)
    assert state.converged

    ref = SkSVC(kernel="precomputed", C=C, tol=TOL).fit(K64, y)
    ref_coef = np.zeros(y.shape[0])
    ref_coef[ref.support_] = ref.dual_coef_[0]

    ours = _dual(state.alpha * y, K64)
    theirs = _dual(ref_coef, K64)
    # One-sided where mlxlearn may legitimately stop lower (fp32_policy §3).
    assert ours <= theirs + 1e-5 * abs(theirs), (ours, theirs)
    assert ours == pytest.approx(theirs, rel=1e-4)
    assert _kkt_violation(state.alpha, y, Cs, K64) < TOL * 1.01
    # LIBSVM's decision function is K @ coef + intercept_; the solver's rho is its
    # bias with LIBSVM's internal sign, which scikit-learn flips for two classes.
    ours_decision = K64 @ (state.alpha * y) - state.rho
    theirs_decision = K64 @ ref_coef + ref.intercept_[0]
    agreement = np.mean(np.sign(ours_decision) == np.sign(theirs_decision))
    assert agreement >= 0.99


def test_box_constraints_and_the_equality_hold_exactly():
    _, y, K32 = _problem(n=500, seed=4)
    C = np.random.default_rng(0).uniform(0.2, 3.0, y.shape[0])
    state, _ = _solve(y, K32, C)
    assert np.all(state.alpha >= 0.0) and np.all(state.alpha <= C)
    # The box branches clip *to* the bounds, which is what makes exact status
    # tests downstream meaningful.
    assert np.any(state.alpha == C) and np.any(state.alpha == 0.0)
    assert abs(state.alpha @ y) < 1e-9


# ------------------------------------------------------- the budget, max_iter


@pytest.mark.parametrize("shrinking", [False, True])
def test_max_iter_zero_returns_the_initial_iterate(shrinking):
    _, y, K32 = _problem(n=200)
    state, source = _solve(y, K32, np.ones(200), max_iter=0, shrinking=shrinking)
    assert state.n_iter == 0 and not state.converged
    np.testing.assert_array_equal(state.alpha, 0.0)
    assert np.isfinite(state.rho)
    assert source.single == 0 and source.batches == []
    # The violation at α = 0, where yg = −y, as the one selection of max_iter=1
    # measures it — not an unmeasured 0.0 beside stop_reason="max_iter".
    one, _ = _solve(y, K32, np.ones(200), max_iter=1, shrinking=shrinking)
    assert state.max_kkt_violation == one.max_kkt_violation == 2.0


@pytest.mark.parametrize("shrinking", [False, True])
def test_an_exhausted_budget_is_reported_not_hidden(shrinking):
    _, y, K32 = _problem(n=600, seed=5, flip_y=0.2)
    state, _ = _solve(y, K32, np.full(600, 10.0), max_iter=50, shrinking=shrinking)
    assert state.n_iter == 50
    assert not state.converged
    assert state.stop_reason == "max_iter"
    assert state.max_kkt_violation > TOL


@pytest.mark.parametrize("shrinking", [False, True])
def test_a_converged_solve_says_so(shrinking):
    _, y, K32 = _problem(n=300, seed=2)
    state, _ = _solve(y, K32, np.ones(300), shrinking=shrinking)
    assert state.converged and state.stop_reason == "converged"


# ------------------------------------------------- a non-finite kernel is not a solution


@pytest.mark.parametrize("shrinking", [False, True])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("where", ["column", "row-and-column", "diagonal"])
def test_a_non_finite_kernel_is_never_reported_as_converged(shrinking, bad, where):
    """NaN gradients left no eligible working pair, which the loop read as convergence.

    The state claimed ``converged=True`` after one iteration with a NaN ``rho``, and
    only SVC's post-fit check stood between that and a model predicting one class.
    An infinite diagonal entry instead made every step through its sample zero-length,
    and the solver spun on one pair until the budget ran out.
    """
    _, y, K32 = _problem(n=300, seed=3)
    K32 = K32.copy()
    k_diag = np.diag(K32).astype(np.float64)
    if where == "column":
        K32[:, 7] = bad  # every row the solver reads carries it
    elif where == "row-and-column":
        K32[250, :] = bad
        K32[:, 250] = bad
    else:
        k_diag[11] = bad
    source = _Source(K32)
    with np.errstate(invalid="ignore", over="ignore"):
        state = solve_c_svc(
            y, np.ones(300), k_diag, source.row,
            SMOConfig(tol=TOL, cache_rows=300, shrinking=shrinking, max_iter=100_000),
            fetch_rows=source.rows,
        )
    assert not state.converged
    assert state.stop_reason == "non-finite"
    assert np.isnan(state.max_kkt_violation)


# ------------------------------------------- the cache cannot change the answer


def test_cache_capacity_changes_calls_not_the_solution():
    _, y, K32 = _problem(n=500, seed=6)
    C = np.ones(500)
    whole, whole_src = _solve(y, K32, C, fetch_rows=False)
    tiny, tiny_src = _solve(y, K32, C, fetch_rows=False, cache_rows=2)
    _assert_same_state(whole, tiny)
    assert tiny_src.single > whole_src.single


@pytest.mark.parametrize("cache_rows, fetch_rows", [(3, True), (2, False)])
def test_cache_capacity_does_not_move_a_shrinking_solve(cache_rows, fetch_rows):
    """With shrinking, the capacity also sets the batches the gradient is rebuilt in.

    Each batch used to be reduced with a matrix product, so the capacity — and the
    BLAS build — decided how the rebuilt gradient was rounded, and near a tie which
    pair came next: this problem took a different path at each of these
    capacities. The terms are now added one at a time in index order, however the
    rows were read.
    """
    _, y, K32 = _problem(n=500, seed=3, kernel="poly", flip_y=0.1)
    C = np.full(500, 10.0)
    whole, _ = _solve(y, K32, C, shrinking=True)
    small, _ = _solve(y, K32, C, shrinking=True, cache_rows=cache_rows, fetch_rows=fetch_rows)
    assert whole.n_iter > 500  # past the first shrinking pass, at min(n, 1000)
    _assert_same_state(whole, small)


@pytest.mark.parametrize("cache_rows", [3, 40, 250, 500])
def test_prefetch_changes_calls_not_the_solution(cache_rows):
    """Batched prefetch, into free room and — once full — over the LRU tail."""
    _, y, K32 = _problem(n=500, seed=7, flip_y=0.1)
    C = np.full(500, 5.0)
    plain, plain_src = _solve(y, K32, C, cache_rows=cache_rows, fetch_rows=False)
    prefetched, pre_src = _solve(y, K32, C, cache_rows=cache_rows, prefetch_rows=16)
    _assert_same_state(plain, prefetched)

    calls = pre_src.single + len(pre_src.batches)
    assert max(pre_src.batches) > 1
    # A batch never exceeds the prefetch count, nor what the cache can hold.
    assert max(pre_src.batches) - 1 <= min(16, cache_rows - 1)
    if cache_rows >= 40:
        assert calls < 0.75 * plain_src.single
    else:
        # capacity // 8 == 0: once full, a tiny cache degrades to single rows
        # rather than thrashing.
        assert calls <= plain_src.single


def test_prefetched_rows_own_their_memory_and_capacity_holds():
    """A prefetched row kept as a view would pin its whole batch after eviction."""
    K = np.arange(100, dtype=np.float32).reshape(10, 10)
    cache = _RowCache(lambda i: K[i], 4, 10, fetch_rows=lambda idx: K[np.asarray(idx)] + 0, prefetch=8)
    scores = np.arange(10, dtype=np.float64)
    cache.get(0, lambda: scores)
    assert len(cache) == 4
    for index in range(10):
        row = cache.peek(index)
        if row is not None:
            assert row.base is None or row.base.shape == (1, 10)
            np.testing.assert_array_equal(row, K[index])
    # Full now: a further miss displaces at most capacity // 8 = 0 rows beyond
    # its own, i.e. it degrades to a single fetch.
    before = len(cache)
    cache.get(1, lambda: scores)
    assert len(cache) == before


# ----------------------------------------------------------------- pinned paths

#: ``(kernel, n, d, seed, C, per-sample boxes, max_iter)`` to
#: ``(n_iter, stop_reason, rho, alpha digest)``.
_PINNED_PLAIN = {
    ("linear", 300, 4, 0, 1.0, False, 10_000_000):
        (2371, "converged", "-0x1.a6a4c80d1eb72p-4", "56f8203634578a6f"),
    ("poly", 300, 4, 1, 0.05, False, 10_000_000):
        (4062, "converged", "0x1.619fa2b81d7a8p-1", "391564d06a4cbdc2"),
    ("linear", 400, 5, 2, 0.5, True, 10_000_000):
        (6506, "converged", "0x1.61ce468977635p-3", "df9242d8db0db65c"),
    ("linear", 1200, 5, 4, 0.5, False, 10_000_000):
        (8520, "converged", "0x1.43b864c24f4d2p-4", "df4a1e65c7bb0651"),
    ("linear", 1200, 6, 7, 1.0, True, 1800):
        (1800, "max_iter", "0x1.170d1115c9b5dp-3", "ee14dfc2877d5cce"),
}


def _pinned_solve(case, **config):
    kernel, n, d, seed, C, per_sample, max_iter = case
    y, K32 = _integer_problem(n, d, seed, kernel)
    Cs = np.full(n, C)
    if per_sample:
        Cs = C * np.random.RandomState(seed + 100).randint(2, 9, size=n, dtype=np.int64) / 4.0
    state, _ = _solve(y, K32, Cs, max_iter=max_iter, **config)
    return state


@pytest.mark.parametrize("cache_rows", [None, 2])
@pytest.mark.parametrize("case", list(_PINNED_PLAIN), ids=lambda case: "-".join(map(str, case[:5])))
def test_the_non_shrinking_path_is_pinned(case, cache_rows):
    """``shrinking=False`` takes exactly the steps it always has, bit for bit.

    The pins were taken from the plain loop before the shrinking loop absorbed it,
    so they are what shows the two were the same loop, and what notices when the
    statements they share change. Nothing on this path reduces through BLAS and the
    kernel is exact, so the pins hold on every NumPy and BLAS build (checked on
    NumPy 1.23 with OpenBLAS and 2.5 with Accelerate). ``n = 1200`` runs past the
    1000-iteration shrinking period, where an ungated shrink would move the path.
    """
    config = {} if cache_rows is None else {"cache_rows": cache_rows}
    assert _pin(_pinned_solve(case, **config)) == _PINNED_PLAIN[case]


_PINNED_SHRINKING = {
    ("linear", 300, 4, 0, 1.0, False, 10_000_000):
        (2389, "converged", "-0x1.a6ac2a478c996p-4", "08a098a44ef23833"),
    ("poly", 300, 4, 1, 0.05, False, 10_000_000):
        (3876, "converged", "0x1.6194aa39f5147p-1", "23f5434e94de4c0b"),
    ("linear", 400, 5, 2, 0.5, True, 10_000_000):
        (5895, "converged", "0x1.61a42c4966343p-3", "653f821c971726d8"),
}


@pytest.mark.parametrize("cache_rows", [None, 2])
@pytest.mark.parametrize(
    "case", list(_PINNED_SHRINKING), ids=lambda case: "-".join(map(str, case[:5]))
)
def test_the_shrinking_path_is_pinned(case, cache_rows):
    """Shrinking fits are the same bits on every cache capacity and BLAS build.

    The rebuilt gradient decides working pairs near a tie, so its rounding is part
    of the path. When each batch of rows was reduced with a matrix product, every
    one of these problems took a different path at each of the two capacities, and
    the third also on NumPy 1.23 with OpenBLAS against 2.5 with Accelerate at the
    same capacity. The pins hold on both builds.
    """
    config = {"shrinking": True}
    if cache_rows is not None:
        config["cache_rows"] = cache_rows
    assert _pin(_pinned_solve(case, **config)) == _PINNED_SHRINKING[case]


# ------------------------------------------------------------------- shrinking


@pytest.mark.parametrize(
    "kernel, C, n",
    [
        # Small C converges fastest: at n = 1200 this case stopped at 791
        # iterations, before the first shrinking pass, and never shrank.
        ("rbf", 0.1, 2000),
        ("rbf", 1.0, 1200),
        ("rbf", 10.0, 1200),
        ("poly", 1.0, 1200),
        ("linear", 0.1, 1200),
    ],
)
def test_shrinking_reaches_the_same_optimum(kernel, C, n, monkeypatch):
    """LIBSVM's shrinking is a speed heuristic; the optimum must not move.

    Per-sample boxes stand in for sample and class weights. Every case runs past
    the first shrinking pass, which comes after ``min(n, 1000)`` iterations, and
    fails if nothing was set aside. The path itself may change — a set-aside
    sample can become a violator again before the next reconstruction, and then
    the two solvers select differently, as LIBSVM's do — so the models are
    compared at the float32-policy tolerance and the optimum on the objective and
    the KKT conditions.
    """
    _, y, K32 = _problem(n=n, d=10, seed=8, kernel=kernel, flip_y=0.1)
    K64 = K32.astype(np.float64)
    Cs = C * np.random.default_rng(1).uniform(0.5, 2.0, y.shape[0])

    set_aside = []
    original = smo._be_shrunk

    def counting(*args):
        shrunk = original(*args)
        set_aside.append(int(shrunk.sum()))
        return shrunk

    plain, _ = _solve(y, K32, Cs)
    monkeypatch.setattr(smo, "_be_shrunk", counting)
    shrunk, _ = _solve(y, K32, Cs, shrinking=True)
    assert plain.converged and shrunk.converged
    assert _kkt_violation(shrunk.alpha, y, Cs, K64) < TOL * 1.01
    j_plain = _dual(plain.alpha * y, K64)
    j_shrunk = _dual(shrunk.alpha * y, K64)
    assert j_shrunk == pytest.approx(j_plain, rel=1e-6)
    np.testing.assert_allclose(
        K64 @ (shrunk.alpha * y) - shrunk.rho,
        K64 @ (plain.alpha * y) - plain.rho,
        atol=2e-2,
        rtol=5e-3,
    )
    assert sum(set_aside) > 0, (
        f"nothing was set aside in {shrunk.n_iter} iterations; the case tests nothing"
    )


def test_shrinking_keeps_the_path_when_nothing_ties():
    """Ascending active order keeps LIBSVM's last-index tie-breaking intact.

    With every reconstruction exact enough not to flip a selection, shrinking
    takes the very same steps; measured bit-identical across the problems here.
    """
    _, y, K32 = _problem(n=1500, d=10, seed=9, flip_y=0.05)
    Cs = np.full(y.shape[0], 10.0)
    plain, _ = _solve(y, K32, Cs)
    shrunk, _ = _solve(y, K32, Cs, shrinking=True)
    assert plain.n_iter > 2000
    assert shrunk.n_iter == plain.n_iter
    np.testing.assert_allclose(shrunk.alpha, plain.alpha, rtol=0, atol=1e-9)
    assert shrunk.rho == pytest.approx(plain.rho, abs=1e-9)


def test_shrinking_rebuilds_the_gradient_before_an_out_of_budget_exit():
    """rho is read off the gradient, so a budget exit while shrunk must rebuild it."""
    _, y, K32 = _problem(n=1200, d=10, seed=10, flip_y=0.2)
    Cs = np.full(y.shape[0], 10.0)
    budget = 1600  # past the first shrink at pass 1001, well short of convergence
    plain, _ = _solve(y, K32, Cs, max_iter=budget)
    shrunk, _ = _solve(y, K32, Cs, max_iter=budget, shrinking=True)
    assert not plain.converged and not shrunk.converged
    assert shrunk.n_iter == plain.n_iter == budget
    np.testing.assert_allclose(shrunk.alpha, plain.alpha, rtol=0, atol=1e-9)
    assert shrunk.rho == pytest.approx(plain.rho, abs=1e-7)


@pytest.mark.parametrize("cache_rows", [2, 64])
def test_shrinking_without_batches_or_with_a_tiny_cache(cache_rows):
    """Reconstruction falls back to single rows, and never through the LRU."""
    _, y, K32 = _problem(n=1100, d=10, seed=11, flip_y=0.1)
    K64 = K32.astype(np.float64)
    Cs = np.full(y.shape[0], 5.0)
    plain, _ = _solve(y, K32, Cs)
    shrunk, _ = _solve(y, K32, Cs, shrinking=True, fetch_rows=False, cache_rows=cache_rows)
    assert shrunk.converged
    assert _dual(shrunk.alpha * y, K64) == pytest.approx(_dual(plain.alpha * y, K64), rel=1e-6)


# --------------------------------------- the KernelRows path, as a fit takes it


BACKENDS = [
    pytest.param(
        "mlx",
        marks=pytest.mark.skipif(not gpu_available(), reason="needs an MLX GPU device"),
    ),
    "cpu",
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel", ["rbf", "poly", "linear", "sigmoid"])
def test_kernel_rows_with_a_tiny_cache_reach_the_gram_optimum(backend, kernel):
    """The on-demand row path — every pair above ~7 240 rows — against the Gram path.

    This is how ``SVC._solve_pair`` drives the solver on either side of its Gram
    cutoff, shrunk to a size that runs in well under a second: rows from
    ``KernelRows`` through a two-row cache with the device's prefetch, against a
    materialized Gram. On the GPU the rows equal the Gram's bit for bit (measured,
    not promised), so the whole solve does; elsewhere the objective must agree.
    """
    X, y, _ = _problem(n=700, d=12, seed=12)
    X32 = X.astype(np.float32)
    spec = KernelSpec(kernel=kernel, gamma=1.0 / 12, degree=3, coef0=0.5)
    # The linear kernel's iteration count grows fastest with C; keep it small so
    # a two-row cache — a device round trip per miss — stays quick.
    C = np.full(y.shape[0], 0.05 if kernel == "linear" else 1.0)
    k_diag = kernel_diagonal(spec, X32.astype(np.float64))

    gram = kernel_matrix(spec, X32, backend=backend)
    via_gram = solve_c_svc(y, C, k_diag, gram.__getitem__, SMOConfig(cache_rows=700))

    rows = build_kernel_rows(X32, spec, backend=backend)
    via_rows = solve_c_svc(
        y, C, k_diag, rows.row,
        SMOConfig(cache_rows=2, prefetch_rows=rows.prefetch_rows),
        fetch_rows=rows.rows,
    )
    K64 = gram.astype(np.float64)
    assert via_rows.converged
    if backend == "mlx" and kernel != "sigmoid":
        _assert_same_state(via_gram, via_rows)
    j_gram = _dual(via_gram.alpha * y, K64)
    j_rows = _dual(via_rows.alpha * y, K64)
    if kernel == "sigmoid":
        # Indefinite: two correct runs may reach different local optima
        # (fp32_policy §3.2.1). Only "no worse, within tolerance" is meaningful.
        assert j_rows <= j_gram + 1e-3 * abs(j_gram)
    else:
        assert j_rows == pytest.approx(j_gram, rel=1e-5)
        assert _kkt_violation(via_rows.alpha, y, C, K64) < TOL * 1.01


@pytest.mark.skipif(not gpu_available(), reason="needs an MLX GPU device")
def test_multiclass_fit_prefetches_on_the_row_path(monkeypatch):
    """End to end: a cache too small for the Gram, several pairs, prefetch engaged.

    Each one-vs-one pair decides Gram versus rows on its own size, so this runs
    the row path three times over. The model must match the Gram-path fit.
    """
    from mlxlearn._common.config import tuning_context
    from mlxlearn._kernels import kernels
    from mlxlearn.svm import SVC

    X, labels = make_classification(
        n_samples=1500, n_features=12, n_informative=6, n_classes=3, random_state=13
    )
    sizes = []
    original = kernels.KernelRows.rows

    def counting(self, indices):
        sizes.append(len(indices))
        return original(self, indices)

    with tuning_context(force_backend="mlx"):
        large = SVC(cache_size=200).fit(X, labels)
        monkeypatch.setattr(kernels.KernelRows, "rows", counting)
        small = SVC(cache_size=0.5).fit(X, labels)

    assert small.execution_backend_ == "mlx"
    assert sizes and max(sizes) > 1, "the row path did not prefetch"
    np.testing.assert_allclose(
        small.decision_function(X[:300]), large.decision_function(X[:300]), atol=2e-2, rtol=5e-3
    )
    assert np.mean(small.predict(X) == large.predict(X)) >= 0.99


# ------------------------------------------------------------- finishing elsewhere


def _finish(y, K32, C, alpha, n_iter, **config):
    return smo.finish_c_svc(
        y, C, np.diag(K32).astype(np.float64), K32, alpha, n_iter,
        SMOConfig(tol=TOL, **config),
    )


def test_a_solution_that_meets_the_stop_is_accepted_as_it_is():
    """The host's own solution passes its own stop; the bias is read off the gradient
    recomputed in float64, which agrees with the solve's incremental one."""
    _, y, K32 = _problem(n=400)
    C = np.full(400, 1.0)
    done, _ = _solve(y, K32, C)
    again = _finish(y, K32, C, done.alpha, done.n_iter)
    assert again.converged and again.stop_reason == "converged"
    np.testing.assert_array_equal(again.alpha, done.alpha)
    assert again.n_iter == done.n_iter
    assert abs(again.rho - done.rho) < 1e-12
    assert again.max_kkt_violation < TOL


@pytest.mark.parametrize("start", ["zero", "early", "perturbed"])
def test_a_solution_that_fails_the_stop_is_finished_to_it(start):
    """From α = 0, from a solve stopped early, and from a feasible perturbation of the
    optimum: the warm-started loop reaches the same optimum, counting on from the
    candidate's updates."""
    _, y, K32 = _problem(n=500, seed=3)
    K64 = K32.astype(np.float64)
    C = np.full(500, 2.0)
    done, _ = _solve(y, K32, C)
    if start == "zero":
        alpha, n_iter = np.zeros(500), 0
    elif start == "early":
        early, _ = _solve(y, K32, C, max_iter=done.n_iter // 3)
        alpha, n_iter = early.alpha, early.n_iter
    else:
        alpha, n_iter = done.alpha.copy(), done.n_iter
        free = np.flatnonzero((alpha > 0.1) & (alpha < 1.9))
        i = next(t for t in free if y[t] > 0)
        j = next(t for t in free if y[t] > 0 and t != i)
        alpha[i] += 0.05
        alpha[j] -= 0.05
    finished = _finish(y, K32, C, alpha, n_iter)
    assert finished.converged
    assert finished.n_iter > n_iter
    assert _kkt_violation(finished.alpha, y, C, K64) < TOL * 1.01
    np.testing.assert_allclose(
        _dual(finished.alpha * y, K64), _dual(done.alpha * y, K64), rtol=1e-6
    )
    assert abs(float(finished.alpha @ y)) < 1e-9


def test_a_candidate_out_of_budget_is_reported_as_exhausted():
    _, y, K32 = _problem(n=300)
    C = np.full(300, 1.0)
    early, _ = _solve(y, K32, C, max_iter=5)
    finished = _finish(y, K32, C, early.alpha, 5, max_iter=5)
    assert finished.stop_reason == "max_iter" and not finished.converged
    assert finished.n_iter == 5
    np.testing.assert_array_equal(finished.alpha, early.alpha)
    assert finished.max_kkt_violation > TOL


def test_a_non_finite_candidate_is_reported_for_a_fresh_solve():
    _, y, K32 = _problem(n=200)
    C = np.full(200, 1.0)
    alpha = np.zeros(200)
    alpha[3] = np.nan
    finished = _finish(y, K32, C, alpha, 7)
    assert finished.stop_reason == "non-finite" and not finished.converged


def test_a_candidate_is_clipped_into_its_box():
    """A double-float value a hair past its bound is the bound, in float64."""
    _, y, K32 = _problem(n=300)
    C = np.full(300, 1.0)
    done, _ = _solve(y, K32, C)
    nudged = done.alpha.copy()
    nudged[done.alpha >= 1.0] += 1e-15
    nudged[done.alpha <= 0.0] -= 1e-15
    finished = _finish(y, K32, C, nudged, done.n_iter)
    assert finished.alpha.min() >= 0.0 and finished.alpha.max() <= 1.0
    np.testing.assert_array_equal(finished.alpha >= 1.0, done.alpha >= 1.0)


def test_finishing_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="agree in size"):
        smo.finish_c_svc(
            np.ones(3), np.ones(3), np.ones(3), np.ones((4, 4), np.float32), np.zeros(3), 0,
            SMOConfig(),
        )
