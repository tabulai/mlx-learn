"""Native hierarchical C-SVC optimization with double-precision certification.

Working-set decomposition follows the algorithmic idea described by Wen et al.
(ThunderSVM, JMLR 2018). The implementation
is independent: analytic float64 diagonals, float32 kernel rows, float64 optimizer
state, true pair-update budgets, and a reconstructed full KKT stopping test retain
mlxlearn's numerical contract. No upstream implementation code is incorporated.
"""

from __future__ import annotations

import importlib

import numpy as np

from .smo import SMOConfig, SMOState

try:
    _smo_block_native = importlib.import_module(f"{__package__}._smo_block_native")
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}._smo_block_native":
        raise
    _smo_block_native = None


def available() -> bool:
    return _smo_block_native is not None


def solve_c_svc(
    y, C, k_diag, fetch_rows, config: SMOConfig, *, gram=None, working_set_size=128,
) -> SMOState | None:
    """Solve with native local subproblems and one global update per subproblem.

    ``fetch_rows(indices)`` receives a Python list of row indices and returns a
    C-contiguous float32 matrix shaped ``(len(indices), len(y))``. Missing rows
    are batched and copied into an LRU as needed. The estimator reserves one
    working set of ``config.cache_rows`` for the callback and uses the remaining
    rows for the LRU. An explicitly requested working set always has room for
    its own rows. A supplied ``gram`` bypasses the row cache and callback.

    ``max_iter`` counts actual pair updates, including a stalled attempted update.
    Convergence requires a freshly reconstructed gradient on every sample; final
    residuals and rho also use a fresh gradient at exhausted/zero budgets. A
    numerical stall returns ``converged=False`` rather than weakening tolerance.
    Fixed zero-width bounds are excluded from feasible-direction selection.
    """
    if _smo_block_native is None:
        return None
    labels = np.require(y, dtype=np.float64, requirements=["C", "A"])
    bounds = np.require(C, dtype=np.float64, requirements=["C", "A"])
    diagonal = np.require(k_diag, dtype=np.float64, requirements=["C", "A"])
    alpha = np.empty(labels.shape, dtype=np.float64)
    cache_arg = ()
    if gram is not None:
        gram = np.require(gram, dtype=np.float32, requirements=["C", "A"])
    elif getattr(_smo_block_native, "SUPPORTS_ROW_CACHE", False):
        # Older editable builds accept just the original ten arguments.
        width = min(len(labels), int(working_set_size))
        retained = max(width, int(config.cache_rows) - width)
        cache_arg = (min(len(labels), retained),)
    rho, iterations, converged, gap = _smo_block_native.solve(
        labels, bounds, diagonal, alpha, gram, fetch_rows,
        float(config.tol), int(config.max_iter), int(working_set_size), float(config.tau), *cache_arg,
    )
    return SMOState(alpha, rho, iterations, converged, gap)
