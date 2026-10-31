"""Optional native SMO: fused scans and updates, with double-precision state.

The full-Gram path enters native code once for the entire optimization. The
bounded row-cache path returns to Python only on a missing row, never for
working-set arithmetic. NumPy is used here only for buffer conversion and output
allocation; the optimizer has no NumPy C API dependency.
"""

from __future__ import annotations

import importlib

import numpy as np

from .smo import SMOConfig, SMOState

try:
    _smo_native = importlib.import_module(f"{__package__}._smo_native")
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}._smo_native":
        raise
    _smo_native = None


def available() -> bool:
    """Whether the optional native extension was built for this installation."""
    return _smo_native is not None


def solve_c_svc(y, C, k_diag, fetch_row, config: SMOConfig, *, gram=None) -> SMOState | None:
    """Solve one binary C-SVC problem, or return ``None`` without the extension.

    ``gram`` optionally supplies the already budgeted dense float32 kernel
    matrix. Otherwise ``fetch_row(i)`` must return a contiguous float32 vector;
    the native LRU owns copies so callbacks may safely reuse a scratch buffer.
    The reported KKT violation is evaluated on the returned iterate, including
    zero and exhausted budgets. ``converged`` retains the existing convention:
    it is true only when an iteration's stopping test ran before budget expiry.
    With ``config.shrinking``, bound coordinates are provisionally screened;
    all gradients are reconstructed before convergence or a budgeted return.
    """
    if _smo_native is None:
        return None
    labels = np.require(y, dtype=np.float64, requirements=["C", "A"])
    bounds = np.require(C, dtype=np.float64, requirements=["C", "A"])
    diagonal = np.require(k_diag, dtype=np.float64, requirements=["C", "A"])
    alpha = np.empty(labels.shape, dtype=np.float64)
    if gram is not None:
        gram = np.require(gram, dtype=np.float32, requirements=["C", "A"])
    rho, n_iter, converged, violation = _smo_native.solve(
        labels, bounds, diagonal, alpha, gram, fetch_row,
        float(config.tol), int(config.max_iter), int(config.cache_rows), float(config.tau),
        bool(getattr(config, "shrinking", False)),
    )
    return SMOState(
        alpha=alpha, rho=rho, n_iter=n_iter, converged=converged,
        max_kkt_violation=violation,
    )
