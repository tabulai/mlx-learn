"""Linear models, accelerated with MLX.

0.1.0 ships two estimators. :class:`LogisticRegression` covers one solver:
L-BFGS on the L2-penalized (or unpenalized) multinomial/binomial log-loss.
:class:`LinearRegression` covers dense ordinary least squares, including
multi-output targets and sample weights. ``positive=True`` raises rather than
being quietly ignored — non-negative least squares is a different solver.
"""

from __future__ import annotations

from ._linear import LinearRegression
from ._logistic import LogisticRegression

__all__ = ["LinearRegression", "LogisticRegression"]
