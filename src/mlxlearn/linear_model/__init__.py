"""Linear models, accelerated with MLX.

0.1.0 ships one estimator, :class:`LogisticRegression`, and it covers one solver:
L-BFGS on the L2-penalized (or unpenalized) multinomial/binomial log-loss. Every
other ``solver`` scikit-learn offers raises rather than being quietly redirected to
this one — ``liblinear`` and ``saga`` minimize the same objective by different
routes and stop at different points, so serving them from here would return
answers that are close enough to look right and different enough to matter.
"""

from __future__ import annotations

from ._logistic import LogisticRegression

__all__ = ["LogisticRegression"]
