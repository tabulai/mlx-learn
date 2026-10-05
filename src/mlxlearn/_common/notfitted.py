"""The not-fitted error for an estimator whose fit did not complete.

It lives apart from :mod:`mlxlearn.exceptions` for one reason: it subclasses
scikit-learn's ``NotFittedError``, and ``import mlxlearn`` deliberately does not
import scikit-learn — that import costs most of a second, and it initializes
the runtime state that makes MLX unusable in a forked child. Nothing reaches this
module before an estimator module has loaded scikit-learn anyway.
"""

from __future__ import annotations

from sklearn.exceptions import NotFittedError

from ..exceptions import MLXLearnError

__all__ = ["IncompleteFitError"]


class IncompleteFitError(MLXLearnError, NotFittedError):
    """Inference was asked of an estimator whose last ``fit`` did not complete.

    A failed refit leaves the previous fit's public attributes in place, so the
    estimator looks fitted, but mlxlearn has forgotten which backend owns the
    state and will not guess. Both an :class:`~mlxlearn.exceptions.MLXLearnError`
    and scikit-learn's ``NotFittedError`` (so also a ``ValueError`` and an
    ``AttributeError``), which is what code written against scikit-learn catches.
    """
