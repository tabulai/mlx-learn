"""Layer 2 estimator classes: Layer 1, plus permission to fall back.

Each class here is three lines. That is the entire adapter, and it is deliberate.

A Layer 1 estimator raises when it cannot serve a request, because a library that
quietly does something other than what was asked is worse than one that says no.
A Layer 2 estimator, installed by :func:`~mlxlearn.patch_sklearn` into
scikit-learn's own namespace, is standing in for a class the user never chose,
so raising would break working code; for *capability* mismatches it hands the
work to scikit-learn instead. The whole difference is
``_mlxlearn_allow_fallback``, read by ``BackendMixin._select_backend``.

What this flag does **not** do is make failures disappear. An unexpected MLX error
still raises under every policy, because rerunning a crash on scikit-learn turns a
bug into a silent success and guarantees nobody ever fixes it.

These are real module-level classes rather than ones synthesized with ``type()``
so that a fitted model pickles. ``__name__`` is set to the scikit-learn name so
that ``repr`` reads ``SVC(C=1.0)`` and not ``PatchedSVC(C=1.0)``; ``__qualname__``
is left alone, because that is what :mod:`pickle` looks up.
"""

from __future__ import annotations

from ..linear_model import LogisticRegression as _LogisticRegression
from ..neighbors import KNeighborsClassifier as _KNeighborsClassifier
from ..neighbors import KNeighborsRegressor as _KNeighborsRegressor
from ..neighbors import NearestNeighbors as _NearestNeighbors
from ..svm import SVC as _SVC

__all__ = [
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "LogisticRegression",
    "SVC",
]

_PATCHED_NOTE = """

    Notes
    -----
    This is the patched variant installed by :func:`mlxlearn.patch_sklearn`. It
    behaves exactly like its Layer 1 counterpart except that a *capability*
    mismatch — sparse input, an unimplemented parameter, a problem below the
    measured crossover — is handed to scikit-learn instead of raising. Unexpected
    MLX failures still raise.

    Import the Layer 1 class directly for strict behavior.
    """


def _patched(base: type, sklearn_name: str) -> type:
    """Build the Layer 2 twin of ``base``.

    Used only to attach the shared docstring and display name; the classes below
    are still statically defined, so ``pickle`` and ``inspect`` behave normally.
    """
    base.__doc__ = (base.__doc__ or "") + _PATCHED_NOTE
    base.__name__ = sklearn_name
    return base


class NearestNeighbors(_NearestNeighbors):
    _mlxlearn_allow_fallback = True


class KNeighborsClassifier(_KNeighborsClassifier):
    _mlxlearn_allow_fallback = True


class KNeighborsRegressor(_KNeighborsRegressor):
    _mlxlearn_allow_fallback = True


class LogisticRegression(_LogisticRegression):
    _mlxlearn_allow_fallback = True


class SVC(_SVC):
    _mlxlearn_allow_fallback = True


for _cls, _name in (
    (NearestNeighbors, "NearestNeighbors"),
    (KNeighborsClassifier, "KNeighborsClassifier"),
    (KNeighborsRegressor, "KNeighborsRegressor"),
    (LogisticRegression, "LogisticRegression"),
    (SVC, "SVC"),
):
    _patched(_cls, _name)
