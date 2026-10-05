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
so that a fitted model pickles, and each is named exactly as its scikit-learn
counterpart, so ``repr`` reads ``SVC(C=1.0)`` and :mod:`pickle` finds the class
under the name it records.
"""

from __future__ import annotations

import inspect
import re
import textwrap

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

#: A plain paragraph, not a section. Four of the five Layer 1 docstrings already
#: have a Notes section, and scikit-learn's vendored numpydoc — which parses the
#: docstring on every ``_repr_html_()`` — raises on a duplicate section and warns
#: on an unknown one, so either would break or clutter notebook display.
_PATCHED_NOTE = """\
This is the patched variant installed by :func:`mlxlearn.patch_sklearn`. It
differs from :class:`{layer1}` only in that it may hand work to scikit-learn: on a
capability mismatch — sparse input, an unimplemented parameter — where the
directly imported class raises, and by design on problems stock scikit-learn
serves faster, such as those below the measured crossover, which the directly
imported class runs on MLX regardless. Unexpected MLX failures still raise.
Import :class:`{layer1}` directly for strict behavior."""

#: The first numpydoc section heading: a line of text underlined with dashes.
_FIRST_SECTION = re.compile(r"^\S[^\n]*\n-{3,}[ \t]*$", re.MULTILINE)


def _patched(cls: type, layer1_name: str) -> type:
    """Give a Layer 2 class its Layer 1 documentation plus the patched-variant note.

    The class body has no docstring of its own, so ``cls.__doc__`` is ``None``;
    the note goes into the extended summary of the inherited text, just before
    Parameters, and ``help(sklearn.svm.SVC)`` under ``patch_sklearn()`` still
    documents every parameter. ``inspect.getdoc`` also normalizes indentation,
    which Python 3.13 strips at compile time and 3.10-3.12 do not.
    """
    doc = inspect.getdoc(cls.__mro__[1]) or ""
    note = textwrap.fill(
        " ".join(_PATCHED_NOTE.format(layer1=layer1_name).split()),
        width=79, break_long_words=False, break_on_hyphens=False,
    )
    section = _FIRST_SECTION.search(doc)
    if section is None:
        cls.__doc__ = f"{doc}\n\n{note}" if doc else note
    else:
        head, tail = doc[: section.start()].rstrip(), doc[section.start():]
        cls.__doc__ = f"{head}\n\n{note}\n\n{tail}" if head else f"{note}\n\n{tail}"
    return cls


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


for _cls, _layer1 in (
    (NearestNeighbors, "mlxlearn.neighbors.NearestNeighbors"),
    (KNeighborsClassifier, "mlxlearn.neighbors.KNeighborsClassifier"),
    (KNeighborsRegressor, "mlxlearn.neighbors.KNeighborsRegressor"),
    (LogisticRegression, "mlxlearn.linear_model.LogisticRegression"),
    (SVC, "mlxlearn.svm.SVC"),
):
    _patched(_cls, _layer1)
