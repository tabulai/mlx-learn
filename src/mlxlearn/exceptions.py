"""Public exception and warning hierarchy.

The whole point of this module is that mlxlearn's failures are *legible*. A user
who imports an estimator directly gets a precise error naming what mlxlearn could
not do; a user running under :func:`mlxlearn.patch_sklearn` gets the same
information as a recorded diagnostic event, and — for capability mismatches only —
a transparent fall back to stock scikit-learn.

The distinction that matters is between:

**Capability mismatch** (:class:`UnsupportedParameterError`,
:class:`UnsupportedInputError`, :class:`SparseInputError`)
    mlxlearn understood the request and knows it cannot serve it. Falling back to
    scikit-learn produces a correct answer, so patched mode does exactly that.

**Backend failure** (:class:`BackendExecutionError`, :class:`DeviceError`)
    Something went wrong inside MLX. Rerunning on scikit-learn would produce a
    correct-looking answer that conceals a real bug, so patched mode does *not*
    do that: it raises. This is a deliberate reversal of the "patched mode can
    never raise" idea, which trades away every chance of ever finding such bugs.

Every class inherits from :class:`MLXLearnError` so callers can catch the family,
and the input-validation classes additionally inherit from :class:`ValueError` so
that code written against scikit-learn's error contract keeps working.

Every class also survives :mod:`pickle`. That is not a nicety: joblib sends a
worker's exception back to the parent by pickling it, so an error that cannot be
unpickled turns ``cross_val_score(..., n_jobs=2)`` into an opaque
``BrokenProcessPool`` with the precise message gone from the traceback.
"""

from __future__ import annotations

import pickle
from typing import Any

__all__ = [
    "MLXLearnError",
    "CapabilityError",
    "UnsupportedParameterError",
    "UnsupportedInputError",
    "SparseInputError",
    "InvalidInputError",
    "UnsupportedEstimatorStateError",
    "DeviceError",
    "BackendExecutionError",
    "MLXLearnFallbackWarning",
    "MLXLearnPerformanceWarning",
    "MLXLearnPatchWarning",
]


def _rebuild_exception(cls: type, args: tuple) -> BaseException:
    """Recreate an mlxlearn exception on unpickling, without calling ``__init__``.

    ``BaseException.__new__`` restores ``args``; the attributes follow from the
    pickled ``__dict__``. Going through ``__init__`` instead — the default — calls
    ``cls(*args)``, which fails for any class with a required keyword-only
    argument, such as :class:`UnsupportedParameterError`'s ``parameter``.
    """
    return cls.__new__(cls, *args)


def _portable_state(state: dict[str, Any]) -> dict[str, Any]:
    """``state`` with every value that cannot be pickled replaced by its ``repr``.

    ``UnsupportedParameterError.value`` can be whatever the user passed — a lambda
    as ``weights`` or ``kernel``, say. Losing the object is fine; losing the whole
    exception to a ``PicklingError`` in a worker process is not.
    """
    portable = {}
    for key, value in state.items():
        try:
            pickle.dumps(value)
        except (pickle.PicklingError, TypeError, AttributeError):
            value = repr(value)
        portable[key] = value
    return portable


class MLXLearnError(Exception):
    """Base class for every error mlxlearn raises on purpose."""

    def __reduce__(self):
        return (_rebuild_exception, (type(self), self.args), _portable_state(self.__dict__))


class CapabilityError(MLXLearnError):
    """mlxlearn understood the request but cannot serve it.

    Patched mode may fall back to scikit-learn for this family, because doing so
    yields a correct result. Direct use raises, because silently doing something
    other than what was asked is worse than an error.

    Parameters
    ----------
    message : str
        Human-readable explanation, phrased as what mlxlearn does not support.
    reason : str
        Short machine-readable slug recorded in diagnostics, e.g.
        ``"sparse-input"`` or ``"unsupported-solver"``. Used to deduplicate
        warnings and to group fallback statistics.
    advice : str, optional
        What a caller using mlxlearn *directly* can do instead — typically
        "run under ``patch_sklearn()``". Appended to the message. Kept apart
        because a patched estimator is already under ``patch_sklearn()``: it
        reports the mismatch as :attr:`summary`, without the advice, and says
        what it did about it.

    Attributes
    ----------
    reason : str
    summary : str
        The message without :attr:`advice`.
    advice : str
    """

    def __init__(self, message: str, *, reason: str = "capability", advice: str = "") -> None:
        super().__init__(f"{message} {advice}" if advice else message)
        self.reason = reason
        self.summary = message
        self.advice = advice


class UnsupportedParameterError(CapabilityError, ValueError):
    """A constructor parameter has a value mlxlearn's implementation does not cover.

    The value is typically valid scikit-learn input — ``solver="liblinear"``,
    ``kernel="precomputed"`` — just not something the MLX path implements.
    """

    def __init__(
        self, message: str, *, parameter: str, value: object = None, advice: str = ""
    ) -> None:
        super().__init__(message, reason=f"unsupported-parameter:{parameter}", advice=advice)
        self.parameter = parameter
        self.value = value


class UnsupportedInputError(CapabilityError, ValueError):
    """The input data has a property the MLX path cannot handle.

    Object dtypes and float64 values that overflow float32 are the common cases.
    """

    def __init__(
        self, message: str, *, reason: str = "unsupported-input", advice: str = ""
    ) -> None:
        super().__init__(message, reason=reason, advice=advice)


class SparseInputError(CapabilityError, TypeError):
    """Sparse input was given to a dense-only MLX path.

    Inherits :class:`TypeError`, not :class:`ValueError`, because that is what
    scikit-learn's own estimators raise for unsupported sparse input and what its
    estimator checks look for. The message always contains the word "sparse", for
    the same reason.
    """

    def __init__(self, message: str, *, advice: str = "") -> None:
        super().__init__(message, reason="sparse-input", advice=advice)


class UnsupportedEstimatorStateError(CapabilityError, ValueError):
    """The fitted state cannot be represented or reused on the MLX path.

    For an estimator fitted elsewhere and handed to an MLX inference path with
    state it cannot interpret.

    Reserved: nothing in 0.1.x raises it, because inference never re-decides the
    backend — a model fitted on scikit-learn predicts on scikit-learn. It is part of
    the hierarchy now so that the first path to need it does not have to extend
    the public API to say so.
    """

    def __init__(self, message: str, *, reason: str = "unsupported-state") -> None:
        super().__init__(message, reason=reason)


class DeviceError(MLXLearnError, RuntimeError):
    """A requested device is unavailable or unusable.

    ``device="gpu"`` on a machine with no working Metal device raises this rather
    than quietly computing on the CPU, because a user who named a device wants
    that device.
    """


class BackendExecutionError(MLXLearnError, RuntimeError):
    """An MLX computation failed unexpectedly.

    This is a bug — in mlxlearn, in MLX, or in the driver — and it is never
    resolved by rerunning the work on scikit-learn. The original exception is
    attached as ``__cause__`` and also as :attr:`original`.
    """

    def __init__(self, message: str, *, original: BaseException | None = None) -> None:
        super().__init__(message)
        self.original = original


class MLXLearnFallbackWarning(UserWarning):
    """Patched mode handed a call to stock scikit-learn.

    Emitted once per (estimator class, reason) pair, because an estimator that
    falls back once in a loop falls back every iteration and the second warning
    onward is noise. The full, undeduplicated record is always available from
    :func:`mlxlearn.get_backend_diagnostics`.
    """


class MLXLearnPerformanceWarning(UserWarning):
    """A request will run, but on a path known to be slow.

    Never emitted for the routine "problem is small, scikit-learn is faster here"
    dispatch, which is the system working as designed.

    Reserved: nothing in 0.1.x emits it. The paths that are not accelerated — such
    as ``radius_neighbors`` — are documented rather than warned about on every call,
    and are no slower than stock scikit-learn. The category exists so that warning
    filters written against it keep working once a genuinely slow path does warn.
    """


class MLXLearnPatchWarning(UserWarning):
    """Something about the patching operation itself needs attention.

    For example: patching was requested for an estimator with no mlxlearn
    implementation, or the attribute being replaced is not the object mlxlearn
    recorded when it patched.
    """


class InvalidInputError(MLXLearnError, ValueError):
    """The input is invalid — scikit-learn would reject it too.

    Not a capability mismatch: falling back cannot help, so patched mode raises
    it like any other bad input. It exists so that a deliberate input error
    raised from inside an MLX-guarded block is told apart from an MLX-internal
    ``ValueError``, which :func:`~mlxlearn._common.base.mlx_guard` reports as a
    :class:`BackendExecutionError`. A ``ValueError`` subclass, so code written
    against scikit-learn's error contract keeps working.
    """

