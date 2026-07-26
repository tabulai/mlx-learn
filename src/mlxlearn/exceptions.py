"""Public exception and warning hierarchy.

The whole point of this module is that mlxlearn's failures are *legible*. A user
who imports an estimator directly gets a precise error naming what mlxlearn could
not do; a user running under :func:`mlxlearn.patch_sklearn` gets the same
information as a recorded diagnostic event, and — for capability mismatches only —
a transparent fall back to stock scikit-learn.

The distinction that matters is between:

**Capability mismatch** (:class:`UnsupportedParameterError`,
:class:`UnsupportedInputError`, :class:`UnsupportedEstimatorStateError`)
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
"""

from __future__ import annotations

__all__ = [
    "MLXLearnError",
    "CapabilityError",
    "UnsupportedParameterError",
    "UnsupportedInputError",
    "SparseInputError",
    "UnsupportedEstimatorStateError",
    "DeviceError",
    "BackendExecutionError",
    "NotSupportedByBackend",
    "MLXLearnFallbackWarning",
    "MLXLearnPerformanceWarning",
    "MLXLearnPatchWarning",
]


class MLXLearnError(Exception):
    """Base class for every error mlxlearn raises on purpose."""


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
    """

    def __init__(self, message: str, *, reason: str = "capability") -> None:
        super().__init__(message)
        self.reason = reason


class UnsupportedParameterError(CapabilityError, ValueError):
    """A constructor parameter has a value mlxlearn's implementation does not cover.

    The value is typically valid scikit-learn input — ``solver="liblinear"``,
    ``kernel="precomputed"`` — just not something the MLX path implements.
    """

    def __init__(self, message: str, *, parameter: str, value: object = None) -> None:
        super().__init__(message, reason=f"unsupported-parameter:{parameter}")
        self.parameter = parameter
        self.value = value


class UnsupportedInputError(CapabilityError, ValueError):
    """The input data has a property the MLX path cannot handle.

    Object dtypes and float64 values that overflow float32 are the common cases.
    """

    def __init__(self, message: str, *, reason: str = "unsupported-input") -> None:
        super().__init__(message, reason=reason)


class SparseInputError(CapabilityError, TypeError):
    """Sparse input was given to a dense-only MLX path.

    Inherits :class:`TypeError`, not :class:`ValueError`, because that is what
    scikit-learn's own estimators raise for unsupported sparse input and what its
    estimator checks look for. The message always contains the word "sparse", for
    the same reason.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="sparse-input")


class UnsupportedEstimatorStateError(CapabilityError, ValueError):
    """The fitted state cannot be represented or reused on the MLX path.

    Raised when an estimator fitted elsewhere is handed to an MLX inference path
    with state it cannot interpret.
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


#: Historical alias. Some early notes used this name for the capability family.
NotSupportedByBackend = CapabilityError


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
    """


class MLXLearnPatchWarning(UserWarning):
    """Something about the patching operation itself needs attention.

    For example: patching was requested for an estimator with no mlxlearn
    implementation, or the attribute being replaced is not the object mlxlearn
    recorded when it patched.
    """
