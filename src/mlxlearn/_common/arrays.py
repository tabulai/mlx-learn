"""Conversions across the NumPy/MLX boundary, and what survives them in float32.

Written to be public-safe from the start, because the 0.2.x experimental
``mx.array`` boundary will expose it: no module-level state, no caches keyed on
the identity of a user's array, and device and dtype always explicit.

The no-cache rule is not stylistic. The ancestor kept module-level weakrefs to the
last training and query matrices so it could skip re-uploading them. That is a
correctness hazard in exactly the situation mlxlearn targets — an agent refitting
a probe on new data — because a caller who mutates an array in place, or gets a
new array at the same address after a free, silently gets stale device data. Any
caching that happens here must be owned by a fitted estimator and die with it.

**fp32 policy.** MLX computes in float32 on the GPU; float64 is not available
there. Rather than pretend otherwise, mlxlearn converts float64 input to float32
for computation and says so. Where that changes an answer relative to
scikit-learn's float64 arithmetic, the deviation is documented and tested to
float32 tolerances — see ``docs/fp32_policy.md``. Integer and boolean inputs are
promoted to float32 the same way scikit-learn promotes them to float64.

**Range.** Converting is only half the question; the other half is whether the
values stay representable through the arithmetic done on them. :func:`finite_range`
measures an array cheaply, and :func:`check_squared_float32_range` answers the
question kernels actually face — whether *squares* of the data stay inside float32
— for a value that fits in float32 but whose square does not. No 0.1.x estimator
needs that guard: each rescales its data, and the kernels check their own compute
frames. What is here stays free of scikit-learn, which ``validation`` imports and a
kernel module must not, so the kernels share its constants — the float32 limits,
and the ``patch_sklearn()`` advice their fit-time errors carry.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from ..exceptions import UnsupportedInputError

__all__ = [
    "to_mlx",
    "to_numpy",
    "as_float32",
    "is_mlx_array",
    "sync",
    "finite_range",
    "check_squared_float32_range",
    "ValueRange",
    "MLX_FLOAT",
    "NUMPY_FLOAT",
    "INDEX_DTYPE",
    "FLOAT32_MAX",
    "FLOAT32_TINY",
]

#: The compute dtype for every MLX path in 0.1.0.
NUMPY_FLOAT = np.float32

#: NumPy dtype used for returned neighbor indices, matching scikit-learn.
INDEX_DTYPE = np.int64

#: float32's largest finite value. A float64 above this becomes ``inf`` on conversion.
FLOAT32_MAX = float(np.finfo(np.float32).max)

#: float32's smallest *normal* value. Below it lie the subnormals, which the GPU
#: flushes to zero.
FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _mx():
    import mlx.core as mx

    return mx


def _mlx_float():
    return _mx().float32


class _LazyMLXFloat:
    """``MLX_FLOAT`` without importing mlx at module import time.

    Importing ``mlx.core`` costs real time and initializes Metal. A user who
    imports ``mlxlearn`` to read ``__version__`` should not pay for that.
    """

    def __repr__(self) -> str:  # pragma: no cover - display only
        return "mlx.core.float32"

    def __call__(self):
        return _mlx_float()


MLX_FLOAT = _LazyMLXFloat()


def is_mlx_array(obj: Any) -> bool:
    """Whether ``obj`` is an ``mx.array``, without importing mlx if it obviously is not."""
    module = type(obj).__module__
    # "mlx" or "mlx.<anything>", but not "mlxlearn": a prefix test alone would
    # import MLX to ask the question about every mlxlearn object.
    if module != "mlx" and not module.startswith("mlx."):
        return False
    return isinstance(obj, _mx().array)


def as_float32(array: Any, *, copy: bool = False) -> np.ndarray:
    """Return ``array`` as a C-contiguous float32 NumPy array.

    Parameters
    ----------
    array : array-like
        Anything ``np.asarray`` accepts, or an ``mx.array``.
    copy : bool, default=False
        Force a copy even when the input already satisfies the requirements.
        Use this when the result will be mutated. The copy is a fresh base
        ``ndarray`` that shares memory with nothing — including for ``np.matrix``,
        ``np.memmap`` and pandas input, where ``np.asarray`` hands back a *new
        object* that still views the caller's buffer.

    Notes
    -----
    Non-finite values pass through untouched. Rejecting them is validation's job,
    not conversion's, and doing it here would make this function expensive for
    callers that already checked.

    Without ``copy``, the result may be a read-only view: of an ``mx.array``'s
    buffer (see :func:`to_numpy`), or of a pandas frame under copy-on-write.
    """
    if is_mlx_array(array):
        array = to_numpy(array)
    if copy:
        return np.array(array, dtype=NUMPY_FLOAT, order="C", copy=True)
    return np.asarray(array, dtype=NUMPY_FLOAT, order="C")


def to_mlx(array: Any, *, dtype: Any = None, device: str | None = None):
    """Move ``array`` onto an MLX device.

    Parameters
    ----------
    array : array-like or mx.array
    dtype : mlx dtype, optional
        Defaults to ``mx.float32``. Pass explicitly when a different dtype is
        genuinely wanted; there is no implicit dtype policy hidden in here.
    device : {"auto", "gpu", "cpu"}, optional
        Where the conversion runs. Defaults to the configured device.

    Returns
    -------
    mx.array
        Never a view of a NumPy input: ``mx.array`` copies host memory, so
        mutating the caller's array afterwards changes nothing on the device.
    """
    from .device import use_device

    mx = _mx()
    target_dtype = dtype if dtype is not None else mx.float32

    if is_mlx_array(array):
        with use_device(device):
            return array if array.dtype == target_dtype else array.astype(target_dtype)

    host = np.asarray(array)
    if host.dtype == np.float64 and target_dtype == mx.float32:
        # Downcast on the host: mx.array would otherwise materialize a float64
        # buffer MLX cannot use on the GPU anyway.
        host = host.astype(np.float32, copy=False)
    host = np.ascontiguousarray(host)
    with use_device(device):
        return mx.array(host, dtype=target_dtype)


def to_numpy(array: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Bring ``array`` back to the host as a NumPy array.

    Evaluation is forced first. MLX is lazy, so without an explicit
    :func:`sync` the copy would silently include the cost of every pending
    operation, which makes benchmark numbers meaningless and makes exceptions
    surface far from the code that caused them.

    For an ``mx.array`` the result is a zero-copy view of MLX's buffer — on
    unified memory there is nothing to copy — and it is marked **read-only**.
    Writing through it would change the ``mx.array`` underneath, and every lazy
    graph built from it that has not been evaluated yet; a caller that wants to
    mutate asks for a copy (``np.array(...)`` or ``as_float32(..., copy=True)``).
    A requested ``dtype`` that differs from the array's produces a fresh, writable
    array.
    """
    if is_mlx_array(array):
        sync(array)
        out = np.asarray(array)
        out.flags.writeable = False
    else:
        out = np.asarray(array)
    if dtype is not None and out.dtype != dtype:
        out = out.astype(dtype, copy=False)
    return out


def sync(*arrays: Any) -> None:
    """Force evaluation of pending MLX work.

    Call before timing anything and before crossing back to the host. Passing
    non-MLX values is harmless, so callers do not need to filter.
    """
    pending = [a for a in arrays if is_mlx_array(a)]
    if pending:
        _mx().eval(*pending)


# ------------------------------------------------------------------ float32 range


class ValueRange(NamedTuple):
    """The smallest and largest finite values of an array, as Python floats."""

    low: float
    high: float

    @property
    def magnitude(self) -> float:
        """The largest absolute value, ``max(|low|, |high|)``."""
        return max(abs(self.low), abs(self.high))

    @property
    def span(self) -> float:
        """``high - low``.

        A bound on every coordinate difference ``|a_j - b_j|`` between two rows,
        and on ``|x_j - c_j|`` for any center ``c`` inside the data's range — the
        quantities a distance or a centered kernel actually squares.
        """
        return self.high - self.low


def finite_range(array: Any) -> ValueRange | None:
    """The smallest and largest finite values of a real-numeric array.

    Returns ``None`` when there is nothing to report: an empty array, one with no
    finite value at all, or a dtype that is not real-numeric (object, string,
    complex). Reading such data as numbers is validation's decision, not this
    function's.

    Two reductions and no temporaries in the common all-finite case, split by rows
    across threads for a large matrix (:func:`~mlxlearn._common.host.extremes`). A
    NaN or an infinity makes the maximum or minimum non-finite, and only then does a
    masked pass run — so the cost of non-finite input is paid only by non-finite
    input, and even then without copying the finite values out.
    """
    from .host import extremes

    array = np.asarray(array)
    if array.size == 0 or array.dtype.kind not in "biuf":
        return None
    high, low = extremes(array)
    if not (np.isfinite(high) and np.isfinite(low)):
        finite = np.isfinite(array)
        high = np.max(array, where=finite, initial=-np.inf)
        low = np.min(array, where=finite, initial=np.inf)
        if high < low:  # no finite value: the initial sentinels survived
            return None
    return ValueRange(float(low), float(high))


#: The Layer 1 advice for data float32 cannot hold, at ``fit``: a patched estimator
#: hands such data to scikit-learn's float64 path (and leaves this out of its own
#: messages; see CapabilityError.advice).
_FLOAT64_FIT_ADVICE = (
    "Or run under patch_sklearn(), which hands a fit on such data to scikit-learn's "
    "float64 path."
)

#: Factor of headroom the squared-range guard keeps below float32's largest value:
#: the Gram form ``‖a‖² + ‖b‖² − 2⟨a, b⟩`` adds, and doubles, squared norms before
#: its subtraction brings the result back down.
_SQUARED_HEADROOM = 4.0

#: Squares at or above ``FLOAT32_TINY * 2**24`` keep float32's full 24-bit
#: resolution above the subnormal range: anything the data's own precision can
#: distinguish from its largest square is still a normal number.
_SQUARED_FLOOR = FLOAT32_TINY * 2.0**24


def check_squared_float32_range(
    magnitude: float, *, n_terms: int = 1, name: str = "X"
) -> None:
    """Reject data whose squares leave float32's range, though the data itself does not.

    A value can be comfortably finite in float32 and still have a square that is
    not: ``|x| ≈ 1e19`` squares to ``1e38``, and a sum of ten of those is past
    float32's ``3.4e38``. At the other end, ``|x| ≈ 1e-19`` squares into the
    subnormal range, which the GPU flushes to zero. Distances and Gram-form kernels
    computed in float32 on such data come out infinite or zero — silently, and
    with a successful status.

    Exact power-of-two rescaling avoids both ends where an algorithm can undo the
    scale afterwards, and is the better fix when available. This guard is for the
    paths that cannot, and for queries arriving outside the range a fitted scale
    was chosen for.

    Parameters
    ----------
    magnitude : float
        Bound on the absolute values that get squared. For a distance, or a kernel
        on centered data, that is the per-coordinate :attr:`ValueRange.span`; for
        an uncentered inner product it is :attr:`ValueRange.magnitude`.
    n_terms : int, default=1
        How many squares are summed — the number of features, for a squared norm.
    name : str, default="X"
        What to call the data in the message.

    Raises
    ------
    UnsupportedInputError
        ``reason="float32-overflow"`` when ``n_terms · magnitude²``, with a factor
        of four headroom, exceeds float32's largest finite value (or ``magnitude``
        is not finite); ``reason="float32-underflow"`` when ``magnitude²`` is
        within 24 bits of float32's smallest normal value, where the squares of
        the data's finer differences become subnormal.
    """
    magnitude = abs(float(magnitude))
    squared = magnitude * magnitude  # a Python float: large inputs give inf, never an error
    if not math.isfinite(magnitude) or _SQUARED_HEADROOM * n_terms * squared > FLOAT32_MAX:
        raise UnsupportedInputError(
            f"{name} has values spanning {magnitude:.3g}; summed over {n_terms} "
            f"feature(s), their squares exceed float32's largest finite value "
            f"({FLOAT32_MAX:.3g}), so float32 distances and kernels on this data "
            "would overflow to infinity. mlxlearn computes in float32 and will not "
            "return those silently. Rescale the data.",
            reason="float32-overflow",
            advice=_FLOAT64_FIT_ADVICE,
        )
    if 0.0 < magnitude and squared < _SQUARED_FLOOR:
        raise UnsupportedInputError(
            f"{name} has values spanning only {magnitude:.3g}; their squares fall "
            "into float32's subnormal range, which the GPU flushes to zero, so "
            "float32 distances and kernels on this data would collapse to zero. "
            "mlxlearn computes in float32 and will not return those silently. "
            "Rescale the data.",
            reason="float32-underflow",
            advice=_FLOAT64_FIT_ADVICE,
        )
