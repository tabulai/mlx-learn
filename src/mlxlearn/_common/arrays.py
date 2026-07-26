"""Conversions across the NumPy/MLX boundary.

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
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "to_mlx",
    "to_numpy",
    "as_float32",
    "is_mlx_array",
    "sync",
    "MLX_FLOAT",
    "NUMPY_FLOAT",
    "INDEX_DTYPE",
]

#: The compute dtype for every MLX path in 0.1.0.
NUMPY_FLOAT = np.float32

#: NumPy dtype used for returned neighbor indices, matching scikit-learn.
INDEX_DTYPE = np.int64


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
    if not module.startswith("mlx"):
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
        Use this when the result will be mutated.

    Notes
    -----
    Non-finite values pass through untouched. Rejecting them is validation's job,
    not conversion's, and doing it here would make this function expensive for
    callers that already checked.
    """
    if is_mlx_array(array):
        array = to_numpy(array)
    out = np.asarray(array, dtype=NUMPY_FLOAT, order="C")
    if copy and out is array:
        out = out.copy()
    return out


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
    """
    if is_mlx_array(array):
        sync(array)
        out = np.asarray(array)
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
