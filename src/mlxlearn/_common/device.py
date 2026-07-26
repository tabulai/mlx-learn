"""Device selection and capability probing.

MLX has more than one backend, so the public device names are MLX's — ``"auto"``,
``"gpu"``, ``"cpu"`` — and never ``"mps"``. ``"mps"`` names a PyTorch backend; using
it here would be wrong and would age badly.

The rules this module enforces:

* ``device="gpu"`` on a machine with no usable GPU is a :class:`DeviceError`. A
  user who named a device wants that device; quietly computing somewhere else and
  reporting success is the kind of thing that makes a benchmark meaningless.
* ``device="auto"`` prefers the GPU and degrades to the CPU without complaint.
* Probing happens once per process and is cached, because it costs a small MLX
  allocation and the answer cannot change while the process runs.
* :func:`use_device` restores the previous MLX default device on the way out,
  including when the body raises, so mlxlearn never leaves a process-global
  setting changed behind a user's back.
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass

from ..exceptions import DeviceError
from .config import get_config

__all__ = [
    "DeviceInfo",
    "resolve_device",
    "use_device",
    "gpu_available",
    "device_info",
    "physical_memory_bytes",
]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """What mlxlearn knows about where it is running."""

    kind: str  # "gpu" or "cpu"
    gpu_available: bool
    mlx_version: str
    physical_memory_bytes: int | None

    def __str__(self) -> str:  # pragma: no cover - display only
        mem = (
            f"{self.physical_memory_bytes / 1024**3:.0f} GiB"
            if self.physical_memory_bytes
            else "unknown memory"
        )
        return f"mlx {self.mlx_version} on {self.kind} ({mem})"


@functools.lru_cache(maxsize=1)
def physical_memory_bytes() -> int | None:
    """Total physical memory, or ``None`` where the platform will not say.

    Used to bound the SVM kernel cache and the distance-block target. On unified
    memory this is also, usefully, the GPU's memory.
    """
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return page_size * page_count


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Whether a Metal device is present *and* actually executes work.

    Importing ``mlx`` succeeding is not evidence that the GPU works, so this runs
    a tiny kernel and forces evaluation. The result is cached for the process.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return False
    try:
        probe = mx.array([1.0, 2.0], dtype=mx.float32)
        with mx.stream(mx.gpu):
            result = mx.sum(probe * probe)
            mx.eval(result)
        return float(result) == 5.0
    except Exception:
        # Any failure here means "no usable GPU". Deliberately broad: MLX raises
        # several unrelated exception types when Metal is missing or unusable,
        # and the answer is the same for all of them.
        return False


@functools.lru_cache(maxsize=1)
def _mlx_version() -> str:
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - mlx is a hard dependency
        return "unavailable"
    return getattr(mx, "__version__", "unknown")


def resolve_device(device: str | None = None) -> str:
    """Resolve a device request to ``"gpu"`` or ``"cpu"``.

    Parameters
    ----------
    device : {"auto", "gpu", "cpu"}, optional
        Defaults to the current configuration's ``device``.

    Returns
    -------
    str
        ``"gpu"`` or ``"cpu"``.

    Raises
    ------
    DeviceError
        If ``"gpu"`` was requested explicitly and no usable GPU exists.
    """
    requested = device if device is not None else get_config().device
    if requested not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu' or 'cpu'; got {requested!r}")

    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not gpu_available():
            raise DeviceError(
                "device='gpu' was requested but no usable MLX GPU device is available. "
                "mlxlearn will not silently compute on the CPU when a device was named "
                "explicitly; pass device='auto' if a CPU fallback is acceptable."
            )
        return "gpu"
    return "gpu" if gpu_available() else "cpu"


def device_info(device: str | None = None) -> DeviceInfo:
    """Describe the device a request would resolve to."""
    return DeviceInfo(
        kind=resolve_device(device),
        gpu_available=gpu_available(),
        mlx_version=_mlx_version(),
        physical_memory_bytes=physical_memory_bytes(),
    )


@contextmanager
def use_device(device: str | None = None):
    """Run a block on the resolved MLX device.

    Yields the resolved kind (``"gpu"`` or ``"cpu"``). Uses ``mx.stream`` rather
    than ``mx.set_default_device`` so that concurrent work in other threads is not
    disturbed by a process-global mutation.

    Examples
    --------
    >>> from mlxlearn._common.device import use_device
    >>> with use_device("cpu") as kind:
    ...     kind
    'cpu'
    """
    import mlx.core as mx

    kind = resolve_device(device)
    target = mx.gpu if kind == "gpu" else mx.cpu
    with mx.stream(target):
        yield kind


def reset_device_cache() -> None:
    """Forget the cached probe results. Tests only."""
    gpu_available.cache_clear()
    physical_memory_bytes.cache_clear()
    _mlx_version.cache_clear()
