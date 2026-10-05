"""Device selection and capability probing.

MLX has more than one backend, so the public device names are MLX's — ``"auto"``,
``"gpu"``, ``"cpu"`` — and never ``"mps"``. ``"mps"`` names a PyTorch backend; using
it here would be wrong and would age badly.

The rules this module enforces:

* ``device="gpu"`` on a machine with no usable GPU is a :class:`DeviceError`. A
  user who named a device wants that device; quietly computing somewhere else and
  reporting success is the kind of thing that makes a benchmark meaningless. The
  device choice applies to work that size dispatch has sent to MLX; it does not
  override the crossover (see :class:`~mlxlearn._common.config.Config`).
* ``device="auto"`` prefers the GPU and degrades to the CPU without complaint.
* Probing happens once per process and is cached, because it costs a small MLX
  allocation and the answer cannot change while the process runs.
* MLX's default *device* is process-global (its default streams are per thread),
  so :func:`use_device` cannot scope it to one thread. What it does instead is
  make mlxlearn's own device blocks agree: blocks on the same device run
  concurrently, a block that needs the other device waits until they finish,
  and the application's default device is restored when the last block exits.
  Other MLX code running concurrently in the same process, outside mlxlearn, can
  still observe the temporary default. Passing ``stream=`` to every operation
  would be the complete fix, and :func:`use_device` yields the stream for that
  purpose, but it is not reachable yet: as of MLX 0.32, array indexing and
  slicing, ``.T`` and ``mx.compile``'d functions take no ``stream=``, and
  ``with mx.stream(...)`` in one thread changes another thread's default device.
* A process forked after MLX or scikit-learn was loaded cannot use MLX: Metal and
  the Objective-C runtime are not fork-safe, and the child dies with a signal on
  its first MLX operation, even on the CPU. mlxlearn raises
  :class:`DeviceError` there instead, naming the fix (the ``spawn`` or
  ``forkserver`` start method).
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from ..exceptions import DeviceError
from .config import get_config

__all__ = [
    "DeviceInfo",
    "resolve_device",
    "inference_device",
    "use_device",
    "gpu_available",
    "device_info",
    "physical_memory_bytes",
    "forked_after_mlx_init",
    "check_fork_safety",
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


# ------------------------------------------------------------------------- fork safety
#
# The one piece of mutable module state here, and it is not a cache: it records a
# fact about this process's lineage that cannot be recovered after the fact. The
# hook runs in a forked child and notes whether the parent had already loaded MLX
# or scikit-learn (whose import initializes the Objective-C runtime through
# Accelerate). Either makes the child's first MLX operation die with SIGSEGV or
# SIGABRT; `import mlxlearn` alone does not, so it is not flagged. Numerical NumPy
# work in the parent can have the same effect and cannot be detected; that case
# is documented rather than guarded.

_FORK_UNSAFE_MODULES = ("mlx.core", "sklearn")
_forked_after_init = False


def _note_fork_in_child() -> None:
    global _forked_after_init
    if not _forked_after_init:
        _forked_after_init = any(name in sys.modules for name in _FORK_UNSAFE_MODULES)


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX
    os.register_at_fork(after_in_child=_note_fork_in_child)


def forked_after_mlx_init() -> bool:
    """Whether this process was forked from one that had already loaded MLX or scikit-learn."""
    return _forked_after_init


def check_fork_safety() -> None:
    """Raise :class:`DeviceError` if MLX cannot be used in this (forked) process."""
    if _forked_after_init:
        raise DeviceError(
            "This process was forked from one that had already loaded MLX or "
            "scikit-learn, and MLX (Metal and the Objective-C runtime) cannot run in "
            "such a child — any MLX call would crash the process, on the CPU too. "
            "Start worker processes with the 'spawn' or 'forkserver' method "
            "(multiprocessing.get_context('spawn'); joblib's default loky backend "
            "already does), or import mlxlearn and scikit-learn only after the fork."
        )


# ----------------------------------------------------------------------------- probing


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
    a tiny kernel and forces evaluation. The result is cached for the process. In
    a child forked after MLX was loaded the parent's cached answer is inherited;
    without one, the probe itself would crash, so it is not run and the answer is
    ``False``.
    """
    if _forked_after_init:
        return False
    try:
        import mlx.core as mx
    except ImportError:
        return False
    try:
        # Explicit streams, not `with mx.stream(mx.gpu)`: the probe must not
        # touch the process-global default device.
        probe = mx.array([1.0, 2.0], dtype=mx.float32)
        result = mx.sum(mx.multiply(probe, probe, stream=mx.gpu), stream=mx.gpu)
        mx.eval(result)
        return float(result.item()) == 5.0
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
                "device='gpu' was requested but no usable MLX GPU device is available "
                "in this process. mlxlearn does not substitute the CPU for a device "
                "that was named explicitly; use device='auto' to let it run MLX on "
                "the CPU instead."
            )
        return "gpu"
    return "gpu" if gpu_available() else "cpu"


def inference_device(backend: str, device: str | None = None) -> str:
    """Where inference for a model fitted on ``backend`` runs: ``"gpu"`` or ``"cpu"``.

    The fitted backend is sticky, and this does not change that: a model fitted
    on the MLX GPU backend predicts on the GPU. The one exception is a GPU-fitted
    model loaded (unpickled) in a process with no usable GPU. MLX on the CPU runs
    the identical float32 code, so under ``device="auto"`` or ``"cpu"`` it runs
    there — a model file stays usable on a GPU-less host — while an explicit
    ``device="gpu"`` still refuses to substitute the CPU.

    Parameters
    ----------
    backend : {"mlx", "cpu"}
        The model's ``execution_backend_``.
    device : {"auto", "gpu", "cpu"}, optional
        Defaults to the current configuration's ``device``.

    Raises
    ------
    DeviceError
        For an MLX-fitted model when no GPU is usable and ``device="gpu"``.
    """
    if backend == "cpu":
        return "cpu"
    if backend != "mlx":
        raise ValueError(f"inference_device takes an MLX backend ('mlx' or 'cpu'); got {backend!r}")
    if gpu_available():
        return "gpu"
    requested = device if device is not None else get_config().device
    if requested == "gpu":
        raise DeviceError(
            "This model was fitted on the MLX GPU backend, device='gpu' is set, and no "
            "usable MLX GPU is available in this process. Set device='auto' or "
            "device='cpu' to run it on MLX's CPU backend — the same float32 "
            "computation — or refit it here."
        )
    return "cpu"


def device_info(device: str | None = None) -> DeviceInfo:
    """Describe the device a request would resolve to."""
    return DeviceInfo(
        kind=resolve_device(device),
        gpu_available=gpu_available(),
        mlx_version=_mlx_version(),
        physical_memory_bytes=physical_memory_bytes(),
    )


# --------------------------------------------------------------------- the device gate


class _DeviceGate:
    """Makes mlxlearn's blocks agree on MLX's process-global default device.

    Interim, until every MLX operation takes an explicit ``stream=``. Blocks that
    want the device already in force join it and run concurrently — the common
    all-GPU case costs one uncontended lock. A block that wants the other device
    waits until the current ones finish, and new arrivals for the current device
    queue behind it rather than starving it. The first block in switches the
    default only if it differs; the last one out restores it.

    A thread already inside a block that asks for the other device (nesting) gets
    it once its own blocks are the only ones left, exclusively, and the outer
    device is put back when the nested block exits. Two threads doing that at the
    same moment would wait on each other forever, so the second one is refused
    with a :class:`DeviceError` rather than deadlocking.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._mode: str | None = None  # device every active block runs on
        self._holders = 0  # active blocks, over all threads
        self._held: dict[int, int] = {}  # active blocks per thread
        self._owner: int | None = None  # thread holding an exclusive nested switch
        self._restore = None  # MLX default device to put back at the end, if changed
        self._waiting = {"gpu": 0, "cpu": 0}
        self._nested_waiter: int | None = None

    def acquire(self, kind: str, mx) -> tuple | None:
        me = threading.get_ident()
        target = mx.gpu if kind == "gpu" else mx.cpu
        with self._cond:
            mine = self._held.get(me, 0)
            if mine and self._mode == kind:
                self._held[me] = mine + 1
                self._holders += 1
                return None
            if mine:
                return self._switch_nested(me, mine, kind, target, mx)

            self._waiting[kind] += 1
            try:
                while not self._may_join(kind):
                    self._cond.wait()
            finally:
                self._waiting[kind] -= 1
            if self._holders == 0:
                current = mx.default_device()
                if current != target:
                    self._restore = current
                    mx.set_default_device(target)
                else:
                    self._restore = None
                self._mode = kind
            self._held[me] = 1
            self._holders += 1
            return None

    def _may_join(self, kind: str) -> bool:
        if self._holders == 0:
            return True
        if self._owner is not None or self._mode != kind:
            return False
        other = "cpu" if kind == "gpu" else "gpu"
        return self._waiting[other] == 0

    def _switch_nested(self, me: int, mine: int, kind: str, target, mx) -> tuple:
        if self._holders != mine:
            if self._nested_waiter is not None:
                raise DeviceError(
                    "Two threads each asked for a different MLX device from inside an "
                    "mlxlearn device block at the same time; granting either would "
                    "deadlock. This is a bug in mlxlearn — please report it."
                )
            self._nested_waiter = me
            self._waiting[kind] += 1
            try:
                while self._holders != mine:
                    self._cond.wait()
            finally:
                self._waiting[kind] -= 1
                self._nested_waiter = None
        undo = (self._mode, self._owner)
        self._owner = me
        self._mode = kind
        mx.set_default_device(target)
        self._held[me] = mine + 1
        self._holders += 1
        return undo

    def release(self, undo: tuple | None, mx) -> None:
        me = threading.get_ident()
        with self._cond:
            remaining = self._held[me] - 1
            if remaining:
                self._held[me] = remaining
            else:
                del self._held[me]
            self._holders -= 1
            if undo is not None:
                outer_mode, outer_owner = undo
                mx.set_default_device(mx.gpu if outer_mode == "gpu" else mx.cpu)
                self._mode = outer_mode
                self._owner = outer_owner
            elif self._holders == 0:
                if self._restore is not None:
                    mx.set_default_device(self._restore)
                self._restore = None
                self._mode = None
            self._cond.notify_all()


_gate = _DeviceGate()


@contextmanager
def use_device(device: str | None = None):
    """Run a block on the resolved MLX device.

    Yields the ``mx.Stream`` the block's operations run on — this thread's
    default stream for the resolved device. Passing it as ``stream=`` makes an
    operation independent of the process-global default device; operations that
    do not take it run there anyway for the duration of the block (see the module
    docstring for what "for the duration" guarantees across threads).

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlxlearn._common.device import use_device
    >>> with use_device("cpu") as stream:
    ...     stream.device == mx.cpu
    True
    """
    check_fork_safety()
    import mlx.core as mx

    kind = resolve_device(device)
    undo = _gate.acquire(kind, mx)
    try:
        yield mx.default_stream(mx.gpu if kind == "gpu" else mx.cpu)
    finally:
        _gate.release(undo, mx)


def reset_device_cache() -> None:
    """Forget the cached probe results. Tests only."""
    gpu_available.cache_clear()
    physical_memory_bytes.cache_clear()
    _mlx_version.cache_clear()
