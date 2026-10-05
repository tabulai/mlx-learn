"""mlxlearn's own Metal kernels: when one may run, and how one is built and launched.

``mx.fast.metal_kernel`` compiles a Metal source string at run time, so a pure-Python
wheel can carry a kernel MLX has no operation for. Every rule here was measured before
the first kernel shipped, on MLX 0.31.2 and 0.32.2 (Apple M4 Max), and each one
answers a way a kernel went wrong there:

* **Only for work resolved to the GPU, on MLX 0.32 or later.** A kernel launched
  without ``stream=`` ran on the GPU under ``use_device("cpu")`` on 0.32, silently,
  and raised on 0.31; so the decision is made from the device the caller resolved
  (:func:`kernels_enabled`), never from MLX's default device, and every launch names
  ``mx.gpu`` (:func:`launch`). MLX 0.31 keeps compiled libraries in an unlocked map
  keyed by name alone: one kernel fed inputs of two sizes from two threads crashed
  the process, and from one thread rebuilt its library on every call. It has no
  ``compile_options`` either. A gate closed on 0.31 cost less than a lock around
  every launch, and the MLX CPU device and 0.31 run MLX's built-in operations, as
  they always did.
* **``#pragma METAL fp contract(off)`` first in every source** (:data:`PRELUDE`). In
  Metal's default mode ``a*b + c`` written as one expression is fused into an fma,
  which rounds once where MLX's own operations round twice; with the pragma only an
  explicit ``fma()`` fuses, in every math mode. ``math_mode`` is pinned to
  ``"safe"``: the relaxed and fast modes fuse across statements too.
* **Precise math only.** On 0.32 the bare ``metal::exp``, ``sqrt``, ``rsqrt`` and
  ``log`` are the fast approximations (``exp`` matched ``mx.exp`` on 7% of
  elements); a kernel that needs one spells out ``metal::precise::``.
* **No input shorter than eight elements** (:func:`pad_min8`). MLX declares such an
  input ``constant`` rather than ``device``, which generates, and compiles, a second
  variant of the source and breaks a helper taking a ``device`` pointer. Run-time
  sizes travel in one int32 array of eight (:func:`int_params`), never as template
  values, which would compile a library per size.
* **ASCII sources with fixed names**, built once per static variant from a
  ``functools.lru_cache`` factory keyed by static parameters only — never by data,
  and with nothing in a name or a source that changes between processes. MLX keeps
  each library in memory for the process; the operating system's Metal shader cache
  keeps the compiled source across processes, as long as the source and name stay
  the same.

When a kernel compiles is each kernel's policy; see :mod:`._topk` and :mod:`._ovo`.
A library MLX builds holds a lock that stalls every other MLX GPU call in the process
while it builds, so nothing here builds ahead of need.
"""

from __future__ import annotations

import functools

from .._common import device as _device
from .._common.config import get_tuning

__all__ = [
    "MIN_MLX",
    "PRELUDE",
    "kernels_enabled",
    "metal_kernel",
    "launch",
    "pad_min8",
    "int_params",
    "mlx_version",
]

#: The MLX release custom kernels need. See the module docstring.
MIN_MLX = (0, 32)

#: First in every kernel's header. See the module docstring.
PRELUDE = "#pragma METAL fp contract(off)\n"

#: Every kernel's name starts with this, so it never collides with another library's.
NAME_PREFIX = "mlxlearn_"

#: The fewest elements MLX passes as a ``device`` buffer.
MIN_INPUT = 8


@functools.lru_cache(maxsize=1)
def mlx_version() -> tuple[int, int, int]:
    """MLX's ``(major, minor, patch)``, read from ``mx.__version__``.

    Development and local builds append to the patch (``0.32.2.dev20261001``); only
    the leading digits of each part count.
    """
    import mlx.core as mx

    parts = []
    for piece in str(getattr(mx, "__version__", "0")).split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    parts += [0] * (3 - len(parts))
    return (parts[0], parts[1], parts[2])


@functools.lru_cache(maxsize=1)
def _runtime_ready() -> bool:
    """MLX new enough, and a Metal device that executes work. Fixed for the process."""
    if mlx_version()[:2] < MIN_MLX:
        return False
    import mlx.core as mx

    try:
        if not mx.metal.is_available():
            return False
    except AttributeError:  # pragma: no cover - an MLX build without the metal module
        return False
    return _device.gpu_available()


def kernels_enabled(device: str) -> bool:
    """Whether a custom kernel may run for work the caller resolved to ``device``.

    ``device`` is the resolved ``"gpu"`` or ``"cpu"`` the caller passes its built-in
    path too — not MLX's default device, which a kernel would otherwise follow onto the
    GPU from a CPU block. ``Tuning.metal_kernels`` turns every kernel off.
    """
    return device == "gpu" and get_tuning().metal_kernels and _runtime_ready()


def _check_ascii(text: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - a bug in a kernel's source
        raise ValueError(f"a Metal kernel's source must be ASCII: {exc}") from None


def metal_kernel(name: str, *, inputs, outputs, source: str, header: str = ""):
    """``mx.fast.metal_kernel`` under this module's rules.

    The name gets :data:`NAME_PREFIX`, the header :data:`PRELUDE`, and the source is
    checked for ASCII; ``math_mode`` is pinned to ``"safe"``; inputs are made
    row-contiguous, so a row slice reaches the kernel with its offset and anything
    else as a copy. Call it from an ``lru_cache`` factory keyed by static variant
    parameters. Building the object compiles nothing: MLX compiles the library on its
    first launch.
    """
    import mlx.core as mx

    full_header = PRELUDE + header
    _check_ascii(full_header + source + name)
    return mx.fast.metal_kernel(
        name=NAME_PREFIX + name,
        input_names=list(inputs),
        output_names=list(outputs),
        source=source,
        header=full_header,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def launch(kernel, *, inputs, output_shapes, output_dtypes, grid, threadgroup):
    """Run ``kernel`` on the GPU's stream, whatever MLX's default device is.

    The caller has already checked :func:`kernels_enabled` for its resolved device;
    the explicit stream is what keeps a launch made from inside a CPU block — a bug —
    from running on the GPU silently.
    """
    import mlx.core as mx

    return kernel(
        inputs=list(inputs),
        output_shapes=list(output_shapes),
        output_dtypes=list(output_dtypes),
        grid=tuple(grid),
        threadgroup=tuple(threadgroup),
        stream=mx.gpu,
    )


def pad_min8(mx, a, fill):
    """The 1-D array ``a``, padded with ``fill`` to at least :data:`MIN_INPUT` elements."""
    n = int(a.shape[0])
    if n >= MIN_INPUT:
        return a
    return mx.concatenate([a, mx.full((MIN_INPUT - n,), fill, dtype=a.dtype)])


def int_params(mx, *values: int):
    """Run-time sizes as one int32 array of exactly :data:`MIN_INPUT` entries."""
    vals = [int(v) for v in values]
    if len(vals) > MIN_INPUT:  # pragma: no cover - a bug in a kernel's caller
        raise OverflowError(f"{len(vals)} parameters do not fit {MIN_INPUT}")
    for v in vals:
        if not -(2**31) <= v < 2**31:  # pragma: no cover - the callers' gates prevent it
            raise OverflowError(f"{v} does not fit int32")
    return mx.array(vals + [0] * (MIN_INPUT - len(vals)), dtype=mx.int32)
