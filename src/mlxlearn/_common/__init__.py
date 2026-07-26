"""Shared runtime: configuration, device, arrays, validation, RNG, diagnostics, dispatch.

Private. Nothing here is public API; import from ``mlxlearn`` instead. The one
exception in spirit is ``arrays``, which is written to public-safe standards
because the 0.2.x experimental MLX boundary will expose it.
"""

from __future__ import annotations

from .arrays import INDEX_DTYPE, NUMPY_FLOAT, as_float32, sync, to_mlx, to_numpy
from .base import BackendMixin, mlx_guard, problem_work
from .config import (
    Config,
    Tuning,
    config_context,
    get_config,
    get_tuning,
    reset_config,
    set_config,
    tuning_context,
)
from .device import DeviceInfo, device_info, gpu_available, resolve_device, use_device
from .diagnostics import (
    BackendEvent,
    clear_backend_diagnostics,
    diagnostics_summary,
    get_backend_diagnostics,
    get_last_backend_event,
    record_event,
)
from .rng import derive_seed, mlx_key, numpy_generator, resolve_seed

__all__ = [
    "INDEX_DTYPE",
    "NUMPY_FLOAT",
    "BackendEvent",
    "BackendMixin",
    "Config",
    "DeviceInfo",
    "Tuning",
    "as_float32",
    "clear_backend_diagnostics",
    "config_context",
    "derive_seed",
    "device_info",
    "diagnostics_summary",
    "get_backend_diagnostics",
    "get_config",
    "get_last_backend_event",
    "get_tuning",
    "gpu_available",
    "mlx_guard",
    "mlx_key",
    "numpy_generator",
    "problem_work",
    "record_event",
    "reset_config",
    "resolve_device",
    "resolve_seed",
    "set_config",
    "sync",
    "to_mlx",
    "to_numpy",
    "tuning_context",
    "use_device",
]
