"""mlxlearn — MLX-accelerated classical machine learning for Apple silicon.

mlxlearn is not officially associated with scikit-learn or PROBABL, nor with Apple.

Two layers. Import estimators directly for strict, explicit behavior::

    from mlxlearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)

or patch scikit-learn so existing code is accelerated unchanged::

    from mlxlearn import patch_sklearn
    patch_sklearn()

    from sklearn.svm import SVC   # now resolves to the accelerated class

Patch before importing the symbols you want accelerated. Patching rebinds
attributes on scikit-learn's modules, so it cannot reach names that were already
bound into your own namespace — that is Python name binding, not a limitation
mlxlearn can engineer around.

The estimator subpackages are also reachable as attributes, as scikit-learn's are
(``import mlxlearn; mlxlearn.svm.SVC``). They load on first access, so
``import mlxlearn`` itself imports neither scikit-learn nor MLX.
"""

from __future__ import annotations

import importlib

from ._common.config import (
    Config,
    config_context,
    get_config,
    reset_config,
    set_config,
)
from ._common.device import device_info, gpu_available
from ._common.diagnostics import (
    BackendEvent,
    clear_backend_diagnostics,
    diagnostics_summary,
    get_backend_diagnostics,
    get_last_backend_event,
    reset_fallback_warnings,
)
from .exceptions import (
    BackendExecutionError,
    CapabilityError,
    DeviceError,
    InvalidInputError,
    MLXLearnError,
    MLXLearnFallbackWarning,
    MLXLearnPatchWarning,
    MLXLearnPerformanceWarning,
    SparseInputError,
    UnsupportedInputError,
    UnsupportedParameterError,
)
from .patching import (
    ESTIMATOR_REGISTRY,
    get_patch_map,
    is_patched,
    patch_sklearn,
    unpatch_sklearn,
)

__version__ = "0.1.0a2"

__all__ = [
    # patching
    "patch_sklearn",
    "unpatch_sklearn",
    "is_patched",
    "get_patch_map",
    "ESTIMATOR_REGISTRY",
    # configuration
    "set_config",
    "get_config",
    "config_context",
    "reset_config",
    "Config",
    # diagnostics
    "get_backend_diagnostics",
    "get_last_backend_event",
    "clear_backend_diagnostics",
    "reset_fallback_warnings",
    "diagnostics_summary",
    "BackendEvent",
    # environment
    "device_info",
    "gpu_available",
    # errors
    "MLXLearnError",
    "CapabilityError",
    "UnsupportedParameterError",
    "UnsupportedInputError",
    "SparseInputError",
    "InvalidInputError",
    "DeviceError",
    "BackendExecutionError",
    "MLXLearnFallbackWarning",
    "MLXLearnPerformanceWarning",
    "MLXLearnPatchWarning",
    "__version__",
]


#: Subpackages resolved on first attribute access (PEP 562). Loading one imports
#: scikit-learn, so none is loaded by ``import mlxlearn`` itself.
_LAZY_SUBMODULES = ("linear_model", "neighbors", "svm")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        # import_module binds the subpackage as an attribute of this module, so
        # the hook runs once per name.
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*__all__, *_LAZY_SUBMODULES, "exceptions"})
