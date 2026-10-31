"""Device-native, NumPy-free algorithms for mlx-learn.

Unlike the sklearn-compatible ``mlxlearn`` interface, this package accepts and
returns MLX arrays and never imports NumPy or scikit-learn. The binary logistic
model stores its intercept in centered coordinates, retaining numerical
precision for shifted features. Neighbor search shares its device core with
the sklearn-compatible interface. See :func:`fit_logistic` and
:func:`fit_neighbors` for supported inputs.
"""

from ._neighbors import DeviceNeighborIndex, fit_neighbors

__all__ = ["DeviceConvergenceWarning", "DeviceLogisticModel", "fit_logistic",
           "DeviceNeighborIndex", "fit_neighbors"]


def __getattr__(name):
    # A neighbor-only workload should not load the native logistic optimizer,
    # parse its source generator, or depend on its optional compiled extension.
    if name in {"DeviceConvergenceWarning", "DeviceLogisticModel", "fit_logistic"}:
        from . import _logistic

        return getattr(_logistic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
