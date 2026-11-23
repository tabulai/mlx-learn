"""Patched-layer delegation of whole radius calls to stock scikit-learn.

A patched estimator hands a call to scikit-learn when the call is outside the
owned engines' capability (sparse queries, non-scalar or negative radii,
values beyond float32, callable weights), or when the state-free router
predicts that scikit-learn is faster and can preserve exact membership, or
when stock is explicitly requested. The stock estimator mirrors the user's
configuration (algorithm, leaf_size, n_jobs, weights, outlier_label, ...),
is fitted once on the float64 image of the owned float32 training rows, and
its outputs are returned unchanged.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

__all__ = ["capability_reason", "stock_model"]


def capability_reason(estimator, X, radius) -> str | None:
    """Why a patched call cannot run on the owned engines, or ``None``."""
    from scipy import sparse

    if radius is None:
        return "radius is None"
    if np.ndim(radius) != 0:
        return "per-query radii"
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (Real, np.number)):
        return "non-numeric radius"
    value = float(radius)
    if math.isnan(value) or value < 0:
        return "negative or NaN radius"
    if X is not None and sparse.issparse(X):
        return "sparse query"
    weights = getattr(estimator, "weights", "uniform")
    if callable(weights):
        return "callable weights"
    if getattr(estimator, "metric_params", None):
        return "metric_params"
    return None


#: Parameters scikit-learn reads at query time; the rest take effect at fit.
_QUERY_TIME_PARAMS = ("radius", "weights", "n_jobs")


def stock_model(estimator):
    """The fitted stock twin of a patched, MLX-fitted radius estimator.

    Built once per fit from the fit-time parameters, on the float64 image of
    the owned float32 training rows, with the fitted feature names; query-time
    parameters (radius, weights, n_jobs) follow the estimator's current values.
    """
    from ._base import _stock_neighbor_class

    state = estimator._knn_query_state_
    model = state.get("stock_radius")
    current = estimator.get_params(deep=False)
    if model is None:
        params = dict(estimator.__dict__.get("_radius_fit_params_") or current)
        model = _stock_neighbor_class(estimator._sklearn_class_name)(**params)
        X = np.asarray(estimator._index_.training_data, dtype=np.float64)
        if hasattr(estimator, "classes_"):
            model.fit(X, estimator.classes_[estimator._y])
        elif estimator._sklearn_class_name == "RadiusNeighborsRegressor":
            y = np.asarray(estimator._y, dtype=estimator.__dict__.get("_y_out_dtype_", np.float64))
            if estimator.__dict__.get("_y_2d_", False):
                y = y.reshape(-1, 1)
            model.fit(X, y)
        else:
            model.fit(X)
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            model.feature_names_in_ = names
        state["stock_radius"] = model
    updates = {key: current[key] for key in _QUERY_TIME_PARAMS
               if key in current and getattr(model, key, _MISSING) is not current[key]
               and getattr(model, key, _MISSING) != current[key]}
    if updates:
        model.set_params(**updates)
    return model


_MISSING = object()


def stock_query(X, reason: str):
    """Round a delegated query to float32 like the owned engines would.

    Values beyond float32 (the reason for that delegation) are passed through.
    Containers keep their type (and feature names) where possible.
    """
    if X is None or reason == "query values outside float32":
        return X
    from scipy import sparse

    try:
        if sparse.issparse(X):
            rounded = X.astype(np.float32)
            if rounded.nnz and not np.all(np.isfinite(rounded.data)):
                return X
            return rounded.astype(np.float64)
        if hasattr(X, "columns") and hasattr(X, "astype"):  # DataFrame-like
            values = np.asarray(X, dtype=np.float64)
            if not np.all(np.abs(values[np.isfinite(values)]) <= np.finfo(np.float32).max):
                return X
            return X.astype(np.float32).astype(np.float64)
        values = np.asarray(X)
        if values.dtype.kind not in "fiu":
            return X
        values = values.astype(np.float64)
        finite = np.isfinite(values)
        if not np.all(np.abs(values[finite]) <= np.finfo(np.float32).max):
            return X
        return values.astype(np.float32).astype(np.float64)
    except (TypeError, ValueError):
        return X
