"""Input validation, built only on scikit-learn's public API.

The ancestor reached into ``sklearn._*`` in five places. mlxlearn does not: every
helper here comes from ``sklearn.utils.validation``, ``sklearn.utils.multiclass``,
``sklearn.utils.class_weight`` or ``sklearn.base``, all of which are documented
public API. ``tools/compliance.py`` fails CI if a private import reappears outside
the quarantined ``patching/`` package.

Two responsibilities, kept apart:

*Validation* answers "is this input valid at all?" and raises the same errors
scikit-learn would, because code written against scikit-learn's error contract has
to keep working.

*Capability checking* answers "can the MLX path serve this valid input?" and
raises :class:`~mlxlearn.exceptions.CapabilityError`, which is the family patched
mode is allowed to fall back on. Conflating the two is how a genuine user error
ends up silently producing an answer from a different code path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets, type_of_target, unique_labels
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_random_state,
    validate_data,
)

from ..exceptions import SparseInputError, UnsupportedInputError, UnsupportedParameterError
from .arrays import NUMPY_FLOAT

__all__ = [
    "check_array",
    "check_consistent_length",
    "check_is_fitted",
    "check_random_state",
    "validate_data",
    "check_classification_targets",
    "type_of_target",
    "unique_labels",
    "compute_class_weight",
    "is_sparse",
    "require_dense",
    "SparseInputError",
    "require_supported_params",
    "check_sample_weight",
    "check_finite_float32",
    "class_weight_vector",
]


def is_sparse(X: Any) -> bool:
    """Whether ``X`` is a SciPy sparse matrix or array, without importing SciPy eagerly."""
    module = type(X).__module__
    if not module.startswith("scipy.sparse"):
        return False
    from scipy import sparse

    return sparse.issparse(X)


def require_dense(X: Any, *, estimator: str, operation: str = "fit") -> None:
    """Raise a capability error for sparse input.

    mlxlearn's kernels are dense. That is a real limitation, and naming it as one
    is what lets patched mode fall back correctly instead of guessing.
    """
    if is_sparse(X):
        raise SparseInputError(
            f"{estimator}.{operation} received sparse input, but mlxlearn's MLX "
            "kernels are dense. Convert with X.toarray() to use mlxlearn, or run "
            "under patch_sklearn() to have sparse input handled by scikit-learn "
            "automatically."
        )


def require_supported_params(
    params: dict[str, Any],
    supported: dict[str, tuple[Any, ...]],
    *,
    estimator: str,
) -> None:
    """Check parameter values against what the MLX path implements.

    Parameters
    ----------
    params : dict
        The estimator's current parameter values.
    supported : dict
        Maps parameter name to the tuple of values the MLX path implements. A
        parameter absent from this mapping is not checked here.
    estimator : str
        Class name, used in the message and in the diagnostics slug.

    Raises
    ------
    UnsupportedParameterError
        Naming the parameter, the offending value, and what is supported.
    """
    for name, allowed in supported.items():
        if name not in params:
            continue
        value = params[name]
        if value not in allowed:
            options = ", ".join(repr(a) for a in allowed)
            raise UnsupportedParameterError(
                f"{estimator} does not implement {name}={value!r} on the MLX path. "
                f"Supported values are: {options}.",
                parameter=name,
                value=value,
            )


def check_sample_weight(
    sample_weight: Any,
    X: np.ndarray,
    *,
    dtype: Any = NUMPY_FLOAT,
) -> np.ndarray | None:
    """Validate ``sample_weight`` against ``X``, returning ``None`` when unweighted.

    scikit-learn's own helper for this is private, so this reimplements the parts
    of its contract mlxlearn relies on: scalars broadcast, 1-D arrays must match
    ``n_samples``, weights must be finite and non-negative.

    Returning ``None`` for the unweighted case — rather than an array of ones —
    is deliberate: solvers can then skip the weighted code path entirely instead
    of multiplying by ones and losing exactness.
    """
    if sample_weight is None:
        return None

    n_samples = X.shape[0]
    if np.isscalar(sample_weight):
        value = float(sample_weight)
        if value == 1.0:
            return None
        weights = np.full(n_samples, value, dtype=dtype)
    else:
        weights = np.asarray(sample_weight, dtype=dtype)
        if weights.ndim == 0:
            weights = np.full(n_samples, float(weights), dtype=dtype)
        elif weights.ndim != 1:
            raise ValueError(
                f"sample_weight must be 1-dimensional, got shape {weights.shape}."
            )
        if weights.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight has {weights.shape[0]} elements but X has "
                f"{n_samples} samples."
            )

    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight contains NaN or infinity.")
    if np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative.")
    return weights


def check_finite_float32(X: np.ndarray, *, name: str = "X") -> None:
    """Reject values that survive float64 but not float32.

    ``1e300`` is a perfectly ordinary float64 and becomes ``inf`` in float32. If
    that conversion happened silently, the result would be garbage attributable to
    nothing, so mlxlearn refuses the input and says which value caused it.
    """
    if X.dtype != NUMPY_FLOAT:
        return
    if not np.all(np.isfinite(X)):
        raise UnsupportedInputError(
            f"{name} contains values that are not finite in float32. mlxlearn "
            "computes in float32, and float64 values with magnitude beyond "
            "~3.4e38 overflow to infinity in the conversion.",
            reason="float32-overflow",
        )


def class_weight_vector(
    class_weight: Any,
    classes: np.ndarray,
    y: np.ndarray,
) -> np.ndarray | None:
    """Resolve ``class_weight`` to a per-class weight vector aligned with ``classes``.

    Returns ``None`` when every weight is 1, so the caller can skip the weighted
    path entirely.
    """
    if class_weight is None:
        return None
    weights = compute_class_weight(class_weight, classes=classes, y=y)
    weights = np.asarray(weights, dtype=np.float64)
    if np.allclose(weights, 1.0):
        return None
    return weights
