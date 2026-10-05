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

The capability checks run on input *as the caller passed it* — before
``validate_data`` — because that is the only window in which patched mode can still
fall back. So they have to read anything scikit-learn reads: NumPy arrays, lists,
pandas frames of mixed dtypes, ``mx.array``, torch tensors. When a check cannot read
something as numbers, it says nothing and leaves ``validate_data`` to reject it with
scikit-learn's own error; a capability check never invents a validation error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets, type_of_target, unique_labels
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_random_state,
    validate_data,
)
from sklearn.utils.validation import (
    check_is_fitted as _sklearn_check_is_fitted,
)

from ..exceptions import SparseInputError, UnsupportedInputError, UnsupportedParameterError
from .arrays import (
    _FLOAT64_FIT_ADVICE,
    FLOAT32_MAX,
    FLOAT32_TINY,
    NUMPY_FLOAT,
    ValueRange,
    check_squared_float32_range,
    finite_range,
)

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
    "require_nonnegative_weights",
    "check_sample_weight",
    "count_labels",
    "check_float32_range",
    "check_squared_float32_range",
    "finite_range",
    "validate_float32_input",
    "ValueRange",
    "FLOAT32_MAX",
    "FLOAT32_TINY",
]


def check_is_fitted(estimator: Any, *args: Any, **kwargs: Any) -> None:
    """scikit-learn's ``check_is_fitted``, answered at once for an mlxlearn-fitted estimator.

    An estimator whose fit completed carries ``_execution_backend_``
    (:meth:`~mlxlearn._common.base.BackendMixin.__sklearn_is_fitted__`), and
    scikit-learn's check would read that same fact after building the estimator's
    tags — a few microseconds on every inference call, twice for a ``predict`` that
    reaches ``decision_function``, which on a 0.2 ms single-row call was a
    measurable share of it. Everything else — unfitted, mid-fit, state assigned
    from outside, explicit ``attributes`` — goes to scikit-learn's check unchanged.
    """
    if not args and not kwargs and "_execution_backend_" in getattr(estimator, "__dict__", ()):
        return
    _sklearn_check_is_fitted(estimator, *args, **kwargs)


def _is_pandas(X: Any) -> bool:
    return type(X).__module__.startswith("pandas")


def _safe_shape(X: Any) -> tuple[int, int] | None:
    """``(n_samples, n_features)`` of input as the caller passed it, or ``None``.

    ``None`` for anything without a two-dimensional ``shape`` — a list, say —
    whose size is known only once it has been validated.
    """
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def is_sparse(X: Any) -> bool:
    """Whether ``X`` is sparse input, without importing SciPy or pandas eagerly.

    Two kinds count: a SciPy sparse matrix or array, and a pandas ``DataFrame``
    whose *every* column has a ``SparseDtype``. The pandas rule is scikit-learn's
    own — ``check_array`` converts a frame to COO only when all of its columns are
    sparse, and densifies a partly sparse one with a warning. Calling a partly
    sparse frame sparse here would hand a frame the MLX path serves correctly to a
    fallback.

    The type's module is checked first, so an ``ndarray`` costs one string
    comparison and neither library is imported to answer for it.
    """
    module = type(X).__module__
    if module.startswith("scipy.sparse"):
        from scipy import sparse

        return sparse.issparse(X)
    if module.startswith("pandas") and getattr(X, "ndim", 0) == 2:
        # X is a pandas object, so pandas is already imported: this costs nothing.
        from pandas import SparseDtype

        dtypes = X.dtypes
        return len(dtypes) > 0 and all(isinstance(dtype, SparseDtype) for dtype in dtypes)
    return False


def require_dense(X: Any, *, estimator: str, operation: str = "fit") -> None:
    """Raise a capability error for sparse input.

    mlxlearn's kernels are dense. That is a real limitation, and naming it as one
    is what lets patched mode fall back correctly instead of guessing.
    """
    if is_sparse(X):
        convert = "X.sparse.to_dense()" if _is_pandas(X) else "X.toarray()"
        raise SparseInputError(
            f"{estimator}.{operation} received sparse input, but mlxlearn's MLX "
            f"kernels are dense. Convert with {convert} to use mlxlearn.",
            advice="Under patch_sklearn(), sparse input is handed to scikit-learn "
            "automatically.",
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


def _negative_weight_error(estimator: str | None) -> UnsupportedInputError:
    # A capability error, not a plain ValueError. scikit-learn accepts negative
    # sample weights; mlxlearn's logistic solver does not, because a negative weight
    # makes the objective non-convex and the convergence guarantees stop holding.
    # Since scikit-learn *can* serve the request, patched mode must be able to fall
    # back to it — which it can only do if this is in the CapabilityError family,
    # raised inside the capability check (see require_nonnegative_weights).
    where = f"{estimator}: " if estimator else ""
    return UnsupportedInputError(
        f"{where}sample_weight contains negative values. mlxlearn's solvers require "
        "non-negative weights, because a negative weight makes the objective "
        "non-convex.",
        reason="negative-sample-weight",
        advice="scikit-learn accepts them, so under patch_sklearn() this falls back "
        "rather than failing.",
    )


def require_nonnegative_weights(sample_weight: Any, *, estimator: str) -> None:
    """Raise :class:`UnsupportedInputError` for a negative ``sample_weight``, before validation.

    :func:`check_sample_weight` reports the same thing, but it needs the validated
    ``X`` and so runs after backend selection — where a capability error can no
    longer fall back. This is the half that belongs in ``capability_check``: it
    looks only at the weights, and it leaves anything it cannot read as numbers for
    :func:`check_sample_weight` to reject with a proper error afterwards.
    """
    if sample_weight is None:
        return
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return
    if np.any(weights < 0):
        raise _negative_weight_error(estimator)


def check_sample_weight(
    sample_weight: Any,
    X: np.ndarray,
    *,
    dtype: Any = NUMPY_FLOAT,
) -> np.ndarray | None:
    """Validate ``sample_weight`` against ``X``, returning ``None`` when unweighted.

    scikit-learn's own helper for this is private, so this reimplements the parts
    of its contract mlxlearn relies on: scalars broadcast, 1-D arrays must match
    ``n_samples``, weights must be finite and non-negative. A negative weight is
    refused before this runs, where it can still fall back
    (:func:`require_nonnegative_weights`), or given a meaning first — SVC clips it
    to zero, which is what LIBSVM makes of it.

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
        raise _negative_weight_error(None)
    return weights


# ------------------------------------------------------------------ class counts


def count_labels(y: Any) -> int | None:
    """How many distinct labels ``y`` holds, read cheaply from the raw target; or ``None``.

    A size hint for dispatch, taken before validation, so it must neither fail nor
    cost what validation costs. Two vectorized passes find the common case, two
    labels, with no sort: every label either equals the first one or, failing that,
    the first one that differs. Only a target with more labels than that is counted
    in full — a sort for numbers and fixed-width strings, a hash set for Python
    objects, which is what a pandas ``Series`` of strings or categories arrives as.
    Counting object labels as binary, as the dispatch rules once did, made where a
    fit ran depend on whether its labels were integers or strings.

    ``None`` when the labels cannot be read that way — not a single column, not
    hashable, not comparable — leaving the decision to the caller and the error, if
    there is one, to scikit-learn's validation.
    """
    try:
        labels = np.asarray(y)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.ravel()
        if labels.ndim != 1 or labels.dtype.kind not in "biufUSO":
            return None
        if labels.size == 0:
            return 0
        other = np.asarray(labels != labels[0], dtype=bool)
        rest = np.flatnonzero(other)
        if rest.size == 0:
            return 1
        more = np.asarray(labels != labels[rest[0]], dtype=bool)
        if not np.any(other & more):
            return 2
        if labels.dtype.kind == "O":
            return len(set(labels.tolist()))
        return int(np.unique(labels).shape[0])
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ float32 range


def _numeric_kind(dtype: Any) -> tuple[str, int]:
    """``(kind, itemsize)`` of a NumPy dtype, or of a pandas masked dtype's NumPy twin.

    ``("O", 0)`` for anything else — pandas strings, categoricals, sparse columns —
    which the caller then tries to read as float64.
    """
    numpy_dtype = dtype if isinstance(dtype, np.dtype) else getattr(dtype, "numpy_dtype", None)
    if isinstance(numpy_dtype, np.dtype):
        return numpy_dtype.kind, numpy_dtype.itemsize
    return "O", 0


def _cannot_overflow(kind: str, itemsize: int) -> bool:
    """Integer, boolean and float32-or-narrower values cannot reach float32's limit."""
    return kind in "biu" or (kind == "f" and itemsize <= 4)


def _array_range(array: np.ndarray, *, need_range: bool) -> ValueRange | None:
    kind, itemsize = _numeric_kind(array.dtype)
    if kind == "c":
        return None  # complex input: scikit-learn rejects it itself
    if kind in "biuf":
        if _cannot_overflow(kind, itemsize) and not need_range:
            return None
        return finite_range(array)
    # Object, string or record data: numbers NumPy could not unify, such as a
    # float64 column mixed with a bool one. validate_data reads them as floats, so
    # the check does too, and anything that does not convert is left for
    # validate_data to reject with scikit-learn's error.
    try:
        return finite_range(array.astype(np.float64))
    except (TypeError, ValueError):
        return None


def _frame_range(frame: Any, *, need_range: bool) -> ValueRange | None:
    # A frame mixing dtypes is an object array to np.asarray, built only to be
    # scanned and thrown away: ~140 ms for 200k x 51 float64 + bool. Reading the
    # columns by dtype instead costs a few milliseconds and never goes near object.
    positions_by_dtype: dict[Any, list[int]] = {}
    for position, dtype in enumerate(frame.dtypes):
        positions_by_dtype.setdefault(dtype, []).append(position)

    found = []
    for dtype, positions in positions_by_dtype.items():
        kind, itemsize = _numeric_kind(dtype)
        if kind == "c" or (_cannot_overflow(kind, itemsize) and not need_range):
            continue
        if isinstance(dtype, np.dtype) and kind in "biuf":
            # All columns of one NumPy dtype: a single block, read in one pass.
            block = frame if len(positions) == frame.shape[1] else frame.iloc[:, positions]
            found.append(finite_range(np.asarray(block)))
            continue
        # Nullable, string, categorical, sparse or object columns: one at a time,
        # with missing values read as NaN the way validate_data reads them. A
        # column that is not numbers is validate_data's to reject.
        for position in positions:
            try:
                values = frame.iloc[:, position].to_numpy(dtype=np.float64, na_value=np.nan)
            except (TypeError, ValueError):
                continue
            found.append(finite_range(values))

    found = [r for r in found if r is not None]
    if not found:
        return None
    return ValueRange(min(r.low for r in found), max(r.high for r in found))


def _input_range(X: Any, *, need_range: bool) -> ValueRange | None:
    if _is_pandas(X) and getattr(X, "ndim", 0) == 2:
        return _frame_range(X, need_range=need_range)
    # Everything else is read the way validate_data is about to read it. For an
    # ndarray that is the array itself; for an mx.array or a CPU torch tensor it is
    # a zero-copy view. Their own dtypes are not NumPy dtypes — mlx.core.Dtype has
    # no .kind — so reading .dtype off the raw object would crash here, outside the
    # CapabilityError family, on input scikit-learn accepts.
    try:
        array = np.asarray(X)
    except (TypeError, ValueError):
        return None  # not an array at all: validate_data raises scikit-learn's error
    return _array_range(array, need_range=need_range)


def check_float32_range(
    X: Any, *, name: str = "X", need_range: bool = False, fallback_advice: bool = True
) -> ValueRange | None:
    """Reject values that are finite as given but would become infinite in float32.

    ``1e300`` is a perfectly ordinary float64 and is ``inf`` in float32. Converting
    it silently would produce garbage attributable to nothing.

    This must run **before** ``validate_data``, because ``validate_data(dtype=float32)``
    performs the conversion itself and then raises scikit-learn's generic "contains
    infinity or a value too large" error. That error is not wrong, but it is a plain
    ``ValueError``, so patched mode could not fall back — even though scikit-learn's
    float64 path would handle the input perfectly well. Raising a
    :class:`~mlxlearn.exceptions.UnsupportedInputError` here makes the fallback work
    and explains *why* mlxlearn specifically cannot take the input.

    What gets scanned is decided by what can overflow. Float64 (and wider) data is,
    and so is anything not already numeric — an object array, or a pandas frame
    mixing float64 with bool or nullable columns, which ``np.asarray`` turns into an
    object array — because a float64 hidden in an object array overflows just the
    same. Float32, integer and boolean data cannot reach ``3.4e38`` and is skipped
    unless ``need_range`` asks for its range anyway. The scan is two reductions
    with no temporaries on finite data; see :func:`finite_range`.

    Parameters
    ----------
    X : array-like
        Dense input as the caller passed it; run :func:`require_dense` first.
    name : str, default="X"
        What to call the data in the message.
    need_range : bool, default=False
        Scan data that cannot overflow too, so the range is always returned when
        there is one. For callers that go on to :func:`check_squared_float32_range`.
    fallback_advice : bool, default=True
        Advise running under ``patch_sklearn()``, whose fallback to scikit-learn's
        float64 path exists only at ``fit``. Inference cannot fall back — the
        fitted backend is sticky, in both layers — so it passes ``False``.

    Returns
    -------
    ValueRange or None
        The smallest and largest finite values, when the data was scanned and has
        any; otherwise ``None``.

    Raises
    ------
    UnsupportedInputError
        With ``reason="float32-overflow"``.
    """
    value_range = _input_range(X, need_range=need_range)
    if value_range is not None and value_range.magnitude > FLOAT32_MAX:
        raise UnsupportedInputError(
            f"{name} contains a value of magnitude {value_range.magnitude:.3g}, which "
            f"is finite in float64 but infinite in float32 (the limit is "
            f"{FLOAT32_MAX:.3g}). mlxlearn computes in float32 and will not silently "
            "convert it to infinity. Rescale the data.",
            reason="float32-overflow",
            advice=_FLOAT64_FIT_ADVICE if fallback_advice else "",
        )
    return value_range


def validate_float32_input(
    estimator: Any,
    X: Any,
    y: Any = "no_validation",
    *,
    operation: str = "predict",
    dtype: Any = NUMPY_FLOAT,
    reset: bool | None = None,
    **check_params: Any,
):
    """``validate_data`` for an MLX path, with mlxlearn's input capability checks in front.

    Every path that computes in float32 needs the same three steps, in this order:

    1. :func:`require_dense` — a :class:`~mlxlearn.exceptions.SparseInputError`
       naming the operation, rather than scikit-learn's generic sparse error.
    2. :func:`check_float32_range` — an
       :class:`~mlxlearn.exceptions.UnsupportedInputError` for a value float32
       cannot hold, rather than scikit-learn's "contains infinity" or, worse, a
       silent cast to infinity.
    3. ``validate_data(estimator, X, y, accept_sparse=False, dtype=dtype, ...)``.

    With a ``dtype`` wider than float32 — SVC validates in float64 and casts later
    — step 2 runs on the *validated* array instead: scikit-learn has already done
    the conversion, so the check costs one min/max pass rather than a second
    conversion, and it still fires before the cast would turn the value into
    infinity. A tuple of dtypes, as ``validate_data`` takes it, counts as wide
    when any member is: the validated array then keeps float32 input as float32,
    which the range check skips, and scans only what came out wider.

    At fit time the same checks belong inside ``_select_backend``, the only place a
    capability error can still fall back. At inference the backend is sticky, so
    they are simply raised — an MLX-fitted model has no scikit-learn state to hand
    ``predict`` to.

    Parameters
    ----------
    estimator : estimator instance
        Passed to ``validate_data``; its class name is used in messages.
    X : array-like
    y : array-like, default="no_validation"
        As for ``validate_data``: when given, ``(X, y)`` is returned.
    operation : str, default="predict"
        Named in the sparse-input message.
    dtype : NumPy floating dtype or tuple of them, default=float32
    reset : bool, optional
        Passed to ``validate_data``. Defaults to ``operation == "fit"``, so an
        inference call cannot reset ``n_features_in_`` by forgetting to say so.
    **check_params
        Passed through to ``validate_data``.

    Returns
    -------
    X or (X, y)
        Whatever ``validate_data`` returns.
    """
    if reset is None:
        reset = operation == "fit"
    dtypes = dtype if isinstance(dtype, (tuple, list)) else (dtype,)
    wide = any(np.dtype(d).itemsize > np.dtype(NUMPY_FLOAT).itemsize for d in dtypes)
    advice = operation == "fit"  # only a fit has a scikit-learn path to fall back to
    require_dense(X, estimator=type(estimator).__name__, operation=operation)
    if not wide:
        check_float32_range(X, fallback_advice=advice)
    validated = validate_data(
        estimator, X, y, accept_sparse=False, dtype=dtype, reset=reset, **check_params
    )
    if wide:
        check_float32_range(
            validated[0] if isinstance(validated, tuple) else validated,
            fallback_advice=advice,
        )
    return validated
