"""Ordinary least squares on the host, usually faster than scikit-learn's ``gelsd``.

Subclassing :class:`sklearn.linear_model.LinearRegression` is what makes the
drop-in promise real: the constructor signature, ``get_params``/``set_params``,
the parameter constraints and the estimator tags are all inherited, so ``clone``,
``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back
to.

mlxlearn's path covers dense ordinary least squares, including multi-output
targets and sample weights, with scikit-learn's rank cutoff for the installed
version. ``positive=True`` is a capability mismatch — non-negative least squares
is a different solver — and raises rather than being quietly ignored.
"""

from __future__ import annotations

import numbers

import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import LinearRegression as _SkLinearRegression
from sklearn.utils.validation import check_array, check_non_negative

from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_config, get_tuning
from .._common.diagnostics import record_event
from .._common.validation import check_is_fitted, require_dense, validate_data
from .._solvers.ols import OLSConfig, solve_ols
from ..exceptions import UnsupportedParameterError

__all__ = ["LinearRegression"]

_SKLEARN = tuple(int(part) for part in sklearn.__version__.split(".")[:2])

#: scikit-learn 1.9 passes ``tol`` to ``lstsq`` as the dense rank cutoff; 1.7 and
#: 1.8 use ``max(n_samples, n_features) * eps`` of the working dtype instead.
_DENSE_CUTOFF_IS_TOL = _SKLEARN >= (1, 9)


def _sample_weight_as_stock(sample_weight, X) -> np.ndarray:
    """scikit-learn's ``_check_sample_weight(..., dtype=X.dtype, ensure_non_negative=True)``.

    Rebuilt from public API (the project allows no private scikit-learn imports
    outside ``patching/``), with each version's dtype rule and messages: float64
    weights unless X is float32 or float64; the short 1-D message on 1.7; the
    all-zero check from 1.9, before the sign check.
    """
    n = X.shape[0]
    dtype = X.dtype if X.dtype in (np.float32, np.float64) else np.float64
    if isinstance(sample_weight, numbers.Number):
        weights = np.full(n, sample_weight, dtype=dtype)
    else:
        weights = check_array(sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
                              order="C", input_name="sample_weight")
        if weights.ndim != 1:
            if _SKLEARN >= (1, 8):
                raise ValueError(
                    f"Sample weights must be 1D array or scalar, got {weights.ndim}D array. "
                    f"Expected either a scalar value or a 1D array of length {n}."
                )
            raise ValueError("Sample weights must be 1D array or scalar")
        if weights.shape != (n,):
            raise ValueError(f"sample_weight.shape == {weights.shape}, expected {(n,)}!")
    if _SKLEARN >= (1, 9) and np.all(weights == 0):
        raise ValueError("Sample weights must contain at least one non-zero number.")
    check_non_negative(weights, "`sample_weight`")
    return weights


class LinearRegression(BackendMixin, _SkLinearRegression):
    """Ordinary least squares, fitted by mlxlearn's host solver.

    A drop-in replacement for :class:`sklearn.linear_model.LinearRegression` on
    dense data. Multi-output targets and ``sample_weight`` are implemented;
    ``positive=True`` is not.

    Parameters
    ----------
    fit_intercept : bool, default=True
    copy_X : bool, default=True
        Accepted. ``X`` is read but never written: the centred working design is
        always a separate array, so the caller's array is not modified even when
        ``copy_X=False`` (scikit-learn centres ``X`` in place then).
    tol : float, default=1e-6
        On scikit-learn 1.9 and later, the relative cutoff on the singular values
        of the (centred, weighted) design, as scikit-learn passes it to
        ``lstsq``. On 1.7 and 1.8 the dense cutoff is scikit-learn's
        ``max(n_samples, n_features) * eps`` of the working dtype, as stock uses.
    n_jobs : int, default=None
        Accepted and ignored, as scikit-learn ignores it for dense data. The host
        solver threads its row passes itself.
    positive : bool, default=False
        Only ``False`` is implemented. ``True`` raises
        :class:`~mlxlearn.exceptions.UnsupportedParameterError` on direct use
        and falls back under ``patch_sklearn()``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        float64.
    intercept_ : float or ndarray of shape (n_targets,)
        float64. ``0.0`` when ``fit_intercept=False``, matching scikit-learn.
    n_features_in_ : int
    rank_ : int
        Effective rank of the (centred, weighted) design under scikit-learn's
        cutoff for the installed version.
    singular_ : ndarray of shape (min(n_samples, n_features),)
        Singular values of that design, descending, float64.
    execution_backend_ : {"cpu", "sklearn"}
        ``"cpu"`` whenever mlxlearn fitted the model. There is no Metal path for
        ordinary least squares; ``device="gpu"`` is honoured by solving on the
        host and recording a ``host-solve`` diagnostic.

    Notes
    -----
    **Precision and parity.** The accelerated paths compute in float64,
    including for float32 input (whose values, and ``y`` cast to float32 as
    scikit-learn casts it, are widened exactly). Rank decisions are certified
    with rigorous rounding bounds. A decision that falls inside scikit-learn's
    own rounding of the cutoff, a near-square design, a ``tol`` outside
    ``(0, 1)`` and a tiny problem all run scikit-learn's own arithmetic instead
    (in the working dtype, without its extra copies), so the result is stock's
    to the bit. ``rank_`` therefore always matches stock.

    **No caches.** ``fit`` is a pure function of ``(X, y, sample_weight, params)``.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.linear_model import LinearRegression
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([1.0, 3.0, 5.0, 7.0])
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.coef_.shape
    (1,)
    >>> reg.predict([[4.0]])
    array([9.])
    """

    #: Ordinary least squares is natively multi-output; mlxlearn's path is too.
    _mlxlearn_supports_multi_output = True

    def _check_linear_capability(self, X) -> None:
        """Raise :class:`CapabilityError` for anything mlxlearn's path cannot serve."""
        name = type(self).__name__
        require_dense(X, estimator=name)
        if bool(self.positive):
            raise UnsupportedParameterError(
                f"{name} implements positive=False only; got positive={self.positive!r}. "
                "Non-negative least squares is a different solver.",
                parameter="positive",
                value=self.positive,
            )

    def _below_crossover(self, n_samples: int, n_features: int, n_targets: int = 1) -> tuple[bool, str]:
        """Apply measured size floors to automatic fits."""
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < n_features:
            below = n_features > tuning.linreg_wide_max_features or work < tuning.linreg_min_work
        else:
            below = (
                n_samples < tuning.linreg_min_samples
                or n_features < tuning.linreg_min_features
                or work < tuning.linreg_min_work
                or n_features > tuning.linreg_max_features
                or n_samples < tuning.linreg_min_aspect * n_features
                # More targets than features: on graded designs the tiled-QR
                # rescue is not available, and a refused attempt costs a share
                # of stock's own time.
                or n_targets > tuning.linreg_max_target_ratio * n_features
            )
        return below, (f"{n_samples} samples x {n_features} features x {n_targets} targets "
                       "is outside the OLS crossover")

    def fit(self, X, y, sample_weight=None):
        """Fit the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : LinearRegression
        """
        if not self._mlxlearn_allow_fallback:
            # Layer 1 validates first, so its parameter errors are stock's. The
            # patched class leaves that to whichever fit runs: stock's own fit
            # validates its parameters, so a fit handed to it pays once.
            self._validate_params()

        self._clear_backend()
        shape = _safe_shape(X)
        if shape is None and not hasattr(X, "shape"):
            # Lists and other shape-less array-likes: convert once so the size
            # crossover applies. Validation (and stock, in patched mode) gets the
            # conversion only when it is an ordinary numeric array; anything else
            # (ragged rows, None, strings, generators) goes on as the caller
            # passed it, so the errors and their order are stock's.
            try:
                converted = np.asarray(X)
            except (ValueError, TypeError):
                converted = None
            if converted is not None:
                shape = _safe_shape(converted)
                if converted.dtype.kind in "biuf" and converted.ndim >= 1:
                    X = converted
        below, detail = False, ""
        if shape is not None and get_config().device == "auto":
            y_shape = np.shape(y) if hasattr(y, "shape") or isinstance(y, (list, tuple)) else ()
            n_targets = int(y_shape[1]) if len(y_shape) == 2 else 1
            below, detail = self._below_crossover(*shape, n_targets)
        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_linear_capability(X),
            below_crossover=below,
            crossover_detail=detail,
            prefer_cpu=True,
            shape=shape,
        )
        if backend == "sklearn":
            super().fit(X, y, sample_weight)
            self._set_backend("sklearn")
            return self
        if self._mlxlearn_allow_fallback:
            self._validate_params()

        # Validate as stock does: numeric first (MLX float32 stays float32, strings
        # are rejected), keeping the caller's memory layout (the exact fallback
        # reproduces stock's column means, whose rounding depends on it).
        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=True,
            ensure_min_samples=1,
            reset=True,
        )
        weights = None
        if sample_weight is not None:
            weights = _sample_weight_as_stock(sample_weight, X_checked)
        # _preprocess_data's second check_array. A floating X is already checked;
        # anything else (integers, bools, an object array validate_data left
        # unconverted, such as a list holding None) gets stock's own call, with
        # its conversion and its errors. y is cast to X's dtype, with stock's
        # check where it can fail.
        if X_checked.dtype not in (np.float64, np.float32, np.float16):
            X_checked = check_array(X_checked, dtype=[np.float64, np.float32, np.float16])
        y_arr = _targets_as_stock(y_checked, X_checked.dtype)
        if weights is not None and self.fit_intercept and float(np.sum(weights, dtype=np.float64)) == 0.0:
            # Where stock's weighted mean (np.average) raises it, after the y
            # checks (1.7 and 1.8; 1.9 rejects all-zero weights during
            # validation). NaN weights pass on, as in stock, to the solver's
            # finiteness check.
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        y_is_1d = y_arr.ndim == 1
        if _DENSE_CUTOFF_IS_TOL:
            rcond = float(self.tol)
        else:
            rcond = float(max(X_checked.shape) * np.finfo(X_checked.dtype).eps)

        with mlx_guard(type(self).__name__, "fit"):
            solution = solve_ols(
                X_checked,
                y_arr,
                sample_weight=weights,
                config=OLSConfig(
                    fit_intercept=bool(self.fit_intercept),
                    rcond=rcond,
                    threads=int(get_tuning().linreg_threads),
                ),
            )

        if y_is_1d:
            self.coef_ = np.ravel(solution.coef)
        else:
            self.coef_ = solution.coef
        if self.fit_intercept:
            self.intercept_ = (
                np.float64(solution.intercept.ravel()[0])
                if y_is_1d
                else np.ascontiguousarray(solution.intercept, dtype=np.float64)
            )
        else:
            self.intercept_ = 0.0
        self.rank_ = int(solution.rank)
        self.singular_ = np.ascontiguousarray(solution.singular, dtype=np.float64)
        if backend == "mlx":
            # Ordinary least squares has no Metal path; the host solved it.
            backend = "cpu"
            record_event(
                type(self).__name__, "fit", backend, reason="host-solve",
                detail="ordinary least squares is solved on the host", shape=X_checked.shape,
            )
        # Which solver path ran (gram, ritz, qr or the bit-identical lstsq), so
        # internal fallbacks are visible in the diagnostics.
        record_event(type(self).__name__, "fit", backend, reason=f"ols-{solution.method}",
                     shape=X_checked.shape)
        self._set_backend(backend)
        return self

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_targets)
            float64.
        """
        check_is_fitted(self)
        if getattr(self, "_execution_backend_", None) is not None:
            # A model whose coef_/intercept_ were set by hand has no recorded
            # backend; stock scikit-learn predicts from it, and so does this.
            self._note_inference("predict", shape=_safe_shape(X))
        return super().predict(X)


def _targets_as_stock(y, dtype) -> np.ndarray:
    """``check_array(y, dtype=X.dtype, ensure_2d=False)`` as stock's _preprocess_data runs it.

    validate_data has already checked a numeric y, so a clean one only needs the
    cast and one finiteness pass; sparse, string or object targets, and any
    non-finite value (from a narrowing cast, or None/'nan' in an object or
    string y, which validate_data does not catch), go through stock's own call
    for its exact error. It honours ``assume_finite`` as stock does.
    """
    if sp.issparse(y) or getattr(y, "dtype", None) is None or y.dtype.kind not in "biuf":
        return check_array(y, dtype=dtype, ensure_2d=False)
    y_arr = np.asarray(y, dtype=dtype)
    if not np.isfinite(y_arr).all():
        return check_array(y, dtype=dtype, ensure_2d=False)
    return y_arr


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))
