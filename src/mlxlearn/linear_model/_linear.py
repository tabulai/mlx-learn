"""Ordinary least squares, accelerated with MLX.

Subclassing :class:`sklearn.linear_model.LinearRegression` is what makes the
drop-in promise real: the constructor signature, ``get_params``/``set_params``,
the parameter constraints and the estimator tags are all inherited, so ``clone``,
``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back
to.

The MLX path covers dense ordinary least squares, including multi-output targets
and sample weights. ``positive=True`` is a capability mismatch — non-negative
least squares is a different solver — and raises rather than being quietly
ignored.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression as _SkLinearRegression

from .._common.arrays import NUMPY_FLOAT
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.validation import (
    check_finite_float32,
    check_float32_range,
    check_is_fitted,
    check_sample_weight,
    require_dense,
    validate_data,
)
from .._solvers.ols import OLSConfig, solve_ols
from ..exceptions import UnsupportedParameterError

__all__ = ["LinearRegression"]


class LinearRegression(BackendMixin, _SkLinearRegression):
    """Ordinary least squares, fitted on MLX.

    A drop-in replacement for :class:`sklearn.linear_model.LinearRegression` on
    dense data. Multi-output targets and ``sample_weight`` are implemented;
    ``positive=True`` is not.

    Parameters
    ----------
    fit_intercept : bool, default=True
    copy_X : bool, default=True
        Accepted. The MLX path always copies into float32, so the caller's
        array is never modified.
    tol : float, default=1e-6
        Relative cutoff on singular values of the (centered, weighted) design,
        forwarded as ``rcond`` to the least-squares factorization. Absent on
        scikit-learn versions that do not yet expose it; those use ``1e-6``.
    n_jobs : int, default=None
        Accepted and ignored; the MLX path is not joblib-parallel.
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
        Effective rank of the (centered, weighted) design.
    singular_ : ndarray of shape (min(n_samples, n_features),)
        Singular values of that design, descending, float64.
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    **Precision.** Computation is float32 (``docs/fp32_policy.md``); ``coef_``
    and ``intercept_`` are float64 containers. The device computes the Gram
    products of the (centered, weighted) design; the small factorization that
    follows is host-side, because MLX's SVD is not implemented on the GPU.

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

    #: Ordinary least squares is natively multi-output; the MLX path is too.
    _mlxlearn_supports_multi_output = True

    def _check_linear_capability(self, X, y) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve."""
        name = type(self).__name__
        require_dense(X, estimator=name)
        check_float32_range(X)
        check_float32_range(y, name="y")

        if bool(self.positive):
            raise UnsupportedParameterError(
                f"{name} implements positive=False only; got positive={self.positive!r}. "
                "Non-negative least squares is a different solver.",
                parameter="positive",
                value=self.positive,
            )

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        No dedicated crossover has been measured yet, so both thresholds are
        high on purpose. See ``docs/benchmarks.md``. A problem must be large on
        both counts before patched dispatch will keep it on MLX.
        """
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < tuning.linreg_min_samples or work < tuning.linreg_min_work:
            return True, (
                f"{n_samples} samples x {n_features} features is below the "
                f"linear-regression crossover ({tuning.linreg_min_samples} samples "
                f"and {tuning.linreg_min_work} element-ops, both required)"
            )
        return False, ""

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
        self._validate_params()

        self._clear_backend()
        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_linear_capability(X, y),
            shape=_safe_shape(X),
        )
        if backend == "sklearn":
            super().fit(X, y, sample_weight)
            self._set_backend("sklearn")
            return self

        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            dtype=NUMPY_FLOAT,
            order="C",
            y_numeric=True,
            multi_output=True,
            ensure_min_samples=1,
            reset=True,
        )
        check_finite_float32(X_checked)

        below, detail = self._below_crossover(*X_checked.shape)
        if below and backend != "cpu":
            backend = self._select_backend(
                "fit",
                below_crossover=True,
                crossover_detail=detail,
                shape=X_checked.shape,
            )
            if backend == "sklearn":
                super().fit(X, y, sample_weight)
                self._set_backend("sklearn")
                return self

        y_arr = np.asarray(y_checked)
        y_is_1d = y_arr.ndim == 1
        weights = check_sample_weight(sample_weight, X_checked, dtype=np.float64)
        rcond = float(getattr(self, "tol", 1e-6))

        with mlx_guard(type(self).__name__, "fit"):
            solution = solve_ols(
                X_checked,
                y_arr,
                sample_weight=weights,
                config=OLSConfig(
                    fit_intercept=bool(self.fit_intercept),
                    rcond=rcond,
                    device="gpu" if backend == "mlx" else "cpu",
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
        self._note_inference("predict", shape=_safe_shape(X))
        return super().predict(X)


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))
