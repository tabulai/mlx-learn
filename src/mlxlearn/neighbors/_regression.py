"""k-nearest-neighbors regression."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor as _SkKNeighborsRegressor

from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin, compute_weights

__all__ = ["KNeighborsRegressor"]


class KNeighborsRegressor(MLXNeighborsMixin, _SkKNeighborsRegressor):
    """k-nearest-neighbors regressor, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.KNeighborsRegressor` for
    exact brute-force Euclidean search on dense data.

    Parameters
    ----------
    n_neighbors : int, default=5
    weights : {"uniform", "distance"} or callable, default="uniform"
        A callable is applied to the neighbor distances as in scikit-learn; see
        :class:`~mlxlearn.neighbors.KNeighborsClassifier`.
    algorithm : {"auto", "brute"}, default="auto"
        See :class:`~mlxlearn.neighbors.KNeighborsClassifier` for how ``"auto"``
        is dispatched on low-dimensional data under ``patch_sklearn()``.
    leaf_size : int, default=30
    p : float, default=2
    metric : {"euclidean", "l2", "minkowski", "sqeuclidean"}, default="minkowski"
    metric_params : dict, default=None
        Minkowski's ``p`` only; anything else raises.
    n_jobs : int, default=None

    Attributes
    ----------
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    Multi-output targets are accelerated, and a column-vector ``y`` of shape
    ``(n_samples, 1)`` gives predictions of shape ``(n_queries, 1)``, as it does
    in scikit-learn.

    A ``y`` of strings raises ``ValueError`` at ``fit`` on the MLX path.
    scikit-learn stores it and fails in ``predict`` instead; it is never parsed
    into numbers.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.neighbors import KNeighborsRegressor
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> reg = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    >>> reg.predict([[1.5]])
    array([1.5])
    """

    def fit(self, X, y):
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self : KNeighborsRegressor
        """
        if self._fit_neighbors(X, y, target="regression") == "sklearn":
            super().fit(X, y)
            self._set_backend("sklearn")
        return self

    def _fit_targets(self, y: np.ndarray) -> None:
        # A copy, always. validate_data hands back the caller's own buffer when y
        # is already float64, and a regressor holding a view of it changes its
        # predictions when the caller writes into y — "fit copies" applies to the
        # target as much as to X.
        self._y = np.array(y, dtype=np.float64, copy=True)

    def predict(self, X):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            float64. Two-dimensional exactly when ``y`` was at fit.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        dist, idx = self._query_indices(X, "predict", need_distance=self.weights != "uniform")
        weights = compute_weights(dist, self.weights)

        # The same arithmetic, in the same order, as scikit-learn's predict.
        _y = self._y if self._y.ndim == 2 else self._y.reshape((-1, 1))
        if weights is None:
            y_pred = np.mean(_y[idx], axis=1)
        else:
            y_pred = np.empty((idx.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)
            for j in range(_y.shape[1]):
                y_pred[:, j] = np.sum(_y[idx, j] * weights, axis=1) / denom

        return y_pred.ravel() if self._y.ndim == 1 else y_pred
