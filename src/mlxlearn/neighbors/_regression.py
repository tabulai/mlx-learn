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
    exact brute-force Euclidean search on dense, single-output data.

    Parameters
    ----------
    n_neighbors : int, default=5
    weights : {"uniform", "distance"}, default="uniform"
    algorithm : {"auto", "brute"}, default="auto"
    leaf_size : int, default=30
    p : float, default=2
    metric : {"euclidean", "l2", "minkowski"}, default="minkowski"
    metric_params : dict, default=None
    n_jobs : int, default=None

    Attributes
    ----------
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

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
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : KNeighborsRegressor
        """
        if self._fit_neighbors(X, y, target="regression") == "sklearn":
            super().fit(X, y)
            self._set_backend("sklearn")
            return self

        self._y = np.asarray(self._y_checked_, dtype=np.float64)
        return self

    def predict(self, X):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y : ndarray of shape (n_queries,)
            float64.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        dist, idx = self._query_indices(X, need_distance=self.weights == "distance")
        neighbors = self._y[idx]

        weights = compute_weights(dist, self.weights)
        if weights is None:
            return np.mean(neighbors, axis=1)
        return np.sum(neighbors * weights, axis=1) / np.sum(weights, axis=1)
