"""k-nearest-neighbors regression."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin, compute_weights

__all__ = ["KNeighborsRegressor"]


class KNeighborsRegressor(MLXNeighborsMixin, RegressorMixin, BaseEstimator):
    """k-nearest-neighbors regressor, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.KNeighborsRegressor` for
    exact brute-force Euclidean search on dense, single-output data.
    Concrete sklearn neighbor subclass identity is preserved only by patched
    variants; direct instances expose their own ``kneighbors_graph`` method.

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

    _sklearn_class_name = "KNeighborsRegressor"

    def __init__(self, n_neighbors=5, *, weights="uniform", algorithm="auto", leaf_size=30,
                 p=2, metric="minkowski", metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

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
            return self._fit_sklearn(X, y)

        # Keep the sklearn target contract, including values outside float32's
        # range and cancellation-sensitive float64 targets. The device search
        # stays shared; only this final small gather/reduction remains on host.
        self._y = np.array(self._y_checked_, dtype=np.float64, copy=True)
        del self._y_checked_
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
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().predict(X)

        dist, idx = self._query_indices(X, need_distance=self.weights == "distance")
        neighbors = self._y[idx]

        weights = compute_weights(dist, self.weights)
        if weights is None:
            return np.mean(neighbors, axis=1)
        return np.sum(neighbors * weights, axis=1) / np.sum(weights, axis=1)
