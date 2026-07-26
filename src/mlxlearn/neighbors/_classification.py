"""k-nearest-neighbors classification."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as _SkKNeighborsClassifier

from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin, class_probabilities, compute_weights, weighted_mode

__all__ = ["KNeighborsClassifier"]


class KNeighborsClassifier(MLXNeighborsMixin, _SkKNeighborsClassifier):
    """k-nearest-neighbors classifier, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.KNeighborsClassifier` for
    exact brute-force Euclidean search on dense, single-output data.

    Parameters
    ----------
    n_neighbors : int, default=5
    weights : {"uniform", "distance"}, default="uniform"
        A callable is not implemented on the MLX path.
    algorithm : {"auto", "brute"}, default="auto"
    leaf_size : int, default=30
        Accepted for signature compatibility; unused by brute-force search.
    p : float, default=2
    metric : {"euclidean", "l2", "minkowski"}, default="minkowski"
        ``"minkowski"`` at ``p=2`` only.
    metric_params : dict, default=None
        Not implemented; raises rather than being silently ignored.
    n_jobs : int, default=None
        Accepted and ignored.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    Ties are resolved exactly as scikit-learn resolves them: neighbors are ordered
    by ``(distance, training index)``, and a tied vote goes to the smallest class
    code, i.e. the class that sorts first in ``classes_``.

    ``weights="distance"`` gives a query that coincides exactly with training
    points all of its weight on those points, matching scikit-learn, rather than
    producing an infinity.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.neighbors import KNeighborsClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    >>> clf.predict([[1.1]])
    array([0])
    """

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : KNeighborsClassifier
        """
        if self._fit_neighbors(X, y, target="classification") == "sklearn":
            super().fit(X, y)
            self._set_backend("sklearn")
            return self

        y = self._y_checked_
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._y = encoded.astype(np.intp, copy=False)
        self.outputs_2d_ = False
        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Labels drawn from ``classes_``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        dist, idx = self._query_indices(X, need_distance=self.weights == "distance")
        weights = compute_weights(dist, self.weights)
        codes = self._y[idx]
        winners = weighted_mode(codes, weights, len(self.classes_))
        return self.classes_.take(winners)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes)
            Rows sum to 1, float64, columns ordered by ``classes_``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict_proba") == "sklearn":
            self._note_inference("predict_proba")
            return super().predict_proba(X)

        dist, idx = self._query_indices(X, need_distance=self.weights == "distance")
        weights = compute_weights(dist, self.weights)
        codes = self._y[idx]
        return class_probabilities(codes, weights, len(self.classes_))
