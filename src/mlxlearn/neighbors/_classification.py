"""k-nearest-neighbors classification."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .._common.base import mlx_guard
from .._common.device import use_device
from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin, compute_weights

__all__ = ["KNeighborsClassifier"]


class KNeighborsClassifier(MLXNeighborsMixin, ClassifierMixin, BaseEstimator):
    """k-nearest-neighbors classifier, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.KNeighborsClassifier` for
    exact brute-force Euclidean search on dense, single-output data.
    Concrete sklearn neighbor subclass identity is preserved only by patched
    variants; direct instances expose their own ``kneighbors_graph`` method.

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
    Neighbors are ordered by true distance and then training index. A tied vote
    goes to the smallest class code, i.e. the class that sorts first in
    ``classes_``. scikit-learn may choose different members when equal-distance
    candidates cross the kth-neighbor boundary.

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

    _sklearn_class_name = "KNeighborsClassifier"

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
            return self._fit_sklearn(X, y)

        y = self._y_checked_
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._y = encoded.astype(np.intp, copy=False)
        del self._y_checked_
        self.outputs_2d_ = False
        if self.execution_policy_ != "auto" or self.execution_backend_ == "mlx":
            with mlx_guard(type(self).__name__, "fit"):
                self._ensure_device_targets()
        return self

    @property
    def _y_device_(self):
        return self.__dict__.get("_knn_query_state_", {}).get("targets")

    @_y_device_.setter
    def _y_device_(self, value):
        self._knn_query_state_["targets"] = value

    def _ensure_device_targets(self):
        """Upload encoded labels once per fit, rebuilding only after pickle."""
        import mlx.core as mx

        if getattr(self, "_y_device_", None) is None:
            backend = "mlx" if self.execution_policy_ == "auto" else self._fitted_backend()
            with use_device("gpu" if backend == "mlx" else "cpu"):
                self._y_device_ = mx.array(self._y, dtype=mx.int32)
                mx.eval(self._y_device_)
        return self._y_device_

    def _predict_on_device(self, X, *, probabilities):
        """Search and vote before transferring the public result to NumPy."""
        if self.weights not in (None, "uniform", "distance"):
            raise ValueError(
                "weights not recognized: should be 'uniform' or 'distance', "
                f"got {self.weights!r}"
            )
        operation = "predict_proba" if probabilities else "predict"
        backend = self._fitted_backend(operation)
        prepared = None
        if self.execution_policy_ == "auto":
            prepared = self._prepare_neighbor_query(X)
            Q, k, query_is_train = prepared
            policy = self._index_._query_policy(Q.shape[0], k, Q)
            backend = policy[0]
            if backend == "cpu":
                with mlx_guard(type(self).__name__, operation):
                    dist, idx = self._index_.query(
                        Q, k, return_distance=self.weights == "distance",
                        exclude_self=query_is_train, _policy=policy,
                    )
                    weights = compute_weights(dist, self.weights)
                    votes = np.zeros((len(idx), len(self.classes_)), dtype=np.float64)
                    rows = np.arange(len(idx))[:, None]
                    np.add.at(votes, (rows, self._y[idx]), 1.0 if weights is None else weights)
                    self._record_neighbor_query(operation, Q.shape)
                    if probabilities:
                        return votes / votes.sum(axis=1, keepdims=True)
                    return self.classes_.take(votes.argmax(axis=1))
        import mlx.core as mx

        from mlxlearn_device._neighbors_votes import class_predictions, class_probabilities

        with mlx_guard(type(self).__name__, operation), use_device(
            "gpu" if backend == "mlx" else "cpu"
        ):
            if prepared is None:
                dist, idx = self._query_device_indices(
                    X, need_distance=self.weights == "distance", squared=True
                )
            else:
                dist, idx = self._index_.query_device(
                    Q, k, return_distance=self.weights == "distance",
                    exclude_self=query_is_train, _squared=True,
                )
                self._record_neighbor_query(operation, Q.shape)
            aggregate = class_probabilities if probabilities else class_predictions
            result = aggregate(
                idx, self._ensure_device_targets(), len(self.classes_), distances=dist,
                squared_distances=True,
            )
            mx.eval(result)
            if probabilities:
                output = np.asarray(result, dtype=np.float64)
                invalid = not np.isfinite(output).all()
            else:
                output = np.asarray(result, dtype=np.intp)
                invalid = np.any(output < 0)
            if invalid:
                raise ValueError("All neighbors of some sample are getting zero weights.")
            return output if probabilities else self.classes_.take(output)

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
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().predict(X)

        return self._predict_on_device(X, probabilities=False)

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
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().predict_proba(X)

        return self._predict_on_device(X, probabilities=True)
