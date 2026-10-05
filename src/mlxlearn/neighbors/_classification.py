"""k-nearest-neighbors classification."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.neighbors import KNeighborsClassifier as _SkKNeighborsClassifier

from .._common.validation import check_classification_targets, check_is_fitted
from ._base import MLXNeighborsMixin, class_probabilities, compute_weights, weighted_mode

__all__ = ["KNeighborsClassifier"]


class KNeighborsClassifier(MLXNeighborsMixin, _SkKNeighborsClassifier):
    """k-nearest-neighbors classifier, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.KNeighborsClassifier` for
    exact brute-force Euclidean search on dense data.

    Parameters
    ----------
    n_neighbors : int, default=5
    weights : {"uniform", "distance"} or callable, default="uniform"
        A callable takes the array of neighbor distances and returns weights of
        the same shape, as in scikit-learn; it runs on the host, on the float64
        distances :meth:`kneighbors` returns.
    algorithm : {"auto", "brute"}, default="auto"
        See Notes for how ``"auto"`` is dispatched under ``patch_sklearn()``.
    leaf_size : int, default=30
        Accepted for signature compatibility; unused by brute-force search.
    p : float, default=2
    metric : {"euclidean", "l2", "minkowski", "sqeuclidean"}, default="minkowski"
        ``"minkowski"`` at ``p=2`` only.
    metric_params : dict, default=None
        Only Minkowski's ``p``, which overrides the ``p`` parameter as it does in
        scikit-learn. Anything else — a weight vector ``w``, another metric's
        parameters — raises rather than being silently ignored.
    n_jobs : int, default=None
        Accepted and ignored.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of such arrays
        A list, one array per output, when ``y`` is multi-output.
    outputs_2d_ : bool
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    Neighbors are the first ``k`` training rows under ``(distance, training
    index)``, a total order over the whole training set, so rows tied at the
    k-th distance are decided by index. scikit-learn defines no order among
    exactly equal distances and keeps an arbitrary subset of such rows, so on
    tie-heavy data a vote can legitimately differ from its vote; this one is
    reproducible (``docs/fp32_policy.md`` section 3.1). A vote tied between
    classes goes to the class that sorts first in ``classes_``, as in
    scikit-learn.

    ``weights="distance"`` gives a query that coincides exactly with training
    points all of its weight on those points, matching scikit-learn, rather than
    producing an infinity.

    Multi-output targets are accelerated: the neighbor search does not depend on
    ``y``, and the vote is taken per output, as scikit-learn takes it.

    With at most 15 features scikit-learn's ``algorithm="auto"`` builds a
    KD-tree, which answered a single-row query faster than an MLX call at every
    size measured up to 12 features, and at 2-5 features every batch. Under
    ``patch_sklearn()`` those fits, and any other with fewer than 16 features, are
    handed to scikit-learn. Imported directly, this class always searches by brute
    force, so on that kind of data stock scikit-learn is the faster choice unless
    the queries come in large batches.

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
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self : KNeighborsClassifier
        """
        if self._fit_neighbors(X, y, target="classification") == "sklearn":
            super().fit(X, y)
            self._set_backend("sklearn")
        return self

    def _fit_targets(self, y: np.ndarray) -> None:
        """Encode the validated target exactly as scikit-learn's ``_fit`` does.

        Including its column-vector warning: a ``(n, 1)`` target is a single
        output, so scikit-learn ravels it and says so, and so does this.
        """
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            if y.ndim != 1:
                warnings.warn(
                    (
                        "A column-vector y was passed when a 1d array was expected. "
                        "Please change the shape of y to (n_samples,), for example "
                        "using ravel()."
                    ),
                    DataConversionWarning,
                    stacklevel=4,
                )
            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        check_classification_targets(y)
        classes = []
        # np.unique returns fresh arrays, so nothing here aliases the caller's y.
        encoded = np.empty(y.shape, dtype=np.intp)
        for k in range(y.shape[1]):
            classes_k, encoded[:, k] = np.unique(y[:, k], return_inverse=True)
            classes.append(classes_k)

        if self.outputs_2d_:
            self.classes_ = classes
            self._y = encoded
        else:
            self.classes_ = classes[0]
            self._y = encoded.ravel()

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Labels drawn from ``classes_``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        dist, idx = self._query_indices(X, "predict", need_distance=self.weights != "uniform")
        weights = _check_nonzero_weights(compute_weights(dist, self.weights))
        if not self.outputs_2d_:
            winners = weighted_mode(self._y[idx], weights, len(self.classes_))
            return self.classes_.take(winners)

        # scikit-learn allocates the result with the first output's label dtype.
        y_pred = np.empty((idx.shape[0], len(self.classes_)), dtype=self.classes_[0].dtype)
        for k, classes_k in enumerate(self.classes_):
            winners = weighted_mode(self._y[idx, k], weights, len(classes_k))
            y_pred[:, k] = classes_k.take(winners)
        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of them
            Rows sum to 1, float64, columns ordered by ``classes_``. One array per
            output when ``y`` was multi-output.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict_proba") == "sklearn":
            self._note_inference("predict_proba")
            return super().predict_proba(X)

        dist, idx = self._query_indices(
            X, "predict_proba", need_distance=self.weights != "uniform"
        )
        weights = _check_nonzero_weights(compute_weights(dist, self.weights))
        if not self.outputs_2d_:
            return class_probabilities(self._y[idx], weights, len(self.classes_))
        return [
            class_probabilities(self._y[idx, k], weights, len(classes_k))
            for k, classes_k in enumerate(self.classes_)
        ]


def _check_nonzero_weights(weights: np.ndarray | None) -> np.ndarray | None:
    """Refuse a query whose neighbors all have zero weight, as scikit-learn's classifier does.

    Only a callable can produce one, and there is no vote to take for it.
    """
    if weights is not None and np.any(np.all(weights == 0, axis=1)):
        raise ValueError(
            "All neighbors of some sample is getting zero weights. "
            "Please modify 'weights' to avoid this case if you are "
            "using a user-defined function."
        )
    return weights
