"""Radius-neighbor classification and regression."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier as _SkRadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor as _SkRadiusNeighborsRegressor

from .._common.arrays import INDEX_DTYPE
from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin, MLXRadiusMixin

__all__ = ["RadiusNeighborsClassifier", "RadiusNeighborsRegressor"]


class RadiusNeighborsClassifier(MLXRadiusMixin, MLXNeighborsMixin, _SkRadiusNeighborsClassifier):
    """Radius-neighbors classifier, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.RadiusNeighborsClassifier`
    for exact brute-force Euclidean search on dense, single-output data. Every
    training row within ``radius`` of a query votes; an empty neighborhood is
    handled by ``outlier_label``.

    Parameters
    ----------
    radius : float, default=1.0
    weights : {"uniform", "distance"}, default="uniform"
        A callable is not implemented on the MLX path.
    algorithm : {"auto", "brute"}, default="auto"
    leaf_size : int, default=30
        Accepted for signature compatibility; unused by brute-force search.
    p : float, default=2
    metric : {"euclidean", "l2", "minkowski"}, default="minkowski"
        ``"minkowski"`` at ``p=2`` only.
    outlier_label : {manual label, "most_frequent"}, default=None
        Label for queries with no neighbor inside ``radius``. ``None`` raises.
    metric_params : dict, default=None
        Not implemented; raises rather than being silently ignored.
    n_jobs : int, default=None
        Accepted and ignored.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    outlier_label_ : list or None
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    Ties are resolved as scikit-learn resolves them: a tied vote goes to the
    smallest class code, i.e. the class that sorts first in ``classes_``.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.neighbors import RadiusNeighborsClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = RadiusNeighborsClassifier(radius=1.0).fit(X, y)
    >>> clf.predict([[1.5]])
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
        self : RadiusNeighborsClassifier
        """
        if self._fit_neighbors(X, y, target="classification") == "sklearn":
            super().fit(X, y)
            self._set_backend("sklearn")
            return self

        y = self._y_checked_
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._y = encoded.astype(np.intp, copy=False)
        self.outputs_2d_ = False
        self.outlier_label_ = _resolve_outlier_label(
            self.outlier_label, self.classes_, self._y
        )
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

        proba = self.predict_proba(X)
        labels = self.classes_.take(proba.argmax(axis=1))
        empty = np.all(proba == 0.0, axis=1)
        if np.any(empty):
            labels = np.array(labels, copy=True)
            labels[empty] = self.outlier_label_[0]
        return labels

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes)
            Rows sum to 1, float64, columns ordered by ``classes_``. An outlier
            whose label is in ``classes_`` gets a one-hot row; an outlier whose
            label is not in the training set gets a zero row.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict_proba") == "sklearn":
            self._note_inference("predict_proba")
            return super().predict_proba(X)

        dist, idx = self.radius_neighbors(X, return_distance=True)
        n_queries = len(idx)
        n_classes = len(self.classes_)
        empty = np.array([np.asarray(row).size == 0 for row in idx], dtype=bool)
        outliers = np.flatnonzero(empty)
        if outliers.size > 0 and self.outlier_label_ is None:
            raise ValueError(
                "No neighbors found for test samples "
                f"{outliers!r}, "
                "you can try using larger radius, "
                "giving a label for outliers, "
                "or considering removing them from your dataset."
            )

        proba = np.zeros((n_queries, n_classes), dtype=np.float64)
        weights = _ragged_weights(dist, self.weights)
        for i in range(n_queries):
            neighbors = np.asarray(idx[i], dtype=np.intp)
            if neighbors.size == 0:
                continue
            codes = self._y[neighbors]
            if weights is None:
                votes = np.bincount(codes, minlength=n_classes).astype(np.float64)
            else:
                votes = np.bincount(
                    codes, weights=np.asarray(weights[i], dtype=np.float64), minlength=n_classes
                )
            total = votes.sum()
            if total > 0:
                proba[i] = votes / total

        if outliers.size > 0:
            label = self.outlier_label_[0]
            match = np.flatnonzero(self.classes_ == label)
            if match.size == 1:
                proba[outliers] = 0.0
                proba[outliers, match[0]] = 1.0
            else:
                warnings.warn(
                    f"Outlier label {label} is not in training classes. All class "
                    "probabilities of outliers will be assigned with 0.",
                    stacklevel=2,
                )
        return proba


class RadiusNeighborsRegressor(MLXRadiusMixin, MLXNeighborsMixin, _SkRadiusNeighborsRegressor):
    """Radius-neighbors regressor, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.RadiusNeighborsRegressor`
    for exact brute-force Euclidean search on dense, single-output data.

    Parameters
    ----------
    radius : float, default=1.0
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

    Notes
    -----
    A query with no neighbor inside ``radius`` predicts ``NaN`` and emits a
    warning, matching scikit-learn.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.neighbors import RadiusNeighborsRegressor
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> reg = RadiusNeighborsRegressor(radius=1.0).fit(X, y)
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
        self : RadiusNeighborsRegressor
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
            float64. ``NaN`` where a query has no neighbor inside ``radius``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        dist, idx = self.radius_neighbors(X, return_distance=True)
        n_queries = len(idx)
        y_pred = np.empty(n_queries, dtype=np.float64)
        weights = _ragged_weights(dist, self.weights)
        empty = False
        for i in range(n_queries):
            neighbors = np.asarray(idx[i], dtype=INDEX_DTYPE)
            if neighbors.size == 0:
                y_pred[i] = np.nan
                empty = True
                continue
            values = self._y[neighbors]
            if weights is None:
                y_pred[i] = float(np.mean(values))
            else:
                w = np.asarray(weights[i], dtype=np.float64)
                y_pred[i] = float(np.average(values, weights=w))
        if empty:
            warnings.warn(
                "One or more samples have no neighbors within specified radius; predicting NaN.",
                stacklevel=2,
            )
        return y_pred


def _resolve_outlier_label(outlier_label, classes, encoded_y):
    """Populate ``outlier_label_`` the way scikit-learn does for a single output."""
    if outlier_label is None:
        return None
    if outlier_label == "most_frequent":
        counts = np.bincount(encoded_y, minlength=len(classes))
        return [classes[int(counts.argmax())]]

    if not np.isscalar(outlier_label) and not isinstance(outlier_label, (str, bytes, np.generic)):
        raise TypeError(
            f"The outlier_label of classes {classes} is supposed to be a scalar, "
            f"got {outlier_label}."
        )
    label_arr = np.append(classes, outlier_label)
    if label_arr.dtype != classes.dtype:
        raise TypeError(
            f"The dtype of outlier_label {outlier_label} is inconsistent with "
            f"classes {classes} in y."
        )
    return [outlier_label]


def _ragged_weights(neigh_dist, weights):
    """Per-query distance weights for object arrays of 1-d distance rows."""
    if weights in (None, "uniform"):
        return None
    if weights != "distance":
        raise ValueError(
            f"weights not recognized: should be 'uniform' or 'distance', got {weights!r}"
        )
    out = np.empty(len(neigh_dist), dtype=object)
    for i, dist in enumerate(neigh_dist):
        d = np.asarray(dist, dtype=np.float64)
        if d.size == 0:
            out[i] = d
            continue
        with np.errstate(divide="ignore"):
            w = 1.0 / d
        if np.any(d == 0):
            w = (d == 0).astype(np.float64)
        out[i] = w
    return out
