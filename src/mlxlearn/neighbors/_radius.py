"""Radius-neighbor classification and regression."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin, MLXRadiusMixin

__all__ = ["RadiusNeighborsClassifier", "RadiusNeighborsRegressor"]


class RadiusNeighborsClassifier(MLXRadiusMixin, MLXNeighborsMixin, ClassifierMixin, BaseEstimator):
    """Radius-neighbors classifier, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.RadiusNeighborsClassifier`
    for exact Euclidean radius search on dense, single-output data. Every
    training row within ``radius`` of a query votes; an empty neighborhood is
    handled by ``outlier_label``.

    Parameters
    ----------
    radius : float, default=1.0
    weights : {"uniform", "distance"}, default="uniform"
        A callable is not implemented on the MLX path.
    algorithm : {"auto", "brute"}, default="auto"
    leaf_size : int, default=30
        Accepted for signature compatibility.
    p : float, default=2
    metric : {"euclidean", "l2", "minkowski"}, default="minkowski"
        ``"minkowski"`` at ``p=2`` only.
    outlier_label : {manual label, "most_frequent"}, default=None
        Label for queries with no neighbor inside ``radius``. ``None`` raises.
    metric_params : dict, default=None
        Not implemented; raises rather than being silently ignored.
    n_jobs : int, default=None
        Worker threads for the host radius engines; ``None`` and ``-1`` use all
        cores.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    outlier_label_ : list or None
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}
        Which backend initially stored the fitted training data;
        ``last_query_backend_`` reports where each query ran.

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

    _sklearn_class_name = "RadiusNeighborsClassifier"
    # Radius estimators have no k; the shared parameter check still reads it.
    n_neighbors = None

    def __init__(self, radius=1.0, *, weights="uniform", algorithm="auto", leaf_size=30,
                 p=2, metric="minkowski", outlier_label=None, metric_params=None, n_jobs=None):
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.outlier_label = outlier_label
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
        self : RadiusNeighborsClassifier
        """
        if self._fit_neighbors(X, y, target="classification") == "sklearn":
            return self._fit_sklearn(X, y)

        y = self._y_checked_
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._y = encoded.astype(np.intp, copy=False)
        del self._y_checked_
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
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().predict(X)

        proba, stock = self._radius_proba(X, "predict")
        if stock is not None:
            return stock[0].predict(stock[1])
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
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().predict_proba(X)

        proba, stock = self._radius_proba(X, "predict_proba")
        if stock is not None:
            return stock[0].predict_proba(stock[1])
        return proba

    def _radius_proba(self, X, operation):
        """Vote over every neighbor; ``(proba, None)`` or ``(None, (stock_model, query))``."""
        weights = self.weights
        if weights not in (None, "uniform", "distance") and not callable(weights):
            raise ValueError(
                f"weights not recognized: should be 'uniform' or 'distance', got {weights!r}"
            )
        if callable(weights) and not self._mlxlearn_allow_fallback:
            raise ValueError(
                "A callable weights function is not implemented on the MLX path; "
                "use weights='uniform' or 'distance', or run under patch_sklearn()."
            )
        weighted = weights == "distance"
        parts, stock = self._radius_parts(X, None, need_distance=weighted, sort_results=False,
                                          operation=operation, raw=True)
        if stock is not None:
            return None, (self._radius_stock(operation, stock), self._stock_query(X, stock))
        from .._kernels.radius_reduce import votes_of_parts

        n_classes = len(self.classes_)
        votes, counts = votes_of_parts(parts, self._y, n_classes, weighted=weighted)
        outliers = np.flatnonzero(counts == 0)
        if outliers.size > 0 and self.outlier_label_ is None:
            raise ValueError(
                "No neighbors found for test samples "
                f"{outliers!r}, "
                "you can try using larger radius, "
                "giving a label for outliers, "
                "or considering removing them from your dataset."
            )
        if outliers.size > 0:
            label = self.outlier_label_[0]
            match = np.flatnonzero(self.classes_ == label)
            if match.size == 1:
                votes[outliers, match[0]] = 1.0
            else:
                warnings.warn(
                    f"Outlier label {label} is not in training classes. All class "
                    "probabilities of outliers will be assigned with 0.",
                    stacklevel=3,
                )
        normalizer = votes.sum(axis=1)[:, None]
        normalizer[normalizer == 0.0] = 1.0
        votes /= normalizer
        return votes, None


class RadiusNeighborsRegressor(MLXRadiusMixin, MLXNeighborsMixin, RegressorMixin, BaseEstimator):
    """Radius-neighbors regressor, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.RadiusNeighborsRegressor`
    for exact Euclidean radius search on dense, single-output data.

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
        Worker threads for the host radius engines; ``None`` and ``-1`` use all
        cores.

    Attributes
    ----------
    n_features_in_ : int
    n_samples_fit_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}
        Which backend initially stored the fitted training data;
        ``last_query_backend_`` reports where each query ran.

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

    _sklearn_class_name = "RadiusNeighborsRegressor"
    # Radius estimators have no k; the shared parameter check still reads it.
    n_neighbors = None

    def __init__(self, radius=1.0, *, weights="uniform", algorithm="auto", leaf_size=30,
                 p=2, metric="minkowski", metric_params=None, n_jobs=None):
        self.radius = radius
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
        self : RadiusNeighborsRegressor
        """
        y_in = y
        column = False
        if self._mlxlearn_allow_fallback:
            # Patched twins advertise multi-output targets like scikit-learn's
            # regressor, which keeps a single-column target two-dimensional.
            # (Layer 1 is single-output: such a y warns and is raveled.)
            values = np.asarray(y)
            column = values.ndim == 2 and values.shape[1] == 1
            if column:
                y = values[:, 0]
        if self._fit_neighbors(X, y, target="regression") == "sklearn":
            return self._fit_sklearn(X, y_in)

        checked = self._y_checked_
        self._y = np.array(checked, dtype=np.float64, copy=True)
        self._y_out_dtype_ = np.float32 if checked.dtype == np.float32 else np.float64
        self._y_2d_ = bool(column)
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
            float64. ``NaN`` where a query has no neighbor inside ``radius``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            self._set_query_backend("sklearn")
            return self._sklearn_delegate().predict(X)

        weights = self.weights
        if weights not in (None, "uniform", "distance") and not callable(weights):
            raise ValueError(
                f"weights not recognized: should be 'uniform' or 'distance', got {weights!r}"
            )
        if callable(weights) and not self._mlxlearn_allow_fallback:
            raise ValueError(
                "A callable weights function is not implemented on the MLX path; "
                "use weights='uniform' or 'distance', or run under patch_sklearn()."
            )
        weighted = weights == "distance"
        parts, stock = self._radius_parts(X, None, need_distance=weighted, sort_results=False,
                                          operation="predict", raw=True)
        if stock is not None:
            return self._radius_stock("predict", stock).predict(self._stock_query(X, stock))
        from .._kernels.radius_reduce import mean_of_parts

        y_pred, counts = mean_of_parts(parts, self._y, weighted=weighted)
        if np.any(counts == 0):
            warnings.warn(
                "One or more samples have no neighbors within specified radius; predicting NaN.",
                stacklevel=2,
            )
        # Like scikit-learn: float32 targets give float32 predictions, and a
        # column-vector target gives column-vector predictions.
        y_pred = y_pred.astype(self.__dict__.get("_y_out_dtype_", np.float64), copy=False)
        if self.__dict__.get("_y_2d_", False):
            y_pred = y_pred.reshape(-1, 1)
        return y_pred


def _resolve_outlier_label(outlier_label, classes, encoded_y):
    """Populate ``outlier_label_`` the way scikit-learn does for a single output."""
    if outlier_label is None:
        return None
    if isinstance(outlier_label, str) and outlier_label == "most_frequent":
        counts = np.bincount(encoded_y, minlength=len(classes))
        return [classes[int(counts.argmax())]]

    def arraylike(value):
        return (hasattr(value, "__len__") or hasattr(value, "shape")
                or hasattr(value, "__array__")) and not isinstance(value, (str, bytes))

    if arraylike(outlier_label):
        if len(outlier_label) != 1:
            raise ValueError(
                f"The length of outlier_label: {outlier_label} is inconsistent with the "
                "output length: 1"
            )
        label = list(outlier_label)[0]
    else:
        label = outlier_label
    if arraylike(label):
        raise TypeError(
            f"The outlier_label of classes {classes} is supposed to be a scalar, "
            f"got {label}."
        )
    if np.append(classes, label).dtype != classes.dtype:
        raise TypeError(
            f"The dtype of outlier_label {label} is inconsistent with "
            f"classes {classes} in y."
        )
    return [label]
