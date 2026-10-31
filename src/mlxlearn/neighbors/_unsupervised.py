"""Unsupervised nearest neighbors."""

from __future__ import annotations

from sklearn.base import BaseEstimator

from .._common.validation import check_is_fitted
from ._base import MLXNeighborsMixin

__all__ = ["NearestNeighbors"]


class NearestNeighbors(MLXNeighborsMixin, BaseEstimator):
    """Unsupervised nearest-neighbor search, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.NearestNeighbors` for
    exact brute-force Euclidean search. Parameters, fitted attributes and error
    messages follow scikit-learn. Concrete sklearn neighbor subclass identity
    is preserved only by patched variants; direct instances expose their own
    ``kneighbors_graph`` and ``radius_neighbors_graph`` methods.

    Parameters
    ----------
    n_neighbors : int, default=5
    radius : float, default=1.0
        Used by :meth:`radius_neighbors`, which runs on scikit-learn — see Notes.
    algorithm : {"auto", "brute"}, default="auto"
        Only brute-force search is accelerated. ``"kd_tree"`` and ``"ball_tree"``
        raise :class:`~mlxlearn.exceptions.UnsupportedParameterError` on direct
        use and fall back to scikit-learn under ``patch_sklearn()``.
    leaf_size : int, default=30
        Accepted for signature compatibility; brute-force search does not use it.
    metric : {"euclidean", "l2", "minkowski"}, default="minkowski"
        ``"minkowski"`` is supported at ``p=2`` only.
    p : float, default=2
    metric_params : dict, default=None
        Not implemented on the MLX path. mlxlearn raises rather than silently
        computing unweighted Euclidean distances.
    n_jobs : int, default=None
        Accepted and ignored; the MLX path does not use joblib parallelism.

    Attributes
    ----------
    n_samples_fit_ : int
    n_features_in_ : int
    effective_metric_ : str
    execution_backend_ : {"mlx", "cpu", "sklearn"}
        Which backend initially stored the fitted training data.
    execution_policy_ : {"auto", "fixed"}
        The policy captured at fit. ``"auto"`` selects the owned CPU or GPU
        search for each query batch; explicit device choices are ``"fixed"``.
    last_query_backend_ : {"mlx", "cpu", "sklearn"}
        Backend used by the most recent :meth:`kneighbors` call. Defined after
        the first completed query. Delegated radius queries are not included.

    Notes
    -----
    :meth:`radius_neighbors` delegates to scikit-learn and is **not**
    accelerated. It uses the fitted float32 compatibility data on the CPU,
    as the preceding inherited implementation did. ``sklearn.neighbors`` is
    loaded only when that optional operation is requested.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.neighbors import NearestNeighbors
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> nn = NearestNeighbors(n_neighbors=2).fit(X)
    >>> dist, ind = nn.kneighbors([[0.5]])
    >>> ind
    array([[0, 1]])
    """

    _sklearn_class_name = "NearestNeighbors"

    def __init__(self, *, n_neighbors=5, radius=1.0, algorithm="auto", leaf_size=30,
                 metric="minkowski", p=2, metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : NearestNeighbors
        """
        if self._fit_neighbors(X, None) == "sklearn":
            self._fit_sklearn(X, y)
        return self

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Find points within radius using sklearn's public CPU implementation."""
        check_is_fitted(self)
        return self._radius_delegate().radius_neighbors(X, radius, return_distance, sort_results)

    def radius_neighbors_graph(self, X=None, radius=None, mode="connectivity", sort_results=False):
        """Return sklearn's CSR radius graph, preserving self-query exclusion."""
        check_is_fitted(self)
        return self._radius_delegate().radius_neighbors_graph(X, radius, mode, sort_results)
