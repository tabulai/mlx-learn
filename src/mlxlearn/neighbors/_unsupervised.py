"""Unsupervised nearest neighbors."""

from __future__ import annotations

from sklearn.base import BaseEstimator

from ._base import MLXNeighborsMixin, MLXRadiusMixin

__all__ = ["NearestNeighbors"]


class NearestNeighbors(MLXRadiusMixin, MLXNeighborsMixin, BaseEstimator):
    """Unsupervised nearest-neighbor search, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.NearestNeighbors` for
    exact Euclidean search. Parameters, fitted attributes and error
    messages follow scikit-learn. Concrete sklearn neighbor subclass identity
    is preserved only by patched variants; direct instances expose their own
    ``kneighbors_graph`` and ``radius_neighbors_graph`` methods.

    Parameters
    ----------
    n_neighbors : int, default=5
    radius : float, default=1.0
        Used by :meth:`radius_neighbors`.
    algorithm : {"auto", "brute"}, default="auto"
        ``"auto"`` can use a low-dimensional tree or an index of duplicate rows.
        ``"brute"`` keeps every query on brute-force engines. Results are the
        same either way. Explicit ``"kd_tree"`` and ``"ball_tree"`` raise
        :class:`~mlxlearn.exceptions.UnsupportedParameterError` on direct use and
        fall back to scikit-learn under ``patch_sklearn()``.
    leaf_size : int, default=30
        Accepted for signature compatibility. The owned engines choose their
        own tree leaves; a call delegated to scikit-learn uses this value.
    metric : {"euclidean", "l2", "minkowski"}, default="minkowski"
        ``"minkowski"`` is supported at ``p=2`` only.
    p : float, default=2
    metric_params : dict, default=None
        Not implemented on the MLX path. mlxlearn raises rather than silently
        computing unweighted Euclidean distances.
    n_jobs : int, default=None
        Worker threads for the host radius engines. ``None`` and ``-1`` use all
        cores. k-neighbor search ignores it.

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
        Backend used by the most recent :meth:`kneighbors` or
        :meth:`radius_neighbors` call. Defined after the first completed query.

    Notes
    -----
    :meth:`radius_neighbors` returns exact neighbor sets for the float32
    coordinates: a training row is returned iff its exact squared distance is
    at most ``fl64(radius ** 2)``. Automatic queries can use a tree, host brute
    force, a Metal Gram screen, or exact row lookup for batched zero-radius
    searches. Each engine returns indices in ascending order unless
    ``sort_results=True``. Patched estimators can delegate unsupported inputs
    to scikit-learn. For supported inputs, automatic delegation requires a
    faster stock route that preserves exact membership.
    Neighborhoods are ragged arrays with one row per query.

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
