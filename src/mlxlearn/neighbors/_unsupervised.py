"""Unsupervised nearest neighbors."""

from __future__ import annotations

from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors

from ._base import MLXNeighborsMixin

__all__ = ["NearestNeighbors"]


class NearestNeighbors(MLXNeighborsMixin, _SkNearestNeighbors):
    """Unsupervised nearest-neighbor search, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.NearestNeighbors` for
    exact brute-force Euclidean search. Parameters, fitted attributes and error
    messages follow scikit-learn.

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
        Which backend fitted this estimator. Every subsequent query uses it.

    Notes
    -----
    :meth:`radius_neighbors` is inherited from scikit-learn and is **not**
    accelerated. It is correct — mlxlearn keeps the fitted attributes
    scikit-learn's implementation needs — but it runs in float64 on the CPU. This
    is stated rather than left for a user to discover from a profiler; the
    ancestor inherited the same method and never mentioned it.

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
            super().fit(X, y)
            self._set_backend("sklearn")
        return self
