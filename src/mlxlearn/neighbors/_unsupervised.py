"""Unsupervised nearest neighbors."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors

from .._common.arrays import NUMPY_FLOAT
from .._common.validation import (
    _is_pandas,
    _safe_shape,
    check_float32_range,
    check_is_fitted,
    is_sparse,
)
from ._base import MLXNeighborsMixin, _reject_string_data, _sparse_query_for_sklearn

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
        use and fall back to scikit-learn under ``patch_sklearn()``. See Notes for
        how ``"auto"`` is dispatched on low-dimensional data.
    leaf_size : int, default=30
        Accepted for signature compatibility; brute-force search does not use it.
    metric : {"euclidean", "l2", "minkowski", "sqeuclidean"}, default="minkowski"
        ``"minkowski"`` is supported at ``p=2`` only.
    p : float, default=2
    metric_params : dict, default=None
        Only Minkowski's ``p``, which overrides the ``p`` parameter as it does in
        scikit-learn. Anything else — a weight vector ``w``, another metric's
        parameters — raises rather than being silently ignored, so mlxlearn never
        computes unweighted Euclidean distances for a weighted request.
    n_jobs : int, default=None
        Accepted and ignored; the MLX path does not use joblib parallelism.

    Attributes
    ----------
    n_samples_fit_ : int
    n_features_in_ : int
    effective_metric_ : str
        As scikit-learn sets it: ``"euclidean"`` for ``metric="minkowski"`` at
        ``p=2``, otherwise the metric as given.
    execution_backend_ : {"mlx", "cpu", "sklearn"}
        Which backend fitted this estimator. Every subsequent query uses it.

    Notes
    -----
    :meth:`radius_neighbors` and :meth:`radius_neighbors_graph` are **not**
    accelerated. mlxlearn keeps the fitted state scikit-learn's brute-force
    implementation reads, so they run scikit-learn's own code on the CPU — at
    stock speed, because the query is handed over in float32 to match the float32
    training copy. That copy is the training data rounded to float32 as given:
    unlike :meth:`kneighbors`, these methods neither translate float64 input by
    its mean nor keep its float32 residual, so their distances are rounded at the
    data's own magnitude (``docs/fp32_policy.md`` §3.4). On data with a large
    constant offset — Unix timestamps round to 128-second steps, UTM northings to
    half-metre ones — that can change which points fall inside the radius;
    subtract the offset before fitting, or use stock scikit-learn. Being
    scikit-learn's code, they also take a sparse query in both layers, as
    scikit-learn does, recorded as an ordinary inference call rather than a
    fallback. Stated rather than left for a user to discover from a profiler; the
    ancestor inherited the same methods and never mentioned it.

    With at most 15 features scikit-learn's ``algorithm="auto"`` builds a
    KD-tree, which answered a single-row query faster than an MLX call at every
    size measured up to 12 features, and at 2-5 features every batch. Under
    ``patch_sklearn()`` those fits, and any other with fewer than 16 features, are
    handed to scikit-learn. Imported directly, this class always searches by brute
    force.

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

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Find the neighbors within ``radius`` of each point.

        scikit-learn's implementation, run on the fitted state. See the class
        Notes for what that means for speed and precision; the parameters and
        return values are scikit-learn's.
        """
        check_is_fitted(self)
        if self._fitted_backend("radius_neighbors") == "sklearn":
            self._note_inference("radius_neighbors")
            return super().radius_neighbors(X, radius, return_distance, sort_results)

        if X is not None and is_sparse(X):
            X = _sparse_query_for_sklearn(X, self._fit_X.dtype)
        elif X is not None:
            # The training copy is float32. scikit-learn validates the query
            # without a dtype, so a float64 query stayed float64, its Cython
            # radius search requires both sides to share a dtype, and it fell to
            # a chunked pairwise-distance path about 4.5x slower than stock. The
            # range check runs before that cast, so a float64 value float32 cannot
            # hold is an UnsupportedInputError rather than scikit-learn's generic
            # "contains infinity" about a value the caller never wrote. The cast
            # keeps a DataFrame a DataFrame: scikit-learn validates the query
            # itself, and a bare array made it warn that a correctly named query
            # had no feature names.
            X = _reject_string_data(X)
            check_float32_range(X, fallback_advice=False)
            X = _as_float32_query(X)
        shape = self._fit_X.shape if X is None else _safe_shape(X)
        # scikit-learn's host code on the fitted state: no MLX device is involved,
        # so a GPU-fitted model needs none to answer this.
        self._note_inference("radius_neighbors", shape=shape, host=True, compute="scikit-learn")
        return super().radius_neighbors(X, radius, return_distance, sort_results)


def _as_float32_query(X):
    """A dense query cast to float32 for scikit-learn's radius search, container kept.

    A pandas frame of NumPy numeric columns is cast with ``astype``, which keeps
    its column names for scikit-learn's feature-name check. A frame from another
    library, nullable or object columns, and complex data are handed over as
    given: scikit-learn validates them in its own words and answers correctly,
    only off the matched-dtype fast path.
    """
    if _is_pandas(X):
        numeric = getattr(X, "ndim", 0) == 2 and all(
            isinstance(dtype, np.dtype) and dtype.kind in "biuf" for dtype in X.dtypes
        )
        return X.astype(NUMPY_FLOAT) if numeric else X
    if hasattr(X, "columns"):
        return X  # another library's frame: np.asarray would drop its column names
    try:
        array = np.asarray(X)
    except (TypeError, ValueError):
        return X  # not an array at all: scikit-learn's validation reports it
    if array.dtype.kind not in "biuf":
        return array
    return array.astype(NUMPY_FLOAT, order="C", copy=False)
