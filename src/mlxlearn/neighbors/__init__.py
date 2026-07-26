"""Nearest-neighbor estimators, accelerated with MLX.

Exact brute-force Euclidean search on dense data. There is no approximate index
here and there will not be one under a scikit-learn estimator name: an
``ANN``-backed ``KNeighborsClassifier`` would silently return different answers
from the class it claims to replace.
"""

from __future__ import annotations

from ._classification import KNeighborsClassifier
from ._regression import KNeighborsRegressor
from ._unsupervised import NearestNeighbors

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor", "NearestNeighbors"]
