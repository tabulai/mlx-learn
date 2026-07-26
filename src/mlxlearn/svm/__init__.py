"""Support-vector estimators, accelerated with MLX.

The exact C-SVC dual and nothing else. There is no random-feature approximation
and no subsampled surrogate here, and there will not be one under a scikit-learn
estimator name: an approximate ``SVC`` returns different support vectors,
different decision values and different predictions from the class it claims to
replace, and no amount of speed makes that the same estimator.
"""

from __future__ import annotations

from ._classes import SVC

__all__ = ["SVC"]
