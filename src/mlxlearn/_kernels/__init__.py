"""MLX kernels: the array-level operations estimators are built from.

Private. Nothing here knows what an estimator is; everything takes and returns
arrays and typed plain-data objects.
"""

from __future__ import annotations

from .distance import NeighborIndex, build_index, resolve_blocks

__all__ = ["NeighborIndex", "build_index", "resolve_blocks"]
