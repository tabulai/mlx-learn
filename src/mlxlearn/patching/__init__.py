"""Layer 2: the patching shell.

The only package in mlxlearn permitted anywhere near scikit-learn's namespace, and
the only one where ``_mlxlearn_allow_fallback`` is true. Everything numerical
lives in Layer 1 and stays fully usable and testable with this package unimported.

``tools/compliance.py`` enforces the quarantine: a private ``sklearn._*`` import
anywhere else fails CI, and even here it must be on an explicit allowlist. As of
0.1.0a2 that allowlist is empty.
"""

from __future__ import annotations

from ._patch import (
    get_patch_map,
    is_patched,
    patch_names,
    patch_sklearn,
    patched_estimators,
    unpatch_sklearn,
)
from ._registry import ESTIMATOR_REGISTRY, EstimatorSpec

__all__ = [
    "patch_sklearn",
    "unpatch_sklearn",
    "is_patched",
    "get_patch_map",
    "patch_names",
    "patched_estimators",
    "ESTIMATOR_REGISTRY",
    "EstimatorSpec",
]
