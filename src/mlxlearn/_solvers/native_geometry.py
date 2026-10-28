"""Launch geometry shared by the fused native logistic line-search kernels.

The MLX-driven solver step and the optional native command loop consume the
same speculative Metal kernel, so both must specialize it identically. This
module has no imports: ``mlxlearn_device`` may load it by path without
importing ``mlxlearn``. The one-pass Hessian-vector kernel in
``mlxlearn_device`` keeps its own table; it coincides with this one only
for 512 to 4096 features.
"""

from __future__ import annotations

__all__ = ["CLUSTER_MAX_FEATURES", "NARROW_MAX_FEATURES", "speculative_geometry"]

MAX_FUSED_FEATURES = 4096
LARGE_ROW_COUNT = 16384
# Rows up to this width run the four-rows-per-SIMD-group kernel; up to
# CLUSTER_MAX_FEATURES eight lanes share one row inside it.
NARROW_MAX_FEATURES = 128
CLUSTER_MAX_FEATURES = 32


def speculative_geometry(n: int, d: int) -> tuple[int, int, int, int]:
    """Return ``(groups, feature_groups, rows, blocks)`` for ``n`` rows and ``d`` features.

    Each SIMD group retains its feature slice in registers and accumulates
    across ``rows`` rows before writing partial gradients. Rows wider than 1024
    features span four SIMD groups to bound register pressure. Below 512
    features the per-row dot product is short, so on large designs eight
    SIMD groups per threadgroup keep enough rows in flight, except that up
    to 32 features four groups already saturate the bandwidth. Up to 128
    features the narrow kernels batch four rows per SIMD group, so
    ``feature_groups`` stays 1 and ``rows`` is a multiple of four. Designs
    below 16384 rows keep more, shorter row batches for launch occupancy.
    """
    if not 1 <= d <= MAX_FUSED_FEATURES:
        raise ValueError(f"native fusion requires 1 to {MAX_FUSED_FEATURES} features")
    if n >= LARGE_ROW_COUNT:
        if d <= CLUSTER_MAX_FEATURES:
            groups, feature_groups, rows = 4, 1, 32
        elif d < 512:
            groups, feature_groups, rows = 8, 1, 32
        elif d <= 1024:
            groups, feature_groups, rows = 4, 1, 32
        else:
            groups, feature_groups, rows = 8, 4, 32 if d <= 2048 else 64
    elif d < 512:
        groups, feature_groups, rows = 8, 1, 4
    elif d <= 1024:
        groups, feature_groups, rows = 2, 1, 16
    else:
        groups, feature_groups, rows = 8, 4, 8
    rows_per_block = (groups // feature_groups) * rows
    blocks = (n + rows_per_block - 1) // rows_per_block
    return groups, feature_groups, rows, blocks
