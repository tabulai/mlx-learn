"""Two-query direct scans reuse training loads with independent Kahan sums.

Internal launch helper; shape and dtype validation belongs to fused_neighbors.
Each query retains the same per-pair arithmetic and distance/index ordering as
the one-query scan, including odd query batches and sliced self queries. Feature
counts divisible by four use packed loads without requiring 16-byte alignment;
k=5 uses named candidate scalars rather than a dynamically indexed array.
"""

from functools import lru_cache

import mlx.core as mx

from mlxlearn_device._neighbors_fused import _HEADER

from ._neighbors_precision import PRECISION_HEADER

# Four source variants cover k=5 scalar state and general k, each with packed
# or scalar feature loads. This bounds generated source and kernel-factory state.
from ._neighbors_registers import paired_scan_source

_SCAN = paired_scan_source(registers=False, packed=False)


@lru_cache(maxsize=4)
def _scan_kernel(registers=False, packed=False):
    return mx.fast.metal_kernel(
        name=f"mlxlearn_paired_neighbors_scan_{int(registers)}_{int(packed)}",
        input_names=["train", "queries", "shape"],
        output_names=["out_distances", "out_indices"],
        source=paired_scan_source(registers=registers, packed=packed),
        header=_HEADER + PRECISION_HEADER,
    )


def paired_neighbors(
    train,
    queries,
    k,
    *,
    partitions,
    exclude_self=False,
    query_offset=0,
):
    from ._neighbors_screen import screen_neighbors

    if queries.shape[0] == 0:
        return mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32)
    return screen_neighbors(
        train, queries, int(k), partitions=partitions, paired=True,
        exclude_self=exclude_self, query_offset=query_offset,
    )
