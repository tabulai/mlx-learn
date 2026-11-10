"""Single-pass Metal kernels for PCA input validation, column means, and total variance."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["pca_stats", "prepare_pca_centered"]

_pca_stats_kernel = mx.fast.metal_kernel(
    name="mlxlearn_pca_stats",
    input_names=["X"],
    output_names=["parts_sum", "parts_sq", "flags"],
    source=r"""
uint t=thread_position_in_threadgroup.x,lane=t%32,sg=t/32;
uint cb=threadgroup_position_in_grid.x,j=cb*32+lane;
uint b=threadgroup_position_in_grid.y;
float anchor=j<D?X[j]*.25f:0.0f;
float acc_sum=0.0f,acc_sq=0.0f;
bool finite=true;
if(j<D){
 ulong r0=ulong(b)*ROWS_PER_BLOCK+sg;
 ulong r1=min(ulong(N),(ulong(b)+1)*ROWS_PER_BLOCK);
 for(ulong row=r0;row<r1;row+=4){
  float val=X[row*D+j];
  finite=finite && isfinite(val);
  float diff=val*.25f-anchor;
  acc_sum+=diff*(1.0f/N);
  acc_sq+=diff*diff;
 }
}
threadgroup float sums[128],sqs[128];
threadgroup uint valid[4];
sums[t]=acc_sum;
sqs[t]=acc_sq;
bool good=simd_all(finite);
if(lane==0)valid[sg]=uint(good);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && j<D){
 parts_sum[ulong(b)*D+j]=(sums[lane]+sums[lane+32])+(sums[lane+64]+sums[lane+96]);
 parts_sq[ulong(b)*D+j]=(sqs[lane]+sqs[lane+32])+(sqs[lane+64]+sqs[lane+96]);
}
if(t==0)flags[ulong(b)*COL_BLOCKS+cb]=valid[0]&valid[1]&valid[2]&valid[3];
""",
)

_pca_center_copy_kernel = mx.fast.metal_kernel(
    name="mlxlearn_pca_center_copy",
    input_names=["X", "center"],
    output_names=["owned"],
    source=r"""
uint t=thread_position_in_threadgroup.x;
ulong block=ulong(threadgroup_position_in_grid.x)+
    ulong(threadgroup_position_in_grid.y)*BLOCK_COLUMNS;
ulong size=ulong(N)*D;
for(uint k=0;k<4;++k){
 ulong i=block*1024+t+k*256;
 if(i<size){owned[i]=X[i]-center[i%D];}
}
""",
)


@mx.compile
def _run_pca_stats(X: mx.array, rows_per_block: int) -> tuple[mx.array, mx.array, mx.array]:
    n, d = X.shape
    nb = (n + rows_per_block - 1) // rows_per_block
    cols = (d + 31) // 32
    parts_sum, parts_sq, flags = _pca_stats_kernel(
        inputs=[X],
        template=[
            ("N", n),
            ("D", d),
            ("COL_BLOCKS", cols),
            ("ROWS_PER_BLOCK", rows_per_block),
        ],
        grid=(cols * 128, nb, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nb, d), (nb, d), (nb, cols)],
        output_dtypes=[mx.float32, mx.float32, mx.uint32],
    )
    mean_diff = mx.sum(parts_sum, axis=0)
    bound = 3.4028234663852886e38 * 0.25
    mean = mx.clip(X[0] * 0.25 + mean_diff, -bound, bound) * 4.0
    col_ss = mx.maximum((mx.sum(parts_sq, axis=0) - float(n) * (mean_diff * mean_diff)) * 16.0, 0.0)
    return mean, col_ss, mx.all(flags != 0)


def pca_stats(X: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Compute column mean, per-column centered sum of squares, and finiteness in one GPU pass.

    Parameters
    ----------
    X : mx.array
        Nonempty 2D float32 array of shape ``(n_samples, n_features)``.

    Returns
    -------
    mean : mx.array of shape ``(n_features,)``
    col_ss : mx.array of shape ``(n_features,)``
        Per-column sum of squared deviations from ``mean``.
    finite : mx.array of shape ``()``, dtype bool
        True if and only if every element of ``X`` is finite.
    """
    if X.ndim != 2 or min(X.shape) == 0 or X.dtype != mx.float32:
        raise ValueError("pca_stats requires a nonempty 2D float32 array")
    n = X.shape[0]
    rows_per_block = 256 if n <= 262_144 else 1024
    return _run_pca_stats(X, rows_per_block)


def prepare_pca_centered(X: mx.array, mean: mx.array) -> mx.array:
    """Return a newly allocated centered float32 matrix ``X - mean`` on the GPU."""
    if X.ndim != 2 or min(X.shape) == 0 or X.dtype != mx.float32:
        raise ValueError("prepare_pca_centered requires a nonempty 2D float32 array")
    n, d = X.shape
    blocks = (X.size + 1023) // 1024
    columns = min(blocks, 65535)
    return _pca_center_copy_kernel(
        inputs=[X, mean],
        template=[
            ("N", n),
            ("D", d),
            ("BLOCK_COLUMNS", columns),
        ],
        grid=(columns * 256, (blocks + columns - 1) // columns, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[X.shape],
        output_dtypes=[mx.float32],
    )[0]
