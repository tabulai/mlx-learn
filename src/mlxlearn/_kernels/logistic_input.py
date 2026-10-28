"""Validate a borrowed design while preparing owned, optionally centered GPU storage.

One pass over the borrowed input writes the owned copy, checks that every
original value is finite and, for binary fits, accumulates the solver's
initial gradient and curvature diagonal from the actual owned values. With
an intercept the copy is centered first; the mean is computed from the
original values in a separate pass, with a shifted, scaled sum that cannot
overflow and preserves constant columns.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["prepare_centered"]

_center_stats = mx.fast.metal_kernel(
    name="mlxlearn_logistic_center_stats",
    input_names=["X"],
    output_names=["parts"],
    source=r"""
uint t=thread_position_in_threadgroup.x,lane=t%32,sg=t/32;
uint j=threadgroup_position_in_grid.x*32+lane;
uint b=threadgroup_position_in_grid.y;
float anchor=j<D?X[j]*.25f:0.0f,acc=0.0f;
if(j<D){for(ulong row=ulong(b)*256+sg;row<min(ulong(N),(ulong(b)+1)*256);row+=4){
 acc+=(X[row*D+j]*.25f-anchor)*(1.0f/N);
}}
threadgroup float sums[128];
sums[t]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && j<D)parts[ulong(b)*D+j]=(sums[lane]+sums[lane+32])+(sums[lane+64]+sums[lane+96]);
""",
)

_center_copy = mx.fast.metal_kernel(
    name="mlxlearn_logistic_center_copy",
    input_names=["X", "center"],
    output_names=["owned", "flags"],
    source=r"""
uint t=thread_position_in_threadgroup.x;
ulong block=ulong(threadgroup_position_in_grid.x)+
    ulong(threadgroup_position_in_grid.y)*BLOCK_COLUMNS;
ulong size=ulong(N)*D;
bool finite=true;
for(uint k=0;k<4;++k){
 ulong i=block*1024+t+k*256;
 if(i<size){float value=X[i];finite=finite && isfinite(value);owned[i]=CENTER?value-center[i%D]:value;}
}
threadgroup uint valid[8];
bool good=simd_all(finite);
if((t%32)==0)valid[t/32]=uint(good);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(t==0 && block<BLOCKS){uint result=1;for(uint s=0;s<8;++s)result&=valid[s];flags[block]=result;}
""",
)

_center_initial = mx.fast.metal_kernel(
    name="mlxlearn_logistic_center_initial",
    input_names=["X", "center", "y", "sw"],
    output_names=["owned", "parts", "flags"],
    source=r"""
// The row partition and accumulation order define the native solver's
// initial statistics. Every feature value is copied and checked, including
// rows with zero sample weight; column D, when present, is the intercept.
uint tid=thread_position_in_threadgroup.x;
uint cb=threadgroup_position_in_grid.x,col=cb*32+tid%32;
uint block=threadgroup_position_in_grid.y,sg=tid/32;
float grad=0,diag=0;
bool finite=true;
if(col<P){for(ulong row=ulong(block)*256+sg;row<min(ulong(N),(ulong(block)+1)*256);row+=4){
 float x=1.0f;
 if(col<D){ulong i=row*D+col;float value=X[i];finite=finite && isfinite(value);x=CENTER?value-center[col]:value;owned[i]=x;}
 float weight=sw[row];
 if(weight>0){grad+=x*(.5f-y[row])*weight;diag+=.25f*x*x*weight;}
}}
threadgroup float gp[128],hp[128];
threadgroup uint valid[4];
gp[tid]=grad;hp[tid]=diag;
bool good=simd_all(finite);
if(tid%32==0)valid[sg]=uint(good);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && col<P){
 parts[ulong(block)*P+col]=gp[tid]+gp[tid+32]+gp[tid+64]+gp[tid+96];
 parts[ulong(B)*P+ulong(block)*P+col]=hp[tid]+hp[tid+32]+hp[tid+64]+hp[tid+96];
}
if(tid==0)flags[ulong(block)*COL_BLOCKS+cb]=valid[0]&valid[1]&valid[2]&valid[3];
""",
)


@mx.compile
def _column_center(X):
    n, d = X.shape
    nb, cols = (n + 255) // 256, (d + 31) // 32
    parts = _center_stats(
        inputs=[X],
        template=[("N", n), ("D", d)],
        grid=(cols * 128, nb, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nb, d)],
        output_dtypes=[mx.float32],
    )[0]
    # Quarter-scaled differences cannot exceed half the float32 range. Clipping
    # the rounded mean before restoring its scale also protects finite constants.
    bound = 3.4028234663852886e38 * 0.25
    return mx.clip(X[0] * 0.25 + mx.sum(parts, axis=0), -bound, bound) * 4.0


def prepare_centered(X, y=None, sw=None, *, center=True, intercept=None):
    """Return lazy owned data, its applied center, finiteness, and initial parts.

    ``X`` must be a nonempty float32 MLX matrix. With ``center`` the owned
    copy is ``X`` minus its per-column mean and that mean is returned; without
    it the copy holds the original values and the returned center is ``None``.
    When ``y`` is supplied, it contains binary labels and the returned
    unnormalized initial parts have shape ``(2, ceil(n / 256), p)``; with
    ``intercept`` (by default, exactly when centering) ``p = d + 1`` and the
    last column is the intercept, otherwise ``p = d``. Without labels, the
    initial-parts result is ``None``.

    The caller must evaluate every returned array before releasing or modifying
    borrowed input storage. Finiteness describes the original input, not the
    centered differences, matching ordinary input validation.
    """
    if X.ndim != 2 or min(X.shape) == 0 or X.dtype != mx.float32:
        raise ValueError("prepare_centered requires a nonempty float32 matrix")
    if intercept is None:
        intercept = center
    n, d = X.shape
    shift = _column_center(X) if center else mx.zeros(d, dtype=X.dtype)
    if y is None:
        blocks = (X.size + 1023) // 1024
        columns = min(blocks, 65535)
        owned, flags = _center_copy(
            inputs=[X, shift],
            template=[("N", n), ("D", d), ("CENTER", int(center)),
                      ("BLOCKS", blocks), ("BLOCK_COLUMNS", columns)],
            grid=(columns * 256, (blocks + columns - 1) // columns, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[X.shape, (blocks,)],
            output_dtypes=[mx.float32, mx.uint32],
        )
        return owned, (shift if center else None), mx.all(flags != 0), None
    if y.shape != (n,) or (sw is not None and sw.shape != (n,)):
        raise ValueError("labels and sample weights must match the design's rows")
    if sw is None:
        sw = mx.ones(n, dtype=X.dtype)
    p, nb = d + int(intercept), (n + 255) // 256
    cols = (p + 31) // 32
    owned, parts, flags = _center_initial(
        inputs=[X, shift, y, sw],
        template=[("N", n), ("D", d), ("P", p), ("B", nb), ("CENTER", int(center)),
                  ("COL_BLOCKS", cols)],
        grid=(cols * 128, nb, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[X.shape, (2, nb, p), (nb, cols)],
        output_dtypes=[mx.float32, mx.float32, mx.uint32],
    )
    return owned, (shift if center else None), mx.all(flags != 0), parts
