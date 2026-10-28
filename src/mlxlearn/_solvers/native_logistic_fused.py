"""One matrix pass for line search and its first candidate's gradient.

The selected candidate is still decided by the ordinary Armijo kernel. This
module computes the gradient for the first candidate speculatively; callers may
reuse it only when that candidate was actually accepted. Other accepted steps
retain the normal gradient evaluation, and final convergence is checked directly.

Each SIMD group retains its feature slice in registers and accumulates across
several rows before writing partial gradients. Wider rows span four SIMD groups
to bound register pressure. The shared workspace is reused for row dot products,
losses/intercepts, and feature gradients, never exceeding 32 KiB. The kernel is
feature-count agnostic: every binary fit of at most 4096 features uses it, with
the launch geometry chosen by :mod:`native_geometry`.

Rows of at most 128 features use two further kernels with the same contract.
One SIMD group per row leaves most lanes idle on such short dot products, so
a SIMD group carries four rows at once and every lane evaluates one
line-search candidate of its row; at most 32 features, eight lanes share a
row outright. Above 128 features the register copy of four rows no longer
pays for itself and the wide kernel's one-row layout already streams at the
memory ceiling.
"""

from __future__ import annotations

import mlx.core as mx

from .native_geometry import CLUSTER_MAX_FEATURES, NARROW_MAX_FEATURES, speculative_geometry

# Imported only when native_logistic.step first needs the kernel. That module
# is fully initialized before this module is loaded.
from .native_logistic import _REDUCE

__all__ = ["forward_gradient"]

_speculative = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_speculative",
    header=_REDUCE,
    input_names=["X", "direction", "raw", "y", "sw", "stats", "denom"],
    output_names=["dr", "partial", "gradient_parts"],
    source=r"""

uint tid=thread_position_in_threadgroup.x,lane=tid%32,sg=tid/32;
uint group=threadgroup_position_in_grid.x;
constexpr uint V=(D+32*FEATURE_GROUPS-1)/(32*FEATURE_GROUPS);
constexpr uint CHUNK=V*32,DPAD=CHUNK*FEATURE_GROUPS;
constexpr uint ROW_GROUPS=GROUPS/FEATURE_GROUPS;
constexpr uint WS_GRAD=ROW_GROUPS*DPAD,WS_LOSS=GROUPS*(CAND+1);
constexpr uint SCRATCH=WS_GRAD>WS_LOSS?WS_GRAD:WS_LOSS;
uint fs=sg%FEATURE_GROUPS,rs=sg/FEATURE_GROUPS;
threadgroup float scratch[SCRATCH];
float accumulator[V];for(uint k=0;k<V;++k)accumulator[k]=0;
float local_loss=0,local_bias=0;
float alpha0=stats[8];bool active=stats[1]==0;
for(uint rr=0;rr<ROWS;++rr){
 uint row=group*(ROW_GROUPS*ROWS)+rr*ROW_GROUPS+rs;
 float values[V],z=0;
 if(row<N && active){for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;values[k]=j<D?X[ulong(row)*D+j]:0.0f;if(j<D)z=fma(values[k],direction[j],z);}}
 else{for(uint k=0;k<V;++k)values[k]=0;}
 z=simd_sum(z);
 if(FEATURE_GROUPS>1){
  if(lane==0)scratch[sg]=z;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  z=0;for(uint f=0;f<FEATURE_GROUPS;++f)z+=scratch[rs*FEATURE_GROUPS+f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(P>D && active)z+=direction[D];
 if(lane==0 && fs==0 && row<N)dr[row]=z;
 float first_residual=0;
 if(row<N && active && sw[row]>0){
  float sign=1-2*y[row],u=sign*raw[row];
  if(fs==0 && lane<CAND){
   float alpha=alpha0*exp2(-float(lane)),dz=sign*z*alpha;
   float diff=abs(dz)<.5f?log1p_stable(sigmoid_stable(u)*expm1_small(dz)):softplus_stable(u+dz)-softplus_stable(u);
   local_loss+=sw[row]*diff/denom;
  }
  if(lane==0){first_residual=sw[row]*sign*sigmoid_stable(sign*fma(alpha0,z,raw[row]));if(fs==0)local_bias+=first_residual;}
 }
 first_residual=simd_shuffle(first_residual,ushort(0));
 for(uint k=0;k<V;++k)accumulator[k]=fma(values[k],first_residual,accumulator[k]);
}
if(lane<CAND)scratch[sg*(CAND+1)+lane]=local_loss;
if(lane==0)scratch[sg*(CAND+1)+CAND]=local_bias;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){
 if(lane<CAND){float total=0;for(uint s=0;s<GROUPS;++s)total+=scratch[s*(CAND+1)+lane];partial[ulong(lane)*B+group]=total;}
 if(lane==0 && P>D){float total=0;for(uint s=0;s<GROUPS;++s)total+=scratch[s*(CAND+1)+CAND];gradient_parts[ulong(group)*P+D]=total;}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;scratch[rs*DPAD+j]=accumulator[k];}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(rs==0){for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;if(j<D){float total=0;for(uint r=0;r<ROW_GROUPS;++r)total+=scratch[r*DPAD+j];gradient_parts[ulong(group)*P+j]=total;}}}

""",
)


# Both narrow kernels share these fragments. Prologue: a converged or failed
# status leaves every output defined without reading X. Epilogue: candidate
# and bias partials are reduced across the eight-lane row clusters, then
# across SIMD groups, next to the feature gradient partials.
_NARROW_PROLOGUE = r"""
constexpr uint Q=(D+3)/4;
constexpr uint RB=4;
constexpr uint LC=(CAND+7)/8;
uint tid=thread_position_in_threadgroup.x,lane=tid%32,sg=tid/32;
uint group=threadgroup_position_in_grid.x,r=lane/8,k=lane%8;
threadgroup float sg_g[GROUPS*DPAD];
threadgroup float sg_l[GROUPS*CAND];
threadgroup float sg_b[GROUPS];
if(stats[1]!=0){
 for(uint c=tid;c<CAND;c+=GROUPS*32)partial[ulong(c)*B+group]=0.0f;
 for(uint j=tid;j<P;j+=GROUPS*32)gradient_parts[ulong(group)*P+j]=0.0f;
 for(uint i=tid;i<GROUPS*ROWS;i+=GROUPS*32){ulong row=ulong(group)*GROUPS*ROWS+i;if(row<N)dr[row]=0.0f;}
 return;
}
ulong base=(ulong(group)*GROUPS+sg)*ROWS;
float alpha0=stats[8];
float lacc[LC];for(uint i=0;i<LC;++i)lacc[i]=0.0f;
float bacc=0.0f;
"""

# Candidate c of row r is evaluated on lane 8*r+(c%8); lane 8*r keeps the
# first candidate's residual for the gradient accumulators.
_NARROW_CANDIDATES = r"""
 float res=0.0f;
 if(ok && sw[row]>0.0f){float w=sw[row],sign=1.0f-2.0f*y[row],u=sign*raw[row];
  for(uint c=k,i=0;c<CAND;c+=8,++i){float alpha=alpha0*exp2(-float(c)),dz=sign*zr*alpha;
   float diff=abs(dz)<0.5f?log1p_stable(sigmoid_stable(u)*expm1_small(dz)):softplus_stable(u+dz)-softplus_stable(u);
   lacc[i]+=w*diff/denom;}
  if(k==0){res=w*sign*sigmoid_stable(sign*fma(alpha0,zr,raw[row]));bacc+=res;}
 }
"""

_NARROW_EPILOGUE = r"""
for(uint i=0;i<LC;++i){float t=lacc[i];t+=simd_shuffle_xor(t,ushort(8));t+=simd_shuffle_xor(t,ushort(16));
 uint c=k+8*i;if(r==0 && c<CAND)sg_l[sg*CAND+c]=t;}
{float t=bacc;t+=simd_shuffle_xor(t,ushort(8));t+=simd_shuffle_xor(t,ushort(16));if(lane==0)sg_b[sg]=t;}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){
 if(lane<CAND){float t=0.0f;for(uint s=0;s<GROUPS;++s)t+=sg_l[s*CAND+lane];partial[ulong(lane)*B+group]=t;}
 if(lane==0 && P>D){float t=0.0f;for(uint s=0;s<GROUPS;++s)t+=sg_b[s];gradient_parts[ulong(group)*P+D]=t;}
 for(uint j=lane;j<D;j+=32){float t=0.0f;for(uint s=0;s<GROUPS;++s)t+=sg_g[s*DPAD+j];gradient_parts[ulong(group)*P+j]=t;}
}
"""

# Four consecutive scalar loads: the compiler vectorizes them, and unlike a
# packed_float4 cast they compile whichever address space MLX binds X in.
_NARROW_QUAD = r"""
  if(j+4<=D)x=float4(X[o],X[o+1],X[o+2],X[o+3]);
  else x=float4(X[o],j+1<D?X[o+1]:0.0f,j+2<D?X[o+2]:0.0f,0.0f);
"""

_cluster = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_cluster",
    header=_REDUCE,
    input_names=["X", "direction", "raw", "y", "sw", "stats", "denom"],
    output_names=["dr", "partial", "gradient_parts"],
    source=r"""
// Rows of at most 32 features: lane 8*r+q holds quad q of row r of its SIMD
// group's four-row batch, so the row dot product closes with three shuffles
// inside the eight-lane cluster and all 32 lanes evaluate candidates.
constexpr uint DPAD=32;
""" + _NARROW_PROLOGUE + r"""
float4 dir=float4(0.0f);
if(k<Q){uint j=4*k;dir=float4(direction[j],j+1<D?direction[j+1]:0.0f,j+2<D?direction[j+2]:0.0f,j+3<D?direction[j+3]:0.0f);}
float4 acc=float4(0.0f);
for(uint b=0;b<ROWS;b+=RB){
 ulong row=base+b+r;bool ok=row<N;
 float4 x=float4(0.0f);
 if(ok && k<Q){uint j=4*k;ulong o=row*D+j;
""" + _NARROW_QUAD + r"""
 }
 float zr=dot(x,dir);
 zr+=simd_shuffle_xor(zr,ushort(1));zr+=simd_shuffle_xor(zr,ushort(2));zr+=simd_shuffle_xor(zr,ushort(4));
 if(P>D)zr+=direction[D];
 if(k==0 && ok)dr[row]=zr;
""" + _NARROW_CANDIDATES + r"""
 acc=fma(x,simd_shuffle(res,ushort(8*r)),acc);
}
acc.x+=simd_shuffle_xor(acc.x,ushort(8));acc.x+=simd_shuffle_xor(acc.x,ushort(16));
acc.y+=simd_shuffle_xor(acc.y,ushort(8));acc.y+=simd_shuffle_xor(acc.y,ushort(16));
acc.z+=simd_shuffle_xor(acc.z,ushort(8));acc.z+=simd_shuffle_xor(acc.z,ushort(16));
acc.w+=simd_shuffle_xor(acc.w,ushort(8));acc.w+=simd_shuffle_xor(acc.w,ushort(16));
if(r==0){uint j=4*k;sg_g[sg*DPAD+j]=acc.x;sg_g[sg*DPAD+j+1]=acc.y;sg_g[sg*DPAD+j+2]=acc.z;sg_g[sg*DPAD+j+3]=acc.w;}
""" + _NARROW_EPILOGUE,
)

_rows = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_rows",
    header=_REDUCE,
    input_names=["X", "direction", "raw", "y", "sw", "stats", "denom"],
    output_names=["dr", "partial", "gradient_parts"],
    source=r"""
// Rows of 33 to 128 features: every lane holds one quad of each of its SIMD
// group's four rows; the dot products close with simd_sum one row at a time,
// then candidate c of row r is evaluated on lane 8*r+(c%8).
constexpr uint DPAD=128;
""" + _NARROW_PROLOGUE + r"""
float4 dir=float4(0.0f);
if(lane<Q){uint j=4*lane;dir=float4(direction[j],j+1<D?direction[j+1]:0.0f,j+2<D?direction[j+2]:0.0f,j+3<D?direction[j+3]:0.0f);}
float4 acc=float4(0.0f);
for(uint b=0;b<ROWS;b+=RB){
 float4 v[RB];float z[RB];
 for(uint rr=0;rr<RB;++rr){ulong rw=base+b+rr;bool okr=rw<N;
  float4 x=float4(0.0f);
  if(okr && lane<Q){uint j=4*lane;ulong o=rw*D+j;
""" + _NARROW_QUAD + r"""
  }
  v[rr]=x;
  float t=simd_sum(dot(x,dir));
  if(P>D)t+=direction[D];
  if(lane==0 && okr)dr[rw]=t;
  z[rr]=t;}
 ulong row=base+b+r;bool ok=row<N;
 float zr=r==0?z[0]:(r==1?z[1]:(r==2?z[2]:z[3]));
""" + _NARROW_CANDIDATES + r"""
 for(uint rr=0;rr<RB;++rr)acc=fma(v[rr],simd_shuffle(res,ushort(8*rr)),acc);
}
{uint j=4*lane;sg_g[sg*DPAD+j]=acc.x;sg_g[sg*DPAD+j+1]=acc.y;sg_g[sg*DPAD+j+2]=acc.z;sg_g[sg*DPAD+j+3]=acc.w;}
""" + _NARROW_EPILOGUE,
)


@mx.compile
def forward_gradient(X, direction, raw, y, sw, stats, denom, candidates):
    """Return projections, candidate loss changes, and the first sample gradient.

    The gradient is normalized by ``denom`` and excludes regularization.
    ``stats[8]`` supplies the initial step, including a resumed line search.
    All inputs are explicit; no training buffer is retained between fits.
    """
    n, d = X.shape
    p = direction.size
    if not 1 <= candidates <= 32:
        raise ValueError("native fusion requires 1 to 32 candidates")
    if p not in (d, d + 1):
        raise ValueError("direction must hold one entry per feature plus at most an intercept")
    groups, feature_groups, rows, blocks = speculative_geometry(n, d)
    template = [
        ("N", n),
        ("D", d),
        ("P", p),
        ("B", blocks),
        ("GROUPS", groups),
        ("ROWS", rows),
        ("CAND", candidates),
    ]
    if d <= CLUSTER_MAX_FEATURES:
        kernel = _cluster
    elif d <= NARROW_MAX_FEATURES:
        kernel = _rows
    else:
        kernel = _speculative
        template.append(("FEATURE_GROUPS", feature_groups))
    dr, partial, gradient_parts = kernel(
        inputs=[X, direction, raw, y, sw, stats, denom],
        template=template,
        grid=(blocks * groups * 32, 1, 1),
        threadgroup=(groups * 32, 1, 1),
        output_shapes=[(n,), (candidates, blocks), (blocks, p)],
        output_dtypes=[mx.float32] * 3,
    )
    return dr, mx.sum(partial, axis=1), mx.sum(gradient_parts, axis=0) / denom
