"""NumPy-free, bounded inexact Newton systems on a native Metal command queue."""
from __future__ import annotations

import ast
import functools
import math
import re
from pathlib import Path

import mlx.core as mx

from ._hessian_vector import _geometry
from ._native import metal_loop

_MAX_WORKSPACE_BYTES = 64 * 1024 * 1024


def _workspace_supported(n, d):
    """Whether the shared Hessian-product partials fit the native CG budget."""
    _, _, _, blocks = _geometry(n, d)
    return blocks * (d + 1) * 4 <= _MAX_WORKSPACE_BYTES


@functools.lru_cache(maxsize=16)
def _source(n, d):
    p = d + 1
    groups, feature_groups, rows, blocks = _geometry(n, d)
    # Share the single-pass arithmetic with the separately tested MLX HVP.
    tree = ast.parse(Path(__file__).with_name("_hessian_vector.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_one_pass_hvp" for t in node.targets
        ):
            body = next(ast.literal_eval(kw.value) for kw in node.value.keywords if kw.arg == "source")
            break
    else:
        raise RuntimeError("Hessian-vector source unavailable")
    definitions = {"N": n, "D": d, "P": p, "B": blocks, "GROUPS": groups,
                   "FEATURE_GROUPS": feature_groups, "ROWS": rows}
    header = "#include <metal_stdlib>\nusing namespace metal;\n"
    header += "".join(f"#define {key} {value}\n" for key, value in definitions.items())
    header += r"""
inline float cg_sum(float value, threadgroup float* scratch, uint tid) {
 value=simd_sum(value);
 if(tid%32==0)scratch[tid/32]=value;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 value=scratch[0]+scratch[1]+scratch[2]+scratch[3];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 return value;
}
kernel void cg_initialize(device const float* gradient [[buffer(0)]],
 device float* x [[buffer(1)]], device float* r [[buffer(2)]],
 device float* direction [[buffer(3)]], device float* meta [[buffer(4)]],
 constant float& forcing [[buffer(5)]], uint t [[thread_position_in_threadgroup]]) {
 threadgroup float scratch[4]; float rr=0;
 for(uint j=t;j<P;j+=128){float v=-gradient[j];x[j]=0;r[j]=v;direction[j]=v;rr=fma(v,v,rr);}
 rr=cg_sum(rr,scratch,t);
 if(t==0){meta[0]=!isfinite(rr)?3:rr==0?1:0;meta[1]=0;meta[2]=rr;meta[3]=rr;
  meta[4]=0;meta[5]=max(rr*forcing*forcing,1e-20f);}
}
kernel void cg_reduce(device const float* parts [[buffer(0)]],
 device const float* direction [[buffer(1)]], device const float* reg [[buffer(2)]],
 device float* hd [[buffer(3)]], device const float* meta [[buffer(4)]],
 uint j [[thread_position_in_grid]]) {
 if(meta[0]!=0 || j>=P)return;
 float value=0;for(uint i=0;i<B;++i)value+=parts[ulong(i)*P+j];
 hd[j]=fma(reg[j],direction[j],value);
}
kernel void cg_update(device float* x [[buffer(0)]], device float* r [[buffer(1)]],
 device float* direction [[buffer(2)]],device const float* hd [[buffer(3)]],
 device float* meta [[buffer(4)]], uint t [[thread_position_in_threadgroup]]) {
 if(meta[0]!=0)return;
 threadgroup float scratch[4];
 float dot=0,pp=0;
 for(uint j=t;j<P;j+=128){dot=fma(direction[j],hd[j],dot);pp=fma(direction[j],direction[j],pp);}
 dot=cg_sum(dot,scratch,t);pp=cg_sum(pp,scratch,t);
 float rayleigh=dot/max(pp,1e-30f),oldrr=meta[2];
 bool invalid=!isfinite(dot)||!isfinite(pp)||!isfinite(oldrr)||!isfinite(rayleigh);
 if(invalid || dot<=0 || rayleigh<=meta[4]*(8*1.1920928955078125e-7f)){
  if(t==0)meta[0]=invalid?3:2;return;
 }
 float alpha=oldrr/dot,rr=0;
 // Separate multiplication and subtraction matches the residual of the
 // represented product; fast-math contraction is disabled for this pipeline.
 for(uint j=t;j<P;j+=128){x[j]=x[j]+alpha*direction[j];r[j]=r[j]-alpha*hd[j];rr=fma(r[j],r[j],rr);}
 rr=cg_sum(rr,scratch,t);
 float beta=rr/oldrr;
 for(uint j=t;j<P;j+=128)direction[j]=r[j]+beta*direction[j];
 if(t==0){meta[1]+=1;meta[2]=rr;meta[4]=max(meta[4],rayleigh);
  if(!isfinite(rr))meta[0]=3;else if(rr<=meta[5])meta[0]=1;}
}
kernel void cg_product(device const float* X [[buffer(0)]],
 device const float* curvature [[buffer(1)]],device const float* vector [[buffer(2)]],
 device float* parts [[buffer(3)]],device const float* meta [[buffer(4)]],
 uint3 thread_position_in_grid [[thread_position_in_grid]],
 uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
 uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
 if(meta[0]!=0)return;
"""
    return header + body + "\n}\n"


def solve_cg(X, curvature, gradient, regularization, *, forcing=0.01, max_iter=128):
    """Solve H direction = -gradient approximately, retaining float32 rank safety.

    Relative residual stopping avoids chasing operator roundoff; unresolved
    curvature truncates the solve. Neither is an outer convergence certificate.
    All arrays must be validated, finite, contiguous float32 GPU allocations;
    the caller must independently score and line-search the returned direction.
    The native code checks exact pipeline limits and retains DLPack owners until
    external work reaches a terminal state. Only scalar diagnostics leave GPU.
    """
    if metal_loop is None or not hasattr(metal_loop, "run_cg"):
        raise RuntimeError("native Metal CG extension unavailable")
    version = getattr(mx, "__version__", None)
    if not isinstance(version, str) or re.fullmatch(r"0\.32\.(?:0|[1-9]\d{0,2})", version) is None:
        raise RuntimeError("native CG supports MLX 0.32.x Metal DLPack only")
    if (not isinstance(X, mx.array) or X.ndim != 2 or X.dtype != mx.float32
        or not 1 <= X.shape[1] <= 4096 or not 1 <= X.shape[0] <= 2**31 - 1):
        raise ValueError("X must be nonempty float32 with at most 4096 features")
    n, d = X.shape
    p = d + 1
    for value, shape in ((curvature, (n,)), (gradient, (p,)), (regularization, (p,))):
        if not isinstance(value, mx.array) or value.shape != shape or value.dtype != mx.float32:
            raise ValueError("CG inputs must be matching float32 arrays")
    if (isinstance(forcing, bool) or not isinstance(forcing, (float, int))
        or not math.isfinite(forcing) or not 1e-4 <= forcing <= 0.5
        or type(max_iter) is not int or not 1 <= max_iter <= 512):
        raise ValueError("invalid CG forcing or iteration budget")
    groups, _, _, blocks = _geometry(n, d)
    if not _workspace_supported(n, d):
        raise ValueError("CG workspace exceeds 64 MiB")
    with mx.stream(mx.gpu):
        output = mx.zeros((p,), dtype=mx.float32)
        arrays = (X, curvature, gradient, regularization, output)
        mx.eval(*arrays)
        mx.synchronize()
        capsules = tuple(value.__dlpack__() for value in arrays)
        owners = (capsules, *arrays)
        metadata = metal_loop.run_cg(_source(n, d), capsules, n, d, p, blocks,
                                     groups, max_iter, float(forcing), owners)
    return output, metadata
