"""Native Metal binary logistic L-BFGS with a curvature-scaled initial metric.

The optimizer and its ring history remain on the GPU. Native Metal C++ kernels
fuse the two-loop recurrence, directional projection and Armijo loss differences,
accepted-step residuals, and gradient reductions. Host checks occur once per
bounded chunk, then the caller's direct objective certifies the final result.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mlx.core as mx
import numpy as np

from . import native_command
from .lbfgs import LBFGSConfig, LBFGSState, ObjectiveGradient
from .native_geometry import MAX_FUSED_FEATURES

__all__ = ["minimize_binary"]

_REDUCE = r"""
inline float sigmoid_stable(float z) {
    float e = exp(-abs(z));
    return z >= 0.0f ? 1.0f/(1.0f+e) : e/(1.0f+e);
}
inline float expm1_small(float x) {
    // Metal has no expm1 intrinsic. On |x| < 0.5, the omitted terms are below
    // one float32 ULP; factoring out x also preserves very small changes.
    return x*(1.0f+x*(0.5f+x*(1.0f/6.0f+x*(1.0f/24.0f+x*(1.0f/120.0f+x*(1.0f/720.0f+x*(1.0f/5040.0f+x/40320.0f)))))));
}
inline float log1p_stable(float x) {
    // The loss-difference path has x in (exp(-0.5)-1, exp(0.5)-1),
    // and softplus uses x in [0, 1]. In [-0.5, 1], r=x/(2+x) has
    // |r| <= 1/3; the omitted atanh term is below 1.1e-9 in absolute
    // value. This form retains x even when 1+x rounds to 1, and does not
    // depend on the compiler preserving a compensated log(1+x) expression.
    // Outside that interval (including near -1), use the standard logarithm.
    if (x < -0.5f || x > 1.0f) return log(1.0f+x);
    float r=x/(2.0f+x), z=r*r;
    float p=1.0f/15.0f;
    p=fma(z,p,1.0f/13.0f); p=fma(z,p,1.0f/11.0f);
    p=fma(z,p,1.0f/9.0f);  p=fma(z,p,1.0f/7.0f);
    p=fma(z,p,1.0f/5.0f);  p=fma(z,p,1.0f/3.0f);
    return (2.0f*r)*fma(z,p,1.0f);
}
inline float softplus_stable(float z) {
    return max(z,0.0f)+log1p_stable(exp(-abs(z)));
}
"""

_DIRECTION_REDUCE = r"""

inline float direction_sum(float value, uint t, threadgroup float* shared, uint width) {
 if(width==32)return simd_sum(value);
 value=simd_sum(value);
 if(t%32==0)shared[t/32]=value;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 value=0.0f;for(uint i=0;i<width/32;++i)value+=shared[i];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 return value;
}
inline float direction_max(float value, uint t, threadgroup float* shared, uint width) {
 if(width==32)return simd_max(value);
 value=simd_max(value);
 if(t%32==0)shared[t/32]=value;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 value=shared[0];for(uint i=1;i<width/32;++i)value=max(value,shared[i]);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 return value;
}
"""

# meta: accepted iterations, status (0 active,1 converged,2 search failure,
# 3 nonfinite), history count, next slot, previous accepted loss change,
# directional derivative, penalty linear and quadratic terms, alpha, loss
# evaluations, last accepted loss change, and running objective estimate.
# The direction stage sets alpha to the initial candidate; acceptance replaces
# it with the actual accepted step (zero on rejection).
_direction = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_direction",
    header=_REDUCE + _DIRECTION_REDUCE,
    input_names=[
        "w",
        "g",
        "oldw",
        "oldg",
        "hs",
        "hy",
        "hc",
        "meta",
        "reg",
        "tol",
        "h0inv",
        "min_curvature",
        "alpha_start",
        "center",
    ],
    output_names=["direction", "newhs", "newhy", "newhc", "stats"],
    source=r"""
// Distribute wide vector recurrences across SIMD groups to avoid spilling
// a long private q array. The scalar initial metric and ring order are unchanged.
uint t=thread_position_in_threadgroup.x;
threadgroup float shared[THREADS/32];
float alphas[M];
float q[(P+THREADS-1)/THREADS];
float localmax=0.0f, bad=0.0f, gg=0.0f;
for(uint k=0;k<(P+THREADS-1)/THREADS;++k){
    uint j=t+k*THREADS; float v=j<P?g[j]:0.0f; q[k]=v;
    float original=(P>D && j<D)?fma(center[j],g[D],v):v;
    localmax=max(localmax,abs(original));bad+=(isfinite(v) && isfinite(original))?0.0f:1.0f;
    gg+=j<P?v*v*h0inv[j]:0.0f;
}
float maxgrad=direction_max(localmax,t,shared,THREADS);
bad=direction_sum(bad,t,shared,THREADS);
gg=direction_sum(gg,t,shared,THREADS);
uint status=uint(meta[1]), count=uint(meta[2]), head=uint(meta[3]);
// Individual gradient components can be small while their combined objective
// improvement is material. Weighted fits and wide fits use a tighter two-step
// loss-progress test; this is a heuristic, not an objective-error bound.
float recent=abs(meta[10]);
if(meta[0]>=2)recent=0.5f*(recent+abs(meta[4]));
bool progress=(!WEIGHTED && (D<2048 || meta[0]==0)) || maxgrad==0 ||
    ((!WEIGHTED || meta[0]>=2) && recent<=tol*((WEIGHTED || D>=2048) ? 0.025f : 1.0f)*max(abs(meta[11]),1e-12f));
if(status==0){if(bad>0 || !isfinite(meta[11]))status=3;else if(maxgrad<=tol && progress)status=1;}
bool active=status==0;
for(uint j=t;j<M*P;j+=THREADS){newhs[j]=hs[j];newhy[j]=hy[j];}
for(uint j=t;j<M*2;j+=THREADS)newhc[j]=hc[j];
threadgroup_barrier(mem_flags::mem_device);
if(active && meta[0]>0){
    float sy=0,yy=0;
    for(uint j=t;j<P;j+=THREADS){float s=w[j]-oldw[j],y=g[j]-oldg[j];sy+=s*y;yy+=y*y*h0inv[j];}
    sy=direction_sum(sy,t,shared,THREADS);yy=direction_sum(yy,t,shared,THREADS);
    // The floor is relative to the pair's own metric norm: an absolute floor
    // rejects every valid pair once one feature's scale makes y'H0y small.
    if(isfinite(sy) && yy>0 && sy>min_curvature*yy){
        for(uint j=t;j<P;j+=THREADS){newhs[head*P+j]=w[j]-oldw[j];newhy[head*P+j]=g[j]-oldg[j];}
        if(t==0){newhc[2*head]=1.0f/sy;newhc[2*head+1]=sy/yy;}
        head=(head+1)%M;count=min(count+1,uint(M));
    }
}
threadgroup_barrier(mem_flags::mem_device);
if(active){
    for(uint a=0;a<count;++a){
        uint h=(head+M-1-a)%M;float dot=0;
        for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+THREADS*k;if(j<P)dot+=newhs[h*P+j]*q[k];}
        float alpha=direction_sum(dot,t,shared,THREADS)*newhc[2*h];
        alphas[a]=alpha;
        for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+THREADS*k;if(j<P)q[k]-=alpha*newhy[h*P+j];}
    }
    float gamma=count?newhc[2*((head+M-1)%M)+1]:1.0f;
    for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+THREADS*k;if(j<P)q[k]*=gamma*h0inv[j];}
    for(uint a=0;a<count;++a){
        uint h=(head+M-count+a)%M;float dot=0;
        for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+THREADS*k;if(j<P)dot+=newhy[h*P+j]*q[k];}
        float beta=direction_sum(dot,t,shared,THREADS)*newhc[2*h];
        float alpha=alphas[count-1-a];
        for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+THREADS*k;if(j<P)q[k]+=newhs[h*P+j]*(alpha-beta);}
    }
}
float slope=0;
for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+k*THREADS;if(j<P)slope-=q[k]*g[j];}
slope=direction_sum(slope,t,shared,THREADS);
bool reset=active && (!isfinite(slope) || slope>=0);
if(reset){count=0;head=0;slope=-gg;}
float penlin=0,penquad=0;
for(uint k=0;k<(P+THREADS-1)/THREADS;++k){uint j=t+k*THREADS;
 if(j<P){float d=active?(reset?-g[j]*h0inv[j]:-q[k]):0.0f;direction[j]=d;if(reg[j]>0){penlin+=reg[j]*w[j]*d;penquad+=reg[j]*d*d;}}}
penlin=direction_sum(penlin,t,shared,THREADS);penquad=direction_sum(penquad,t,shared,THREADS);
if(t==0){
 stats[0]=meta[0];stats[1]=float(status);stats[2]=float(count);stats[3]=float(head);
 stats[4]=meta[4];stats[5]=active?slope:0.0f;stats[6]=penlin;stats[7]=penquad;
 stats[8]=alpha_start;stats[9]=meta[9];stats[10]=meta[10];stats[11]=meta[11];
}
""",
)

_accept = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_accept",
    header=_REDUCE,
    input_names=["w", "raw", "direction", "dr", "y", "sw", "stats", "diffs", "c1"],
    output_names=["neww", "newraw", "residual", "newmeta"],
    source=r"""
uint i=thread_position_in_grid.x;
uint status=uint(stats[1]);float alpha=0,accepted_diff=0;bool accepted=false;
if(status==0){
 for(uint k=0;k<CAND;++k){
  float a=stats[8]*exp2(-float(k));
  float df=diffs[k]+a*stats[6]+0.5f*a*a*stats[7];
  if(isfinite(df) && df<=c1*a*stats[5] && stats[5]<0){alpha=a;accepted_diff=df;accepted=true;break;}
 }
 if(!accepted)status=2;
}
if(i<P)neww[i]=accepted?fma(alpha,direction[i],w[i]):w[i];
if(i<N){float z=accepted?fma(alpha,dr[i],raw[i]):raw[i];newraw[i]=z;float sign=1.0f-2.0f*y[i];residual[i]=sw[i]>0?sw[i]*sign*sigmoid_stable(sign*z):0.0f;}
if(i<12)newmeta[i]=i==0?stats[0]+float(accepted):(i==1?float(status):(i==4?(accepted?stats[10]:stats[4]):(i==8?alpha:(i==9?stats[9]+(stats[1]==0?CAND:0):(i==10?(accepted?accepted_diff:stats[10]):(i==11?stats[11]+accepted_diff:stats[i]))))));
""",
)


_backward = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_backward_bias",
    input_names=["X", "r", "active"],
    output_names=["parts"],
    source=r"""
// Coalesced feature loads, four row accumulators per feature, then a small
// partial reduction. The intercept is implicit, avoiding an augmented X copy.
uint col=threadgroup_position_in_grid.x*32+(thread_position_in_threadgroup.x % 32);
uint block=threadgroup_position_in_grid.y,sg=(thread_position_in_threadgroup.x / 32);
uint start=block*256+sg;float value=0;
if(col<P && active){for(uint row=start;row<min(uint(N),(block+1)*256);row+=4)value=fma(col<D?X[ulong(row)*D+col]:1.0f,r[row],value);}
threadgroup float partial[128];uint tid=thread_position_in_threadgroup.x;
partial[tid]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && col<P)parts[ulong(block)*P+col]=partial[tid]+partial[tid+32]+partial[tid+64]+partial[tid+96];
""",
)


def bwd(X, r, p, active=None):
    n, d = X.shape
    nb = (n + 255) // 256
    if active is None:
        active = mx.array(True)
    return mx.sum(
        _backward(
            inputs=[X, r, active],
            template=[("N", n), ("D", d), ("P", p)],
            grid=(((p + 31) // 32) * 128, nb, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(nb, p)],
            output_dtypes=[mx.float32],
        )[0],
        axis=0,
    )


_direct_forward = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_direct_forward",
    header=_REDUCE,
    input_names=["X", "theta", "y", "sw"],
    output_names=["loss", "residual"],
    source=r"""
uint tid=thread_position_in_threadgroup.x,lane=tid%32;
uint row=threadgroup_position_in_grid.x*4+tid/32;
float z=0.0f;
if(row<N){for(uint j=lane;j<D;j+=32)z=fma(X[ulong(row)*D+j],theta[j],z);}
z=simd_sum(z);
if(lane==0 && row<N){
 if(P>D)z+=theta[D];
 float weight=sw[row];
 if(weight>0){
  float sign=1.0f-2.0f*y[row],u=sign*z;
  loss[row]=weight*softplus_stable(u);
  residual[row]=weight*sign*sigmoid_stable(u);
 }else{loss[row]=0.0f;residual[row]=0.0f;}
}
""",
)


@mx.compile
def direct_arrays(theta, X, y, sw, denom, l2, fit_intercept):
    """Evaluate loss and gradient from actual parameters using native kernels.

    ``l2`` is a Python float, making the unpenalized branch static. The caller
    reduces returned sample losses in blocks and finishes accumulation in host
    float64, exactly as for the generic logistic objective.
    """
    n, d = X.shape
    p = d + int(fit_intercept)
    if sw is None:
        sw = mx.ones(n, dtype=X.dtype)
    loss, residual = _direct_forward(
        inputs=[X, theta, y, sw],
        template=[("N", n), ("D", d), ("P", p)],
        grid=(((n + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    gradient = bwd(X, residual, p) / denom
    if l2 > 0:
        weights = theta[0, :d]
        grad_penalty = l2 * weights
        if fit_intercept:
            grad_penalty = mx.concatenate([grad_penalty, mx.zeros(1, dtype=X.dtype)])
        gradient = gradient + grad_penalty
        penalty = (0.5 * l2) * mx.sum(weights * weights)
    else:
        # Keep the zero output dependent on an input for mx.compile, without
        # squaring unpenalized parameters or multiplying a possible infinity.
        penalty = denom - denom
    return loss, penalty, gradient.reshape(theta.shape)


def initial(X, y, sw, p, reg, denom):
    """Initial gradient and scalar metric for a design without prepared parts.

    The estimator collects ``initial_parts`` while copying the design. A
    direct caller gets the same statistics from the same kernel, at the cost
    of one discarded copy of its already owned ``X``.
    """
    from .._kernels.logistic_input import prepare_centered

    if p not in (X.shape[1], X.shape[1] + 1):
        raise ValueError("p must be the feature count, plus one for an intercept")
    _, _, _, parts = prepare_centered(X, y, sw, center=False, intercept=p == X.shape[1] + 1)
    return initial_from_parts(parts, reg, denom)


@mx.compile
def initial_from_parts(parts, reg, denom):
    """Normalize initial statistics collected from the actual solver design."""
    out = mx.sum(parts, axis=1) / denom
    diagonal = mx.mean(out[1] + reg)
    # A scalar metric preserves the data's row space on rank-deficient designs.
    # A per-feature Jacobi metric can introduce null-space coefficients that a
    # weak penalty does not remove before the gradient stopping criterion fires.
    scale = mx.where(mx.isfinite(diagonal) & (diagonal > 1e-12), 1 / diagonal, 1.0)
    return out[0], mx.ones_like(out[0]) * scale


@mx.compile
def step(
    X,
    y,
    sw,
    reg,
    denom,
    tol,
    h0inv,
    c1,
    min_curvature,
    candidates,
    alpha_start,
    center,
    w,
    g,
    oldw,
    oldg,
    hs,
    hy,
    hc,
    meta,
    raw,
    weighted=False,
):
    n, d = X.shape
    p = w.shape[0]
    m = hs.shape[0]
    direction_threads = 256 if d >= 2048 else 32
    direction, hs, hy, hc, stats = _direction(
        inputs=[w, g, oldw, oldg, hs, hy, hc, meta, reg, tol, h0inv, min_curvature, alpha_start, center],
        template=[("P", p), ("D", d), ("M", m), ("WEIGHTED", int(weighted)),
                  ("THREADS", direction_threads)],
        grid=(direction_threads, 1, 1),
        threadgroup=(direction_threads, 1, 1),
        output_shapes=[(p,), (m, p), (m, p), (m, 2), (12,)],
        output_dtypes=[mx.float32] * 5,
    )
    # One fused pass over X serves every width: projections, all candidate
    # loss changes, and the first candidate's gradient. The module is imported
    # here because it takes this module's shared kernel header at load time.
    from .native_logistic_fused import forward_gradient

    dr, diffs, first_gradient = forward_gradient(
        X, direction, raw, y, sw, stats, denom, candidates
    )
    neww, newraw, resid, meta = _accept(
        inputs=[w, raw, direction, dr, y, sw, stats, diffs, c1],
        template=[("N", n), ("P", p), ("CAND", candidates)],
        grid=(((max(n, p) + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(p,), (n,), (n,), (12,)],
        output_dtypes=[mx.float32] * 4,
    )
    # A rejected first batch is resumed from exactly the same gradient and
    # history. Recomputing an unchanged gradient can perturb the direction.
    active = meta[1] == 0
    # The acceptance kernel supplies the chosen step, so this does not
    # repeat its floating-point predicate with a different rounding order.
    first_accepted = active & (meta[8] == alpha_start)
    # mx.where alone would evaluate both branches. Pass the predicate into
    # the backward Metal kernel so first acceptance performs no second X read.
    fallback = bwd(X, resid, p, active & ~first_accepted) / denom
    sample_gradient = mx.where(first_accepted, first_gradient, fallback)
    newg = mx.where(active, sample_gradient + reg * neww, g)
    return neww, newg, w, g, hs, hy, hc, meta, newraw


#: Two direct objective evaluations of nearly identical parameters differ by
#: float32 rounding of the per-sample losses. The measured difference between a
#: native iterate and its continuation is 1e-9 to 4e-9 relative; a band of a
#: few float32 epsilons separates such noise from an actual objective increase.
_RESUME_OBJECTIVE_BAND = 4.0 * float(np.finfo(np.float32).eps)


def _resume_keeps_native_iterate(
    previous: LBFGSState, objective: float, grad_max: float, finite: bool
) -> bool:
    """Decide whether an uncertified continuation must yield to the native iterate.

    A non-finite or clearly higher objective keeps the native parameters. Inside
    the rounding band the objectives are indistinguishable, so the iterate with
    the smaller direct gradient wins. A clearly lower objective keeps the
    continuation. The caller never applies this to a certified continuation.
    """
    if not finite:
        return True
    band = _RESUME_OBJECTIVE_BAND * abs(previous.objective)
    if objective > previous.objective + band:
        return True
    if objective >= previous.objective - band:
        return previous.grad_max_abs < grad_max
    return False


def _conservative_config_for_skewed_curvature(
    parts: mx.array | None, d: int, sw_sum: float, l2: float, config: LBFGSConfig
) -> LBFGSConfig:
    """Tighten the internal gradient threshold for strongly uneven feature scales.

    Initial weighted per-feature squared moments are already produced during
    input preparation. A large diagonal spread is a cheap warning sign that
    one coordinate's gradient can meet the requested maximum-norm tolerance
    while slow coordinates still have material objective progress remaining.
    This is a conservative extra check, not an objective-optimality bound or
    a general condition-number estimate: correlations can be ill-conditioned
    even when all feature diagonals match. The public tolerance and iteration
    budget are never relaxed.
    """
    if (
        parts is None or d < 512 or config.max_iter == 0 or config.tol <= 0
        or parts.ndim != 3 or parts.shape[0] != 2 or parts.shape[2] < d
    ):
        return config
    diagonal = mx.sum(parts[1, :, :d], axis=0) / sw_sum + l2
    extremes = mx.stack([mx.min(diagonal), mx.max(diagonal)])
    mx.eval(extremes)
    minimum, maximum = (float(value) for value in extremes.tolist())
    if not (math.isfinite(minimum) and math.isfinite(maximum)) or maximum <= 0:
        return config
    if minimum <= 0 or maximum > 100.0 * minimum:
        return replace(config, tol=config.tol * 0.5)
    return config


def minimize_binary(
    X: mx.array,
    y: mx.array,
    sw: mx.array | None,
    sw_sum: float,
    l2: float,
    fit_intercept: bool,
    config: LBFGSConfig,
    fun: ObjectiveGradient,
    *,
    initial_parts: mx.array | None = None,
    convergence_center: mx.array | None = None,
    initial_history=None,
    use_native_loop: bool = True,
) -> LBFGSState:
    """Minimize a binary logistic objective on the current GPU stream.

    ``X`` is already prepared by the caller; this function performs no host
    preprocessing or dtype conversion. Backtracking first evaluates eight
    geometrically decreasing steps. A rejected batch resumes the same direction
    with the remaining search budget, up to 32 candidates in total.
    The line search uses Armijo decrease rather than strong Wolfe curvature.
    A positive curvature check protects each L-BFGS history insertion.

    Cached logits avoid repeated matrix-vector products. They are never the
    authority for returned objective or convergence: ``fun`` recomputes both
    directly, and a premature cached convergence restarts from that gradient
    while preserving the remaining accepted-step budget.

    ``initial_parts`` may contain statistics collected while copying exactly
    this design. The same normalization and scalar initial metric apply.
    """
    if not 1 <= config.max_line_search <= 32:
        raise ValueError("native binary L-BFGS requires max_line_search in [1, 32]")
    if initial_history is not None and not use_native_loop:
        raise ValueError("initial_history requires the native command loop")
    n, d = X.shape
    if not 1 <= d <= MAX_FUSED_FEATURES:
        raise ValueError(f"native binary L-BFGS requires 1 to {MAX_FUSED_FEATURES} features")
    weighted = sw is not None
    p = d + int(fit_intercept)
    if convergence_center is not None and (
        not fit_intercept or convergence_center.shape != (d,) or convergence_center.dtype != X.dtype
    ):
        raise ValueError("convergence_center must match the centered feature dimension and dtype")
    from .fused_lbfgs import _convergence_gradient_max

    w = mx.zeros(p, dtype=X.dtype)

    if config.max_iter == 0:
        objective, gradient = fun(w.reshape(1, p))
        grad_max = float(_convergence_gradient_max(gradient, convergence_center))
        finite = math.isfinite(objective) and math.isfinite(grad_max)
        converged = finite and grad_max <= config.tol and (not weighted or grad_max == 0)
        return LBFGSState(
            x=w.reshape(1, p),
            objective=objective,
            grad_max_abs=grad_max,
            n_iter=0,
            n_fev=1,
            converged=converged,
            stop_reason="gtol" if converged else ("max_iter" if finite else "non-finite-objective"),
        )

    # Native command submission requires prepared, privately owned inputs.
    # Preserve its state and spent budget if direct certification fails; never
    # rerun from zero after an actual native attempt.
    native_attempt = None
    if initial_parts is not None and use_native_loop:
        native_kwargs = {"initial_parts": initial_parts, "convergence_center": convergence_center,
                         "return_attempt": True}
        if initial_history is not None:
            native_kwargs["initial_history"] = initial_history
        native_state = native_command.run_native_loop(
            X, y, sw, sw_sum, l2, fit_intercept, config, fun, **native_kwargs
        )
        if isinstance(native_state, native_command.NativeAttempt):
            native_attempt = native_state
            result = native_attempt.state
            if (
                result.converged or initial_history is not None
                or result.stop_reason in ("non-finite-objective", "max_iter")
                or result.n_iter >= config.max_iter
                or native_attempt.status not in (1, 2)
                # A failed search may not be relabeled as convergence without
                # accepting another step, even if a direct check is small.
                or (native_attempt.status == 2 and result.grad_max_abs <= config.tol)
            ):
                return result
        elif native_state is not None:
            return native_state

    if sw is None:
        sw = mx.ones(n, dtype=X.dtype)
    reg = mx.full(d, l2, dtype=X.dtype)
    if fit_intercept:
        reg = mx.concatenate([reg, mx.zeros(1, dtype=X.dtype)])
    denom = mx.array(sw_sum, dtype=X.dtype)
    # Rounding the threshold upward could let the device stop at a gradient
    # that the host rejects, without taking any step after a reanchor.
    # A finite host tolerance can exceed float32's range. Clamp before the
    # cast so the device remains conservative without an overflow warning.
    device_tol = np.float32(min(config.tol, float(np.finfo(np.float32).max)))
    if float(device_tol) > config.tol:
        device_tol = np.nextafter(device_tol, np.float32(0))
    tol = mx.array(device_tol, dtype=X.dtype)
    device_center = convergence_center if convergence_center is not None else mx.zeros(d, dtype=X.dtype)
    c1 = mx.array(config.c1, dtype=X.dtype)
    min_curvature = mx.array(config.min_curvature, dtype=X.dtype)
    if initial_parts is None:
        g, h0inv = initial(X, y, sw, p, reg, denom)
    else:
        if initial_parts.shape != (2, (n + 255) // 256, p) or initial_parts.dtype != X.dtype:
            raise ValueError("initial_parts must match the native solver design")
        g, h0inv = initial_from_parts(initial_parts, reg, denom)
    hs = mx.zeros((config.history_size, p), dtype=X.dtype)
    hy = mx.zeros_like(hs)
    hc = mx.zeros((config.history_size, 2), dtype=X.dtype)
    if native_attempt is None:
        meta = mx.concatenate([mx.zeros(11, dtype=X.dtype), mx.array([math.log(2)], dtype=X.dtype)])
        raw = mx.zeros(n, dtype=X.dtype)
        n_iter = 0
        direct_evaluations = 0
        last_reanchor_iteration = -1
    else:
        # Reanchor at the actual accepted native parameters, with an exactly
        # recomputed objective, centered gradient and logits. Discard history:
        # it was formed from the native loop's cached, rounded logits. Keep
        # previous accepted steps and evaluations in the original budget.
        result = native_attempt.state
        w = result.x.reshape(p)
        g = native_attempt.gradient.reshape(p)
        raw = X @ w[:d]
        if fit_intercept:
            raw = raw + w[d]
        host_meta = np.zeros(12, dtype=np.float32)
        host_meta[0] = result.n_iter
        host_meta[4] = np.inf
        host_meta[9] = result.n_fev - 1  # Direct evaluation tracked separately.
        host_meta[10] = np.inf
        host_meta[11] = result.objective
        meta = mx.array(host_meta, dtype=X.dtype)
        n_iter = result.n_iter
        direct_evaluations = 1
        last_reanchor_iteration = result.n_iter
    state = (w, g, w, g, hs, hy, hc, meta, raw)
    first_batch = min(8, config.max_line_search)
    search_offset = 0
    stall_reset = False

    while True:
        # Four steps amortize Python evaluation boundaries without allowing a
        # long tail of redundant GEMVs after device convergence.
        candidates = first_batch if search_offset == 0 else config.max_line_search - search_offset
        alpha_start = mx.array(2.0 ** (-search_offset), dtype=X.dtype)
        # A resumed search executes only one step: the next iteration starts at
        # unit step size again rather than inheriting the continuation offset.
        chunk = 4 if search_offset == 0 else 1
        chunk_start, chunk_raw = n_iter, state[8]
        for _ in range(min(chunk, config.max_iter - n_iter)):
            step_args = (
                X,
                y,
                sw,
                reg,
                denom,
                tol,
                h0inv,
                c1,
                min_curvature,
                candidates,
                alpha_start,
                device_center,
                *state,
            )
            state = step(*step_args, weighted=True) if weighted else step(*step_args)
        # The last accepted update may meet the tolerance before a following
        # direction kernel can set status=1. Check its gradient at this same
        # evaluation boundary, avoiding another whole frozen iteration chunk.
        chunk_grad_max = _convergence_gradient_max(state[1], convergence_center)
        # Accepted steps can leave every cached logit bit-for-bit unchanged
        # when the parameter change is below float32 resolution of the model:
        # the gradient then stops changing, no curvature pair can form, and
        # the same penalty-only Armijo decrease would be accepted until the
        # budget is spent without the model moving.
        logits_unchanged = mx.array_equal(state[8], chunk_raw)
        mx.eval(state, chunk_grad_max, logits_unchanged)
        host_meta = np.asarray(state[7]).copy()
        n_iter, status = int(host_meta[0]), int(host_meta[1])
        recent = abs(float(host_meta[10]))
        if n_iter >= 2:
            recent = 0.5 * (recent + abs(float(host_meta[4])))
        progress = (not weighted and (d < 2048 or n_iter == 0)) or float(chunk_grad_max) == 0 or (
            (not weighted or n_iter >= 2)
            and
            math.isfinite(float(host_meta[11]))
            and recent <= float(device_tol) * (0.025 if weighted or d >= 2048 else 1.0) * max(abs(float(host_meta[11])), 1e-12)
        )
        if status == 0 and float(chunk_grad_max) <= float(device_tol) and progress:
            status = 1
        stalled = status == 0 and n_iter > chunk_start and bool(logits_unchanged)
        if stalled and not stall_reset:
            # The frozen history cannot produce a different direction. Retry
            # once from a scalar-metric steepest-descent step.
            stall_reset = True
            host_meta[2] = host_meta[3] = 0.0
            search_offset = 0
            state = (*state[:4], mx.zeros_like(hs), mx.zeros_like(hy), mx.zeros_like(hc),
                     mx.array(host_meta, dtype=X.dtype), state[8])
            continue
        if stalled:
            # Report precision stagnation as a failed search, so the caller's
            # intercept polish and float64 certificate run now instead of
            # after the whole iteration budget.
            status = 2
        if (
            status == 2 and not stalled and search_offset == 0
            and first_batch < config.max_line_search
        ):
            search_offset = first_batch
            host_meta[1] = 0
            state = (*state[:7], mx.array(host_meta, dtype=X.dtype), state[8])
            continue
        if status == 0:
            search_offset = 0
        if status == 0 and n_iter < config.max_iter:
            continue

        x = state[0].reshape(1, p)
        objective, direct_gradient = fun(x)
        direct_evaluations += 1
        grad_max = float(_convergence_gradient_max(direct_gradient, convergence_center))
        finite = math.isfinite(objective) and math.isfinite(grad_max)
        direct_progress = (not weighted and (d < 2048 or n_iter == 0)) or grad_max == 0 or (
            (not weighted or n_iter >= 2)
            and
            math.isfinite(recent)
            and recent <= config.tol * (0.025 if weighted or d >= 2048 else 1.0) * max(abs(objective), 1e-12)
        )
        # A fully rejected line search is not convergence, even if its
        # independently recomputed gradient happens to be below tolerance.
        converged = status in (0, 1) and finite and grad_max <= config.tol and direct_progress
        if (
            native_attempt is not None and not converged
            and _resume_keeps_native_iterate(native_attempt.state, objective, grad_max, finite)
        ):
            # The resumed path must not return a worse candidate than the
            # already direct-scored native iterate. Preserve its parameters and
            # loss while accounting for all accepted continuation steps. A
            # continuation the direct check certifies is never discarded.
            previous = native_attempt.state
            return LBFGSState(
                x=previous.x, objective=previous.objective,
                grad_max_abs=previous.grad_max_abs, n_iter=n_iter,
                n_fev=int(host_meta[9]) + direct_evaluations, converged=False,
                stop_reason="non-finite-objective" if not finite else "line-search-failure",
            )
        if status == 1 and finite and not converged and n_iter < config.max_iter:
            if n_iter == last_reanchor_iteration:
                # No accepted step separates these two direct checks. Report
                # precision stagnation through the existing failure contract.
                return LBFGSState(
                    x=x,
                    objective=objective,
                    grad_max_abs=grad_max,
                    n_iter=n_iter,
                    n_fev=int(host_meta[9]) + direct_evaluations,
                    converged=False,
                    stop_reason="line-search-failure",
                )
            last_reanchor_iteration = n_iter
            # Rebuild from the actual parameters, not from accumulated logits.
            # A history reset avoids mixing gradients from the two paths.
            w = state[0]
            g = direct_gradient.reshape(p)
            raw = X @ w[:d]
            if fit_intercept:
                raw = raw + w[d]
            host_meta[1:9] = 0
            # A reanchor replaces the cached gradient; require a fresh accepted
            # steps before relying on the previous loss-progress statistics.
            host_meta[4] = np.inf
            host_meta[10] = np.inf
            host_meta[11] = objective
            search_offset = 0
            meta = mx.array(host_meta, dtype=X.dtype)
            state = (w, g, w, g, mx.zeros_like(hs), mx.zeros_like(hy), mx.zeros_like(hc), meta, raw)
            continue

        if not finite:
            reason = "non-finite-objective"
        elif converged:
            reason = "gtol"
        elif status == 2:
            reason = "line-search-failure"
        elif status == 3:
            reason = "non-finite-objective"
        else:
            reason = "max_iter"
        return LBFGSState(
            x=x,
            objective=objective,
            grad_max_abs=grad_max,
            n_iter=n_iter,
            n_fev=int(host_meta[9]) + direct_evaluations,
            converged=converged,
            stop_reason=reason,
        )
