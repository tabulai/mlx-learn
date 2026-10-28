"""Optional batched Metal command loop for the wide binary solver.

The extension consumes Metal DLPack buffers, and completes all external work
before returning.  A direct objective/gradient evaluation remains authoritative;
unhandled line searches and precision reanchors use the existing MLX solver.
"""

from __future__ import annotations

import ast
import functools
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .lbfgs import LBFGSState
from .native_geometry import speculative_geometry


@dataclass(frozen=True, slots=True)
class NativeAttempt:
    """Direct-scored native result and centered gradient for a budgeted resume."""

    state: LBFGSState
    gradient: mx.array
    status: int

try:
    _metal_loop = importlib.import_module(f"{__package__}._metal_loop")
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}._metal_loop":
        raise
    _metal_loop = None


def _kernel_source(path, name):
    """Extract the existing single source of truth for an MLX Metal kernel."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in item.targets
        ):
            if isinstance(item.value, ast.Constant):
                return item.value.value
            for keyword in item.value.keywords:
                if keyword.arg == "source":
                    return ast.literal_eval(keyword.value)
    raise RuntimeError(f"missing Metal source for {name}")


def _kernel(name, args, body):
    declaration = ",\n".join(f"{kind} {arg} [[buffer({i})]]" for i, (kind, arg) in enumerate(args))
    return f"kernel void {name}(\n{declaration},\nuint3 thread_position_in_grid [[thread_position_in_grid]],\nuint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\nuint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {{\n{body}\n}}\n"


@functools.lru_cache(maxsize=32)
def _source(n, d, p, m, cand, blocks, groups, rows, feature_groups, weighted=False, direction_threads=None):
    directory = Path(__file__).parent
    core = directory / "native_logistic.py"
    fused = directory / "native_logistic_fused.py"
    if direction_threads is None:
        direction_threads = 256 if d >= 2048 else 32
    if direction_threads not in (32, 256):
        raise ValueError("unsupported direction thread count")
    defines = f"#define DIRECTION_THREADS {direction_threads}\n#define THREADS DIRECTION_THREADS\n#define N {n}\n#define D {d}\n#define P {p}\n#define M {m}\n#define CAND {cand}\n#define B {blocks}\n#define GROUPS {groups}\n#define ROWS {rows}\n#define FEATURE_GROUPS {feature_groups}\n#define WEIGHTED {int(weighted)}\n"
    # Both MLX and the command loop use the same explicit small-loss helper.
    header = "#include <metal_stdlib>\nusing namespace metal;\n" + defines + _kernel_source(core, "_REDUCE") + _kernel_source(core, "_DIRECTION_REDUCE")
    def inputs(*names):
        return [("device const float*", name) for name in names]

    def outputs(*names):
        return [("device float*", name) for name in names]
    direction = (inputs("w", "g", "oldw", "oldg", "hs", "hy", "hc", "meta", "reg")
                 + [("constant float&", "tol")] + inputs("h0inv")
                 + [("constant float&", "min_curvature"), ("constant float&", "alpha_start")]
                 + inputs("center")
                 + outputs("direction", "newhs", "newhy", "newhc", "stats"))
    speculative = inputs("X", "direction", "raw", "y", "sw", "stats") + [("constant float&", "denom")] + outputs("dr", "partial", "gradient_parts")
    accept = inputs("w", "raw", "direction", "dr", "y", "sw", "stats", "diffs") + [("constant float&", "c1")] + outputs("neww", "newraw", "residual", "newmeta")
    backward = inputs("X", "r") + [("device const bool&", "active")] + outputs("parts")
    header += _kernel("direction_kernel", direction, _kernel_source(core, "_direction"))
    header += _kernel("speculative_kernel", speculative, _kernel_source(fused, "_speculative"))
    header += _kernel("accept_kernel", accept, _kernel_source(core, "_accept"))
    header += _kernel("backward_kernel", backward, _kernel_source(core, "_backward"))
    header += r"""
kernel void reduce_spec(device const float* losses [[buffer(0)]],
                        device const float* parts [[buffer(1)]],
                        constant float& denom [[buffer(2)]],
                        device float* diffs [[buffer(3)]],
                        device float* gradient [[buffer(4)]],
                        uint t [[thread_position_in_threadgroup]],
                        uint group [[threadgroup_position_in_grid]]) {
 uint lane=t%32,sg=t/32,col=group*32+lane;
 float value=0, loss=0;
 for(uint i=sg;i<B;i+=4){
   if(col<P)value+=parts[ulong(i)*P+col];
   if(group==0 && lane<CAND)loss+=losses[ulong(lane)*B+i];
 }
 threadgroup float shared[128], shared_loss[128];
 shared[t]=value;shared_loss[t]=loss;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0){
   if(col<P)gradient[col]=(shared[t]+shared[t+32]+shared[t+64]+shared[t+96])/denom;
   if(group==0 && lane<CAND)diffs[lane]=shared_loss[t]+shared_loss[t+32]+shared_loss[t+64]+shared_loss[t+96];
 }
}
kernel void make_active(device const float* meta [[buffer(0)]],
                        constant float& alpha [[buffer(1)]],
                        device bool* active [[buffer(2)]]) {
 active[0]=meta[1]==0 && meta[8]!=alpha;
}
kernel void finish_grad(device const float* parts [[buffer(0)]],
                        device const float* first [[buffer(1)]],
                        device const float* meta [[buffer(2)]],
                        constant float& alpha [[buffer(3)]],
                        device const float* oldg [[buffer(4)]],
                        device const float* w [[buffer(5)]],
                        device const float* reg [[buffer(6)]],
                        constant float& denom [[buffer(7)]],
                        device float* g [[buffer(8)]],
                        uint t [[thread_position_in_grid]]) {
 if(t<P){float value=first[t];
  if(meta[1]==0 && meta[8]!=alpha){value=0;for(uint i=0;i<(N+255)/256;i++)value+=parts[ulong(i)*P+t];value/=denom;}
  g[t]=meta[1]==0?value+reg[t]*w[t]:oldg[t];
 }
}
"""
    return header


def _supports_metal_dlpack():
    """Use only MLX releases with documented Metal-buffer DLPack ownership.

    MLX 0.29–0.31 can report a Metal device while exporting a host pointer;
    treating that pointer as an MTLBuffer is unsafe. Only the inspected 0.32.x
    release series is enabled; prereleases, unknown and future versions use
    the existing MLX solver until their export format has been verified.
    """
    version = getattr(mx, "__version__", None)
    if not isinstance(version, str):
        return False
    return re.fullmatch(r"0\.32\.(?:0|[1-9]\d{0,2})", version) is not None


def _native_loop_tol(requested, weighted):
    """Reserve weighted progress margin and round down to float32.

    The native running loss is an estimate; the caller recomputes the final
    objective independently. A tighter internal tolerance leaves room for
    their rounding difference without relaxing the final acceptance check.
    Never increase a subnormal requested tolerance to the normal minimum.
    """
    minimum_normal = float(np.finfo(np.float32).tiny)
    target = max(min(requested, minimum_normal), 0.95 * requested) if weighted else requested
    # Larger finite host tolerances use the largest representable threshold;
    # clamping before conversion avoids a spurious overflow warning.
    rounded = np.float32(min(target, float(np.finfo(np.float32).max)))
    if float(rounded) > target:
        rounded = np.nextafter(rounded, np.float32(0))
    return float(rounded)


def run_native_loop(
    X, y, sw, sw_sum, l2, fit_intercept, config, fun, *, initial_parts=None,
    convergence_center=None, initial_history=None, return_attempt=False,
):
    """Return a certified native result, or None when the MLX solver should run."""
    if _metal_loop is None or initial_parts is None or not fit_intercept or not _supports_metal_dlpack():
        return None
    n, d = X.shape
    if not (512 <= d <= 4096 and config.history_size == 10 and config.max_line_search >= 8):
        return None
    from .native_logistic import initial_from_parts

    p = d + 1
    if initial_parts.shape != (2, (n + 255) // 256, p) or initial_parts.dtype != mx.float32:
        raise ValueError("initial_parts must match the native solver design")
    weighted = sw is not None
    if sw is None:
        sw = mx.ones(n, dtype=mx.float32)
    reg = mx.concatenate([mx.full(d, l2, dtype=mx.float32), mx.zeros(1, dtype=mx.float32)])
    denom = mx.array(sw_sum, dtype=mx.float32)
    if initial_history is None:
        g, h0 = initial_from_parts(initial_parts, reg, denom)
    else:
        from mlxlearn_device.hessian_sketch import initial_metric_from_parts

        g, h0 = initial_metric_from_parts(initial_parts, reg, denom)
    # Export synchronizes the producer; keep every capsule and array alive
    # until the native command buffer has finished executing.
    dependencies = (X, y, sw, g, h0, reg)
    if convergence_center is not None:
        dependencies += (convergence_center,)
    seed_arrays = ()
    seed_count = 0
    if initial_history is not None:
        seed_s, seed_y, seed_hc, seed_count = initial_history
        if (
            not isinstance(seed_count, int)
            or not 0 <= seed_count <= config.history_size
            or seed_s.shape != (config.history_size, p)
            or seed_y.shape != (config.history_size, p)
            or seed_hc.shape != (config.history_size, 2)
            or any(value.dtype != mx.float32 for value in (seed_s, seed_y, seed_hc))
        ):
            raise ValueError("initial_history must match the native solver history")
        if seed_count:
            seed_arrays = (seed_s, seed_y, seed_hc)
            dependencies += seed_arrays
    mx.eval(*dependencies)
    mx.synchronize()
    arrays = (X, y, sw, g, h0, reg)
    capsules = tuple(value.__dlpack__() for value in arrays)
    # Preserve both the managed DLPack handles and the original MLX arrays if
    # a submitted command cannot be observed to complete on an error path.
    owners = (capsules, *arrays)
    seed_capsules = None
    if seed_arrays:
        seed_capsules = tuple(value.__dlpack__() for value in seed_arrays)
        owners += (seed_capsules, *seed_arrays)
    center = (
        np.zeros(d, dtype=np.float32)
        if convergence_center is None
        else np.ascontiguousarray(np.asarray(convergence_center), dtype=np.float32)
    )
    if center.shape != (d,):
        raise ValueError("convergence_center must match the feature dimension")
    groups, feature_groups, rows, blocks = speculative_geometry(n, d)
    # Older optional extensions hard-code a single SIMD group. Keep their
    # source specialization consistent until that extension is rebuilt.
    direction_threads = 256 if d >= 2048 and hasattr(_metal_loop, "run_cg") else 32
    src = _source(n, d, p, config.history_size, 8, blocks, groups, rows, feature_groups, weighted, direction_threads)
    tol = _native_loop_tol(config.tol, weighted)
    arguments = (src, capsules, n, d, p, config.history_size, blocks, groups,
                 config.max_iter, float(np.float32(sw_sum)), tol, config.c1,
                 config.min_curvature, owners, center.tobytes())
    if weighted:
        arguments += (seed_capsules, seed_count, None, 1)
    elif seed_capsules is not None:
        arguments += (seed_capsules, seed_count)
    result = _metal_loop.run(*arguments)
    raw, n_iter, status, n_fev, last_difference, previous_difference = result
    if status != 1 and initial_history is None and not return_attempt:
        return None
    w = mx.array(np.frombuffer(raw, dtype=np.float32).copy()).reshape(1, p)
    objective, gradient = fun(w)
    from .fused_lbfgs import _convergence_gradient_max

    grad_max = float(_convergence_gradient_max(gradient, convergence_center))
    recent = abs(last_difference)
    if n_iter >= 2:
        recent = 0.5 * (recent + abs(previous_difference))
    progress = (not weighted and (d < 2048 or n_iter == 0)) or grad_max == 0 or (
        (not weighted or n_iter >= 2)
        and
        math.isfinite(recent)
        and recent <= config.tol * (0.025 if weighted or d >= 2048 else 1.0) * max(abs(objective), 1e-12)
    )
    finite = math.isfinite(objective) and math.isfinite(grad_max)
    converged = status in (0, 1) and finite and grad_max <= config.tol and progress
    if not converged:
        if initial_history is None and not return_attempt:
            return None
        # The native loop already spent accepted iterations. Restarting from
        # zero with a fresh budget would misreport max_iter and could hide a
        # failed sketch; report the directly checked state instead.
        reason = (
            "non-finite-objective" if status == 3 or not finite else
            "max_iter" if n_iter >= config.max_iter else "line-search-failure"
        )
        state = LBFGSState(x=w, objective=objective, grad_max_abs=grad_max, n_iter=n_iter,
                           n_fev=n_fev + 1, converged=False, stop_reason=reason)
    else:
        state = LBFGSState(x=w, objective=objective, grad_max_abs=grad_max, n_iter=n_iter,
                           n_fev=n_fev + 1, converged=True, stop_reason="gtol")
    return NativeAttempt(state=state, gradient=gradient, status=status) if return_attempt else state
