"""NumPy-free GPU-resident binary logistic regression.

This API is separate from the sklearn wrapper. It accepts float32 MLX arrays
and returns a centered-coordinate model with MLX arrays. Metal runs the
optimizer and gradient check; Python coordinates command submissions and reads
small scalar status values. No CPU array or NumPy conversion is used.
"""

from __future__ import annotations

import math
import re
import struct
import warnings
from dataclasses import dataclass

import mlx.core as mx

from ._compensated import logits_parts, score_original
from ._native import metal_loop, prepare_centered, source_for, speculative_geometry
from .hessian_sketch import build_sketch_history, initial_metric_from_parts

__all__ = ["DeviceConvergenceWarning", "DeviceLogisticModel", "fit_logistic"]


class DeviceConvergenceWarning(UserWarning):
    """The GPU optimizer did not pass its direct numerical convergence check."""


@dataclass(frozen=True, slots=True)
class DeviceLogisticModel:
    """Binary coefficients and intercept in centered feature coordinates.

    ``coef_`` has shape ``(1, n_features)``. The model score is
    ``(X - center_) @ coef_[0] + centered_intercept_``; retaining this form is
    important when the original features have a large offset. All three model
    arrays remain MLX arrays. Python numeric status fields are synchronized
    metadata, not host copies of training data or coefficients.

    ``converged`` means the native stopping conditions and a compensated GPU
    check of the original float32 data passed with a safety margin. The latter
    is not a formal error bound; callers needing rigorous numerical guarantees
    should use independent validation. Unsuccessful fits report a warning and
    retain the actual parameters and reason.
    """

    coef_: mx.array
    center_: mx.array
    centered_intercept_: mx.array
    n_iter: int
    converged: bool
    stop_reason: str
    objective: float
    gradient_max_abs: float
    sketch_count: int
    used_fallback: bool
    discarded_tail_bound: float
    newton_iterations: int = 0
    newton_hvp_count: int = 0
    relative_duality_gap: float | None = None

    @property
    def sketch_pairs_(self) -> int:
        """Number of accepted GPU sketch pairs in the initial attempt."""
        return self.sketch_count

    @property
    def used_unsketched_fallback_(self) -> bool:
        """Whether the fit restarted without sketch pairs."""
        return self.used_fallback

    def _decision_parts(self, X: mx.array):
        if not isinstance(X, mx.array) or X.ndim != 2 or X.shape[1] != self.center_.size:
            raise ValueError("X must be an MLX matrix with the fitted feature count")
        if X.dtype != mx.float32:
            raise TypeError("X must have float32 dtype")
        with mx.stream(mx.gpu):
            if X.shape[0] == 0:
                return mx.zeros((0,), mx.float32), mx.zeros((0,), mx.float32)
            theta = mx.concatenate([self.coef_[0], self.centered_intercept_.reshape(1)])
            return logits_parts(X, self.center_, theta)

    def decision_function(self, X: mx.array) -> mx.array:
        """Return compensated binary logits as a float32 MLX array on the GPU."""
        hi, lo = self._decision_parts(X)
        with mx.stream(mx.gpu):
            return hi + lo

    def predict(self, X: mx.array) -> mx.array:
        """Return device labels using both components near a decision boundary."""
        hi, lo = self._decision_parts(X)
        with mx.stream(mx.gpu):
            return ((hi > 0) | ((hi == 0) & (lo >= 0))).astype(mx.int32)


def _device_tol(value: float) -> float:
    """Round a nonnegative tolerance toward zero to float32."""
    rounded = struct.unpack("=f", struct.pack("=f", value))[0]
    if rounded > value:
        bits = struct.unpack("=I", struct.pack("=f", rounded))[0]
        rounded = struct.unpack("=f", struct.pack("=I", bits - 1))[0]
    return rounded


def _seeded_native_tol(certificate_tol: float) -> float:
    """Reserve the certificate's safety margin in the seeded native solve.

    The final compensated score is still checked against ``certificate_tol``;
    this merely asks the native loop to continue past gradients near its
    independently checked threshold. Keep the tolerance normal because Metal
    may flush float32 subnormals to zero.
    """
    min_normal = 1.1754943508222875e-38
    return max(min_normal, _device_tol(certificate_tol * 0.95))


def _weight_sum(sw: mx.array, n: int) -> float:
    """Reduce blocks on GPU and finish only a short scalar list on the host."""
    padding = (-n) % 256
    values = mx.pad(sw, (0, padding)) if padding else sw
    partial = mx.sum(values.reshape((-1, 256)), axis=1)
    mx.eval(partial)
    return math.fsum(partial.tolist())



def _polish_intercept(X, centered, center, output, y, sw, sw_sum, l2,
                      iterations, budget, effective_tol, objective_value, norm_value, tail_value):
    """Try bounded GPU intercept-only Newton steps after a stalled search.

    Bias changes use a stable signed loss difference and are accepted only if
    the direct compensated original-coordinate gradient improves. The wide
    loss-progress guard cannot be certified by a bias-only step, so callers
    restrict this helper to fewer than 2,048 features.
    """
    if iterations >= budget or not math.isfinite(norm_value):
        return output, iterations, objective_value, norm_value, tail_value, False
    linear = centered @ output[:-1]
    weights = sw / float(sw_sum)
    sign = 1.0 - 2.0 * y
    mx.eval(linear, weights, sign)
    accepted_any = False
    for _ in range(min(4, budget - iterations)):
        _, _, grad_hi, grad_lo, _ = score_original(X, center, output, y, sw, sw_sum, l2)
        raw = linear + output[-1]
        curvature = mx.sum(weights * mx.sigmoid(raw) * mx.sigmoid(-raw))
        mx.eval(grad_hi, grad_lo, curvature)
        gradient = float(grad_hi[-1]) + float(grad_lo[-1])
        h = float(curvature)
        if not math.isfinite(gradient) or not math.isfinite(h) or h <= 0:
            break
        proposed = max(-0.5, min(0.5, -gradient / h))
        old_bias = float(output[-1])
        if not math.isfinite(proposed) or proposed == 0:
            break
        accepted = False
        for _ in range(8):
            new_bias = mx.array(old_bias + proposed, dtype=mx.float32)
            mx.eval(new_bias)
            actual = float(new_bias) - old_bias
            if not math.isfinite(actual) or actual == 0:
                break
            # Difference is evaluated directly, so tiny improvements do not
            # disappear when subtracting two large float32 loss totals.
            delta = sign * actual
            difference = mx.sum(weights * mx.log1p(mx.sigmoid(sign * raw) * mx.expm1(delta)))
            mx.eval(difference)
            if math.isfinite(float(difference)) and float(difference) < 0:
                candidate = mx.concatenate([output[:-1], new_bias.reshape(1)])
                new_obj, new_norm, _, _, new_tail = score_original(X, center, candidate, y, sw, sw_sum, l2)
                mx.eval(new_obj, new_norm, new_tail)
                candidate_obj, candidate_norm, candidate_tail = float(new_obj), float(new_norm), float(new_tail)
                if (math.isfinite(candidate_obj) and math.isfinite(candidate_norm)
                        and math.isfinite(candidate_tail) and candidate_obj <= objective_value
                        and candidate_norm + candidate_tail < norm_value + tail_value):
                    output, objective_value, norm_value, tail_value = (
                        candidate, candidate_obj, candidate_norm, candidate_tail
                    )
                    iterations += 1
                    accepted = True
                    accepted_any = True
                    break
            proposed *= 0.5
        if not accepted or norm_value + tail_value <= effective_tol * 0.95:
            break
    return (output, iterations, objective_value, norm_value, tail_value,
            accepted_any and norm_value + tail_value <= effective_tol * 0.95)


def fit_logistic(
    X: mx.array,
    y: mx.array,
    *,
    sample_weight: mx.array | None = None,
    C: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 200,
    sketch_size: int = 4,
    seed: int = 0,
) -> DeviceLogisticModel:
    """Fit binary logistic regression on a GPU using device-resident MLX arrays.

    Parameters are binary 0/1 targets, finite nonnegative weights, and a fitted
    intercept. The current native path requires nonempty float32 ``X`` with
    512–4096 features and MLX 0.32.x on Apple silicon. ``sketch_size=0`` disables
    the GPU Hessian-sketch warm start. The default four sketch directions stay
    in the centered feature row space in exact arithmetic. Positive L2
    regularization is required; a single-class target is rejected.

    A tightened internal gradient tolerance and the native loss-progress check
    reduce early stopping. The final compensated GPU check uses the original
    float32 input, not merely its rounded centered representation. It is an
    empirical precision safeguard rather than an objective-gap certificate.
    If the loop cannot meet the requested accuracy within ``max_iter`` or a
    numerical check fails, the result is explicitly marked nonconverged.
    """
    if not isinstance(X, mx.array) or X.ndim != 2 or X.dtype != mx.float32:
        raise TypeError("X must be a two-dimensional float32 MLX array")
    n, d = X.shape
    if n == 0 or not 512 <= d <= 4096:
        raise ValueError("device logistic currently requires nonempty X and 512–4096 features")
    if not isinstance(y, mx.array) or y.shape != (n,):
        raise ValueError("y must be a row-aligned MLX vector")
    if sample_weight is not None and (
        not isinstance(sample_weight, mx.array) or sample_weight.shape != (n,)
    ):
        raise ValueError("sample_weight must be a row-aligned MLX vector")
    label_dtypes = (mx.bool_, mx.int8, mx.int16, mx.int32, mx.int64,
                    mx.uint8, mx.uint16, mx.uint32, mx.uint64, mx.float16,
                    mx.bfloat16, mx.float32)
    if y.dtype not in label_dtypes:
        raise TypeError("y must have a supported integer, boolean, or float32-or-narrower dtype")
    if sample_weight is not None and sample_weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError("sample_weight must have float32, float16, or bfloat16 dtype")
    if not math.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and strictly positive")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and strictly positive")
    effective_tol = tol * 0.25
    if not 1.1754943508222875e-38 <= effective_tol <= 3.4028234663852886e38:
        raise ValueError("tol produces an effective tolerance outside normal float32 range")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(sketch_size, int) or isinstance(sketch_size, bool) or not 0 <= sketch_size <= 10:
        raise ValueError("sketch_size must be an integer between 0 and 10")
    version = getattr(mx, "__version__", "")
    if not isinstance(version, str) or re.fullmatch(r"0\.32\.(?:0|[1-9]\d{0,2})", version) is None:
        raise RuntimeError("The native device bridge currently supports MLX 0.32.x only")
    if metal_loop is None:
        raise RuntimeError("The native Apple Metal extension is unavailable")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    with mx.stream(mx.gpu):
        X = mx.contiguous(X)
        valid_labels = mx.all((y == 0) | (y == 1)) & mx.any(y == 0) & mx.any(y == 1)
        y = mx.contiguous(y.astype(mx.float32))
        sw = mx.ones(n, dtype=mx.float32) if sample_weight is None else mx.contiguous(sample_weight.astype(mx.float32))
        valid_weights = mx.all(mx.isfinite(sw) & (sw >= 0))
        unit_weights = mx.all(sw == 1) if sample_weight is not None else None
        mx.eval(valid_weights, valid_labels, *(() if unit_weights is None else (unit_weights,)))
        if not bool(valid_weights):
            raise ValueError("sample_weight must be finite and nonnegative")
        weighted = unit_weights is not None and not bool(unit_weights)
        if not bool(valid_labels):
            raise ValueError("y must contain both binary labels 0 and 1")
        weighted_classes = mx.any((y == 0) & (sw > 0)) & mx.any((y == 1) & (sw > 0))
        mx.eval(weighted_classes)
        if not bool(weighted_classes):
            raise ValueError("positive sample weights must include both binary classes")
        sw_sum = float(n) if sample_weight is None else _weight_sum(sw, n)
        if not math.isfinite(sw_sum) or sw_sum <= 0 or sw_sum > 3.4028234663852886e38:
            raise ValueError("sample_weight must have a finite, positive float32 sum")
        l2 = 1.0 / (C * sw_sum)
        if not math.isfinite(l2) or not 1.1754943508222875e-38 <= l2 <= 3.4028234663852886e38:
            raise ValueError("C and sample_weight produce an L2 scale outside normal float32 range")
        centered, center, finite, parts = prepare_centered(X, y, sw)
        reg = mx.concatenate((mx.full((d,), l2, mx.float32), mx.zeros((1,), mx.float32)))
        denominator = mx.array(sw_sum, mx.float32)
        g, h0 = initial_metric_from_parts(parts, reg, denominator)
        mx.eval(X, centered, center, finite, y, sw, g, h0, reg)
        if not bool(finite):
            raise ValueError("X must contain only finite float32 values")
        if not bool(mx.all(mx.isfinite(centered))):
            raise ValueError("centering overflowed float32; rescale X before fitting")

        seed_arrays = None
        count = 0
        if sketch_size:
            seed_result = build_sketch_history(
                centered, sw, sw_sum, l2, True, h0,
                history_size=10, sketch_size=sketch_size, seed=seed,
            )
            *seed_arrays, count = seed_result
            seed_arrays = tuple(seed_arrays)
        dependencies = (centered, y, sw, g, h0, reg)
        if seed_arrays is not None:
            mx.eval(*seed_arrays)
        groups, feature_groups, rows, blocks = speculative_geometry(n, d)
        direction_threads = 256 if d >= 2048 and hasattr(metal_loop, "run_cg") else 32
        source = source_for(n, d, d + 1, 10, 8, blocks, groups, rows, feature_groups,
                            weighted, direction_threads)
        # Asking the optimizer for a tighter gradient reduces observed objective
        # discrepancies from componentwise stopping, without changing tol's
        # meaning as the final upper bound.
        device_tol = _device_tol(effective_tol)
        if device_tol == 0:
            raise ValueError("tol is too small to represent on the GPU")

        def attempt(seed_values, seed_count, budget):
            # Every attempt gets fresh, independently owned capsules and output.
            # Native WorkGuard retains these through submitted command completion.
            output = mx.zeros((d + 1,), mx.float32)
            mx.eval(output)
            mx.synchronize()
            capsules = tuple(array.__dlpack__() for array in dependencies)
            owners = (capsules, *dependencies)
            seed_capsules = None
            if seed_values is not None and seed_count:
                seed_capsules = tuple(array.__dlpack__() for array in seed_values)
                owners += (seed_capsules, *seed_values)
            device_buffers = (center.__dlpack__(), output.__dlpack__())
            owners += (device_buffers, center, output)
            # Reserve once for seeded or weighted execution. The compensated
            # final score and its original safety margin remain authoritative.
            native_tol = _seeded_native_tol(device_tol) if (weighted or (seed_values is not None and seed_count)) else device_tol
            arguments = (
                source, capsules, n, d, d + 1, 10, blocks, groups, budget,
                sw_sum, native_tol, 1e-4, 1e-12, owners, b"",
                seed_capsules, seed_count, device_buffers,
            )
            if weighted:
                arguments += (1,)
            _, iterations, status, _, last_delta, previous_delta = metal_loop.run(*arguments)
            objective, norm, _, _, tail = score_original(X, center, output, y, sw, sw_sum, l2)
            mx.eval(objective, norm, tail)
            objective_value, norm_value, tail_value = float(objective), float(norm), float(tail)
            recent = (
                abs(last_delta) if iterations < 2
                else 0.5 * (abs(last_delta) + abs(previous_delta))
            )
            wide_progress = (not weighted and (d < 2048 or iterations == 0)) or norm_value == 0 or (
                (not weighted or iterations >= 2) and math.isfinite(recent)
                and recent <= device_tol * (0.025 if weighted or d >= 2048 else 1.0) * max(abs(objective_value), 1e-12)
            )
            numerically_valid = (
                math.isfinite(objective_value) and math.isfinite(norm_value) and math.isfinite(tail_value)
            )
            # A failed search must never be promoted, even if its last gradient
            # happens to fall below the requested tolerance.
            converged = bool(
                status in (0, 1) and numerically_valid
                and norm_value + tail_value <= device_tol * 0.95 and wide_progress
            )
            return (output, iterations, status, objective_value, norm_value, tail_value,
                    numerically_valid, converged, recent, wide_progress)

        from ._newton import minimize_newton, use_newton_for_curvature

        use_newton = use_newton_for_curvature(parts, n, d, sw_sum, l2)
        (output, iterations, status, objective_value, norm_value, tail_value, finite_result,
         converged, recent, progress_ok) = attempt(
             seed_arrays if count else None, count, min(32, max_iter) if use_newton else max_iter,
         )
        used_fallback = False
        newton_iterations = newton_products = 0
        newton_gap = newton_reason = None
        if use_newton:
            if iterations < max_iter and finite_result and status != 3:
                refinement = minimize_newton(
                    X, centered, center, y, sw, sw_sum, l2,
                    tol=device_tol, max_iter=max_iter - iterations, initial=output,
                )
                output = refinement.theta
                newton_iterations, newton_products = refinement.n_iter, refinement.hvp_count
                iterations += newton_iterations
                objective_value, norm_value, tail_value = (
                    refinement.objective, refinement.gradient_max_abs, refinement.tail_bound
                )
                converged = refinement.converged
                newton_gap, newton_reason = refinement.relative_gap, refinement.stop_reason
                finite_result = all(math.isfinite(value) for value in (objective_value, norm_value, tail_value))
            else:
                converged = False
                newton_reason = "non-finite-objective" if not finite_result or status == 3 else "max_iter"
        # Small intercept refinements can repair shifted fits when line-search
        # loss differences have become too small for ordinary float32 totals.
        # The helper uses the remaining accepted-iteration budget, and never
        # bypasses the independent wide progress condition.
        if (not use_newton and not converged and status == 2 and finite_result and progress_ok and d < 2048
                and norm_value <= tol * 100 and iterations < max_iter):
            output, iterations, objective_value, norm_value, tail_value, converged = _polish_intercept(
                X, centered, center, output, y, sw, sw_sum, l2, iterations,
                max_iter, device_tol, objective_value, norm_value, tail_value,
            )
            if weighted and converged:
                converged = recent <= device_tol * 0.025 * max(abs(objective_value), 1e-12)
            if converged:
                status = 1
        # A sketch can produce an invalid or poor direction on some inputs.
        # Retry with the ordinary scalar metric and no history, but never spend
        # more than the user's total accepted-iteration budget.
        if not use_newton and not converged and count and iterations < max_iter and finite_result and status != 3:
            used_fallback = True
            remaining = max_iter - iterations
            previous = (output, status, objective_value, norm_value, tail_value, finite_result, recent, progress_ok)
            (retry_output, retry_iterations, retry_status, retry_objective, retry_norm,
             retry_tail, retry_finite, retry_converged, retry_recent, retry_progress) = attempt(None, 0, remaining)
            iterations += retry_iterations
            # Retain the lower finite objective if the retry does not converge,
            # but never hide a non-finite failure behind the prior candidate.
            if (not retry_converged and previous[5]
                    and (not retry_finite or retry_objective > previous[2])):
                (output, previous_status, objective_value, norm_value, tail_value,
                 finite_result, recent, progress_ok) = previous
                status = 3 if retry_status == 3 or not retry_finite else previous_status
                converged = False
            else:
                (output, status, objective_value, norm_value, tail_value,
                 finite_result, converged, recent, progress_ok) = (retry_output, retry_status, retry_objective,
                     retry_norm, retry_tail, retry_finite, retry_converged, retry_recent, retry_progress)
            if (not converged and status == 2 and finite_result and progress_ok and d < 2048
                    and norm_value <= tol * 100 and iterations < max_iter):
                output, iterations, objective_value, norm_value, tail_value, converged = _polish_intercept(
                    X, centered, center, output, y, sw, sw_sum, l2, iterations,
                    max_iter, device_tol, objective_value, norm_value, tail_value,
                )
                if weighted and converged:
                    converged = recent <= device_tol * 0.025 * max(abs(objective_value), 1e-12)
                if converged:
                    status = 1
        reason = "gtol" if converged else (
            "non-finite-objective" if status == 3 or not finite_result else
            "line-search-failure" if status == 2 else
            "max_iter" if iterations >= max_iter else "uncertified"
        )
        if newton_reason is not None:
            reason = newton_reason
        model = DeviceLogisticModel(
            coef_=output[:d].reshape(1, d), center_=center, centered_intercept_=output[d],
            n_iter=iterations, converged=converged, stop_reason=reason,
            objective=objective_value, gradient_max_abs=norm_value, sketch_count=count,
            used_fallback=used_fallback, discarded_tail_bound=tail_value,
            newton_iterations=newton_iterations, newton_hvp_count=newton_products,
            relative_duality_gap=newton_gap,
        )
    if not converged:
        warnings.warn(
            f"Device logistic did not converge ({reason}) after {iterations} iterations; "
            f"original-coordinate max gradient {norm_value:.3g} versus tol={tol:g}",
            DeviceConvergenceWarning,
            stacklevel=2,
        )
    return model
