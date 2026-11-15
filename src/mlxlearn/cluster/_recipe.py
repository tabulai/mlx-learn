"""Published e-means laws and explicitly versioned Faiss-compatible recipes.

Formulae are from Bessa et al., arXiv:2611.09501v1, Equations 5--7.
``faiss`` reproduces the constants and boundary conventions in
facebookresearch/faiss/contrib/e_means/base.py (accessed 2026-11-14).
"""

from __future__ import annotations

import math
from numbers import Integral, Real


def round_power_of_two(value: float) -> int:
    """Round in log space, resolving geometric midpoint ties upwards."""
    if value <= 1:
        return 1
    low = 2 ** math.floor(math.log2(value))
    return low if value / low < 2 * low / value else 2 * low


def epoch_rows(n: int, epochs: float) -> list[int]:
    """Keep one complete closing pass whenever the budget permits one.

    For example 2.5 epochs is a full pass, a half pass, and a full closing
    pass. This matches Faiss's fractional-budget extension to Algorithm 1.
    """
    whole = math.floor(epochs)
    partial = round((epochs - whole) * n)
    if whole == 0:
        return [partial]
    return [n] * (whole - 1) + ([partial] if partial else []) + [n]


def scheduled_alpha(alpha0: float, step: int, total_steps: int) -> float:
    """Equation 5; update indices are zero-based, as in the reference code."""
    fraction = min(step, total_steps) / max(total_steps, 1)
    return alpha0 * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))


def _positive_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def resolve_recipe(
    n: int,
    k: int,
    d: int,
    epochs: float,
    batch_size: int | None = None,
    alpha0: float | None = None,
    batch_law: str = "auto",
    recipe: str = "paper",
    unused_threshold: float | None = None,
) -> dict:
    """Resolve the full statistical recipe, including differences from Faiss.

    ``paper`` follows Table 2, uses 1e-10 initial mass, and BF16 epsilon for
    splits. ``faiss`` uses the more precise shipped fit, its half-dataset
    batch caps and low-dimensional heuristics, 2**-16 mass, and 1e-4 splits.
    Both retain remainder batches and support fractional epoch budgets.
    """
    for name, value in (("n", n), ("k", k), ("d", d)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if k > n:
        raise ValueError("n_clusters must not exceed the number of samples")
    _positive_real("max_iter", epochs)
    if recipe not in {"paper", "faiss"}:
        raise ValueError("recipe must be 'paper' or 'faiss'")
    if batch_law not in {"auto", "quality", "throughput"}:
        raise ValueError("batch_law must be 'auto', 'quality', or 'throughput'")
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer or None")
    if alpha0 is not None:
        _positive_real("alpha0", alpha0)
        if alpha0 > 1:
            raise ValueError("alpha0 must be in (0, 1]")
    if unused_threshold is not None and (
        not isinstance(unused_threshold, Real)
        or not math.isfinite(unused_threshold)
        or not 0 <= unused_threshold < 1
    ):
        raise ValueError("unused_threshold must be finite and in [0, 1)")
    if batch_law == "auto":
        if recipe == "paper":
            law = "quality" if n > 2_000_000 else "throughput"
            reason = "paper: n > 2M" if law == "quality" else "paper: n <= 2M"
        else:
            law = "quality" if d <= 16 else "throughput"
            reason = "faiss: d <= 16" if law == "quality" else "faiss: d > 16"
    else:
        law, reason = batch_law, "requested"
    if recipe == "paper":
        coefficient, k_exp, n_exp, t_exp = 0.035, 0.13, 0.88, 0.81
        step_law = (0.824, -0.46, 0.90, -0.45, -0.11)
        mass, split_epsilon = 1e-10, 2**-7
    else:
        coefficient, k_exp, n_exp, t_exp = 0.035134, 0.134749, 0.876473, 0.813344
        step_law = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
        mass, split_epsilon = 2**-16, 1e-4
    if law == "quality":
        coefficient /= 8
    if batch_size is None:
        raw = coefficient * k**k_exp * n**n_exp * epochs**t_exp
        batch = round_power_of_two(raw)
        if recipe == "faiss":
            cap = round_power_of_two(n / 2) if law == "throughput" else n // 2
            batch = min(batch, cap)
        batch = max(1, min(batch, n))
    else:
        batch = min(int(batch_size), n)
    if batch > 2**24:
        raise ValueError("batch_size must not exceed 2**24 for exact float32 batch counts")
    if alpha0 is None:
        c, ak, ab, an, at = step_law
        alpha = min(c * k**ak * batch**ab * n**an * epochs**at, 1.0)
        if recipe == "faiss":
            alpha = max(alpha, 1e-4)
    else:
        alpha = float(alpha0)
    rows = epoch_rows(n, epochs)
    if sum(rows) == 0:
        raise ValueError("max_iter is too small to include even one training sample")
    steps = [(r + batch - 1) // batch for r in rows]
    threshold = unused_threshold
    if threshold is None:
        threshold = 0.0 if recipe == "faiss" and d <= 2 else 0.1
    return {
        "name": recipe,
        "batch_law": law if batch_size is None else None,
        "batch_law_requested": batch_law,
        "batch_law_reason": reason,
        "batch_size": batch,
        "alpha0": alpha,
        "nepochs": float(epochs),
        "epoch_rows": rows,
        "epoch_steps": steps,
        "total_steps": sum(steps),
        "alpha_min_factor": 0.1,
        "init_usage": mass,
        "split_epsilon": split_epsilon,
        "unused_threshold": float(threshold),
        "check_unused_every": 5,
        "tail_policy": "keep",
        "fractional_policy": "full_closing_pass_when_T_ge_1",
    }
