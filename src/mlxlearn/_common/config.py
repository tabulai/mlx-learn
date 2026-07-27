"""Configuration: a small public surface over a larger private one.

Two layers, on purpose.

:class:`Config` is public and deliberately tiny — six options that change what a
user observes: which device to use, what to do when mlxlearn cannot serve a
request, what array type comes back, whether results are reproducible, the seed,
and whether diagnostics are recorded. It is thread-local and scoped, so a library
that sets a policy inside :func:`config_context` cannot leak it into a caller's
thread.

:class:`Tuning` is private. Crossover thresholds, block sizes, cache fractions and
solver limits are implementation details that will move as the benchmarks improve;
promoting them to public options would freeze today's implementation into the API.
They are overridable through ``MLXLEARN_*`` environment variables for CI and
debugging, and that is the whole supported surface for them.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, get_args

__all__ = [
    "Config",
    "Tuning",
    "get_config",
    "set_config",
    "config_context",
    "get_tuning",
    "tuning_context",
    "reset_config",
]

Device = Literal["auto", "gpu", "cpu"]
FallbackPolicy = Literal["warn", "raise", "silent"]
OutputType = Literal["numpy", "input", "mlx"]


# --------------------------------------------------------------------------------------
# public configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """User-visible configuration. Immutable; replace it, do not mutate it.

    Attributes
    ----------
    device : {"auto", "gpu", "cpu"}
        Which MLX device to compute on. ``"auto"`` prefers the GPU and falls back
        to the CPU when no Metal device is usable. The names are MLX's, not
        Metal's: there is deliberately no ``"mps"``.
    fallback_policy : {"warn", "raise", "silent"}
        What patched mode does on a *capability* mismatch. ``"warn"`` falls back
        and warns once per estimator class and reason; ``"raise"`` turns any
        fallback into an error, which is how you assert that a workload is fully
        accelerated; ``"silent"`` falls back quietly, for production and agent
        harnesses where a warning stream is just noise. Unexpected backend
        failures raise under every policy.
    output_type : {"numpy", "input", "mlx"}
        What array type estimators return. 0.1.0 supports ``"numpy"`` only;
        ``"input"`` and ``"mlx"`` are reserved for the 0.2.x experimental MLX
        boundary and are rejected today rather than silently ignored.
    deterministic : bool
        When true, mlxlearn takes reproducible paths — fixed reduction orders,
        seeded initialization, stable tie-breaking — even where a faster
        nondeterministic path exists.
    random_state : int or None
        Seed for the library-level RNG. Estimator-level ``random_state``
        parameters take precedence over this.
    diagnostics : bool
        Whether backend events are recorded. Recording is cheap and it is what
        makes a fallback debuggable after the fact, so it defaults on.
    """

    device: Device = "auto"
    fallback_policy: FallbackPolicy = "warn"
    output_type: OutputType = "numpy"
    deterministic: bool = True
    random_state: int | None = None
    diagnostics: bool = True

    def __post_init__(self) -> None:
        _validate_choice("device", self.device, get_args(Device))
        _validate_choice("fallback_policy", self.fallback_policy, get_args(FallbackPolicy))
        _validate_choice("output_type", self.output_type, get_args(OutputType))
        if self.output_type != "numpy":
            raise ValueError(
                f"output_type={self.output_type!r} is reserved for the 0.2.x experimental "
                "MLX array boundary and is not implemented in this release. "
                "mlxlearn 0.1.0 takes NumPy in and returns NumPy out."
            )
        if not isinstance(self.deterministic, bool):
            raise TypeError(f"deterministic must be a bool, got {type(self.deterministic).__name__}")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(
                f"random_state must be an int or None, got {type(self.random_state).__name__}"
            )
        if not isinstance(self.diagnostics, bool):
            raise TypeError(f"diagnostics must be a bool, got {type(self.diagnostics).__name__}")


def _validate_choice(name: str, value: object, allowed: tuple[Any, ...]) -> None:
    if value not in allowed:
        options = ", ".join(repr(a) for a in allowed)
        raise ValueError(f"{name} must be one of {options}; got {value!r}")


# --------------------------------------------------------------------------------------
# private tuning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tuning:
    """Private tuning knobs. Not public API; may change in any release.

    Every default here is a *measurement*, or a placeholder waiting for one. The
    crossover thresholds are the sizes below which stock scikit-learn wins, taken
    from ``docs/benchmarks.md``; dispatch reads them so that "we are never slower"
    is enforced by handing small problems away rather than by hoping.
    """

    # ---- dispatch crossovers: below these, stock scikit-learn is used ----------------
    #
    # Measured, not guessed. See docs/benchmarks.md for the runs and
    # `benchmarks/run.py` to reproduce them. A problem is "below the crossover"
    # only if it is small on *both* counts, so wide data at low row counts still
    # reaches MLX.
    #
    # KNN queries beat scikit-learn from about 250 rows upward — 2.9x to 17x on
    # medians, 1.2x to 5.6x comparing best case to best case. The threshold is
    # therefore low. It was 4096 as a placeholder, which gave away most of that.
    knn_min_train_samples: int = 256
    knn_min_work: int = 2_048  # n_train * n_features
    #
    # Logistic regression is the opposite story, and the thresholds are high on
    # purpose. Its L-BFGS matches scikit-learn's iteration count but pays a host
    # synchronization per line-search evaluation, so an iteration costs ~5.9 ms
    # where its matmul costs 163 us. Measured 0.02x-0.11x below 60 000 x 128, and
    # 1.26x-1.34x only at 1024 features. Both conditions must be met (see
    # LogisticRegression._below_crossover) rather than either.
    logreg_min_samples: int = 20_000
    logreg_min_work: int = 20_000_000  # n_samples * n_features
    #
    # SVC is the same story again, and worse -- but the axis that matters is
    # *width*, not height. Over a 64-configuration grid at d in {10, 20} mlxlearn
    # was slower in 58 of 64, worst 0.013x. What separates the wins is the cost of
    # a kernel row: rbf at 4000 x 32 is 0.47x, and the same n at 4000 x 256 is
    # 2.93x. Both conditions must hold (see SVC._below_crossover), and the linear
    # kernel is excluded at every size.
    svc_min_samples: int = 2_048
    svc_min_work: int = 1_000_000  # n_samples * n_features

    # ---- blocking ---------------------------------------------------------------------
    distance_block_bytes: int = 256 * 1024 * 1024
    distance_block_bytes_max: int = 1024 * 1024 * 1024
    query_block_rows: int = 1024
    query_block_align: int = 128
    cpu_distance_block_bytes: int = 128 * 1024 * 1024
    cpu_query_block_rows: int = 256

    # ---- solvers ----------------------------------------------------------------------
    lbfgs_history: int = 10
    lbfgs_max_line_search: int = 32
    smo_kernel_cache_fraction: float = 0.05  # of physical memory
    smo_kernel_cache_min_bytes: int = 32 * 1024 * 1024
    smo_kernel_cache_max_bytes: int = 2 * 1024 * 1024 * 1024
    smo_max_iter_factor: int = 100  # max_iter=-1 -> factor * n_samples, bounded below
    smo_min_iter_budget: int = 10_000_000

    # ---- behavior ---------------------------------------------------------------------
    #: Force a specific backend regardless of size. For tests and bisection only.
    force_backend: Literal["", "mlx", "cpu", "sklearn"] = ""
    diagnostics_capacity: int = 512

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, int) and not isinstance(value, bool) and value < 0:
                raise ValueError(f"tuning value {f.name} must be non-negative, got {value}")
        if not 0.0 < self.smo_kernel_cache_fraction <= 1.0:
            raise ValueError(
                "smo_kernel_cache_fraction must be in (0, 1], got "
                f"{self.smo_kernel_cache_fraction}"
            )


_ENV_PREFIX = "MLXLEARN_"


def _tuning_from_env(base: Tuning) -> Tuning:
    """Apply ``MLXLEARN_*`` overrides on top of ``base``.

    A malformed value is an error, not a silent fallback to the default: an
    environment variable that quietly does nothing is worse than no override at
    all, because you cannot tell the two apart from the outside.
    """
    overrides: dict[str, Any] = {}
    for f in fields(Tuning):
        key = _ENV_PREFIX + f.name.upper()
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            if f.type is int or f.type == "int":
                overrides[f.name] = int(raw)
            elif f.type is float or f.type == "float":
                overrides[f.name] = float(raw)
            else:
                overrides[f.name] = raw
        except ValueError as exc:
            raise ValueError(f"{key}={raw!r} is not a valid value for {f.name}") from exc
    return replace(base, **overrides) if overrides else base


# --------------------------------------------------------------------------------------
# thread-local state
# --------------------------------------------------------------------------------------


class _State(threading.local):
    config: Config
    tuning: Tuning

    def __init__(self) -> None:
        # threading.local re-runs __init__ per thread, which is exactly what we
        # want: a new thread starts from the defaults rather than inheriting a
        # scoped override from whoever spawned it.
        self.config = Config()
        self.tuning = _tuning_from_env(Tuning())


_state = _State()


def get_config() -> Config:
    """Return the current thread's configuration."""
    return _state.config


def set_config(**kwargs: Any) -> Config:
    """Update the current thread's configuration and return the new value.

    Only the fields named are changed. Unknown field names are an error rather
    than being stored and ignored.

    Examples
    --------
    >>> import mlxlearn
    >>> _ = mlxlearn.set_config(fallback_policy="raise")
    >>> mlxlearn.get_config().fallback_policy
    'raise'
    >>> _ = mlxlearn.set_config(fallback_policy="warn")
    """
    known = {f.name for f in fields(Config)}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(
            f"unknown configuration option(s): {', '.join(sorted(unknown))}. "
            f"Valid options are: {', '.join(sorted(known))}."
        )
    _state.config = replace(_state.config, **kwargs)
    return _state.config


@contextmanager
def config_context(**kwargs: Any):
    """Temporarily override configuration for the current thread.

    The previous configuration is restored even if the body raises.

    Examples
    --------
    >>> import mlxlearn
    >>> with mlxlearn.config_context(fallback_policy="raise"):
    ...     mlxlearn.get_config().fallback_policy
    'raise'
    >>> mlxlearn.get_config().fallback_policy
    'warn'
    """
    previous = _state.config
    try:
        set_config(**kwargs)
        yield _state.config
    finally:
        _state.config = previous


def get_tuning() -> Tuning:
    """Return the current thread's private tuning values."""
    return _state.tuning


@contextmanager
def tuning_context(**kwargs: Any):
    """Temporarily override private tuning. Tests and debugging only."""
    known = {f.name for f in fields(Tuning)}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(f"unknown tuning option(s): {', '.join(sorted(unknown))}")
    previous = _state.tuning
    try:
        _state.tuning = replace(previous, **kwargs)
        yield _state.tuning
    finally:
        _state.tuning = previous


def reset_config() -> None:
    """Restore this thread's configuration and tuning to their defaults."""
    _state.config = Config()
    _state.tuning = _tuning_from_env(Tuning())


# Importable so a test can assert the public surface has not silently grown. The
# plan fixes this list at six entries; the 50 environment variables of the
# ancestor do not get to become 50 public options by accident.
PUBLIC_CONFIG_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Config))
