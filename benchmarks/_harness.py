"""Timing that can be believed.

MLX is lazy. A benchmark that calls a fit and then stops the clock is timing how
long it took to *schedule* the work, and the number it reports will be
spectacular and meaningless. Everything here forces evaluation before stopping
the clock.

The other rules, each of which exists because averaging hides something:

* **Warmup is excluded.** The first MLX calls in a process pay for Metal
  initialization and kernel compilation, so every measurement is preceded by
  untimed warmup calls. The harness does *not* report that first-call cost; a
  user who makes exactly one call pays it, and no number here describes them.
* **Fit and predict are reported separately.** An estimator can be twice as fast
  to fit and half as fast to predict; a combined number says nothing about either.
* **Median of at least five after warmup, plus the interquartile range.** A mean
  over a noisy machine is dominated by whatever else was running. The spread is
  what tells you whether a difference is real, and it has to be a real IQR: the
  full min-max range lets one garbage-collection pause hide a 2x regression.

Comparisons against scikit-learn are paired and interleaved rather than "all of A
then all of B", so that a thermal ramp or a background process during the run
degrades both sides equally instead of whichever went second. Within a pair the
two sides take turns going first — A B, B A, A B — and every timed call gets the
same preparation, a garbage collection before it and a synchronization inside its
window: a call made straight after ``gc.collect()`` runs on cold caches, and when
that was always the mlxlearn side, identical work on both sides read as a 10-25%
loss and failed the gate.
"""

from __future__ import annotations

import gc
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

__all__ = ["Measurement", "Comparison", "measure", "compare", "environment", "interquartile_range"]


@dataclass(frozen=True, slots=True)
class Measurement:
    """One timed operation."""

    label: str
    median_s: float
    min_s: float
    max_s: float
    iqr_s: float
    iterations: int

    def as_dict(self) -> dict:
        return asdict(self)


#: Fewer samples than this cannot support a significance claim: the quartiles of
#: three numbers are two of the three numbers.
MIN_SIGNIFICANCE_SAMPLES = 5

#: How many standard errors apart two medians must be to count as different. 3.0
#: rather than the textbook 1.96 because the standard errors are themselves
#: estimated from a handful of samples: at 7 iterations this keeps the false-alarm
#: rate per case near or below 1% under Gaussian, log-normal and outlier-contaminated
#: noise (simulated), while still catching a 1.3x regression that carries a 3x
#: outlier. That rate is per case, and a nightly run gates well over a hundred: with
#: 137 cases of identical work (simulated, Gaussian and log-normal noise) 73-79% of
#: nights failed on one measurement each. So the runner re-measures any case the gate
#: would fail, and the gate fails it only if the loss reproduces there
#: (``CONFIRMATION_FACTOR`` in ``benchmarks/run.py``): 0.4-0.7% of nights.
SIGNIFICANCE_Z = 3.0


def _sync() -> None:
    """Force every pending MLX computation to complete."""
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover
        return
    mx.synchronize()


def interquartile_range(samples: list[float]) -> float:
    """The IQR, with linear interpolation between order statistics.

    The one place the spread is computed, so the single-callable and paired
    measurements cannot drift apart again. The previous ``samples[-q] - samples[q-1]``
    with ``q = n // 4`` was the full range for n <= 7 — the default — so a single
    outlier set the "noise" for the whole case.
    """
    if len(samples) < 2:
        return 0.0
    q1, _median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return q3 - q1


def median_standard_error(measurement: Measurement) -> float:
    """Standard error of a sample median, estimated robustly from the IQR.

    ``1.2533 * sigma / sqrt(n)`` for a normal distribution, with ``sigma`` estimated
    as ``IQR / 1.349`` so that one outlier does not inflate it.
    """
    sigma = measurement.iqr_s / 1.349
    return 1.2533 * sigma / measurement.iterations**0.5


def _summarize(samples: list[float], label: str) -> Measurement:
    samples = sorted(samples)
    return Measurement(
        label=label,
        median_s=statistics.median(samples),
        min_s=samples[0],
        max_s=samples[-1],
        iqr_s=interquartile_range(samples),
        iterations=len(samples),
    )


def _timed(fn: Callable[[], object], setup: Callable[[], object] | None = None) -> float:
    """One synchronized call, after ``setup``, a collection, and with the collector off."""
    if setup is not None:
        setup()
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        fn()
        _sync()
        return time.perf_counter() - start
    finally:
        gc.enable()


def measure(
    fn: Callable[[], object],
    *,
    label: str,
    warmup: int = 2,
    iterations: int = 7,
) -> Measurement:
    """Time ``fn`` with warmup, synchronization, and a median over ``iterations``."""
    for _ in range(warmup):
        fn()
    _sync()
    return _summarize([_timed(fn) for _ in range(iterations)], label)


@dataclass(frozen=True, slots=True)
class Comparison:
    """mlxlearn against stock scikit-learn on the same work."""

    label: str
    shape: tuple[int, ...]
    mlxlearn: Measurement
    sklearn: Measurement
    extra: dict = field(default_factory=dict)

    @property
    def speedup(self) -> float:
        """Median against median — what a user typically experiences."""
        return self.sklearn.median_s / self.mlxlearn.median_s

    @property
    def speedup_best(self) -> float:
        """Best case against best case.

        Reported alongside :attr:`speedup` because at small sizes scikit-learn's
        timings are heavy-tailed — thread-pool scheduling produces a median
        several times its own minimum — and a ratio of medians can then flatter
        mlxlearn. Measured on 500x8 neighbor queries: scikit-learn's median was
        26.3 ms and its minimum 1.55 ms, so the same case reads as 17x on medians
        and 1.2x on minima. Both numbers are true; publishing only the first
        would not be.
        """
        return self.sklearn.min_s / self.mlxlearn.min_s

    @property
    def significant(self) -> bool:
        """Whether the two medians differ by more than their uncertainty.

        Without a test like this, a 3% "speedup" on a machine with 10% run-to-run
        spread gets reported as a win. The gate in ``benchmarks/gate.py`` reads
        this field, not the raw ratio.

        The uncertainty is the standard error *of the median* on each side,
        combined in quadrature, and the medians must be ``SIGNIFICANCE_Z`` of those
        apart. Two things matter here. It is the median's standard error, not the
        spread of individual samples: scikit-learn's timings are heavy-tailed, and
        comparing against the raw spread declared almost nothing significant,
        including a measured 10x. And the spread is a real IQR, so a single outlier
        can no longer declare a real regression "within noise".
        """
        if min(self.mlxlearn.iterations, self.sklearn.iterations) < MIN_SIGNIFICANCE_SAMPLES:
            return False
        uncertainty = (
            median_standard_error(self.mlxlearn) ** 2 + median_standard_error(self.sklearn) ** 2
        ) ** 0.5
        return abs(self.sklearn.median_s - self.mlxlearn.median_s) > SIGNIFICANCE_Z * uncertainty

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "shape": list(self.shape),
            "mlxlearn": self.mlxlearn.as_dict(),
            "sklearn": self.sklearn.as_dict(),
            "speedup": self.speedup,
            "speedup_best": self.speedup_best,
            "significant": self.significant,
            **self.extra,
        }


def compare(
    mlx_fn: Callable[[], object],
    sk_fn: Callable[[], object],
    *,
    label: str,
    shape: tuple[int, ...],
    warmup: int = 2,
    iterations: int = 7,
    mlx_setup: Callable[[], object] | None = None,
    **extra,
) -> Comparison:
    """Time both implementations, interleaved, each side going first in turn.

    Interleaving matters more than it sounds: a run that measures all of A and
    then all of B attributes any thermal ramp or background load entirely to
    whichever ran second. The order within a pair matters too, which is why it
    alternates and why each call is timed exactly as :func:`measure` times one —
    collected before, synchronized inside — whichever backend it ran on.

    ``mlx_setup``, when given, runs before every mlxlearn call, warm-up included,
    outside the timed window: to time a first call — one that builds something the
    model then keeps — as often as the comparison needs.
    """
    for _ in range(warmup):
        if mlx_setup is not None:
            mlx_setup()
        mlx_fn()
        sk_fn()
    _sync()

    mlx_samples: list[float] = []
    sk_samples: list[float] = []
    for i in range(iterations):
        if i % 2 == 0:
            mlx_samples.append(_timed(mlx_fn, mlx_setup))
            sk_samples.append(_timed(sk_fn))
        else:
            sk_samples.append(_timed(sk_fn))
            mlx_samples.append(_timed(mlx_fn, mlx_setup))

    return Comparison(
        label=label,
        shape=shape,
        mlxlearn=_summarize(mlx_samples, f"{label}[mlxlearn]"),
        sklearn=_summarize(sk_samples, f"{label}[sklearn]"),
        extra=extra,
    )


def environment() -> dict:
    """Record what the numbers were produced on.

    A benchmark result without its hardware is not a result. Absolute timings are
    only comparable across runs on the same machine, which is why the nightly job
    pins itself to fixed self-hosted hardware.
    """
    info: dict = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import mlx.core as mx

        info["mlx"] = getattr(mx, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        info["mlx"] = "unavailable"
    for module in ("numpy", "scipy", "sklearn"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:  # pragma: no cover
            info[module] = "unavailable"

    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.split("\n")
            info["cpu"] = out[0].strip()
            info["memory_bytes"] = int(out[1])
            info["cores"] = int(out[2])
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            pass

    try:
        from mlxlearn._common.device import device_info

        info["device"] = str(device_info())
    except Exception:  # pragma: no cover
        pass

    import mlxlearn
    from mlxlearn._common.config import get_tuning

    info["mlxlearn"] = mlxlearn.__version__
    # The thresholds dispatch ran with, and any MLXLEARN_* variable behind them: a
    # result the gate judges against Tuning has to say which Tuning that was.
    info["tuning"] = asdict(get_tuning())
    info["mlxlearn_environment"] = {
        name: value for name, value in sorted(os.environ.items()) if name.startswith("MLXLEARN_")
    }
    return info
