"""Backend event recording.

Every dispatch decision mlxlearn makes is recorded: which estimator, which
operation, which backend ran it, and — when it did not run on MLX — why not.
This is what makes a fallback debuggable after the fact instead of at the moment
it happens, and it is recorded under *every* fallback policy, including
``"silent"``. Silencing a warning stream is a reasonable thing to want; losing the
evidence is not.

The record is a bounded ring buffer. An agent harness fitting in a loop for hours
must not accumulate an unbounded event list, and the interesting events are the
recent ones. Warnings are deduplicated per ``(estimator, reason)`` because an
estimator that falls back once inside a loop falls back every iteration.
"""

from __future__ import annotations

import threading
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..exceptions import MLXLearnFallbackWarning
from .config import get_config, get_tuning

__all__ = [
    "BackendEvent",
    "record_event",
    "get_backend_diagnostics",
    "get_last_backend_event",
    "clear_backend_diagnostics",
    "diagnostics_summary",
]

Backend = Literal["mlx", "cpu", "sklearn"]


@dataclass(frozen=True, slots=True)
class BackendEvent:
    """One dispatch decision.

    Attributes
    ----------
    estimator : str
        Class name, e.g. ``"KNeighborsClassifier"``.
    operation : str
        ``"fit"``, ``"predict"``, ``"kneighbors"``, and so on.
    backend : {"mlx", "cpu", "sklearn"}
        What actually ran the work.
    reason : str
        Why this backend. ``"selected"`` when MLX ran it; otherwise a slug such as
        ``"sparse-input"``, ``"below-crossover"``, ``"unsupported-parameter:solver"``,
        or ``"sticky"`` for an inference call following its fit.
    detail : str
        Human-readable explanation, empty when there is nothing to add.
    shape : tuple of int or None
        ``(n_samples, n_features)`` when known. Makes threshold decisions
        auditable after the fact.
    fallback : bool
        Whether this event represents work mlxlearn handed to scikit-learn.
    """

    estimator: str
    operation: str
    backend: Backend
    reason: str = "selected"
    detail: str = ""
    shape: tuple[int, ...] | None = None
    fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" {self.shape}" if self.shape else ""
        why = "" if self.reason == "selected" else f" ({self.reason})"
        return f"{self.estimator}.{self.operation}{where} -> {self.backend}{why}"


class _Recorder:
    """Process-wide, lock-guarded ring buffer.

    Process-wide rather than thread-local on purpose: a user debugging why their
    fit fell back should not have to know which thread scikit-learn's ``n_jobs``
    machinery happened to run it on.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[BackendEvent] = deque(maxlen=get_tuning().diagnostics_capacity)
        self._warned: set[tuple[str, str]] = set()

    def record(self, event: BackendEvent) -> None:
        with self._lock:
            self._events.append(event)

    def should_warn(self, estimator: str, reason: str) -> bool:
        key = (estimator, reason)
        with self._lock:
            if key in self._warned:
                return False
            self._warned.add(key)
            return True

    def events(self) -> list[BackendEvent]:
        with self._lock:
            return list(self._events)

    def last(self) -> BackendEvent | None:
        with self._lock:
            return self._events[-1] if self._events else None

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._warned.clear()

    def resize(self, capacity: int) -> None:
        with self._lock:
            self._events = deque(self._events, maxlen=capacity)


_recorder = _Recorder()


def record_event(
    estimator: str,
    operation: str,
    backend: Backend,
    *,
    reason: str = "selected",
    detail: str = "",
    shape: tuple[int, ...] | None = None,
    fallback: bool = False,
    warn: bool = False,
    **extra: Any,
) -> BackendEvent:
    """Record one dispatch decision, optionally warning about it.

    Parameters
    ----------
    warn : bool, default=False
        Emit :class:`~mlxlearn.exceptions.MLXLearnFallbackWarning` the first time
        this ``(estimator, reason)`` pair is seen, subject to
        ``fallback_policy``. Recording happens regardless.

    Returns
    -------
    BackendEvent
        The event, whether or not diagnostics are enabled, so callers can attach
        it to an estimator without re-deriving it.
    """
    config = get_config()
    event = BackendEvent(
        estimator=estimator,
        operation=operation,
        backend=backend,
        reason=reason,
        detail=detail,
        shape=tuple(shape) if shape is not None else None,
        fallback=fallback,
        extra=dict(extra),
    )
    if config.diagnostics:
        _recorder.record(event)

    if warn and config.fallback_policy == "warn" and _recorder.should_warn(estimator, reason):
        message = detail or f"{estimator}.{operation} fell back to scikit-learn ({reason})."
        warnings.warn(
            f"{message} This warning is shown once per estimator and reason; "
            "call mlxlearn.get_backend_diagnostics() for the full record.",
            MLXLearnFallbackWarning,
            stacklevel=3,
        )
    return event


def get_backend_diagnostics() -> list[BackendEvent]:
    """Return the recorded backend events, oldest first.

    Returns an empty list when ``diagnostics=False``, which is
    indistinguishable from "nothing happened" — that is the cost of turning
    recording off.
    """
    return _recorder.events()


def get_last_backend_event() -> BackendEvent | None:
    """Return the most recent backend event, or ``None``."""
    return _recorder.last()


def clear_backend_diagnostics() -> None:
    """Drop all recorded events and reset warning deduplication."""
    _recorder.clear()


def diagnostics_summary() -> dict[str, int]:
    """Count recorded events by ``"estimator.operation -> backend"``.

    A quick way to answer "did any of this actually run on the GPU?" without
    reading a few hundred events.
    """
    counts: dict[str, int] = {}
    for event in _recorder.events():
        key = f"{event.estimator}.{event.operation} -> {event.backend}"
        counts[key] = counts.get(key, 0) + 1
    return counts
