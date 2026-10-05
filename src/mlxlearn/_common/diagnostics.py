"""Backend event recording.

Every dispatch decision mlxlearn makes is recorded: which estimator, which
operation, which backend ran it, and — when it did not run on MLX — why not.
This is what makes a fallback debuggable after the fact instead of at the moment
it happens, and it is recorded under *every* fallback policy, including
``"silent"``. Silencing a warning stream is a reasonable thing to want; losing the
evidence is not. A fit's decision is recorded when it is made, so a fit that then
fails is still on the record; a fit that revises its decision records the revision
in the same place, so one fit is one event.

The record is a bounded ring buffer. An agent harness fitting in a loop for hours
must not accumulate an unbounded event list, and the interesting events are the
recent ones. Each fitted estimator also keeps the event its own ``fit``
committed (``fit_event_``), which survives the ring buffer's eviction and travels
with the model when it is pickled.

Warnings are deduplicated per ``(estimator, reason)`` because an estimator that
falls back once inside a loop falls back every iteration. A key is used up only
by a warning that was actually shown: one hidden by an ``"ignore"`` filter, or
turned into an exception by an ``"error"`` filter, leaves it armed.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..exceptions import MLXLearnFallbackWarning
from .config import _process_tuning, get_config

__all__ = [
    "BackendEvent",
    "record_event",
    "get_backend_diagnostics",
    "get_last_backend_event",
    "clear_backend_diagnostics",
    "reset_fallback_warnings",
    "diagnostics_summary",
    "external_stacklevel",
]

Backend = Literal["mlx", "cpu", "sklearn"]

#: Operations that are bookkeeping about patching rather than work on data, left
#: out of :func:`diagnostics_summary`.
_NON_WORK_OPERATIONS = frozenset({"patch", "unpatch"})


@dataclass(frozen=True, slots=True)
class BackendEvent:
    """One dispatch decision.

    Attributes
    ----------
    estimator : str
        Class name, e.g. ``"KNeighborsClassifier"``.
    operation : str
        ``"fit"``, ``"predict"``, ``"kneighbors"``, and so on. For inference, the
        public method the caller invoked, even when it reached the work through
        another one.
    backend : {"mlx", "cpu", "sklearn"}
        What ran the work. For inference events this is the backend that owns
        the fitted state (the sticky backend), which is what routed the call.
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
    extra : dict
        Anything else the recorder attached — ``raised=True`` on a request that
        was refused with an error rather than served, ``fit_reason`` on inference
        events. Not part of equality or the hash.
    """

    estimator: str
    operation: str
    backend: Backend
    reason: str = "selected"
    detail: str = ""
    shape: tuple[int, ...] | None = None
    fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __reduce__(self):
        # Rebuilt through __init__ rather than the default slots/frozen state
        # protocol, which has been fragile across the Python versions supported
        # here. A fitted estimator carries one of these in its pickle.
        return (
            type(self),
            (
                self.estimator,
                self.operation,
                self.backend,
                self.reason,
                self.detail,
                self.shape,
                self.fallback,
                dict(self.extra),
            ),
        )

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" {self.shape}" if self.shape else ""
        why = "" if self.reason == "selected" else f" ({self.reason})"
        return f"{self.estimator}.{self.operation}{where} -> {self.backend}{why}"


class _Recorder:
    """Process-wide, lock-guarded ring buffer.

    Process-wide rather than thread-local on purpose: a user debugging why their
    fit fell back should not have to know which thread scikit-learn's ``n_jobs``
    machinery happened to run it on. It is not *cross*-process: events recorded
    in a process worker (joblib's loky backend, multiprocessing) stay in that
    worker, and so do its warnings' deduplication keys. A model fitted there still
    carries its own ``fit_event_`` back when it is returned.

    Its capacity follows the process-wide ``diagnostics_capacity`` (the
    ``MLXLEARN_DIAGNOSTICS_CAPACITY`` environment variable, read at import and on
    ``reset_config``). A ``tuning_context`` cannot resize it: one thread shrinking
    a buffer every thread shares would silently drop the others' events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[BackendEvent] = deque(
            maxlen=_process_tuning().diagnostics_capacity
        )
        self._warned: set[tuple[str, str]] = set()

    def record(
        self, event: BackendEvent | tuple, replacing: BackendEvent | None = None
    ) -> None:
        """Append ``event`` — or the field tuple of one, built only when it is read.

        An inference call records its event as the tuple (:func:`record_inference`):
        building the frozen dataclass cost a few microseconds on every call, a
        measurable share of a 0.1 ms single-row ``predict``, for an event that is
        usually never read. :meth:`events` and :meth:`last` build it on first read,
        in place, so it is the same object on every read after that.
        """
        capacity = _process_tuning().diagnostics_capacity
        with self._lock:
            if self._events.maxlen != capacity:
                self._events = deque(self._events, maxlen=capacity)
            if replacing is not None:
                # Newest first: the event a fit is revising is usually the last one.
                for position in range(len(self._events) - 1, -1, -1):
                    if self._events[position] is replacing:
                        self._events[position] = event
                        return
            self._events.append(event)

    def _built(self, position: int) -> BackendEvent:
        # Under the lock.
        event = self._events[position]
        if type(event) is tuple:
            event = BackendEvent(*event)
            self._events[position] = event
        return event

    def has_warned(self, key: tuple[str, str]) -> bool:
        with self._lock:
            return key in self._warned

    def claim_warning(self, key: tuple[str, str]) -> bool:
        """Mark ``key`` as warned; ``False`` if another thread got there first."""
        with self._lock:
            if key in self._warned:
                return False
            self._warned.add(key)
            return True

    def reset_warnings(self) -> None:
        with self._lock:
            self._warned.clear()

    def events(self) -> list[BackendEvent]:
        with self._lock:
            return [self._built(position) for position in range(len(self._events))]

    def last(self) -> BackendEvent | None:
        with self._lock:
            return self._built(-1) if self._events else None

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._warned.clear()


_recorder = _Recorder()


# ------------------------------------------------------------------ warning plumbing

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _is_internal(frame) -> bool:
    filename = frame.f_code.co_filename
    if filename.startswith(_PACKAGE_DIR) or os.path.abspath(filename).startswith(_PACKAGE_DIR):
        return True
    # scikit-learn 1.9's callback decorator, which LogisticRegression.fit carries:
    # scikit-learn's frame, but between mlxlearn's and the caller's. Stopping there
    # attributed a patched fit's fallback warning to scikit-learn's module.
    return frame.f_code.co_name == "callback_managed_method" and str(
        frame.f_globals.get("__name__", "")
    ).startswith("sklearn.callback")


def external_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside mlxlearn.

    For a ``warnings.warn`` call made directly by the caller of this function. A
    hard-coded level points wherever the call depth happened to be when it was
    written — for fallback warnings that was always ``_common/base.py``, which no
    module-based filter a user writes can match.
    """
    level, _ = _external_frame(sys._getframe(1))
    return level


def _external_frame(frame) -> tuple[int, Any]:
    # `frame` is the frame that will call warnings.warn, i.e. stacklevel 1.
    level = 1
    while frame is not None and _is_internal(frame):
        frame = frame.f_back
        level += 1
    return level, frame


def _matches(pattern: Any, text: str) -> bool:
    # Mirrors the warnings module: None matches anything, a plain string (an
    # internal default filter) must match exactly, anything else is a regex.
    if pattern is None:
        return True
    if isinstance(pattern, str):
        return pattern == text
    return pattern.match(text) is not None


def _filter_action(category: type[Warning], message: str, frame) -> str:
    """The action Python's warnings filters would take for this warning at ``frame``.

    The first matching entry of ``warnings.filters`` wins, exactly as in
    ``warnings.warn_explicit``; ``warnings.defaultaction`` applies if none does.
    """
    if frame is None:
        module, lineno = "<string>", 0
    else:
        module = frame.f_globals.get("__name__", "<string>")
        if not isinstance(module, str):
            module = "<string>"
        lineno = frame.f_lineno
    for action, message_re, filter_category, module_re, filter_lineno in warnings.filters:
        if (
            _matches(message_re, message)
            and issubclass(category, filter_category)
            and _matches(module_re, module)
            and (filter_lineno == 0 or filter_lineno == lineno)
        ):
            return action
    return warnings.defaultaction


def _warn_fallback(estimator: str, reason: str, message: str) -> None:
    key = (estimator, reason)
    if _recorder.has_warned(key):
        return
    level, frame = _external_frame(sys._getframe())
    action = _filter_action(MLXLearnFallbackWarning, message, frame)
    if action == "ignore":
        return
    if action != "error" and not _recorder.claim_warning(key):
        return
    warnings.warn(message, MLXLearnFallbackWarning, stacklevel=level)


# ------------------------------------------------------------------------ public API


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
    replacing: BackendEvent | None = None,
    **extra: Any,
) -> BackendEvent:
    """Record one dispatch decision, optionally warning about it.

    Parameters
    ----------
    warn : bool, default=False
        Emit :class:`~mlxlearn.exceptions.MLXLearnFallbackWarning` the first time
        this ``(estimator, reason)`` pair is shown, subject to
        ``fallback_policy``. Recording happens regardless. The warning is
        attributed to the first stack frame outside mlxlearn — the user's call.
    replacing : BackendEvent, optional
        An event this one supersedes: the same fit's decision, made again or
        amended. The new event takes its place in the record instead of being
        appended, so a fit stays one event however often it revises its decision.
        Appended as usual if ``replacing`` is no longer in the record.

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
        _recorder.record(event, replacing)

    if warn and config.fallback_policy == "warn":
        message = detail or f"{estimator}.{operation} fell back to scikit-learn ({reason})."
        _warn_fallback(
            estimator,
            reason,
            f"{message} This warning is shown once per estimator and reason; "
            "call mlxlearn.get_backend_diagnostics() for the full record.",
        )
    return event


def record_inference(
    estimator: str,
    operation: str,
    backend: Backend,
    reason: str,
    detail: str,
    shape: tuple[int, ...] | None,
    extra: dict[str, Any],
) -> None:
    """:func:`record_event` for an inference call: never a warning, nothing returned.

    The event is kept as its field tuple until someone reads it
    (:meth:`_Recorder.record`). ``extra`` is taken as given, not copied.
    """
    if get_config().diagnostics:
        _recorder.record(
            (estimator, operation, backend, reason, detail, shape, backend == "sklearn", extra)
        )


def get_backend_diagnostics() -> list[BackendEvent]:
    """Return the recorded backend events, oldest first.

    ``diagnostics=False`` stops new events from being recorded while it is in
    effect; events recorded before are still returned. The record covers this
    process only — see ``fit_event_`` on a fitted estimator for the decision its
    own fit made, wherever that ran.
    """
    return _recorder.events()


def get_last_backend_event() -> BackendEvent | None:
    """Return the most recent backend event, or ``None``."""
    return _recorder.last()


def clear_backend_diagnostics() -> None:
    """Drop all recorded events and reset warning deduplication."""
    _recorder.clear()


def reset_fallback_warnings() -> None:
    """Re-arm fallback warnings without dropping the recorded events.

    After this, each ``(estimator, reason)`` pair warns again the next time it
    falls back.
    """
    _recorder.reset_warnings()


def diagnostics_summary() -> dict[str, int]:
    """Count recorded work by ``"estimator.operation -> backend"``.

    A quick way to answer "did any of this actually run on the GPU?" without
    reading a few hundred events. Requests that were refused with an error, and
    patch/unpatch bookkeeping, did no work and are not counted.
    """
    counts: dict[str, int] = {}
    for event in _recorder.events():
        if event.operation in _NON_WORK_OPERATIONS or event.extra.get("raised"):
            continue
        key = f"{event.estimator}.{event.operation} -> {event.backend}"
        counts[key] = counts.get(key, 0) + 1
    return counts
