"""The backend-selection mixin every mlxlearn estimator inherits.

This is where the dispatch policy from the refactoring plan §2 actually lives, so
that no estimator has to reimplement it and drift.

Two invariants matter more than anything else here.

**The sticky backend.** The backend is chosen once, during ``fit``, and stored on
``_execution_backend_``. Every subsequent ``predict`` / ``predict_proba`` /
``transform`` / ``kneighbors`` uses *that* backend. Inference never re-decides.
An estimator whose MLX fit produced MLX-shaped state must not later find itself in
scikit-learn's inference path holding state that path cannot read — and the
converse, an estimator that fell back to scikit-learn at fit time, must not have
its predictions computed from attributes the MLX path never populated. A fresh
``fit`` clears the state and may choose differently.

**Fallback distinguishes why.** A capability mismatch — sparse input, an
unimplemented parameter — means scikit-learn can produce the right answer, so
patched mode uses it. An unexpected MLX failure means something is *broken*, and
rerunning it on scikit-learn would turn a bug report into a silent success. So it
raises, under every policy. That reverses the tempting "patched mode can never
raise" rule, which buys apparent robustness by making its own defects
undiscoverable.

Layer 1 classes (``mlxlearn.svm.SVC``) set ``_mlxlearn_allow_fallback = False``
and therefore only ever raise. Layer 2 classes, defined in
``mlxlearn.patching``, set it to ``True``. That single flag is the whole
difference between the two layers.

**One fit, one event; one call, one event.** A fit's decision is recorded when
it is made, fallback warning included, so a fit that then fails is still on the
record. A fit may consult the selection logic more than once — before validation
for capability, after it for the crossover — and each later decision takes the
earlier one's place in the record; :meth:`BackendMixin._set_backend` commits the
last one when the fit completes. A public inference call may reach the work
through other public methods (``predict`` through ``kneighbors``, scikit-learn's
``predict`` through the overridden ``decision_function``), so each public method
is entered through a thin wrapper that makes the outermost call the one that
records.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Literal

from ..exceptions import (
    BackendExecutionError,
    CapabilityError,
    MLXLearnError,
    UnsupportedInputError,
)
from .config import get_config, get_tuning
from .device import (
    check_fork_safety,
    forked_after_mlx_init,
    inference_device,
    physical_memory_bytes,
    resolve_device,
)
from .diagnostics import BackendEvent, record_event, record_inference

__all__ = ["BackendMixin", "Backend", "mlx_guard", "release_mlx_cache"]

Backend = Literal["mlx", "cpu", "sklearn"]


# ------------------------------------------------------------------ the MLX guard

#: Deliberate errors pass through ``mlx_guard`` untouched. ``Warning`` is here
#: because a warning a user's filter escalated to an error is theirs, not a bug.
_PASS_THROUGH = (MLXLearnError, MemoryError, Warning)


@contextmanager
def mlx_guard(estimator: str, operation: str):
    """Turn an unexpected MLX failure into a :class:`BackendExecutionError`.

    Deliberate errors — anything in the :class:`MLXLearnError` family, which
    includes :class:`~mlxlearn.exceptions.InvalidInputError` for bad input
    detected inside a guarded block — pass through untouched, as do
    ``MemoryError`` and warnings escalated to errors. Everything else is
    wrapped with the estimator and operation that produced it, keeping the
    original as ``__cause__``: that includes the ``ValueError`` and ``TypeError``
    MLX raises for its own shape and argument errors, which would otherwise reach
    the user looking like a complaint about their input.

    Wrapping is what makes the "never silently rerun on scikit-learn" rule
    enforceable: callers can tell a capability mismatch from a crash by type.

    The block is also a top-level MLX operation, so it releases MLX's buffer
    cache on the way out when it grew it substantially (:func:`release_mlx_cache`).
    """
    with release_mlx_cache():
        try:
            yield
        except _PASS_THROUGH:
            raise
        except Exception as exc:
            raise _backend_failure(estimator, operation, exc) from exc


def _backend_failure(estimator: str, operation: str, exc: Exception) -> BackendExecutionError:
    text = str(exc).rstrip(". ")
    where = f"{estimator}.{operation} failed inside the MLX backend: {type(exc).__name__}: {text}."
    if _is_allocation_failure(exc, text):
        return BackendExecutionError(
            f"{where} MLX could not allocate the memory this problem needed: the "
            "device ran out of memory, or one buffer exceeded its maximum size. "
            "mlxlearn did not silently rerun the computation on scikit-learn; use a "
            "smaller problem, or scikit-learn, for this call. If the problem fits "
            "comfortably in this machine's memory, please report it — an mlxlearn "
            "work buffer should have been bounded and was not.",
            original=exc,
        )
    return BackendExecutionError(
        f"{where} This is a bug in mlxlearn or MLX, not a capability limit, so "
        "mlxlearn did not silently rerun the computation on scikit-learn. Please "
        "report it with the traceback below.",
        original=exc,
    )


def _is_allocation_failure(exc: Exception, text: str) -> bool:
    # MLX reports allocator failures as RuntimeError("[metal::malloc] ..."). Only
    # the message's wording changes; a miss falls back to the generic text.
    return isinstance(exc, RuntimeError) and ("malloc" in text or "allocate" in text.lower())


# ----------------------------------------------------------- MLX buffer-cache release

_CACHE_SCOPE_DEPTH: ContextVar[int] = ContextVar("mlxlearn_cache_scope", default=0)
_FALLBACK_MEMORY_BYTES = 16 * 1024**3


def _mlx_module():
    # Only an already-loaded MLX is consulted: a top-level operation that never
    # touched MLX must not import it just to measure a cache it cannot have grown.
    if forked_after_mlx_init():
        return None
    return sys.modules.get("mlx.core")


def _cache_memory(mx) -> int:
    getter = getattr(mx, "get_cache_memory", None) or mx.metal.get_cache_memory
    return int(getter())


def _clear_cache(mx) -> None:
    clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
    clear()


@contextmanager
def release_mlx_cache():
    """Return MLX's idle buffers to the OS after a top-level operation that grew the cache.

    MLX keeps freed buffers in its own cache, up to a default limit of the whole
    memory limit, and blocked kernels whose buffer sizes vary from call to call
    cannot recycle them. Left alone, one large ``kneighbors`` call keeps
    gigabytes of unified memory after the model is gone. On exit from the
    outermost such block, if the cache grew by more than
    ``Tuning.mlx_cache_release_fraction`` of physical memory, ``mx.clear_cache()``
    is called. That frees idle buffers only and changes no MLX setting, though it
    is process-wide: idle buffers of other MLX work in the process go too, which
    costs them a reallocation and nothing else. Nested blocks do nothing.
    """
    depth = _CACHE_SCOPE_DEPTH.get()
    token = _CACHE_SCOPE_DEPTH.set(depth + 1)
    before = None
    if depth == 0:
        mx = _mlx_module()
        before = _cache_memory(mx) if mx is not None else 0
    try:
        yield
    finally:
        _CACHE_SCOPE_DEPTH.reset(token)
        if before is not None:
            _release_if_grown(before)


def _release_if_grown(before: int) -> None:
    fraction = get_tuning().mlx_cache_release_fraction
    mx = _mlx_module()
    if fraction <= 0 or mx is None:
        return
    memory = physical_memory_bytes() or _FALLBACK_MEMORY_BYTES
    if _cache_memory(mx) - before > int(fraction * memory):
        _clear_cache(mx)


# ---------------------------------------------------------------- public-call scope


class _CallScope:
    """The outermost public inference call currently running on one estimator."""

    __slots__ = ("owner", "operation", "recorded")

    def __init__(self, owner: object, operation: str) -> None:
        self.owner = owner
        self.operation = operation
        self.recorded = False


_CALL_SCOPE: ContextVar[_CallScope | None] = ContextVar("mlxlearn_call_scope", default=None)

#: Public inference methods whose outermost call records the one diagnostics
#: event. Wrapped wherever an mlxlearn class defines them (see __init_subclass__).
_PUBLIC_INFERENCE = (
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "kneighbors",
    "kneighbors_graph",
    "radius_neighbors",
    "radius_neighbors_graph",
    "transform",
    "score",
)


def _public_call(name: str, method: Callable) -> Callable:
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        scope = _CALL_SCOPE.get()
        if scope is not None and scope.owner is self:
            return method(self, *args, **kwargs)
        token = _CALL_SCOPE.set(_CallScope(self, name))
        try:
            return method(self, *args, **kwargs)
        finally:
            _CALL_SCOPE.reset(token)

    call._mlxlearn_public_call = True  # type: ignore[attr-defined]
    return call


def _array_api_dispatch_enabled() -> bool:
    # scikit-learn is always loaded by the time an estimator fits; importing it
    # here rather than at module level keeps `import mlxlearn` free of it.
    import sklearn

    return bool(sklearn.get_config().get("array_api_dispatch", False))


# ------------------------------------------------------------------------ the mixin


class BackendMixin:
    """Backend selection, stickiness, and diagnostics for mlxlearn estimators.

    Subclasses get two things without asking. Every public inference method a
    subclass defines or inherits as a plain function (``predict``, ``kneighbors``,
    scikit-learn's ``score`` ... — see ``_PUBLIC_INFERENCE``) is wrapped so that
    nested calls on the same estimator record one event, under the name of the
    method the caller invoked. And the fitted attributes named in
    any ``_mlxlearn_mlx_attrs`` tuple along the MRO are deleted at the start of
    every fit.
    """

    #: Layer 1 raises on a capability mismatch; Layer 2 falls back. Nothing else differs.
    _mlxlearn_allow_fallback: ClassVar[bool] = False

    #: Set during ``fit``. Absent on an unfitted estimator, which is how
    #: ``check_is_fitted`` and the sticky-backend assertions detect misuse.
    _execution_backend_: str

    #: Narrowed for Layer 1 by :meth:`__sklearn_tags__`. An estimator that
    #: genuinely handles multi-output targets on the MLX path sets this
    #: ``True``, as the neighbor estimators do.
    _mlxlearn_supports_multi_output: ClassVar[bool] = False

    #: Fitted attributes only an MLX-path fit creates — a device index, pairwise
    #: solver state. Deleted at the start of every fit, so a refit that goes to
    #: scikit-learn (or fails) does not keep the previous training data alive in
    #: memory and in pickles. Each mixin names its own; the tuples of every class
    #: in the MRO are combined.
    _mlxlearn_mlx_attrs: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in _PUBLIC_INFERENCE:
            # Inherited methods too: scikit-learn's own kneighbors_graph, score or
            # radius_neighbors_graph reach the work through a method mlxlearn
            # overrides, and without a wrapper of their own the event was named
            # after that inner method rather than the call the user made. Only
            # plain functions, read from the class dicts rather than through
            # getattr: an ``available_if`` descriptor (SVC.predict_proba) resolves
            # to a function on the class, and replacing it with a wrapper would make
            # the method exist on estimators that do not have it.
            method = next((k.__dict__[name] for k in cls.__mro__ if name in k.__dict__), None)
            if inspect.isfunction(method) and not getattr(method, "_mlxlearn_public_call", False):
                setattr(cls, name, _public_call(name, method))

    # --------------------------------------------------------------------- tags

    def __sklearn_tags__(self):
        """Advertise what this estimator actually accepts.

        Inherited tags describe scikit-learn's capabilities, not mlxlearn's, so
        they have to be narrowed. An estimator that advertises sparse support and
        then raises on sparse input is lying to every meta-estimator in the
        ecosystem that reads its tags to decide what to pass it.

        The narrowing is conditional on ``_mlxlearn_allow_fallback``, because for
        a Layer 2 class the wide tags are *true*: a patched estimator really does
        accept sparse input, multi-output targets and array-API inputs, by handing
        them to scikit-learn. Layer 1 does not, and says so.
        """
        tags = super().__sklearn_tags__()
        if not self._mlxlearn_allow_fallback:
            tags.input_tags.sparse = False
            # scikit-learn 1.9 advertises array-API support for lbfgs
            # LogisticRegression; mlxlearn converts everything to NumPy.
            tags.array_api_support = False
            if not self._mlxlearn_supports_multi_output:
                tags.target_tags.multi_output = False
                if getattr(tags, "classifier_tags", None) is not None:
                    tags.classifier_tags.multi_label = False
        return tags

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted means a fit completed — not merely that fitted attributes exist.

        A refit that failed leaves the previous fit's public attributes behind,
        which scikit-learn's attribute scan would call fitted; mlxlearn has by
        then forgotten which backend owns them. State assigned from outside,
        with no mlxlearn fit involved, is judged by scikit-learn's own rule.
        """
        if "_execution_backend_" in self.__dict__:
            return True
        if self.__dict__.get("_mlxlearn_fit_incomplete", False):
            return False
        return any(k.endswith("_") and not k.startswith("__") for k in vars(self))

    # ---------------------------------------------------------------- selection

    def _select_backend(
        self,
        operation: str,
        *,
        capability_check: Callable[[], None] | None = None,
        below_crossover: bool = False,
        crossover_detail: str = "",
        shape: tuple[int, ...] | None = None,
        crossover: Callable[[int, int], tuple[bool, str]] | None = None,
        handoff: Callable[[int, int], tuple[str, str] | None] | None = None,
        data_check: Callable[[], None] | None = None,
    ) -> Backend:
        """Choose the backend for a ``fit``, and record the decision.

        May be called more than once per fit; each decision takes the previous
        one's place in the record, and :meth:`_set_backend` commits the last. The
        order of precedence is: capability, then ``MLXLEARN_FORCE_BACKEND``, then
        size, then the data-dependent capability checks, then the device. A
        by-design handoff by size therefore skips ``data_check``: a patched fit
        below the crossover whose only mismatch is data-dependent (values past
        float32's range, say) is recorded as ``"below-crossover"``, and does not
        raise under ``fallback_policy="raise"``.

        With ``crossover`` given and no ``shape`` — input that has none before
        validation — the size, and so the device, cannot be decided yet: this
        returns ``"cpu"`` provisionally, having recorded nothing, and the caller
        decides again with the validated shape.

        Parameters
        ----------
        operation : str
            Usually ``"fit"``. Recorded in diagnostics.
        capability_check : callable, optional
            Raises :class:`CapabilityError` if the MLX path cannot serve this
            request. Called before any size heuristic, because "we cannot do this
            at all" outranks "this is small". Keep it cheap — parameters and
            sparsity — and put O(n·d) scans in ``data_check``.
        below_crossover : bool, default=False
            True when the problem is smaller than the measured point at which MLX
            overtakes scikit-learn. Handing these away is the mechanism behind the
            "never slower" goal, not an admission of defeat.
        crossover_detail : str
            Human-readable size explanation for the diagnostics record.
        shape : tuple of int, optional
            ``(n_samples, n_features)``, recorded so a threshold decision can be
            audited later.
        crossover : callable, optional
            ``crossover(n_samples, n_features) -> (below, detail)``. When given
            and ``shape`` is known, the size decision is made here, from the raw
            shape, before any validation or conversion: a patched fit headed for
            scikit-learn then pays for none of mlxlearn's own input preparation.
        handoff : callable, optional
            ``handoff(n_samples, n_features) -> (reason, detail)`` or ``None``: a
            by-design handoff to scikit-learn that only patched mode makes, on
            the raw shape, after the crossover — the neighbor estimators'
            narrow-data rule. Directly imported, the estimator runs on MLX regardless.
        data_check : callable, optional
            Data-dependent capability checks (float32 range, target type). Run
            after the size decision, and skipped when the work is going to
            scikit-learn anyway.

        Returns
        -------
        {"mlx", "cpu", "sklearn"}
        """
        self._begin_fit()
        name = type(self).__name__
        tuning = get_tuning()

        try:
            if capability_check is not None:
                capability_check()
            self._check_array_api_capability()
        except CapabilityError as exc:
            return self._handle_capability_error(exc, operation, shape=shape)

        if tuning.force_backend:
            backend: Backend = tuning.force_backend  # type: ignore[assignment]
            if backend == "sklearn" and not self._mlxlearn_allow_fallback:
                raise MLXLearnError(
                    "MLXLEARN_FORCE_BACKEND='sklearn' has no meaning for a directly "
                    f"imported {name}: Layer 1 estimators never dispatch to scikit-learn. "
                    "Use the patched class, or force 'mlx' or 'cpu'."
                )
            if backend != "sklearn":
                refused = self._run_data_check(data_check, operation, shape)
                if refused is not None:
                    return refused
                check_fork_safety()
            self._stage_event(
                operation, backend, reason="forced", shape=shape, fallback=backend == "sklearn"
            )
            return backend

        detail = crossover_detail
        sized = shape is not None and len(shape) == 2
        if not below_crossover and crossover is not None and sized:
            below_crossover, detail = crossover(int(shape[0]), int(shape[1]))

        if self._mlxlearn_allow_fallback:
            handed = None
            if below_crossover:
                handed = ("below-crossover", detail)
            elif handoff is not None and sized:
                handed = handoff(int(shape[0]), int(shape[1]))
            if handed is not None:
                reason, handed_detail = handed
                # Not a warning: dispatching this to scikit-learn is the design
                # working, not a limitation being hit.
                self._stage_event(
                    operation, "sklearn", reason=reason, detail=handed_detail, shape=shape,
                    fallback=True,
                )
                return "sklearn"

        refused = self._run_data_check(data_check, operation, shape)
        if refused is not None:
            return refused

        if forked_after_mlx_init():
            if self._mlxlearn_allow_fallback:
                return self._handle_capability_error(
                    CapabilityError(
                        f"{name} cannot use MLX in this process: it was forked from one "
                        "that had already loaded MLX or scikit-learn, and MLX is not "
                        "fork-safe. Use the 'spawn' or 'forkserver' start method to "
                        "get MLX in worker processes.",
                        reason="forked-process",
                    ),
                    operation,
                    shape=shape,
                )
            check_fork_safety()

        if crossover is not None and not sized:
            return "cpu"  # provisional: the caller decides again once it has a shape

        if below_crossover:
            self._stage_event(operation, "cpu", reason="below-crossover", detail=detail, shape=shape)
            return "cpu"

        # An explicit device="gpu" that cannot be honored raises DeviceError here,
        # and it is not a capability mismatch scikit-learn could paper over.
        device = resolve_device(get_config().device)
        backend = "mlx" if device == "gpu" else "cpu"
        self._stage_event(operation, backend, reason="selected", shape=shape)
        return backend

    def _run_data_check(
        self,
        data_check: Callable[[], None] | None,
        operation: str,
        shape: tuple[int, ...] | None,
    ) -> Backend | None:
        if data_check is None:
            return None
        try:
            data_check()
        except CapabilityError as exc:
            return self._handle_capability_error(exc, operation, shape=shape)
        return None

    def _check_array_api_capability(self) -> None:
        """Under ``array_api_dispatch``, hand what scikit-learn serves natively back to it.

        scikit-learn returns arrays in the input's namespace (torch in, torch
        out); mlxlearn computes in MLX and returns NumPy. Where the scikit-learn
        estimator advertises array-API support — which only the Layer 2 tags pass
        through — that is a capability mismatch.
        """
        if _array_api_dispatch_enabled() and self.__sklearn_tags__().array_api_support:
            raise UnsupportedInputError(
                f"{type(self).__name__} does not implement scikit-learn's array API "
                "dispatch (array_api_dispatch=True): mlxlearn computes in MLX and "
                "returns NumPy arrays.",
                reason="array-api-dispatch",
            )

    def _handle_capability_error(
        self,
        exc: CapabilityError,
        operation: str,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Backend:
        """Apply ``fallback_policy`` to a capability mismatch.

        Layer 1 re-raises ``exc`` as it is, advice included. A patched estimator
        is already under ``patch_sklearn()``, so what it records, warns and raises
        is the mismatch without that advice (``exc.summary``), followed by what it
        did about it.

        The warning comes with the decision, before scikit-learn does any of the
        work: a handoff decided here is final, and a user's "error" filter — the
        documented way to assert that nothing falls back — then fails fast.
        """
        policy = get_config().fallback_policy

        if not self._mlxlearn_allow_fallback:
            self._stage_event(
                operation, "cpu", reason=exc.reason, detail=str(exc), shape=shape, raised=True
            )
            raise exc

        # getattr: an exception unpickled from an older mlxlearn has no summary.
        summary = getattr(exc, "summary", None) or str(exc)
        if policy == "raise":
            detail = (
                f"{summary} fallback_policy='raise' is set, so mlxlearn did not hand "
                f"this {operation} to scikit-learn. Set fallback_policy='warn' or "
                "'silent' to allow the fallback."
            )
            self._stage_event(
                operation, "sklearn", reason=exc.reason, detail=detail, shape=shape,
                fallback=True, raised=True,
            )
            raise CapabilityError(detail, reason=exc.reason) from exc

        self._stage_event(
            operation,
            "sklearn",
            reason=exc.reason,
            detail=f"{summary} mlxlearn handed this {operation} to scikit-learn.",
            shape=shape,
            fallback=True,
            warn=True,
        )
        return "sklearn"

    # ------------------------------------------------------------------ sticky

    def _begin_fit(self) -> None:
        # A fit is in progress from here until _set_backend. Also set by
        # _clear_backend; repeated here so that an estimator that forgot to
        # clear still records its decisions as one event.
        self.__dict__["_mlxlearn_fit_incomplete"] = True

    def _stage_event(self, operation: str, backend: Backend, **kwargs: Any) -> BackendEvent:
        """Record a decision; during a fit, in place of the one the fit recorded before.

        The fit's latest decision stays on the estimator, for :meth:`_set_backend`
        to commit or a later decision in the same fit to replace. A fit that fails
        after deciding therefore leaves its decision on the record — once — with
        its fallback warning already shown.
        """
        state = self.__dict__
        in_fit = state.get("_mlxlearn_fit_incomplete", False)
        event = record_event(
            type(self).__name__,
            operation,
            backend,
            replacing=state.get("_mlxlearn_pending_event") if in_fit else None,
            **kwargs,
        )
        if in_fit:
            state["_mlxlearn_pending_event"] = event
        return event

    def _amend_fit_event(self, detail: str = "", **extra: Any) -> None:
        """Add what the work found to this fit's decision: ``detail`` appended, ``extra`` merged.

        How a solver stopped, say. It goes on the fit's one event, in its place in
        the record, rather than on a second event.
        """
        event = self.__dict__["_mlxlearn_pending_event"]
        self.__dict__["_mlxlearn_pending_event"] = record_event(
            event.estimator,
            event.operation,
            event.backend,
            reason=event.reason,
            detail=f"{event.detail} {detail}".strip(),
            shape=event.shape,
            fallback=event.fallback,
            replacing=event,
            **{**event.extra, **extra},
        )

    def _set_backend(self, backend: Backend) -> None:
        """Commit the backend chosen by ``fit``. Call exactly once per fit, at its end.

        The fit's decision, already recorded, is kept on the estimator as
        ``fit_event_``, and the estimator is fitted from here on.
        """
        event = self.__dict__.pop("_mlxlearn_pending_event", None)
        if event is None or event.backend != backend:
            # A decision this mixin did not make; record what actually happened.
            event = record_event(
                type(self).__name__, "fit", backend, fallback=backend == "sklearn",
                replacing=event,
            )
        self._execution_backend_ = backend
        self._fit_event_ = event
        self.__dict__.pop("_mlxlearn_fit_incomplete", None)

    def _clear_backend(self) -> None:
        """Forget the fitted backend. Call at the start of every ``fit``.

        Without this, a fit that raises partway through leaves a stale
        ``_execution_backend_`` behind, and the next ``predict`` would route on
        the strength of a fit that never completed. Until the fit completes the
        estimator reports itself unfitted (:meth:`__sklearn_is_fitted__`).

        MLX-owned fitted state (``_mlxlearn_mlx_attrs``) goes too: a refit that
        falls back to scikit-learn must not keep the previous fit's device index
        and training copy alive, in memory or in a pickle.
        """
        state = self.__dict__
        for attribute in ("_execution_backend_", "_fit_event_", "_mlxlearn_pending_event"):
            state.pop(attribute, None)
        for klass in type(self).__mro__:
            for attribute in klass.__dict__.get("_mlxlearn_mlx_attrs", ()):
                state.pop(attribute, None)
        self._begin_fit()

    def _fitted_backend(self, operation: str = "predict") -> Backend:
        """Return the backend ``fit`` chose, for an inference call to reuse.

        Raises
        ------
        IncompleteFitError
            Both an :class:`MLXLearnError` and scikit-learn's ``NotFittedError``.
        """
        backend = self.__dict__.get("_execution_backend_")
        if backend is None:
            from .notfitted import IncompleteFitError

            raise IncompleteFitError(
                f"{type(self).__name__} has no recorded execution backend. This "
                "instance was not fitted by mlxlearn, or its fit did not complete."
            )
        return backend  # type: ignore[return-value]

    def _claim_call_event(self, operation: str) -> str | None:
        """Claim the one event of the public call in progress on this estimator.

        Returns the name to record it under — the method the caller invoked, or
        ``operation`` outside any public call — or ``None`` when the call has
        already recorded its event. For an inference path that records something
        other than a sticky event (a capability fallback at inference) and must
        still count as the call's one event.
        """
        scope = _CALL_SCOPE.get()
        if scope is None or scope.owner is not self:
            return operation
        if scope.recorded:
            return None
        scope.recorded = True
        return scope.operation

    def _note_inference(
        self,
        operation: str,
        *,
        shape: tuple[int, ...] | None = None,
        detail: str = "",
        host: bool = False,
        **extra: Any,
    ) -> Backend:
        """Record that an inference call reused the fitted backend, and return it.

        Every inference path calls this instead of re-running selection. The
        recorded reason is ``"sticky"``, which is exactly the evidence needed to
        show that inference did not wander — or ``"gpu-unavailable"`` when a
        GPU-fitted model runs on MLX's CPU backend because this process has no
        GPU (:func:`~mlxlearn._common.device.inference_device`). That is also
        where an explicit ``device="gpu"`` that cannot be honored raises
        :class:`~mlxlearn.exceptions.DeviceError`.

        One public call records one event, named after the method the caller
        invoked: a nested call on the same estimator (``predict`` reaching
        ``kneighbors``) returns the backend without recording again. ``detail``
        and ``extra`` describe how the call was served — a neighbor search that
        needed its exact fallback, say — and go on that one event.

        ``host=True`` says the call computes on the host from the fitted state
        (scikit-learn's own ``radius_neighbors`` on an MLX-fitted index,
        ``LogisticRegression``'s float64 inference on its coefficients): no MLX
        device is used, so none is resolved, none can be missing, and an
        explicit ``device="gpu"`` has nothing to refuse.
        """
        backend = self._fitted_backend(operation)
        claimed = self._claim_call_event(operation)
        if claimed is None:
            return backend
        operation = claimed

        reason = "sticky"
        if backend == "mlx" and not host and inference_device(backend) == "cpu":
            reason = "gpu-unavailable"
        fit_event = self.__dict__.get("_fit_event_")
        if fit_event is not None:
            extra["fit_reason"] = fit_event.reason
        record_inference(
            type(self).__name__,
            operation,
            backend,
            reason,
            detail,
            None if shape is None else tuple(shape),
            extra,
        )
        return backend

    # ------------------------------------------------------------ introspection

    @property
    def execution_backend_(self) -> str:
        """Which backend fitted this estimator: ``"mlx"``, ``"cpu"`` or ``"sklearn"``.

        Raises
        ------
        AttributeError
            If the estimator is not fitted, matching scikit-learn's convention for
            fitted attributes.
        """
        backend = self.__dict__.get("_execution_backend_")
        if backend is None:
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet; execution_backend_ is "
                "only defined after fit."
            )
        return backend

    @property
    def fit_event_(self) -> BackendEvent:
        """The diagnostics event this estimator's own ``fit`` committed.

        Why the fit ran where it did — ``reason`` and ``detail`` — kept with the
        model: it outlives the process-wide ring buffer, is recorded even with
        ``diagnostics=False``, and pickles with the estimator, so a model fitted
        in a worker process still explains itself.

        Raises
        ------
        AttributeError
            If the estimator is not fitted.
        """
        event = self.__dict__.get("_fit_event_")
        if event is None:
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet; fit_event_ is only "
                "defined after fit."
            )
        return event

    def _backend_event(self, operation: str, backend: Backend, **kwargs: Any) -> BackendEvent:
        return record_event(type(self).__name__, operation, backend, **kwargs)

