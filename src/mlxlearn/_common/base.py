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
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

import numpy as np

from ..exceptions import BackendExecutionError, CapabilityError, DeviceError, MLXLearnError
from .config import get_config, get_tuning
from .device import resolve_device
from .diagnostics import BackendEvent, record_event

__all__ = ["BackendMixin", "Backend", "mlx_guard"]

Backend = Literal["mlx", "cpu", "sklearn"]


@contextmanager
def mlx_guard(estimator: str, operation: str):
    """Turn an unexpected MLX failure into a :class:`BackendExecutionError`.

    Deliberate errors — anything already in the :class:`MLXLearnError` family, and
    the ``ValueError``/``TypeError`` that validation raises — pass through
    untouched, so a user's bad input still reports as bad input. Everything else
    is a bug, gets wrapped with the estimator and operation that produced it, and
    keeps the original as ``__cause__``.

    Wrapping is what makes the "never silently rerun on scikit-learn" rule
    enforceable: callers can tell a capability mismatch from a crash by type.
    """
    try:
        yield
    except (MLXLearnError, ValueError, TypeError, KeyboardInterrupt, MemoryError):
        raise
    except Exception as exc:
        raise BackendExecutionError(
            f"{estimator}.{operation} failed inside the MLX backend: "
            f"{type(exc).__name__}: {exc}. This is a bug in mlxlearn or MLX, not a "
            "capability limit, so mlxlearn did not silently rerun the computation on "
            "scikit-learn. Please report it with the traceback below.",
            original=exc,
        ) from exc


class BackendMixin:
    """Backend selection, stickiness, and diagnostics for mlxlearn estimators."""

    #: Layer 1 raises on a capability mismatch; Layer 2 falls back. Nothing else differs.
    _mlxlearn_allow_fallback: ClassVar[bool] = False

    #: Set during ``fit``. Absent on an unfitted estimator, which is how
    #: ``check_is_fitted`` and the sticky-backend assertions detect misuse.
    _execution_backend_: str

    #: Narrowed for Layer 1 by :meth:`__sklearn_tags__`. An estimator that
    #: genuinely handles multi-output targets on the MLX path can set this
    #: ``True``; none of the 0.1.0 estimators do.
    _mlxlearn_supports_multi_output: ClassVar[bool] = False

    # --------------------------------------------------------------------- tags

    def __sklearn_tags__(self):
        """Advertise what this estimator actually accepts.

        Inherited tags describe scikit-learn's capabilities, not mlxlearn's, so
        they have to be narrowed. An estimator that advertises sparse support and
        then raises on sparse input is lying to every meta-estimator in the
        ecosystem that reads its tags to decide what to pass it.

        The narrowing is conditional on ``_mlxlearn_allow_fallback``, because for
        a Layer 2 class the wide tags are *true*: a patched estimator really does
        accept sparse input and multi-output targets, by handing them to
        scikit-learn. Layer 1 does not, and says so.
        """
        tags = super().__sklearn_tags__()
        if not self._mlxlearn_allow_fallback:
            tags.input_tags.sparse = False
            if not self._mlxlearn_supports_multi_output:
                tags.target_tags.multi_output = False
                if getattr(tags, "classifier_tags", None) is not None:
                    tags.classifier_tags.multi_label = False
        return tags

    # ---------------------------------------------------------------- selection

    def _select_backend(
        self,
        operation: str,
        *,
        capability_check: Callable[[], None] | None = None,
        below_crossover: bool = False,
        crossover_detail: str = "",
        shape: tuple[int, ...] | None = None,
    ) -> Backend:
        """Choose the backend for a ``fit``, recording the decision.

        Parameters
        ----------
        operation : str
            Usually ``"fit"``. Recorded in diagnostics.
        capability_check : callable, optional
            Raises :class:`CapabilityError` if the MLX path cannot serve this
            request. Called before any size heuristic, because "we cannot do this
            at all" outranks "this is small".
        below_crossover : bool, default=False
            True when the problem is smaller than the measured point at which MLX
            overtakes scikit-learn. Handing these away is the mechanism behind the
            "never slower" goal, not an admission of defeat.
        crossover_detail : str
            Human-readable size explanation for the diagnostics record.
        shape : tuple of int, optional
            ``(n_samples, n_features)``, recorded so a threshold decision can be
            audited later.

        Returns
        -------
        {"mlx", "cpu", "sklearn"}
        """
        name = type(self).__name__
        tuning = get_tuning()
        config = get_config()

        if capability_check is not None:
            try:
                capability_check()
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
            record_event(name, operation, backend, reason="forced", shape=shape)
            return backend

        if below_crossover:
            if self._mlxlearn_allow_fallback:
                record_event(
                    name,
                    operation,
                    "sklearn",
                    reason="below-crossover",
                    detail=crossover_detail,
                    shape=shape,
                    fallback=True,
                    # Not a warning: dispatching a small problem to scikit-learn is
                    # the design working, not a limitation being hit.
                    warn=False,
                )
                return "sklearn"
            record_event(
                name,
                operation,
                "cpu",
                reason="below-crossover",
                detail=crossover_detail,
                shape=shape,
            )
            return "cpu"

        try:
            device = resolve_device(config.device)
        except DeviceError:
            # An explicit device="gpu" that cannot be honored is an error, and it
            # is not a capability mismatch scikit-learn could paper over.
            raise
        backend = "mlx" if device == "gpu" else "cpu"
        record_event(name, operation, backend, reason="selected", shape=shape)
        return backend

    def _handle_capability_error(
        self,
        exc: CapabilityError,
        operation: str,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Backend:
        """Apply ``fallback_policy`` to a capability mismatch."""
        name = type(self).__name__
        policy = get_config().fallback_policy

        if not self._mlxlearn_allow_fallback:
            record_event(name, operation, "cpu", reason=exc.reason, detail=str(exc), shape=shape)
            raise exc

        if policy == "raise":
            record_event(
                name, operation, "sklearn", reason=exc.reason, detail=str(exc),
                shape=shape, fallback=True,
            )
            raise CapabilityError(
                f"{exc} — and fallback_policy='raise' is set, so mlxlearn did not "
                "hand this to scikit-learn. Set fallback_policy='warn' or 'silent' "
                "to allow the fallback.",
                reason=exc.reason,
            ) from exc

        record_event(
            name,
            operation,
            "sklearn",
            reason=exc.reason,
            detail=str(exc),
            shape=shape,
            fallback=True,
            warn=True,
        )
        return "sklearn"

    # ------------------------------------------------------------------ sticky

    def _set_backend(self, backend: Backend) -> None:
        """Record the backend chosen by ``fit``. Call exactly once per fit."""
        self._execution_backend_ = backend

    def _clear_backend(self) -> None:
        """Forget the fitted backend. Call at the start of every ``fit``.

        Without this, a fit that raises partway through leaves a stale
        ``_execution_backend_`` behind, and the next ``predict`` would route on
        the strength of a fit that never completed.
        """
        for attribute in ("_execution_backend_", "_mlx_state_"):
            if hasattr(self, attribute):
                delattr(self, attribute)

    def _fitted_backend(self, operation: str = "predict") -> Backend:
        """Return the backend ``fit`` chose, for an inference call to reuse."""
        backend = getattr(self, "_execution_backend_", None)
        if backend is None:
            raise MLXLearnError(
                f"{type(self).__name__} has no recorded execution backend. This "
                "instance was not fitted by mlxlearn, or its fit did not complete."
            )
        return backend  # type: ignore[return-value]

    def _note_inference(
        self,
        operation: str,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Backend:
        """Record that an inference call reused the fitted backend, and return it.

        Every inference path calls this instead of re-running selection. The
        recorded reason is ``"sticky"``, which is exactly the evidence needed to
        show that inference did not wander.
        """
        backend = self._fitted_backend(operation)
        record_event(
            type(self).__name__,
            operation,
            backend,
            reason="sticky",
            shape=shape,
            fallback=backend == "sklearn",
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
        backend = getattr(self, "_execution_backend_", None)
        if backend is None:
            raise AttributeError(
                f"{type(self).__name__} is not fitted yet; execution_backend_ is "
                "only defined after fit."
            )
        return backend

    def _backend_event(self, operation: str, backend: Backend, **kwargs: Any) -> BackendEvent:
        return record_event(type(self).__name__, operation, backend, **kwargs)


def problem_work(X: np.ndarray, *, extra: int = 1) -> int:
    """A crude scalar proxy for how much arithmetic a problem needs.

    ``n_samples * n_features * extra``. Crude on purpose: the crossover thresholds
    it feeds are measured, and a measured threshold on a simple proxy beats a
    sophisticated model nobody calibrated.
    """
    if X.ndim != 2:
        return 0
    return int(X.shape[0]) * int(X.shape[1]) * int(extra)
