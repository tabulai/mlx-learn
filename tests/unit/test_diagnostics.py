"""Diagnostics: what ran where, and why."""

from __future__ import annotations

import threading
import warnings

import pytest

import mlxlearn
from mlxlearn._common.config import tuning_context
from mlxlearn._common.diagnostics import diagnostics_summary, record_event, record_inference
from mlxlearn.exceptions import MLXLearnFallbackWarning


def test_events_are_recorded_in_order():
    record_event("A", "fit", "mlx")
    record_event("B", "predict", "cpu", reason="below-crossover")
    events = mlxlearn.get_backend_diagnostics()
    assert [e.estimator for e in events] == ["A", "B"]
    assert events[-1].reason == "below-crossover"


def test_last_event():
    assert mlxlearn.get_last_backend_event() is None
    record_event("A", "fit", "mlx")
    assert mlxlearn.get_last_backend_event().estimator == "A"


def test_a_revised_decision_takes_the_earlier_ones_place():
    """How a fit that decides again stays one event, wherever it sits in the record."""
    first = record_event("A", "fit", "mlx")
    other = record_event("B", "predict", "cpu")
    revised = record_event("A", "fit", "cpu", reason="below-crossover", replacing=first)
    events = mlxlearn.get_backend_diagnostics()
    assert len(events) == 2 and events[0] is revised and events[1] is other
    # One no longer in the record is appended.
    mlxlearn.clear_backend_diagnostics()
    again = record_event("A", "fit", "sklearn", replacing=revised)
    assert mlxlearn.get_backend_diagnostics() == [again]


def test_an_inference_event_is_built_when_it_is_read_and_is_then_the_same_object():
    """Inference calls record their event's fields and build the event on first read,
    in place: a later read returns the object an earlier one did."""
    record_inference("A", "predict", "sklearn", "sticky", "", (3, 2), {"fit_reason": "x"})
    record_event("A", "fit", "mlx")
    first = mlxlearn.get_backend_diagnostics()
    assert isinstance(first[0], mlxlearn.BackendEvent)
    assert (first[0].operation, first[0].shape, first[0].fallback) == ("predict", (3, 2), True)
    assert first[0].extra == {"fit_reason": "x"}
    assert mlxlearn.get_backend_diagnostics()[0] is first[0]
    record_inference("A", "kneighbors", "mlx", "sticky", "", None, {})
    last = mlxlearn.get_last_backend_event()
    assert isinstance(last, mlxlearn.BackendEvent) and last.operation == "kneighbors"
    assert last is mlxlearn.get_backend_diagnostics()[-1]
    assert diagnostics_summary() == {
        "A.predict -> sklearn": 1, "A.fit -> mlx": 1, "A.kneighbors -> mlx": 1,
    }
    with mlxlearn.config_context(diagnostics=False):
        record_inference("A", "predict", "mlx", "sticky", "", None, {})
    assert len(mlxlearn.get_backend_diagnostics()) == 3


def test_clearing_resets_warning_deduplication():
    with pytest.warns(MLXLearnFallbackWarning):
        record_event("A", "fit", "sklearn", reason="sparse-input", fallback=True, warn=True)

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        record_event("A", "fit", "sklearn", reason="sparse-input", fallback=True, warn=True)
    assert not second

    mlxlearn.clear_backend_diagnostics()
    with pytest.warns(MLXLearnFallbackWarning):
        record_event("A", "fit", "sklearn", reason="sparse-input", fallback=True, warn=True)


def test_a_different_reason_warns_again():
    """Deduplication is per (estimator, reason), so a new limitation is still reported."""
    with pytest.warns(MLXLearnFallbackWarning):
        record_event("A", "fit", "sklearn", reason="sparse-input", fallback=True, warn=True)
    with pytest.warns(MLXLearnFallbackWarning):
        record_event("A", "fit", "sklearn", reason="unsupported-parameter:solver", fallback=True, warn=True)


def test_silent_policy_records_but_does_not_warn():
    """Silencing the warning stream must not cost the evidence."""
    with mlxlearn.config_context(fallback_policy="silent"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record_event("A", "fit", "sklearn", reason="sparse-input", fallback=True, warn=True)
    assert not caught
    assert mlxlearn.get_backend_diagnostics()[-1].fallback is True


def test_diagnostics_can_be_turned_off():
    with mlxlearn.config_context(diagnostics=False):
        record_event("A", "fit", "mlx")
        assert mlxlearn.get_backend_diagnostics() == []


def test_ring_buffer_is_bounded(monkeypatch):
    """An agent harness fitting in a loop for hours must not accumulate forever.

    The capacity is process-wide, set through the environment; the autouse
    fixture's reset_config restores the default after the monkeypatch is undone.
    """
    monkeypatch.setenv("MLXLEARN_DIAGNOSTICS_CAPACITY", "8")
    mlxlearn.reset_config()
    for i in range(50):
        record_event(f"E{i}", "fit", "mlx")
    events = mlxlearn.get_backend_diagnostics()
    assert len(events) == 8
    assert events[-1].estimator == "E49"


def test_tuning_context_cannot_resize_the_shared_buffer():
    """One thread shrinking a buffer every thread shares would drop their events."""
    with tuning_context(diagnostics_capacity=8):
        for i in range(20):
            record_event(f"E{i}", "fit", "mlx")
    assert len(mlxlearn.get_backend_diagnostics()) == 20


def test_capacity_returns_to_the_default_after_a_reset(monkeypatch):
    """The old test resized the recorder by hand and leaked capacity 8 into the session."""
    from mlxlearn._common.config import Tuning
    from mlxlearn._common.diagnostics import _recorder

    monkeypatch.setenv("MLXLEARN_DIAGNOSTICS_CAPACITY", "4")
    mlxlearn.reset_config()
    record_event("A", "fit", "mlx")
    assert _recorder._events.maxlen == 4

    monkeypatch.delenv("MLXLEARN_DIAGNOSTICS_CAPACITY")
    mlxlearn.reset_config()
    record_event("B", "fit", "mlx")
    assert _recorder._events.maxlen == Tuning().diagnostics_capacity
    assert [e.estimator for e in mlxlearn.get_backend_diagnostics()] == ["A", "B"]


def test_summary_counts_by_estimator_operation_and_backend():
    record_event("A", "fit", "mlx")
    record_event("A", "fit", "mlx")
    record_event("A", "predict", "mlx")
    summary = diagnostics_summary()
    assert summary["A.fit -> mlx"] == 2
    assert summary["A.predict -> mlx"] == 1


def test_recording_is_shared_across_threads():
    """Process-wide on purpose: a user debugging a fallback should not have to know
    which thread scikit-learn's n_jobs machinery happened to run it on."""

    def worker():
        record_event("Worker", "fit", "mlx")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert any(e.estimator == "Worker" for e in mlxlearn.get_backend_diagnostics())


def test_event_carries_shape_for_auditing_a_threshold(small_classification_data):
    from mlxlearn.neighbors import KNeighborsClassifier

    X, y = small_classification_data
    KNeighborsClassifier(n_neighbors=3).fit(X, y)
    fit_events = [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"]
    assert fit_events[-1].shape == X.shape
    assert "below the measured KNN crossover" in fit_events[-1].detail
