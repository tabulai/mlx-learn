"""Diagnostics: what ran where, and why."""

from __future__ import annotations

import threading
import warnings

import pytest

import mlxlearn
from mlxlearn._common.config import tuning_context
from mlxlearn._common.diagnostics import diagnostics_summary, record_event
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


def test_ring_buffer_is_bounded():
    """An agent harness fitting in a loop for hours must not accumulate forever."""
    mlxlearn.clear_backend_diagnostics()
    with tuning_context(diagnostics_capacity=8):
        from mlxlearn._common.diagnostics import _recorder

        _recorder.resize(8)
        for i in range(50):
            record_event(f"E{i}", "fit", "mlx")
        events = mlxlearn.get_backend_diagnostics()
    assert len(events) == 8
    assert events[-1].estimator == "E49"


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
