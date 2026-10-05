"""The exception hierarchy survives the trip through a worker process.

joblib returns a worker's exception to the parent by pickling it. An exception
whose ``__init__`` has a required keyword-only argument cannot be rebuilt by the
default ``cls(*args)``, so the parent got ``BrokenProcessPool`` instead of the
precise error — with the message gone from the traceback entirely.
"""

from __future__ import annotations

import copy
import pickle
import sys

import numpy as np
import pytest

from mlxlearn import exceptions
from mlxlearn.exceptions import (
    BackendExecutionError,
    CapabilityError,
    MLXLearnError,
    UnsupportedParameterError,
)

#: One instance of every public class, built with non-default attributes so a
#: round trip that dropped any of them would show.
INSTANCES = {
    "MLXLearnError": lambda: exceptions.MLXLearnError("base"),
    "CapabilityError": lambda: exceptions.CapabilityError("capability", reason="custom-slug"),
    "UnsupportedParameterError": lambda: exceptions.UnsupportedParameterError(
        "no liblinear", parameter="solver", value="liblinear"
    ),
    "UnsupportedInputError": lambda: exceptions.UnsupportedInputError(
        "too large", reason="float32-overflow", advice="Or run under patch_sklearn()."
    ),
    "SparseInputError": lambda: exceptions.SparseInputError(
        "sparse input", advice="Under patch_sklearn() it is handed over."
    ),
    "InvalidInputError": lambda: exceptions.InvalidInputError("class 3 has zero weight"),
    "UnsupportedEstimatorStateError": lambda: exceptions.UnsupportedEstimatorStateError(
        "state", reason="custom-state"
    ),
    "DeviceError": lambda: exceptions.DeviceError("no gpu"),
    "BackendExecutionError": lambda: exceptions.BackendExecutionError(
        "kernel failed", original=RuntimeError("metal said no")
    ),
    "MLXLearnFallbackWarning": lambda: exceptions.MLXLearnFallbackWarning("fell back"),
    "MLXLearnPerformanceWarning": lambda: exceptions.MLXLearnPerformanceWarning("slow"),
    "MLXLearnPatchWarning": lambda: exceptions.MLXLearnPatchWarning("patch"),
}


def _state(exc):
    """What a round trip must preserve, with nested exceptions compared by value."""
    state = {}
    for key, value in vars(exc).items():
        if isinstance(value, BaseException):
            value = (type(value), value.args)
        state[key] = value
    return state


def test_every_public_class_has_an_instance_here():
    """A new class must be added above, or its pickling goes untested."""
    assert set(INSTANCES) == set(exceptions.__all__)


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_round_trips_through_pickle(name, protocol):
    original = INSTANCES[name]()
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))

    assert type(restored) is type(original)
    assert restored.args == original.args
    assert str(restored) == str(original)
    assert _state(restored) == _state(original)


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_copies(name):
    original = INSTANCES[name]()
    for duplicate in (copy.copy(original), copy.deepcopy(original)):
        assert type(duplicate) is type(original)
        assert _state(duplicate) == _state(original)


def test_advice_is_part_of_the_message_and_kept_apart_from_it():
    """Layer 1 shows the advice; a patched estimator reports ``summary`` without it."""
    exc = UnsupportedParameterError(
        "gamma=0 unsupported.", parameter="gamma", value=0.0, advice="Use patch_sklearn()."
    )
    assert str(exc) == "gamma=0 unsupported. Use patch_sklearn()."
    assert exc.summary == "gamma=0 unsupported."
    assert exc.advice == "Use patch_sklearn()."
    restored = pickle.loads(pickle.dumps(exc))
    assert (restored.summary, restored.advice) == (exc.summary, exc.advice)

    plain = CapabilityError("no advice")
    assert str(plain) == plain.summary == "no advice" and plain.advice == ""


def test_restored_error_keeps_its_family_and_attributes():
    exc = UnsupportedParameterError("p=1 unsupported", parameter="p", value=1)
    restored = pickle.loads(pickle.dumps(exc))

    assert isinstance(restored, CapabilityError)
    assert isinstance(restored, ValueError)
    assert restored.parameter == "p"
    assert restored.value == 1
    assert restored.reason == "unsupported-parameter:p"


def test_an_unpicklable_value_degrades_to_its_repr():
    """``value`` is whatever the user passed; a lambda must not sink the whole error."""
    weights = lambda d: 1 / d  # noqa: E731 - the point is that it cannot be pickled
    exc = UnsupportedParameterError("callable weights", parameter="weights", value=weights)

    restored = pickle.loads(pickle.dumps(exc))

    assert str(restored) == "callable weights"
    assert restored.parameter == "weights"
    assert restored.value == repr(weights)
    assert exc.value is weights  # the live exception is untouched


def test_backend_error_keeps_its_original():
    exc = BackendExecutionError("failed", original=ValueError("inner"))
    restored = pickle.loads(pickle.dumps(exc))
    assert isinstance(restored.original, ValueError)
    assert restored.original.args == ("inner",)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="add_note is new in Python 3.11")
def test_notes_survive():
    exc = MLXLearnError("with a note")
    exc.add_note("context")
    assert pickle.loads(pickle.dumps(exc)).__notes__ == ["context"]


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_parallel_cross_validation_surfaces_the_precise_error():
    """The case that broke: n_jobs > 1 with error_score='raise'."""
    from sklearn.model_selection import cross_val_score

    from mlxlearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)

    with pytest.raises(UnsupportedParameterError) as excinfo:
        cross_val_score(KNeighborsClassifier(p=1), X, y, cv=2, n_jobs=2, error_score="raise")
    assert excinfo.value.parameter == "p"
