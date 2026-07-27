"""Cross-cutting invariants that hold for every estimator.

These are the properties the ancestor violated, asserted once across the whole
public surface rather than re-derived per estimator. A new estimator gets them
for free by appearing in ``ESTIMATORS`` — and fails loudly if it does not honor
them.
"""

from __future__ import annotations

import pickle
import threading

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from mlxlearn._common.config import tuning_context
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.svm import SVC

SUPERVISED = [
    (KNeighborsClassifier, {"n_neighbors": 5}),
    (KNeighborsRegressor, {"n_neighbors": 5}),
    (LogisticRegression, {}),
    (SVC, {}),
]
ALL = SUPERVISED + [(NearestNeighbors, {"n_neighbors": 5})]


@pytest.fixture(scope="module")
def data():
    return make_classification(
        n_samples=3000, n_features=12, n_informative=6, n_redundant=2, random_state=0
    )


def _fit(cls, kwargs, X, y, backend="mlx"):
    with tuning_context(force_backend=backend):
        return cls(**kwargs).fit(X, y)


def _output(model, X):
    if hasattr(model, "predict"):
        return model.predict(X)
    return model.kneighbors(X)[1]


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_pickle_round_trip_preserves_predictions(cls, kwargs, data):
    """Device-resident state must not make a fitted model unpicklable.

    ``mx.array`` objects are dropped on pickle and rebuilt lazily from the host
    copy, so the file stays portable to a machine with a different MLX build or
    no GPU at all.
    """
    X, y = data
    model = _fit(cls, kwargs, X, y)
    restored = pickle.loads(pickle.dumps(model))

    assert restored.execution_backend_ == model.execution_backend_
    np.testing.assert_array_equal(_output(model, X[:200]), _output(restored, X[:200]))


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_clone_produces_an_unfitted_equal_estimator(cls, kwargs, data):
    X, y = data
    model = _fit(cls, kwargs, X, y)
    fresh = clone(model)

    assert fresh.get_params() == model.get_params()
    with pytest.raises(AttributeError):
        fresh.execution_backend_


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_mutating_the_training_array_after_fit_changes_nothing(cls, kwargs, data):
    """No aliasing of the caller's buffer, no cache keyed on its identity.

    The ancestor kept module-level weakrefs to the last training matrix and
    ``id()``-keyed LRU caches. CPython recycles ``id()`` after a collection, so a
    freed array whose address was reused produced predictions computed on the
    wrong data — silently.
    """
    X, y = data
    X = np.array(X)
    query = X[:100].copy()

    model = _fit(cls, kwargs, X, y)
    before = _output(model, query)

    X[:] = 0.0
    np.testing.assert_array_equal(before, _output(model, query))


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_fit_is_a_pure_function_of_its_arguments(cls, kwargs, data):
    """``fit(A); fit(B)`` must equal ``fit(B)``.

    The ancestor's continuation cache warm-started from whatever had been fitted
    earlier in the process, on data of the same shape.
    """
    X, y = data
    X_other, y_other = make_classification(
        n_samples=3000, n_features=12, n_informative=6, n_redundant=2, random_state=99
    )

    with tuning_context(force_backend="mlx"):
        sequential = cls(**kwargs).fit(X_other, y_other).fit(X, y)
        fresh = cls(**kwargs).fit(X, y)

    np.testing.assert_array_equal(_output(sequential, X[:200]), _output(fresh, X[:200]))


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_concurrent_fits_match_sequential_fits(cls, kwargs):
    """Threads must not contend through a process-global device setting.

    ``use_device`` scopes with ``mx.stream`` rather than mutating MLX's default
    device, so two threads fitting at once cannot disturb each other.
    """
    def build(seed):
        return make_classification(
            n_samples=1500, n_features=8, n_informative=5, n_redundant=1, random_state=seed
        )

    seeds = (1, 2, 3, 4)
    expected = {}
    for seed in seeds:
        X, y = build(seed)
        expected[seed] = _output(_fit(cls, kwargs, X, y), X[:80])

    observed: dict[int, np.ndarray] = {}
    errors: list[BaseException] = []

    def worker(seed):
        try:
            X, y = build(seed)
            observed[seed] = _output(_fit(cls, kwargs, X, y), X[:80])
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in seeds]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    for seed in seeds:
        np.testing.assert_array_equal(observed[seed], expected[seed])


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_repeated_predictions_are_deterministic(cls, kwargs, data):
    X, y = data
    model = _fit(cls, kwargs, X, y)
    first = _output(model, X[:200])
    for _ in range(3):
        np.testing.assert_array_equal(_output(model, X[:200]), first)


@pytest.mark.parametrize("cls, kwargs", ALL, ids=lambda v: getattr(v, "__name__", ""))
def test_backends_reach_the_same_answer(cls, kwargs, data):
    """The MLX and internal-CPU paths are one implementation on two devices."""
    X, y = data
    gpu = _output(_fit(cls, kwargs, X, y, backend="mlx"), X[:200])
    cpu = _output(_fit(cls, kwargs, X, y, backend="cpu"), X[:200])

    if gpu.dtype.kind == "f":
        np.testing.assert_allclose(gpu, cpu, atol=1e-3, rtol=1e-3)
    else:
        agreement = np.mean(gpu == cpu)
        assert agreement >= 0.99, f"backends disagree on {1 - agreement:.1%} of predictions"


#: The only places allowed to catch ``Exception``, each with the reason it must.
#:
#: A bare ``except Exception`` around accelerated work is how the ancestor made
#: its own bugs invisible: a padding crash reached users as a slightly slow but
#: correct answer, and survived for exactly as long as the handler did. Neither
#: entry below wraps accelerated work.
ALLOWED_BROAD_EXCEPTS = {
    "_common/base.py": "mlx_guard re-raises as BackendExecutionError; it never swallows",
    "_common/device.py": (
        "the GPU probe answers one boolean question, and MLX raises several "
        "unrelated exception types when Metal is missing or unusable"
    ),
}


def test_broad_exception_handlers_are_only_where_they_are_justified():
    """No new ``except Exception`` without a deliberate entry above."""
    import pathlib

    import mlxlearn

    root = pathlib.Path(mlxlearn.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("except Exception") and relative not in ALLOWED_BROAD_EXCEPTS:
                offenders.append(f"{relative}:{number}")

    assert not offenders, (
        "broad exception handlers outside the justified set: "
        f"{offenders}. Add an entry to ALLOWED_BROAD_EXCEPTS with a reason, or "
        "catch something specific."
    )
