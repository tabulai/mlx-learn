"""Shared fixtures.

Two rules the whole suite depends on.

**Every test starts from default configuration.** Configuration is thread-local
and scoped, but a test that calls ``set_config`` without a context manager would
leak into whatever pytest runs next, and the resulting failure would point at the
wrong test.

**Nothing is patched unless a test patches it.** Patching mutates scikit-learn's
module attributes process-wide. A leaked patch would make the Layer 1 suite
silently exercise Layer 2 classes, which is exactly the confusion the two-layer
split exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlxlearn._common.config import reset_config
from mlxlearn._common.diagnostics import clear_backend_diagnostics
from mlxlearn.patching import unpatch_sklearn


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    reset_config()
    clear_backend_diagnostics()
    yield
    unpatch_sklearn()
    reset_config()
    clear_backend_diagnostics()


@pytest.fixture
def rng():
    return np.random.default_rng(20260726)


@pytest.fixture
def classification_data(rng):
    """A dataset large enough to select the MLX backend, small enough to be quick."""
    n_samples, n_features, n_classes = 6000, 24, 3
    centers = rng.normal(scale=2.0, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, n_samples)
    X = centers[y] + rng.normal(size=(n_samples, n_features))
    return X, y


@pytest.fixture
def small_classification_data(rng):
    """Below every crossover, so Layer 1 takes its internal CPU path."""
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def regression_data(rng):
    X = rng.normal(size=(6000, 16))
    w = rng.normal(size=16)
    return X, X @ w + rng.normal(scale=0.1, size=6000)
