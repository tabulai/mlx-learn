"""Adversarial float32 cases for the neighbor kernel.

These exist because the expanded Gram identity ``‖q‖² + ‖t‖² − 2⟨q,t⟩`` is the fast
way to compute distances and the numerically hostile one. Each test pins a specific
claim from ``docs/fp32_policy.md`` §2.1.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

from mlxlearn._kernels.distance import build_index
from mlxlearn.neighbors import NearestNeighbors
from tests._helpers import assert_neighbor_parity, assert_ties_break_on_smallest_index

BACKENDS = ["mlx", "cpu"]


def _adversarial_column(n=120_000):
    """``n - 5`` points bunched at 1.0004 and five true nearest at 1.0000.

    The five correct answers are separated from the decoys by 4e-4 while being
    separated from *each other* by 1e-8, so a kernel that ranks on unrefined
    float32 Gram values returns the wrong five.
    """
    X = np.empty((n, 1), dtype=np.float64)
    X[: n - 5, 0] = 1.0004 + np.arange(n - 5) * 1e-8
    X[n - 5 :, 0] = 1.0000 + np.arange(5) * 1e-8
    return X


@pytest.mark.parametrize("backend", BACKENDS)
def test_refinement_recovers_the_true_neighbors(backend):
    """Without the refinement pass this returns indices from the decoy cluster."""
    X = _adversarial_column()
    _, idx = build_index(X, backend=backend).query(np.zeros((1, 1), np.float32), 5)
    np.testing.assert_array_equal(np.sort(idx[0]), np.arange(len(X) - 5, len(X)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_large_offset_does_not_destroy_accuracy(backend, rng):
    """Coordinates at 1e4 with separations at 1e-2.

    This is the case centering exists for: without subtracting the training mean,
    ‖q‖² and ‖t‖² are ~1e8 and their difference has to resolve ~1e-4, which
    float32 cannot do. Distances must still match a float64 reference exactly.
    """
    X = (1e4 + rng.normal(scale=0.05, size=(50_000, 4))).astype(np.float32)
    Q = (1e4 + rng.normal(scale=0.05, size=(200, 4))).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query(Q, 7)
    ref = SkNearestNeighbors(n_neighbors=7, algorithm="brute").fit(X.astype(np.float64))
    ref_dist, ref_idx = ref.kneighbors(Q.astype(np.float64))

    # Distances must be exact. float32 quantizes this data to ~370 levels per
    # coordinate, so exact ties at the k-th boundary are common and the neighbor
    # set is genuinely ambiguous there; the helper checks that every disagreement
    # is such a tie.
    np.testing.assert_allclose(dist, ref_dist, atol=1e-6, rtol=1e-6)
    assert_neighbor_parity(dist, idx, ref_dist, ref_idx)
    assert_ties_break_on_smallest_index(dist, idx)


@pytest.mark.parametrize("backend", BACKENDS)
def test_accuracy_is_relative_to_the_float32_input(backend):
    """A limit that is about storage, not arithmetic — policy §3.2.

    Shift the adversarial data to coordinate 1e4 and float32 has a spacing of
    ~1e-3 there, so the 1e-8 increments are gone before any distance is computed:
    120 000 points collapse to three distinct values. mlxlearn must be correct for
    the data it was given, and it cannot be correct for data it never saw.
    """
    X32 = (_adversarial_column() + 1e4).astype(np.float32)
    assert len(np.unique(X32)) < 10, "precondition: the input itself has collapsed"

    query = np.array([[1e4]], dtype=np.float32)
    dist, idx = build_index(X32, backend=backend).query(query, 5)

    # The honest reference is float64 arithmetic on the float32-rounded data.
    ref = SkNearestNeighbors(n_neighbors=5, algorithm="brute").fit(X32.astype(np.float64))
    ref_dist, ref_idx = ref.kneighbors(query.astype(np.float64))

    np.testing.assert_array_equal(np.sort(idx, axis=1), np.sort(ref_idx, axis=1))
    np.testing.assert_allclose(dist, ref_dist, atol=1e-6)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_float64_values_that_overflow_float32_are_rejected():
    """1e300 is an ordinary float64 and is infinity in float32.

    Converting it silently would produce garbage attributable to nothing. NumPy
    emits its own overflow warning during the cast, which is filtered here
    because the error is the assertion.
    """
    X = np.array([[1e300], [1.0], [2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        NearestNeighbors(n_neighbors=2).fit(X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_duplicate_rows_are_ordered_by_index(backend):
    """Exact ties must produce a deterministic, index-ascending order."""
    X = np.repeat(np.array([[3.0]], dtype=np.float32), 50, axis=0)
    _, idx = build_index(X, backend=backend).query(np.zeros((1, 1), np.float32), 10)
    np.testing.assert_array_equal(idx[0], np.arange(10))


@pytest.mark.parametrize("backend", BACKENDS)
def test_high_dimensional_accumulation(backend, rng):
    """1024 features — the embedding case — still matches a float64 reference."""
    X = rng.normal(size=(4000, 1024)).astype(np.float32)
    Q = rng.normal(size=(16, 1024)).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref = SkNearestNeighbors(n_neighbors=5, algorithm="brute").fit(X.astype(np.float64))
    ref_dist, ref_idx = ref.kneighbors(Q.astype(np.float64))

    np.testing.assert_array_equal(np.sort(idx, axis=1), np.sort(ref_idx, axis=1))
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-4, atol=1e-4)
