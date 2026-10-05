"""Adversarial float32 cases for the neighbor kernel.

These exist because the expanded Gram identity ``‖q‖² + ‖t‖² − 2⟨q,t⟩`` is the fast
way to compute distances and the numerically hostile one. Each test pins a specific
claim from ``docs/fp32_policy.md`` §2.1 or §3.1.

Most compare against :func:`_exact_neighbors` — float64 direct differences ranked by
``(distance, index)`` — rather than against scikit-learn, which defines no order
among ties and whose own brute-force path uses the same identity in float64.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.neighbors import KNeighborsRegressor as SkKNeighborsRegressor
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

from mlxlearn import clear_backend_diagnostics, get_backend_diagnostics
from mlxlearn._common.config import tuning_context
from mlxlearn._kernels import distance
from mlxlearn._kernels.distance import build_index
from mlxlearn.exceptions import MLXLearnFallbackWarning, UnsupportedInputError
from mlxlearn.neighbors import NearestNeighbors
from mlxlearn.patching import _estimators as patched
from tests._helpers import assert_neighbor_parity, assert_ties_break_on_smallest_index

BACKENDS = ["mlx", "cpu"]


def _exact_neighbors(X, Q, k):
    """The first ``k`` training rows under ``(distance, index)``, in float64.

    Direct differences, not the Gram identity, so the reference has no cancellation
    of its own; and a lexicographic sort, so ties are ranked the way mlxlearn
    promises to rank them.
    """
    X = np.asarray(X, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    order = np.arange(X.shape[0])
    dist = np.empty((Q.shape[0], k))
    idx = np.empty((Q.shape[0], k), dtype=np.int64)
    for row, q in enumerate(Q):
        d2 = np.sum((X - q) ** 2, axis=1)
        first = np.lexsort((order, d2))[:k]
        idx[row] = first
        dist[row] = np.sqrt(d2[first])
    return dist, idx


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
    """A limit that is about storage, not arithmetic — policy §3.4.

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


# ----------------------------------------------------------------------------------
# selection the Gram noise cannot fool — the certificate, §2.1
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_tight_clusters_return_the_exact_neighbors(backend, rng):
    """Clusters whose spread is 1e-3 of their distance from the mean.

    Every member of such a cluster sits inside the Gram noise band together, so a
    fixed over-selection margin kept arbitrary members: half the rows came back
    with a wrong neighbor set, on the CPU path too, and nothing was raised. The
    certificate has to catch every one of those rows.
    """
    centers = rng.normal(size=(200, 32))
    X = (np.repeat(centers, 40, axis=0) + rng.normal(scale=1e-3, size=(8000, 32))).astype(
        np.float32
    )
    Q = (centers[rng.integers(0, 200, 60)] + rng.normal(scale=1e-3, size=(60, 32))).astype(
        np.float32
    )

    stats = {}
    dist, idx = build_index(X, backend=backend).query(Q, 5, stats=stats)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    assert stats["widened_rows"] > 0, "precondition: the first pass was fooled"
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_make_blobs_with_a_small_cluster_std(backend):
    """The same failure on stock ``make_blobs`` data, as float64 input."""
    X, _ = make_blobs(n_samples=8000, centers=40, n_features=16, cluster_std=0.01, random_state=0)
    Q = X[::131] + 0.005

    dist, idx = build_index(X, backend=backend).query(Q, 10)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 10)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_unscaled_column_does_not_hide_neighbors(backend, rng):
    """Ordinary unscaled tabular data: one column spanning 0-1e4 beside unit ones.

    The wide column dominates every norm, so the Gram noise swamps the unit
    columns' contribution; at 50 000 rows the fixed margin returned a wrong set
    for 12% of queries and moved regressor predictions by up to 0.85.
    """
    X = rng.normal(size=(20_000, 16))
    X[:, 0] = rng.uniform(0, 1e4, 20_000)
    Q = rng.normal(size=(100, 16))
    Q[:, 0] = rng.uniform(0, 1e4, 100)

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_far_training_row_does_not_disable_the_first_pass(backend, rng):
    """The certificate's bound is norm-aware.

    Bounding the Gram error with the single largest training norm would let one
    outlier make every row uncertifiable — and send all of them to the fallback.
    Ordinary data must certify on the first pass, outlier or not.
    """
    X = rng.normal(size=(20_000, 32)).astype(np.float32)
    X[123] = 1e3
    Q = rng.normal(size=(200, 32)).astype(np.float32)

    stats = {}
    build_index(X, backend=backend).query(Q, 10, stats=stats)
    assert stats == {"rows": 200, "widened_rows": 0, "widen_skipped_rows": 0, "exact_rows": 0}


def _far_cluster(rng, members, *, d=4):
    """``members`` rows within 1e-4 of a point 1000 out on every axis, and queries
    among them.

    Every member sits inside the Gram noise band of a query there, together, so the
    first pass cannot certify one; a second pass carrying 265 candidates can when
    the cluster is smaller than that, and cannot when it is larger.
    """
    X = np.vstack(
        [
            1000.0 + 1e-4 * rng.normal(size=(members, d)),
            -1000.0 + rng.normal(size=(members, d)),
            50.0 * rng.normal(size=(3000, d)),
        ]
    )
    return X, 1000.0 + 1e-4 * rng.normal(size=(20, d))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("members", "skipped", "exact"),
    [
        (100, 0, 0),  # the second pass certifies every row
        (400, 0, 20),  # it cannot, but the count is short of twice its carry: tried
        (2000, 20, 20),  # a count past twice its carry: straight to the exact search
    ],
)
def test_the_second_pass_is_skipped_only_where_it_cannot_certify(
    backend, members, skipped, exact, rng, monkeypatch
):
    """Which search answers a row changes its time, never its answer: forcing every
    uncertified row through the second pass, or past it, returns the same bits."""
    X, Q = _far_cluster(rng, members)
    index = build_index(X, backend=backend)

    stats = {}
    dist, idx = index.query(Q, 5, stats=stats)
    assert stats == {"rows": 20, "widened_rows": 20, "widen_skipped_rows": skipped,
                     "exact_rows": exact}
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)

    for forced in (True, False):
        with monkeypatch.context() as patch:
            patch.setattr(
                distance._Search, "cannot_widen",
                lambda self, qb, gmax, k_wide, forced=forced: np.full(qb.n, forced),
            )
            _assert_same_bits(index.query(Q, 5), (dist, idx))


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_repeating_in_step_with_the_sample_still_get_the_second_pass(backend, rng):
    """1008 rows tiled 40 times, and queries beside the rows at multiples of 16.

    Each query has the 40 copies of its row in its Gram noise band: too many for the
    first pass, which carries 14, and few enough for the second, which carries 270
    and certifies every one. The count before it samples one training row in 16.
    Every 16th row, which it used to take, held all 40 copies of each of these rows
    — 1008 is a multiple of 16 — so it counted 640, past twice the second pass's
    carry, and sent every query straight to the exact search.
    """
    base = rng.normal(size=(1008, 16))
    X = np.tile(base, (40, 1))
    Q = base[::16][:32] + 1e-6 * rng.normal(size=(32, 16))
    index = build_index(X, backend=backend)

    stats = {}
    dist, idx = index.query(Q, 10, stats=stats)
    assert stats == {"rows": 32, "widened_rows": 32, "widen_skipped_rows": 0, "exact_rows": 0}
    ref_dist, ref_idx = _exact_neighbors(X, Q, 10)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


#: Far points for clusters of 1e-4 spread, and what the search does with ten queries
#: in each: 2000 members are past twice the second pass's carry (the count skips it),
#: 100 within its carry (it certifies them), 400 between (it is tried, and fails).
_CLUSTER_CENTERS = {
    2000: np.full(4, 1000.0),
    100: np.full(4, -1000.0),
    400: np.array([1000.0, -1000.0, 1000.0, -1000.0]),
}


@pytest.mark.parametrize(
    ("members", "skipped", "exact", "account"),
    [
        (
            (2000,), 10, 10,
            "10 of 10 query rows could not be certified by the first float32 pass; 10 of "
            "them went straight to the exact search, where a count on a sample of the "
            "training rows estimated that more candidates could not certify them either",
        ),
        (
            (2000, 100), 10, 10,
            "20 of 20 query rows could not be certified by the first float32 pass; 10 of "
            "them went straight to the exact search, where a count on a sample of the "
            "training rows estimated that more candidates could not certify them either; "
            "10 were searched again with more candidates, which certified them",
        ),
        (
            (2000, 100, 400), 10, 20,
            "30 of 30 query rows could not be certified by the first float32 pass; 10 of "
            "them went straight to the exact search, where a count on a sample of the "
            "training rows estimated that more candidates could not certify them either; "
            "20 were searched again with more candidates, and 10 of those needed the exact "
            "search too",
        ),
        (
            (100, 400), 0, 10,
            "20 of 20 query rows could not be certified by the first float32 pass and were "
            "searched again with more candidates; 10 of them needed the exact search",
        ),
    ],
)
def test_the_diagnostics_account_for_every_row_once(members, skipped, exact, account, rng):
    """One event per call, and in it each row the first pass could not certify, once:
    skipped to the exact search, certified by the second pass, or searched exactly
    after it. The time a query took is explained, and the parts add up."""
    clusters = [_CLUSTER_CENTERS[m] + 1e-4 * rng.normal(size=(m, 4)) for m in members]
    X = np.vstack([*clusters, 50.0 * rng.normal(size=(3000, 4))])
    Q = np.vstack([_CLUSTER_CENTERS[m] + 1e-4 * rng.normal(size=(10, 4)) for m in members])
    with tuning_context(force_backend="cpu"):
        model = NearestNeighbors(n_neighbors=5).fit(X)
    clear_backend_diagnostics()
    model.kneighbors(Q)
    (event,) = get_backend_diagnostics()
    assert {key: event.extra[key] for key in ("search", "widened_rows", "widen_skipped_rows",
                                              "exact_rows")} == {
        "search": "certificate-fallback",
        "widened_rows": len(Q),
        "widen_skipped_rows": skipped,
        "exact_rows": exact,
    }
    assert event.detail == account + ". The neighbors returned are exact either way."


def _assert_same_bits(got, want):
    for a, b in zip(got, want, strict=True):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


def _kind(kind, n, d, rng):
    """Training rows and 24 queries of a kind that reaches every route of the search."""
    def draw(m):
        return rng.normal(size=(m, d))

    if kind == "gaussian":
        return draw(n), draw(24)
    if kind == "unscaled":
        X, Q = draw(n), draw(24)
        X[:, 0] = rng.uniform(0, 1e4, n)
        Q[:, 0] = rng.uniform(0, 1e4, 24)
        return X, Q
    if kind in ("duplicates", "jittered"):
        base = draw(max(n // 20, 1))
        jitter = 1e-4 if kind == "jittered" else 0.0
        X = base[rng.integers(0, len(base), n)] + jitter * draw(n)
        Q = base[rng.integers(0, len(base), 24)] + 1e-4 * draw(24)
        return X, Q
    if kind == "binary":
        return (rng.random((n, d)) < 0.2).astype(np.float64), (rng.random((24, d)) < 0.2) * 1.0
    if kind == "counts":
        return rng.poisson(2.0, (n, d)).astype(np.float64), rng.poisson(2.0, (24, d)) * 1.0
    raise ValueError(kind)


def _every_route(index, Q, k, monkeypatch, **kwargs):
    """``index.query`` under each way of answering a row, which must agree bit for bit:
    as it chooses; every uncertified row through the second pass; every one past it;
    the exact search in the old fixed-width tiles; and 64-bit keys throughout."""
    answer = index.query(Q, k, **kwargs)
    forced = [
        (distance._Search, "cannot_widen", lambda self, qb, gmax, k_wide: np.zeros(qb.n, bool)),
        (distance._Search, "cannot_widen", lambda self, qb, gmax, k_wide: np.ones(qb.n, bool)),
        (distance, "_exact_cols", lambda budget, rows, pair_bytes: 4096),
        (distance, "_int_key_bits", lambda max_sq, n_train: None),
    ]
    for owner, name, replacement in forced:
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, replacement)
            _assert_same_bits(index.query(Q, k, **kwargs), answer)
    return answer


_KINDS = ["gaussian", "unscaled", "duplicates", "jittered", "binary", "counts"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(("n", "d"), [(3000, 4), (20_000, 16)])
def test_every_route_returns_the_same_neighbors(backend, dtype, kind, n, d, rng, monkeypatch):
    """How a row is answered — first pass, second pass, the exact search in any tile
    width, 32- or 64-bit keys — is a matter of time only; each is the first ``k``
    under ``(refined distance, index)``, and that is the float64 answer."""
    X, Q = (a.astype(dtype) for a in _kind(kind, n, d, rng))
    index = build_index(X, backend=backend)

    dist, idx = _every_route(index, Q, 10, monkeypatch)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 10)
    if kind in ("binary", "counts"):
        np.testing.assert_array_equal(idx, ref_idx)
        np.testing.assert_array_equal(dist, ref_dist)
    else:
        assert_neighbor_parity(dist, idx, ref_dist, ref_idx)
    if n <= 3000:
        _every_route(index, index.X, 7, monkeypatch, exclude_self=True)


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(("n", "d"), [(1_000_000, 16), (65_536, 256)])
def test_every_route_returns_the_same_neighbors_at_scale(
    backend, dtype, kind, n, d, rng, monkeypatch
):
    """At a million rows the unscaled column sends some rows past the second pass on
    its own, and most of the rest to the exact search after it."""
    X, Q = (a.astype(dtype) for a in _kind(kind, n, d, rng))
    dist, idx = _every_route(build_index(X, backend=backend), Q, 10, monkeypatch)
    ref_dist, ref_idx = _exact_neighbors(X, Q[:4], 10)
    assert_neighbor_parity(dist[:4], idx[:4], ref_dist, ref_idx)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["gaussian", "duplicates", "binary"])
@pytest.mark.parametrize(("n", "k"), [(300, 300), (400, 100), (600, 100), (1030, 1000)])
def test_every_route_agrees_for_k_near_the_number_of_rows(backend, kind, n, k, rng, monkeypatch):
    """With ``k`` near ``n`` the second pass may carry every row, or the first may."""
    X, Q = _kind(kind, n, 16, rng)
    index = build_index(X, backend=backend)
    dist, idx = _every_route(index, Q, k, monkeypatch)
    assert_neighbor_parity(dist, idx, *_exact_neighbors(X, Q, k))
    _every_route(index, index.X, k - 1, monkeypatch, exclude_self=True)


# ----------------------------------------------------------------------------------
# refinement from the input's own coordinates, §2.1
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_cluster_far_from_the_global_mean(backend, rng):
    """A cluster at the origin with spread 1e-3, and another at 1000.

    The mean sits near 500, where float32's spacing is 3e-5 — thirty times the
    near cluster's own spread — so translating the rows by it erased the cluster
    before a distance was computed. Selection has to be certified and refinement
    has to use the input's own coordinates to get these right.
    """
    X = np.vstack(
        [rng.normal(scale=1e-3, size=(2000, 4)), 1000 + rng.normal(size=(2000, 4))]
    ).astype(np.float32)
    Q = np.vstack(
        [rng.normal(scale=1e-3, size=(30, 4)), 1000 + rng.normal(size=(30, 4))]
    ).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_refined_distances_are_relative_to_the_input(backend, rng):
    """Selection is easy here, and the neighbor sets were always right.

    But distances were refined from mean-translated coordinates, whose rounding is
    set by the distance to the mean (~960) rather than by the data, and most of
    them missed the documented ``atol=rtol=1e-6``.
    """
    X = np.vstack(
        [rng.normal(scale=10, size=(200, 4)), 1000 + rng.normal(size=(3000, 4))]
    ).astype(np.float32)
    Q = rng.normal(scale=10, size=(60, 4)).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------------------
# the (distance, index) order across the k-th boundary, §3.1
# ----------------------------------------------------------------------------------


def _subsampled_grid(rng, side=20):
    """Integer points on a 3-D grid, with 40% removed at random.

    Subsampled so that the mean is not a lattice point: translating a full grid by
    its own center leaves a symmetric lattice on which the identity happens to be
    exact, and nothing would be tested.
    """
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return grid[rng.random(len(grid)) < 0.6].astype(np.float32)


#: (grid spacing, query offset in grid units). Integer data with integer and with
#: half-integer queries takes the exact small-integer path; half-integer data is not
#: integral, so quarter-integer queries go through the certificate and its fallback.
#: Every distance is exact in all three.
_TIE_CASES = {
    "integer-queries": (1.0, 0.0),
    "half-integer-queries": (1.0, 0.5),
    "certified-path": (0.5, 0.5),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(_TIE_CASES))
def test_boundary_ties_resolve_to_the_smallest_indices(backend, case, rng):
    """More rows tie at the k-th distance than any fixed margin carries.

    Which of them are returned must follow the global ``(distance, index)`` order —
    not merely an ordering of whichever tied rows survived selection, which is
    what the fixed margin produced on a third of these rows.
    """
    spacing, offset = _TIE_CASES[case]
    X = _subsampled_grid(rng) * spacing
    Q = (rng.integers(0, 20, size=(60, 3)) + offset).astype(np.float32) * spacing

    stats = {}
    dist, idx = build_index(X, backend=backend).query(Q, 7, stats=stats)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 7)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(dist, ref_dist)
    if case == "certified-path":
        assert stats["widened_rows"] > 0, "precondition: ties defeated the first pass"
    else:
        assert stats["widened_rows"] == 0, "small-integer data should need no fallback"


@pytest.mark.parametrize("backend", BACKENDS)
def test_k_neighbors_are_a_prefix_of_more_neighbors(backend, rng):
    """``kneighbors(k)`` must equal ``kneighbors(k + m)[:, :k]``, ties included."""
    X = _subsampled_grid(rng)
    Q = rng.integers(0, 20, size=(60, 3)).astype(np.float32) + 0.5
    index = build_index(X, backend=backend)

    _, wide = index.query(Q, 16)
    for k in (1, 2, 5, 9):
        _, idx = index.query(Q, k)
        np.testing.assert_array_equal(idx, wide[:, :k], err_msg=f"k={k}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_self_query_neighbors_follow_the_order_without_the_row_itself(backend, rng):
    """``kneighbors()`` with no query, on binary data, where duplicates abound.

    A row with more than ``k`` exact duplicates before it is not among its own
    ``k + 1`` candidates, so one candidate has to go. Dropping column 0, as
    scikit-learn does, removed the lowest-indexed duplicate — the one the
    ``(distance, index)`` order keeps — so k=3 was not a prefix of k=6 on 96 of 3000
    rows. The answer is the first ``k`` under that order once the row is left out.
    """
    X = (rng.random((3000, 5)) < 0.5).astype(np.float32)
    index = build_index(X, backend=backend)

    order = np.arange(len(X))
    expected = np.empty((len(X), 12), dtype=np.int64)
    for row, x in enumerate(X.astype(np.float64)):
        d2 = np.sum((X - x) ** 2, axis=1)
        d2[row] = np.inf
        expected[row] = np.lexsort((order, d2))[:12]

    for k in (1, 2, 3, 6, 12):
        _, idx = index.query(index.X, k, exclude_self=True)
        np.testing.assert_array_equal(idx, expected[:, :k], err_msg=f"k={k}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("spacing", [1.0, 0.5], ids=["integer-path", "certified-path"])
def test_tie_heavy_results_do_not_depend_on_blocking(backend, spacing, rng):
    """Binary features: the train block size used to change the CPU path's answer.

    ``{0, 1}`` takes the exact small-integer path; ``{0, 0.5}`` with queries a
    quarter off it keeps every distance exact but is not integral, so it goes
    through the certificate and its fallback instead.
    """
    X = (rng.random((2000, 12)) < 0.5).astype(np.float32) * spacing
    Q = (rng.random((30, 12)) < 0.5).astype(np.float32) * spacing
    if spacing != 1.0:
        Q = Q + spacing / 2
    index = build_index(X, backend=backend)

    _, reference = index.query(Q, 7)
    np.testing.assert_array_equal(reference, _exact_neighbors(X, Q, 7)[1])
    for query_block in (1, 13, None):
        for train_block in (64, 257, None):
            _, idx = index.query(
                Q, 7, query_block_size=query_block, train_block_size=train_block
            )
            np.testing.assert_array_equal(idx, reference, err_msg=f"{query_block=} {train_block=}")


def test_exact_ties_resolve_identically_on_both_devices(rng):
    """Where the refined distances are exact, the device cannot matter.

    It did: on binary data the GPU and CPU returned different neighbor sets in
    every trial, from identical distances, and KNeighborsClassifier's predictions
    followed.
    """
    X = (rng.random((4000, 12)) < 0.5).astype(np.float32)
    Q = (rng.random((60, 12)) < 0.5).astype(np.float32)

    _, on_gpu = build_index(X, backend="mlx").query(Q, 7)
    _, on_cpu = build_index(X, backend="cpu").query(Q, 7)
    np.testing.assert_array_equal(on_gpu, on_cpu)


# ----------------------------------------------------------------------------------
# magnitudes and offsets the float32 cast used to destroy
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1e19, 1e-20])
def test_magnitudes_whose_squares_leave_float32(backend, scale):
    """1e19 and 1e-20 are ordinary float32 values; their squares are not.

    At 1e19 every squared distance overflowed to infinity, and at 1e-20 they
    underflowed — the GPU flushes those to zero — so every neighbor set was
    arbitrary. Scaling by an exact power of two brings both back.
    """
    X, _ = make_classification(n_samples=1000, n_features=8, random_state=0)
    X = X * scale
    Q = X[:40] + 0.01 * scale

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


def _with_a_fill_row(rng, fill, n=2000, d=4):
    """Unit-scale rows and one row of an unmasked fill value, as float32."""
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[123] = fill
    return X


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fill", [1e18, 1e20], ids=["1e18", "CMIP-fill-1e20"])
def test_one_fill_value_does_not_erase_the_other_rows_distances(backend, fill, rng):
    """One huge value used to set the scale for everyone.

    The span was normalized into [0.5, 1), which scaled every unit-scale gap down
    by the fill value and squared it below float32's normal range: from about 1e18
    the GPU returned every distance as zero and the neighbors in index order, the
    CPU followed from about 1e21, and nothing was raised. The scale-down now stops
    where the squares stop overflowing.
    """
    X = _with_a_fill_row(rng, fill)
    Q = rng.normal(size=(100, 4)).astype(np.float32)

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fill", [1e30, 9.97e36], ids=["1e30", "NetCDF-fill"])
def test_a_fill_value_float32_cannot_resolve_beside_is_refused_at_fit(backend, fill, rng):
    """No scale holds both a 1e37 span and unit gaps in float32: refused, not zeros.

    It is found at fit, where patched mode can still fall back.
    """
    with pytest.raises(UnsupportedInputError, match="resolve") as raised:
        build_index(_with_a_fill_row(rng, fill), backend=backend)
    assert raised.value.reason == "float32-overflow"
    assert "scikit-learn" in raised.value.advice


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_query_nearer_than_the_scale_resolves_is_rejected(backend, rng):
    """What fit could not rule out is still never answered with an underflowed zero.

    An exact duplicate of a training row is the one neighbor that legitimately has
    distance zero there.
    """
    X = _with_a_fill_row(rng, 1e20)
    index = build_index(X, backend=backend)

    _, idx = index.query(X[:3], 1)
    np.testing.assert_array_equal(idx[:, 0], [0, 1, 2])
    with pytest.raises(UnsupportedInputError, match="resolve") as raised:
        index.query(X[:3].astype(np.float64) + 1e-9, 1)
    assert raised.value.reason == "float32-overflow"


def test_a_patched_fit_on_fill_values_is_served_or_falls_back(rng):
    """Under patch_sklearn(): a 1e20 fill value is served on MLX and agrees with
    scikit-learn; a 9.97e36 one falls back at fit instead of predicting from zeros."""
    X = _with_a_fill_row(rng, 1e20, n=3000, d=8).astype(np.float64)
    y = X[:, 0] + rng.normal(scale=0.1, size=len(X))
    Q = rng.normal(size=(200, 8))
    expected = SkKNeighborsRegressor(n_neighbors=5).fit(X, y).predict(Q)
    no_handoff = {
        "knn_min_train_samples": 0,
        "knn_min_work": 0,
        "knn_min_features": 0,
        "knn_narrow_min_samples": 0,
        "knn_probe_max_uncertified": 1.0,
        "knn_wide_min_samples": 0,
    }

    with tuning_context(**no_handoff):
        served = patched.KNeighborsRegressor(n_neighbors=5).fit(X, y)
        assert served.execution_backend_ in ("mlx", "cpu")
        np.testing.assert_allclose(served.predict(Q), expected, rtol=1e-6, atol=1e-6)

        X[123] = 9.97e36
        with pytest.warns(MLXLearnFallbackWarning, match="resolve"):
            fallen = patched.KNeighborsRegressor(n_neighbors=5).fit(X, y)
    assert fallen.execution_backend_ == "sklearn"
    assert fallen.fit_event_.reason == "float32-overflow"
    np.testing.assert_array_equal(
        fallen.predict(Q), SkKNeighborsRegressor(n_neighbors=5).fit(X, y).predict(Q)
    )


def _timestamps(rng, n):
    """Unix timestamps inside one hour, beside a column in seconds."""
    return np.column_stack([1.7e9 + rng.uniform(0, 3600, n), rng.normal(scale=60, size=n)])


@pytest.mark.parametrize("backend", BACKENDS)
def test_float64_offsets_survive_the_float32_cast(backend, rng):
    """float32 spaces values near 1.7e9 128 apart.

    Cast first, 5000 such timestamps collapsed to 29 distinct values and no row had
    its true neighbors. The index keeps float64 input as a float32 value plus a
    float32 residual and translates it by its float64 mean, so the answer is
    relative to the data the caller supplied — to ~2⁻⁴⁸ of the coordinate, which is
    ~6e-6 here.
    """
    X = _timestamps(rng, 5000)
    Q = _timestamps(rng, 60)

    dist, idx = build_index(X, backend=backend).query(Q, 5)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=0, atol=2e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_float64_offset_does_not_cost_the_first_pass(backend, rng):
    """Timestamps a few seconds apart.

    The float64 mean, rounded to float32, is up to 64 from the true one near 1.7e9,
    and that constant offset in every translated row swamped the certificate's
    noise band: every query took the wider search, at 4-5x the time. The center is
    now carried as a float32 pair, like the rows. Queries with no residual — float32
    ones — have to be translated into the same frame; certified against a
    different one, they come back with wrong neighbors and nothing raised.
    """

    def window(m):
        return np.column_stack([1.7e9 + 61.3 + rng.uniform(0, 5, m), rng.uniform(0, 5, m)])

    X, Q = window(20_000), window(200)
    index = build_index(X, backend=backend)

    stats = {}
    dist, _ = index.query(Q, 5, stats=stats)
    assert stats["widened_rows"] == 0
    np.testing.assert_allclose(dist, _exact_neighbors(X, Q, 5)[0], rtol=0, atol=2e-5)

    Q32 = Q.astype(np.float32)
    dist, _ = index.query(Q32, 5)
    np.testing.assert_allclose(dist, _exact_neighbors(X, Q32, 5)[0], rtol=0, atol=2e-5)


def test_the_estimator_passes_float64_through(rng):
    """The same data through the public estimator, which must not cast it first."""
    X = _timestamps(rng, 5000)
    Q = _timestamps(rng, 60)

    model = NearestNeighbors(n_neighbors=5).fit(X)
    dist, idx = model.kneighbors(Q)
    ref_dist, ref_idx = _exact_neighbors(X, Q, 5)

    assert model.execution_backend_ in ("mlx", "cpu")
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=0, atol=2e-5)


def test_a_query_far_outside_the_training_range_is_rejected(rng):
    """Even after scaling float32 cannot square this, and inference cannot fall back."""
    X = rng.normal(size=(500, 4)).astype(np.float32)
    index = build_index(X, backend="cpu")
    with pytest.raises(UnsupportedInputError, match="float32"):
        index.query(np.full((1, 4), 1e30, dtype=np.float32), 3)
