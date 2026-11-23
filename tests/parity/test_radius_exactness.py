"""Route exactness: every radius route against the exact oracle.

The contract (docs/radius-certificate.md) is checked on every generator in
``tests/_radius_oracle.py``, for each route forced through
``tuning_context(radius_route=...)``:

* (M) membership equals the oracle bit for bit: ``S(q, x) <= fl64(r * r)`` on
  the float32 coordinates, ``X=None`` excluding only ``j == i``;
* (O) rows ascend by training index with ``sort_results=False`` and by
  ``(distance, index)`` with ``sort_results=True``;
* (D) distances are float64, within ``1e-7`` relative of ``sqrt(S)`` (plus the
  documented ``d * 2**-90`` absolute allowance of the Gram/DS route on DS-safe
  rows), never above the radius, and zero exactly for identical points;
* (R) radius classifier and regressor predictions match a direct
  reimplementation on the oracle's neighbor sets.

The route tests skip until ``Tuning.radius_route`` exists. The host-engine
tests at the end exercise ``_kernels/radius_host`` directly and always run.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from mlxlearn._common.config import Tuning, tuning_context
from mlxlearn.neighbors import (
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from tests import _radius_oracle as oracle

HAS_ROUTE_KNOB = "radius_route" in {f.name for f in dataclasses.fields(Tuning)}
needs_route_knob = pytest.mark.skipif(
    not HAS_ROUTE_KNOB, reason="Tuning.radius_route does not exist yet (route override lands in WI 1.1)"
)

ROUTES = ("tree", "brute", pytest.param("metal", marks=pytest.mark.gpu), "auto")
#: Plan names a forced route may report (the interim and final GPU engines).
ROUTE_NAMES = {
    "tree": {"tree"},
    "brute": {"brute"},
    "metal": {"metal", "metal_v1", "gpu_direct", "gpu_gram"},
}
CASES = oracle.all_cases()
LABELED = [case for case in CASES if case.id in ("label-sorted", "blobs-d8")]
DS_MIN, DS_MAX = 2.0**-30, 2.0**30

_GPU: bool | None = None


def _require(route: str) -> None:
    global _GPU
    if route != "metal":
        return
    if _GPU is None:
        from mlxlearn._common.device import gpu_available

        _GPU = gpu_available()
    if not _GPU:
        pytest.skip("the metal route needs a working Metal GPU")


# ----------------------------------------------------------------------------------
# expected results, computed once per case
# ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Expected:
    indptr: np.ndarray
    indices: np.ndarray
    rows: np.ndarray
    squared: np.ndarray  # correctly rounded float64 S of each member
    atol: np.ndarray  # per-member absolute distance allowance


_EXPECTED: dict[tuple[str, bool], Expected] = {}


def _ds_safe(A32: np.ndarray) -> np.ndarray:
    magnitude = np.abs(A32)
    return np.all((magnitude == 0) | ((magnitude >= DS_MIN) & (magnitude <= DS_MAX)), axis=1)


def expected(case: oracle.RadiusCase, *, self_query: bool) -> Expected:
    key = (case.id, self_query)
    if key not in _EXPECTED:
        Q = None if self_query else case.Q
        indptr, indices = oracle.oracle_csr(case.X, Q, case.radius, exclude_self=self_query)
        rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        squared = oracle.oracle_distances(case.X, Q, rows, indices)
        X32 = oracle.to_float32(case.X)
        Q32 = X32 if self_query else oracle.to_float32(case.Q)
        both = _ds_safe(Q32)[rows] & _ds_safe(X32)[indices]
        atol = np.where(both, math.sqrt(X32.shape[1] * 2.0**-90) * 1.001, 0.0)
        _EXPECTED[key] = Expected(indptr, indices, rows, squared, atol)
    return _EXPECTED[key]


# ----------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------


def _flatten(rows_obj, dtype_kind: str, what: str) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(rows_obj, np.ndarray) and rows_obj.dtype == object and rows_obj.ndim == 1, what
    lengths = np.fromiter(map(len, rows_obj), dtype=np.int64, count=len(rows_obj))
    for row in rows_obj:
        assert isinstance(row, np.ndarray) and row.ndim == 1 and row.dtype.kind == dtype_kind, what
        if dtype_kind == "f":
            assert row.dtype == np.float64, what
    flat = (np.concatenate(list(rows_obj)) if len(rows_obj) and lengths.sum()
            else np.empty(0, dtype=np.float64 if dtype_kind == "f" else np.int64))
    return lengths, flat


def _explain(case, exp: Expected, row: int, got_cols, *, self_query: bool) -> str:
    want = set(exp.indices[exp.indptr[row]:exp.indptr[row + 1]].tolist())
    got = set(np.asarray(got_cols).tolist())
    R2 = oracle.squared_radius(case.radius)
    Q = None if self_query else case.Q
    lines = [f"{case.id}: row {row} differs from the oracle (R2={R2!r})"]
    if want == got:
        lines.append(f"  same set, wrong order: {np.asarray(got_cols)[:10].tolist()}")
    for label, cols in (("missing", sorted(want - got)[:5]), ("extra", sorted(got - want)[:5])):
        for col in cols:
            S = oracle.oracle_distances(case.X, Q, [row], [col], kind="fraction")[0]
            gap = float(S - Fraction(R2)) if math.isfinite(R2) else -math.inf
            lines.append(f"  {label} col {col}: S={float(S)!r}, S - R2 = {gap!r}")
    return "\n".join(lines)


def check_neighbors(case, exp: Expected, ind, dist, *, sort_results: bool, self_query: bool,
                    what: str) -> None:
    """Membership, order and distances of one radius_neighbors result."""
    nq = len(exp.indptr) - 1
    lengths, cols = _flatten(ind, "i", what)
    assert len(ind) == nq, what
    want_lengths = np.diff(exp.indptr)
    if not np.array_equal(lengths, want_lengths) or (
            not sort_results and not np.array_equal(cols, exp.indices)):
        for row in range(nq):
            got_row = ind[row]
            if sort_results:
                got_row = np.sort(got_row)
            if not np.array_equal(got_row, exp.indices[exp.indptr[row]:exp.indptr[row + 1]]):
                raise AssertionError(f"{what}: " + _explain(case, exp, row, got_row,
                                                            self_query=self_query))
    rows = exp.rows
    if sort_results:
        # Membership as sets, then the (distance, index) order.
        order = np.lexsort((cols, rows))
        if not np.array_equal(cols[order], exp.indices):
            for row in range(nq):
                got_row = np.sort(ind[row])
                if not np.array_equal(got_row, exp.indices[exp.indptr[row]:exp.indptr[row + 1]]):
                    raise AssertionError(f"{what}: " + _explain(case, exp, row, got_row,
                                                                self_query=self_query))
    else:
        order = np.arange(cols.size)
        same = rows[1:] == rows[:-1]
        assert np.all(cols[1:][same] > cols[:-1][same]), f"{what}: rows are not in ascending index order"
    if dist is None:
        return
    dist_lengths, values = _flatten(dist, "f", what)
    assert np.array_equal(dist_lengths, lengths), f"{what}: distance rows do not match index rows"
    if sort_results:
        same = rows[1:] == rows[:-1]
        ordered = (values[1:] > values[:-1]) | ((values[1:] == values[:-1]) & (cols[1:] > cols[:-1]))
        assert np.all(ordered[same]), f"{what}: rows are not ordered by (distance, index)"
    aligned = values[order]  # now in the oracle's (row, ascending col) order
    check_distances(case, exp, aligned, what)


def check_distances(case, exp: Expected, values: np.ndarray, what: str) -> None:
    reference = np.sqrt(exp.squared)
    error = np.abs(values - reference)
    tolerance = 1e-7 * reference + exp.atol
    bad = np.flatnonzero(~(error <= tolerance))
    assert bad.size == 0, (
        f"{what}: {bad.size} distances off; first at row {exp.rows[bad[0]]} col "
        f"{exp.indices[bad[0]]}: got {values[bad[0]]!r}, sqrt(S)={reference[bad[0]]!r}")
    radius = float(case.radius)
    assert np.all(values <= radius), f"{what}: a returned distance exceeds the radius"
    assert np.array_equal(values == 0, exp.squared == 0), f"{what}: d == 0 must mean identical points"


def check_plan(estimator, route: str, what: str) -> None:
    search = getattr(getattr(estimator, "_index_", None), "_radius_search", None)
    plan = getattr(search, "last_plan", None)
    if plan is None or route == "auto":
        return
    ran = str(getattr(plan, "route", None))
    # A forced GPU route may merge range-escalated rows from a host pass, and
    # data outside the float32 Gram screen's range escalates to the exact host
    # brute engine as a whole (documented in the radius certificate).
    escalated = route == "metal" and ran == "brute" and "outside the float32" in str(
        getattr(plan, "reason", ""))
    ok = escalated or ran in ROUTE_NAMES[route] or (route == "metal" and (
        ran.startswith(("metal", "gpu")) or ran == "mixed"))
    assert ok, f"{what}: forced route {route!r} ran {ran!r} ({getattr(plan, 'reason', '')})"


# ----------------------------------------------------------------------------------
# routes against the oracle
# ----------------------------------------------------------------------------------


@needs_route_knob
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_radius_neighbors_match_the_oracle(case, route):
    _require(route)
    on_q = expected(case, self_query=False)
    on_x = expected(case, self_query=True)
    with tuning_context(radius_route=route):
        nn = NearestNeighbors(radius=case.radius).fit(case.X)

        dist, ind = nn.radius_neighbors(case.Q)
        check_neighbors(case, on_q, ind, dist, sort_results=False, self_query=False,
                        what=f"{route} radius_neighbors(Q)")
        check_plan(nn, route, "radius_neighbors(Q)")

        dist, ind = nn.radius_neighbors(case.Q, sort_results=True)
        check_neighbors(case, on_q, ind, dist, sort_results=True, self_query=False,
                        what=f"{route} radius_neighbors(Q, sort_results=True)")

        ind = nn.radius_neighbors(case.Q, return_distance=False)
        check_neighbors(case, on_q, ind, None, sort_results=False, self_query=False,
                        what=f"{route} radius_neighbors(Q, return_distance=False)")

        dist, ind = nn.radius_neighbors()
        check_neighbors(case, on_x, ind, dist, sort_results=False, self_query=True,
                        what=f"{route} radius_neighbors()")
        check_plan(nn, route, "radius_neighbors()")

        dist, ind = nn.radius_neighbors(sort_results=True)
        check_neighbors(case, on_x, ind, dist, sort_results=True, self_query=True,
                        what=f"{route} radius_neighbors(sort_results=True)")

        # The radius given at query time instead of construction.
        dist, ind = NearestNeighbors().fit(case.X).radius_neighbors(case.Q, radius=case.radius)
        check_neighbors(case, on_q, ind, dist, sort_results=False, self_query=False,
                        what=f"{route} radius_neighbors(Q, radius=r)")


@needs_route_knob
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_radius_graph_matches_the_oracle(case, route):
    _require(route)
    on_q = expected(case, self_query=False)
    with tuning_context(radius_route=route):
        nn = NearestNeighbors(radius=case.radius).fit(case.X)
        for mode in ("connectivity", "distance"):
            graph = nn.radius_neighbors_graph(case.Q, mode=mode).tocsr()
            graph.sort_indices()
            assert graph.shape == (len(on_q.indptr) - 1, len(oracle.to_float32(case.X)))
            assert np.array_equal(graph.indptr, on_q.indptr), f"{route} {mode}: row counts"
            assert np.array_equal(graph.indices, on_q.indices), f"{route} {mode}: columns"
            if mode == "connectivity":
                assert np.all(graph.data == 1)
            else:
                check_distances(case, on_q, np.asarray(graph.data, dtype=np.float64),
                                f"{route} radius_neighbors_graph(mode='distance')")
        on_x = expected(case, self_query=True)
        graph = nn.radius_neighbors_graph(mode="connectivity").tocsr()
        graph.sort_indices()
        assert np.array_equal(graph.indptr, on_x.indptr) and np.array_equal(graph.indices, on_x.indices)


REPEAT_CASES = [case for case in CASES if case.id in (
    "perm-tie", "perm-mixed-d16-below", "gauss-d16-edge", "gauss-d64-edge", "offset-1e12-tie",
    "lattice-z3-sqrt3", "huge-r6e38", "hubs-d8")]


@needs_route_knob
@pytest.mark.parametrize("case", REPEAT_CASES, ids=lambda c: c.id)
def test_routes_agree_bitwise_and_repeat(case):
    """Membership is identical across routes and runs; host-computed distances
    (tree and brute) are bitwise identical."""
    results = {}
    for route in ("tree", "brute", "metal", "auto"):
        if route == "metal":
            try:
                _require(route)
            except pytest.skip.Exception:
                continue
        with tuning_context(radius_route=route):
            nn = NearestNeighbors(radius=case.radius).fit(case.X)
            runs = [nn.radius_neighbors(case.Q) for _ in range(3)]
        flat = [(np.concatenate(list(d)) if len(d) else np.empty(0),
                 np.concatenate(list(i)) if len(i) else np.empty(0, np.int64)) for d, i in runs]
        for d, i in flat[1:]:
            assert np.array_equal(i, flat[0][1]), f"{route}: membership changed between runs"
            assert np.array_equal(d, flat[0][0]), f"{route}: distances changed between runs"
        results[route] = flat[0]
    first = next(iter(results.values()))[1]
    for route, (_, indices) in results.items():
        assert np.array_equal(indices, first), f"{route} membership differs from another route"
    assert np.array_equal(results["tree"][0], results["brute"][0]), "tree and brute distances differ"


# ----------------------------------------------------------------------------------
# classifier and regressor reductions
# ----------------------------------------------------------------------------------


def _reference_votes(case, exp: Expected, codes, n_classes, weights):
    nq = len(exp.indptr) - 1
    proba = np.zeros((nq, n_classes))
    dist = np.sqrt(exp.squared)
    for i in range(nq):
        lo, hi = exp.indptr[i], exp.indptr[i + 1]
        if lo == hi:
            continue
        if weights == "uniform":
            w = np.ones(hi - lo)
        elif np.any(dist[lo:hi] == 0):
            w = (dist[lo:hi] == 0).astype(np.float64)
        else:
            w = 1.0 / dist[lo:hi]
        votes = np.bincount(codes[exp.indices[lo:hi]], weights=w, minlength=n_classes)
        proba[i] = votes / votes.sum()
    return proba


def _reference_means(case, exp: Expected, target, weights):
    nq = len(exp.indptr) - 1
    out = np.full(nq, np.nan)
    dist = np.sqrt(exp.squared)
    for i in range(nq):
        lo, hi = exp.indptr[i], exp.indptr[i + 1]
        if lo == hi:
            continue
        values = target[exp.indices[lo:hi]]
        if weights == "uniform":
            out[i] = values.mean()
        else:
            d = dist[lo:hi]
            w = (d == 0).astype(np.float64) if np.any(d == 0) else 1.0 / d
            out[i] = np.average(values, weights=w)
    return out


@needs_route_knob
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", LABELED, ids=lambda c: c.id)
def test_classifier_matches_the_oracle(case, route, weights):
    _require(route)
    exp = expected(case, self_query=False)
    classes, codes = np.unique(case.y, return_inverse=True)
    reference = _reference_votes(case, exp, codes, len(classes), weights)
    empty = np.diff(exp.indptr) == 0
    most_frequent = classes[np.bincount(codes).argmax()]
    reference[empty] = (classes == most_frequent).astype(np.float64)
    with tuning_context(radius_route=route):
        clf = RadiusNeighborsClassifier(radius=case.radius, weights=weights,
                                        outlier_label="most_frequent").fit(case.X, case.y)
        proba = clf.predict_proba(case.Q)
        labels = clf.predict(case.Q)
    np.testing.assert_allclose(proba, reference, rtol=1e-9, atol=1e-12)
    expected_labels = classes[reference.argmax(axis=1)]  # ties go to the smallest class
    expected_labels[empty] = most_frequent
    if weights == "uniform":
        decided = np.ones(len(labels), dtype=bool)  # exact counts: ties must match too
    else:
        top_two = np.sort(reference, axis=1)[:, -2:]
        decided = top_two[:, 1] - top_two[:, 0] > 1e-9
    assert np.array_equal(labels[decided], expected_labels[decided])


@needs_route_knob
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", LABELED, ids=lambda c: c.id)
def test_regressor_matches_the_oracle(case, route, weights):
    _require(route)
    exp = expected(case, self_query=False)
    X32 = oracle.to_float32(case.X).astype(np.float64)
    target = X32[:, 0] * 2.0 - X32[:, -1] + case.y
    reference = _reference_means(case, exp, target, weights)
    with tuning_context(radius_route=route):
        reg = RadiusNeighborsRegressor(radius=case.radius, weights=weights).fit(case.X, target)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            predicted = reg.predict(case.Q)
    assert predicted.dtype == np.float64
    assert np.array_equal(np.isnan(predicted), np.isnan(reference))
    np.testing.assert_allclose(predicted, reference, rtol=1e-9, atol=1e-12, equal_nan=True)


# ----------------------------------------------------------------------------------
# host engines (radius_host) against the oracle; independent of the route knob
# ----------------------------------------------------------------------------------


def _csr_neighbors(result):
    """A RadiusCSR as ragged object arrays (indices, distances)."""
    bounds = result.indptr.tolist()
    ind = np.empty(len(bounds) - 1, dtype=object)
    dist = None if result.distances is None else np.empty(len(bounds) - 1, dtype=object)
    for i in range(len(bounds) - 1):
        ind[i] = result.indices[bounds[i]:bounds[i + 1]]
        if dist is not None:
            dist[i] = result.distances[bounds[i]:bounds[i + 1]]
    return ind, dist


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_host_engines_match_the_oracle(case):
    radius_host = pytest.importorskip("mlxlearn._kernels.radius_host")
    from mlxlearn._kernels import radius_contract as rc
    from mlxlearn._kernels.radius_result import sort_rows

    X32 = oracle.to_float32(case.X)
    Q32 = oracle.to_float32(case.Q)
    radius = rc.validate_radius(case.radius)
    R2 = radius * radius
    tree = radius_host.build_tree(X32)
    state = radius_host.BruteState(X32)
    distances = {}
    for self_query in (False, True):
        queries = X32 if self_query else Q32
        thr = rc.thresholds(X32.shape[1], R2, integer_exact=rc.integer_exact(X32, queries))
        exp = expected(case, self_query=self_query)
        for engine in ("tree", "brute"):
            if engine == "tree":
                result = radius_host.tree_query(tree, X32, queries, thr, exclude_self=self_query,
                                                need_distance=True, threads=1)
            else:
                result = radius_host.brute_host(state, X32, queries, thr, exclude_self=self_query,
                                                need_distance=True)
            what = f"radius_host.{engine} ({'X=None' if self_query else 'Q'})"
            ind, dist = _csr_neighbors(result)
            check_neighbors(case, exp, ind, dist, sort_results=False, self_query=self_query,
                            what=what)
            ind, dist = _csr_neighbors(sort_rows(result))
            check_neighbors(case, exp, ind, dist, sort_results=True, self_query=self_query,
                            what=what + " + sort_rows")
            distances[engine, self_query] = result.distances
    for self_query in (False, True):
        assert np.array_equal(distances["tree", self_query], distances["brute", self_query]), (
            "tree and brute host distances must be bitwise identical")
