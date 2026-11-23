"""The radius benchmark harness: grids, data, variants, arguments, analysis, one micro run.

Fast tests only. The end-to-end test runs the micro grid (n=2000, nq=50) in a
subprocess against the mlxlearn this test process imports, with a control child on
the same tree.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from benchmarks.radius import analyze, radius_calibrate
from benchmarks.radius import bench_radius as B

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "radius"

#: (configs, sha256 of the newline-joined config keys) of the review driver's grids.
REVIEW_GRIDS = {
    "smoke": (2, "971a6d509a15e578843db0a32cd740b80975f677415a8c4dd4781805a5802756"),
    "core": (161, "5935d33c1a50f40a469b60b7fb7d49536115f4951ef83f8c3b37a1e66277e003"),
    "real": (24, "cb1fa8ed15e241141046e432bede2c2ba73bddc35663679da830a8f43d83adb3"),
    "ops": (44, "2590f1874b0b408f1f941f6719d3c10e99d47302e003d72593f98fab26a6ac5c"),
}


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(B.GRIDS))
def test_grid_configs_are_well_formed(name):
    cfgs = B.GRIDS[name]()
    assert cfgs
    keys = [B.config_key(c) for c in cfgs]
    assert len(set(keys)) == len(keys), "duplicate configs in a grid"
    for cfg in cfgs:
        assert {"data", "n", "nq", "m", "op"} <= set(cfg)
        assert cfg["data"] in B.DATA_KINDS and cfg["op"] in B.OPS
        assert cfg["n"] > 0 and cfg["m"] >= 1 and cfg["nq"] >= 0
        assert (cfg["nq"] == 0) == (cfg["op"] == "self")
        assert ("d" in cfg) == (cfg["data"] not in B.REAL_DATA)
        if "weights" in cfg:
            assert cfg["op"] in ("clf", "reg") and cfg["weights"] in ("uniform", "distance")
        assert B.canonical_key(cfg) == analyze.canonical_key(cfg)


@pytest.mark.parametrize("name", sorted(REVIEW_GRIDS))
def test_review_grids_keep_their_configs_and_keys(name):
    count, digest = REVIEW_GRIDS[name]
    keys = [B.config_key(c) for c in B.GRIDS[name]()]
    assert len(keys) == count
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == digest


def test_extra_grid_covers_the_planned_cells():
    cfgs = B.GRIDS["extra"]()
    assert len(cfgs) == 85
    assert {c.get("offset") for c in cfgs if c["data"] == "offset"} == {1_000, 1_000_000}
    assert any(c.get("order") == "label" for c in cfgs)
    assert {c["d"] for c in cfgs if c["data"] == "hubs"} == {3, 16, 64}
    selfq = {(c["n"], c["d"]) for c in cfgs if c["op"] == "self"}
    assert {n for n, _ in selfq} == {50_000, 1_000_000} and {d for _, d in selfq} == {3, 8, 16}
    latency = {(c["nq"], c["d"], c["n"]) for c in cfgs if c["op"] == "rn" and c["nq"] <= 64}
    assert latency == {(q, d, n) for q in (1, 16, 64) for d in (3, 16, 64)
                       for n in (10_000, 100_000)}
    assert sum(c["op"] == "graph_conn" for c in cfgs) == 6
    distance = [c for c in cfgs if c.get("weights") == "distance"]
    assert {c["op"] for c in distance} == {"clf", "reg"}
    assert any(c.get("queries") == "train" for c in distance)
    assert "mlx_patched_nj" in [v.name for v in B.select_variants(16, "extra")]
    assert "mlx_patched_nj" not in [v.name for v in B.select_variants(16, "core")]


def test_canonical_key_ignores_the_implied_dimension_of_real_data():
    cfg = {"data": "mnist", "n": 50_000, "nq": 1_000, "m": 10, "op": "rn"}
    assert B.canonical_key({**cfg, "d": 784}) == B.canonical_key(cfg) == B.config_key(cfg)
    synthetic = {"data": "gauss", "n": 10, "d": 3, "nq": 2, "m": 1, "op": "rn"}
    assert B.canonical_key(synthetic) == B.config_key(synthetic)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def review_make_data(cfg, seed=0):
    """The review driver's generator for the synthetic kinds, verbatim."""
    kind, n, d, nq = cfg["data"], cfg["n"], cfg.get("d"), cfg["nq"]
    rng = np.random.default_rng(seed)
    total = n + nq
    labels = None
    if kind == "gauss":
        A = rng.standard_normal((total, d), dtype=np.float32)
    elif kind == "uniform":
        A = rng.random((total, d), dtype=np.float32)
    elif kind == "blobs":
        k = 24
        C = (rng.standard_normal((k, d)) * 6.0).astype(np.float32)
        labels = rng.integers(0, k, total)
        scale = rng.uniform(0.3, 1.2, k).astype(np.float32)
        A = C[labels] + rng.standard_normal((total, d), dtype=np.float32) * scale[labels, None]
    elif kind == "offset":
        A = rng.standard_normal((total, d), dtype=np.float32) + np.float32(1e3)
    A = np.ascontiguousarray(A, dtype=np.float32)
    d = A.shape[1]
    X, Q = A[:n].astype(np.float64), A[n:].astype(np.float64)
    if labels is None:
        z = A[:, 0] + 0.5 * A[:, min(1, d - 1)]
        labels = np.digitize(z, np.quantile(z, [0.33, 0.66]))
    y_clf = (np.asarray(labels) % 5)[:n]
    y_reg = (A[:n, 0].astype(np.float64) + np.sin(A[:n, min(1, d - 1)].astype(np.float64)))
    return X, Q, y_clf, y_reg


@pytest.mark.parametrize("cfg", [
    {"data": "gauss", "n": 20000, "d": 3, "nq": 1000, "m": 10, "op": "rn"},
    {"data": "uniform", "n": 100_000, "d": 2, "nq": 1_000, "m": 10, "op": "rn"},
    {"data": "blobs", "n": 100_000, "d": 8, "nq": 1_000, "m": 10, "op": "rn"},
    {"data": "blobs", "n": 50_000, "d": 3, "nq": 0, "m": 10, "op": "self"},
    {"data": "blobs", "n": 20_000, "d": 16, "nq": 2_000, "m": 10, "op": "clf",
     "weights": "uniform"},
    {"data": "offset", "n": 2000, "d": 4, "nq": 50, "m": 5, "op": "rn"},
])
def test_make_data_reproduces_the_review_driver(cfg):
    for new, old in zip(B.make_data(dict(cfg)), review_make_data(dict(cfg)), strict=True):
        np.testing.assert_array_equal(new, old)


def test_new_data_options():
    base = {"data": "blobs", "n": 4000, "d": 5, "nq": 100, "m": 5, "op": "rn"}
    X, Q, _, _ = B.make_data(dict(base))
    Xs, Qs, _, _ = B.make_data({**base, "order": "label"})
    np.testing.assert_array_equal(Q, Qs)
    np.testing.assert_array_equal(np.sort(X, axis=0), np.sort(Xs, axis=0))
    assert not np.array_equal(X, Xs)
    Xt, Qt, _, _ = B.make_data({**base, "queries": "train"})
    np.testing.assert_array_equal(X, Xt)
    np.testing.assert_array_equal(Qt, X[np.arange(100) * 40])
    Xo, _, _, _ = B.make_data({**base, "data": "offset", "offset": 1_000_000})
    assert abs(Xo.mean() - 1e6) < 1
    np.testing.assert_array_equal(Xo, Xo.astype(np.float32))
    Xh, Qh, _, _ = B.make_data({**base, "data": "hubs"})
    Xh2, _, _, _ = B.make_data({**base, "data": "hubs"})
    np.testing.assert_array_equal(Xh, Xh2)
    assert Xh.shape == (4000, 5) and Qh.shape == (100, 5)
    fp = B.data_fingerprint(X, Q, *B.make_data(dict(base))[2:])
    assert fp == B.data_fingerprint(*B.make_data(dict(base)))
    assert fp != B.data_fingerprint(Xs, Qs, *B.make_data(dict(base))[2:])


# ---------------------------------------------------------------------------
# variants, routes and arguments
# ---------------------------------------------------------------------------

def test_variant_selection_and_user_stock():
    names = [v.name for v in B.select_variants(8)]
    assert names == list(B.BASE_VARIANTS)
    assert [v.name for v in B.select_variants(16)] == ["sk_auto", "sk_brute", "mlx",
                                                       "mlx_patched"]
    plus = [v.name for v in B.select_variants(16, requested=["+mlx_patched_nj"])]
    assert plus[-1] == "mlx_patched_nj"
    exact = [v.name for v in B.select_variants(8, requested=["mlx", "sk_auto"])]
    assert exact == ["sk_auto", "mlx"]
    with pytest.raises(ValueError):
        B.select_variants(8, requested=["mlx", "+mlx_patched_nj"])
    with pytest.raises(ValueError):
        B.select_variants(8, requested=["nope"])
    routes = B.select_variants(8, routes=["tree", "stock"])
    assert [(v.name, v.family, v.route) for v in routes[-2:]] == [
        ("mlx_route=tree", "mlx", "tree"), ("mlx_route=stock", "patched", "stock")]
    present = {"sk_auto", "sk_brute", "sk_auto_nj"}
    assert B.user_stock("mlx_patched", present, 8) == "sk_auto"
    assert B.user_stock("mlx_patched_nj", present, 8) == "sk_auto_nj"
    assert B.user_stock("ctl_mlx_patched_nj", {"sk_auto"}, 64) == "sk_auto"
    assert B.user_stock("mlx_patched_nj", {"sk_auto"}, 8) is None
    assert B.user_stock("mlx_patched_brute", present, 8) == "sk_brute"
    assert B.user_stock("mlx", present, 8) is None


def test_parse_args_defaults_and_options():
    args = B.parse_args(["smoke", "--out", "x.jsonl"])
    assert (args.gap, args.pause, args.hot_primers, args.cold, args.warm) == (
        "gapped", 0.3, 1, 2, 5)
    assert args.routes is None and args.control is None and args.label == "run"
    assert not args.no_drain and B.parse_args(["smoke", "--out", "x", "--no-drain"]).no_drain
    args = B.parse_args(["extra", "--out", "x", "--gap", "hot", "--hot-primers", "2",
                         "--routes", "tree,metal", "--control", "/tmp/old/src", "--label", "m1",
                         "--variants", "+mlx_patched_brute", "--thermal-max", "9"])
    assert (args.gap, args.hot_primers, args.routes, args.control, args.label) == (
        "hot", 2, "tree,metal", "/tmp/old/src", "m1")
    assert args.variants == ["+mlx_patched_brute"] and args.thermal_max == 9
    assert B.parse_args(["core", "--list"]).list
    assert B.parse_args(["--serve"]).serve
    for bad in (["smoke"], ["nope", "--out", "x"], ["smoke", "--out", "x", "--gap", "warm"],
                ["smoke", "--out", "x", "--variants", "bogus"], ["--out", "x"]):
        with pytest.raises(SystemExit):
            B.parse_args(bad)


def test_route_knob_and_route_resolution(monkeypatch):
    from mlxlearn._common import config

    @dataclasses.dataclass(frozen=True)
    class WithKnob:
        radius_route: Literal["auto", "tree", "brute", "metal", "stock"] = "auto"

    @dataclasses.dataclass(frozen=True)
    class WithoutKnob:
        force_backend: str = ""

    monkeypatch.setattr(config, "Tuning", WithKnob)
    allowed = B.route_knob()
    assert allowed == ["auto", "tree", "brute", "metal", "stock"]
    assert B.resolve_routes("all", allowed) == ["tree", "brute", "metal", "stock"]
    assert B.resolve_routes("tree,gpu_direct", allowed) == ["tree"]
    assert B.resolve_routes(None, allowed) == []
    monkeypatch.setattr(config, "Tuning", WithoutKnob)
    assert B.route_knob() is None
    assert B.resolve_routes("all", None) == []


def test_capture_route_prefers_last_plan_and_never_reports_a_stale_one():
    search = SimpleNamespace(last_plan=SimpleNamespace(route="tree", reason="d<=3", threads=4))
    est = SimpleNamespace(_index_=SimpleNamespace(_radius_search=search))
    routes, plan = B.capture_route(est)
    assert routes == ["tree"] and plan == {"route": "tree", "reason": "d<=3", "threads": 4}
    B.clear_plan(est)
    assert search.last_plan is None
    B.ROUTES[:] = ["cpu"]
    try:
        assert B.capture_route(est) == (["cpu"], None)
        assert B.capture_route(SimpleNamespace()) == (["cpu"], None)
    finally:
        B.ROUTES.clear()


# ---------------------------------------------------------------------------
# correctness checks
# ---------------------------------------------------------------------------

def _ragged(rows, dists=None):
    ind = np.empty(len(rows), dtype=object)
    dist = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        ind[i] = np.asarray(r, dtype=np.int64)
        dist[i] = np.asarray(dists[i] if dists else np.zeros(len(r)), dtype=np.float64)
    return dist, ind


def test_compare_membership_and_digests():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    Q = rng.standard_normal((4, 3))
    radius = 1.0
    rows = [np.flatnonzero(((X - q) ** 2).sum(1) <= radius**2) for q in Q]
    exact = [np.sqrt(((X[r] - q) ** 2).sum(1)) for r, q in zip(rows, Q, strict=True)]
    ref = _ragged(rows, exact)
    shuffled = _ragged([r[::-1] for r in rows], [e[::-1] for e in exact])
    check = B.compare("rn", shuffled, ref, X, Q, radius)
    assert check["bad_rows"] == 0 and check["off_boundary"] == 0
    assert check["dist_rel_err"] < 1e-12
    assert B.membership_digest(B.membership("rn", shuffled, 50)) == \
        B.membership_digest(B.membership("rn", ref, 50))
    broken = [r.copy() for r in rows]
    far = int(np.argmax(((X - Q[1]) ** 2).sum(1)))
    broken[1] = np.append(broken[1], far)
    check = B.compare("rn", _ragged(broken), ref, X, Q, radius)
    assert check["bad_rows"] == 1 and check["off_boundary"] == 1
    assert B.membership_digest(B.membership("rn", _ragged(broken), 50)) != \
        B.membership_digest(B.membership("rn", ref, 50))
    unsorted = _ragged(list(rows), [e[::-1] if len(e) > 1 else e for e in exact])
    assert B.compare("rns", unsorted, ref, X, Q, radius)["unsorted_rows"] >= 1
    assert B.compare("clf", np.array([1, 2, 3]), np.array([1, 2, 4]), X, Q, radius) == {
        "disagree": 1}


def test_compare_graphs():
    from scipy import sparse

    a = sparse.csr_matrix((np.ones(3), np.array([2, 0, 1]), np.array([0, 2, 3])), shape=(2, 4))
    b = sparse.csr_matrix((np.ones(3), np.array([0, 2, 1]), np.array([0, 2, 3])), shape=(2, 4))
    check = B.compare("graph_conn", a, b, np.zeros((4, 1)), np.zeros((2, 1)), 1.0)
    assert check["same_pattern"] and check["nnz_diff"] == 0 and check["max_abs_err"] == 0
    payload = B.check_payload("graph_conn", a, 4)
    assert payload["pairs"] == 3
    assert payload["digest"] == B.membership_digest(B.membership("graph_conn", b, 4))


# ---------------------------------------------------------------------------
# calibration framework
# ---------------------------------------------------------------------------

def test_protocols_and_interleaved_runner():
    rc = radius_calibrate
    assert rc.parse_protocol("hot") == rc.Protocol("hot", 0.0, 1)
    assert rc.parse_protocol("hot:3").primer_count() == 3
    assert rc.parse_protocol("gapped:0.002").idle_before() == 0.002
    with pytest.raises(ValueError):
        rc.parse_protocol("warm")
    assert rc.rotation("abc", 1) == ["a", "c", "b"]
    assert rc.rotation("abc", 2, reverse_odd=False) == ["c", "a", "b"]
    calls = {"a": 0, "b": 0}
    order = []

    def case(name):
        def setup():
            def fn():
                calls[name] += 1
                order.append(name)
            return fn
        return rc.Case(name, setup, work=10)

    results = rc.run_interleaved([case("a"), case("b")], rc.Protocol("hot", 0.0, 2, drain=False),
                                 rounds=3, warmup_rounds=1)
    assert calls == {"a": 12, "b": 12}  # (1 primer pair + 1 timed) x 4 rounds, 2 primers
    assert [r["summary"]["n"] for r in results] == [3, 3]
    assert all(r["summary"]["per_unit_ns"] > 0 for r in results)
    results = rc.run_interleaved([case("a")], rc.Protocol("gapped", 0.001, 0, drain=False),
                                 rounds=2, warmup_rounds=0)
    assert results[0]["summary"]["n"] == 2 and calls["a"] == 14
    a, b = rc.fit_affine([1, 10, 100, 1000], [1.1, 2.0, 11.0, 101.0])
    assert a == pytest.approx(1.0) and b == pytest.approx(0.1)


def test_drain_waits_while_this_process_burns_cpu():
    import threading

    rc = radius_calibrate
    stop = time.monotonic() + 0.15

    def spin():
        while time.monotonic() < stop:
            pass

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        waited = rc.drain_idle(max_s=1.0)
    finally:
        worker.join()
    assert 0.1 <= waited < 1.0
    assert rc.drain_idle(max_s=1.0) < 0.5


# ---------------------------------------------------------------------------
# analyze.py on a synthetic JSONL fixture
# ---------------------------------------------------------------------------

def _summary(times):
    return {name: {"warm_med": t, "warm_min": t * 0.9, "cold_med": 2 * t, "fit_med": t}
            for name, t in times.items()}


def _record(cfg, d, times, *, label="fx", routes=None, **extra):
    rec = {"label": label, "cfg": cfg, "d": d, "radius": 0.5, "variants": list(times),
           "summary": _summary(times), "ref_summary": {"pairs": 1000},
           "routes": routes or {}, "check": {n: {"bad_rows": 0} for n in times
                                             if n != "sk_brute"}}
    rec.update(extra)
    return rec


@pytest.fixture
def fixture_jsonl(tmp_path):
    low = {"data": "uniform", "n": 1000, "d": 3, "nq": 100, "m": 10, "op": "rn"}
    mid = {"data": "gauss", "n": 1000, "d": 12, "nq": 100, "m": 10, "op": "rn"}
    high = {"data": "gauss", "n": 1000, "d": 64, "nq": 100, "m": 10, "op": "rn"}
    ops = {"data": "blobs", "n": 1000, "d": 3, "nq": 100, "m": 10, "op": "clf",
           "weights": "distance"}
    real = {"data": "mnist", "n": 1000, "nq": 100, "m": 10, "op": "rn"}
    records = [
        # superseded by the later record with the same key
        _record(low, 3, {"sk_auto": 1.0, "sk_brute": 1.0, "mlx": 100.0}),
        _record(low, 3, {"sk_auto": 2.0, "sk_brute": 4.0, "sk_auto_nj": 1.0, "mlx": 1.0,
                         "mlx_patched": 2.0, "mlx_patched_nj": 0.5, "ctl_mlx": 2.0,
                         "ctl_mlx_patched": 2.0, "mlx_route=tree": 0.8,
                         "mlx_route=metal": 3.0, "mlx_route=stock": 2.1},
                routes={"mlx:warm": ["tree"], "mlx_route=tree:warm": ["tree"],
                        "mlx_route=metal:warm": ["metal"], "mlx_route=stock:warm": ["stock"],
                        "mlx_patched:warm": ["tree"]},
                plans={"mlx:warm": {"route": "tree", "reason": "d<=3"}},
                user_stock={"mlx_patched": "sk_auto", "mlx_patched_nj": "sk_auto_nj",
                            "ctl_mlx_patched": "sk_auto"}),
        _record(mid, 12, {"sk_auto": 1.0, "sk_brute": 1.0, "mlx": 2.0, "mlx_patched": 1.0,
                          "ctl_mlx": 1.0}),
        # d > 15: the n_jobs user's stock falls back to sk_auto (no user_stock field)
        _record(high, 64, {"sk_auto": 3.0, "sk_brute": 3.0, "mlx": 1.0, "mlx_patched": 3.0,
                           "mlx_patched_nj": 6.0},
                variant_errors={"mlx_route=metal": "ValueError('infeasible')"}),
        _record(ops, 3, {"sk_auto": 4.0, "sk_brute": 4.0, "mlx": 2.0, "mlx_patched": 2.0}),
        _record({**real, "d": 784}, 784, {"sk_auto": 1.0, "sk_brute": 2.0, "mlx": 4.0,
                                           "mlx_patched": 1.1}),
        {"label": "fx", "cfg": {"data": "gauss", "n": 5, "d": 3, "nq": 1, "m": 1, "op": "rn"},
         "error": "RuntimeError('boom')"},
    ]
    path = tmp_path / "fixture.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_analyze_cells_regions_and_gates(fixture_jsonl):
    rows, errors = analyze.load([fixture_jsonl])
    assert len(rows) == 5 and len(errors) == 1
    cells = {c.key.split(" ")[0] + f" d={c.rec['d']}": c
             for c in (analyze.build_cell(r) for r in rows)}
    low = cells["uniform/rn d=3"]
    assert low.region == "low-d (<=8)"
    assert low.values["mlx vs sk_auto"] == 2.0 and low.values["mlx vs sk_brute"] == 4.0
    assert low.values["mlx vs sk_auto_nj"] == 1.0 and low.values["mlx vs sk_best"] == 1.0
    assert low.values["mlx_patched vs user stock"] == 1.0
    assert low.values["mlx_patched_nj vs user stock"] == 2.0
    assert low.values["mlx: new vs ctl"] == 2.0
    assert low.values["mlx_patched: new vs ctl"] == 1.0
    assert low.values["regret mlx"] == pytest.approx(1.0 / 0.8)
    assert low.values["regret mlx_patched"] == pytest.approx(2.0 / 0.8)
    high = cells["gauss/rn d=64"]
    assert high.region == "high-d (>=16)" and high.stocks["mlx_patched_nj"] == "sk_auto"
    assert high.values["mlx_patched_nj vs user stock"] == 0.5
    assert cells["gauss/rn d=12"].region == "mid-d (9-15)"
    assert cells["blobs/clf/distance d=3"].region == "ops"
    assert cells["mnist/rn d=784"].region == "real"

    agg = analyze.aggregate(list(cells.values()))
    allx = agg["mlx vs sk_auto"]["all"]
    assert allx["n"] == 5 and allx["wins"] == 3
    assert allx["geo"] == pytest.approx((2.0 * 0.5 * 3.0 * 2.0 * 0.25) ** (1 / 5))
    assert agg["mlx vs sk_auto"]["low-d (<=8)"]["n"] == 1
    assert agg["mlx_patched vs user stock"]["all"]["pass"] == 4
    assert agg["mlx: new vs ctl"]["all"] == pytest.approx(
        {"n": 2, "geo": 1.0, "min": 0.5, "max": 2.0, "wins": 1, "pass": 1})
    bars = {b: (k, n) for b, k, n in analyze.acceptance(list(cells.values()))}
    assert bars["Layer 1 >= 1.0x both sk_auto and sk_brute"] == (3, 5)
    assert bars["mlx_patched_nj >= 0.95x user stock"] == (1, 2)
    assert bars["regret mlx <= 1.25"] == (1, 1)
    honored = analyze.forced_route_honored(list(cells.values()))
    assert ("mlx_route=metal", 1, 1) in honored


def test_paired_metric_uses_per_round_ratios():
    cfg = {"data": "gauss", "n": 1000, "d": 3, "nq": 100, "m": 10, "op": "rn"}
    # Rounds 3 and 4 are a slow phase for everything; mlx has no round-2 sample.
    warm = {"sk_auto": [2.0, 2.0, 6.0, 6.0], "mlx": [1.0, 3.0, 3.0],
            "ctl_mlx": [1.5, 1.5, 4.5, 4.5]}
    rounds = {"sk_auto": [1, 2, 3, 4], "mlx": [1, 3, 4], "ctl_mlx": [1, 2, 3, 4]}
    rec = _record(cfg, 3, {n: float(np.median(v)) for n, v in warm.items()}, warm=warm,
                  warm_round=rounds)
    paired = analyze.build_cell(rec, "paired").values
    assert paired["mlx vs sk_auto"] == 2.0  # rounds 1, 3, 4: 2/1, 6/3, 6/3
    assert paired["mlx: new vs ctl"] == 1.5
    medians = analyze.build_cell(rec).values  # ratio of medians mixes the phases
    assert medians["mlx vs sk_auto"] == pytest.approx(4.0 / 3.0)
    assert medians["mlx: new vs ctl"] == 1.0
    del rec["warm_round"]  # older records: samples are one per timed round, in order
    assert analyze.build_cell(rec, "paired").values["mlx vs sk_auto"] == pytest.approx(
        statistics.median([2.0 / 1.0, 2.0 / 3.0, 6.0 / 3.0]))


def test_analyze_main_writes_text_and_markdown(fixture_jsonl, tmp_path, capsys):
    md = tmp_path / "summary.md"
    assert analyze.main([str(fixture_jsonl), "--md", str(md), "--metric", "e2e"]) == 0
    text = capsys.readouterr().out
    assert "5 configs, 1 errors" in text and "Acceptance bars" in text
    assert "RuntimeError('boom')" in text and "infeasible" in text
    body = md.read_text()
    assert body.startswith("# Radius benchmark summary (e2e")
    assert "| comparison | region | n | wins | gate | pass | geomean | min | max |" in body
    assert "| mlx_route=metal | 1 | 1 |" in body
    assert analyze.main([str(fixture_jsonl), "--label", "other"]) == 0
    assert "0 configs" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# one tiny end-to-end run
# ---------------------------------------------------------------------------

def test_micro_grid_end_to_end_with_control(tmp_path):
    pytest.importorskip("mlx.core")
    import mlxlearn

    src = Path(mlxlearn.__file__).resolve().parents[1]
    out = tmp_path / "micro.jsonl"
    env = {k: v for k, v in os.environ.items() if k != "MLXLEARN_RADIUS_ROUTE"}
    env["PYTHONPATH"] = str(src)
    cmd = [sys.executable, str(BENCH / "bench_radius.py"), "micro", "--out", str(out),
           "--cold", "1", "--warm", "1", "--pause", "0", "--thermal-max", "9",
           "--label", "test", "--control", str(src)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["cfg"]["d"] for r in records] == [3, 32]
    for rec in records:
        assert "error" not in rec, rec.get("trace")
        assert not rec["variant_errors"]
        assert {"sk_auto", "sk_brute", "mlx", "mlx_patched", "ctl_mlx",
                "ctl_mlx_patched"} <= set(rec["variants"])
        assert rec["meta"]["mlxlearn_file"].startswith(str(src))
        assert rec["control"]["src"] == str(src)
        for name in ("mlx", "mlx_patched", "sk_auto"):
            assert rec["check"][name]["off_boundary"] == 0
        assert rec["check"]["ctl_mlx"]["same_membership_as_mlx"] is True
        assert rec["routes"]["mlx:warm"]
        assert rec["user_stock"]["mlx_patched"] == "sk_auto"
        assert all(rec["summary"][n]["warm_med"] > 0 for n in rec["variants"])
        assert rec["protocol"] == {"gap": "gapped", "pause": 0.0, "hot_primers": 0, "cold": 1,
                                   "warm": 1, "drain": True, "rounds": 1}
        assert rec["drain_s"]["parent"] > 0
    # resuming skips both configs
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and len(out.read_text().splitlines()) == 2
    assert analyze.main([str(out), "--md", str(tmp_path / "micro.md")]) == 0
