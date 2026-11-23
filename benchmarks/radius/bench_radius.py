#!/usr/bin/env python
"""Radius-neighbor benchmark: mlxlearn (Layer 1 and patched) against scikit-learn.

Every config fits each variant on the same float64 data. The values are
representable in float32, so mlxlearn's float32 copy is exact. Each config then
times two phases:

* cold: construct, fit, then the first query, in rotated variant order, ``--cold``
  times, with a ``--pause`` idle before every call.
* warm: repeated queries on the fitted estimators. Each round makes one call per
  variant, with the order rotated each round and reversed on odd rounds. Round 0 is
  an untimed warm-up that also runs the correctness checks. ``--gap`` sets how the
  timed calls are spaced:

  - ``gapped`` (the default) pauses ``--pause`` seconds before every call. The GPU
    clocks wind down during the pause, which is what a sporadic caller sees.
  - ``hot`` uses no pause. Each timed call immediately follows ``--hot-primers``
    untimed calls of the same variant, which is what a back-to-back caller sees.
    Variants are still interleaved from slot to slot.

Variants:

* ``sk_auto``, ``sk_brute`` and ``sk_auto_nj`` are stock scikit-learn: default, then
  ``algorithm="brute"``, then ``n_jobs=-1`` (``sk_auto_nj`` only when d <= 15).
* ``mlx`` is Layer 1 (``mlxlearn.neighbors``).
* ``mlx_patched`` is the patched class with a default user configuration.
  ``mlx_patched_nj`` is a user who asked for ``n_jobs=-1`` (on by default in the
  extra grid), and ``mlx_patched_brute`` is a user who asked for
  ``algorithm="brute"``. The record names each patched variant's
  ``user_stock``, the scikit-learn variant that configuration runs.
* ``mlx_route=<r>`` (``--routes``) is Layer 1 with ``radius_route`` forced through
  ``tuning_context``. The patched class is used for ``r="stock"``.
* ``ctl_<variant>`` (``--control SRC``) is the same variant, run by an older
  mlxlearn source tree in a child process and interleaved with the others in the
  same rounds.

Before every call, cold or warm, the driver waits until the previous call's
threads have stopped spinning (``radius_calibrate.drain_idle``; scikit-learn's
OpenMP threads spin about 200 ms after each parallel region). The pause is then
true idle time whatever ran before. ``--no-drain`` restores the review driver's
fixed pause, under which a call right after ``sk_brute`` measured 2x faster on
the M4 Max.

Correctness is checked against scikit-learn brute force in float64. Routes are
read from ``est._index_._radius_search.last_plan`` when it exists, and otherwise
from a probe on ``RadiusSearch.route`` (PR #3). Results are appended to a JSONL
file keyed by (label, config), so a rerun resumes where it stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import select
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass

import numpy as np

try:  # imported as benchmarks.radius.bench_radius
    from . import radius_calibrate as rc
except ImportError:  # run as a script, including the --control child
    import radius_calibrate as rc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("RADIUS_BENCH_DATA", os.path.join(HERE, "data"))
#: The e_means harness caches real datasets as <cache>/<name>/{train,test}.npy.
E_MEANS_CACHE = os.path.expanduser("~/.cache/mlxlearn/e_means/arrays")
REAL_DATA = ("mnist", "clip", "spacev")
RAGGED_OPS = ("rn", "rns", "self")
GRAPH_OPS = ("graph", "graph_conn")
OPS = RAGGED_OPS + GRAPH_OPS + ("clf", "reg")
DATA_KINDS = ("gauss", "uniform", "blobs", "offset", "hubs") + REAL_DATA
#: Route names of the forced-route knob, used when its type does not list them.
DEFAULT_ROUTES = ("tree", "brute", "metal", "stock")
PROTOCOL_MARK = "@@radius-bench@@ "
#: Rows examined in detail (boundary distance check) per variant and config.
MAX_DETAIL_ROWS = 2000


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _data_dirs() -> list[str]:
    dirs = [p for p in DATA_DIR.split(os.pathsep) if p]
    default = os.path.join(HERE, "data")
    if default not in dirs:
        dirs.append(default)
    return dirs


def find_data(name: str, split: str | None = None) -> str:
    """Locate a real-data array.

    Looks for ``<name>-<split>.npy`` (or ``<name>.npy``) in ``RADIUS_BENCH_DATA``
    (os.pathsep-separated) and ``benchmarks/radius/data``, then for the e_means
    cache layout ``~/.cache/mlxlearn/e_means/arrays/<name>/<split>.npy``.
    """
    filename = f"{name}-{split}.npy" if split else f"{name}.npy"
    tried = []
    for directory in _data_dirs():
        path = os.path.join(directory, filename)
        tried.append(path)
        if os.path.exists(path):
            return path
    if split:
        path = os.path.join(E_MEANS_CACHE, name, f"{split}.npy")
        tried.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found; tried {tried} (set RADIUS_BENCH_DATA)")


def make_data(cfg, seed=0):
    """Return ``X, Q, y_clf, y_reg`` (float64 values exact in float32).

    Sets ``cfg["d"]`` to the data's dimension. Configs of the original grids
    produce exactly the arrays of the original review driver. Extra-grid keys:

    * ``offset`` (kind ``offset``): the shift of the Gaussian, default 1e3.
    * ``order="label"``: training rows sorted by cluster label (stable).
    * ``queries="train"``: queries are copies of strided training rows, so every
      query has a zero-distance neighbor.
    """
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
        # Gaussian far from the origin: Gram cancellation stress.
        A = rng.standard_normal((total, d), dtype=np.float32) + np.float32(cfg.get("offset", 1e3))
    elif kind == "hubs":
        # Heavy-tailed density: Zipf cluster sizes and a log-normal spread per
        # point, so a few dense cores hold most of the neighbor pairs.
        k = 32
        C = (rng.standard_normal((k, d)) * 4.0).astype(np.float32)
        weights = 1.0 / np.arange(1, k + 1) ** 1.2
        labels = rng.choice(k, total, p=weights / weights.sum())
        spread = np.exp(rng.normal(-1.0, 1.0, total)).astype(np.float32)
        A = C[labels] + rng.standard_normal((total, d), dtype=np.float32) * spread[:, None]
    elif kind == "mnist":
        X = np.load(find_data("mnist", "X"))
        y = np.load(find_data("mnist", "y"))
        perm = rng.permutation(len(X))[:total]
        A, labels = X[perm], y[perm]
    elif kind in ("clip", "spacev"):
        name = {"clip": "imagenet-clip-512-normalized", "spacev": "spacev-1M"}[kind]
        tr = np.load(find_data(name, "train"), mmap_mode="r")
        te = np.load(find_data(name, "test"), mmap_mode="r")
        idx = np.sort(rng.choice(len(tr), n, replace=False)) if n < len(tr) else np.arange(len(tr))
        X = np.asarray(tr[idx], dtype=np.float32)
        Q = np.asarray(te[:nq], dtype=np.float32)
        A = np.vstack([X, Q])
    else:
        raise ValueError(kind)
    A = np.ascontiguousarray(A, dtype=np.float32)
    d = A.shape[1]
    cfg["d"] = d
    if labels is None:
        z = A[:, 0] + 0.5 * A[:, min(1, d - 1)]
        labels = np.digitize(z, np.quantile(z, [0.33, 0.66]))
    if cfg.get("order") == "label":
        perm = np.argsort(labels[:n], kind="stable")
        A[:n] = A[:n][perm]
        labels = np.concatenate([np.asarray(labels)[:n][perm], np.asarray(labels)[n:]])
    if cfg.get("queries") == "train" and nq:
        A[n:] = A[np.arange(nq, dtype=np.int64) * max(1, n // nq) % n]
    X, Q = A[:n].astype(np.float64), A[n:].astype(np.float64)
    y_clf = (np.asarray(labels) % 5)[:n]
    y_reg = (A[:n, 0].astype(np.float64) + np.sin(A[:n, min(1, d - 1)].astype(np.float64)))
    return X, Q, y_clf, y_reg


def data_fingerprint(X, Q, y_clf, y_reg) -> str:
    """A cheap digest of strided samples, to prove a child generated the same data."""
    h = hashlib.sha256()
    for a in (X, Q, np.asarray(y_clf), np.asarray(y_reg)):
        h.update(f"{a.shape}{a.dtype}".encode())
        step = max(1, len(a) // 4096)
        h.update(np.ascontiguousarray(a[::step]).tobytes())
    return h.hexdigest()[:24]


def calibrate_radius(X, Q, m, self_query, rng):
    from sklearn.neighbors import NearestNeighbors as SkNN

    src = X if self_query else Q
    take = rng.choice(len(src), min(256, len(src)), replace=False)
    k = min(len(X), m + (1 if self_query else 0))
    dist, _ = SkNN(n_neighbors=k, algorithm="brute").fit(X).kneighbors(src[take])
    return float(np.median(dist[:, -1]))


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """One estimator configuration in a config's rotation."""

    name: str
    family: str  # "sk", "mlx" (Layer 1) or "patched"
    kwargs: tuple = ()
    stock: str | None = None  # the scikit-learn variant this patched user runs
    max_d: int | None = None  # only measured when d <= max_d
    route: str | None = None  # forced Tuning.radius_route


REGISTRY: dict[str, Variant] = {v.name: v for v in (
    Variant("sk_auto", "sk"),
    Variant("sk_brute", "sk", (("algorithm", "brute"),)),
    Variant("mlx", "mlx"),
    Variant("mlx_patched", "patched", stock="sk_auto"),
    Variant("sk_auto_nj", "sk", (("n_jobs", -1),), max_d=15),
    Variant("mlx_patched_nj", "patched", (("n_jobs", -1),), stock="sk_auto_nj"),
    Variant("mlx_patched_brute", "patched", (("algorithm", "brute"),), stock="sk_brute"),
)}
#: The review driver's variants, in its rotation order.
BASE_VARIANTS = ("sk_auto", "sk_brute", "mlx", "mlx_patched", "sk_auto_nj")
#: Variants a grid adds to the base set by default.
GRID_VARIANTS = {"extra": ("mlx_patched_nj",)}
CONTROL_FAMILIES = ("mlx", "patched")


def route_variant(route: str) -> Variant:
    return Variant(f"mlx_route={route}", "patched" if route == "stock" else "mlx", route=route)


def select_variants(d, grid=None, requested=None, routes=()) -> list[Variant]:
    """The variants of one config, in rotation order.

    ``requested`` is the ``--variants`` list: plain names select exactly those
    variants, and ``+name`` entries add to the grid's default set.
    """
    names = list(BASE_VARIANTS) + list(GRID_VARIANTS.get(grid, ()))
    if requested:
        adds = [r[1:] for r in requested if r.startswith("+")]
        exact = [r for r in requested if not r.startswith("+")]
        if adds and exact:
            raise ValueError("--variants takes either plain names or +name additions, not both")
        unknown = [r for r in adds + exact if r not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown variant(s) {unknown}; known: {sorted(REGISTRY)}")
        if exact:
            names = [n for n in REGISTRY if n in exact]
        else:
            names += [a for a in adds if a not in names]
    out = [REGISTRY[n] for n in names
           if REGISTRY[n].max_d is None or d <= REGISTRY[n].max_d]
    out += [route_variant(r) for r in routes]
    return out


def user_stock(name: str, present, d) -> str | None:
    """The scikit-learn variant whose configuration a patched variant's user chose.

    For d > 15, scikit-learn's default algorithm is brute force, whose radius
    reductions ignore ``n_jobs``. ``sk_auto_nj`` is not measured there, and
    ``sk_auto`` stands in for it.
    """
    base = name[4:] if name.startswith("ctl_") else name
    variant = REGISTRY.get(base)
    if variant is None or variant.stock is None:
        return None
    if variant.stock in present:
        return variant.stock
    if variant.stock == "sk_auto_nj" and d > 15 and "sk_auto" in present:
        return "sk_auto"
    return None


def estimator_class(variant: Variant, op: str):
    cls_name = {"clf": "RadiusNeighborsClassifier", "reg": "RadiusNeighborsRegressor"}.get(
        op, "NearestNeighbors")
    if variant.family == "sk":
        import sklearn.neighbors as module
    elif variant.family == "mlx":
        import mlxlearn.neighbors as module
    else:
        import mlxlearn.patching._estimators as module
    return getattr(module, cls_name)


def variant_specs(op, d, include):
    """``{name: (class, kwargs)}`` for the base variants (review-driver interface)."""
    return {v.name: (estimator_class(v, op), dict(v.kwargs))
            for v in select_variants(d) if include is None or v.name in include}


def variant_context(variant: Variant):
    if variant.route is None:
        return contextlib.nullcontext()
    from mlxlearn._common.config import tuning_context

    return tuning_context(radius_route=variant.route)


def route_knob() -> list[str] | None:
    """The installed mlxlearn's forced-route values, or None without the knob."""
    try:
        import typing

        from mlxlearn._common.config import Tuning
    except Exception:
        return None
    if "radius_route" not in {f.name for f in dataclasses.fields(Tuning)}:
        return None
    try:
        values = typing.get_args(typing.get_type_hints(Tuning).get("radius_route"))
    except Exception:
        values = ()
    values = [v for v in values if isinstance(v, str)]
    return values or ["auto", *DEFAULT_ROUTES]


def resolve_routes(spec: str | None, allowed: list[str] | None) -> list[str]:
    """``--routes`` (``all`` or a comma list) against the installed knob's values."""
    if not spec:
        return []
    if allowed is None:
        print("[routes] the installed mlxlearn has no Tuning.radius_route; "
              "forced-route variants skipped", file=sys.stderr, flush=True)
        return []
    if spec == "all":
        return [r for r in allowed if r != "auto"]
    routes = [r.strip() for r in spec.split(",") if r.strip()]
    unknown = [r for r in routes if r not in allowed]
    if unknown:
        print(f"[routes] skipping {unknown}: not in {allowed}", file=sys.stderr, flush=True)
    return [r for r in routes if r in allowed and r != "auto"]


def build_call(op, Q):
    if op == "rn":
        return lambda est: est.radius_neighbors(Q)
    if op == "rns":
        return lambda est: est.radius_neighbors(Q, sort_results=True)
    if op == "self":
        return lambda est: est.radius_neighbors()
    if op == "graph":
        return lambda est: est.radius_neighbors_graph(Q, mode="distance")
    if op == "graph_conn":
        return lambda est: est.radius_neighbors_graph(Q, mode="connectivity")
    if op in ("clf", "reg"):
        return lambda est: est.predict(Q)
    raise ValueError(op)


def make_estimator(spec, op, radius, weights):
    cls, extra = spec
    kw = dict(extra)
    kw["radius"] = radius
    if op == "clf":
        kw["outlier_label"] = "most_frequent"
        kw["weights"] = weights
    elif op == "reg":
        kw["weights"] = weights
    return cls(**kw)


def fit_estimator(est, op, X, y_clf, y_reg):
    if op == "clf":
        return est.fit(X, y_clf)
    if op == "reg":
        return est.fit(X, y_reg)
    return est.fit(X)


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------

def membership(op, out, n_fit):
    """Per-row counts and the column indices sorted within each row."""
    if op in GRAPH_OPS:
        a = out.tocsr(copy=True)
        a.sort_indices()
        return np.diff(a.indptr).astype(np.int64), a.indices.astype(np.int64)
    ind = out[1] if isinstance(out, tuple) else out
    counts = np.fromiter(map(len, ind), dtype=np.int64, count=len(ind))
    if not counts.sum():
        return counts, np.empty(0, dtype=np.int64)
    flat = np.concatenate(list(ind)).astype(np.int64, copy=False)
    rows = np.repeat(np.arange(len(ind), dtype=np.int64), counts)
    if flat.min() < 0 or flat.max() >= n_fit:
        order = np.lexsort((flat, rows))
        return counts, flat[order]
    key = rows * np.int64(n_fit) + flat
    key.sort()
    return counts, key - rows * np.int64(n_fit)


def membership_digest(mem) -> str:
    counts, flat = mem
    h = hashlib.sha256(np.ascontiguousarray(counts, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(flat, dtype=np.int64).tobytes())
    return h.hexdigest()[:32]


def _row_hash(counts, flat):
    """An order-independent 64-bit hash of each row's index set."""
    h = np.zeros(len(counts), dtype=np.uint64)
    if not flat.size:
        return h
    with np.errstate(over="ignore"):
        x = (flat.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0x94D049BB133111EB)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    nonempty = counts > 0
    h[nonempty] = np.add.reduceat(x, starts[nonempty])
    return h


def compare(op, out, ref, X, Q, radius, *, n_fit=None, mem=None, ref_mem=None):
    """Return a dict summarizing disagreement with the reference output."""
    n_fit = len(X) if n_fit is None else n_fit
    if op in RAGGED_OPS:
        dist, ind = out
        rdist, rind = ref
        mem = membership(op, out, n_fit) if mem is None else mem
        ref_mem = membership(op, ref, n_fit) if ref_mem is None else ref_mem
        counts, flat = mem
        rcounts, rflat = ref_mem
        if counts.shape == rcounts.shape and np.array_equal(counts, rcounts) and \
                np.array_equal(flat, rflat):
            bad = np.empty(0, dtype=np.int64)
        elif counts.shape != rcounts.shape:
            bad = np.arange(max(len(counts), len(rcounts)))
        else:
            bad = np.flatnonzero((counts != rcounts)
                                 | (_row_hash(counts, flat) != _row_hash(rcounts, rflat)))
        src = X if op == "self" else Q
        off_boundary = 0
        worst = 0.0
        for i in bad[:MAX_DETAIL_ROWS].tolist():
            if i >= len(ind) or i >= len(rind):
                continue
            diff = np.setxor1d(np.asarray(ind[i]), np.asarray(rind[i]))
            exact = np.sqrt(((X[diff] - src[i]) ** 2).sum(axis=1))
            rel = np.abs(exact - radius) / max(radius, 1e-300)
            worst = max(worst, float(rel.max(initial=0)))
            off_boundary += int(np.count_nonzero(rel > 1e-5))
        # distance accuracy on a sample of rows
        derr = 0.0
        for i in range(0, len(ind), max(1, len(ind) // 64)):
            a = np.asarray(ind[i])
            if a.size:
                exact = np.sqrt(((X[a] - src[i]) ** 2).sum(axis=1))
                derr = max(derr, float(np.max(np.abs(np.asarray(dist[i]) - exact)
                                                / np.maximum(exact, 1e-30))))
        unsorted = 0
        if op == "rns" and counts.sum():
            values = np.concatenate(list(dist))
            rows = np.repeat(np.arange(len(ind), dtype=np.int64), counts)
            down = (np.diff(values) < 0) & (rows[1:] == rows[:-1])
            unsorted = int(np.unique(rows[1:][down]).size)
        result = {"bad_rows": int(bad.size), "off_boundary": off_boundary, "worst_rel": worst,
                  "dist_rel_err": derr, "unsorted_rows": unsorted}
        if bad.size > MAX_DETAIL_ROWS:
            result["detail_rows"] = MAX_DETAIL_ROWS
        return result
    if op in GRAPH_OPS:
        a = out.tocsr(copy=True)
        b = ref.tocsr(copy=True)
        a.sort_indices()
        b.sort_indices()
        same_pattern = (a.indptr.shape == b.indptr.shape and np.array_equal(a.indptr, b.indptr)
                        and np.array_equal(a.indices, b.indices))
        nnz_diff = abs(a.nnz - b.nnz)
        verr = float(np.max(np.abs(a.data - b.data))) if same_pattern and a.nnz else None
        return {"same_pattern": bool(same_pattern), "nnz_diff": int(nnz_diff), "max_abs_err": verr,
                "type": type(out).__name__, "ref_type": type(ref).__name__,
                "indices_dtype": str(out.indices.dtype), "ref_indices_dtype": str(ref.indices.dtype)}
    if op == "clf":
        return {"disagree": int(np.count_nonzero(np.asarray(out) != np.asarray(ref)))}
    if op == "reg":
        a = np.asarray(out, dtype=np.float64)
        b = np.asarray(ref, dtype=np.float64)
        nan_mismatch = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
        both = ~np.isnan(a) & ~np.isnan(b)
        err = float(np.max(np.abs(a[both] - b[both]), initial=0))
        return {"nan_mismatch": nan_mismatch, "max_abs_err": err}
    return {}


def summarize(out, op):
    if op in RAGGED_OPS:
        ind = out[1]
        counts = np.fromiter((len(r) for r in ind), dtype=np.int64, count=len(ind))
        if not counts.size:
            return {"pairs": 0, "mean_count": 0.0, "median_count": 0.0, "max_count": 0,
                    "empty_rows": 0}
        return {"pairs": int(counts.sum()), "mean_count": float(counts.mean()),
                "median_count": float(np.median(counts)), "max_count": int(counts.max(initial=0)),
                "empty_rows": int(np.count_nonzero(counts == 0))}
    if op in GRAPH_OPS:
        return {"pairs": int(out.nnz)}
    return {}


def check_payload(op, out, n_fit):
    """What a control child returns for the parent's correctness cross-check."""
    if op in RAGGED_OPS or op in GRAPH_OPS:
        mem = membership(op, out, n_fit)
        return {"digest": membership_digest(mem), "pairs": int(mem[0].sum())}
    return {"pred": np.asarray(out).tolist()}


# ---------------------------------------------------------------------------
# route capture (any mlxlearn version)
# ---------------------------------------------------------------------------

ROUTES: list = []


def install_route_probe():
    """Record every ``RadiusSearch.route`` result (PR #3 and its thin-wrapper successor)."""
    try:
        from mlxlearn._kernels import radius as R
    except Exception:
        return False
    cls = getattr(R, "RadiusSearch", None)
    if cls is None or not hasattr(cls, "route"):
        return False
    original = cls.route
    if getattr(original, "_radius_bench_probe", False):
        return True

    def route(self, *a, **k):
        r = original(self, *a, **k)
        ROUTES.append(r)
        return r

    route._radius_bench_probe = True
    cls.route = route
    return True


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return str(value)


def _search_of(est):
    index = getattr(est, "__dict__", {}).get("_index_")
    return getattr(index, "_radius_search", None) if index is not None else None


def clear_plan(est):
    """Forget the previous call's plan so a stale one is never recorded."""
    search = _search_of(est)
    if search is not None and getattr(search, "last_plan", None) is not None:
        with contextlib.suppress(Exception):
            search.last_plan = None


def plan_info(plan) -> dict | None:
    if plan is None:
        return None
    return {attr: _jsonable(getattr(plan, attr))
            for attr in ("route", "reason", "threads", "predicted_ms") if hasattr(plan, attr)}


def capture_route(est):
    """``(routes, plan)`` of the call that just ran: ``last_plan``, else the probe."""
    search = _search_of(est)
    info = plan_info(getattr(search, "last_plan", None) if search is not None else None)
    if info is not None and info.get("route") is not None:
        return [str(info["route"])], info
    return [str(getattr(r, "route", r)) for r in ROUTES], None


def mlx_backend(est):
    for attr in ("last_query_backend_", "execution_backend_"):
        try:
            return getattr(est, attr)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# variant runners (in-process and control child)
# ---------------------------------------------------------------------------

class LocalVariant:
    """Fits and calls one variant in this process."""

    def __init__(self, variant: Variant, op, data, radius, weights):
        self.variant = variant
        self.op = op
        self.X, self.Q, self.y_clf, self.y_reg = data
        self.radius = radius
        self.weights = weights
        self.fn = build_call(op, self.Q)
        self.est = None

    def cold(self) -> dict:
        with variant_context(self.variant):
            est = make_estimator((estimator_class(self.variant, self.op), dict(self.variant.kwargs)),
                                 self.op, self.radius, self.weights)
            t0 = time.perf_counter()
            fit_estimator(est, self.op, self.X, self.y_clf, self.y_reg)
            t1 = time.perf_counter()
            ROUTES.clear()
            t2 = time.perf_counter()
            out = self.fn(est)
            t3 = time.perf_counter()
        routes, plan = capture_route(est)
        self.est = est
        return {"fit": t1 - t0, "cold": t3 - t2, "routes": routes, "plan": plan,
                "backend": mlx_backend(est), "out": out}

    def call(self, primers=0, keep=False) -> dict:
        est = self.est
        with variant_context(self.variant):
            # Primer outputs stay alive until after the timed call, so freeing
            # them never sits between the primer and the timed call.
            primed = [self.fn(est) for _ in range(primers)]
            clear_plan(est)
            ROUTES.clear()
            t0 = time.perf_counter()
            out = self.fn(est)
            t1 = time.perf_counter()
        routes, plan = capture_route(est)
        del primed
        return {"t": t1 - t0, "routes": routes, "plan": plan, "backend": mlx_backend(est),
                "out": out if keep else None}


class ControlError(RuntimeError):
    pass


class ControlChild:
    """An older mlxlearn source tree, served by this script in a child process.

    The child imports mlxlearn from ``PYTHONPATH=<src>`` only. Each command is one
    JSON line, and the child times its own calls, so the pipe is never inside a
    timed region. Pauses and thermal waits happen in the parent, under the same
    protocol as the in-process variants.
    """

    def __init__(self, src, *, timeout=900.0, python=None):
        self.src = os.path.abspath(src)
        if not os.path.isdir(os.path.join(self.src, "mlxlearn")):
            raise ControlError(f"{self.src} has no mlxlearn package")
        self.timeout = timeout
        env = dict(os.environ)
        env["PYTHONPATH"] = self.src
        self.proc = subprocess.Popen(
            [python or sys.executable, os.path.abspath(__file__), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )
        self._buf = b""
        hello = self._read(max(timeout, 120.0))
        if "error" in hello:
            self.close()
            raise ControlError(f"control child failed to start: {hello['error']}")
        self.meta = hello.get("meta", {})
        for name, path in hello.get("packages", {}).items():
            inside = os.path.abspath(path).startswith(self.src + os.sep)
            if os.path.isdir(os.path.join(self.src, name)) and not inside:
                self.close()
                raise ControlError(f"control child imported {name} from {path}, not {self.src}")
        self.meta["packages"] = hello.get("packages", {})

    @property
    def pid(self):
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read(self, timeout):
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" not in self._buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.kill()
                    raise ControlError(f"control child timed out after {timeout:.0f} s")
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    continue
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    raise ControlError(f"control child exited (code {self.proc.poll()})")
                self._buf += chunk
            line, _, self._buf = self._buf.partition(b"\n")
            text = line.decode(errors="replace")
            if text.startswith(PROTOCOL_MARK):
                return json.loads(text[len(PROTOCOL_MARK):])

    def request(self, msg, timeout=None) -> dict:
        if not self.alive():
            raise ControlError("control child is not running")
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ControlError(f"control child pipe closed: {exc!r}") from None
        reply = self._read(self.timeout if timeout is None else timeout)
        if "error" in reply:
            raise ControlError(f"{reply['error']}\n{reply.get('trace', '')}")
        return reply

    def kill(self):
        with contextlib.suppress(Exception):
            self.proc.kill()
            self.proc.wait(5)

    def close(self):
        if self.alive():
            with contextlib.suppress(Exception):
                self.request({"cmd": "exit"}, timeout=30)
            with contextlib.suppress(Exception):
                self.proc.wait(10)
        if self.alive():
            self.kill()


class RemoteVariant:
    """A ``ctl_<base>`` variant: ``base`` timed by the control child."""

    def __init__(self, name, base, child: ControlChild):
        self.name = name
        self.base = base
        self.child = child

    def cold(self) -> dict:
        return self.child.request({"cmd": "cold", "variant": self.base})

    def call(self, primers=0, keep=False) -> dict:
        return self.child.request({"cmd": "call", "variant": self.base, "primers": primers,
                                   "check": bool(keep)})


class ChildSession:
    """The control child's state for one config."""

    def __init__(self, msg):
        cfg = dict(msg["cfg"])
        self.op = cfg["op"]
        X, Q, y_clf, y_reg = make_data(cfg, seed=cfg.get("seed", 0))
        self.n_fit = len(X)
        self.fingerprint = data_fingerprint(X, Q, y_clf, y_reg)
        data = (X, Q, y_clf, y_reg)
        weights = cfg.get("weights", "uniform")
        self.runners = {name: LocalVariant(REGISTRY[name], self.op, data, msg["radius"], weights)
                        for name in msg["variants"]}

    def cold(self, name) -> dict:
        res = self.runners[name].cold()
        res.pop("out", None)
        return res

    def call(self, name, primers, check) -> dict:
        res = self.runners[name].call(primers=primers, keep=check)
        out = res.pop("out", None)
        if check:
            res["payload"] = check_payload(self.op, out, self.n_fit)
        return res


def serve() -> int:
    """Control-child main loop: JSON commands on stdin, marked replies on stdout."""
    warnings.filterwarnings("ignore")
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # stray writes to fd 1 must not corrupt the protocol
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(PROTOCOL_MARK + json.dumps(_jsonable(obj)) + "\n")
        proto.flush()

    try:
        import mlxlearn

        packages = {"mlxlearn": mlxlearn.__file__}
        with contextlib.suppress(Exception):
            import mlxlearn_device

            packages["mlxlearn_device"] = mlxlearn_device.__file__
        install_route_probe()
        send({"hello": True, "meta": rc.machine_meta(mlxlearn_module=mlxlearn),
              "packages": packages, "pid": os.getpid()})
    except Exception as exc:
        send({"error": repr(exc), "trace": traceback.format_exc()[-2000:]})
        return 1
    session = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "exit":
                send({"ok": True})
                break
            if cmd == "setup":
                session = None
                gc.collect()
                t0 = time.perf_counter()
                session = ChildSession(msg)
                send({"ok": True, "fingerprint": session.fingerprint,
                      "setup_s": time.perf_counter() - t0})
            elif cmd == "teardown":
                session = None
                gc.collect()
                send({"ok": True})
            elif cmd == "drain":
                send({"ok": True, "drained_s": rc.drain_idle()})
            elif cmd == "collect":
                gc.collect()
                send({"ok": True})
            elif cmd == "cold":
                send(session.cold(msg["variant"]))
            elif cmd == "call":
                send(session.call(msg["variant"], int(msg.get("primers", 0)),
                                  bool(msg.get("check", False))))
            else:
                send({"error": f"unknown command {cmd!r}"})
        except Exception as exc:
            send({"error": repr(exc), "trace": traceback.format_exc()[-2000:]})
    return 0


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def config_key(cfg):
    return "|".join(f"{k}={cfg[k]}" for k in sorted(cfg) if k not in ("d_out",))


def canonical_key(cfg):
    """``config_key`` without the implied ``d`` of real datasets.

    The review driver stored real-data configs with ``d`` filled in, while the
    grids leave it out. This key matches both forms.
    """
    if cfg.get("data") in REAL_DATA:
        cfg = {k: v for k, v in cfg.items() if k != "d"}
    return config_key(cfg)


def _median(values):
    return float(np.median(values)) if values else None


def _drop(rec, name, exc):
    text = str(exc) if isinstance(exc, ControlError) else repr(exc)
    rec["variant_errors"][name] = text[:2000]
    print(f"   [{name}] error: {text.splitlines()[0][:200] if text else ''}", file=sys.stderr,
          flush=True)


def _control_check(op, payload, ref, ref_digest, local, base, X, Q, radius, n_fit) -> dict:
    """A control output against the reference and its in-process counterpart."""
    if "digest" in payload:
        check = {"pairs": payload["pairs"], "same_membership_as_ref": payload["digest"] == ref_digest}
        if local is not None:
            check["same_membership_as_" + base] = payload["digest"] == local
        return check
    pred = np.asarray(payload["pred"])
    check = compare(op, pred, ref, X, Q, radius, n_fit=n_fit)
    if local is not None:
        check["same_as_" + base] = bool(np.array_equal(
            pred.astype(local.dtype, copy=False), local, equal_nan=pred.dtype.kind == "f"))
    return check


def run_config(cfg, args, label, *, control=None, routes=(), gate=None):
    gate = gate if gate is not None else rc.ThermalGate(max_level=9)
    # ``ps`` forks this process; data generation below rewrites the pages, so the
    # copy-on-write faults that follow a fork never land in a timed call.
    load_start = rc.foreign_load(exclude=[control.pid] if control else ())
    rng = np.random.default_rng(1234)
    t_data = time.perf_counter()
    X, Q, y_clf, y_reg = make_data(dict(cfg), seed=cfg.get("seed", 0))
    d = int(X.shape[1])
    op = cfg["op"]
    self_query = op == "self"
    radius = cfg.get("radius") or calibrate_radius(X, Q, cfg["m"], self_query, rng)
    weights = cfg.get("weights", "uniform")
    variants = select_variants(d, getattr(args, "grid", None), args.variants, routes)
    data = (X, Q, y_clf, y_reg)
    runners = {v.name: LocalVariant(v, op, data, radius, weights) for v in variants}
    names = [v.name for v in variants]
    t_data = time.perf_counter() - t_data

    rec = {"label": label, "cfg": cfg, "radius": radius, "d": d, "host": os.uname().nodename,
           "variants": names, "cold": {}, "warm": {}, "fit": {}, "routes": {}, "plans": {},
           "backend": {}, "check": {}, "setup_s": t_data, "variant_errors": {},
           "route_trace": {}, "warm_round": {},
           "protocol": {"gap": args.gap, "pause": args.pause,
                        "hot_primers": args.hot_primers if args.gap == "hot" else 0,
                        "cold": args.cold, "warm": args.warm,
                        "drain": not getattr(args, "no_drain", False)},
           "thermal_start": rc.thermal_level(),
           "load_start": load_start["top"], "load_start_total": load_start["total_pcpu"]}

    if control is not None:
        bases = [v.name for v in variants if v.route is None and v.family in CONTROL_FAMILIES]
        try:
            ready = control.request({"cmd": "setup", "cfg": cfg, "radius": radius,
                                     "variants": bases})
            fingerprint = data_fingerprint(X, Q, y_clf, y_reg)
            if ready.get("fingerprint") != fingerprint:
                raise ControlError(f"control data differs: {ready.get('fingerprint')} != "
                                   f"{fingerprint}")
            for base in bases:
                runners["ctl_" + base] = RemoteVariant("ctl_" + base, base, control)
                names.append("ctl_" + base)
            rec["control"] = {"src": control.src, "variants": ["ctl_" + b for b in bases],
                              "setup_s": ready.get("setup_s")}
        except ControlError as exc:
            rec["control"] = {"src": control.src, "error": str(exc)[:2000]}
        rec["variants"] = names

    present = set(names)
    rec["user_stock"] = {n: s for n in names if (s := user_stock(n, present, d)) is not None}
    rec["user_config"] = {n: dict(REGISTRY[n[4:] if n.startswith("ctl_") else n].kwargs)
                          for n in rec["user_stock"]}
    if routes:
        rec["forced_routes"] = {v.name: v.route for v in variants if v.route is not None}
    errors = rec["variant_errors"]
    pause = args.pause
    hot = args.gap == "hot"
    drain = rec["protocol"]["drain"]
    drained = rec["drain_s"] = {"parent": 0.0, "child": 0.0}

    def settle(previous):
        """Thermal gate, then wait out any spinning threads of the previous call."""
        gate.wait()
        if not drain:
            return
        drained["parent"] += rc.drain_idle()
        if previous and previous.startswith("ctl_") and control is not None and control.alive():
            with contextlib.suppress(ControlError):
                drained["child"] += control.request({"cmd": "drain"}).get("drained_s", 0.0)

    previous = None

    # ---- cold: fit + first query, rotated order
    slow = set()
    ref = None
    for r in range(args.cold):
        for name in rc.rotation(names, r, reverse_odd=False):
            if name in slow or name in errors:
                continue
            # Collect before the drain and pause, so the collector's work never
            # sits right before a timed fit.
            gc.collect()
            if name.startswith("ctl_") and control is not None and control.alive():
                with contextlib.suppress(ControlError):
                    control.request({"cmd": "collect"})
            settle(previous)
            if pause:
                time.sleep(pause)
            previous = name
            try:
                res = runners[name].cold()
            except Exception as exc:
                _drop(rec, name, exc)
                continue
            rec["fit"].setdefault(name, []).append(res["fit"])
            rec["cold"].setdefault(name, []).append(res["cold"])
            if r == 0:
                rec["routes"][name + ":cold"] = res["routes"]
                if res.get("plan"):
                    rec["plans"][name + ":cold"] = res["plan"]
                rec["backend"][name + ":cold"] = res["backend"]
                if name == "sk_brute":
                    ref = res["out"]
                    rec["ref_summary"] = summarize(ref, op)
            if res["cold"] > args.slow:
                slow.add(name)
            res = None

    # reference output (cold outputs are not kept; recompute once if needed)
    if ref is None:
        ref_est = fit_estimator(make_estimator(
            (estimator_class(REGISTRY["sk_brute"], op), {"algorithm": "brute"}), op, radius,
            weights), op, X, y_clf, y_reg)
        ref = build_call(op, Q)(ref_est)
        rec["ref_summary"] = summarize(ref, op)
        del ref_est
    n_fit = len(X)
    ref_mem = membership(op, ref, n_fit) if op in RAGGED_OPS + GRAPH_OPS else None
    ref_digest = membership_digest(ref_mem) if ref_mem is not None else None
    local_payload = {}
    remote_payload = {}

    # ---- warm: repeated queries on fitted estimators
    budget_rounds = args.warm
    first = {k: min(v) for k, v in rec["cold"].items()}
    longest = max(first.values()) if first else 0
    if longest > 4.0:
        budget_rounds = max(2, min(budget_rounds, int(40 / longest)))
    rec["protocol"]["rounds"] = budget_rounds
    for r in range(budget_rounds + 1):  # round 0 is an untimed warm-up
        for name in rc.rotation(names, r):
            if name in errors or (name in slow and r > 1):
                continue
            settle(previous)
            if pause and not hot:
                time.sleep(pause)
            previous = name
            primers = args.hot_primers if hot and r > 0 else 0
            try:
                res = runners[name].call(primers=primers, keep=(r == 0))
            except Exception as exc:
                _drop(rec, name, exc)
                continue
            if r == 0:
                rec["routes"][name + ":warm"] = res["routes"]
                if res.get("plan"):
                    rec["plans"][name + ":warm"] = res["plan"]
                rec["backend"][name + ":warm"] = res["backend"]
                if name.startswith("ctl_"):
                    remote_payload[name] = res.get("payload")
                elif name != "sk_brute":
                    out = res["out"]
                    mem = membership(op, out, n_fit) if ref_mem is not None else None
                    rec["check"][name] = compare(op, out, ref, X, Q, radius, n_fit=n_fit,
                                                 mem=mem, ref_mem=ref_mem)
                    if not name.startswith("sk_"):
                        local_payload[name] = (membership_digest(mem) if mem is not None
                                               else np.asarray(out))
                    del out
            else:
                rec["warm"].setdefault(name, []).append(res["t"])
                rec["warm_round"].setdefault(name, []).append(r)
                if not name.startswith("sk_"):
                    rec["route_trace"].setdefault(name, []).append(",".join(res["routes"]))
            res = None
        if r == 0:
            # Check the control outputs, then free the reference before timing.
            for name, payload in remote_payload.items():
                if payload is not None:
                    rec["check"][name] = _control_check(op, payload, ref, ref_digest,
                                                        local_payload.get(name[4:]), name[4:],
                                                        X, Q, radius, n_fit)
            ref = ref_mem = None
            local_payload.clear()
            remote_payload.clear()

    if control is not None and "error" not in rec.get("control", {}):
        with contextlib.suppress(ControlError):
            control.request({"cmd": "teardown"})
    rec["thermal_end"] = rc.thermal_level()
    load = rc.foreign_load(exclude=[control.pid] if control else ())
    rec["load_end"] = load["top"]
    rec["load_end_total"] = load["total_pcpu"]
    rec["thermal_gate"] = gate.snapshot()

    summary = {}
    for name in names:
        summary[name] = {
            "warm_med": _median(rec["warm"].get(name)),
            "warm_min": float(min(rec["warm"][name])) if rec["warm"].get(name) else None,
            "cold_med": _median(rec["cold"].get(name)),
            "fit_med": _median(rec["fit"].get(name)),
        }
    rec["summary"] = summary
    base = (summary.get("sk_auto") or {}).get("warm_med") or \
        (summary.get("sk_auto") or {}).get("cold_med")
    sk_warm = [s["warm_med"] for n, s in summary.items() if n.startswith("sk_") and s["warm_med"]]
    sk_e2e = [s["cold_med"] + s["fit_med"] for n, s in summary.items()
              if n.startswith("sk_") and s["cold_med"] is not None and s["fit_med"] is not None]
    sk_best_warm = min(sk_warm, default=None)
    sk_best_e2e = min(sk_e2e, default=None)
    for name, s in summary.items():
        if name.startswith("sk_"):
            continue
        warm = s["warm_med"]
        e2e = (s["cold_med"] + s["fit_med"]) if s["cold_med"] is not None and \
            s["fit_med"] is not None else None
        s["speedup_vs_sk_auto_warm"] = (base / warm) if base and warm else None
        s["speedup_vs_sk_best_warm"] = (sk_best_warm / warm) if sk_best_warm and warm else None
        s["speedup_vs_sk_best_e2e"] = (sk_best_e2e / e2e) if sk_best_e2e and e2e else None
        stock = rec["user_stock"].get(name)
        if stock:
            stock_warm = summary.get(stock, {}).get("warm_med")
            s["user_stock"] = stock
            s["speedup_vs_user_stock_warm"] = (stock_warm / warm) if stock_warm and warm else None
        if name.startswith("ctl_") and name[4:] in summary:
            new = summary[name[4:]]["warm_med"]
            s["new_speedup_vs_ctl_warm"] = (warm / new) if warm and new else None
    return rec


GRIDS = {}


def grid(name):
    def deco(fn):
        GRIDS[name] = fn
        return fn
    return deco


@grid("micro")
def _micro():
    """Tiny configs for tests and quick plumbing checks."""
    return [
        {"data": "gauss", "n": 2000, "d": 3, "nq": 50, "m": 5, "op": "rn"},
        {"data": "gauss", "n": 2000, "d": 32, "nq": 50, "m": 5, "op": "rn"},
    ]


@grid("smoke")
def _smoke():
    return [
        {"data": "gauss", "n": 20000, "d": 3, "nq": 1000, "m": 10, "op": "rn"},
        {"data": "gauss", "n": 20000, "d": 32, "nq": 1000, "m": 10, "op": "rn"},
    ]


@grid("core")
def _core():
    cfgs = []
    # low-d: KD-tree territory for scikit-learn
    for data in ("uniform", "blobs"):
        for d in (2, 3, 8):
            for n in (100_000, 1_000_000):
                for nq in (1_000, 10_000):
                    for m in (10, 100):
                        if n == 1_000_000 and nq == 10_000 and m == 100 and d == 8:
                            continue
                        cfgs.append({"data": data, "n": n, "d": d, "nq": nq, "m": m, "op": "rn"})
    # mid-d 9..15: scikit-learn still picks KD-tree; PR uses host screening
    for d in (12,):
        for nq in (1_000, 10_000):
            for m in (10, 100):
                cfgs.append({"data": "gauss", "n": 100_000, "d": d, "nq": nq, "m": m, "op": "rn"})
    # high-d: brute force for scikit-learn
    for data in ("gauss", "blobs"):
        for d in (16, 64, 128):
            for n in (100_000, 1_000_000):
                for nq in (64, 1_000, 10_000):
                    for m in (1, 10, 100, 1000):
                        if n == 1_000_000 and nq == 10_000 and m >= 1000:
                            continue
                        if data == "blobs" and m in (1, 1000):
                            continue
                        cfgs.append({"data": data, "n": n, "d": d, "nq": nq, "m": m, "op": "rn"})
    # small / latency
    for d in (3, 16, 64):
        for nq in (1, 16):
            cfgs.append({"data": "gauss", "n": 10_000, "d": d, "nq": nq, "m": 10, "op": "rn"})
    return cfgs


@grid("real")
def _real():
    cfgs = []
    for m in (1, 10, 100, 1000):
        for nq in (1_000, 10_000):
            cfgs.append({"data": "mnist", "n": 50_000, "nq": nq, "m": m, "op": "rn"})
    for data, n in (("clip", 1_000_000), ("spacev", 1_000_000), ("clip", 200_000)):
        for m in (1, 10, 100):
            for nq in (1_000, 10_000):
                if n == 1_000_000 and nq == 10_000 and m == 100:
                    continue
                cfgs.append({"data": data, "n": n, "nq": nq, "m": m, "op": "rn"})
    return cfgs


@grid("ops")
def _ops():
    cfgs = []
    for d in (3, 16, 64):
        for m in (10, 100):
            base = {"data": "blobs", "n": 100_000, "d": d, "nq": 10_000, "m": m}
            cfgs.append({**base, "op": "rns"})
            cfgs.append({**base, "op": "graph"})
            cfgs.append({**base, "op": "clf", "weights": "uniform"})
            cfgs.append({**base, "op": "clf", "weights": "distance"})
            cfgs.append({**base, "op": "reg", "weights": "uniform"})
            cfgs.append({**base, "op": "reg", "weights": "distance"})
            cfgs.append({**{**base, "nq": 0, "n": 50_000}, "op": "self"})
    for m in (10, 100):
        cfgs.append({"data": "mnist", "n": 50_000, "nq": 10_000, "m": m, "op": "clf",
                     "weights": "uniform"})
    return cfgs


@grid("extra")
def _extra():
    """Stress and usage cells beyond the review grids (design plan WI 1.7).

    Every extra config also times ``mlx_patched_nj``, a user who asked for
    ``n_jobs=-1``.
    """
    cfgs = []
    # offsets: Gram cancellation and centring stress
    for offset in (1_000, 1_000_000):
        for d in (3, 8, 16, 64):
            for m in (10, 100):
                cfgs.append({"data": "offset", "offset": offset, "n": 100_000, "d": d,
                             "nq": 10_000, "m": m, "op": "rn"})
    # training rows in label-sorted order: hits concentrate in a few column chunks
    for d in (3, 16, 64):
        for nq in (1_000, 10_000):
            for m in (10, 100):
                cfgs.append({"data": "blobs", "order": "label", "n": 100_000, "d": d,
                             "nq": nq, "m": m, "op": "rn"})
    # hubs: heavy-tailed density
    for d in (3, 16, 64):
        for nq in (1_000, 10_000):
            for m in (10, 100):
                cfgs.append({"data": "hubs", "n": 100_000, "d": d, "nq": nq, "m": m, "op": "rn"})
    # self queries (X=None)
    for d in (3, 8, 16):
        for n, m in ((50_000, 10), (50_000, 100), (1_000_000, 10)):
            cfgs.append({"data": "uniform", "n": n, "d": d, "nq": 0, "m": m, "op": "self"})
    # latency calls
    for n in (10_000, 100_000):
        for d in (3, 16, 64):
            for nq in (1, 16, 64):
                cfgs.append({"data": "gauss", "n": n, "d": d, "nq": nq, "m": 10, "op": "rn"})
    # connectivity graph
    for d in (3, 16, 64):
        for m in (10, 100):
            cfgs.append({"data": "blobs", "n": 100_000, "d": d, "nq": 10_000, "m": m,
                         "op": "graph_conn"})
    # distance weights, with fresh queries and with queries that coincide with
    # training rows (sklearn's zero-distance rule on every row)
    for d in (3, 16, 64):
        for coincident in (False, True):
            for op in ("clf", "reg"):
                cfg = {"data": "gauss", "n": 100_000, "d": d, "nq": 10_000, "m": 10, "op": op,
                       "weights": "distance"}
                if coincident:
                    cfg["queries"] = "train"
                cfgs.append(cfg)
    return cfgs


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Radius-neighbor benchmark: mlxlearn against scikit-learn.",
        epilog="See benchmarks/radius/README.md.")
    ap.add_argument("grid", nargs="?", help=f"one of {sorted(GRIDS)}")
    ap.add_argument("--out", help="JSONL file to append records to (resumes by label and config)")
    ap.add_argument("--label", default="run", help="run label stored in every record")
    ap.add_argument("--cold", type=int, default=2, help="fresh fit + first query rounds")
    ap.add_argument("--warm", type=int, default=5, help="timed warm rounds (plus one untimed)")
    ap.add_argument("--pause", type=float, default=0.3,
                    help="seconds idle before each cold call, and before each warm call when "
                         "--gap gapped")
    ap.add_argument("--gap", choices=("gapped", "hot"), default="gapped",
                    help="gapped: pause before every warm call; hot: no pause, primed calls")
    ap.add_argument("--hot-primers", type=int, default=1,
                    help="untimed calls right before each timed warm call when --gap hot")
    ap.add_argument("--no-drain", action="store_true",
                    help="do not wait for spinning threads (scikit-learn's OpenMP blocktime) "
                         "before each call, as the review driver did")
    ap.add_argument("--slow", type=float, default=60.0,
                    help="variants whose cold query exceeds this many seconds get one timed "
                         "warm round")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="exact variant names, or +name to add to the grid's defaults")
    ap.add_argument("--routes", default=None,
                    help="forced-route variants: 'all' or a comma list (e.g. tree,brute,metal)")
    ap.add_argument("--control", default=None, metavar="SRC",
                    help="also time an older mlxlearn source tree (its src directory) in a "
                         "child process, merged as ctl_<variant>")
    ap.add_argument("--control-timeout", type=float, default=900.0,
                    help="seconds to wait for one control-child reply")
    ap.add_argument("--only", default=None, help="substring filter on the config key")
    ap.add_argument("--max", type=int, default=None, help="run at most this many configs")
    ap.add_argument("--thermal-max", type=int, default=0,
                    help="wait while the thermal pressure level exceeds this (9 disables)")
    ap.add_argument("--thermal-limit", type=float, default=600.0,
                    help="longest single thermal wait in seconds")
    ap.add_argument("--retry-errors", action="store_true",
                    help="rerun configs whose stored record is an error")
    ap.add_argument("--list", action="store_true", help="print the grid's config keys and exit")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.serve:
        return args
    if not args.grid:
        ap.error("a grid is required")
    if args.grid not in GRIDS:
        ap.error(f"unknown grid {args.grid!r}; choose from {sorted(GRIDS)}")
    if not args.list and not args.out:
        ap.error("--out is required")
    if args.cold < 1 or args.warm < 1:
        ap.error("--cold and --warm must be at least 1")
    if args.hot_primers < 0 or args.pause < 0:
        ap.error("--hot-primers and --pause must be non-negative")
    try:
        select_variants(3, args.grid, args.variants)
    except ValueError as exc:
        ap.error(str(exc))
    return args


def select_configs(args) -> list[dict]:
    cfgs = GRIDS[args.grid]()
    if args.only:
        cfgs = [c for c in cfgs if args.only in config_key(c)]
    if args.max:
        cfgs = cfgs[: args.max]
    return cfgs


def load_done(path, retry_errors=False) -> set:
    done = set()
    if not path or not os.path.exists(path):
        return done
    last = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                last[(r["label"], canonical_key(r["cfg"]))] = "error" in r
            except Exception:
                pass
    for key, failed in last.items():
        if not (failed and retry_errors):
            done.add(key)
    return done


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _brief(rec) -> str:
    s = rec.get("summary", {})
    times = " ".join(f"{k}={v['warm_med'] * 1e3:.1f}ms" for k, v in s.items()
                     if v.get("warm_med") is not None)
    ratios = []
    for k, v in s.items():
        if k.startswith("sk_"):
            continue
        part = f"{k}:{v.get('speedup_vs_sk_best_warm') or 0:.2f}x"
        if v.get("speedup_vs_user_stock_warm") is not None:
            part += f"(user {v['speedup_vs_user_stock_warm']:.2f}x)"
        if v.get("new_speedup_vs_ctl_warm") is not None:
            part += f"(new/ctl {v['new_speedup_vs_ctl_warm']:.2f}x)"
        ratios.append(part)
    routes = " ".join(f"{k}={','.join(v)}" for k, v in rec.get("routes", {}).items()
                      if k.endswith(":warm") and v)
    return f"{times} | {' '.join(ratios)} | {routes}"


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.serve:
        return serve()
    warnings.filterwarnings("ignore")
    cfgs = select_configs(args)
    if args.list:
        for c in cfgs:
            print(config_key(c))
        print(f"{len(cfgs)} configs", file=sys.stderr)
        return 0

    import mlx.core  # noqa: F401  (Metal initialization outside any timed region)

    import mlxlearn

    probe = install_route_probe()
    allowed = route_knob()
    routes = resolve_routes(args.routes, allowed)
    gate = rc.ThermalGate(max_level=args.thermal_max, limit_s=args.thermal_limit)
    meta = rc.machine_meta(mlxlearn_module=mlxlearn)
    meta["route_probe"] = probe
    meta["route_knob"] = allowed
    meta["routes"] = routes
    if meta["env_overrides"].get("MLXLEARN_RADIUS_ROUTE"):
        print("[warning] MLXLEARN_RADIUS_ROUTE is set: every mlxlearn variant is forced",
              file=sys.stderr, flush=True)
    control = None
    if args.control:
        control = ControlChild(args.control, timeout=args.control_timeout)
        meta["control"] = control.meta
        if sorted(control.meta.get("native_modules", [])) != sorted(meta.get("native_modules", [])):
            print(f"[control] native modules differ: control {control.meta.get('native_modules')}"
                  f" vs current {meta.get('native_modules')}", file=sys.stderr, flush=True)
    print(json.dumps({"meta": meta, "configs": len(cfgs)}, default=_json_default), flush=True)
    done = load_done(args.out, args.retry_errors)
    try:
        for i, cfg in enumerate(cfgs):
            key = (args.label, canonical_key(cfg))
            if key in done:
                continue
            if control is not None and not control.alive():
                print("[control] child died; restarting", file=sys.stderr, flush=True)
                try:
                    control = ControlChild(args.control, timeout=args.control_timeout)
                except ControlError as exc:
                    print(f"[control] restart failed, control disabled: {exc}", file=sys.stderr,
                          flush=True)
                    control = None
            gate.reset()
            t0 = time.time()
            try:
                rec = run_config(dict(cfg), args, args.label, control=control, routes=routes,
                                 gate=gate)
                rec["meta"] = meta
            except Exception as exc:  # keep the campaign going
                rec = {"label": args.label, "cfg": cfg, "error": repr(exc),
                       "trace": traceback.format_exc()[-2000:]}
            rec["wall_s"] = time.time() - t0
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
            pairs = rec.get("ref_summary", {}).get("pairs")
            print(f"[{i + 1}/{len(cfgs)}] {config_key(cfg)} r={rec.get('radius', 0):.4g} "
                  f"pairs={pairs} | {_brief(rec)} | {rec.get('error', '')} "
                  f"({rec['wall_s']:.0f}s)", flush=True)
    finally:
        if control is not None:
            control.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
