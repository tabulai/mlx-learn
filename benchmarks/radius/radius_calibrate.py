#!/usr/bin/env python
"""Microbenchmark framework for radius-search calibration.

This is the skeleton that work item 4.3 fills in (routing calibration, step 1). It
holds the pieces that every calibration measurement and ``bench_radius.py`` share:

* Environment capture. The thermal pressure level is the notify state that
  ``notifyutil -g`` prints, read in-process so that no call follows a fork. The
  foreign CPU load comes from ``ps``. The machine, GPU generation, library
  versions, source revision and native modules come from the running interpreter.
* :class:`ThermalGate`. Before every timed call it waits while the thermal pressure
  level is above a limit, and it records how long it waited.
* :class:`Protocol`. This sets how timed calls are spaced:

  - ``gapped`` idles ``gap_s`` seconds before each timed call. The GPU clocks wind
    down during that time, which is what a sporadic caller sees. The default gap is
    0.3 s.
  - ``hot`` runs ``primers`` untimed calls of the same callable immediately before
    each timed call, with no idle between them. This is what a back-to-back
    caller sees.

* :func:`run_interleaved`. Each round visits every case once. The order rotates by
  one each round and is reversed on odd rounds, and round 0 is an untimed warm-up.
  A thermal ramp or a background process therefore degrades every case equally,
  instead of hurting whichever case happens to come last.
* Summaries, and :func:`fit_affine`, a least-squares fit with relative weights for
  the affine cost terms of the routing model.
* A registry of suites. ``round_trip``, ``host_distances`` and ``ragged`` work today.
  :data:`PLANNED_SUITES` lists the suites that work item 4.3 adds.

The callables never call ``mx.synchronize``, because it holds the GIL and can
deadlock. MLX results are forced with ``mx.eval`` on whatever the callable returns.

Usage (from the repository root)::

    PYTHONPATH=src python benchmarks/radius/radius_calibrate.py --list
    PYTHONPATH=src python benchmarks/radius/radius_calibrate.py round_trip \\
        --protocols hot gapped:0.002 gapped:0.3 --rounds 15 --out /tmp/round_trip.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "Case",
    "PLANNED_SUITES",
    "Protocol",
    "SUITES",
    "drain_idle",
    "ThermalGate",
    "fit_affine",
    "force",
    "foreign_load",
    "machine_meta",
    "parse_protocol",
    "rotation",
    "run_interleaved",
    "source_info",
    "suite",
    "summarize_samples",
    "thermal_level",
]

THERMAL_KEY = "com.apple.system.thermalpressurelevel"


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

_NOTIFY = None


def _notify_reader():
    """Read the thermal notification state in-process through libSystem's notify API.

    ``notifyutil -g`` reads the same state, but spawning it before every call
    forks this process. Afterwards the next writes to every page take a
    copy-on-write fault: a 3 ms in-process query measured 15-20% slower than the
    same code in a process that had not forked.
    """
    import ctypes

    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.notify_register_check.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.notify_register_check.restype = ctypes.c_uint32
    lib.notify_get_state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    lib.notify_get_state.restype = ctypes.c_uint32
    token = ctypes.c_int()
    if lib.notify_register_check(THERMAL_KEY.encode(), ctypes.byref(token)) != 0:
        raise OSError("notify_register_check failed")
    state = ctypes.c_uint64()

    def read() -> int:
        if lib.notify_get_state(token, ctypes.byref(state)) != 0:
            return -1
        return int(state.value)

    return read


def thermal_level() -> int:
    """macOS thermal pressure level (0 nominal); -1 when it cannot be read."""
    global _NOTIFY
    if _NOTIFY is None:
        try:
            _NOTIFY = _notify_reader()
        except Exception:
            _NOTIFY = False
    if _NOTIFY:
        return _NOTIFY()
    try:  # pragma: no cover - only without libSystem's notify API
        out = subprocess.run(["notifyutil", "-g", THERMAL_KEY], capture_output=True,
                             text=True, timeout=5).stdout.split()
        return int(out[-1])
    except Exception:
        return -1


@dataclass
class ThermalGate:
    """Wait while the thermal pressure level exceeds ``max_level``.

    ``max_level=0`` (the default) admits only nominal pressure; 9 disables the
    gate. A wait never exceeds ``limit_s`` per call; the totals are recorded so a
    campaign can tell which cells ran warm.
    """

    max_level: int = 0
    limit_s: float = 600.0
    poll_s: float = 5.0
    waits: int = 0
    wait_s: float = 0.0
    timeouts: int = 0

    def wait(self) -> float:
        if self.max_level >= 9:
            return 0.0
        level = thermal_level()
        if level <= self.max_level:
            return 0.0
        t0 = time.monotonic()
        print(f"[thermal] pressure level {level} > {self.max_level}; waiting", file=sys.stderr,
              flush=True)
        while level > self.max_level:
            if time.monotonic() - t0 >= self.limit_s:
                self.timeouts += 1
                break
            time.sleep(self.poll_s)
            level = thermal_level()
        waited = time.monotonic() - t0
        self.waits += 1
        self.wait_s += waited
        return waited

    def snapshot(self) -> dict:
        return {"waits": self.waits, "wait_s": round(self.wait_s, 3), "timeouts": self.timeouts}

    def reset(self) -> None:
        self.waits, self.wait_s, self.timeouts = 0, 0.0, 0


def foreign_load(exclude: Sequence[int] = (), top: int = 4) -> dict:
    """The busiest processes other than ``exclude`` (and this one), for the record.

    ``ps`` reports a decaying average, so this is a coarse indicator of other
    sessions benchmarking on the same machine, not a precise measurement.
    """
    skip = {os.getpid(), *exclude}
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,pcpu=,comm="], capture_output=True, text=True,
                             timeout=5).stdout.splitlines()
    except Exception:
        return {"top": [], "total_pcpu": None}
    rows = []
    for line in out:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid, pcpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid in skip:
            continue
        rows.append((pcpu, parts[2][-60:]))
    rows.sort(reverse=True)
    return {"top": [list(r) for r in rows[:top]],
            "total_pcpu": round(sum(r[0] for r in rows), 1)}


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def source_info(path: str | os.PathLike | None) -> dict:
    """Git revision and dirty flag of the tree that contains ``path``."""
    if path is None:
        return {}
    path = os.fspath(path)
    info: dict = {"path": path}
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
        if head.returncode == 0:
            info["git_head"] = head.stdout.strip()
            status = subprocess.run(["git", "-C", path, "status", "--porcelain", "--", "."],
                                    capture_output=True, text=True, timeout=30)
            info["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except Exception:
        pass
    return info


def native_modules(package_dir: str | os.PathLike) -> list[str]:
    """Compiled extension modules importable by this interpreter under ``package_dir``."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    root = Path(package_dir)
    found = []
    for path in root.rglob("*"):
        name = path.name
        for suffix in suffixes:
            # A stem that still has a dot ("x.cpython-311-darwin" under the bare
            # ".so" suffix) belongs to another interpreter.
            if name.endswith(suffix) and "." not in name[: -len(suffix)]:
                rel = path.relative_to(root.parent).with_name(name[: -len(suffix)])
                found.append(".".join(rel.parts))
                break
    return sorted(set(found))


def gpu_architecture() -> str | None:
    try:
        import mlx.core as mx
    except Exception:
        return None
    for getter in (getattr(mx, "device_info", None),
                   getattr(getattr(mx, "metal", None), "device_info", None)):
        if getter is None:
            continue
        try:
            return str(getter().get("architecture"))
        except Exception:
            continue
    return None


def machine_meta(*, mlxlearn_module=None) -> dict:
    """Machine, library and source metadata for a benchmark record."""
    import numpy as np

    meta = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0] or None,
        "machine": _sysctl("hw.model"),
        "cpu_count": os.cpu_count(),
        "p_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "e_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "device": gpu_architecture(),
        "thermal": thermal_level(),
        "env_overrides": {k: v for k, v in sorted(os.environ.items()) if k.startswith("MLXLEARN_")},
    }
    for name in ("mlx", "scipy", "sklearn"):
        try:
            module = __import__(name if name != "mlx" else "mlx.core", fromlist=["__version__"])
            meta[name] = getattr(module, "__version__", None)
        except Exception:
            meta[name] = None
    module = mlxlearn_module
    if module is None:
        try:
            import mlxlearn as module
        except Exception:
            module = None
    if module is not None:
        package_dir = os.path.dirname(os.path.abspath(module.__file__))
        meta["mlxlearn"] = getattr(module, "__version__", None)
        meta["mlxlearn_file"] = os.path.abspath(module.__file__)
        meta["mlxlearn_source"] = source_info(os.path.dirname(package_dir))
        meta["native_modules"] = native_modules(package_dir)
    return meta


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------

def drain_idle(max_s: float = 1.0, window_s: float = 0.02, busy_cores: float = 0.25) -> float:
    """Wait until this process stops burning CPU; return the seconds waited.

    After a parallel region, scikit-learn's OpenMP threads spin for about 200 ms
    (the default blocktime) on every core. A fixed 0.3 s gap therefore leaves only
    about 0.1 s of true idle after a scikit-learn call, and the clocks are still
    high. Measured on the M4 Max (3 ms query, 0.3 s gap), the next call took
    4.3 ms after ``sk_brute`` and 9.6 ms after ``mlx_patched``; with a 0.6 s gap,
    both took 8.3 ms. The fixed rotation gives every variant the same two
    neighbors, so this was a systematic bias per variant. Draining first makes
    ``gap_s`` real idle time whatever ran before. It waits while the process uses
    more than ``busy_cores`` cores, measured over ``window_s`` windows.
    """
    t0 = time.monotonic()
    while True:
        c0 = time.process_time()
        time.sleep(window_s)
        if time.process_time() - c0 < busy_cores * window_s:
            break
        if time.monotonic() - t0 >= max_s:
            break
    return time.monotonic() - t0


@dataclass(frozen=True)
class Protocol:
    """How timed calls are spaced: ``gapped`` (idle before each) or ``hot`` (primed).

    With ``drain`` (the default), each call first waits until the process is idle
    (:func:`drain_idle`). In gapped mode the gap is then true idle time, and in hot
    mode no spinning thread from the previous case competes with the primed call.
    """

    mode: str = "gapped"
    gap_s: float = 0.3
    primers: int = 1
    drain: bool = True

    def __post_init__(self):
        if self.mode not in ("gapped", "hot"):
            raise ValueError(f"protocol mode must be 'gapped' or 'hot', got {self.mode!r}")
        if self.gap_s < 0 or self.primers < 0:
            raise ValueError("gap_s and primers must be non-negative")

    @property
    def label(self) -> str:
        return f"gapped:{self.gap_s:g}" if self.mode == "gapped" else f"hot:{self.primers}"

    def idle_before(self) -> float:
        """Seconds to sleep before a timed call (0 for hot)."""
        return self.gap_s if self.mode == "gapped" else 0.0

    def primer_count(self) -> int:
        """Untimed calls immediately before a timed call (0 for gapped)."""
        return self.primers if self.mode == "hot" else 0


def parse_protocol(text: str) -> Protocol:
    """``hot``, ``hot:N`` (N primers), ``gapped`` or ``gapped:SECONDS``."""
    mode, _, value = text.partition(":")
    if mode == "hot":
        return Protocol("hot", 0.0, int(value) if value else 1)
    if mode == "gapped":
        return Protocol("gapped", float(value) if value else 0.3, 0)
    raise ValueError(f"unknown protocol {text!r}; use hot[:primers] or gapped[:seconds]")


def rotation(names: Sequence[str], r: int, *, reverse_odd: bool = True) -> list[str]:
    """Round ``r``'s visiting order: rotate by ``r``; reverse on odd rounds."""
    names = list(names)
    if not names:
        return names
    k = r % len(names)
    order = names[k:] + names[:k]
    return order[::-1] if reverse_odd and r % 2 else order


def force(result) -> None:
    """Make an MLX result concrete without ``mx.synchronize``."""
    items = result if isinstance(result, (tuple, list)) else (result,)
    arrays = [x for x in items if type(x).__module__.startswith("mlx")]
    if arrays:
        import mlx.core as mx

        mx.eval(*arrays)


# ---------------------------------------------------------------------------
# cases and the interleaved runner
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """A named timed callable.

    ``setup`` builds the state and returns the zero-argument callable to time. It
    runs once, before any timing. ``work`` is the number of units (pairs, rows or
    queries) one call processes, used to report a per-unit cost.
    """

    name: str
    setup: Callable[[], Callable[[], object]]
    tags: dict = field(default_factory=dict)
    work: float | None = None


def summarize_samples(samples: Sequence[float]) -> dict:
    if not samples:
        return {"n": 0, "median_s": None, "min_s": None, "max_s": None, "iqr_s": None}
    s = sorted(samples)
    q = statistics.quantiles(s, n=4) if len(s) >= 2 else [s[0], s[0], s[0]]
    return {"n": len(s), "median_s": statistics.median(s), "min_s": s[0], "max_s": s[-1],
            "iqr_s": q[2] - q[0]}


def run_interleaved(cases: Sequence[Case], protocol: Protocol, *, rounds: int = 7,
                    warmup_rounds: int = 1, gate: ThermalGate | None = None) -> list[dict]:
    """Time every case under ``protocol`` in interleaved rounds.

    Returns one result per case, with the raw samples, a summary and the per-unit
    cost in nanoseconds when the case declares its ``work``.
    """
    gate = gate if gate is not None else ThermalGate(max_level=9)
    callables = {c.name: c.setup() for c in cases}
    samples: dict[str, list[float]] = {c.name: [] for c in cases}
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")
    for r in range(warmup_rounds + rounds):
        for name in rotation(names, r):
            fn = callables[name]
            gate.wait()
            if protocol.drain:
                drain_idle()
            idle = protocol.idle_before()
            if idle:
                time.sleep(idle)
            for _ in range(protocol.primer_count()):
                force(fn())
            t0 = time.perf_counter()
            force(fn())
            elapsed = time.perf_counter() - t0
            if r >= warmup_rounds:
                samples[name].append(elapsed)
    results = []
    for case in cases:
        summary = summarize_samples(samples[case.name])
        if case.work and summary["median_s"] is not None:
            summary["per_unit_ns"] = summary["median_s"] / case.work * 1e9
        results.append({"case": case.name, "tags": case.tags, "work": case.work,
                        "protocol": asdict(protocol), "summary": summary,
                        "samples": samples[case.name]})
    del callables
    gc.collect()
    return results


def fit_affine(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares ``y ~ a + b * x`` with relative residuals ``(fit - y) / y``.

    Cost-model terms are affine in the work (``c0 + W * c``) and are measured over
    orders of magnitude of ``W``. Relative weights keep the small calls from being
    ignored.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("need at least two (x, y) points of equal length")
    if np.any(y <= 0):
        raise ValueError("times must be positive")
    w = 1.0 / y
    A = np.stack([w, w * x], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, w * y, rcond=None)
    return float(a), float(b)


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------

class SuiteUnavailable(RuntimeError):
    """The installed mlxlearn (or the machine) lacks what a suite measures."""


SUITES: dict[str, Callable[[argparse.Namespace], list[Case]]] = {}

#: Suites that work item 4.3 adds (routing calibration, step 1).
PLANNED_SUITES: dict[str, str] = {
    "gpu_direct": "lane-per-query scan T(W): D in {2,3,4,6,8,12,16}, runtime d in {24,32,64}, "
                  "W in {1e8,1e9,1e10}, density m in {1,10,100,1000}; cap/overflow on "
                  "label-sorted data",
    "gpu_gram": "binary32 GEMM + ordered compaction at d in {17..784}; DS cost per candidate",
    "finisher": "slab finisher cost per pair for each mode bit set",
    "tree_build": "native tree build for n in {1e4,1e5,1e6}, d in {2,3,5,8,12,16}",
    "tree_pilot": "c_v(d), c_h, P_eff(threads) from pilot counters on uniform/gauss/blobs",
    "brute": "native brute c_bd and the fan-out threshold",
    "output": "CSR fill, per-row sort and object-array costs",
    "ckdtree": "cKDTree v2 cost per pair and per query (NumPy installs)",
    "sk_kdtree": "scikit-learn KD-tree tau_sk(d, n, m), fixed overheads, fit time, n_jobs=-1 "
                 "efficiency and joblib floor",
    "sk_brute": "scikit-learn brute GEMM rate F64(d), per-pair costs, RadiusNeighborsClassMode",
}


def suite(name: str):
    def deco(fn):
        SUITES[name] = fn
        return fn
    return deco


@suite("round_trip")
def _round_trip(args) -> list[Case]:
    """One tiny GPU kernel per call: the dispatch and completion round trip."""
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - machines without MLX
        raise SuiteUnavailable(f"mlx is not importable: {exc!r}") from None

    def setup():
        # Eight elements: kernel inputs shorter than 8 change the generated source
        # before MLX 0.32.
        a = mx.zeros((8,), dtype=mx.float32, stream=mx.gpu)
        mx.eval(a)
        return lambda: mx.add(a, 1.0, stream=mx.gpu)

    return [Case("gpu_round_trip", setup, tags={"elements": 8})]


@suite("host_distances")
def _host_distances(args) -> list[Case]:
    """Canonical host float64 distances (radius_host.host_distances), per pair."""
    try:
        from mlxlearn._kernels.radius_host import host_distances
    except Exception as exc:
        raise SuiteUnavailable(f"radius_host is not importable: {exc!r}") from None
    import numpy as np

    cases = []
    pairs = int(getattr(args, "pairs", 0) or 1 << 18)
    for d in (3, 16, 64):
        def setup(d=d):
            rng = np.random.default_rng(d)
            train = rng.standard_normal((100_000, d), dtype=np.float32)
            queries = rng.standard_normal((1_000, d), dtype=np.float32)
            rows = rng.integers(0, len(queries), pairs)
            cols = np.sort(rng.integers(0, len(train), pairs))
            return lambda: host_distances(train, queries, rows, cols, threads=1)

        cases.append(Case(f"host_distances_d{d}", setup, tags={"d": d, "pairs": pairs},
                          work=pairs))
    return cases


@suite("ragged")
def _ragged(args) -> list[Case]:
    """Object arrays of CSR row views (radius_result.csr_to_objects), per row."""
    try:
        from mlxlearn._kernels.radius_result import csr_to_objects
    except Exception as exc:
        raise SuiteUnavailable(f"radius_result is not importable: {exc!r}") from None
    import numpy as np

    cases = []
    for rows in (10_000, 100_000):
        def setup(rows=rows):
            lengths = np.random.default_rng(rows).integers(0, 20, rows)
            indptr = np.concatenate(([0], np.cumsum(lengths)))
            values = np.arange(indptr[-1], dtype=np.int64)
            return lambda: csr_to_objects(indptr, values)

        cases.append(Case(f"csr_to_objects_{rows}", setup, tags={"rows": rows}, work=rows))
    return cases


def run_suite(name: str, protocols: Sequence[Protocol], args) -> dict:
    cases = SUITES[name](args)
    gate = ThermalGate(max_level=args.thermal_max, limit_s=args.thermal_limit)
    results = []
    for protocol in protocols:
        results.extend(run_interleaved(cases, protocol, rounds=args.rounds,
                                       warmup_rounds=args.warmup, gate=gate))
    return {"suite": name, "meta": machine_meta(), "thermal_gate": gate.snapshot(),
            "load_end": foreign_load(), "results": results}


def _format(result: dict) -> str:
    s = result["summary"]
    unit = f" {s['per_unit_ns']:.2f} ns/unit" if s.get("per_unit_ns") is not None else ""
    proto = result["protocol"]
    label = (f"gapped:{proto['gap_s']:g}" if proto["mode"] == "gapped"
             else f"hot:{proto['primers']}")
    return (f"{result['case']:28s} {label:14s} median {s['median_s'] * 1e3:9.4f} ms "
            f"min {s['min_s'] * 1e3:9.4f} ms iqr {s['iqr_s'] * 1e3:8.4f} ms{unit}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("suite", nargs="?", help="suite name (see --list)")
    ap.add_argument("--list", action="store_true", help="list available and planned suites")
    ap.add_argument("--protocols", nargs="+", default=["hot", "gapped:0.3"],
                    help="hot[:primers] and/or gapped[:seconds]; each is run in turn")
    ap.add_argument("--rounds", type=int, default=7, help="timed rounds per protocol")
    ap.add_argument("--warmup", type=int, default=1, help="untimed rounds before timing")
    ap.add_argument("--pairs", type=int, default=0, help="pairs per call (host_distances)")
    ap.add_argument("--thermal-max", type=int, default=0,
                    help="wait while the thermal pressure level exceeds this (9 disables)")
    ap.add_argument("--thermal-limit", type=float, default=600.0,
                    help="longest single thermal wait in seconds")
    ap.add_argument("--out", default=None, help="write the JSON result here")
    args = ap.parse_args(argv)
    args.protocol_objects = [parse_protocol(p) for p in args.protocols]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list or not args.suite:
        for name, fn in sorted(SUITES.items()):
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            print(f"{name:16s} {doc}")
        for name, text in sorted(PLANNED_SUITES.items()):
            print(f"{name:16s} (planned, WI 4.3) {text}")
        return 0
    if args.suite not in SUITES:
        print(f"unknown suite {args.suite!r}; see --list", file=sys.stderr)
        return 2
    try:
        record = run_suite(args.suite, args.protocol_objects, args)
    except SuiteUnavailable as exc:
        print(f"suite {args.suite} unavailable: {exc}", file=sys.stderr)
        return 3
    for result in record["results"]:
        print(_format(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
