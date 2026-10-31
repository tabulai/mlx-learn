"""Diagnose public KNN timing stages and process-boundary effects; no source edits."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def profile(fixtures):
    import ctypes
    import resource

    import mlx.core as mx
    import numpy as np
    from threadpoolctl import threadpool_info

    import mlxlearn
    from benchmarks.knn_pipeline import source_hashes
    from mlxlearn._common.config import get_tuning, tuning_context
    from mlxlearn._kernels.distance import resolve_blocks
    from mlxlearn.neighbors import NearestNeighbors
    from mlxlearn_device._neighbors_fused import _scan_partitions, fused_neighbors

    data = dict(np.load(Path(fixtures) / "batch_50000_32_k5.npz"))
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(data["X"])
        xd = model._index_.device_index.X
        qd = mx.array(data["Q"])
        mx.eval(xd, qd)
        phases = {}
        tracking = False

        def wrap(obj, name, label):
            original = getattr(obj, name)

            def measured(*a, **kw):
                if not tracking:
                    return original(*a, **kw)
                start = time.perf_counter_ns()
                result = original(*a, **kw)
                phases[label] = phases.get(label, 0) + (time.perf_counter_ns() - start) / 1e6
                return result

            setattr(obj, name, measured)

        wrap(mx, "eval", "mx.eval")
        wrap(mx, "synchronize", "mx.synchronize")
        wrap(type(model), "_prepare_neighbor_query", "prepare_query")
        wrap(type(model._index_), "query_device", "index.query_device")
        wrap(type(model._index_), "query", "index.query")
        wrap(type(model._index_.device_index), "query", "device.query")
        qos = None
        try:
            lib = ctypes.CDLL(None)
            lib.pthread_self.restype = ctypes.c_void_p
            lib.pthread_get_qos_class_np.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            qos = lib.pthread_get_qos_class_np(lib.pthread_self(), None)
        except (AttributeError, OSError):
            pass
        result = {
            "pid": os.getpid(),
            "qos": qos,
            "nice": os.nice(0),
            "hardware": dict(mx.device_info()),
            "mlx_version": mx.__version__,
            "package": mlxlearn.__file__,
            "threadpools": threadpool_info(),
            "source_hashes": source_hashes(ROOT / "src"),
            "shape": {"X": list(data["X"].shape), "Q": list(data["Q"].shape)},
            "contiguous": {k: bool(data[k].flags.c_contiguous) for k in ("X", "Q")},
            "numpy_strides": {k: data[k].strides for k in ("X", "Q")},
            "dtype": {k: str(data[k].dtype) for k in ("X", "Q")},
            "blocks": str(resolve_blocks(256, 50000, backend="mlx", k_search=9)),
            "parts": _scan_partitions(256, 50000),
            "tuning": str(get_tuning()),
            "operations": {},
        }
        fns = {
            "primitive": lambda: fused_neighbors(xd, qd, 5),
            "device_query_validated": lambda: model._index_.device_index.query(
                qd, 5, _validated=True, _squared=True
            ),
            "index_query": lambda: model._index_.query(data["Q"], 5),
            "public_query": lambda: model.kneighbors(data["Q"]),
        }
        for name, fn in fns.items():
            samples = []
            for rep in range(33):
                mx.synchronize()
                phases.clear()
                before = resource.getrusage(resource.RUSAGE_SELF)
                tracking = True
                start = time.perf_counter_ns()
                value = fn()
                mx.eval(*(v for v in value if isinstance(v, mx.array)))
                mx.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                tracking = False
                after = resource.getrusage(resource.RUSAGE_SELF)
                idx = np.asarray(value[1])
                assert np.array_equal(idx, data["indices"])
                if rep >= 8:
                    samples.append(
                        {
                            "ms": elapsed,
                            "phases": dict(phases),
                            "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
                            "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
                            "cpu_ms": 1000
                            * (after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime),
                        }
                    )
            result["operations"][name] = {
                "median_ms": statistics.median(x["ms"] for x in samples),
                "phase_medians": {
                    p: statistics.median(x["phases"].get(p, 0) for x in samples) for p in phases
                },
                "samples": samples,
            }
        result["source_integrity_pass"] = source_hashes(ROOT / "src") == result["source_hashes"]
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/results/knn-latency-profile-v2-2026-10-28.json",
    )
    a = p.parse_args()
    if a.child:
        print(json.dumps(profile(a.child)))
        return
    from benchmarks.knn_pipeline import CASES, Worker, make_fixture

    with tempfile.TemporaryDirectory(prefix="knn-profile-v2-") as tmp:
        fixture = make_fixture(CASES[-1], 51, Path(tmp))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
        child = subprocess.run(
            [sys.executable, __file__, "--child", tmp],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        c = json.loads(child.stdout)
        # Run the exact unmodified audited worker immediately after the profile.
        w = Worker("current", ROOT / "src", Path(tmp))
        try:
            from dataclasses import asdict

            req = {"case": asdict(CASES[-1])}
            cold = w.request({"op": "cold", **req})
            block = w.request(
                {"op": "block", "operation": "kneighbors", "warmup": 8, "repeats": 25, **req}
            )
            info = w.info
        finally:
            w.close()
        direct = profile(tmp)
    result = {
        "fixture": fixture,
        "child_profile": c,
        "audited_worker": {"info": info, "cold": cold, "block": block},
        "same_process_profile": direct,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for label, r in [("child", c), ("same_process", direct)]:
        print(
            label,
            "qos",
            r["qos"],
            "parts",
            r["parts"],
            {k: round(v["median_ms"], 3) for k, v in r["operations"].items()},
        )
        print("public phases", r["operations"]["public_query"]["phase_medians"])
    print("audited_worker", block["median_ms"])
    print(a.output)


if __name__ == "__main__":
    main()
