"""Exercise worker import/timing order with fake public libraries, without workloads."""

import builtins
import contextlib
import io
import json
import runpy
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("worker", ["knn_cold_worker.py", "knn_first_query_worker.py"])
@pytest.mark.parametrize("variant", ["current", "previous_auto", "sklearn_default"])
def test_public_imports_are_timed_without_stock_neighbor_or_query_prewarm(worker, variant, monkeypatch, tmp_path):
    events, messages = [], []
    clock_ns = [0]
    public_name = "sklearn.neighbors" if variant.startswith("sklearn") else "mlxlearn.neighbors"

    def tick():
        clock_ns[0] += 1000
        return clock_ns[0]
    source_root = tmp_path / "source"
    fixture_path = tmp_path / "fixture.npz"
    fixture_path.write_bytes(b"synthetic protocol-only fixture")
    data = {"X": object(), "Q": object()}

    class Fixture:
        files = list(data)

        def __enter__(self):
            events.append("fixture_load")
            return self

        def __exit__(self, *_):
            return False

        def __getitem__(self, key):
            return data[key]

    class Estimator:
        def __init__(self, **_):
            events.append("construct")

        def fit(self, _):
            events.append("fit")
            return self

        def kneighbors(self, _):
            events.append("query")
            return [[0.0]], [[0]]

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        return value

    def audit(*_):
        events.append("audit")
        return {"pass": True}

    libraries = {
        "numpy": module("numpy", load=lambda _: Fixture(), __version__="fake"),
        "mlxlearn": module("mlxlearn", __file__=str(source_root / "mlxlearn/__init__.py")),
        "mlxlearn._common.config": module("mlxlearn._common.config",
            config_context=lambda **_: contextlib.nullcontext(), tuning_context=lambda **_: contextlib.nullcontext()),
        "mlxlearn.neighbors": module("mlxlearn.neighbors", NearestNeighbors=Estimator),
        "sklearn.neighbors": module("sklearn.neighbors", NearestNeighbors=Estimator),
        "sklearn": module("sklearn", __version__="fake"),
        "scipy": module("scipy", __version__="fake"),
        "threadpoolctl": module("threadpoolctl", threadpool_info=lambda: []),
        "benchmarks.knn_broad_worker": module("benchmarks.knn_broad_worker", audit_search=audit,
            backend_metadata=lambda *_, **__: {}, native_cpu_metadata=lambda _: {"available": False}),
        "benchmarks.knn_pipeline": module("benchmarks.knn_pipeline", sha256=lambda _: "fake",
            source_hashes=lambda _: {}),
    }
    original_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        if name in libraries:
            events.append("import:" + name)
            if name == public_name:
                clock_ns[0] += 100_000_000  # Synthetic 100 ms public import, never a sleep.
            return libraries[name]
        if name == "mlx.core":
            raise AssertionError("Harness imported GPU runtime before public estimator use")
        return original_import(name, *args, **kwargs)

    def tracked_print(value, **_):
        message = json.loads(value)
        messages.append(message)
        events.append("emit:" + message.get("kind", "sample"))

    command = {"op": "fresh_model", "case": {"name": "fixture", "k": 1}, "round": 0}
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", tracked_import)
        patch.setattr(builtins, "print", tracked_print)
        patch.setattr(sys, "stdin", io.StringIO(json.dumps(command) + "\n"))
        patch.setattr(time, "perf_counter_ns", tick)
        namespace = runpy.run_path(str(Path(__file__).with_name(worker)), run_name="protocol_test")
        namespace["run"](SimpleNamespace(variant=variant, source_root=source_root, fixture=fixture_path, fixtures=tmp_path, k=1))
    candidate_import = "import:" + public_name
    assert events.index(candidate_import) < events.index("construct") < events.index("fit") < events.index("query")
    assert events.count("construct") == events.count("fit") == 1
    if not variant.startswith("sklearn"):
        assert "import:sklearn.neighbors" not in events
    if worker == "knn_cold_worker.py":
        assert events.count("query") == 1
        assert events.index("query") < events.index("emit:result_ready") < events.index("audit")
        assert events.index("emit:result_ready") < events.index("import:sklearn")
        assert messages[0]["timings"]["estimator_to_result_ms"] >= messages[0]["timings"]["fit_query_ms"]
        assert messages[0]["timings"]["estimator_setup_ms"] >= 100
        assert "sklearn.neighbors" in messages[0]["loaded_modules"]["before_fit"]
    else:
        assert events.count("query") == 2
        assert events.index("emit:initialized") < events.index("construct")
        assert max(i for i, event in enumerate(events) if event == "query") < events.index("audit")
        assert messages[0]["setup"]["query_prewarm_count"] == 0
        assert messages[0]["setup"]["estimator_import_setup_ms"] >= 100
        assert "sklearn.neighbors" in messages[0]["setup"]["loaded_modules_after_imports"]
