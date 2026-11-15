"""CPU-only harness/controller/report checks, without MLX imports or datasets."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h100  # noqa: E402
import h100_report as report  # noqa: E402
from test_comparison_report import row as old_row  # noqa: E402


@pytest.fixture
def config(tmp_path):
    return {"dataset": "example", "mode": "optimized", "seed": 10, "threads": 8,
            "memory": True, "memory_interval_seconds": .01,
            "output": str(tmp_path / "result.json"), "worker_output": str(tmp_path / "worker.tmp"),
            "expected_source_sha256": {"h100.py": "source"}, "expected_versions": {"python": "version"},
            "expected_host": {"cpu": "test"}}


@pytest.fixture
def fake_worker(monkeypatch):
    import memory
    state = {"start_error": False, "stop_error": False, "complete": True, "starts": 0, "stops": 0}
    processes = []
    original = subprocess.Popen

    class Sampler:
        def __init__(self, pid, interval_seconds):
            self.pid = pid

        def start(self):
            state["starts"] += 1
            if state["start_error"]:
                raise OSError("start failed")
            return self

        def stop(self):
            state["stops"] += 1
            if state["stop_error"]:
                raise RuntimeError("stop failed")
            return {"complete": state["complete"]}

    monkeypatch.setattr(memory, "MemorySampler", Sampler)

    def install(program):
        def spawn(command, **kwargs):
            p = original([sys.executable, "-c", program, command[-1]], **kwargs)
            processes.append(p)
            return p
        monkeypatch.setattr(h100.subprocess, "Popen", spawn)
        return state, processes
    return install


def program(seconds=3., count=1, mutation=""):
    return f'''
import json, sys, time
from pathlib import Path
config=json.loads(sys.argv[1])
for i in range({count}):
 print(json.dumps({{'phase':'FIT_READY','call':i}}),flush=True)
 assert sys.stdin.readline().strip()=='GO'
 print(json.dumps({{'phase':'FIT_DONE','call':i,'train_seconds':{seconds}}}),flush=True)
 assert sys.stdin.readline().strip()=='STOP_ACK'
result={{'status':'completed','kind':'h100_protocol_metal','config':config,
 'source_sha256':config['expected_source_sha256'],'versions':config['expected_versions'],
 'hardware':config['expected_host'],'train_seconds':{seconds},'train_seconds_samples':[{seconds}]*{count},'test_mse':1.}}
{mutation}
Path(config['worker_output']).write_text(json.dumps(result))
'''


@pytest.mark.parametrize("seconds,count", [(3., 1), (.1, 3), (2., 1)])
def test_adaptive_controller_publishes_all_memory_windows(config, fake_worker, seconds, count):
    state, processes = fake_worker(program(seconds, count))
    h100.measured_run(config, 2)
    result = json.loads(Path(config["output"]).read_text())
    assert len(result["memory_samples"]) == count
    assert state["starts"] == state["stops"] == count
    assert processes[0].poll() == 0


@pytest.mark.parametrize("change", ["crash", "timeout", "start", "stop", "incomplete", "provenance", "wrong_count"])
def test_controller_failures_never_publish(config, fake_worker, change):
    script = program()
    if change == "crash":
        script = "import sys;sys.exit(7)"
    elif change == "timeout":
        script = "import time;time.sleep(30)"
    elif change == "provenance":
        script = program(mutation="result['versions']={'wrong':'version'}")
    elif change == "wrong_count":
        script = program(.1, 1)
    state, processes = fake_worker(script)
    if change in ("start", "stop"):
        state[f"{change}_error"] = True
    if change == "incomplete":
        state["complete"] = False
    Path(config["worker_output"]).write_text('{"status":"completed"}')
    with pytest.raises((OSError, RuntimeError, TimeoutError)):
        h100.measured_run(config, .15 if change == "timeout" else 2)
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("phase", ["register", "close"])
def test_selector_failures_always_cleanup_child(config, fake_worker, monkeypatch, phase):
    _, processes = fake_worker(program())
    original = h100.selectors.DefaultSelector

    class Selector:
        def __init__(self):
            self.inner = original()

        def register(self, *args):
            if phase == "register":
                raise RuntimeError("register failed")
            return self.inner.register(*args)

        def select(self, *args):
            return self.inner.select(*args)

        def close(self):
            self.inner.close()
            if phase == "close":
                raise RuntimeError("close failed")

    monkeypatch.setattr(h100.selectors, "DefaultSelector", Selector)
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        h100.measured_run(config, 2)
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not Path(config["output"]).exists()


def test_dataset_fingerprint_not_only_shape_is_required():
    identity = {"train_sample_sha256": "train", "test_sha256": "test", "train_rows_used": 10,
                "test_rows_used": 3, "dimension": 4}
    c = {"rows": None, "protocol_config": {"n": 10, "d": 4, "dataset_identity": identity}}
    h100.validate_dataset(identity, 10, 4, c)
    with pytest.raises(RuntimeError, match="identity"):
        h100.validate_dataset({**identity, "train_sample_sha256": "changed"}, 10, 4, c)


def row(mode="optimized", seed=10):
    r = old_row(seed=seed)
    r["kind"] = report.KIND
    r["dataset"].update(train_rows_used=r["n"], test_rows_used=10000, dimension=r["d"])
    c = r["config"]
    c.update(mode=mode, assignment="gemm" if mode == "baseline" else "auto", warmup=True, precision="bfloat16",
             capture_assignments=True, k_override=None, epochs_override=None, memory=True,
             protocol_config={"dataset": c["dataset"], "n": r["n"], "d": r["d"], "k": r["k"],
                              "epochs": 3, "seed_ids": [10, 11], "reported_h100_emeans_seconds": None,
                              "dataset_identity": r["dataset"].copy()})
    r["train_seconds"] = 1 if mode == "optimized" else 3
    r["train_seconds_samples"] = [1, 1, 1] if mode == "optimized" else [3]
    for key in ("benchmarks/e_means/h100.py", "benchmarks/e_means/h100_protocol.json", "src/mlxlearn/_kernels/e_means_bf16.py"):
        r["source_sha256"][key] = "source"
    c["expected_source_sha256"] = r["source_sha256"].copy()
    r["recipe_resolved"].update(precision="bfloat16", statistics_representation="bfloat16", assignment_centering="none", finalization="last_epoch", assignment=c["assignment"])
    r["quality_fit_index"] = 0
    r["captured_assignments"] = {"shape": [r["n"]], "dtype": "mlx.core.int64"}
    r["iteration_history"] = [{"moving_objective": 1}]
    r["memory_samples"] = [copy.deepcopy(r["memory"]) for _ in r["train_seconds_samples"]]
    r["mlx_memory_samples"] = [r["mlx_memory"] for _ in r["train_seconds_samples"]]
    r["_path"] = Path(f"/repo/results/{mode}-s{seed}.json")
    return r


def test_report_extra_optimized_seeds_do_not_inflate_paired_sample():
    valid, pairs, excluded = report.validate_and_pair([row(), row("baseline"), row(seed=11)])
    assert not excluded
    assert len(pairs) == 1
    grouped = report.groups(valid, pairs)
    assert grouped[0]["optimized_seeds"] == [10, 11]
    assert len(grouped[0]["pairs"]) == 1
    assert grouped[0]["paired_speedup"] == 3
    assert grouped[0]["h100_context_ratio"] is None
    assert grouped[0]["optimized_std_seconds"] == 0


def test_time_quartiles_and_single_seed_dispersion():
    assert report.percentile([1,2,3,4], .25) == 1.75
    assert report.percentile([1,2,3,4], .75) == 3.25
    valid, pairs, _ = report.validate_and_pair([row()])
    assert report.groups(valid, pairs)[0]["optimized_std_seconds"] is None


@pytest.mark.parametrize("field", ["initial_centroids_sha256", "dataset", "source_sha256", "hardware"])
def test_report_never_pairs_mismatched_evidence(field):
    a, b = row(), row("baseline")
    if field == "initial_centroids_sha256":
        b[field] = "other"
    else:
        b[field] = {**b[field], "extra": "different"}
        if field in ("source_sha256", "hardware"):
            b["config"]["expected_source_sha256" if field == "source_sha256" else "expected_host"] = b[field].copy()
    assert not report.validate_and_pair([a,b])[1]


def test_report_rejects_smoke_and_wrong_recipe_and_missing_kernel_hash():
    for mutation in (lambda r: r["config"].update(rows=1000),
                     lambda r: r["recipe_resolved"].update(statistics_representation="float32"),
                     lambda r: r["source_sha256"].pop("src/mlxlearn/_kernels/e_means_bf16.py")):
        r = row()
        mutation(r)
        assert report.problem(r)


def test_report_rejects_off_manifest_seed_and_same_shape_data_replacement():
    assert "seed" in report.problem(row(seed=999))
    r = row()
    r["dataset"]["test_sha256"] = "different"
    assert "fingerprint" in report.problem(r)


def test_report_incomplete_memory_suppressed_and_h100_ratio_contextual():
    r = row()
    r["memory_samples"][1] = None
    assert report.memory_peak(r) is None
    r["config"]["protocol_config"]["reported_h100_emeans_seconds"] = .1
    valid, pairs, excluded = report.validate_and_pair([r])
    grouped = report.groups(valid,pairs)
    assert grouped[0]["h100_context_ratio"] == 10
    text = report.make_report(valid,pairs,grouped,excluded,Path("/repo/report.md"),Path("/repo/summary.csv"),Path("/repo/manifest.json"))
    assert "descriptive context, not a matched hardware speedup" in text
    assert "No absolute H100 e-means time" in text


def test_imagenet_labels_distinguish_core_and_vibe():
    assert report.label("imagenet-clip-512-normalized") == "ImageNet-CLIP (1M)"
    assert report.label("vibe-imagenet") == "ImageNet-CLIP (VIBE)"


def test_memory_appendix_aggregates_all_seeds_using_per_seed_peaks():
    a, b, c = row(), row(seed=11), row("baseline")
    b["memory_samples"][0]["peak_physical_footprint_bytes"] += 4 * 2**30
    b["memory_samples"][0]["physical_footprint_peak_increase_bytes"] += 4 * 2**30
    valid, pairs, _ = report.validate_and_pair([a,b,c])
    groups = report.memory_groups(valid)
    optimized = next(g for g in groups if g["mode"] == "optimized")
    assert len(groups) == 2
    assert optimized["seed_count"] == 2
    assert optimized["median_peak_footprint_bytes"] == 5 * 2**30
    assert len(optimized["raw_evidence_paths"]) == 2
    assert len(pairs) == 1


def test_memory_appendix_incomplete_samples_report_actual_counts():
    a, b = row(), row(seed=11)
    b["memory_samples"][0] = None
    groups = report.memory_groups([a,b])
    assert groups[0]["seed_count"] == 2
    assert groups[0]["complete_memory_seeds"] == 1
    assert groups[0]["complete_allocator_seeds"] == 2
    assert groups[0]["median_peak_footprint_bytes"] == 3 * 2**30


def test_allocator_tradeoff_uses_paired_seed_maxima_and_separate_footprint():
    optimized, baseline, unpaired = row(), row("baseline"), row(seed=11)
    optimized["mlx_memory_samples"] = [{"peak_active_bytes": value * 2**30} for value in (1, 3, 1)]
    baseline["mlx_memory_samples"] = [{"peak_active_bytes": 2 * 2**30}]
    unpaired["mlx_memory_samples"] = [{"peak_active_bytes": 100 * 2**30}] * 3
    for sample in optimized["memory_samples"]:
        sample["peak_physical_footprint_bytes"] -= 2**30
        sample["physical_footprint_peak_increase_bytes"] -= 2**30
    valid, pairs, excluded = report.validate_and_pair([optimized, baseline, unpaired])
    grouped = report.groups(valid, pairs)
    tradeoff, = report.paired_allocator_increases(grouped)
    assert tradeoff["paired_seeds"] == 1
    assert tradeoff["baseline_active_bytes"] == 2 * 2**30
    assert tradeoff["optimized_active_bytes"] == 3 * 2**30
    assert tradeoff["active_increase_pct"] == 50
    assert tradeoff["footprint_change_pct"] == pytest.approx(-100 / 3)
    text = report.make_report(valid, pairs, grouped, excluded, Path("/repo/report.md"),
                              Path("/repo/summary.csv"), Path("/repo/manifest.json"))
    assert "Memory metrics do not all improve" in text
    assert "+50.0%; 1 paired seeds" in text
    assert "(-33.3%)" in text


def test_allocator_tradeoff_excludes_incomplete_pairs_and_does_not_infer_footprint():
    a, b, c, d = row(), row("baseline"), row(seed=11), row("baseline", seed=11)
    a["mlx_memory_samples"] = [{"peak_active_bytes": 3}] * 3
    b["mlx_memory_samples"] = [{"peak_active_bytes": 2}]
    c["mlx_memory_samples"] = [{"peak_active_bytes": 100}] * 3
    d["mlx_memory_samples"] = []
    a["memory_samples"][0] = None
    valid, pairs, _ = report.validate_and_pair([a, b, c, d])
    tradeoff, = report.paired_allocator_increases(report.groups(valid, pairs))
    assert tradeoff["paired_seeds"] == 1
    assert tradeoff["active_increase_pct"] == 50
    assert tradeoff["footprint_change_pct"] is None


@pytest.mark.parametrize("active", [1, 2])
def test_allocator_tradeoff_omits_equal_and_lower_active_peaks(active):
    a, b = row(), row("baseline")
    a["mlx_memory_samples"] = [{"peak_active_bytes": active}] * 3
    b["mlx_memory_samples"] = [{"peak_active_bytes": 2}]
    valid, pairs, _ = report.validate_and_pair([a, b])
    assert report.paired_allocator_increases(report.groups(valid, pairs)) == []


def test_quality_figure_keeps_zero_median_dot_and_observed_min_max_without_plotting():
    class Axis:
        def __init__(self):
            self.points = []

        def errorbar(self, center, position, **kwargs):
            self.points.append((center, position, kwargs))

        def axvline(self, value, **kwargs):
            self.zero_line = value

        def set_xlabel(self, text):
            self.label = text

    axis = Axis()
    pairs = [{"quality_delta_pct": value} for value in (-.02, 0, .03)]
    report.draw_quality(axis, [{"pairs": pairs}])
    center, position, details = axis.points[0]
    assert center == position == 0
    assert details["fmt"] == "o"
    assert details["xerr"] == [[.02], [.03]]
    assert axis.zero_line == 0
    assert "dot: median; whisker: observed min–max" in axis.label


def test_quality_summary_omits_undefined_changes():
    assert report.paired_quality_summary([{"quality_delta_pct": None}]) is None
    assert report.paired_quality_summary([{"quality_delta_pct": float("inf")}]) is None
    assert report.paired_quality_summary([{"quality_delta_pct": None}, {"quality_delta_pct": 0}]) == {
        "median": 0, "min": 0, "max": 0, "count": 1}


def test_report_global_quality_range_describes_seed_pairs_not_medians():
    a, b, c, d = row(), row("baseline"), row(seed=11), row("baseline", seed=11)
    a["test_mse"], b["test_mse"] = 99.99, 100
    c["test_mse"], d["test_mse"] = 100.02, 100
    valid, pairs, excluded = report.validate_and_pair([a, b, c, d])
    grouped = report.groups(valid, pairs)
    text = report.make_report(valid, pairs, grouped, excluded, Path("/repo/report.md"),
                              Path("/repo/summary.csv"), Path("/repo/manifest.json"))
    assert "Across 2 paired dataset/seed comparisons" in text
    assert "ranges from -0.01% to +0.02%" in text
    assert "individual seed pairs; the table reports per-dataset medians" in text
