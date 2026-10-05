"""benchmarks/: the statistics, the gate's logic, the profiles, and one real run."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import _harness, gate, run
from benchmarks._harness import Comparison, _summarize, compare, interquartile_range, measure

from mlxlearn._common.config import reset_config

# --------------------------------------------------------------------- statistics


def test_one_outlier_does_not_set_the_spread():
    # The old formula returned max - min for n <= 7: 2.0 here.
    assert _summarize([1, 1, 1, 1, 1, 1, 3], "x").iqr_s == 0.0


@pytest.mark.parametrize("n", [2, 5, 7, 8, 11, 25])
def test_the_iqr_is_the_interpolated_quartile_range(n):
    samples = list(np.random.default_rng(n).lognormal(size=n))
    q1, q3 = np.percentile(samples, [25, 75])
    assert interquartile_range(samples) == pytest.approx(q3 - q1)


def test_measure_summarizes_like_compare():
    measurement = measure(lambda: None, label="noop", warmup=0, iterations=6)
    assert measurement.iterations == 6
    assert 0.0 <= measurement.iqr_s <= measurement.max_s - measurement.min_s
    # The cold-start measurement was documented but never wired up; the claim went
    # with the field rather than staying as a promise nothing kept.
    assert not hasattr(measurement, "cold_start_s")


def _comparison(mlx, sk):
    return Comparison(
        label="c", shape=(1, 1), mlxlearn=_summarize(mlx, "m"), sklearn=_summarize(sk, "s")
    )


def test_a_single_outlier_cannot_hide_a_2x_regression():
    rng = np.random.default_rng(0)
    sk = list(1.0 + 0.01 * rng.normal(size=7))
    sk[3] = 5.0  # one garbage-collection pause on the scikit-learn side
    mlx = list(2.0 + 0.01 * rng.normal(size=7))
    comparison = _comparison(mlx, sk)
    assert comparison.speedup < 0.55
    assert comparison.significant


def test_equal_distributions_are_within_noise():
    rng = np.random.default_rng(1)
    comparison = _comparison(list(rng.lognormal(0, 0.1, 7)), list(rng.lognormal(0, 0.1, 7)))
    assert not comparison.significant


def test_too_few_samples_never_claim_significance():
    assert not _comparison([2.0, 2.01, 2.02, 2.03], [1.0, 1.01, 1.02, 1.03]).significant


class _Host:
    """A fake clock and collector where a call made straight after a collection is
    slower — the cold caches that made compare() read identical work as a loss."""

    def __init__(self):
        self.now, self.cold, self.enabled, self.log = 0.0, False, True, []
        self.gc = SimpleNamespace(collect=self._collect, disable=self._disable, enable=self._enable)
        self.time = SimpleNamespace(perf_counter=lambda: self.now)

    def _collect(self):
        self.log.append("collect")
        self.cold = True

    def _disable(self):
        self.enabled = False

    def _enable(self):
        self.enabled = True

    def work(self, side):
        def call():
            self.log.append(f"{side}{'' if self.enabled else ' (gc off)'}")
            self.now += 1.25 if self.cold else 1.0
            self.cold = False

        return call

    def sync(self):
        self.log.append("sync")


@pytest.fixture
def host(monkeypatch):
    fake = _Host()
    monkeypatch.setattr(_harness, "gc", fake.gc)
    monkeypatch.setattr(_harness, "time", fake.time)
    monkeypatch.setattr(_harness, "_sync", fake.sync)
    return fake


def test_compare_alternates_the_first_side_and_times_both_alike(host):
    compare(host.work("mlx"), host.work("sk"), label="c", shape=(1, 1), warmup=0, iterations=4)
    mlx = ["collect", "mlx (gc off)", "sync"]
    sk = ["collect", "sk (gc off)", "sync"]
    warmup = ["sync"]
    assert host.log == [*warmup, *mlx, *sk, *sk, *mlx, *mlx, *sk, *sk, *mlx]


def test_identical_work_compares_as_identical(host):
    work = host.work("same")
    comparison = compare(work, work, label="c", shape=(1, 1), warmup=0, iterations=7)
    assert comparison.speedup == 1.0
    assert not comparison.significant


# --------------------------------------------------------------------------- gate


def _result(label="knn/kneighbors/4000x32", *, speedup, significant=True, n=4000, **extra):
    algorithm, operation = label.split("/")[:2]
    return {
        "label": label,
        "algorithm": algorithm,
        "operation": operation,
        "shape": [n, 32],
        "speedup": speedup,
        "speedup_best": speedup,
        "significant": significant,
        **extra,
    }


def test_a_self_derived_crossover_no_longer_excuses_the_mlx_path():
    # MLX regressed at 500-8000 rows, which moved the crossover measured in the same
    # run up to 32000 — the old gate then called every regression "by design".
    results = [
        _result(f"knn/kneighbors/{n}x32", speedup=0.45, n=n, backend="mlx")
        for n in (500, 2_000, 8_000)
    ] + [_result("knn/kneighbors/32000x32", speedup=3.0, n=32_000, backend="mlx")]
    payload = {
        "layer": "patched",
        "results": results,
        "crossovers": {"knn/kneighbors": {"crossover_n_samples": 32_000}},
    }
    failures, notes = gate.evaluate(payload)
    assert len(failures) == 3
    assert all("Tuning" in failure for failure in failures)
    assert not any("by design" in note for note in notes)
    assert any("measured crossover 32000" in note for note in notes)


def test_patched_handoff_overhead_is_gated():
    payload = {
        "layer": "patched",
        "results": [_result("logreg/fit/5000x32", speedup=0.6, backend="sklearn")],
    }
    failures, _ = gate.evaluate(payload)
    assert len(failures) == 1 and "overhead" in failures[0]


def test_the_layer1_cpu_path_is_not_the_claim():
    payload = {
        "layer": "direct",
        "results": [_result("svc/fit/1000x32", speedup=0.2, backend="cpu")],
    }
    assert gate.evaluate(payload)[0] == []


def test_forced_kernels_below_the_threshold_are_reported_not_gated():
    payload = {
        "layer": "direct",
        "force_backend": "mlx",
        "results": [
            _result("logreg/fit/1000x16", speedup=0.03, backend="mlx", dispatch="sklearn"),
            _result("svc/fit/1000x512", speedup=2.0, backend="mlx", dispatch="sklearn"),
        ],
    }
    failures, notes = gate.evaluate(payload)
    assert failures == []
    assert any("too conservative" in note for note in notes)


def test_forced_kernels_the_threshold_selects_are_gated():
    payload = {
        "layer": "direct",
        "force_backend": "mlx",
        "results": [_result("logreg/fit/60000x1024", speedup=0.5, backend="mlx", dispatch="mlx")],
    }
    assert len(gate.evaluate(payload)[0]) == 1


def test_accepted_regressions_and_noise_pass():
    payload = {
        "results": [
            _result("knn/fit/4000x32", speedup=0.2, backend="mlx"),
            _result("svc/fit/4000x256", speedup=0.7, significant=False, backend="mlx"),
            _result("svc/fit/8000x256", speedup=0.97, backend="mlx"),  # inside TOLERANCE
        ]
    }
    assert gate.evaluate(payload)[0] == []


def test_a_schema_1_payload_without_backends_is_gated():
    payload = {"results": [_result(speedup=0.5)], "crossovers": {}}
    assert len(gate.evaluate(payload)[0]) == 1


def _rechecked(speedup, significant):
    return {"speedup": speedup, "significant": significant, "mlxlearn": {"iterations": 21}}


def test_a_loss_that_does_not_reproduce_is_not_a_failure():
    # About one case in a hundred clears z=3 on noise alone; a nightly run gates more
    # than a hundred, so without a re-measurement most nights failed.
    flaky = _result(
        "logreg/fit/5000x32", speedup=0.8, backend="sklearn", confirmation=_rechecked(1.01, False)
    )
    failures, notes = gate.evaluate({"results": [flaky]})
    assert failures == []
    assert any("did not reproduce (1.01x over 21 iterations)" in note for note in notes)


def test_a_loss_that_reproduces_fails():
    real = _result(
        "logreg/fit/5000x32", speedup=0.8, backend="sklearn", confirmation=_rechecked(0.82, True)
    )
    (failure,) = gate.evaluate({"results": [real]})[0]
    assert "0.80x (reproduced at 0.82x over 21 iterations)" in failure


# ----------------------------------------------------------------------- profiles


@pytest.mark.parametrize("profile", sorted(run.PROFILES))
def test_case_labels_are_unique(profile):
    labels = [case.label for case in run.PROFILES[profile]]
    assert len(labels) == len(set(labels))


def test_the_stock_baseline_is_the_stock_default():
    _, make_sk = run._estimators(run.Case("knn", "fit", 10, 3), "direct")
    assert make_sk().algorithm == "auto"
    _, make_sk = run._estimators(
        run.Case("knn", "fit", 10, 3, options={"sklearn_algorithm": "brute"}), "direct"
    )
    assert make_sk().algorithm == "brute"


def test_the_full_grid_covers_the_regimes_found_missing():
    full = run.PROFILES["full"]
    knn_queries = {(c.n_features, c.n_queries) for c in full if c.operation == "kneighbors"}
    for d in (2, 3, 8):
        assert (d, 256) in knn_queries
    for d in (8, 32):
        assert {(d, 10_000), (d, 50_000)} <= knn_queries
    classes = {c.option("n_classes") for c in full if c.algorithm == "svc"}
    assert {8, 32, 128} <= classes


def test_the_published_profile_encodes_the_documented_headline_cases():
    labels = {case.label for case in run.PROFILES["published"]}
    assert "knn/kneighbors/128000x768/q256/sklearn_algorithm=brute" in labels
    assert "knn/kneighbors/1000x32/q4096/sklearn_algorithm=brute" in labels
    # The mid-size batch the width rule is about, on both sides of it.
    assert "knn/kneighbors/131072x128/q64/sklearn_algorithm=brute" in labels
    assert "knn/kneighbors/1000000x16/q64/sklearn_algorithm=brute" in labels
    assert "svc/fit/4000x256" in labels
    assert "svc/fit/4000x64/C=10.0/kernel=poly" in labels
    assert "svc/fit/2000x128/C=10.0/kernel=poly" in labels
    assert "svc/fit/4000x256/C=10.0/kernel=sigmoid" in labels
    assert "svc/fit/2000x256/kernel=linear" in labels
    assert "logreg/fit/1000000x16" in labels
    assert "logreg/fit/60000x64/n_classes=5" in labels
    assert run.PROFILE_SETTINGS["published"] == {
        "layer": "direct",
        "force_backend": "mlx",
        "iterations": 25,
    }


def _patched_route(case: run.Case) -> str:
    """Where patched dispatch sends a case under the configured Tuning, from its shape."""
    from mlxlearn.patching import _estimators as patched

    n, d = case.n_samples, case.n_features
    n_classes = case.option("n_classes") or 2
    if case.algorithm == "knn":
        model = patched.NearestNeighbors(
            n_neighbors=case.option("n_neighbors"), algorithm=case.option("sklearn_algorithm")
        )
        if model._below_crossover(n, d)[0]:
            return "below-crossover"
        handed = model._patched_handoff(n, d)
        if handed is None and case.option("data"):
            # The data rule reads the data: fit on what the case would generate.
            X, _, _ = run._make_data(case, np.random.default_rng(0))
            event = model.fit(X).fit_event_
            return "mlx" if event.reason == "selected" else "handoff:" + event.reason
        return "handoff:" + handed[0] if handed else "mlx"
    if case.algorithm == "logreg":
        params = {k: case.options[k] for k in ("tol", "max_iter") if k in case.options}
        model = patched.LogisticRegression(**params)
        below, detail = model._below_crossover(n, d, 1 if n_classes == 2 else n_classes)
        if below:
            return "binary:" if "binary" in detail else "below-crossover"
        return "params:" if model._param_handoff() is not None else "mlx"
    kernel = case.option("kernel")
    params = {k: case.options[k] for k in ("degree", "coef0", "cache_size") if k in case.options}
    model = patched.SVC(kernel=kernel, C=case.option("C"), **params)
    below, detail = model._below_crossover(n, d, n_classes=n_classes)
    if below:
        return "pair:" if "one-vs-one" in detail else "below-crossover"
    # The smoke cases pass no weights, so the box the rule reads is C.
    return "kernel:" if model._kernel_handoff(case.option("C")) is not None else "mlx"


def test_smoke_lands_on_both_sides_of_every_mlx_threshold():
    """Every floor in Tuning decides at least one smoke case, and every estimator
    has smoke cases that reach MLX, so a threshold change that stops routing
    anything to MLX — or routes everything — shows up in the smoke gate."""
    from mlxlearn._common.config import get_tuning

    tuning = get_tuning()
    smoke = run.PROFILES["smoke"]

    def cases(algorithm):
        return [(c, _patched_route(c)) for c in smoke if c.algorithm == algorithm]

    knn = cases("knn")
    assert any(r == "mlx" and c.n_features >= tuning.knn_wide_features for c, r in knn)
    assert any(
        r == "mlx"
        and c.n_features < tuning.knn_wide_features
        and c.n_samples >= tuning.knn_narrow_min_samples
        for c, r in knn
    )
    assert any(c.n_samples < tuning.knn_min_train_samples for c, _ in knn)
    assert any(
        r == "below-crossover"
        and c.n_samples >= tuning.knn_min_train_samples
        and c.n_samples * c.n_features < tuning.knn_min_work
        for c, r in knn
    )
    assert any(
        r == "handoff:below-crossover"
        and tuning.knn_min_features <= c.n_features < tuning.knn_wide_features
        and c.n_samples < tuning.knn_narrow_min_samples
        for c, r in knn
    )
    assert any(r == "handoff:tree-wins" for _, r in knn)
    # Too narrow for MLX whatever the rows, where scikit-learn would build no tree;
    # and, from that width, tall enough for MLX.
    assert any(
        r == "handoff:below-crossover"
        and c.n_features < tuning.knn_min_features
        and c.option("sklearn_algorithm") == "brute"
        and c.n_samples >= tuning.knn_narrow_min_samples
        for c, r in knn
    )
    assert any(
        r == "mlx"
        and c.n_features == tuning.knn_min_features
        and c.n_samples >= tuning.knn_narrow_min_samples
        for c, r in knn
    )
    # The data rule, both halves: rows the probe cannot certify at a wide shape, and
    # integer data at a tall narrow one; the same shapes on Gaussian data go to MLX.
    assert any(
        r == "handoff:data-probe" and c.option("data") == "dups"
        and c.n_features >= tuning.knn_wide_features
        for c, r in knn
    )
    assert any(
        r == "handoff:data-probe" and c.option("data") == "binary"
        and c.n_features < tuning.knn_wide_features
        and c.n_samples >= tuning.knn_narrow_min_samples
        for c, r in knn
    )
    # Rows repeated about twice leave only a few probed rows uncertified, and that is
    # enough to hand the fit over.
    assert any(r == "handoff:data-probe" and c.option("data") == "dup2" for c, r in knn)
    # More neighbors than the k rule keeps on MLX, at a shape the size rules send there:
    # asked of the fit, and asked of a query on an index fitted on MLX.
    assert any(
        r == "handoff:below-crossover"
        and c.option("n_neighbors") > tuning.knn_max_neighbors
        and c.n_features >= tuning.knn_min_features
        and c.n_samples >= tuning.knn_narrow_min_samples
        for c, r in knn
    )
    assert any(
        r == "mlx" and (c.option("query_neighbors") or 0) > tuning.knn_max_neighbors
        for c, r in knn
    )
    # Wide, clearing both size floors, short of the wide-data row floor.
    assert any(
        r == "handoff:below-crossover"
        and c.n_features >= tuning.knn_wide_features
        and tuning.knn_min_train_samples <= c.n_samples < tuning.knn_wide_min_samples
        and c.n_samples * c.n_features >= tuning.knn_min_work
        for c, r in knn
    )

    logreg = cases("logreg")
    assert any(r == "mlx" and (c.option("n_classes") or 2) == 2 for c, r in logreg)
    assert any(r == "mlx" and (c.option("n_classes") or 2) > 2 for c, r in logreg)
    assert any(c.n_samples < tuning.logreg_min_samples for c, _ in logreg)
    assert any(
        r == "binary:" and c.n_features > tuning.logreg_binary_max_features for c, r in logreg
    )
    assert any(
        r == "binary:"
        and c.n_features <= tuning.logreg_binary_max_features
        and tuning.logreg_min_samples <= c.n_samples < tuning.logreg_binary_min_samples
        for c, r in logreg
    )
    # Multinomial, with the rows, narrower than the multinomial width floor.
    assert any(
        r == "below-crossover"
        and (c.option("n_classes") or 2) > 2
        and c.n_samples >= tuning.logreg_min_samples
        and c.n_features < tuning.logreg_multinomial_min_features
        for c, r in logreg
    )
    # The solver-parameter rule, both halves, at a shape the size rules send to MLX.
    params = [c for c, r in logreg if r == "params:"]
    assert any(c.options.get("tol", 0.0) > 1e-4 for c in params)  # the default tol
    assert any(c.options.get("max_iter", 200) < tuning.logreg_min_max_iter for c in params)

    svc = cases("svc")
    assert any(r == "mlx" and (c.option("n_classes") or 2) == 2 for c, r in svc)
    assert any(r == "mlx" and (c.option("n_classes") or 2) > 2 for c, r in svc)
    assert any(
        r == "below-crossover"
        and c.n_samples >= tuning.svc_min_samples
        and c.n_features < tuning.svc_min_features
        for c, r in svc
    )
    # Short of the width floor with enough work for the other two.
    assert any(
        r == "below-crossover"
        and c.n_features < tuning.svc_min_features <= 2 * c.n_features
        and c.n_samples * c.n_features >= tuning.svc_min_work
        for c, r in svc
    )
    assert any(r == "pair:" for _, r in svc)
    # The kernel, C and cache rules, at a shape the size floors send to MLX: C above
    # the measured box, the poly kernel at C = 1, and a cache below the default.
    kernel_handoffs = [c for c, r in svc if r == "kernel:"]
    assert any(c.option("C") > tuning.svc_max_c for c in kernel_handoffs)
    assert any(
        c.option("kernel") == "poly" and c.option("C") <= tuning.svc_max_c
        for c in kernel_handoffs
    )
    assert any(c.options.get("cache_size", 200.0) < 200.0 for c in kernel_handoffs)
    assert all(
        c.n_features >= tuning.svc_min_features and c.n_samples * c.n_features >= tuning.svc_min_work
        for c in kernel_handoffs
    )
    # Inference on an MLX fit on both sides of the host threshold: one row is below it
    # whatever the support-vector count (at most the training rows), and a thousand
    # rows above it once a thousand of them are support vectors — about 3 100 of the
    # 4 000 are, on this data (table `svc-predict`).
    predicts = [c for c, r in svc if c.operation == "predict" and r == "mlx"]
    assert any(c.n_queries * c.n_samples * c.n_features <= tuning.svc_host_inference_work
               for c in predicts)
    assert any(c.n_queries * c.n_features * 1_000 > tuning.svc_host_inference_work
               for c in predicts)


def test_the_published_profile_brackets_every_estimators_crossover():
    """Its tables show each estimator on both sides of patched dispatch's decision:
    the forced-MLX timings where dispatch would not use MLX are the evidence for
    the thresholds, and the gate judges the rest."""
    routes = {}
    for case in run.PROFILES["published"]:
        routes.setdefault(case.algorithm, set()).add(_patched_route(case) == "mlx")
    assert routes == {"knn": {True, False}, "svc": {True, False}, "logreg": {True, False}}


# --------------------------------------------------------------------- end to end

_TINY = [
    run.Case("knn", "kneighbors", 200, 8, 32),
    run.Case("logreg", "fit", 300, 4),
    run.Case("svc", "fit", 240, 4, options={"n_classes": 3}),
]


def test_a_run_records_backends_and_the_gate_reads_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(run.PROFILES, "tiny", _TINY)
    output, report = tmp_path / "results.json", tmp_path / "report.md"
    args = ["--profile", "tiny", "--iterations", "1", "--output", str(output)]
    assert run.main([*args, "--report", str(report)]) == 0

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2 and payload["layer"] == "patched"
    backends = {r["label"]: r["backend"] for r in payload["results"]}
    assert set(backends) == {case.label for case in _TINY}
    # Every one of these is far below its threshold: patched dispatch hands it off.
    assert set(backends.values()) == {"sklearn"}
    assert all(r["dispatch"] == r["backend"] for r in payload["results"])
    text = report.read_text()
    assert "interquartile range" in text and "three standard errors" in text

    assert gate.main(["--results", str(output)]) == 0


@pytest.mark.gpu
def test_a_forced_run_records_what_dispatch_would_have_done(tmp_path, monkeypatch):
    monkeypatch.setitem(run.PROFILES, "tiny", _TINY[:1])
    output = tmp_path / "results.json"
    argv = ["--profile", "tiny", "--iterations", "1", "--output", str(output)]
    assert run.main([*argv, "--layer", "direct", "--force-backend", "mlx"]) == 0
    (result,) = json.loads(output.read_text())["results"]
    assert result["backend"] == "mlx" and result["force_backend"] == "mlx"
    assert result["dispatch"] == "sklearn"


@pytest.mark.gpu
def test_a_backend_forced_through_the_environment_is_recorded_as_forced(tmp_path, monkeypatch):
    # The same kernel run as the flag's above, which the gate once judged as Tuning's
    # own decision and failed with "the threshold in Tuning is too low".
    monkeypatch.setenv("MLXLEARN_FORCE_BACKEND", "mlx")
    reset_config()
    monkeypatch.setitem(run.PROFILES, "tiny", _TINY[:1])
    output = tmp_path / "results.json"
    assert run.main(["--profile", "tiny", "--iterations", "1", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    (result,) = payload["results"]
    assert payload["force_backend"] == result["force_backend"] == "mlx"
    assert result["backend"] == "mlx" and result["dispatch"] == "sklearn"
    environment = payload["environment"]
    assert environment["mlxlearn_environment"]["MLXLEARN_FORCE_BACKEND"] == "mlx"
    assert environment["tuning"]["force_backend"] == "mlx"
    assert environment["tuning"]["logreg_min_samples"] > 0


def _fake_compare(calls, slower_first=False):
    """compare() with scripted timings: a 0.5x loss the first time if asked, then even."""

    def fake(mlx_fn, sk_fn, *, label, shape, iterations, **extra):
        mlx_fn()  # the runner reads the backend off the model this fits
        sk_fn()
        calls.append(iterations)
        mlx_time = 2.0 if slower_first and len(calls) == 1 else 1.0
        return Comparison(
            label=label,
            shape=shape,
            mlxlearn=_summarize([mlx_time] * iterations, "m"),
            sklearn=_summarize([1.0] * iterations, "s"),
            extra=extra,
        )

    return fake


def test_a_case_the_gate_would_fail_is_re_measured_before_it_fails(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "compare", _fake_compare(calls, slower_first=True))
    monkeypatch.setitem(run.PROFILES, "tiny", [_TINY[1]])
    output = tmp_path / "results.json"
    assert run.main(["--profile", "tiny", "--iterations", "7", "--output", str(output)]) == 0

    assert calls == [7, 7 * run.CONFIRMATION_FACTOR]
    payload = json.loads(output.read_text())
    (result,) = payload["results"]
    assert result["speedup"] == 0.5 and result["significant"]
    assert result["confirmation"]["speedup"] == 1.0
    assert result["confirmation"]["mlxlearn"]["iterations"] == 21
    assert gate.evaluate(payload)[0] == []


def test_a_case_that_passes_is_measured_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "compare", _fake_compare(calls))
    monkeypatch.setitem(run.PROFILES, "tiny", [_TINY[1]])
    output = tmp_path / "results.json"
    assert run.main(["--profile", "tiny", "--iterations", "7", "--output", str(output)]) == 0
    assert calls == [7]
    assert "confirmation" not in json.loads(output.read_text())["results"][0]


@pytest.mark.parametrize("requested, ran", [(1, 1), (9, 5)])
def test_a_case_count_caps_the_iterations_and_never_raises_them(
    requested, ran, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(run, "compare", _fake_compare(calls))
    monkeypatch.setitem(run.PROFILES, "tiny", [run.Case("logreg", "fit", 300, 4, iterations=5)])
    output = tmp_path / "results.json"
    argv = ["--profile", "tiny", "--iterations", str(requested), "--output", str(output)]
    assert run.main(argv) == 0
    assert calls == [ran]
