"""o-means schedule, an independent training oracle, and estimator contracts.

The training oracle is a float64 NumPy recurrence with its own cell build,
two-level search, EMA and closing means; it shares no code with the Metal
implementation.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from mlxlearn.cluster._o_schedule import (
    closing_weight,
    expected_fraction,
    num_cells,
    num_probes,
    resolve_schedule,
)
from mlxlearn.cluster._recipe import resolve_recipe, scheduled_alpha


@pytest.fixture
def metal():
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("requires Apple Silicon Metal")
    return mx


def test_cells_and_probes_follow_the_reference_rule():
    assert [num_cells(k) for k in (1024, 16384, 20000, 65536)] == [16, 128, 160, 512]
    assert num_cells(12) == 12 and num_cells(1) == 1
    assert num_probes(16384, 96, 128) == 16
    assert num_probes(16384, 768, 128) == 12
    assert num_probes(65536, 96, 512) == num_probes(65536, 768, 512) == 24
    assert num_probes(12, 8, 12) == 12
    recipe = resolve_schedule(1_000_000, 16384, 96, 1)
    assert (recipe["n_cells"], recipe["n_probe"]) == (128, 16)
    recipe = resolve_schedule(1000, 12, 8, 1)
    assert (recipe["n_cells"], recipe["n_probe"]) == (12, 12)
    recipe = resolve_schedule(1_000_000, 16384, 96, 1, n_cells=64, n_probe=5)
    assert (recipe["n_cells"], recipe["n_probe"]) == (64, 5)
    assert expected_fraction(16384, 128, 16) == pytest.approx(2176 / 16384)


@pytest.mark.parametrize("n,k,law", [(1_000_000, 16384, "throughput"), (3_000_000, 65536, "quality")])
@pytest.mark.parametrize("epochs", [1, 1.6, 3])
def test_recipe_is_the_faiss_e_means_recipe(n, k, law, epochs):
    recipe = resolve_schedule(n, k, 96, epochs)
    expected = resolve_recipe(n, k, 96, epochs, batch_law=law, recipe="faiss")
    assert recipe["batch_law"] == law
    for key in ("batch_size", "alpha0", "epoch_rows", "epoch_steps", "total_steps",
                "init_usage", "split_epsilon", "unused_threshold", "check_unused_every"):
        assert recipe[key] == expected[key], key
    assert recipe["algorithm"] == "o-means" and recipe["name"] == "faiss"


def test_below_one_epoch_keeps_the_one_epoch_batch_and_step():
    short, full = resolve_schedule(1_000_000, 16384, 96, 0.8), resolve_schedule(1_000_000, 16384, 96, 1)
    assert short["batch_size"] == full["batch_size"] and short["alpha0"] == full["alpha0"]
    assert short["epoch_rows"] == [800_000] and short["nepochs"] == 0.8
    assert short["total_steps"] == math.ceil(800_000 / short["batch_size"])
    with pytest.raises(ValueError, match="even one training sample"):
        resolve_schedule(100, 4, 3, 0.001)


def test_auto_batch_law_depends_on_rows_not_dimension():
    assert resolve_schedule(2_000_000, 64, 8, 2)["batch_law"] == "throughput"
    assert resolve_schedule(2_000_001, 64, 8, 2)["batch_law"] == "quality"
    assert resolve_schedule(1000, 64, 8, 2, batch_law="quality")["batch_law_reason"] == "requested"
    assert resolve_schedule(1000, 4, 2, 2)["unused_threshold"] == 0.0


def test_closing_weights_and_rebuild_cadence():
    n = 1_000_000
    recipe = resolve_schedule(n, 16384, 96, 1.6)
    assert recipe["closing"] == "recency_weighted"
    total, batch, done = sum(recipe["epoch_rows"]), recipe["batch_size"], 0
    for epoch, rows in enumerate(recipe["epoch_rows"]):
        for start in range(0, rows, batch):
            done += min(batch, rows - start)
            assert closing_weight(recipe, n, epoch, done) == pytest.approx(
                math.exp(-(total - done) / n), abs=1e-12)
    assert closing_weight(recipe, n, 1, total) == 1.0
    recipe = resolve_schedule(n, 16384, 96, 2)
    assert recipe["closing"] == "last_pass"
    assert closing_weight(recipe, n, 0, 5) is None and closing_weight(recipe, n, 1, n + 5) == 1.0
    steps_per_pass = math.ceil(n / recipe["batch_size"])
    assert recipe["rebuild_every"] == max(5, math.ceil(steps_per_pass / 2))
    assert resolve_schedule(100, 8, 4, 2, batch_size=50)["rebuild_every"] == 5
    assert (recipe["first_build_iterations"], recipe["first_pass_rebuild_iterations"],
            recipe["rebuild_iterations"]) == (4, 0, 2)


@pytest.mark.parametrize("kwargs,match", [
    ({"epochs": 0}, "max_iter"),
    ({"k": 2000}, "n_clusters must not exceed"),
    ({"batch_law": "fast"}, "batch_law"),
    ({"n_cells": 0}, "n_cells"),
    ({"n_cells": 65}, "n_cells"),
    ({"n_cells": 8, "n_probe": 9}, "n_probe"),
    ({"n_probe": True}, "n_probe"),
])
def test_schedule_rejects_invalid_arguments(kwargs, match):
    arguments = {"n": 1000, "k": 64, "d": 8, "epochs": 2}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        resolve_schedule(**arguments)


def _nearest(points, centres):
    return ((points[:, None, :] - centres[None, :, :]) ** 2).sum(2).argmin(1)


def _numpy_o_means(x, init, *, seed, passes, batch, peak, n_cells, n_probe, recency, init_mass=2**-16):
    """Float64 o-means without revival: cells, two-level search, EMA, closing."""
    x = x.astype(np.float64)
    n, k = len(x), len(init)
    centers = init.astype(np.float64).copy()
    mass = np.full(k, init_mass)
    sums = centers * mass[:, None]
    cell_rng = np.random.RandomState((seed * 7919 + 17) % 2**32)
    total_steps = sum(math.ceil(rows / batch) for rows in passes)
    total_rows = sum(passes)
    rebuild_every = max(5, math.ceil(math.ceil(n / batch) / 2))
    centres, cell = None, None
    since = rows_done = step = 0
    close_sum, close_count = np.zeros_like(centers), np.zeros(k)
    moving = np.full(n, -1)
    for epoch, rows in enumerate(passes):
        order = np.random.RandomState((seed * 1000003 + epoch) % 2**32).permutation(n)[:rows]
        for start in range(0, rows, batch):
            index = order[start:start + batch]
            rows_x = x[index]
            first_pass = rows_done < n
            if centres is None or since >= (5 if first_pass else rebuild_every):
                if centres is None:
                    centres, iterations = centers[cell_rng.permutation(k)[:n_cells]].copy(), 4
                else:
                    iterations = 0 if first_pass else 2
                for _ in range(iterations):
                    cell = _nearest(centers, centres)
                    for g in range(n_cells):
                        if np.any(cell == g):
                            centres[g] = centers[cell == g].mean(0)
                cell = _nearest(centers, centres)
                since = 0
            since += 1
            coarse = (centres ** 2).sum(1)[None, :] - 2 * rows_x @ centres.T
            coarse[:, np.bincount(cell, minlength=n_cells) == 0] = np.inf
            probes = np.argsort(coarse, axis=1, kind="stable")[:, :n_probe]
            scores = (centers ** 2).sum(1)[None, :] - 2 * rows_x @ centers.T
            allowed = (cell[None, None, :] == probes[:, :, None]).any(1)
            labels = np.where(allowed, scores, np.inf).argmin(1)
            rows_done += len(index)
            batch_sum = np.zeros_like(centers)
            np.add.at(batch_sum, labels, rows_x)
            count = np.bincount(labels, minlength=k).astype(np.float64)
            if recency:
                omega = math.exp(-(total_rows - rows_done) / n)
            else:
                omega = 1.0 if epoch == len(passes) - 1 else None
            if omega is not None:
                close_sum += omega * batch_sum
                close_count += omega * count
            if epoch == len(passes) - 1:
                moving[index] = labels
            alpha = peak * (0.1 + 0.9 * (1 + math.cos(math.pi * step / total_steps)) / 2)
            sums = (1 - alpha) * sums + alpha * batch_sum
            mass = (1 - alpha) * mass + alpha * count
            centers = sums / mass[:, None]
            step += 1
    closing = centers.copy()
    closed = close_count > (1e-3 if recency else 0)
    closing[closed] = close_sum[closed] / close_count[closed, None]
    return centers, closing, moving


@pytest.mark.gpu
@pytest.mark.parametrize("epochs,passes,recency,builds", [
    (1.5, [150, 300], True, 3),
    (2.5, [300, 150, 300], False, 5),
    (0.5, [150], True, 1),
])
def test_training_matches_independent_double_oracle(metal, epochs, passes, recency, builds):
    from mlxlearn.cluster import OMeans

    rng = np.random.default_rng(772)
    x = rng.normal(size=(300, 6)).astype(np.float32)
    init = x[rng.permutation(300)[:24]].copy()
    expected = _numpy_o_means(
        x, init, seed=31, passes=passes, batch=32, peak=0.31, n_cells=4, n_probe=2, recency=recency)
    model = OMeans(
        24, init=init, max_iter=epochs, batch_size=32, alpha0=0.31, random_state=31,
        unused_threshold=0, n_cells=4, n_probe=2, assignment="two_level", compute_labels=False,
    )
    prepared = model.prepare_device(x)
    result = model.train_device(prepared, capture_assignments=True)
    assert model.recipe_["closing"] == ("recency_weighted" if recency else "last_pass")
    assert model.recipe_["search"] == "two_level"
    np.testing.assert_array_equal(np.asarray(result.assignments), expected[2])
    np.testing.assert_allclose(np.asarray(result.ema_centers), expected[0], rtol=1e-5, atol=3e-6)
    np.testing.assert_allclose(np.asarray(result.centers), expected[1], rtol=1e-5, atol=3e-6)
    assert result.assignments.dtype == metal.int64
    assert result.steps == sum(math.ceil(rows / 32) for rows in passes)
    assert [record["rows"] for record in result.history] == passes
    assert np.count_nonzero(np.asarray(result.assignments) == -1) == 300 - passes[-1]
    assert all(0 < record["evaluated_fraction"] < 1 for record in result.history)
    # Cells are re-filed every five steps of the first pass, then rebuilt
    # every half pass (five steps here).
    assert result.history[-1]["cell_builds"] == builds


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_dense_search_from_two_epochs_is_faiss_recipe_e_means(metal, precision):
    from mlxlearn.cluster import EMeans, OMeans

    x = np.random.default_rng(15).normal(size=(211, 9)).astype(np.float32)
    init = x[:13].copy()
    shared = {"init": init, "max_iter": 2.5, "batch_size": 17, "alpha0": 0.3, "random_state": 4,
              "precision": precision}
    expected = EMeans(13, recipe="faiss", batch_law="throughput", **shared).fit(x)
    actual = OMeans(13, assignment="dense", **shared).fit(x)
    np.testing.assert_allclose(actual.cluster_centers_, expected.cluster_centers_, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(actual.ema_centers_, expected.ema_centers_, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(actual.labels_, expected.labels_)
    assert actual.n_splits_ == expected.n_splits_ and actual.n_steps_ == expected.n_steps_
    assert actual.recipe_["search"] == "dense"
    assert all(record["evaluated_fraction"] == 1.0 for record in actual.iteration_stats_)


@pytest.mark.gpu
def test_last_pass_closing_is_the_mean_by_captured_label_with_revival(metal):
    from mlxlearn.cluster import OMeans

    rng = np.random.default_rng(90)
    means = rng.normal(size=(40, 12)) * 3
    x = (means[rng.integers(0, 40, 4000)] + rng.normal(size=(4000, 12)) * 0.3).astype(np.float32)
    model = OMeans(256, max_iter=2, batch_size=250, n_cells=16, n_probe=4, random_state=3,
                   assignment="two_level", compute_labels=False)
    result = model.train_device(model.prepare_device(x), capture_assignments=True)
    labels = np.asarray(result.assignments)
    assert labels.min() >= 0 and labels.max() < 256 and int(result.split_count.item()) > 0
    sums = np.zeros((256, 12))
    np.add.at(sums, labels, x.astype(np.float64))
    counts = np.bincount(labels, minlength=256)
    closing, ema = np.asarray(result.centers), np.asarray(result.ema_centers)
    used = counts > 0
    np.testing.assert_allclose(closing[used], sums[used] / counts[used, None], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(closing[~used], ema[~used])


@pytest.mark.gpu
def test_split_halves_closing_mass(metal):
    from mlxlearn.cluster._e_means import split_plan
    from mlxlearn.cluster._o_means import split_closing

    counts = np.array([16, 0, 8, 0], np.float32)
    plan = split_plan(metal.array(counts), metal.array(np.array([0.9, 0.5, 0.2, 0.5], np.float32)), 0.1)
    close_sums = np.array([[8, -4], [1, 1], [2, 6], [3, 3]], np.float32)
    close_counts = np.array([4, 0.5, 2, 0.25], np.float32)
    sums, mass = split_closing(metal.array(close_sums), metal.array(close_counts), plan)
    source = np.asarray(plan[2])
    donors = source[[1, 3]]
    assert sorted(donors.tolist()) == [0, 2]
    expected_sums, expected_mass = close_sums.copy(), close_counts.copy()
    for slot, donor in zip([1, 3], donors, strict=True):
        expected_sums[[slot, donor]] = close_sums[donor] / 2
        expected_mass[[slot, donor]] = close_counts[donor] / 2
    np.testing.assert_array_equal(np.asarray(sums), expected_sums)
    np.testing.assert_array_equal(np.asarray(mass), expected_mass)


@pytest.mark.gpu
@pytest.mark.parametrize("epochs", [1, 3])
def test_error_is_close_to_e_means_on_a_mixture(metal, epochs):
    from mlxlearn.cluster import EMeans, OMeans

    rng = np.random.default_rng(0)
    n, d, k = 60_000, 32, 1024
    modes = rng.normal(size=(4096, d)).astype(np.float32)
    x = (modes[rng.integers(0, 4096, n + 5000)] + rng.normal(size=(n + 5000, d)).astype(np.float32) * 0.3)
    train, test = x[:n], x[n:]
    reference = EMeans(k, max_iter=epochs, recipe="faiss", batch_law="throughput",
                       compute_labels=False).fit(train)
    # The default rule probes every cell at this size; halve the probes so
    # that the search is genuinely approximate.
    model = OMeans(k, max_iter=epochs, n_probe=8, assignment="two_level",
                   compute_labels=False).fit(train)
    assert model.recipe_["n_cells"] == 16 and model.recipe_["n_probe"] == 8
    ratio = model.score(test) / reference.score(test)
    assert abs(ratio - 1) < 0.02
    assert all(record["evaluated_fraction"] < 0.6 for record in model.iteration_stats_)
    assert np.isfinite(model.cluster_centers_).all() and model.cluster_centers_.dtype == np.float32


@pytest.mark.gpu
def test_fit_labels_and_inference_use_a_full_search(metal):
    from mlxlearn.cluster import OMeans

    rng = np.random.default_rng(2)
    x = rng.normal(size=(900, 10)).astype(np.float32)
    model = OMeans(64, max_iter=2, n_cells=8, n_probe=1, assignment="two_level", random_state=7).fit(x)
    centers = model.cluster_centers_.astype(np.float64)
    distance = ((x.astype(np.float64)[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    np.testing.assert_array_equal(model.labels_, distance.argmin(1))
    np.testing.assert_array_equal(model.predict(x), distance.argmin(1))
    assert model.inertia_ == pytest.approx(distance.min(1).sum(), rel=3e-6)
    assert model.score(x) == pytest.approx(-distance.min(1).sum(), rel=3e-6)
    np.testing.assert_allclose(model.transform(x[:7]), np.sqrt(distance[:7]), rtol=2e-5, atol=2e-5)
    assert model.execution_backend_ == "metal" and model.labels_.dtype == np.int64


@pytest.mark.gpu
def test_seed_reproducibility_and_resident_reuse(metal):
    from mlxlearn.cluster import OMeans

    x = np.random.default_rng(991).normal(size=(400, 5)).astype(np.float32)
    kwargs = {"n_clusters": 32, "max_iter": 2, "batch_size": 40, "n_cells": 4, "n_probe": 2,
              "assignment": "two_level", "random_state": 87, "compute_labels": False}
    first, second = OMeans(**kwargs).fit(x), OMeans(**kwargs).fit(x)
    assert first.labels_ is None and first.inertia_ is None
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_, rtol=3e-6, atol=3e-6)
    other = OMeans(**{**kwargs, "random_state": 88}).fit(x)
    assert not np.allclose(first.cluster_centers_, other.cluster_centers_)
    model = OMeans(**kwargs)
    prepared = model.prepare_device(metal.array(x))
    one, two = model.train_device(prepared), model.train_device(prepared)
    assert one.assignments is None and isinstance(one.centers, metal.array)
    np.testing.assert_allclose(np.asarray(one.centers), np.asarray(two.centers), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(np.asarray(one.centers), first.cluster_centers_, rtol=3e-6, atol=3e-6)
    assert not hasattr(model, "cluster_centers_")
    model.set_params(n_probe=3)
    with pytest.raises(ValueError, match="configuration"):
        model.train_device(prepared)


def test_search_selection(monkeypatch):
    from mlxlearn.cluster import _o_means
    from mlxlearn.cluster._o_means import OMeans

    def search(**kwargs):
        shape = {"n": kwargs.pop("n", 200_000), "d": kwargs.pop("d", 96)}
        recipe = OMeans(**kwargs)._resolve_recipe(**shape)
        return recipe["search"], recipe["search_reason"]

    monkeypatch.setattr(_o_means, "_two_level_device", lambda: True)
    assert search(n_clusters=16384) == ("two_level", "validated")
    assert search(n_clusters=1024) == ("dense", "probes cover most centroids")
    assert search(n_clusters=1024, n_probe=4) == ("two_level", "validated")
    assert search(n_clusters=70_000) == ("dense", "outside the two-level limits")
    assert search(n_clusters=16384, n_cells=1024, n_probe=200) == ("dense", "outside the two-level limits")
    assert search(n_clusters=16384, assignment="dense") == ("dense", "requested")
    assert search(n_clusters=1024, assignment="two_level") == ("two_level", "requested")
    with pytest.raises(ValueError, match="two-level assignment supports"):
        search(n_clusters=70_000, assignment="two_level")
    monkeypatch.setattr(_o_means, "_two_level_device", lambda: False)
    assert search(n_clusters=16384) == ("dense", "two-level scan not validated on this GPU")
    assert search(n_clusters=16384, assignment="two_level") == ("two_level", "requested")


@pytest.mark.gpu
def test_auto_returns_to_dense_when_cells_stop_saving_comparisons(metal, monkeypatch):
    from mlxlearn.cluster import _o_means
    from mlxlearn.cluster._o_means import OMeans

    x = np.random.default_rng(8).normal(size=(3000, 6)).astype(np.float32)
    kwargs = {"n_clusters": 64, "max_iter": 2, "batch_size": 100, "n_cells": 8, "n_probe": 2,
              "random_state": 3, "compute_labels": False}
    monkeypatch.setattr(_o_means, "_two_level_device", lambda: True)
    saving = OMeans(**kwargs).fit(x)
    assert saving.recipe_["search"] == "two_level" and saving.recipe_["search_fallback_step"] is None
    assert [record["search"] for record in saving.iteration_stats_] == ["two_level", "two_level"]
    real = _o_means.two_level_assign

    def uneven(*arguments, **options):
        # Report the comparisons of cells so uneven that every row scans most centroids.
        labels, scores, evaluated = real(*arguments, **options)
        return labels, scores, evaluated * 4

    monkeypatch.setattr(_o_means, "two_level_assign", uneven)
    model = OMeans(**kwargs).fit(x)
    # The check after the first batch already returns to a full search.
    assert model.recipe_["search_fallback_step"] == 1
    first, second = model.iteration_stats_
    assert first["search"] == "two_level" and second["search"] == "dense"
    assert second["evaluated_fraction"] == 1.0 and first["evaluated_fraction"] < 1.05
    assert first["cell_builds"] == second["cell_builds"] == 1
    # An explicit request is never overridden.
    forced = OMeans(assignment="two_level", **kwargs).fit(x)
    assert forced.recipe_["search_fallback_step"] is None
    assert [record["search"] for record in forced.iteration_stats_] == ["two_level", "two_level"]
    assert np.isfinite(model.cluster_centers_).all()


def test_two_level_device_requires_a_validated_gpu_and_the_tiled_scan(monkeypatch):
    from mlxlearn.cluster import _o_means

    mx = pytest.importorskip("mlx.core")
    for name, available, expected in [
            ("Apple M4 Max", True, True), ("Apple M1 Ultra", True, True), ("Apple M5 Ultra", True, True),
            ("Apple M4 Max", False, False), ("Apple M9 Hyper", True, False)]:
        _o_means._two_level_device.cache_clear()
        monkeypatch.setattr(mx, "device_info", lambda name=name: {"device_name": name})
        monkeypatch.setattr(_o_means, "tiled_scan_available", lambda available=available: available)
        assert _o_means._two_level_device() is expected
    _o_means._two_level_device.cache_clear()


def test_estimator_parameters_clone_and_pickle():
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError

    from mlxlearn.cluster import EMeans, OMeans

    model = OMeans(12, max_iter=1.5, n_cells=3, n_probe=2, assignment="two_level")
    assert isinstance(model, EMeans)
    assert set(model.get_params()) == {
        "n_clusters", "max_iter", "batch_size", "alpha0", "batch_law", "n_cells", "n_probe",
        "random_state", "init", "unused_threshold", "assignment", "precision",
        "assignment_batch_size", "compute_labels", "verbose"}
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert pickle.loads(pickle.dumps(model)).get_params() == model.get_params()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(spherical=True)
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((2, 3), np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("kwargs,match", [
    ({"assignment": "fused"}, "assignment must be"),
    ({"precision": "float16"}, "precision must be"),
    ({"assignment_batch_size": 0}, "assignment_batch_size"),
    ({"n_cells": 99}, "n_cells"),
    ({"n_probe": 0}, "n_probe"),
    ({"batch_law": "fast"}, "batch_law"),
    ({"max_iter": 0}, "max_iter"),
    ({"n_clusters": 500}, "n_clusters must not exceed"),
    ({"init": "k-means++"}, "init must be"),
])
def test_invalid_parameters_and_inputs(metal, kwargs, match):
    from mlxlearn.cluster import OMeans

    x = np.random.default_rng(4).normal(size=(60, 4)).astype(np.float32)
    with pytest.raises(ValueError, match=match):
        OMeans(**{"n_clusters": 8, **kwargs}).fit(x)
    with pytest.raises(ValueError, match="finite"):
        OMeans(8).fit(np.full((60, 4), np.nan, np.float32))
    with pytest.raises(ValueError, match="too large"):
        OMeans(8).fit(x * np.float32(1e30))


@pytest.mark.gpu
def test_alpha_schedule_and_step_count_match_e_means(metal):
    from mlxlearn.cluster import OMeans

    x = np.random.default_rng(6).normal(size=(500, 6)).astype(np.float32)
    model = OMeans(40, max_iter=0.8, assignment="two_level", n_cells=5, n_probe=2,
                   compute_labels=False).fit(x)
    recipe = model.recipe_
    assert recipe["epoch_rows"] == [400] and model.n_steps_ == recipe["total_steps"]
    assert scheduled_alpha(recipe["alpha0"], 0, recipe["total_steps"]) == pytest.approx(recipe["alpha0"])
    assert recipe["closing"] == "recency_weighted" and recipe["n_cells"] == 5
