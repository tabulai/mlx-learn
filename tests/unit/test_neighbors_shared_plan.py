"""Shared tiling without changing either interface's dependency or exactness contract."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import Tuning, tuning_context
from mlxlearn._kernels import distance
from mlxlearn_common import _neighbors_plan as planning
from mlxlearn_device import DeviceNeighborIndex, fit_neighbors
from mlxlearn_device import _neighbors_fused as fused
from mlxlearn_device import _neighbors_general as general


def test_planner_import_and_execution_do_not_load_either_array_interface():
    source = """
import importlib.abc
import sys
class BlockInterfaces(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx', 'numpy', 'sklearn', 'mlxlearn', 'mlxlearn_device'):
            raise AssertionError('Array interface imported: ' + fullname)
sys.meta_path.insert(0, BlockInterfaces())
from mlxlearn_common._neighbors_plan import resolve_blocks
plan = resolve_blocks(512, 1048576, backend='mlx', k_search=14, budget_bytes=1 << 30)
assert (plan.query_rows, plan.train_cols) == (64, 1048576)
assert not any(name in sys.modules for name in ('mlx', 'numpy', 'sklearn', 'mlxlearn', 'mlxlearn_device'))
"""
    subprocess.run([sys.executable, "-c", source], check=True, timeout=30)


def test_host_tuning_defaults_come_from_shared_settings():
    host = Tuning()
    for field in fields(planning.PlanSettings):
        assert getattr(host, field.name) == getattr(planning.DEFAULTS, field.name)


@pytest.mark.parametrize("memory,budget", [
    (None, 256 << 20), (8 << 30, 256 << 20), (16 << 30, 512 << 20),
    (128 << 30, 1 << 30),
])
def test_memory_derived_budget_is_capped(memory, budget):
    assert planning.distance_budget_bytes("mlx", planning.DEFAULTS, memory) == budget
    assert planning.distance_budget_bytes("cpu", planning.DEFAULTS, memory) == 128 << 20


@pytest.mark.parametrize("backend,budget", [("mlx", 1 << 30), ("cpu", 128 << 20)])
@pytest.mark.parametrize("n_query,n_train,k", [
    (3, 100, 5), (512, 1048576, 14), (1024, 2**28, 36), (2000, 100000, 100),
])
def test_default_tiles_stay_within_workspace_budget(backend, budget, n_query, n_train, k):
    plan = planning.resolve_blocks(n_query, n_train, backend=backend, k_search=k,
                                   budget_bytes=budget, settings=planning.DEFAULTS)
    assert 1 <= plan.query_rows <= n_query
    assert min(k, n_train) <= plan.train_cols <= n_train
    assert plan.query_rows * plan.train_cols * 16 <= budget


@pytest.mark.parametrize("budget_mib,row_cap,expected", [
    (64, 1024, (64, 65536)),
    (256, 1024, (128, 131072)),
    (512, 48, (48, (512 << 20) // (16 * 48))),
])
def test_tight_gram_tiles_preserve_budget_and_query_row_cap(budget_mib, row_cap, expected):
    budget = budget_mib << 20
    plan = planning.resolve_blocks(512, 2**20, backend="mlx", k_search=36,
                                   budget_bytes=budget,
                                   settings=planning.PlanSettings(query_block_rows=row_cap))
    assert plan.tight_gram_tile == expected
    rows, cols = expected
    assert plan.query_rows < rows <= row_cap
    assert 65536 <= cols < plan.train_cols
    assert rows * cols * 16 <= budget


@pytest.mark.parametrize("override", [
    {"backend": "cpu"}, {"k_search": 37}, {"n_query": 31}, {"n_train": 2**19 - 1},
    {"budget_bytes": (64 << 20) - 1}, {"budget_bytes": (512 << 20) + 1},
    {"query_block_size": 23}, {"train_block_size": 123},
    {"settings": planning.PlanSettings(query_block_rows=16)},
])
def test_tight_gram_tile_is_only_offered_for_eligible_automatic_search(override):
    options = {"n_query": 512, "n_train": 2**20, "backend": "mlx", "k_search": 14,
               "budget_bytes": 256 << 20, "settings": planning.DEFAULTS}
    plan = planning.resolve_blocks(**(options | override))
    assert plan.tight_gram_tile is None
    if "query_block_size" in override:
        assert plan.query_rows == override["query_block_size"]
    if "train_block_size" in override:
        assert plan.train_cols == override["train_block_size"]


def test_custom_query_alignment_keeps_the_aligned_full_width_tile():
    plan = planning.resolve_blocks(512, 2**20, backend="mlx", k_search=14,
                                   budget_bytes=384 << 20,
                                   settings=planning.PlanSettings(query_block_align=13))
    assert plan == planning.BlockPlan(13, 2**20)


def test_host_scoped_tuning_and_standalone_environment_share_the_plan(monkeypatch):
    settings = {"distance_block_bytes": 1 << 20, "distance_block_bytes_max": 1 << 20,
                "query_block_rows": 73, "query_block_align": 8,
                "cpu_distance_block_bytes": 2 << 20, "cpu_query_block_rows": 41}
    for key, value in settings.items():
        monkeypatch.setenv("MLXLEARN_" + key.upper(), str(value))
    for backend in ("mlx", "cpu"):
        with tuning_context(**settings):
            host = distance.resolve_blocks(200, 3000, backend=backend, k_search=14)
        standalone = planning.resolve_blocks(200, 3000, backend=backend, k_search=14)
        assert host == standalone
    assert planning.resolve_blocks(200, 3000, backend="mlx", k_search=14).query_rows == 16


def test_explicit_tiles_and_tiny_budgets_keep_historical_host_semantics():
    # Explicit dimensions and the minimum candidate list can exceed the target;
    # the budget has never been a cap on output or caller-selected tile sizes.
    assert planning.resolve_blocks(100, 50, backend="cpu", k_search=50, budget_bytes=1,
                                   query_block_size=7, train_block_size=3) == planning.BlockPlan(7, 50)
    assert planning.resolve_blocks(0, 50, backend="mlx", k_search=5) == planning.BlockPlan(0, 0)
    # Large CPU budgets must not opt into GPU query alignment.
    assert planning.resolve_blocks(1000, 20000, backend="cpu", k_search=5,
                                   budget_bytes=320000 * 137).query_rows == 137


@pytest.mark.parametrize("raw", ["bad", "-1"])
def test_standalone_malformed_tuning_is_an_error(raw, monkeypatch):
    monkeypatch.setenv("MLXLEARN_DISTANCE_BLOCK_BYTES", raw)
    with pytest.raises(ValueError):
        planning.resolve_blocks(4, 40, backend="mlx", k_search=5)


@pytest.mark.gpu
@pytest.mark.parametrize("memory_gib,family,features,expected", [
    (128, 16, 40, (64, 1048576, False)),
    (8, 16, 40, (128, 131072, True)),
    (8, 17, 40, (16, 1048576, False)),
    (8, None, 40, (16, 1048576, False)),
    (8, 16, 257, (16, 1048576, False)),
])
def test_general_query_interfaces_use_the_same_large_training_plan(
    monkeypatch, memory_gib, family, features, expected,
):
    class ShapeOnlyTraining:
        shape = (1048576, features)

    monkeypatch.setattr(planning, "physical_memory_bytes", lambda: memory_gib << 30)
    monkeypatch.setattr(distance, "physical_memory_bytes", lambda: memory_gib << 30)
    monkeypatch.setattr(general, "_gpu_family", lambda: family)
    calls = []

    def capture(index, query, k, *, rows, cols, bounded_workspace=False, **kwargs):
        calls.append((rows, cols, bounded_workspace))
        return mx.zeros((query.shape[0], k), mx.float32), mx.zeros((query.shape[0], k), mx.int32)

    monkeypatch.setattr(general, "gpu_neighbors", capture)
    # No million-row input allocation is needed to test which actual query plan
    # reaches the kernel. Numerical equivalence is checked below on real data.
    direct = DeviceNeighborIndex(ShapeOnlyTraining())
    queries = mx.zeros((512, features), mx.float32)
    direct.query(queries, 10, _validated=True)
    host = distance.NeighborIndex(None, "mlx", _device_index=direct)
    host.query_device(queries, 10)
    assert calls == [expected, expected]


@pytest.mark.gpu
def test_evaluating_gram_blocks_preserves_exact_distances_and_cross_tile_ties(monkeypatch):
    rng = np.random.default_rng(728)
    train = rng.integers(-5, 6, (65, 40)).astype(np.float32)
    train[[31, 32, 64]] = train[0]
    index = fit_neighbors(mx.array(train))
    k = 10
    values, indices = general.gpu_neighbors(index, index.X, k, rows=65, cols=65,
                                            exclude_self=True)
    # Squared integer distances are exactly representable. Stable sorting also
    # checks the tie break on duplicates, including the singleton final tile.
    squared = np.square(train[:, None, :] - train[None, :, :]).sum(axis=2)
    np.fill_diagonal(squared, np.inf)
    order = np.argsort(squared, axis=1, kind="stable")[:, :k]
    expected_values = np.take_along_axis(squared, order, axis=1)
    np.testing.assert_array_equal(np.asarray(indices), order)
    np.testing.assert_array_equal(np.asarray(values), expected_values)

    evaluated_shapes = []
    original_eval = mx.eval

    def trace_eval(*arrays):
        if len(arrays) == 2 and arrays[0].shape == arrays[1].shape:
            evaluated_shapes.append(arrays[0].shape)
        return original_eval(*arrays)

    monkeypatch.setattr(mx, "eval", trace_eval)
    tiled_values, tiled_indices = general.gpu_neighbors(
        index, index.X, k, rows=16, cols=32, exclude_self=True, bounded_workspace=True,
    )
    assert evaluated_shapes.count((16, k)) == 4
    np.testing.assert_array_equal(np.asarray(tiled_indices), order)
    np.testing.assert_array_equal(np.asarray(tiled_values).view(np.uint32),
                                  expected_values.view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("k", [5, 10])
def test_distant_tied_training_tile_keeps_full_width_certificates_and_distance_bits(k, monkeypatch):
    rng = np.random.default_rng(438)
    train = rng.normal(size=(96, 40)).astype(np.float32)
    # All 32 references in the middle tile tie beyond the global shortlist.
    # A tile-local certificate would fail and send every query to fallback.
    train[32:64] = rng.normal(loc=12.5, size=40).astype(np.float32)
    queries = rng.normal(size=(13, 40)).astype(np.float32)
    index = fit_neighbors(mx.array(train))
    query_array = mx.array(queries)
    certificates = []
    original_certified = fused.certified_neighbors

    def capture(train, query, k, values, indices, certificate, **kwargs):
        certificates.extend(certificate.tolist())
        return original_certified(train, query, k, values, indices, certificate, **kwargs)

    monkeypatch.setattr(fused, "certified_neighbors", capture)
    full_values, full_indices = general.gpu_neighbors(index, query_array, k, rows=13, cols=96)
    assert certificates == [True] * len(queries)
    certificates.clear()
    tiled_values, tiled_indices = general.gpu_neighbors(
        index, query_array, k, rows=4, cols=32, bounded_workspace=True,
    )
    assert certificates == [True] * len(queries)
    np.testing.assert_array_equal(np.asarray(tiled_indices), np.asarray(full_indices))
    np.testing.assert_array_equal(np.asarray(tiled_values).view(np.uint32),
                                  np.asarray(full_values).view(np.uint32))


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_explicit_short_tiles_keep_exact_order_and_self_exclusion(device, exclude_self):
    rng = np.random.default_rng(812)
    train = rng.integers(-2, 3, (73, 40)).astype(np.float32)
    train[8:12] = train[7]
    queries = train if exclude_self else train[:19].copy()
    index = fit_neighbors(mx.array(train), device=device)
    actual = index.query(mx.array(queries), 12, query_block_size=9, train_block_size=3,
                         exclude_self=exclude_self)
    squared = ((queries.astype(np.float64)[:, None] - train.astype(np.float64)) ** 2).sum(axis=2)
    if exclude_self:
        np.fill_diagonal(squared, np.inf)
    order = np.argsort(squared, axis=1, kind="stable")[:, :12]
    np.testing.assert_array_equal(np.asarray(actual[1]), order)
    np.testing.assert_allclose(np.asarray(actual[0]),
                               np.sqrt(np.take_along_axis(squared, order, axis=1)), rtol=1e-6)


@pytest.mark.gpu
def test_host_scoped_plan_is_not_overridden_by_later_environment_changes(monkeypatch):
    train = np.arange(32 * 40, dtype=np.float32).reshape(32, 40)
    host = distance.build_index(train, backend="mlx")
    # Host tuning was parsed before this mutation. Its scoped values remain
    # authoritative, including when a standalone query would reject the env.
    monkeypatch.setenv("MLXLEARN_DISTANCE_BLOCK_BYTES", "bad")
    with tuning_context(query_block_rows=4):
        _, indices = host.query(train[:2], 10)
    np.testing.assert_array_equal(indices[:, 0], [0, 1])
    direct = fit_neighbors(mx.array(train))
    _, explicit = direct.query(mx.array(train[:2]), 10, query_block_size=1, train_block_size=3)
    np.testing.assert_array_equal(np.asarray(explicit)[:, 0], [0, 1])


@pytest.mark.gpu
def test_standalone_general_planning_does_not_import_host_libraries():
    source = """
import importlib.abc
import sys
class BlockHostLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'sklearn', 'mlxlearn'):
            raise AssertionError('Host library imported: ' + fullname)
sys.meta_path.insert(0, BlockHostLibraries())
import mlx.core as mx
from mlxlearn_device import fit_neighbors
train = mx.arange(256 * 40, dtype=mx.float32).reshape(256, 40)
for device in ('cpu', 'gpu'):
    distances, indices = fit_neighbors(train, device=device).query(train[:2], 10)
    mx.eval(distances, indices)
    assert indices[:, 0].tolist() == [0, 1]
assert not any(name in sys.modules for name in ('numpy', 'sklearn', 'mlxlearn'))
"""
    subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), check=True, timeout=30)
