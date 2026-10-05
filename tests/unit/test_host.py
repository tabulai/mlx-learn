"""``_common/host.py``: host passes split by rows across threads."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from mlxlearn._common import host
from mlxlearn._common.arrays import finite_range


def test_blocks_depend_on_the_shape_alone():
    blocks = host.row_blocks(1_000_003, 16)
    assert blocks[0] == (0, host._BLOCK_ELEMENTS // 16)
    assert blocks[-1][1] == 1_000_003
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:], strict=False))
    assert host.row_blocks(10, 4) == [(0, 10)]
    assert host.row_blocks(0, 4) == []


def test_results_come_back_in_block_order_from_the_pool():
    seen = []

    def work(start, stop):
        # Long enough that the blocks overlap: a task that returns at once can let
        # one pool thread drain the whole queue before another wakes.
        time.sleep(0.002)
        seen.append(threading.get_ident())
        return start

    starts = host.map_row_blocks(work, 1_000_000, 32)
    assert starts == [start for start, _ in host.row_blocks(1_000_000, 32)]
    assert len(set(seen)) > 1, "a matrix of many blocks should run on the pool"


def test_the_column_mean_is_numpys_to_rounding_and_reproducible(rng):
    X = rng.normal(loc=3.0, size=(700_001, 8)).astype(np.float32)
    mean = host.column_mean(X)
    assert mean.dtype == np.float64
    np.testing.assert_allclose(mean, X.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(mean, host.column_mean(X))


@pytest.mark.parametrize("poison", [None, np.nan, np.inf, -np.inf])
def test_extremes_are_numpys(poison, rng):
    X = rng.normal(size=(400_000, 12))
    if poison is not None:
        X[123_456, 7] = poison
    high, low = host.extremes(X)
    np.testing.assert_array_equal([high, low], [np.max(X), np.min(X)])
    # And what finite_range makes of them: the finite extremes, or nothing to report.
    expected = X[np.isfinite(X)]
    found = finite_range(X)
    assert (found.low, found.high) == (expected.min(), expected.max())


def test_a_forked_child_gets_a_pool_of_its_own(rng, monkeypatch):
    """A child's copy of the parent's pool has no threads, and submitting to it would
    wait forever, so a process whose id has changed builds its own."""
    X = rng.normal(size=(600_000, 8))
    host.column_mean(X)
    parent = host._executor()
    pid = os.getpid()
    monkeypatch.setattr(host.os, "getpid", lambda: pid + 1)
    child = host._executor()
    assert child is not parent
    np.testing.assert_allclose(host.column_mean(X), X.mean(axis=0), rtol=1e-12)
    monkeypatch.undo()
    assert host._executor() is not child  # and back in the "parent", once more
