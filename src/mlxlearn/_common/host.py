"""Host passes over large arrays, split by rows across a small thread pool.

A NumPy reduction or cast runs on one core, and one core streams a few tens of
GB/s — a fraction of what the memory system of an M-series chip delivers. Before
an MLX fit can start, the host makes several such passes over the caller's
float64 matrix: the float32 range check, the column mean, and the centered
float32 copy. For a fit that converges in one or two L-BFGS iterations those
passes *were* the fit: at 1 000 000 x 32 they took about 30 ms of a 70 ms
logistic regression that scikit-learn finished in 51. NumPy releases the GIL
inside its loops, so row blocks of the same pass run in parallel on threads; on
an M4 Max, eight of them made each pass four to six times faster.

Deterministic by construction. The blocks are fixed by the array's shape — never
by the thread count or by which thread finishes first — and partial results are
combined in block order, so a column mean is the same sum in the same order on
every run and on every machine.

The pool is created on first use and replaced in a forked child, whose copy of it
would have no threads. Its tasks are leaf NumPy calls that never submit work of
their own, so a thread that waits on them cannot deadlock the pool, however many
callers share it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import numpy as np

__all__ = ["row_blocks", "map_row_blocks", "column_mean", "extremes"]

T = TypeVar("T")

#: Elements per block: about 8 MB of float64, large enough that the per-task cost
#: (tens of microseconds) is noise, small enough to keep eight threads busy on a
#: matrix of 100 MB.
_BLOCK_ELEMENTS = 1 << 20

#: Threads. Past eight the passes measured no faster: they are bound by memory, not
#: by cores.
_MAX_THREADS = 8

_pool: ThreadPoolExecutor | None = None
_pool_pid: int | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _pool, _pool_pid
    pid = os.getpid()
    if _pool is None or _pool_pid != pid:
        with _pool_lock:
            if _pool is None or _pool_pid != pid:
                _pool = ThreadPoolExecutor(
                    max_workers=max(1, min(_MAX_THREADS, os.cpu_count() or 1)),
                    thread_name_prefix="mlxlearn-host",
                )
                _pool_pid = pid
    return _pool


def row_blocks(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """``(start, stop)`` row ranges of about ``_BLOCK_ELEMENTS`` elements each.

    A function of the shape alone.
    """
    step = max(1, _BLOCK_ELEMENTS // max(int(n_cols), 1))
    return [(start, min(start + step, n_rows)) for start in range(0, int(n_rows), step)]


def map_row_blocks(
    fn: Callable[[int, int], T], n_rows: int, n_cols: int
) -> list[T]:
    """``[fn(start, stop) for each row block]``, in block order, on the pool.

    A matrix of one block runs inline, on the caller's thread.
    """
    blocks = row_blocks(n_rows, n_cols)
    if len(blocks) <= 1:
        return [fn(start, stop) for start, stop in blocks]
    return list(_executor().map(lambda block: fn(*block), blocks))


def _as_rows(array: np.ndarray) -> np.ndarray | None:
    """``array`` as a 2-D row matrix without copying, or ``None`` when it is not one."""
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return None


def column_mean(X: np.ndarray) -> np.ndarray:
    """``X.mean(axis=0)`` in float64, summed block by block in a fixed order."""
    n_rows, n_cols = X.shape
    parts = map_row_blocks(
        lambda start, stop: np.sum(X[start:stop], axis=0, dtype=np.float64), n_rows, n_cols
    )
    total = parts[0].copy()
    for part in parts[1:]:
        total += part
    return total / max(n_rows, 1)


def extremes(array: np.ndarray) -> tuple[Any, Any]:
    """``(np.max(array), np.min(array))`` of a non-empty numeric array.

    NaN propagates exactly as in the single-threaded pair: a block holding one makes
    its maximum and minimum NaN, and so the combined ones.
    """
    rows = _as_rows(array)
    if rows is None or rows.size < 2 * _BLOCK_ELEMENTS:
        return np.max(array), np.min(array)
    parts = map_row_blocks(
        lambda start, stop: (np.max(rows[start:stop]), np.min(rows[start:stop])),
        rows.shape[0],
        rows.shape[1],
    )
    highs = np.array([high for high, _ in parts])
    lows = np.array([low for _, low in parts])
    return np.max(highs), np.min(lows)
