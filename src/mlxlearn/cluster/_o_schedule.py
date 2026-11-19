"""The o-means training schedule: e-means' Faiss recipe plus the cell rules.

o-means keeps the batch law, step law, cosine schedule, epoch plan and revival
cadence of e-means, and adds the coarse cells and probes of its two-level
assignment, the cadence at which the cells are rebuilt, and each batch's
weight in the closing mean. The rules follow ``contrib/o_means/schedule.py``
of owlgebra-ai/faiss (MIT), accessed 2026-11-18.
"""

from __future__ import annotations

import math
from numbers import Integral

from ._recipe import _positive_real, epoch_rows, resolve_recipe

CELL_SIZE = 128
# Below this many epochs the closing mean weights every batch by its recency;
# from here up it is e-means' mean over the last pass.
RECENCY_CLOSE_BELOW = 2.0
# The throughput batch law up to this many rows, the quality law above.
THROUGHPUT_MAX_ROWS = 2_000_000
FIRST_BUILD_ITERATIONS = 4
FIRST_PASS_REBUILD_ITERATIONS = 0
REBUILD_ITERATIONS = 2


def num_cells(k: int) -> int:
    """Coarse cells for ``k`` centroids: ``k // 128`` rounded up to a multiple of 16."""
    return min(max(16, -(-(k // CELL_SIZE) // 16) * 16), k)


def num_probes(k: int, d: int, cells: int) -> int:
    """Cells each row searches, fitted upstream to about 1% missed neighbours."""
    probes = 24 if k > 16384 else (12 if d >= 512 else 16)
    return min(probes, cells)


def _cell_count(name, value, upper):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer between 1 and {upper}")
    return int(value)


def resolve_schedule(
    n: int,
    k: int,
    d: int,
    epochs: float,
    batch_size: int | None = None,
    alpha0: float | None = None,
    batch_law: str = "auto",
    n_cells: int | None = None,
    n_probe: int | None = None,
    unused_threshold: float | None = None,
) -> dict:
    """Resolve the e-means recipe and the o-means additions for one fit.

    Below one epoch the one-epoch batch and step sizes are kept and the
    cosine schedule spans the shortened run. ``auto`` selects the throughput
    law up to two million rows, unlike the Faiss e-means rule.
    """
    _positive_real("max_iter", epochs)
    if batch_law not in {"auto", "quality", "throughput"}:
        raise ValueError("batch_law must be 'auto', 'quality', or 'throughput'")
    if batch_law == "auto" and isinstance(n, Integral) and not isinstance(n, bool):
        law = "throughput" if n <= THROUGHPUT_MAX_ROWS else "quality"
        reason = "o-means: n <= 2M" if law == "throughput" else "o-means: n > 2M"
    else:
        law, reason = batch_law, "requested"
    recipe = resolve_recipe(n, k, d, max(float(epochs), 1.0), batch_size, alpha0, law,
                            "faiss", unused_threshold)
    batch = recipe["batch_size"]
    rows = epoch_rows(n, epochs)
    if sum(rows) == 0:
        raise ValueError("max_iter is too small to include even one training sample")
    steps = [(r + batch - 1) // batch for r in rows]
    cells = num_cells(k) if n_cells is None else _cell_count("n_cells", n_cells, k)
    probes = (num_probes(k, d, cells) if n_probe is None
              else _cell_count("n_probe", n_probe, cells))
    recency = float(epochs) < RECENCY_CLOSE_BELOW
    recipe.update(
        algorithm="o-means",
        batch_law_requested=batch_law,
        batch_law_reason=reason,
        nepochs=float(epochs),
        epoch_rows=rows,
        epoch_steps=steps,
        total_steps=sum(steps),
        n_cells=cells,
        n_probe=probes,
        # Half a pass, rounded up, and never more often than revival checks.
        rebuild_every=max(recipe["check_unused_every"], ((n + batch - 1) // batch + 1) // 2),
        first_build_iterations=FIRST_BUILD_ITERATIONS,
        first_pass_rebuild_iterations=FIRST_PASS_REBUILD_ITERATIONS,
        rebuild_iterations=REBUILD_ITERATIONS,
        closing="recency_weighted" if recency else "last_pass",
    )
    return recipe


def closing_weight(recipe: dict, n: int, epoch: int, rows_done: int) -> float | None:
    """Weight of a batch in the closing mean, or None when it is not part of it.

    ``rows_done`` counts every row processed up to and including the batch.
    The recency weight is ``exp(-(rows processed afterwards) / n)``.
    """
    if recipe["closing"] == "recency_weighted":
        return math.exp(-((sum(recipe["epoch_rows"]) - rows_done) / n))
    return 1.0 if epoch == len(recipe["epoch_rows"]) - 1 else None


def expected_fraction(k: int, cells: int, probes: int) -> float:
    """Distances a row evaluates, relative to ``k``, for equal-size cells."""
    return (cells + probes * (k / cells)) / k
