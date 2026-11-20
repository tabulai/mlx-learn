"""Memory planning shared by NumPy-facing and standalone MLX neighbor search."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class PlanSettings:
    distance_block_bytes: int = 256 << 20
    distance_block_bytes_max: int = 1024 << 20
    query_block_rows: int = 1024
    query_block_align: int = 128
    cpu_distance_block_bytes: int = 128 << 20
    cpu_query_block_rows: int = 256


DEFAULTS = PlanSettings()


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """Default tile and optional taller tile for the calibrated GPU Gram search."""

    query_rows: int
    train_cols: int
    tight_gram_tile: tuple[int, int] | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"query_rows={self.query_rows}, train_cols={self.train_cols}"


@lru_cache(maxsize=1)
def physical_memory_bytes() -> int | None:
    """Physical memory without loading an array library or probing the GPU."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * page_count if page_size > 0 and page_count > 0 else None


def settings_from_env() -> PlanSettings:
    """Standalone-device settings; host wrappers pass their scoped tuning instead."""
    overrides = {}
    for field in fields(PlanSettings):
        key = "MLXLEARN_" + field.name.upper()
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key}={raw!r} is not a valid value for {field.name}") from exc
        if value < 0:
            raise ValueError(f"tuning value {field.name} must be non-negative, got {value}")
        overrides[field.name] = value
    return PlanSettings(**overrides) if overrides else DEFAULTS


def distance_budget_bytes(backend: str, settings: PlanSettings, total_memory: int | None) -> int:
    """Distance workspace target, capped at the configured maximum on GPUs."""
    if backend != "mlx":
        return settings.cpu_distance_block_bytes
    if total_memory is None:
        return settings.distance_block_bytes
    return int(min(max(total_memory // 32, settings.distance_block_bytes),
                   settings.distance_block_bytes_max))


def resolve_blocks(
    n_query: int,
    n_train: int,
    *,
    backend: str,
    k_search: int,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
    settings: PlanSettings | None = None,
    budget_bytes: int | None = None,
) -> BlockPlan:
    """Choose distance tiles from one memory budget and device-specific row policy.

    Sixteen bytes per pair cover values, ordering keys, and selection indices.
    The target is a workspace budget, not a limit on resident input/output.
    Explicit tile sizes and the minimum candidate width can exceed it, as in
    the host interface's historical policy. CPU alignment is always one even
    when the caller gives it a large budget.
    """
    if n_query <= 0 or n_train <= 0:
        return BlockPlan(0, 0)
    settings = settings_from_env() if settings is None else settings
    if budget_bytes is None:
        budget_bytes = distance_budget_bytes(backend, settings, physical_memory_bytes())
    itemsize = 16
    if query_block_size is not None:
        rows = max(1, min(int(query_block_size), n_query))
    else:
        rows = settings.query_block_rows if backend == "mlx" else settings.cpu_query_block_rows
        max_rows = budget_bytes // max(n_train * itemsize, 1)
        if 0 < max_rows < rows:
            align = settings.query_block_align if backend == "mlx" else 1
            aligned = (max_rows // align) * align if align > 1 else max_rows
            rows = aligned if aligned > 0 else max_rows
        rows = max(1, min(n_query, rows))
    min_cols = min(k_search, n_train)
    if train_block_size is not None:
        cols = max(min_cols, min(int(train_block_size), n_train))
    else:
        cols = max(min_cols, min(n_train, budget_bytes // (rows * itemsize)))
    gram_tile = None
    if (backend == "mlx" and query_block_size is None and train_block_size is None
            and settings.query_block_align == DEFAULTS.query_block_align
            and n_query >= 32 and n_train >= 2**19 and k_search <= 32 + 4
            and (64 << 20) <= budget_bytes <= (512 << 20)):
        # A full-width tile leaves too few queries per matrix product. Trade
        # columns for rows at the same 16 bytes/pair, keeping enough columns
        # for efficient filtered selection even at k=32.
        gram_rows = min(n_query, settings.query_block_rows, 128,
                        budget_bytes // (itemsize * 2**16))
        if gram_rows > rows:
            gram_tile = (gram_rows, min(n_train, budget_bytes // (itemsize * gram_rows)))
    return BlockPlan(int(rows), int(cols), gram_tile)
