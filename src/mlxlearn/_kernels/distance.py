"""Blocked exact Euclidean distance and k-nearest-neighbor selection in MLX.

The whole computation stays on the device. The ancestor materialized each tile's
top-k back to the host and merged it in a Python loop, which meant the GPU did the
matmul and the interpreter did everything else; for a hundred thousand queries and
a hundred tiles that is on the order of 10⁸ interpreted iterations, and it is why
its measured KNN speedups ranged down to 0.4×. Here, the running top-k is an MLX
array, tiles are merged with a concatenate-and-repartition on the device, and the
only host transfer is the final ``(n_query, k)`` result — one copy per query block.

Three correctness properties this module is responsible for:

**Exactness under float32.** Squared distances come from the expanded identity
``‖q‖² + ‖t‖² − 2⟨q,t⟩``, which is one matmul but loses catastrophic precision when
the data sits far from the origin. Mitigated by translating both matrices by the
training mean (exact — distances are translation-invariant), over-selecting
candidates, and recomputing the survivors' distances directly. See
``docs/fp32_policy.md`` §2.1 for what remains.

**A total order.** Results are ordered by ``(distance, index)`` ascending, so ties
break on the smaller training index exactly as scikit-learn's stable sort does, and
so results do not depend on the block sizes.

**Blocking invariance.** Any combination of query- and train-block sizes yields
identical output. The tests sweep this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import physical_memory_bytes, use_device

__all__ = ["NeighborIndex", "build_index", "resolve_blocks"]

#: Extra candidates carried per tile beyond ``k``. Cheap insurance against a
#: candidate that residual float32 noise in the Gram identity ranked just outside
#: the cut; the refinement pass then orders the enlarged set exactly.
_OVERSELECT = 4


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """How a query is split into tiles."""

    query_rows: int
    train_cols: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"query_rows={self.query_rows}, train_cols={self.train_cols}"


def _distance_budget_bytes(backend: str) -> int:
    """Bytes to spend on one distance tile.

    On unified memory a bigger tile means fewer, wider matmuls, which is what the
    GPU wants. The budget is derived from installed memory and clamped, because a
    fixed constant is wrong by an order of magnitude on both an 8 GB Air and a
    128 GB Studio.
    """
    tuning = get_tuning()
    if backend != "mlx":
        return tuning.cpu_distance_block_bytes
    total = physical_memory_bytes()
    if total is None:
        return tuning.distance_block_bytes
    return int(
        min(max(total // 32, tuning.distance_block_bytes), tuning.distance_block_bytes_max)
    )


def resolve_blocks(
    n_query: int,
    n_train: int,
    *,
    backend: str,
    k_search: int,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
) -> BlockPlan:
    """Choose tile sizes so a distance tile fits the budget.

    Unlike the ancestor, the device kind is passed in rather than inferred from
    whether the byte budget happens to exceed a GPU-ish constant — conflating a
    memory budget with a device kind meant an explicit CPU budget above 256 MiB
    silently switched on GPU alignment, and the CPU path could never shrink its
    block no matter how wide the training set was.
    """
    if n_query <= 0 or n_train <= 0:
        return BlockPlan(query_rows=0, train_cols=0)

    tuning = get_tuning()
    budget = _distance_budget_bytes(backend)
    itemsize = 4

    if query_block_size is not None:
        rows = max(1, min(int(query_block_size), n_query))
    else:
        rows = tuning.query_block_rows if backend == "mlx" else tuning.cpu_query_block_rows
        # Keep a full-width tile inside the budget when the training set is small
        # enough to be scanned in one pass; otherwise the train tiling handles it.
        max_rows = budget // max(n_train * itemsize, 1)
        if 0 < max_rows < rows:
            align = tuning.query_block_align if backend == "mlx" else 1
            aligned = (max_rows // align) * align if align > 1 else max_rows
            rows = aligned if aligned > 0 else max_rows
        rows = max(1, min(n_query, rows))

    min_cols = min(k_search, n_train)
    if train_block_size is not None:
        cols = max(min_cols, min(int(train_block_size), n_train))
    else:
        cols = budget // max(rows * itemsize, 1)
        cols = max(min_cols, min(n_train, int(cols)))

    return BlockPlan(query_rows=int(rows), train_cols=int(cols))


@dataclass(slots=True)
class NeighborIndex:
    """Device-resident training data for repeated neighbor queries.

    Owned by the fitted estimator and released with it. Deliberately *not* a
    module-level cache keyed on the identity of a user's array: the ancestor did
    that and it produced stale results whenever a caller mutated an array in
    place, leaked a full float32 copy plus its transpose for the lifetime of the
    process, and was not thread-safe. An agent refitting a probe on new data must
    never get the previous fit's coefficients back.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        A float32 C-ordered *copy* of the training data. A copy, not a view: an
        estimator that aliases the caller's buffer changes its own predictions
        when the caller mutates their array.
    center : ndarray of shape (n_features,)
        The training mean, subtracted from both sides before the Gram identity.
        Exact — distances are translation-invariant — and it is what keeps float32
        cancellation manageable.
    backend : {"mlx", "cpu"}
        Which MLX device the tensors live on and queries run on.
    """

    X: np.ndarray
    center: np.ndarray
    backend: str
    _train: object = None  # mx.array, centered, (n, d)
    _train_t: object = None  # mx.array, (d, n)
    _train_sq: object = None  # mx.array, (n,)

    # ------------------------------------------------------------------ build

    def __post_init__(self) -> None:
        self._ensure_device()

    def _ensure_device(self) -> None:
        """Upload the centered training matrix and its row norms.

        Idempotent, so it doubles as the rebuild path after unpickling.
        """
        if self._train is not None:
            return

        import mlx.core as mx

        with use_device("gpu" if self.backend == "mlx" else "cpu"):
            centered = np.ascontiguousarray(self.X - self.center, dtype=NUMPY_FLOAT)
            train = mx.array(centered, dtype=mx.float32)
            self._train = train
            self._train_t = train.T
            self._train_sq = mx.sum(train * train, axis=1)
            mx.eval(self._train, self._train_t, self._train_sq)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    # ------------------------------------------------------------------ pickle

    def __getstate__(self) -> dict:
        """Pickle the host data only; the device tensors are derived.

        Keeping them would roughly triple the pickle — the matrix, its transpose,
        and its row norms are all recomputable from ``X`` and ``center`` — and
        would embed device buffers in a file that may be loaded on a machine with
        a different MLX build or no GPU at all.
        """
        return {"X": self.X, "center": self.center, "backend": self.backend}

    def __setstate__(self, state: dict) -> None:
        self.X = state["X"]
        self.center = state["center"]
        self.backend = state["backend"]
        self._train = None
        self._train_t = None
        self._train_sq = None
        # Rebuilt lazily on the first query rather than here, so that unpickling
        # on a machine without a GPU does not fail until something actually asks
        # for a computation.

    # ------------------------------------------------------------------ query

    def query(
        self,
        Q: np.ndarray,
        k: int,
        *,
        return_distance: bool = True,
        exclude_self: bool = False,
        query_block_size: int | None = None,
        train_block_size: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return the ``k`` nearest training rows for each row of ``Q``.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Query rows, float32.
        k : int
            Neighbors to return, before ``exclude_self`` removes one.
        return_distance : bool, default=True
            Whether to compute and return Euclidean distances.
        exclude_self : bool, default=False
            Treat ``Q`` as the training matrix itself and drop each row's own
            index. Requires ``len(Q) == n_samples``.

        Returns
        -------
        distances : ndarray of shape (n_queries, k) or None
            Euclidean (not squared) distances, float64, ascending.
        indices : ndarray of shape (n_queries, k)
            Training-row indices, int64, ordered by ``(distance, index)``.

        Notes
        -----
        ``exclude_self`` compares against each row's **global** index. The
        ancestor compared against the block-local row number, so every query block
        after the first tested the wrong identity and a row could be returned as
        its own nearest neighbor — verified on 1500 rows with default blocking.
        """
        import mlx.core as mx

        self._ensure_device()
        n_query = int(Q.shape[0])
        n_train = self.n_samples
        k = int(k)

        if exclude_self and n_query != n_train:
            raise ValueError(
                "exclude_self requires the query matrix to be the training matrix: "
                f"got {n_query} query rows and {n_train} training rows."
            )
        k_search = k + int(exclude_self)
        if k_search > n_train:
            limit = n_train - int(exclude_self)
            raise ValueError(f"n_neighbors must be <= {limit}, got {k}.")

        if n_query == 0:
            empty_idx = np.empty((0, k), dtype=INDEX_DTYPE)
            empty_dist = np.empty((0, k), dtype=np.float64) if return_distance else None
            return empty_dist, empty_idx

        # Carry extra candidates so residual Gram noise cannot push a true
        # neighbor out of the set before refinement can rank it.
        k_carry = min(n_train, k_search + _OVERSELECT)

        plan = resolve_blocks(
            n_query,
            n_train,
            backend=self.backend,
            k_search=k_carry,
            query_block_size=query_block_size,
            train_block_size=train_block_size,
        )

        out_idx = np.empty((n_query, k), dtype=INDEX_DTYPE)
        out_dist = np.empty((n_query, k), dtype=np.float64) if return_distance else None

        with use_device("gpu" if self.backend == "mlx" else "cpu"):
            Qc = mx.array(
                np.ascontiguousarray(Q - self.center, dtype=NUMPY_FLOAT), dtype=mx.float32
            )
            q_sq_all = mx.sum(Qc * Qc, axis=1)

            for start in range(0, n_query, plan.query_rows):
                end = min(start + plan.query_rows, n_query)
                block = Qc[start:end]
                q_sq = q_sq_all[start:end, None]

                cand_idx, _ = self._block_candidates(
                    mx, block, q_sq, n_train, k_carry, plan.train_cols
                )
                refined = self._refine(mx, block, cand_idx)

                # The final ordering is a lexicographic sort on (distance, index)
                # over a (block, k_carry) array -- small enough that host-side
                # np.lexsort is both simpler and faster than any device trick, and
                # it is the one place a *stable, total* order is guaranteed.
                mx.eval(cand_idx, refined)
                idx_np = np.asarray(cand_idx, dtype=INDEX_DTYPE)
                dist_np = np.asarray(refined, dtype=NUMPY_FLOAT)
                order = np.lexsort((idx_np, dist_np), axis=-1)
                rows = np.arange(idx_np.shape[0])[:, None]
                idx_np = idx_np[rows, order]
                dist_np = dist_np[rows, order]

                if exclude_self:
                    idx_np, dist_np = _drop_self(idx_np, dist_np, offset=start)

                out_idx[start:end] = idx_np[:, :k]
                if return_distance:
                    out_dist[start:end] = np.sqrt(
                        np.maximum(dist_np[:, :k].astype(np.float64), 0.0)
                    )

        return out_dist, out_idx

    def query_radius(
        self,
        Q: np.ndarray,
        radius: float,
        *,
        return_distance: bool = True,
        exclude_self: bool = False,
        sort_results: bool = False,
        query_block_size: int | None = None,
        train_block_size: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return every training row within ``radius`` of each row of ``Q``.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Query rows, float32.
        radius : float
            Inclusive Euclidean radius. Negative values are rejected.
        return_distance : bool, default=True
            Whether to return Euclidean distances alongside indices.
        exclude_self : bool, default=False
            Drop each query's own training index. Requires ``len(Q) == n_samples``.
        sort_results : bool, default=False
            If True, each row is ordered by ``(distance, index)``. If False the
            order is still deterministic — the same total order — because an
            unspecified order would make block size and device visible in the
            output. scikit-learn does not define an order unless this flag is
            set; we always have one.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries,) of 1-d arrays
            Euclidean distances, float64. Only when ``return_distance=True``.
        neigh_ind : ndarray of shape (n_queries,) of 1-d arrays
            Training-row indices, int64.

        Notes
        -----
        Membership is decided on *refined* squared distances, not on the Gram
        identity used to gather candidates. The Gram pass uses a small margin so
        a neighbor sitting on the radius is not lost to float32 cancellation;
        the refinement pass then keeps ``√Σ(t − q)² ≤ radius``.
        """
        import mlx.core as mx

        self._ensure_device()
        n_query = int(Q.shape[0])
        n_train = self.n_samples
        radius = float(radius)
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")

        if exclude_self and n_query != n_train:
            raise ValueError(
                "exclude_self requires the query matrix to be the training matrix: "
                f"got {n_query} query rows and {n_train} training rows."
            )

        if n_query == 0:
            empty_ind = np.empty(0, dtype=object)
            if return_distance:
                return np.empty(0, dtype=object), empty_ind
            return empty_ind

        radius_sq = radius * radius
        # Absolute + relative margin on the Gram identity so a true neighbor at
        # the radius is still a candidate for refinement. 4 ulp of the squared
        # radius is the float32 spacing; the extra 1e-4 covers residual
        # cancellation after centering.
        gram_margin = max(1e-4, float(np.finfo(np.float32).eps) * max(radius_sq, 1.0) * 8.0)
        candidate_sq = radius_sq + gram_margin

        plan = resolve_blocks(
            n_query,
            n_train,
            backend=self.backend,
            k_search=1,
            query_block_size=query_block_size,
            train_block_size=train_block_size,
        )

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        dist_parts: list[np.ndarray] = []

        with use_device("gpu" if self.backend == "mlx" else "cpu"):
            Qc = mx.array(
                np.ascontiguousarray(Q - self.center, dtype=NUMPY_FLOAT), dtype=mx.float32
            )
            q_sq_all = mx.sum(Qc * Qc, axis=1)

            for q_start in range(0, n_query, plan.query_rows):
                q_end = min(q_start + plan.query_rows, n_query)
                block = Qc[q_start:q_end]
                q_sq = q_sq_all[q_start:q_end, None]

                for t_start in range(0, n_train, plan.train_cols):
                    t_end = min(t_start + plan.train_cols, n_train)
                    tile = (
                        q_sq
                        + self._train_sq[t_start:t_end][None, :]
                        - 2.0 * (block @ self._train_t[:, t_start:t_end])
                    )
                    tile = mx.maximum(tile, 0.0)
                    cand = tile <= candidate_sq
                    mx.eval(cand)
                    cand_np = np.asarray(cand)
                    local_rows, local_cols = np.nonzero(cand_np)
                    if local_rows.size == 0:
                        continue

                    global_cols = local_cols.astype(INDEX_DTYPE, copy=False) + np.int64(t_start)
                    global_rows = local_rows.astype(INDEX_DTYPE, copy=False) + np.int64(q_start)
                    if exclude_self:
                        keep = global_cols != global_rows
                        local_rows = local_rows[keep]
                        global_cols = global_cols[keep]
                        global_rows = global_rows[keep]
                        if local_rows.size == 0:
                            continue

                    # Direct Σ(t − q)² on the surviving (query, train) pairs.
                    query_pts = mx.take(block, mx.array(local_rows.astype(np.int32)), axis=0)
                    train_pts = mx.take(
                        self._train, mx.array(global_cols.astype(np.int32)), axis=0
                    )
                    diff = train_pts - query_pts
                    refined = mx.sum(diff * diff, axis=1)
                    mx.eval(refined)
                    refined_np = np.asarray(refined, dtype=np.float64)
                    inside = refined_np <= radius_sq
                    if not np.any(inside):
                        continue

                    row_parts.append(global_rows[inside])
                    col_parts.append(global_cols[inside])
                    dist_parts.append(np.sqrt(np.maximum(refined_np[inside], 0.0)))

        # Always apply a total order so tile sizes cannot leak into the output.
        # sort_results=False is still deterministic; scikit-learn does not
        # define an order unless the flag is set.
        sort_results = True
        neigh_ind, neigh_dist = _pack_radius_results(
            n_query, row_parts, col_parts, dist_parts, sort_results=sort_results
        )
        if return_distance:
            return neigh_dist, neigh_ind
        return neigh_ind

    # ----------------------------------------------------------------- internals

    def _block_candidates(self, mx, block, q_sq, n_train: int, k_carry: int, train_cols: int):
        """Running top-``k_carry`` over train tiles, entirely on device."""
        best_idx = None
        best_val = None

        for t_start in range(0, n_train, train_cols):
            t_end = min(t_start + train_cols, n_train)
            width = t_end - t_start

            tile = (
                q_sq
                + self._train_sq[t_start:t_end][None, :]
                - 2.0 * (block @ self._train_t[:, t_start:t_end])
            )
            tile = mx.maximum(tile, 0.0)

            take = min(k_carry, width)
            # One partition pass, not `take` sequential argmin sweeps over the
            # whole tile. The ancestor's iterated argmin also mutated its input in
            # place, so a function that read as pure selection destroyed its
            # argument -- harmless only for as long as the argument stayed a
            # throwaway temporary.
            local = mx.argpartition(tile, kth=take - 1, axis=1)[:, :take]
            rows = mx.arange(tile.shape[0])[:, None]
            vals = tile[rows, local]
            idx = local.astype(mx.int64) + t_start

            if best_idx is None:
                best_idx, best_val = idx, vals
            else:
                merged_idx = mx.concatenate([best_idx, idx], axis=1)
                merged_val = mx.concatenate([best_val, vals], axis=1)
                keep = min(k_carry, merged_val.shape[1])
                sel = mx.argpartition(merged_val, kth=keep - 1, axis=1)[:, :keep]
                mrows = mx.arange(merged_val.shape[0])[:, None]
                best_idx = merged_idx[mrows, sel]
                best_val = merged_val[mrows, sel]

        return best_idx, best_val

    def _refine(self, mx, block, cand_idx):
        """Recompute the candidates' squared distances directly.

        ``Σ(t − q)²`` instead of the expanded identity. Subtracting nearby float32
        values is exact or nearly so, so these values are accurate enough both to
        rank the candidates the way scikit-learn does and to be reported.
        Unlike the ancestor -- where the MLX version of this was written and then
        never called -- this runs on device.
        """
        cand = mx.take(self._train, cand_idx, axis=0)  # (B, k, d)
        diff = cand - block[:, None, :]
        return mx.sum(diff * diff, axis=2)


def _drop_self(idx: np.ndarray, dist: np.ndarray, *, offset: int):
    """Remove each query's own training index from its neighbor list.

    ``offset`` converts the block-local row number into the global row index. That
    single term is the difference between a correct self-query and the ancestor's,
    which returned a row as its own nearest neighbor whenever the training set
    exceeded one query block.

    A row with no self match — possible when duplicate points push the self index
    past the retained candidates — drops its nearest neighbor instead, which is
    what scikit-learn does.
    """
    n_rows, width = idx.shape
    global_rows = (np.arange(n_rows, dtype=idx.dtype) + offset)[:, None]
    mask = idx != global_rows

    missing = np.all(mask, axis=1)
    if np.any(missing):
        mask[missing, 0] = False

    keep = width - 1
    out_idx = idx[mask].reshape(n_rows, keep)
    out_dist = dist[mask].reshape(n_rows, keep)
    return out_idx, out_dist


def _pack_radius_results(
    n_query: int,
    row_parts: list[np.ndarray],
    col_parts: list[np.ndarray],
    dist_parts: list[np.ndarray],
    *,
    sort_results: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ragged ``(n_query,)`` object arrays from concatenated triples.

    Rows are ordered by ``(distance, index)`` when ``sort_results`` is true —
    a total order, so the result does not depend on tile sizes.
    """
    neigh_ind = np.empty(n_query, dtype=object)
    neigh_dist = np.empty(n_query, dtype=object)
    empty_idx = np.empty(0, dtype=INDEX_DTYPE)
    empty_dist = np.empty(0, dtype=np.float64)

    if not row_parts:
        for i in range(n_query):
            neigh_ind[i] = empty_idx
            neigh_dist[i] = empty_dist
        return neigh_ind, neigh_dist

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts).astype(INDEX_DTYPE, copy=False)
    dists = np.concatenate(dist_parts).astype(np.float64, copy=False)

    if sort_results:
        order = np.lexsort((cols, dists, rows))
    else:
        order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    cols = cols[order]
    dists = dists[order]

    counts = np.bincount(rows, minlength=n_query)
    starts = np.cumsum(np.concatenate(([0], counts)))
    for i in range(n_query):
        sl = slice(int(starts[i]), int(starts[i + 1]))
        neigh_ind[i] = cols[sl]
        neigh_dist[i] = dists[sl]
    return neigh_ind, neigh_dist


def build_index(X: np.ndarray, *, backend: str) -> NeighborIndex:
    """Build a :class:`NeighborIndex` from a training matrix.

    ``X`` is copied to float32 C order. The copy is deliberate: an index that
    aliases the caller's array silently changes its own answers when that array is
    mutated.
    """
    Xf = np.array(X, dtype=NUMPY_FLOAT, order="C", copy=True)
    center = Xf.mean(axis=0, dtype=np.float64).astype(NUMPY_FLOAT)
    return NeighborIndex(X=Xf, center=center, backend=backend)
