"""Bounded direct CPU search and certified GPU Gram screening."""

import mlx.core as mx


def _cpu_order(values, indices, k):
    # argsort is stable: establish the secondary index key before distance.
    order = mx.argsort(indices, axis=1)
    values = mx.take_along_axis(values, order, axis=1)
    indices = mx.take_along_axis(indices, order, axis=1)
    order = mx.argsort(values, axis=1)[:, :k]
    return mx.take_along_axis(values, order, axis=1), mx.take_along_axis(indices, order, axis=1)


def cpu_neighbors(train, queries, k, *, rows, cols, exclude_self=False):
    """Exhaustive float64 differences on CPU; no Gram candidate truncation."""
    n, features = train.shape
    # Two float64 feature tensors coexist during direct subtraction/squaring.
    rows = min(rows, max(1, (64 << 20) // (max(features, 1) * 16)))
    cols = min(cols, max(1, (64 << 20) // (rows * features * 16)))
    x64 = train.astype(mx.float64)
    q64 = queries.astype(mx.float64)
    out_values, out_indices = [], []
    for start in range(0, queries.shape[0], rows):
        query = q64[start:start + rows]
        best_values = best_indices = None
        for offset in range(0, n, cols):
            stop = min(n, offset + cols)
            delta = query[:, None, :] - x64[None, offset:stop, :]
            values = mx.sum(delta * delta, axis=2)
            indices = mx.broadcast_to(mx.arange(offset, stop, dtype=mx.int32), values.shape)
            if exclude_self:
                own = mx.arange(start, start + query.shape[0])[:, None]
                values = mx.where(indices == own, float("inf"), values)
            values, indices = _cpu_order(values, indices, min(k, stop - offset))
            if best_values is not None:
                values = mx.concatenate([best_values, values], axis=1)
                indices = mx.concatenate([best_indices, indices], axis=1)
                values, indices = _cpu_order(values, indices, min(k, values.shape[1]))
            best_values, best_indices = values, indices
            mx.eval(best_values, best_indices)
        out_values.append(best_values.astype(mx.float32))
        out_indices.append(best_indices)
    return mx.concatenate(out_values), mx.concatenate(out_indices)


@mx.compile
def _gram_lower(query, train, center, train_sq, error_factor, floor):
    centered = query - center
    norms = mx.sum(centered * centered, axis=1)[:, None] + train_sq[None, :]
    gram = norms - 2.0 * (centered @ train.T)
    safe = (mx.isfinite(norms) & (norms <= 2.0**108) & mx.isfinite(gram)
            & (error_factor < 0.125))
    return mx.where(safe, mx.maximum(gram - error_factor * norms - floor, 0.0), 0.0)


def gpu_neighbors(index, queries, k, *, rows, cols, exclude_self=False):
    """Use Gram values only when an omitted-candidate bound proves safety."""
    from ._neighbors_fused import certified_neighbors, precise_select_candidates, select_topk

    index._prepare_gram()
    train = index.X
    n, features = train.shape
    small_shortlist = 33 <= features <= 256 and k <= 8
    # This margin changes screening cost, not correctness: the same omitted
    # lower bound must still certify membership, or the full scan runs.
    carry = min(n, k + (2 if small_shortlist else 8))
    # Standard dot/norm rounding, centering and final subtraction allowances.
    # Unsafe ranges get a zero lower bound and therefore cannot certify.
    error_factor = (2 * features + 16) * 2.0**-23
    floor = features * 1e-30
    out_values, out_indices = [], []
    for start in range(0, queries.shape[0], rows):
        query = queries[start:start + rows]
        best_values = best_indices = None
        certified = mx.ones((query.shape[0],), mx.bool_)
        own = mx.arange(start, start + query.shape[0])[:, None] if exclude_self else None
        for offset in range(0, n, cols):
            stop = min(offset + cols, n)
            lower = _gram_lower(query, index._train[offset:stop], index._center,
                                index._train_sq[offset:stop], error_factor, floor)
            ids = mx.broadcast_to(mx.arange(offset, stop, dtype=mx.int32), lower.shape)
            if exclude_self:
                lower = mx.where(ids == own, float("inf"), lower)
            width = stop - offset
            take = min(width, carry + 1)
            if take <= 16:
                bounds, candidates = select_topk(lower, take, offset=offset)
            else:
                order = mx.argsort(lower, axis=1)[:, :take]
                bounds = mx.take_along_axis(lower, order, axis=1)
                candidates = mx.take_along_axis(ids, order, axis=1)
            barrier = bounds[:, -1] if width > carry else mx.full((query.shape[0],), float("inf"))
            candidates = candidates[:, :min(carry, width)]
            if exclude_self:
                candidates = mx.where(candidates == own, -1, candidates)
            # A final tile can contain only the excluded identity. An invalid
            # sentinel is retained through the merge, after every real point.
            retained = min(k, width)
            if small_shortlist:
                from ._neighbors_shortlist import refine_certified_candidates

                values, candidates, certified = refine_certified_candidates(
                    train, query, candidates, retained, barrier, certified,
                    check_barrier=width > carry,
                )
            else:
                values, candidates = precise_select_candidates(train, query, candidates, retained)
                if width > carry:
                    upper = values[:, -1] * (1.0 + 6 * 2.0**-23) + floor
                    certified = certified & mx.isfinite(upper) & (barrier > upper)
            if best_values is not None:
                values = mx.concatenate([best_values, values], axis=1)
                candidates = mx.concatenate([best_indices, candidates], axis=1)
                values, candidates = precise_select_candidates(
                    train, query, candidates, min(k, values.shape[1]), values,
                )
            best_values, best_indices = values, candidates
            if stop < n:
                mx.eval(best_values, best_indices, certified)
        values, indices = certified_neighbors(
            train, query, k, best_values, best_indices, certified,
            exclude_self=exclude_self, query_offset=start,
        )
        out_values.append(values)
        out_indices.append(indices)
    return mx.concatenate(out_values), mx.concatenate(out_indices)
