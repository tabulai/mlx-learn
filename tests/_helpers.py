"""Assertions shared across the suite."""

from __future__ import annotations

import numpy as np

__all__ = ["assert_neighbor_parity", "logistic_objective"]


def assert_neighbor_parity(mlx_dist, mlx_idx, ref_dist, ref_idx, *, atol=1e-6):
    """Compare neighbor results the way ``docs/fp32_policy.md`` §3.1 prescribes.

    The substantive claim is about **distances**: the k smallest distances from
    each query must be exactly the k smallest, so those are compared elementwise.

    **Indices** are compared as strictly as they are defined, and no more. Exact
    distance ties make two answers equally correct in two distinct ways:

    * *Order* ties — two returned neighbors at the same distance. scikit-learn's
      brute-force path sorts its candidates with a non-stable sort and defines no
      order among them; mlxlearn defines ``(distance, index)``. Compared only
      where distances are strictly separated.
    * *Boundary* ties — the k-th and (k+1)-th candidates at the same distance, so
      which one makes the cut is arbitrary. Measured on 50 000 points quantized to
      float32 at coordinate scale 1e4, 3 of 200 rows hit this. Membership is
      therefore compared for every neighbor strictly closer than the row's largest
      returned distance, and any disagreement beyond that must sit exactly on that
      distance.

    Both relaxations are checkable rather than assumed: a disagreement that is not
    a tie still fails.
    """
    mlx_idx = np.asarray(mlx_idx)
    ref_idx = np.asarray(ref_idx)
    assert mlx_idx.shape == ref_idx.shape, f"{mlx_idx.shape} != {ref_idx.shape}"

    if mlx_dist is None:
        np.testing.assert_array_equal(
            np.sort(mlx_idx, axis=1), np.sort(ref_idx, axis=1), err_msg="neighbor sets differ"
        )
        return

    mlx_dist = np.asarray(mlx_dist)
    ref_dist = np.asarray(ref_dist)
    np.testing.assert_allclose(
        mlx_dist, ref_dist, atol=atol, rtol=atol, err_msg="neighbor distances differ"
    )

    # Membership: everything strictly inside the boundary must agree.
    for row in range(mlx_idx.shape[0]):
        boundary = ref_dist[row, -1]
        mine = set(mlx_idx[row].tolist())
        theirs = set(ref_idx[row].tolist())
        if mine == theirs:
            continue
        for missing in theirs - mine:
            position = int(np.where(ref_idx[row] == missing)[0][0])
            assert abs(ref_dist[row, position] - boundary) <= atol, (
                f"row {row}: scikit-learn returned index {missing} at distance "
                f"{ref_dist[row, position]} but mlxlearn did not, and it is not a "
                f"boundary tie (boundary={boundary})"
            )
        for extra in mine - theirs:
            position = int(np.where(mlx_idx[row] == extra)[0][0])
            assert abs(mlx_dist[row, position] - boundary) <= atol, (
                f"row {row}: mlxlearn returned index {extra} at distance "
                f"{mlx_dist[row, position]}, which scikit-learn did not, and it is "
                f"not a boundary tie (boundary={boundary})"
            )

    # Order: exact wherever the distances are strictly separated *and* both
    # results agree that the neighbor belongs in the set at all. A boundary tie
    # is invisible from inside the returned row — the (k+1)-th distance is not
    # there to compare against — so a rank occupied by a set disagreement is
    # excluded here rather than being mistaken for an ordering bug.
    separated = np.ones(mlx_idx.shape, dtype=bool)
    tied = np.isclose(np.diff(ref_dist, axis=1), 0.0, atol=atol)
    separated[:, :-1] &= ~tied
    separated[:, 1:] &= ~tied

    shared = np.zeros(mlx_idx.shape, dtype=bool)
    for row in range(mlx_idx.shape[0]):
        common = np.intersect1d(mlx_idx[row], ref_idx[row])
        shared[row] = np.isin(mlx_idx[row], common) & np.isin(ref_idx[row], common)

    violations = separated & shared & (mlx_idx != ref_idx)
    assert not violations.any(), (
        "index order differs where distances are strictly separated at "
        f"{np.argwhere(violations)[:5].tolist()}"
    )


def assert_ties_break_on_smallest_index(dist, idx, *, atol=1e-6):
    """mlxlearn's tie rule: among equal distances, the smaller index comes first.

    A guarantee scikit-learn does not make, so it is asserted directly rather than
    inferred from a comparison against scikit-learn.
    """
    dist = np.asarray(dist)
    idx = np.asarray(idx)
    tied = np.isclose(np.diff(dist, axis=1), 0.0, atol=atol)
    ascending = np.diff(idx, axis=1) > 0
    assert np.all(ascending[tied]), "tied neighbors are not ordered by ascending index"


def logistic_objective(coef, intercept, X, y_encoded, *, C, classes, sample_weight=None):
    """scikit-learn's regularized logistic objective, in float64.

    Used to compare two solvers fairly. Comparing ``coef_`` alone is the weak
    assertion — on an ill-conditioned problem two correct solvers land in
    different places — so every coefficient comparison is paired with this.
    """
    X = np.asarray(X, dtype=np.float64)
    coef = np.atleast_2d(np.asarray(coef, dtype=np.float64))
    intercept = np.atleast_1d(np.asarray(intercept, dtype=np.float64))
    scores = X @ coef.T + intercept
    weights = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, float)

    if len(classes) == 2:
        z = scores[:, 0]
        signs = np.where(y_encoded == 1, 1.0, -1.0)
        loss = np.sum(weights * np.logaddexp(0.0, -signs * z))
    else:
        shifted = scores - scores.max(axis=1, keepdims=True)
        log_norm = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss = -np.sum(weights * log_norm[np.arange(len(y_encoded)), y_encoded])

    return C * loss + 0.5 * np.sum(coef * coef)
