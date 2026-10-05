"""Assertions shared across the suite."""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    "assert_neighbor_parity",
    "join_or_fail",
    "logistic_data",
    "logistic_newton_oracle",
    "logistic_objective",
]


def join_or_fail(threads, *, timeout: float = 120.0) -> None:
    """Join ``threads`` against one deadline, and fail if any of them is still running.

    For worker threads started with ``daemon=True``: a deadlock — in the device
    gate, or an index's lock — then fails the test that caused it, by name,
    rather than hanging the session until the CI job's own timeout, and a stuck
    worker does not keep the interpreter from exiting afterwards either.
    """
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    stuck = sum(thread.is_alive() for thread in threads)
    assert not stuck, f"{stuck} of {len(threads)} threads still running after {timeout:g} s"


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
    """scikit-learn's regularized logistic objective, in float64, in liblinear's scaling.

    ``C·Σ swᵢ·lossᵢ + ½‖W‖²``, which is ``C·sw_sum`` times the objective the parity
    tests compare (:func:`mlxlearn._solvers.logistic.reference_objective`), and
    infinite for ``C = inf``. It is deliberately written differently from that
    function — signed margins for binary, an explicit log-softmax for multiclass —
    because its job is to check the oracle the parity tests rely on, which would
    otherwise be the library grading itself.
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


# --------------------------------------------------------------- logistic test data


def _ar_features(n, d, rho, rng):
    """AR(1) columns: ``corr(xᵢ, xⱼ) = ρ^|i−j|``, unit variance."""
    Z = rng.normal(size=(n, d))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    c = np.sqrt(1.0 - rho * rho)
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + c * Z[:, j]
    return X


def _linear_labels(X, n_classes, rng, noise=0.5):
    """Two classes: the sign of a random direction plus noise. More: Gumbel-max of a
    random linear score — a multinomial logit model."""
    d = X.shape[1]
    if n_classes == 2:
        s = X @ rng.normal(size=d)
        return (s + rng.normal(scale=noise * s.std(), size=len(s)) > 0).astype(int)
    W = rng.normal(size=(d, n_classes))
    return (X @ W / (np.sqrt(d) / 2) + rng.gumbel(size=(len(X), n_classes))).argmax(axis=1)


def logistic_data(law, n, d, n_classes=2, seed=0):
    """``(X, y)`` for the logistic preconditioner's tests, by data law.

    ``iso``: N(0, I) features. ``ar90``/``ar99``: AR(0.9)/AR(0.99) correlated columns.
    ``lowrank5``/``lowrank20``: that many latent factors plus 0.3 noise, labels from the
    factors. ``colscale``: independent columns on scales 10^U(0, 3). ``onehot``: three
    20-level one-hot blocks (an exact dependency beside an intercept) plus AR(0.9)
    numerics, ``d ≥ 62``. ``random``: AR(0.9) features, labels independent of them.
    ``blobs``: separable Gaussian blobs, one per class; ``blobsar`` nearly separable
    ones whose within-class noise is AR(0.9), so that the gate sees correlation and
    the separation limit is what turns the fit away. ``digits`` and
    ``breast_cancer``: scikit-learn's, in raw units, upsampled with 5% jitter when
    ``n`` exceeds their size; ``digits`` relabels ``digit % K`` (``digit >= 5`` for
    two classes) and has ``d = 64``, ``breast_cancer`` ``d = 30``.

    Laws a sample of the rows reads wrong: ``rare``, 16 AR(0.9) numerics beside one
    categorical of ``d − 16`` Zipf(1.5)-distributed levels, one-hot, each level shifting
    the logits — at 60 000 rows the rarest levels have a handful of rows, or none;
    ``twosource``, AR(0.99) rows followed by as many N(0, I) rows, in that order; and
    ``dominant``, AR(0.9) features with the first in units of 1e8, a timestamp's or a
    currency's scale beside unit columns.
    """
    rng = np.random.default_rng(seed)
    if law == "iso":
        X = rng.normal(size=(n, d))
        return X, _linear_labels(X, n_classes, rng)
    if law in ("ar90", "ar99"):
        X = _ar_features(n, d, 0.9 if law == "ar90" else 0.99, rng)
        return X, _linear_labels(X, n_classes, rng)
    if law in ("lowrank5", "lowrank20"):
        rank = int(law.removeprefix("lowrank"))
        Z = rng.normal(size=(n, rank))
        X = Z @ rng.normal(size=(rank, d)) + 0.3 * rng.normal(size=(n, d))
        if n_classes == 2:
            return X, (Z[:, 0] + rng.normal(size=n) > 0).astype(int)
        return X, (Z @ rng.normal(size=(rank, n_classes)) + rng.gumbel(size=(n, n_classes))).argmax(1)
    if law == "colscale":
        X = rng.normal(size=(n, d)) * 10.0 ** rng.uniform(0.0, 3.0, size=d)
        return X, _linear_labels(X / X.std(axis=0), n_classes, rng)
    if law == "onehot":
        latent = rng.normal(size=n)
        blocks = []
        for _ in range(3):
            level = np.clip(((latent + rng.normal(size=n)) * 4 + 10).astype(int), 0, 19)
            blocks.append(np.eye(20)[level])
        numeric = _ar_features(n, max(d - 60, 2), 0.9, rng) + latent[:, None]
        X = np.hstack(blocks + [numeric])
        return X, _linear_labels((X - X.mean(0)) / (X.std(0) + 1e-12), n_classes, rng)
    if law == "random":
        return _ar_features(n, d, 0.9, rng), rng.integers(0, n_classes, size=n)
    if law == "rare":
        levels = d - 16
        numeric = _ar_features(n, 16, 0.9, rng)
        share = 1.0 / np.arange(1, levels + 1) ** 1.5
        level = rng.choice(levels, size=n, p=share / share.sum())
        T = 1 if n_classes == 2 else n_classes
        raw = numeric @ rng.normal(size=(16, T)) / 4.0 + rng.normal(size=(levels, T))[level]
        X = np.hstack([numeric, np.eye(levels)[level]])
        if n_classes == 2:
            return X, (raw[:, 0] + rng.logistic(size=n) > 0).astype(int)
        return X, (raw + rng.gumbel(size=(n, T))).argmax(axis=1)
    if law == "twosource":
        head = n // 2
        X = np.vstack([_ar_features(head, d, 0.99, rng), rng.normal(size=(n - head, d))])
        return X, _linear_labels(X, n_classes, rng)
    if law == "dominant":
        X = _ar_features(n, d, 0.9, rng)
        y = _linear_labels(X, n_classes, rng)
        X[:, 0] *= 1e8
        return X, y
    if law == "blobs":
        y = rng.integers(0, n_classes, size=n)
        return rng.normal(scale=2.0, size=(n_classes, d))[y] + rng.normal(size=(n, d)), y
    if law == "blobsar":
        y = rng.integers(0, n_classes, size=n)
        centers = rng.normal(scale=3.0 / np.sqrt(d), size=(n_classes, d))
        return centers[y] + _ar_features(n, d, 0.9, rng), y
    if law in ("digits", "breast_cancer"):
        from sklearn.datasets import load_breast_cancer, load_digits

        data = load_digits() if law == "digits" else load_breast_cancer()
        X0, y0 = data.data.astype(np.float64), data.target
        if law == "digits" and n_classes != 10:
            y0 = (y0 >= 5).astype(int) if n_classes == 2 else y0 % n_classes
        if n == len(X0):
            return X0, y0
        index = rng.integers(0, len(X0), size=n)
        jitter = rng.normal(size=(n, X0.shape[1])) * 0.05 * (X0.std(axis=0) + 1e-3)
        return X0[index] + jitter, y0[index]
    raise ValueError(f"unknown data law {law!r}")


def logistic_newton_oracle(X, y, n_classes, *, C=1.0, sample_weight=None, fit_intercept=True,
                           tol=1e-10, max_iter=100):
    """The penalized logistic optimum in float64 by damped Newton: ``(coef, intercept)``.

    scikit-learn's objective (the binary one for two classes, the unconstrained
    multinomial otherwise), with minimum-norm steps, in the metric of the Hessian's
    diagonal, where the Hessian is singular — the multinomial's sum over classes, an
    unpenalized intercept. The independent reference the preconditioner's parity
    tests measure "closer to the optimum" against: written from the definitions, not
    from either solver.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    Xa = np.hstack([X, np.ones((n, 1))]) if fit_intercept else X
    p = Xa.shape[1]
    T = 1 if n_classes == 2 else n_classes
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    total = float(w.sum())
    l2 = 0.0 if np.isinf(C) else 1.0 / (C * total)
    mask = np.ones(p)
    if fit_intercept:
        mask[-1] = 0.0
    y = np.asarray(y).astype(np.int64)
    onehot = np.eye(n_classes)[y]

    def value_grad(theta):
        Th = theta.reshape(T, p)
        raw = Xa @ Th.T
        if n_classes == 2:
            r = raw[:, 0]
            f = np.sum(w * (np.logaddexp(0.0, r) - y * r)) / total
            prob = 0.5 * (1.0 + np.tanh(0.5 * r))
            g = ((w * (prob - y)) @ Xa / total)[None, :]
        else:
            top = raw.max(axis=1, keepdims=True)
            lse = top[:, 0] + np.log(np.exp(raw - top).sum(axis=1))
            f = np.sum(w * (lse - raw[np.arange(n), y])) / total
            prob = np.exp(raw - lse[:, None])
            g = ((prob - onehot) * w[:, None]).T @ Xa / total
        f += 0.5 * l2 * np.sum((Th * mask) ** 2)
        return f, (g + l2 * Th * mask).reshape(-1), prob

    def hessian(prob):
        if n_classes == 2:
            H = (Xa * (w * prob * (1.0 - prob) / total)[:, None]).T @ Xa
        else:
            H = np.zeros((T * p, T * p))
            for a in range(T):
                for b in range(a, T):
                    c = prob[:, a] * ((a == b) - prob[:, b]) * w / total
                    block = (Xa * c[:, None]).T @ Xa
                    H[a * p:(a + 1) * p, b * p:(b + 1) * p] = block
                    H[b * p:(b + 1) * p, a * p:(a + 1) * p] = block.T
        return H + l2 * np.diag(np.tile(mask, T))

    theta = np.zeros(T * p)
    f, g, prob = value_grad(theta)
    for _ in range(max_iter):
        if np.max(np.abs(g)) < tol:
            break
        H = hessian(prob)
        # Each step is solved in the metric of the Hessian's diagonal, one scale per
        # feature shared by the classes. Unscaled, a column in units of 1e8 put every
        # other direction below the solve's cutoff and the oracle never moved them
        # (J 1.017 where both mlxlearn paths reached 0.660); a scale per class would
        # let the steps drift along the sum over the classes, which they never enter.
        scale = np.tile(np.sqrt(np.maximum(np.diag(H).reshape(T, p).mean(axis=0), 1e-300)), T)
        step = np.linalg.lstsq(H / np.outer(scale, scale), -g / scale, rcond=1e-13)[0] / scale
        t, slope = 1.0, float(g @ step)
        while True:
            f_new, g_new, prob_new = value_grad(theta + t * step)
            if f_new <= f + 1e-4 * t * slope or t < 1e-10:
                break
            t *= 0.5
        if not f_new <= f:
            break
        theta, f, g, prob = theta + t * step, f_new, g_new, prob_new
    Th = theta.reshape(T, p)
    intercept = Th[:, -1].copy() if fit_intercept else np.zeros(T)
    return Th[:, : X.shape[1]].copy(), intercept
