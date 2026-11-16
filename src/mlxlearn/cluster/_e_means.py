"""Apple Silicon implementation of e-means (arXiv:2611.09501v1).

The numerical work runs on the GPU: tiled nearest-centroid search, grouped
statistics, EMA updates and donor splitting. NumPy generates reproducible
permutations and random donor keys; no centroid statistics are copied to the
CPU during optimization. This implementation follows the published equations
and exposes the current Faiss recipe separately, because its constants differ.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral, Real

import mlx.core as mx
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from mlxlearn_device._binary32 import matmul

from .._kernels.e_means import assign, batch_stats, ema_update, gather_bf16
from ._recipe import resolve_recipe, scheduled_alpha


@dataclass(frozen=True)
class PreparedEMeansFit:
    """Validated resident inputs; preparation is separate from optimization."""
    data: mx.array
    initial: mx.array
    offset: mx.array
    recipe: dict
    seed: int
    signature: tuple


@dataclass(frozen=True)
class DeviceEMeansFit:
    """Device outputs; copying these arrays to NumPy is an explicit operation."""
    centers: mx.array
    ema_centers: mx.array
    last_epoch_centers: mx.array
    slow_ema_centers: mx.array | None
    assignments: mx.array | None
    history: tuple
    split_count: mx.array
    steps: int
    epochs: int
    seconds: float


def _normalize(x):
    """Normalize safely without overflowing the norm; preserve zero rows."""
    maximum = mx.max(mx.abs(x), axis=1, keepdims=True)
    scaled = x / mx.where(maximum > 0, maximum, 1)
    norm = mx.sqrt(mx.sum(scaled * scaled, axis=1, keepdims=True))
    return scaled / mx.where(norm > 0, norm, 1)


def revive(centers, counts, uniform, threshold, split_epsilon, companion=None, *, precision="float32"):
    """Split underused clusters with weighted sampling without replacement.

    ``uniform`` contains independent U(0,1) samples, one per centroid. Gumbel
    top-k implements the reference's mass-proportional donor sampling. The
    whole selection stays on the GPU, including the dynamic number of dead
    clusters. ``companion`` optionally supplies slow-EMA (centers, counts),
    which undergo the same donor/child mapping using their own statistics.
    BF16 repairs a collapsed split by moving the child's first coordinate to
    the next BF16 value toward positive infinity, or inward at the finite
    maximum. Other precisions retain the original multiplicative convention.
    Returns ``centers, counts, number_of_splits, companion``.
    """
    k, d = centers.shape
    ids = mx.arange(k, dtype=mx.int32)
    dead = counts < (threshold * mx.sum(counts) / k)
    available = (~dead) & (counts > 0)
    number = mx.minimum(mx.sum(dead), mx.sum(available)).astype(mx.int32)
    uniform = mx.clip(uniform, 1e-30, 1 - np.finfo(np.float32).eps)
    keys = mx.where(available, mx.log(mx.maximum(counts, 1e-30)) - mx.log(-mx.log(uniform)), -mx.inf)
    donor_order = mx.argsort(-keys)
    target_order = mx.argsort(mx.where(dead, ids, ids + k))
    donor_rank = mx.argsort(donor_order)
    target_rank = mx.argsort(target_order)
    is_child = target_rank < number
    is_donor = donor_rank < number
    source = donor_order[target_rank]
    offset = mx.where(mx.arange(d) % 2 == 0, split_epsilon, -split_epsilon)

    def split(position, mass):
        child = position[source] * (1 + offset)
        donor = position * (1 - offset)
        if precision == "bfloat16":
            donor_bits = donor[source].astype(mx.bfloat16).view(mx.uint16)
            child_bits = child.astype(mx.bfloat16).view(mx.uint16)
            # Compare represented codes, treating signed zeros as equal. Bit
            # operations also preserve subnormals on GPUs that flush FP32
            # subnormal arithmetic, including the first positive BF16 value.
            both_zero = ((donor_bits & 0x7fff) == 0) & ((child_bits & 0x7fff) == 0)
            collapsed = mx.all((donor_bits == child_bits) | both_zero, axis=1)
            bits = child_bits[:, 0].astype(mx.uint32)
            negative, zero = (bits & 0x8000) != 0, (bits & 0x7fff) == 0
            upward = mx.where(zero, 1, mx.where(negative, bits - 1, bits + 1))
            inward = mx.where(zero, 0x8001, mx.where(negative, bits + 1, bits - 1))
            neighbor_bits = mx.where((upward & 0x7fff) < 0x7f80, upward, inward)
            neighbor = (neighbor_bits.astype(mx.uint32) << 16).view(mx.float32)
            child = mx.concatenate((mx.where(collapsed, neighbor, child[:, 0])[:, None], child[:, 1:]), axis=1)
        new_position = mx.where(is_child[:, None], child, mx.where(is_donor[:, None], donor, position))
        new_mass = mx.where(is_child, mass[source] * 0.5, mx.where(is_donor, mass * 0.5, mass))
        return new_position, new_mass

    centers, counts = split(centers, counts)
    if companion is not None:
        companion = split(*companion)
    return centers, counts, number, companion


class EMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """Mini-batch clustering with scheduled EMA updates on the Metal GPU.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of centroids.
    max_iter : float, default=3
        Epoch budget, including fractional epochs. For budgets >= 1 a full
        final pass is reserved for last-epoch means, matching Faiss.
    batch_size : int or None
        Statistical mini-batch size. None uses the fitted batch-size law.
    alpha0 : float or None
        Initial EMA step in (0,1]. None uses the fitted step-size law.
    recipe : {'paper', 'faiss'}, default='paper'
        Published Table 2 recipe or the current reference-code recipe. The
        resolved coefficients and conventions are recorded in ``recipe_``.
    batch_law : {'auto', 'throughput', 'quality'}, default='auto'
        Quality divides the unrounded throughput batch size by eight. Paper
        auto selects quality above 2M rows; Faiss auto selects it at d <= 16.
    finalization : {'last_epoch', 'ema', 'slow_ema'}, default='last_epoch'
        Last-pass assigned means, the primary EMA, or a second, slower EMA.
    slow_ema_half_life : float or None
        Half-life in logical batches for the optional second EMA. Otherwise
        its constant step is ``slow_ema_multiplier * 0.1 * alpha0``.
    slow_ema_multiplier : float, default=1
        Slow-EMA step relative to the final cosine floor (Appendix I).
    random_state : int, RandomState or None, default=1234
        Initialization, epoch shuffling and donor-sampling random seed.
    init : {'random'} or array of shape (n_clusters, n_features)
        Initial centroids. Random initialization samples without replacement.
    spherical : bool, default=False
        Normalize input rows and assignment centroid directions. EMA state
        retains unprojected centers. Zero input rows remain zero.
    unused_threshold : float or None
        Revival threshold as a fraction of mean mass; 0 disables revival.
    assignment : {'auto', 'gemm', 'fused'}, default='auto'
        Metal assignment implementation; useful for performance comparisons.
    precision : {'float32', 'float16', 'bfloat16'}, default='float32'
        Assignment arithmetic. EMA state and accumulated sums remain float32.
        Reduced precision is explicitly approximate. bfloat16 also rounds rows
        before statistics, matching the Faiss H100 cast route; it uses raw
        coordinates, represented-centroid norms, and float32 state.
    assignment_batch_size : int or None
        Maximum rows in an assignment tile, independent of statistical B.
    compute_labels : bool, default=True
        Reassign all input rows after training for sklearn-style ``labels_``
        and ``inertia_``. False omits this extra pass and sets both to None.
    verbose : bool, default=False
        Print one progress line per pass.

    Notes
    -----
    ``last_epoch_centers_`` and ``ema_centers_`` are always available.
    ``slow_ema_centers_`` is available when slow EMA was requested. These
    finalizers use moving assignments, not a fresh Lloyd pass. Final labels
    (when enabled) are reassigned against the chosen final centroids.

    Remaining batches are kept, unlike Algorithm 1's floor notation. GPU
    float32 reductions need not be bitwise identical to the CPU reference.
    Faiss recipe mode matches its statistical recipe, not bitwise execution:
    donor keys use float32. Float32/float16 multiplicative splits leave zero
    donors zero; bfloat16 repairs collapsed represented splits by moving one
    child coordinate to the adjacent representable value, as in Faiss.
    This estimator requires MLX's Metal GPU; it never silently falls back.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        max_iter=3,
        batch_size=None,
        alpha0=None,
        recipe="paper",
        batch_law="auto",
        finalization="last_epoch",
        slow_ema_half_life=None,
        slow_ema_multiplier=1.0,
        random_state=1234,
        init="random",
        spherical=False,
        unused_threshold=None,
        assignment="auto",
        precision="float32",
        assignment_batch_size=None,
        compute_labels=True,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.recipe = recipe
        self.batch_law = batch_law
        self.finalization = finalization
        self.slow_ema_half_life = slow_ema_half_life
        self.slow_ema_multiplier = slow_ema_multiplier
        self.random_state = random_state
        self.init = init
        self.spherical = spherical
        self.unused_threshold = unused_threshold
        self.assignment = assignment
        self.precision = precision
        self.assignment_batch_size = assignment_batch_size
        self.compute_labels = compute_labels
        self.verbose = verbose

    def _validate_parameters(self):
        if self.finalization not in {"last_epoch", "ema", "slow_ema"}:
            raise ValueError("finalization must be 'last_epoch', 'ema', or 'slow_ema'")
        if self.assignment not in {"auto", "gemm", "fused"}:
            raise ValueError("assignment must be 'auto', 'gemm', or 'fused'")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be 'float32', 'float16', or 'bfloat16'")
        if self.assignment_batch_size is not None and (
            isinstance(self.assignment_batch_size, bool)
            or not isinstance(self.assignment_batch_size, Integral)
            or self.assignment_batch_size < 1
        ):
            raise ValueError("assignment_batch_size must be a positive integer or None")
        for name in ("slow_ema_half_life", "slow_ema_multiplier"):
            value = getattr(self, name)
            if value is None and name == "slow_ema_multiplier":
                raise ValueError("slow_ema_multiplier must be finite and positive")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    def _input(self, X, *, fit=False, return_bound=False):
        if isinstance(X, mx.array):
            if X.ndim != 2 or min(X.shape) < 1:
                raise ValueError("X must be a nonempty two-dimensional matrix")
            if X.shape[0] > np.iinfo(np.int32).max:
                raise ValueError("X has too many rows for 32-bit GPU permutation indices")
            if mx.issubdtype(X.dtype, mx.complexfloating):
                raise ValueError("X must contain real values")
            data = X.astype(mx.float32)
            finite, magnitude = mx.all(mx.isfinite(data)), mx.max(mx.abs(data))
            mx.eval(finite, magnitude)
            maximum = float(magnitude.item())
            if not bool(finite.item()) or not math.isfinite(maximum):
                raise ValueError("X must contain finite values representable in float32")
        else:
            array = np.asarray(X)
            if array.ndim != 2 or min(array.shape) < 1:
                raise ValueError("X must be a nonempty two-dimensional matrix")
            if array.shape[0] > np.iinfo(np.int32).max:
                raise ValueError("X has too many rows for 32-bit GPU permutation indices")
            if np.iscomplexobj(array):
                raise ValueError("X must contain real values")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(array, dtype=np.float32, order="C")
            # Reuse validation to obtain a magnitude bound without allocating
            # an array-sized abs/isfinite temporary on large embeddings.
            lower, upper = float(array.min()), float(array.max())
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("X must contain finite values representable in float32")
            maximum = max(abs(lower), abs(upper))
            data = mx.array(array)
        if not fit and data.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {data.shape[1]} features, expected {self.n_features_in_}")
        if self.spherical:
            data, maximum = _normalize(data), 0.0 if maximum == 0 else 1.0
        return (data, maximum) if return_bound else data

    def _assign(self, data, centers):
        if not self.spherical and self.precision != "bfloat16":
            data, centers = data - self._offset_mx, centers - self._offset_mx
        return assign(
            data,
            centers,
            batch_size=self.assignment_batch_size,
            spherical=self.spherical,
            method=self.assignment,
            precision=self.precision,
        )

    def _configuration_signature(self):
        primitive = (str, int, float, bool, type(None))
        return tuple((key, value if isinstance(value, primitive) else id(value))
                     for key, value in sorted(self.get_params(deep=False).items()))

    def prepare_device(self, X):
        """Validate/upload data and explicit initialization before a timed fit.

        Returns reusable resident inputs for :meth:`train_device`. Input scans,
        upload, random initial-center selection and the FP32 assignment origin
        are preparation. Epoch permutations, state initialization, BF16 casts,
        training, online labels and finalization remain in ``train_device``.
        """
        self._validate_parameters()
        if not mx.metal.is_available():
            raise RuntimeError("EMeans requires an available Apple Silicon Metal GPU")
        with mx.stream(mx.gpu):
            data, data_maximum = self._input(X, fit=True, return_bound=True)
            n, d = data.shape
            self.recipe_ = resolve_recipe(
                n, self.n_clusters, d, self.max_iter, self.batch_size, self.alpha0,
                self.batch_law, self.recipe, self.unused_threshold,
            )
            k = int(self.n_clusters)
            if k * d > np.iinfo(np.uint32).max or self.recipe_["batch_size"] * d > np.iinfo(np.uint32).max:
                raise ValueError("centroid or statistical batch size exceeds 32-bit Metal indexing; reduce n_clusters or batch_size")
            rng = check_random_state(self.random_state)
            seed = int(self.random_state) if isinstance(self.random_state, Integral) else int(rng.randint(2**32))
            if not 0 <= seed < 2**32:
                raise ValueError("random_state integer must be in [0, 2**32)")
            if isinstance(self.init, str):
                if self.init != "random":
                    raise ValueError("init must be 'random' or an initial centroid matrix")
                indices = np.random.RandomState(seed).permutation(n)[:k]
                centers = data[mx.array(indices, dtype=mx.int32)]
                initial_maximum = data_maximum
            else:
                centers, initial_maximum = self._input(self.init, fit=True, return_bound=True)
                if centers.shape != (k, d):
                    raise ValueError(f"init must have shape {(k, d)}")
            self._check_float32_range(data_maximum, initial_maximum, n=n, d=d, training=True)
            # BF16 must round original coordinates: recentering before rounding
            # would change the represented values used by the H100 algorithm.
            centered = not self.spherical and self.precision != "bfloat16"
            self._offset_mx = mx.mean(data, axis=0) if centered else mx.zeros((d,), mx.float32)
            use_slow = self.finalization == "slow_ema" or self.slow_ema_half_life is not None
            slow_alpha = (-math.expm1(-math.log(2) / self.slow_ema_half_life)
                          if self.slow_ema_half_life is not None else
                          min(1.0, self.slow_ema_multiplier * 0.1 * self.recipe_["alpha0"]))
            smallest_step = min(self.recipe_["alpha0"] * 0.1, slow_alpha) if use_slow else self.recipe_["alpha0"] * 0.1
            if smallest_step < float(np.finfo(np.float32).tiny):
                raise ValueError("EMA step is too small for float32; increase alpha0 or reduce the slow EMA half-life")
            self.recipe_.update(
                seed=seed, backend="metal", precision=self.precision,
                assignment=self.assignment, finalization=self.finalization,
                spherical=bool(self.spherical), slow_alpha=slow_alpha if use_slow else None,
                assignment_centering="global_mean" if centered else "none",
                statistics_representation="bfloat16" if self.precision == "bfloat16" else "float32",
            )
            if self.precision == "float16":
                self._check_half_range(data, centers, training=True)
            mx.eval(data, centers, self._offset_mx)
            return PreparedEMeansFit(data, centers, self._offset_mx, dict(self.recipe_), seed,
                                     self._configuration_signature())

    def train_device(self, prepared, *, capture_assignments=False):
        """Run a complete fit from prepared inputs, returning resident outputs.

        Reusing ``prepared`` restarts from the same initial centers and seed.
        There is no upload, finite-value scan, or NumPy centroid export here.
        All epoch permutations are generated inside this call. Optional labels
        are captured during the final pass, not reassigned after training.
        This low-level method does not populate sklearn's host fitted attributes.
        """
        if not isinstance(prepared, PreparedEMeansFit) or prepared.signature != self._configuration_signature():
            raise ValueError("prepared inputs do not match this estimator configuration")
        started = time.perf_counter()
        with mx.stream(mx.gpu):
            data, centers = prepared.data, prepared.initial
            n, d = data.shape
            k = int(self.n_clusters)
            self.recipe_, self._offset_mx = dict(prepared.recipe), prepared.offset
            mass = mx.full((k,), self.recipe_["init_usage"], dtype=mx.float32)
            slow_alpha = self.recipe_["slow_alpha"]
            slow = (centers, mass) if slow_alpha is not None else None
            split_rng = np.random.RandomState(prepared.seed)
            total_splits = mx.array(0, dtype=mx.int32)
            history = []
            step = 0
            pass_sums, pass_counts = mx.zeros((k, d), mx.float32), mx.zeros((k,), mx.float32)
            batch_size = self.recipe_["batch_size"]
            final_labels = []
            for epoch, rows in enumerate(self.recipe_["epoch_rows"]):
                epoch_start = time.perf_counter()
                order_seed = (prepared.seed * 1000003 + epoch) % 2**32
                order = mx.array(np.random.RandomState(order_seed).permutation(n), dtype=mx.int32)
                final_epoch = epoch == len(self.recipe_["epoch_rows"]) - 1
                objectives = []
                report_counts = mx.zeros((k,), mx.float32)
                for start in range(0, rows, batch_size):
                    batch_order = order[start:min(start + batch_size, rows)]
                    batch = gather_bf16(data, batch_order) if self.precision == "bfloat16" else data[batch_order]
                    readout = _normalize(centers) if self.spherical else centers
                    labels, distances = self._assign(batch, readout)
                    sums, counts = batch_stats(batch, labels, k)
                    objectives.append(mx.sum(distances))
                    if self.precision == "bfloat16":
                        report_counts = report_counts + counts
                    if final_epoch:
                        pass_sums, pass_counts = pass_sums + sums, pass_counts + counts
                        if capture_assignments:
                            final_labels.append(labels)
                    alpha = scheduled_alpha(self.recipe_["alpha0"], step, self.recipe_["total_steps"])
                    previous_mass = mass
                    if self.precision == "bfloat16":
                        centers, mass = ema_update(centers, mass, sums, counts, alpha, decay=1.0 - alpha)
                    else:
                        centers, mass = ema_update(centers, mass, sums, counts, alpha)
                    if self.recipe == "faiss" and alpha == 1.0:
                        mass = mx.where(counts > 0, mass, previous_mass)
                    if slow is not None:
                        slow = ema_update(*slow, sums, counts, slow_alpha,
                                          **({"decay": 1.0 - slow_alpha} if self.precision == "bfloat16" else {}))
                    if (step + 1) % 5 == 0 and self.recipe_["unused_threshold"] > 0:
                        uniform = mx.array(split_rng.random_sample(k), dtype=mx.float32)
                        centers, mass, nsplit, slow = revive(
                            centers, mass, uniform, self.recipe_["unused_threshold"],
                            self.recipe_["split_epsilon"], companion=slow, precision=self.precision)
                        total_splits = total_splits + nsplit
                    pending = [centers, mass, objectives[-1], total_splits]
                    if self.precision == "bfloat16":
                        pending.append(report_counts)
                    if final_epoch:
                        pending.extend((pass_sums, pass_counts))
                        if capture_assignments:
                            pending.append(labels)
                    if slow is not None:
                        pending.extend(slow)
                    mx.async_eval(*pending)
                    step += 1
                # Match the reference's per-epoch reporting boundary. This
                # diagnostic reduction is FP32: Metal has no FP64 arithmetic.
                objective = float(mx.sum(mx.stack(objectives)).item()) if objectives else 0.0
                mx.eval(centers, mass, total_splits)
                stat = {"epoch": epoch + 1, "rows": rows, "steps": step,
                        "moving_objective": objective, "moving_mse": objective / rows if rows else 0.,
                        "seconds": time.perf_counter() - epoch_start,
                        "cumulative_seconds": time.perf_counter() - started,
                        "splits": int(total_splits.item())}
                history.append(stat)
                if self.precision == "bfloat16":
                    stat["pass_count_squares"] = float(mx.sum(report_counts * report_counts).item())
                if self.verbose:
                    print(f"e-means pass {epoch + 1}: {rows} rows, moving MSE={stat['moving_mse']:.7g}, {stat['seconds']:.3f}s", flush=True)
            closing = mx.where((pass_counts > 0)[:, None], pass_sums / mx.maximum(pass_counts[:, None], 1), centers)
            if self.spherical:
                closing, centers = _normalize(closing), _normalize(centers)
            slow_centers = (_normalize(slow[0]) if self.spherical else slow[0]) if slow is not None else None
            chosen = {"ema": centers, "last_epoch": closing, "slow_ema": slow_centers}[self.finalization]
            captured = None
            if capture_assignments:
                # Metal scatter supports 32-bit integers; widen the completed
                # captured vector to the reference API's int64 representation.
                moving_labels = mx.concatenate(final_labels).astype(mx.int32)
                captured = mx.full((n,), -1, dtype=mx.int32)
                captured[order[:len(moving_labels)]] = moving_labels
                captured = captured.astype(mx.int64)
            pending = [chosen, centers, closing, total_splits]
            if slow_centers is not None:
                pending.append(slow_centers)
            if captured is not None:
                pending.append(captured)
            mx.eval(*pending)
            return DeviceEMeansFit(chosen, centers, closing, slow_centers, captured,
                                   tuple(history), total_splits, step, len(history), time.perf_counter() - started)

    def fit(self, X, y=None):
        """Fit centroids and populate sklearn-style NumPy fitted attributes."""
        started = time.perf_counter()
        prepared = self.prepare_device(X)
        result = self.train_device(prepared)
        with mx.stream(mx.gpu):
            self.offset_ = np.array(prepared.offset)
            self.ema_centers_ = np.array(result.ema_centers)
            self.last_epoch_centers_ = np.array(result.last_epoch_centers)
            self.slow_ema_centers_ = None if result.slow_ema_centers is None else np.array(result.slow_ema_centers)
            self._centers_mx = result.centers
            self.cluster_centers_ = np.array(result.centers)
            self._centers_magnitude_ = max(abs(float(self.cluster_centers_.min())), abs(float(self.cluster_centers_.max())))
            self.n_features_in_ = prepared.data.shape[1]
            self.n_steps_, self.n_iter_ = result.steps, result.epochs
            self.n_splits_ = int(result.split_count.item())
            self.iteration_stats_ = list(result.history)
            self.execution_backend_ = "metal"
            self.optimization_time_ = time.perf_counter() - started
            self.labels_, self.inertia_ = None, None
            if self.compute_labels:
                labels, distances = self._assign(prepared.data, result.centers)
                mx.eval(labels, distances)
                self.labels_ = np.array(labels, dtype=np.int64)
                self.inertia_ = float(np.asarray(distances).sum(dtype=np.float64))
            self.fit_time_ = time.perf_counter() - started
        return self

    def predict(self, X):
        """Return nearest final-centroid indices."""
        check_is_fitted(self, "cluster_centers_")
        with mx.stream(mx.gpu):
            data = self._inference_input(X)
            if self.precision == "float16":
                self._check_half_range(data, self._centers_mx)
            labels, _ = self._assign(data, self._centers_mx)
            return np.array(labels, dtype=np.int64)

    def transform(self, X):
        """Return Euclidean (spherical: chord) distances to all centers."""
        check_is_fitted(self, "cluster_centers_")
        with mx.stream(mx.gpu):
            data = self._inference_input(X)
            centers = self._centers_mx
            if not self.spherical:
                data, centers = data - self._offset_mx, centers - self._offset_mx
            chunks = []
            size = self.assignment_batch_size or max(1, (16 << 20) // (4 * self.n_clusters))
            for start in range(0, len(data), size):
                batch = data[start:start + size]
                if self.spherical:
                    distances = 2 - 2 * matmul(batch, centers, transpose_b=True)
                else:
                    distances = mx.sum(batch * batch, axis=1)[:, None] + mx.sum(centers * centers, axis=1)[None, :] - 2 * matmul(batch, centers, transpose_b=True)
                chunks.append(np.array(mx.sqrt(mx.maximum(distances, 0))))
            return np.concatenate(chunks)

    def fit_predict(self, X, y=None):
        """Fit and return final labels, computing them even when disabled."""
        self.fit(X, y)
        return self.labels_ if self.labels_ is not None else self.predict(X)

    def score(self, X, y=None):
        """Negative sum of squared distances (larger is better)."""
        check_is_fitted(self, "cluster_centers_")
        with mx.stream(mx.gpu):
            data = self._inference_input(X)
            if self.precision == "float16":
                self._check_half_range(data, self._centers_mx)
            _, distances = self._assign(data, self._centers_mx)
            return -float(np.array(distances).sum(dtype=np.float64))

    def _inference_input(self, X):
        data, maximum = self._input(X, return_bound=True)
        # The fixed assignment origin can be larger than the final centers.
        origin = max(abs(float(self.offset_.min())), abs(float(self.offset_.max())))
        self._check_float32_range(max(maximum, origin), self._centers_magnitude_, n=1, d=self.n_features_in_)
        return data

    def _check_float32_range(self, data_maximum, initial_maximum, *, n, d, training=False):
        """Bound float32 distances, sums and objectives before GPU execution.

        A coordinate hull bounds every EMA mean. Every possible revival can
        enlarge that hull by at most (1 + epsilon). Centering can double the
        coordinate magnitude, and subtraction can double it again. The margin
        below bounds squared residuals and their float32 per-pass reduction,
        as well as original-coordinate batch/closing sums. This conservative
        check requires no per-batch synchronization.
        """
        maximum = max(data_maximum, initial_maximum)
        if self.precision == "bfloat16":
            # Cover BF16 rounding and the smallest nonzero repaired donor.
            maximum = max(maximum, 2.0**-133) * (1 + 2.0**-7)
        if training and maximum > 0 and self.recipe_["unused_threshold"] > 0:
            checks = self.recipe_["total_steps"] // 5
            per_check = math.log1p(self.recipe_["split_epsilon"])
            if self.precision == "bfloat16":
                per_check += math.log1p(2.0**-7)
            exponent = checks * per_check
            maximum = math.inf if exponent > 700 else maximum * math.exp(exponent)
        largest = float(np.finfo(np.float32).max)
        safe_maximum = min(largest / (8 * n), math.sqrt(largest / (64 * n * d)))
        if not math.isfinite(maximum) or maximum > safe_maximum:
            raise ValueError("input or initial centroid magnitude is too large for safe float32 distances and accumulation; rescale the data")

    def _check_half_range(self, data, centers, training=False):
        """Reject unsafe half casts without synchronizing every GPU update.

        EMA means remain in the coordinate hull of their operands. Multiplicative
        splits can extend it: conservatively bound that extension over every
        possible revival. This can reject borderline data whose actual run would
        stay in range; float32 remains available for those inputs.
        """
        shift = self._offset_mx
        maximum = mx.maximum(mx.max(mx.abs(data - shift)), mx.max(mx.abs(centers - shift)))
        limit = float(maximum.item())
        if training and not self.spherical and self.recipe_["unused_threshold"] > 0:
            checks = self.recipe_["total_steps"] // 5
            growth_log = checks * math.log1p(self.recipe_["split_epsilon"])
            if growth_log > 700:
                limit = math.inf
            else:
                raw_max = float(mx.maximum(mx.max(mx.abs(data)), mx.max(mx.abs(centers))).item())
                limit += raw_max * math.expm1(growth_log)
        if not math.isfinite(limit) or limit > float(np.finfo(np.float16).max):
            raise ValueError("centered assignment operands may exceed float16 range; use precision='float32'")
