"""Principal Component Analysis (PCA), accelerated on Apple Silicon with Metal and MLX."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA as _SkPCA

from .._common.arrays import to_mlx
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.device import use_device
from .._common.preparation import cast_float32_checked
from .._common.rng import resolve_seed
from .._common.validation import (
    check_array,
    check_float32_range,
    check_is_fitted,
    is_sparse,
    require_dense,
    validate_data,
)
from .._solvers.pca import (
    PCAConfig,
    choose_svd_solver,
    pca_inverse_transform,
    pca_score_samples,
    pca_transform,
    solve_pca,
)
from ..exceptions import CapabilityError

__all__ = ["MLXPCA", "PCA"]

_FLOAT32_MAX_SQUARED = float(np.finfo(np.float32).max) ** 2


def _finite_and_in_float32_range(X: np.ndarray) -> bool:
    """Prove with one BLAS pass that ``X`` is finite and fits float32, when possible.

    The sum of squares bounds the largest magnitude from above, so a finite sum
    below ``float32.max ** 2`` settles both questions at once, and NaN or infinity
    poison it. ``False`` means undecided, not invalid: values whose squares
    overflow the dtype land here too, and the exact checks must then decide.
    """
    if not (X.flags.c_contiguous or X.flags.f_contiguous):
        return False
    flat = X.ravel(order="K")
    # NumPy 2.2 with Accelerate raises spurious floating-point flags from dot.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        total = float(np.dot(flat, flat))
    return total < _FLOAT32_MAX_SQUARED


def _safe_shape(X: Any) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        try:
            return int(shape[0]), int(shape[1])
        except (TypeError, ValueError):
            return None
    return None


def _validate_pca_n_components(
    n_components: Any,
    fit_svd_solver: str,
    n_samples: int,
    n_features: int,
) -> Any:
    """Validate and normalize ``n_components`` matching scikit-learn's exact contract."""
    if n_components is None:
        if fit_svd_solver != "arpack":
            resolved = min(n_samples, n_features)
        else:
            resolved = min(n_samples, n_features) - 1
    else:
        resolved = n_components

    if fit_svd_solver in ("full", "covariance_eigh"):
        if resolved == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= resolved <= min(n_samples, n_features):
            raise ValueError(
                f"n_components={resolved} must be between 0 and "
                f"min(n_samples, n_features)={min(n_samples, n_features)} with "
                f"svd_solver={fit_svd_solver!r}"
            )
    else:
        if isinstance(resolved, str):
            raise ValueError(
                f"n_components={resolved!r} cannot be a string with svd_solver='{fit_svd_solver}'"
            )
        if not 1 <= resolved <= min(n_samples, n_features):
            raise ValueError(
                f"n_components={resolved!r} must be between 1 and "
                f"min(n_samples, n_features)={min(n_samples, n_features)!r} with "
                f"svd_solver='{fit_svd_solver}'"
            )
        if isinstance(resolved, (float, np.floating)):
            raise ValueError(
                f"n_components={resolved!r} must be of type int "
                f"when greater than or equal to 1, was of type={type(resolved)!r}"
            )
        if fit_svd_solver == "arpack" and resolved == min(n_samples, n_features):
            raise ValueError(
                f"n_components={resolved!r} must be strictly less than "
                f"min(n_samples, n_features)={min(n_samples, n_features)!r} with "
                f"svd_solver='{fit_svd_solver}'"
            )
    return resolved


class MLXPCA(BackendMixin):
    """Metal GPU and Unified-Memory Accelerate mixin for :class:`sklearn.decomposition.PCA`."""

    def _narrow_input(
        self, X_checked: np.ndarray, *, use_gpu: bool, need_finite: bool
    ) -> np.ndarray | None:
        """Enforce the float32 capability; return the float32 copy the GPU will read.

        Wider input headed for the GPU has to be narrowed anyway, and that cast
        doubles as the range check: it is finite exactly when no value overflowed.
        Scanning the float64 array for its extremes first would read the whole
        input twice more, which measured as over half of a large fit. float32
        input cannot overflow and is borrowed by the solver as it is; the GPU
        validates its finiteness, so there is nothing to do and nothing to return.

        Input that stays on the host yields ``None``. Its range, and its finiteness
        when the caller has no later check for it (``need_finite``), are settled
        by one sum of squares instead of separate passes for the minimum, the
        maximum and the finiteness test.

        In every case the original array is inspected exactly only when the cheap
        test is inconclusive, so a finite float64 that overflows float32 stays a
        capability error and NaN/Inf stay scikit-learn's invalid-input error.
        """
        narrow = X_checked.dtype == np.float32
        if use_gpu:
            if narrow:
                return None
            X_f32, proven = cast_float32_checked(X_checked)
        else:
            X_f32 = None
            proven = (narrow and not need_finite) or _finite_and_in_float32_range(X_checked)
        if not proven:
            check_float32_range(X_checked, name="X")
            check_array(
                X_checked,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                ensure_all_finite=True,
                input_name="X",
                estimator=self,
            )
        return X_f32

    def _fit_sklearn(self, X: Any, y: Any, return_transformed: bool) -> np.ndarray | None:
        """Hand the whole fit to stock scikit-learn and record that it did the work."""
        if return_transformed:
            raw_fit_transform = getattr(_SkPCA.fit_transform, "__wrapped__", _SkPCA.fit_transform)
            out = raw_fit_transform(self, X, y)
        else:
            _SkPCA.fit(self, X, y)
            out = None
        self._set_backend("sklearn")
        return out

    def _below_crossover(
        self, n_samples: int, n_features: int, fit_svd_solver: str
    ) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        The answer depends on the solver, because the solver decides how much
        arithmetic each element of ``X`` buys and how many GPU round trips the
        fit needs. The floors below are where the GPU path beat scikit-learn for
        every dtype, data distribution, ``n_components`` and operation measured;
        see ``docs/benchmarks.md``.

        ``covariance_eigh`` reads ``X`` twice and then works on a ``d x d``
        matrix, so on narrow data the host BLAS is finished before a GPU round
        trip has paid for itself. From 512 features up the float32 eigensolve and
        the GPU Gram matrix win at every row count the solver is chosen for, so a
        problem is below the crossover only if it is small on both counts.

        ``randomized`` makes two GPU round trips per power iteration; their
        latency is a fixed cost of a few milliseconds that only large inputs repay.

        ``full`` and ``arpack`` stay on the host at every size. The GPU ``full``
        path wins by 3-5x on well-conditioned data, but a spectrum spanning more
        than about 1.5 decades, which is most real data, sends its trailing part
        through a host SVD that costs as much as scikit-learn's whole fit: it
        measured 0.54x-0.88x in float32 from 200 x 200 to 1500 x 1500. ARPACK's
        iteration runs in SciPy on the host either way. Both remain reachable with
        ``force_backend="mlx"``.
        """
        tuning = get_tuning()
        work = n_samples * n_features
        size = f"{n_samples} samples x {n_features} features"
        if fit_svd_solver == "covariance_eigh":
            if (
                work < tuning.pca_covariance_min_work
                and n_features < tuning.pca_covariance_min_features
            ):
                return True, (
                    f"{size} is below the PCA covariance_eigh crossover "
                    f"({tuning.pca_covariance_min_work} elements or "
                    f"{tuning.pca_covariance_min_features} features)"
                )
            return False, ""
        if fit_svd_solver == "randomized":
            if work < tuning.pca_randomized_min_work:
                return True, (
                    f"{size} is below the PCA randomized crossover "
                    f"({tuning.pca_randomized_min_work} elements)"
                )
            return False, ""
        return True, (
            f"svd_solver={fit_svd_solver!r} is served by LAPACK/ARPACK on the host "
            "at every size"
        )

    def _get_device_state(self) -> tuple[Any, Any]:
        """Return cached ``(device_mean, device_components)`` on GPU, rebuilding lazily after unpickling."""
        state = getattr(self, "_mlx_state_", None)
        if isinstance(state, dict):
            return state.get("mean"), state.get("components")
        if getattr(self, "_execution_backend_", None) == "mlx" and hasattr(self, "mean_"):
            dev_mean = to_mlx(self.mean_.astype(np.float32, copy=False), device="gpu")
            dev_comp = (
                to_mlx(self.components_.astype(np.float32, copy=False), device="gpu")
                if self.components_.shape[0] > 0
                else None
            )
            self._mlx_state_ = {"mean": dev_mean, "components": dev_comp}
            return dev_mean, dev_comp
        return None, None

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__() if hasattr(super(), "__getstate__") else self.__dict__.copy()
        if isinstance(state, dict):
            state = state.copy()
            state.pop("_mlx_state_", None)
        return state

    def _fit_impl(self, X: Any, y: Any = None, *, return_transformed: bool = False) -> np.ndarray | None:
        self._validate_params()
        self._clear_backend()

        # Check sparse capability before validate_data so Layer 2 can fall back cleanly
        # while Layer 1 raises SparseInputError (a TypeError subclass). The float32
        # range is enforced after validation, by the cast the GPU path needs anyway.
        def check_capability() -> None:
            require_dense(X, estimator=type(self).__name__, operation="fit")

        # The crossover needs only the shape. When the input already has one, decide
        # before validating, so a fit that scikit-learn will serve is handed over
        # untouched.
        shape = _safe_shape(X)
        below, detail = False, ""
        if shape is not None and min(shape) > 0:
            below, detail = self._below_crossover(
                *shape, choose_svd_solver(*shape, self.n_components, self.svd_solver)
            )
        backend = self._select_backend(
            "fit",
            capability_check=check_capability,
            below_crossover=below,
            crossover_detail=detail,
            shape=shape,
        )
        if backend == "sklearn":
            return self._fit_sklearn(X, y, return_transformed)

        # Validate array shape and convert to float64 / float32 without forcing an
        # upfront host finiteness pass when the GPU 1-pass kernel will validate it.
        X_checked = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_2d=True,
            copy=False,
            force_writeable=True,
            ensure_all_finite=False,
        )

        n_samples, n_features = X_checked.shape
        fit_svd_solver = choose_svd_solver(
            n_samples, n_features, self.n_components, self.svd_solver
        )
        if X_checked.shape != shape:
            # Containers without a shape, or one that validation changed.
            below, detail = self._below_crossover(n_samples, n_features, fit_svd_solver)
            if below and backend != "cpu":
                backend = self._select_backend(
                    "fit",
                    below_crossover=True,
                    crossover_detail=detail,
                    shape=X_checked.shape,
                )
                if backend == "sklearn":
                    return self._fit_sklearn(X, y, return_transformed)

        self._fit_svd_solver = fit_svd_solver
        resolved_n_components = _validate_pca_n_components(
            self.n_components, fit_svd_solver, n_samples, n_features
        )

        try:
            # The solver validates finiteness itself, on the GPU or on the host.
            X_f32 = self._narrow_input(X_checked, use_gpu=backend == "mlx", need_finite=False)
        except CapabilityError as exc:
            # Raises for a directly imported estimator and under
            # fallback_policy="raise"; otherwise scikit-learn's float64 path serves it.
            self._handle_capability_error(exc, "fit", shape=X_checked.shape)
            return self._fit_sklearn(X, y, return_transformed)

        seed = resolve_seed(self.random_state)
        config = PCAConfig(
            n_components=resolved_n_components,
            whiten=bool(self.whiten),
            svd_solver=fit_svd_solver,
            tol=float(self.tol),
            iterated_power=self.iterated_power,
            n_oversamples=int(self.n_oversamples),
            power_iteration_normalizer=self.power_iteration_normalizer,
            seed=seed,
            prefer_gpu=(backend == "mlx"),
        )

        device_target = "gpu" if backend == "mlx" else "cpu"
        with mlx_guard(type(self).__name__, "fit"), use_device(device_target):
            try:
                solution = solve_pca(
                    X_checked,
                    config,
                    return_transformed=return_transformed,
                    X_f32=X_f32,
                )
            except ValueError:
                # Re-run standard sklearn validation so NaN vs Inf error messages match
                # scikit-learn's exact check_array contract.
                check_array(
                    X_checked,
                    dtype=[np.float64, np.float32],
                    ensure_2d=True,
                    ensure_all_finite=True,
                    input_name="X",
                    estimator=self,
                )
                raise

        # Honor copy=False in-place centering contract when X was passed as a writeable
        # float64/float32 ndarray and solver is not covariance_eigh.
        if (
            not self.copy
            and fit_svd_solver != "covariance_eigh"
            and isinstance(X, np.ndarray)
            and X_checked is X
        ):
            X -= solution.mean

        self.mean_ = solution.mean
        self.components_ = np.ascontiguousarray(solution.components)
        self.explained_variance_ = np.ascontiguousarray(solution.explained_variance)
        self.explained_variance_ratio_ = np.ascontiguousarray(solution.explained_variance_ratio)
        self.singular_values_ = np.ascontiguousarray(solution.singular_values)
        self.n_samples_ = solution.n_samples
        self.n_components_ = solution.n_components
        self.noise_variance_ = solution.noise_variance

        if backend == "mlx" and solution.device_mean is not None:
            self._mlx_state_ = {
                "mean": solution.device_mean,
                "components": solution.device_components,
            }
        self._set_backend(backend)
        return solution.transformed

    def fit(self, X: Any, y: Any = None) -> MLXPCA:
        """Fit the PCA model with ``X``."""
        self._fit_impl(X, y, return_transformed=False)
        return self

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the model with ``X`` and apply dimensionality reduction on ``X``."""
        out = self._fit_impl(X, y, return_transformed=True)
        return out  # type: ignore[return-value]

    def transform(self, X: Any) -> np.ndarray:
        """Apply dimensionality reduction to ``X``."""
        check_is_fitted(self)
        backend = self._note_inference("transform", shape=_safe_shape(X))
        if backend == "sklearn":
            raw_transform = getattr(_SkPCA.transform, "__wrapped__", _SkPCA.transform)
            return raw_transform(self, X)

        if is_sparse(X):
            require_dense(X, estimator=type(self).__name__, operation="transform")

        defer_finite = backend == "mlx" and self.n_components_ > 0
        X_checked = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
            reset=False,
        )
        X_f32 = self._narrow_input(X_checked, use_gpu=defer_finite, need_finite=True)
        dev_mean, dev_comp = self._get_device_state()
        device_target = "gpu" if backend == "mlx" else "cpu"
        with mlx_guard(type(self).__name__, "transform"), use_device(device_target):
            out = pca_transform(
                X_checked,
                self.mean_,
                self.components_,
                self.explained_variance_,
                whiten=bool(self.whiten),
                prefer_gpu=(backend == "mlx"),
                device_mean=dev_mean,
                device_components=dev_comp,
                X_f32=X_f32,
            )
        if defer_finite and not bool(np.isfinite(out).all()):
            check_array(
                X_checked,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                ensure_all_finite=True,
                input_name="X",
                estimator=self,
            )
        return out

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Transform data back to its original feature space."""
        check_is_fitted(self)
        backend = self._note_inference("inverse_transform", shape=_safe_shape(X))
        if backend == "sklearn":
            return _SkPCA.inverse_transform(self, X)

        if is_sparse(X):
            require_dense(X, estimator=type(self).__name__, operation="inverse_transform")

        X_checked = check_array(
            X,
            input_name="X",
            dtype=[np.float64, np.float32],
            accept_sparse=False,
        )
        # check_array above already established finiteness.
        X_f32 = self._narrow_input(X_checked, use_gpu=backend == "mlx", need_finite=False)
        dev_mean, dev_comp = self._get_device_state()
        device_target = "gpu" if backend == "mlx" else "cpu"
        with mlx_guard(type(self).__name__, "inverse_transform"), use_device(device_target):
            return pca_inverse_transform(
                X_checked,
                self.mean_,
                self.components_,
                self.explained_variance_,
                whiten=bool(self.whiten),
                prefer_gpu=(backend == "mlx"),
                device_mean=dev_mean,
                device_components=dev_comp,
                X_f32=X_f32,
            )

    def score_samples(self, X: Any) -> np.ndarray:
        """Return the log-likelihood of each sample under the probabilistic PCA model."""
        check_is_fitted(self)
        backend = self._note_inference("score_samples", shape=_safe_shape(X))
        if backend == "sklearn":
            return _SkPCA.score_samples(self, X)

        if is_sparse(X):
            require_dense(X, estimator=type(self).__name__, operation="score_samples")

        nv = float(self.noise_variance_)
        use_woodbury = (
            not self.whiten
            and nv > 0.0
            and self.n_components_ > 0
            and bool(np.all(self.explained_variance_ > nv))
        )
        defer_finite = backend == "mlx" and use_woodbury
        X_checked = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            accept_sparse=False,
            ensure_all_finite=False,
            reset=False,
        )
        X_f32 = self._narrow_input(X_checked, use_gpu=defer_finite, need_finite=True)
        if use_woodbury and X_checked.size >= 16_384:
            device_target = "gpu" if backend == "mlx" else "cpu"
            with mlx_guard(type(self).__name__, "score_samples"), use_device(device_target):
                out = pca_score_samples(
                    X_checked,
                    self.mean_,
                    self.components_,
                    self.explained_variance_,
                    self.noise_variance_,
                    prefer_gpu=(backend == "mlx"),
                    X_f32=X_f32,
                )
            if defer_finite and not bool(np.isfinite(out).all()):
                check_array(
                    X_checked,
                    dtype=[np.float64, np.float32],
                    ensure_2d=True,
                    ensure_all_finite=True,
                    input_name="X",
                    estimator=self,
                )
            return out
        if defer_finite:
            check_array(
                X_checked,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                ensure_all_finite=True,
                input_name="X",
                estimator=self,
            )
        return _SkPCA.score_samples(self, X_checked)

    def score(self, X: Any, y: Any = None) -> float:
        """Return the average log-likelihood of all samples."""
        return float(np.mean(self.score_samples(X)))


class PCA(MLXPCA, _SkPCA):
    """Principal Component Analysis (PCA) accelerated on Apple Silicon with MLX.

    A drop-in replacement for :class:`sklearn.decomposition.PCA` that accelerates
    dense data centering, covariance/Gram matrix formation, randomized subspace
    iteration, projection, reconstruction, and sample log-likelihood evaluation on
    the Metal GPU and Apple Accelerate unified memory.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
    copy : bool, default=True
        If False, data passed to fit are overwritten when using ``"full"``,
        ``"arpack"``, or ``"randomized"`` solvers.
    whiten : bool, default=False
        When True, the ``components_`` vectors are multiplied by the square root of
        ``n_samples`` and divided by the singular values to ensure uncorrelated
        outputs with unit component-wise variances.
    svd_solver : {'auto', 'full', 'covariance_eigh', 'arpack', 'randomized'}, default='auto'
        Solver to use.
    tol : float, default=0.0
        Tolerance for singular values computed by ``svd_solver == 'arpack'``.
    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        ``svd_solver == 'randomized'``.
    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of ``X`` for
        ``svd_solver == 'randomized'``.
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iteration normalizer uses QR, LU, or no normalization.
    random_state : int, RandomState instance or None, default=None
        Used when ``svd_solver`` == 'arpack' or 'randomized'.
    """
