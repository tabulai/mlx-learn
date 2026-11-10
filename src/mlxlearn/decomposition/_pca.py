"""Principal Component Analysis (PCA), accelerated on Apple Silicon with Metal and MLX."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA as _SkPCA

from .._common.arrays import to_mlx
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.device import use_device
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

__all__ = ["MLXPCA", "PCA"]


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

    def _check_pca_capability(self, X: Any, operation: str = "fit") -> None:
        name = type(self).__name__
        require_dense(X, estimator=name, operation=operation)
        check_float32_range(X, name="X")

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < tuning.pca_min_samples and work < tuning.pca_min_work:
            return True, (
                f"{n_samples} samples x {n_features} features is below the "
                f"PCA crossover ({tuning.pca_min_samples} samples and "
                f"{tuning.pca_min_work} elements)"
            )
        return False, ""

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
        # while Layer 1 raises SparseInputError (a TypeError subclass).
        def check_capability() -> None:
            self._check_pca_capability(X, operation="fit")

        backend = self._select_backend(
            "fit",
            capability_check=check_capability,
            shape=_safe_shape(X),
        )
        if backend == "sklearn":
            if return_transformed:
                raw_fit_transform = getattr(_SkPCA.fit_transform, "__wrapped__", _SkPCA.fit_transform)
                out = raw_fit_transform(self, X, y)
            else:
                _SkPCA.fit(self, X, y)
                out = None
            self._set_backend("sklearn")
            return out

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
        check_float32_range(X_checked, name="X")

        n_samples, n_features = X_checked.shape
        below, detail = self._below_crossover(n_samples, n_features)
        if below and backend != "cpu":
            backend = self._select_backend(
                "fit",
                below_crossover=True,
                crossover_detail=detail,
                shape=X_checked.shape,
            )
            if backend == "sklearn":
                if return_transformed:
                    raw_fit_transform = getattr(
                        _SkPCA.fit_transform, "__wrapped__", _SkPCA.fit_transform
                    )
                    out = raw_fit_transform(self, X, y)
                else:
                    _SkPCA.fit(self, X, y)
                    out = None
                self._set_backend("sklearn")
                return out

        fit_svd_solver = choose_svd_solver(
            n_samples, n_features, self.n_components, self.svd_solver
        )
        self._fit_svd_solver = fit_svd_solver
        resolved_n_components = _validate_pca_n_components(
            self.n_components, fit_svd_solver, n_samples, n_features
        )

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
            ensure_all_finite=not defer_finite,
            reset=False,
        )
        check_float32_range(X_checked, name="X")
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
        check_float32_range(X_checked, name="X")
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
            ensure_all_finite=not defer_finite,
            reset=False,
        )
        check_float32_range(X_checked, name="X")
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
