"""Reproduce original-coordinate gradient checks on offset logistic data.

The first case replays the RNG used in the native-initialization scratch probe;
the second meets both ordinary preparation and automatic-dispatch thresholds.
The default objective/gradient tolerance remains unchanged and every result is
recorded, including failures of the independent original-coordinate check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn._common.config import config_context, tuning_context
from mlxlearn._solvers import logistic, native_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator

from .logistic_wide import frozen_reference, observed_implementation


def saved_small_fixture():
    """Replay every earlier scratch case to reproduce its offset RNG state."""
    rng = np.random.default_rng(461)
    for kind in (
        "zero_weight", "ill_conditioned", "near_tolerance_below", "near_tolerance_above", "offset"
    ):
        n, d = (516, 519) if kind.startswith("near_tolerance") else (1027, 520)
        X = rng.normal(size=(n, d)).astype(np.float32)
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        if kind == "zero_weight":
            rng.uniform(0.25, 3, n).astype(np.float32)
        if kind == "offset":
            X += np.float32(1000)
            return X, y
    raise AssertionError("offset fixture not generated")


def production_fixture():
    rng = np.random.default_rng(461)
    X = rng.normal(size=(20_000, 1024)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
    X += np.float32(1000)
    return X, y


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def score(model, X, y, tolerance):
    data = X.astype(np.float64)
    weights = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])
    raw = data @ weights + intercept
    residual = expit(raw) - y
    feature_gradient = residual @ data / len(y) + weights / len(y)
    intercept_gradient = float(residual.mean())
    gradient_max = float(max(np.max(np.abs(feature_gradient)), abs(intercept_gradient)))
    mean = data.mean(axis=0)
    centered_gradient_max = float(np.max(np.abs(feature_gradient - mean * intercept_gradient)))
    objective = float(
        np.logaddexp(0, (1 - 2 * y) * raw).mean() + (weights @ weights) / (2 * len(y))
    )
    return {
        "objective_float64": objective,
        "original_coordinate_gradient_max_abs": gradient_max,
        "original_gradient_within_tolerance": gradient_max <= tolerance,
        "intercept_gradient": intercept_gradient,
        "mean_centered_feature_gradient_max_abs": centered_gradient_max,
        "coef": weights.tolist(),
        "intercept": intercept,
        "predict_from_float64_logits": (raw > 0).astype(int),
    }


def run_fit(cls, settings, tuning, observed, X, y, *, force):
    with (
        settings(device="auto"),
        tuning(force_backend=force),
        observed as records,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        mx.synchronize()
        start = time.perf_counter()
        model = cls(C=1.0, tol=1e-4, max_iter=200, fit_intercept=True).fit(X, y)
        mx.synchronize()
        seconds = time.perf_counter() - start
    return model, {
        "fit_seconds": seconds,
        "execution_backend": getattr(model, "_execution_backend_", "sklearn"),
        "n_iter": model.n_iter_.tolist(),
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
        **records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-archive",
        type=Path,
        default=Path("benchmarks/reference/overhead-logistic-2026-10-26.zip"),
    )
    parser.add_argument("--reference-extension", type=Path)
    parser.add_argument("--output", type=Path, default=Path("bench/offset-baseline-probe.json"))
    args = parser.parse_args()
    tolerance = 1e-4
    # Current estimators prepare every GPU input; only frozen references keep a threshold.
    original_threshold = getattr(estimator, "_CENTERED_INPUT_MIN_ELEMENTS", None)
    payload = {
        "parameters": {"C": 1.0, "tol": tolerance, "max_iter": 200, "fit_intercept": True},
        "reference_archive": str(args.reference_archive.resolve()),
        "reference_sha256": hashlib.sha256(args.reference_archive.read_bytes()).hexdigest(),
        "cases": [],
    }
    try:
        with threadpool_limits(1), frozen_reference(
            archive=args.reference_archive, extension=args.reference_extension
        ) as reference:
            reference_threshold = reference.estimator._CENTERED_INPUT_MIN_ELEMENTS
            for name, make_fixture in (("saved_small", saved_small_fixture), ("production", production_fixture)):
                X, y = make_fixture()
                models = {}
                records = {}
                variants = (
                    ("frozen_default", reference_threshold, ""),
                    ("frozen_prepared", 1, "mlx"),
                    ("current_prepared", 1, "mlx"),
                    ("sklearn", None, ""),
                ) if name == "saved_small" else (
                    ("frozen_default", reference_threshold, ""),
                    ("current_default", original_threshold, ""),
                    ("sklearn", None, ""),
                )
                for variant, threshold, force in variants:
                    if variant.startswith("frozen"):
                        reference.estimator._CENTERED_INPUT_MIN_ELEMENTS = threshold
                        cls = reference.estimator.LogisticRegression
                        settings, tuning = reference.config.config_context, reference.config.tuning_context
                        observed = observed_implementation(
                            reference.estimator, reference.solver, reference.native
                        )
                    else:
                        if original_threshold is not None:
                            estimator._CENTERED_INPUT_MIN_ELEMENTS = original_threshold if threshold is None else threshold
                        cls = SkLogisticRegression if variant == "sklearn" else LogisticRegression
                        settings, tuning = config_context, tuning_context
                        observed = observed_implementation(estimator, logistic, native_logistic)
                    models[variant], records[variant] = run_fit(
                        cls, settings, tuning, observed, X, y, force=force
                    )
                scores = {variant: score(model, X, y, tolerance) for variant, model in models.items()}
                reference_predictions = scores["sklearn"]["predict_from_float64_logits"]
                for score_result in scores.values():
                    predictions = score_result.pop("predict_from_float64_logits")
                    score_result["prediction_agreement_sklearn"] = float(
                        np.mean(predictions == reference_predictions)
                    )
                result = {
                    "name": name,
                    "shape": list(X.shape),
                    "dtype": str(X.dtype),
                    "seed": 461,
                    "offset": 1000,
                    "X_sha256": digest(X),
                    "y_sha256": digest(y),
                    "records": records,
                    "quality": scores,
                }
                payload["cases"].append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(name, {v: {"backend": records[v]["execution_backend"],
                                 "iterations": records[v]["n_iter"],
                                 "gradient": scores[v]["original_coordinate_gradient_max_abs"],
                                 "objective": scores[v]["objective_float64"]} for v in variants_to_names(variants)}, flush=True)
    finally:
        if original_threshold is not None:
            estimator._CENTERED_INPUT_MIN_ELEMENTS = original_threshold
    return 0


def variants_to_names(variants):
    return [variant[0] for variant in variants]


if __name__ == "__main__":
    raise SystemExit(main())
