"""Reproducible broad kNN workloads and exhaustive original-coordinate oracles.

No GPU or network work is performed here. Inputs are stored as float32 before
reference calculation. Genuine ties remain intact and are identified explicitly.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from .knn_pipeline import CASES, Case, sha256

DISTRIBUTIONS = ("gaussian", "clusters", "offset_1e4", "anisotropic")
SEEDS = (151, 251, 252)
REAL_DATASETS = {
    "digits": (1797, 64),
    "breast_cancer": (569, 30),
    "iris": (150, 4),
    "wine": (178, 13),
}


def workload_specs():
    """Return the fixed 72-instance panel: 24 workload families, three seeds."""
    specs = []
    for base_seed in (41, 42, 43):
        for position, original in enumerate(CASES):
            seed = base_seed + position
            case = asdict(original)
            case["name"] = f"shape_{original.name}_seed{seed}"
            specs.append({"case": case, "seed": seed, "base_seed": base_seed,
                          "family": f"shape_{original.name}", "source_kind": "synthetic",
                          "distribution": "gaussian", "suite": "established_shapes"})
    for n_train, n_query, features in ((50_000, 256, 32), (4096, 128, 128)):
        for distribution in DISTRIBUTIONS:
            family = f"distribution_{distribution}_{n_train}x{features}_k5"
            for seed in SEEDS:
                case = Case(f"{family}_seed{seed}", n_train, n_query, features, 5)
                specs.append({"case": asdict(case), "seed": seed, "family": family,
                              "source_kind": "synthetic", "distribution": distribution,
                              "suite": "distribution_panel"})
    for seed in SEEDS:
        family = "large_gaussian_100000x32_k5"
        specs.append({"case": asdict(Case(f"{family}_seed{seed}", 100_000, 256, 32, 5)),
                      "seed": seed, "family": family, "source_kind": "synthetic",
                      "distribution": "gaussian", "suite": "larger_training_set"})
    for dataset, (samples, features) in REAL_DATASETS.items():
        n_query = min(256, math.ceil(0.2 * samples))
        family = f"real_{dataset}_raw_k5"
        for seed in SEEDS:
            case = Case(f"{family}_seed{seed}", samples - n_query, n_query, features, 5)
            specs.append({"case": asdict(case), "seed": seed, "family": family,
                          "source_kind": "bundled_real", "distribution": "raw_real_features",
                          "dataset": dataset, "suite": "bundled_real"})
    assert len(specs) == 72
    assert len({spec["case"]["name"] for spec in specs}) == len(specs)
    return specs


def direct_reference(X, Q, k, *, temporary_bytes=64 << 20):
    """Select every neighbor from direct float64 differences, including k+1.

    The feature-sized difference tensor never exceeds temporary_bytes for the
    supported panel; it is squared in place. Training/query blocking also
    supports larger row counts without using a Gram product or approximate
    candidate shortlist. Only exact per-block lexicographic prefixes are merged.
    """
    import numpy as np

    n, features = X.shape
    if not 1 <= k <= n or Q.ndim != 2 or Q.shape[1] != features:
        raise ValueError("Invalid exhaustive-reference dimensions or k")
    train = X.astype(np.float64)
    query_block = min(4, max(1, temporary_bytes // (max(n, 1) * features * 8)))
    train_block = min(n, max(1, temporary_bytes // (query_block * features * 8)))
    take = min(k + 1, n)
    best_squared = np.empty((len(Q), take), np.float64)
    best_indices = np.empty((len(Q), take), np.int64)
    for start in range(0, len(Q), query_block):
        query = Q[start:start + query_block].astype(np.float64)
        retained_sq = retained_ids = None
        for offset in range(0, n, train_block):
            stop = min(offset + train_block, n)
            delta = query[:, None, :] - train[None, offset:stop, :]
            np.square(delta, out=delta)
            squared = np.sum(delta, axis=2, dtype=np.float64)
            del delta
            ids = np.broadcast_to(np.arange(offset, stop, dtype=np.int64), squared.shape)
            order = np.lexsort((ids, squared), axis=1)[:, :take]
            selected_sq = np.take_along_axis(squared, order, axis=1)
            selected_ids = np.take_along_axis(ids, order, axis=1)
            if retained_sq is not None:
                selected_sq = np.concatenate((retained_sq, selected_sq), axis=1)
                selected_ids = np.concatenate((retained_ids, selected_ids), axis=1)
                order = np.lexsort((selected_ids, selected_sq), axis=1)[:, :take]
                selected_sq = np.take_along_axis(selected_sq, order, axis=1)
                selected_ids = np.take_along_axis(selected_ids, order, axis=1)
            retained_sq, retained_ids = selected_sq, selected_ids
        best_squared[start:start + len(query)] = retained_sq
        best_indices[start:start + len(query)] = retained_ids
    expected_squared = best_squared[:, :k]
    boundary_ties = (best_squared[:, k - 1] == best_squared[:, k] if k < n
                     else np.zeros(len(Q), bool))
    internal_ties = np.any(np.diff(expected_squared, axis=1) == 0, axis=1)
    return {
        "distances": np.sqrt(expected_squared),
        "indices": best_indices[:, :k],
        "expected_squared": expected_squared,
        "oracle_next_squared": best_squared[:, k] if k < n else np.full(len(Q), np.inf),
        "boundary_tie_mask": boundary_ties,
        "internal_tie_mask": internal_ties,
    }


def _synthetic(spec):
    import numpy as np

    case = spec["case"]
    rng = np.random.default_rng(spec["seed"])
    X = rng.normal(size=(case["n_train"], case["n_features"]))
    Q = rng.normal(size=(case["n_query"], case["n_features"]))
    parameters = {"base_distribution": "independent standard normal"}
    distribution = spec["distribution"]
    if distribution == "clusters":
        centers = rng.normal(size=(16, case["n_features"])) * 8.0
        train_groups = rng.integers(0, len(centers), size=case["n_train"])
        query_groups = rng.integers(0, len(centers), size=case["n_query"])
        X = centers[train_groups] + 0.2 * X
        Q = centers[query_groups] + 0.2 * Q
        parameters.update(clusters=16, center_std=8.0, within_cluster_std=0.2)
    elif distribution == "offset_1e4":
        X += 1e4
        Q += 1e4
        parameters["common_offset"] = 1e4
    elif distribution == "anisotropic":
        scales = np.logspace(-2, 2, case["n_features"])
        X *= scales
        Q *= scales
        parameters["feature_scales"] = scales.tolist()
    elif distribution != "gaussian":
        raise ValueError(f"Unknown distribution: {distribution}")
    X, Q = X.astype(np.float32), Q.astype(np.float32)
    copied = min(2, len(Q)) if len(Q) > 1 else 0
    Q[:copied] = X[:copied]
    arrays = {"X": X, "Q": Q, "y": rng.integers(0, 4, size=len(X)),
              "targets": rng.normal(size=len(X))}
    metadata = {
        "source": "deterministic NumPy random-generator fixture",
        "parameters": parameters, "copied_training_queries": copied,
        "classification_response": "independent random four-class labels",
        "regression_response": "independent standard normal targets",
    }
    return arrays, metadata


def _real(spec):
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    loaders = {"digits": datasets.load_digits, "breast_cancer": datasets.load_breast_cancer,
               "iris": datasets.load_iris, "wine": datasets.load_wine}
    dataset = loaders[spec["dataset"]]()
    full_X = np.asarray(dataset.data, np.float32)
    full_y = np.asarray(dataset.target, np.int64)
    case = spec["case"]
    if full_X.shape != (case["n_train"] + case["n_query"], case["n_features"]):
        raise ValueError("Bundled dataset shape differs from the recorded workload specification")
    train_ids, query_ids = train_test_split(
        np.arange(len(full_X), dtype=np.int64), test_size=case["n_query"],
        random_state=spec["seed"], stratify=full_y,
    )
    arrays = {"X": full_X[train_ids], "Q": full_X[query_ids], "y": full_y[train_ids],
              "targets": full_y[train_ids].astype(np.float64),
              "source_train_indices": train_ids, "source_query_indices": query_ids,
              "query_labels": full_y[query_ids]}
    metadata = {
        "source": f"sklearn.datasets.load_{spec['dataset']} (bundled data; no download)",
        "parameters": {"preprocessing": "raw features converted to float32; no scaling or jitter",
                       "split": "stratified deterministic train/query holdout",
                       "test_size": case["n_query"], "random_state": spec["seed"]},
        "copied_training_queries": 0,
        "classification_response": "original dataset class labels",
        "regression_response": "numeric class labels used as a benchmark response; not a regression-quality claim",
        "source_array_sha256": {"data_float32": sha256(full_X.tobytes()),
                                "labels_int64": sha256(full_y.tobytes())},
        "target_names": [str(name) for name in dataset.target_names],
    }
    return arrays, metadata


def _prediction_references(arrays):
    import numpy as np

    distances, indices = arrays["distances"], arrays["indices"]
    y, targets = arrays["y"], arrays["targets"]
    classes = np.unique(y)
    result = {"classes": classes}
    for weighting in ("uniform", "distance"):
        if weighting == "uniform":
            weights = np.ones_like(distances)
        else:
            zeros = distances == 0
            with np.errstate(divide="ignore"):
                weights = np.where(np.any(zeros, axis=1, keepdims=True), zeros, 1 / distances)
        scores = np.stack([np.sum(weights * (y[indices] == label), axis=1) for label in classes], axis=1)
        result[f"proba_{weighting}"] = scores / scores.sum(axis=1, keepdims=True)
        result[f"predict_{weighting}"] = classes[scores.argmax(axis=1)]
        result[f"regress_{weighting}"] = (
            np.sum(weights * targets[indices], axis=1) / weights.sum(axis=1)
        )
    return result


def make_fixture(spec, destination):
    """Write a fixture NPZ and return its complete inspectable metadata."""
    import numpy as np

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    arrays, provenance = _real(spec) if spec["source_kind"] == "bundled_real" else _synthetic(spec)
    arrays.update(direct_reference(arrays["X"], arrays["Q"], spec["case"]["k"]))
    arrays.update(_prediction_references(arrays))
    path = destination / f"{spec['case']['name']}.npz"
    np.savez(path, **arrays)
    return {
        **spec, **provenance,
        "fixture_sha256": sha256(path.read_bytes()), "sha256": sha256(path.read_bytes()),
        "array_sha256": {name: sha256(np.ascontiguousarray(array).tobytes()) for name, array in arrays.items()},
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "input_dtype": "float32",
        "oracle": "exhaustive direct float64 squared differences on actual float32 coordinates; exact lexsort(squared distance,index)",
        "oracle_feature_temporary_budget_bytes": 64 << 20,
        "boundary_tie_query_count": int(np.count_nonzero(arrays["boundary_tie_mask"])),
        "internal_tie_query_count": int(np.count_nonzero(arrays["internal_tie_mask"])),
        "train_min": float(arrays["X"].min()), "train_max": float(arrays["X"].max()),
    }
