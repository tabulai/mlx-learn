"""Compare CPU and GPU PCA query performance after a million-row fit."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA

import mlxlearn.decomposition._pca as pca_module
from mlxlearn.decomposition import PCA

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()

rng = np.random.default_rng(42)
X = rng.standard_normal((1_000_000, 1024), dtype=np.float64)
query = rng.standard_normal((10_000, 1024), dtype=np.float64)
models = {
    "sklearn": SkPCA(n_components=32, random_state=0).fit(X),
    "mlx_gpu": PCA(n_components=32, random_state=0).fit(X),
}
model = models["mlx_gpu"]
use_gpu = pca_module.pca_transform_uses_gpu


def host():
    pca_module.pca_transform_uses_gpu = lambda *args, **kwargs: False
    try:
        return model.transform(query)
    finally:
        pca_module.pca_transform_uses_gpu = use_gpu


calls = {
    "sklearn": lambda: models["sklearn"].transform(query),
    "mlx_gpu": lambda: model.transform(query),
    "mlx_cpu": host,
}
samples = {name: [] for name in calls}
names = tuple(calls)
for i in range(12):
    for name in names[i % len(names) :] + names[: i % len(names)]:
        start = time.perf_counter()
        result = calls[name]()
        if i >= 3:
            samples[name].append((time.perf_counter() - start) * 1000)
expected = (query[:1024] - model.mean_) @ model.components_.T
accuracy = {}
for name, call in (("mlx_gpu", calls["mlx_gpu"]), ("mlx_cpu", host)):
    score = call()[:1024]
    accuracy[name] = float(np.linalg.norm(score - expected) / np.linalg.norm(expected))
record = {
    "shape": list(X.shape),
    "query_shape": list(query.shape),
    "dtype": "float64",
    "samples_ms": samples,
    "median_ms": {k: statistics.median(v) for k, v in samples.items()},
    "projection_relative_error": accuracy,
}
args.out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
