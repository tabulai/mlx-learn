"""Held-out shape guard audit of automatic dispatch against forced GPU."""
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlxlearn._common.config import config_context, tuning_context
from mlxlearn.neighbors import NearestNeighbors


def fit(X, variant):
    with config_context(device='auto'), tuning_context(force_backend='' if variant == 'auto' else 'mlx'):
        return NearestNeighbors(n_neighbors=5, algorithm='brute').fit(X)


def timed(function):
    mx.synchronize()
    start = time.perf_counter_ns()
    result = function()
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1000, result


cases = [(n, d, q, 'gaussian') for n in [128, 512, 1024, 2048]
         for d in [8, 32, 64] for q in [1, 32, 256]]
cases += [(n, 64, 256, 'integer') for n in [512, 1541, 2048]]
results = []
for position, (n, d, queries, distribution) in enumerate(cases):
    rng = np.random.default_rng(62000 + position)
    if distribution == 'integer':
        X = rng.integers(0, 17, size=(n, d)).astype(np.float32)
        Q = rng.integers(0, 17, size=(queries, d)).astype(np.float32)
    else:
        X = rng.normal(size=(n, d)).astype(np.float32)
        Q = rng.normal(size=(queries, d)).astype(np.float32)
    squared = cdist(Q.astype(np.float64), X.astype(np.float64), 'sqeuclidean')
    expected = np.argsort(squared, axis=1, kind='stable')[:, :5]
    models = {variant: fit(X, variant) for variant in ['auto', 'gpu']}
    for model in models.values():
        distances, indices = model.kneighbors(Q)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_allclose(distances, np.sqrt(np.take_along_axis(squared, expected, axis=1)), rtol=1e-6, atol=1e-6)
    samples = {operation: {variant: [] for variant in models} for operation in ['query', 'fresh']}
    for operation in samples:
        for _warm in range(3):
            for variant, model in models.items():
                if operation == 'query':
                    model.kneighbors(Q)
                else:
                    fit(X, variant).kneighbors(Q)
        for repeat in range(7):
            for variant in (['auto', 'gpu'] if repeat % 2 == 0 else ['gpu', 'auto']):
                if operation == 'query':
                    duration, _ = timed(lambda model=models[variant], Q=Q: model.kneighbors(Q))
                else:
                    duration, _ = timed(lambda variant=variant, X=X, Q=Q: fit(X, variant).kneighbors(Q))
                samples[operation][variant].append(duration)
    medians = {operation: {variant: statistics.median(values) for variant, values in variants.items()}
               for operation, variants in samples.items()}
    row = {'shape': [n, d, queries], 'distribution': distribution, 'seed': 62000 + position,
           'medians_us': medians, 'samples_us': samples,
           'auto_over_gpu': {operation: values['auto'] / values['gpu'] for operation, values in medians.items()},
           'auto_backend': models['auto'].last_query_backend_, 'accuracy': 'pass'}
    results.append(row)
    print(row['shape'], distribution, {op: round(ratio, 3) for op, ratio in row['auto_over_gpu'].items()}, flush=True)
    Path('benchmarks/results/knn-auto-heldout-shape-guard-2026-10-30.json').write_text(json.dumps(results, indent=2)+'\n')
