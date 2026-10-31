"""Native feature permutation probe; permutation/preparation is always timed."""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.knn_broad_fixtures import _real, workload_specs

from mlxlearn._kernels import _neighbors_cpu_native as native


def query(train, queries):
    distances = np.empty((len(queries), 5), np.float64)
    indices = np.empty((len(queries), 5), np.int64)
    native.query(train, queries, 5, False, 0, distances, indices)
    return distances, indices


def prepare(train):
    ranges = train.max(axis=0).astype(np.float64) - train.min(axis=0).astype(np.float64)
    order = np.argsort(ranges)[::-1]
    assert np.sum(ranges[order[:8]]**2) >= .99*np.sum(ranges**2)
    return np.ascontiguousarray(train[:, order]), order


def measure(function):
    for _ in range(15):
        function()
    samples = []
    for _ in range(151):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return {"median_us": statistics.median(samples), "samples_us": samples}


results = []
for spec in workload_specs():
    if spec['family'] != 'real_breast_cancer_raw_k5':
        continue
    arrays = _real(spec)[0]
    X, Q = arrays['X'], arrays['Q']
    permuted, order = prepare(X)
    expected = query(X, Q)
    actual = query(permuted, np.ascontiguousarray(Q[:, order]))
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14, atol=0)

    def fresh(X=X, Q=Q):
        train, order = prepare(X)
        return query(train, np.ascontiguousarray(Q[:, order]))

    functions = {
        'original': lambda X=X, Q=Q: query(X, Q),
        'permuted_query_including_Q_copy': lambda train=permuted, Q=Q, order=order: query(train, np.ascontiguousarray(Q[:, order])),
        'fresh_including_ranges_permutation_X_Q': fresh,
    }
    row = {'case': spec['case']['name'], 'shape': [*X.shape, len(Q)], 'order': order.tolist(),
           'timings': {name: measure(function) for name, function in functions.items()}}
    results.append(row)
    print(row['case'], {name: value['median_us'] for name, value in row['timings'].items()}, flush=True)
Path('benchmarks/results/knn-auto-native-permutation-probe-2026-10-30.json').write_text(json.dumps(results, indent=2)+'\n')
