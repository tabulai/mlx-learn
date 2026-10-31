"""CPU-only, default-thread comparison of fused and portable integer selection."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.knn_broad_fixtures import _real, workload_specs

from mlxlearn._kernels import _neighbors_cpu_native as native
from mlxlearn._kernels import neighbors_cpu
from mlxlearn.neighbors import NearestNeighbors


def measure(function):
    for _ in range(12):
        function()
    values = []
    for _ in range(101):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1000)
    return {'median_us': statistics.median(values), 'samples_us': values}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    selector = neighbors_cpu._native_integer_select
    assert selector is not None
    results = []
    for spec in workload_specs():
        if spec['family'] != 'real_digits_raw_k5':
            continue
        arrays = _real(spec)[0]
        X, Q = arrays['X'], arrays['Q']
        full = cdist(Q.astype(np.float64), X.astype(np.float64), 'sqeuclidean')
        expected = np.argsort(full, axis=1, kind='stable')[:, :5]
        expected_distances = np.sqrt(np.take_along_axis(full, expected, axis=1))
        case = {'case': spec['case']['name'], 'shape': [*X.shape, len(Q)], 'modes': {}}
        try:
            for name, entry in [('native_fused', selector), ('portable', None)]:
                neighbors_cpu._native_integer_select = entry
                index = neighbors_cpu.CPUIndex(X)
                model = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(X)
                for result in [index.query(Q, 5), model.kneighbors(Q)]:
                    np.testing.assert_array_equal(result[1], expected)
                    np.testing.assert_array_equal(result[0], expected_distances)
                assert model.last_query_backend_ == 'cpu'
                functions = {
                    'engine_query': lambda index=index, Q=Q: index.query(Q, 5),
                    'engine_fresh_owned_copy': lambda X=X, Q=Q: neighbors_cpu.CPUIndex(X.copy()).query(Q, 5),
                    'public_query': lambda model=model, Q=Q: model.kneighbors(Q),
                    'public_indices': lambda model=model, Q=Q: model.kneighbors(Q, return_distance=False),
                    'public_fresh': lambda X=X, Q=Q: NearestNeighbors(n_neighbors=5, algorithm='brute').fit(X).kneighbors(Q),
                }
                case['modes'][name] = {operation: measure(function) for operation, function in functions.items()}
                print(case['case'], name, {operation: round(value['median_us'], 3)
                      for operation, value in case['modes'][name].items()}, flush=True)
            dots = Q @ X.T
            xn = np.einsum('ij,ij->i', X, X).reshape(1, -1)
            qn = np.einsum('ij,ij->i', Q, Q).reshape(-1, 1)
            values = np.empty((len(Q), 5), np.float64)
            ids = np.empty((len(Q), 5), np.int64)
            case['components'] = {
                'gemm_only': measure(lambda X=X, Q=Q: Q @ X.T),
                'fused_selector_only': measure(lambda dots=dots, xn=xn, qn=qn, values=values, ids=ids: selector(dots, xn, qn, 5, False, 0, values, ids)),
            }
            results.append(case)
        finally:
            neighbors_cpu._native_integer_select = selector
    artifact = {'rows': results, 'numpy_version': np.__version__, 'threadpools': threadpool_info(),
                'thread_policy': 'unchanged default NumPy/Accelerate and process settings',
                'native_binary_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                'native_source_sha256': hashlib.sha256(Path('src/mlxlearn/_kernels/_neighbors_cpu_native.cpp').read_bytes()).hexdigest(),
                'cpu_source_sha256': hashlib.sha256(Path(neighbors_cpu.__file__).read_bytes()).hexdigest()}
    with output.open('x') as stream:
        json.dump(artifact, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
