"""Same-Python paired native A/B/A preservation attribution; never rebuilds."""
import hashlib
import importlib.util
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.knn_broad_fixtures import _real, workload_specs
from mlxlearn._kernels import neighbors_cpu, _neighbors_cpu_native as current
from mlxlearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor

OLD_PATH = Path('/tmp/mlxlearn-knn-digits-gap-baseline/_neighbors_cpu_native.cpython-313-darwin.so')
EXPECTED_OLD = '69b80a68bc3fe5a0937b99ab2d030a37b33b5fdfd6b38d632e8c6611555ff2c3'
EXPECTED_NEW = '5cb92f7ce3d70e8e1803f9d0b81c9753ec8e611f5c02d375c382fccb094d833a'
OUT = ROOT / 'benchmarks/results/knn-digits-breast-native-preservation-2026-10-30.json'
assert not OUT.exists()
def filehash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert filehash(OLD_PATH) == EXPECTED_OLD
assert filehash(current.__file__) == EXPECTED_NEW
spec = importlib.util.spec_from_file_location('breast_preservation_baseline._neighbors_cpu_native', OLD_PATH)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
MODES = {'old': old, 'new': current, 'new_b': current}
ORDERS = list(itertools.permutations(MODES))
HOOKS = {'_native_query': 'query', '_native_query_concentrated': 'query_concentrated',
         '_native_integer_select': 'select_integer_gram', '_native_feature_ranges': 'feature_ranges'}
def bind(module):
    for hook, function in HOOKS.items(): setattr(neighbors_cpu, hook, getattr(module, function))

def arrayhash(value):
    if isinstance(value, tuple): return [arrayhash(part) for part in value]
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()

artifact = {'status': 'running', 'purpose': 'Causal preservation attribution, same Python and arrays; no best-window selection',
            'warmup_rounds': 30, 'retained_rounds': 120, 'orders': ORDERS,
            'order_policy': 'All six mode permutations cycle; operation order rotates by round; every warmup/retained result audited and hashed outside timing',
            'old_native_sha256': EXPECTED_OLD, 'new_native_sha256': EXPECTED_NEW,
            'python_engine_sha256': filehash(neighbors_cpu.__file__), 'driver_sha256': filehash(__file__),
            'threadpools': threadpool_info(), 'thread_policy': 'unchanged process/default threads', 'rows': []}
try:
    for case in workload_specs():
        if case['family'] != 'real_breast_cancer_raw_k5': continue
        arrays = _real(case)[0]
        X, Q, labels, targets = arrays['X'], arrays['Q'], arrays['y'], arrays['targets']
        squared = cdist(Q.astype(np.float64), X.astype(np.float64), 'sqeuclidean')
        expected_ids = np.argsort(squared, axis=1, kind='stable')[:, :5]
        expected_distances = np.sqrt(np.take_along_axis(squared, expected_ids, axis=1))
        classes = np.unique(labels)
        votes = np.stack([(labels[expected_ids] == label).sum(axis=1) for label in classes], axis=1)
        expected_proba = votes / 5
        with np.errstate(divide='ignore'):
            weights = 1 / expected_distances
        zero = expected_distances == 0
        weights[zero.any(axis=1)] = zero[zero.any(axis=1)]
        weighted = np.stack([np.sum(weights * (labels[expected_ids] == label), axis=1)
                             for label in classes], axis=1)
        expected_predict = classes[np.argmax(weighted, axis=1)]
        expected_regression = np.mean(targets[expected_ids], axis=1)
        bind(current)
        index = neighbors_cpu.CPUIndex(X)
        train_permuted, order = index._prepare_native_train(len(Q))
        assert order is not None and index._native_concentrated
        query_permuted = np.ascontiguousarray(Q[:, order])
        models = {}
        for mode, module in MODES.items():
            bind(module)
            models[mode] = {'nn': NearestNeighbors(algorithm='brute').fit(X),
                            'class_distance': KNeighborsClassifier(algorithm='brute', weights='distance').fit(X, labels),
                            'class_uniform': KNeighborsClassifier(algorithm='brute', weights='uniform').fit(X, labels),
                            'reg_uniform': KNeighborsRegressor(algorithm='brute', weights='uniform').fit(X, targets)}
        output_distances = np.empty((len(Q), 5), np.float64)
        output_ids = np.empty((len(Q), 5), np.int64)
        names = ['native_direct', 'native_concentrated', 'public_query', 'first_query',
                 'fit_first_query', 'predict_distance', 'regress_uniform', 'proba_uniform']
        row = {'case': case['case']['name'], 'shape': [*X.shape, len(Q)],
               'arrays_sha256': {name: arrayhash(value) for name, value in arrays.items()},
               'feature_order': order.tolist(), 'samples': [], 'summary': {}}
        artifact['rows'].append(row)
        for round_index in range(150):
            phase = 'warmup' if round_index < 30 else 'retained'
            operation_order = names[round_index % len(names):] + names[:round_index % len(names)]
            for operation in operation_order:
                for mode in ORDERS[round_index % len(ORDERS)]:
                    module = MODES[mode]
                    bind(module)
                    fitted = models[mode]
                    first_model = NearestNeighbors(algorithm='brute').fit(X) if operation == 'first_query' else None
                    start = time.perf_counter_ns()
                    if operation == 'native_direct':
                        module.query(X, Q, 5, False, 0, output_distances, output_ids)
                        result = (output_distances, output_ids)
                    elif operation == 'native_concentrated':
                        module.query_concentrated(train_permuted, query_permuted, 5, False, 0, output_distances, output_ids)
                        result = (output_distances, output_ids)
                    elif operation == 'public_query':
                        result = fitted['nn'].kneighbors(Q)
                    elif operation == 'first_query':
                        result = first_model.kneighbors(Q)
                    elif operation == 'fit_first_query':
                        result = NearestNeighbors(algorithm='brute').fit(X).kneighbors(Q)
                    elif operation == 'predict_distance':
                        result = fitted['class_distance'].predict(Q)
                    elif operation == 'regress_uniform':
                        result = fitted['reg_uniform'].predict(Q)
                    elif operation == 'proba_uniform':
                        result = fitted['class_uniform'].predict_proba(Q)
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    max_error = 0.0
                    if isinstance(result, tuple):
                        np.testing.assert_array_equal(result[1], expected_ids)
                        np.testing.assert_allclose(result[0], expected_distances, rtol=1e-12, atol=1e-12)
                        max_error = float(np.max(np.abs(result[0] - expected_distances)))
                    elif operation == 'predict_distance':
                        np.testing.assert_array_equal(result, expected_predict)
                    elif operation == 'regress_uniform':
                        np.testing.assert_array_equal(result, expected_regression)
                    else:
                        np.testing.assert_array_equal(result, expected_proba)
                    row['samples'].append({'round': round_index, 'phase': phase, 'mode': mode,
                                           'operation': operation, 'elapsed_us': elapsed, 'accuracy_pass': True,
                                           'maximum_absolute_distance_error': max_error, 'result_sha256': arrayhash(result)})
        for operation in names:
            samples = {mode: [sample['elapsed_us'] for sample in row['samples']
                              if sample['phase'] == 'retained' and sample['operation'] == operation and sample['mode'] == mode]
                       for mode in MODES}
            assert all(len(values) == 120 for values in samples.values())
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            comparisons = {mode: {'ratio_of_medians_old_over_new': medians['old'] / medians[mode],
                                  'median_paired_ratio_old_over_new': statistics.median(a / b for a, b in zip(samples['old'], samples[mode], strict=True))}
                           for mode in ['new', 'new_b']}
            row['summary'][operation] = {'median_us': medians, 'comparisons': comparisons,
                                         'new_b_over_new_ratio_of_medians': medians['new_b'] / medians['new']}
        print(row['case'], json.dumps(row['summary']), flush=True)
    assert filehash(OLD_PATH) == EXPECTED_OLD and filehash(current.__file__) == EXPECTED_NEW
    artifact.update(status='complete', accuracy_pass=True,
                    native_hashes_unchanged=True, python_engine_hash_unchanged=filehash(neighbors_cpu.__file__) == artifact['python_engine_sha256'])
except BaseException as error:
    artifact.update(status='error', error=f'{type(error).__name__}: {error}')
    raise
finally:
    bind(current)
    with OUT.open('x') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
