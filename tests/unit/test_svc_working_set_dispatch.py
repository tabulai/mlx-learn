"""Estimator routing must retain native fallbacks and the requested cache bound."""

import numpy as np
import pytest
from sklearn.svm import SVC as SkSVC

from mlxlearn._common.config import tuning_context
from mlxlearn._kernels.kernels import KernelRows
from mlxlearn._solvers import block_smo, native_smo
from mlxlearn.svm import SVC
from mlxlearn.svm import _classes as estimator


def _data():
    rng = np.random.default_rng(208)
    x = rng.normal(size=(300, 9)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    return x, y


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_bounded_working_set_batches_rows_without_a_full_gram(monkeypatch, backend):
    if not block_smo.available():
        pytest.skip("native working-set extension not built")
    x, y = _data()
    calls = []
    original = KernelRows.rows

    def tracked(self, indices):
        calls.append(len(indices))
        return original(self, indices)

    def forbidden(*args, **kwargs):
        raise AssertionError("the limited cache must not allocate a full Gram")

    monkeypatch.setattr(KernelRows, "rows", tracked)
    monkeypatch.setattr(estimator, "kernel_matrix", forbidden)
    with tuning_context(force_backend=backend):
        model = SVC(cache_size=0.012).fit(x, y)
    assert model._pair_solver_ == ("native_working_set",)
    capacity = model._kernel_cache_rows(len(x))
    assert 1 < max(calls) <= capacity // 2
    assert model.fit_status_ == 0
    assert model._pair_kkt_violation_[0] < model.tol
    reference = SkSVC().fit(x, y)
    np.testing.assert_allclose(model.decision_function(x), reference.decision_function(x),
                               atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("shrinking", [False, True])
def test_missing_block_extension_honors_pairwise_shrinking(monkeypatch, shrinking):
    if not native_smo.available():
        pytest.skip("native pair solver not built")
    monkeypatch.setattr(block_smo, "_smo_block_native", None)
    x, y = _data()
    model = SVC(kernel="linear", shrinking=shrinking).fit(x, y)
    assert model._pair_solver_ == (("native_shrinking" if shrinking else "native_pair"),)
    assert model.fit_status_ == 0
    assert model._pair_kkt_violation_[0] < model.tol


def test_missing_both_solvers_retains_portable_fit(monkeypatch):
    monkeypatch.setattr(block_smo, "_smo_block_native", None)
    monkeypatch.setattr(native_smo, "_smo_native", None)
    x, y = _data()
    model = SVC().fit(x, y)
    assert model._pair_solver_ == ("portable",)
    assert model.fit_status_ == 0
    assert np.mean(model.predict(x) == SkSVC().fit(x, y).predict(x)) >= 0.99
