"""Contracts for the optional native Metal command loop and its caller."""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import logistic as logistic_solver
from mlxlearn._solvers import native_command, native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig, LBFGSState


@pytest.fixture
def cpu():
    # Contract tests use a fake extension and must not contend with GPU work.
    with use_device("cpu"):
        yield


def _inputs(n=5, d=513):
    X = mx.ones((n, d), dtype=mx.float32)
    y = mx.array(np.arange(n) % 2, dtype=mx.float32)
    parts = mx.zeros((2, (n + 255) // 256, d + 1), dtype=mx.float32)
    return X, y, parts


def _objective(value=0.5, grad=0.0):
    def direct(theta):
        return value, mx.full(theta.shape, grad, dtype=mx.float32)

    return direct


def _fake_loop(monkeypatch, *, status=1, grad=0.0, callback=None):
    calls = []

    def run(source, capsules, n, d, p, history, blocks, groups, max_iter, *args):
        calls.append((source, capsules, n, d, p, history, blocks, groups, max_iter, args))
        if callback is not None:
            callback(capsules, args[4])
        return np.zeros(p, dtype=np.float32).tobytes(), 2, status, 16, 0.0, 0.0

    monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
    monkeypatch.setattr(native_command, "_source", lambda *args: "fake Metal source")
    return calls


def _run(X, y, parts, *, config=None, fun=None, weights=None, fit_intercept=True):
    return native_command.run_native_loop(
        X,
        y,
        weights,
        float(X.shape[0]),
        0.1,
        fit_intercept,
        LBFGSConfig() if config is None else config,
        _objective() if fun is None else fun,
        initial_parts=parts,
    )


def test_native_source_specializes_weighted_progress():
    unweighted = native_command._source(8, 512, 513, 10, 8, 1, 2, 16, 1, False)
    weighted = native_command._source(8, 512, 513, 10, 8, 1, 2, 16, 1, True)
    assert "#define WEIGHTED 0\n" in unweighted
    assert "#define WEIGHTED 1\n" in weighted
    assert "((WEIGHTED || D>=2048) ? 0.025f : 1.0f)" in weighted


@pytest.mark.parametrize("source, weighted", [("#define WEIGHTED 1\n", 0), ("#define WEIGHTED 0\n", 1)])
def test_native_rejects_inconsistent_weighted_specialization(source, weighted):
    if native_command._metal_loop is None:
        pytest.skip("requires the native extension")
    with pytest.raises(ValueError, match="weighted progress"):
        native_command._metal_loop.run(source, (), 1, 1, 2, 1, 1, 1, 1, 1., 1e-4, 1e-4, 1e-12,
                                      (), b"", None, 0, None, weighted)


def test_missing_extension_declines_without_preparing_inputs(cpu, monkeypatch):
    X, y, parts = _inputs()
    monkeypatch.setattr(native_command, "_metal_loop", None)

    def unexpected(*args, **kwargs):
        pytest.fail("unavailable extension must not prepare or execute the native loop")

    monkeypatch.setattr(native_command.mx, "eval", unexpected)
    assert _run(X, y, parts) is None


@pytest.mark.parametrize(
    "version", [None, "", "0.29.0", "0.30.0", "0.31.0", "0.31.99", "0.32.0rc1",
                "0.32.0.dev1", "0.32.0+local", "0.32.00", "0.33.0", "1.0.0", "unknown"]
)
def test_unsupported_mlx_version_falls_back_before_evaluation_or_export(cpu, monkeypatch, version):
    X, y, parts = _inputs()
    calls = _fake_loop(monkeypatch)
    monkeypatch.setattr(native_command.mx, "__version__", version, raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported MLX releases must not evaluate or export DLPack")

    monkeypatch.setattr(native_command.mx, "eval", unexpected)
    assert _run(X, y, parts) is None
    assert not calls


@pytest.mark.parametrize("version", ["0.32.0", "0.32.2", "0.32.99"])
def test_supported_stable_mlx_versions_allow_native_path(cpu, monkeypatch, version):
    X, y, parts = _inputs()
    monkeypatch.setattr(native_command.mx, "__version__", version, raising=False)
    calls = _fake_loop(monkeypatch)
    assert _run(X, y, parts).converged
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure",
    [
        'ImportError("broken extension load")',
        'ModuleNotFoundError("missing dependency", name="missing_dependency")',
    ],
)
def test_broken_extension_import_is_not_treated_as_absent(failure):
    # A fresh interpreter lets a finder intercept only the optional extension
    # without reloading or corrupting the imported module in this test process.
    script = f"""
import importlib.abc, importlib.machinery, sys
class Fault(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'mlxlearn._solvers._metal_loop':
            return importlib.machinery.ModuleSpec(fullname, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise {failure}
sys.meta_path.insert(0, Fault())
try:
    import mlxlearn._solvers.native_command
except ImportError as error:
    assert str(error) in ('broken extension load', 'missing dependency')
else:
    raise AssertionError('broken extension was silently treated as unavailable')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "d,intercept,history,line_search",
    [(511, True, 10, 32), (4097, True, 10, 32), (513, False, 10, 32),
     (513, True, 3, 32), (513, True, 10, 7)],
)
def test_unsupported_solver_contract_declines(cpu, monkeypatch, d, intercept, history, line_search):
    X, y, parts = _inputs(d=d)
    calls = _fake_loop(monkeypatch)
    config = LBFGSConfig(history_size=history, max_line_search=line_search)
    assert _run(X, y, parts, config=config, fit_intercept=intercept) is None
    assert not calls


def test_missing_or_invalid_initial_statistics(cpu, monkeypatch):
    X, y, parts = _inputs()
    calls = _fake_loop(monkeypatch)
    assert _run(X, y, None) is None
    with pytest.raises(ValueError, match="initial_parts"):
        _run(X, y, parts[:, :, :-1])
    assert not calls


def test_success_requires_direct_objective_and_gradient(cpu, monkeypatch):
    X, y, parts = _inputs()
    calls = _fake_loop(monkeypatch)
    checked = []

    def objective(theta):
        checked.append(np.asarray(theta).copy())
        return 0.25, mx.full(theta.shape, 1e-6, dtype=mx.float32)

    state = _run(X, y, parts, fun=objective)
    assert isinstance(state, LBFGSState)
    assert state.converged and state.stop_reason == "gtol"
    assert state.n_iter == 2 and state.n_fev == 17
    assert state.objective == 0.25 and state.grad_max_abs == pytest.approx(1e-6)
    np.testing.assert_array_equal(checked[-1], np.asarray(state.x))
    assert len(calls) == 1


@pytest.mark.parametrize("status", [0, 2, 3])
def test_unsupported_native_status_uses_existing_solver(cpu, monkeypatch, status):
    X, y, parts = _inputs()
    _fake_loop(monkeypatch, status=status)

    def unexpected(theta):
        pytest.fail("unsuccessful native state must not be presented as a result")

    assert _run(X, y, parts, fun=unexpected) is None


@pytest.mark.parametrize("value,grad", [(0.2, 1e-2), (np.nan, 0), (0.2, np.nan)])
def test_direct_certification_rejects_false_convergence(cpu, monkeypatch, value, grad):
    X, y, parts = _inputs()
    _fake_loop(monkeypatch)
    assert _run(X, y, parts, fun=_objective(value, grad)) is None


def test_extension_and_direct_objective_exceptions_propagate(cpu, monkeypatch):
    X, y, parts = _inputs()
    error = RuntimeError("native command failed")

    def broken(*args):
        raise error

    monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=broken))
    monkeypatch.setattr(native_command, "_source", lambda *args: "source")
    with pytest.raises(RuntimeError, match="native command failed") as caught:
        _run(X, y, parts)
    assert caught.value is error
    _fake_loop(monkeypatch)

    def broken_objective(theta):
        raise RuntimeError("direct objective failed")

    with pytest.raises(RuntimeError, match="direct objective failed"):
        _run(X, y, parts, fun=broken_objective)


def test_exported_capsules_and_owners_live_through_synchronous_native_call(cpu, monkeypatch):
    X, y, parts = _inputs()
    weights = mx.ones(X.shape[0], dtype=mx.float32)
    refs = [weakref.ref(array) for array in (X, y, weights)]
    real_eval = mx.eval
    calls = []

    def observed_eval(*args):
        calls.append("eval")
        return real_eval(*args)

    is_valid = ctypes.pythonapi.PyCapsule_IsValid
    is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
    is_valid.restype = ctypes.c_int

    def inspect(capsules, owners):
        # Producer evaluation must precede submission to the external queue.
        assert calls
        gc.collect()
        assert all(ref() is not None for ref in refs)
        assert len(capsules) == 6
        assert all(is_valid(capsule, b"dltensor") for capsule in capsules)
        assert len(owners) == 7 and owners[0] is capsules
        assert owners[1] is X and owners[2] is y and owners[3] is weights
        assert all(isinstance(array, mx.array) for array in owners[1:])

    monkeypatch.setattr(mx, "eval", observed_eval)
    _fake_loop(monkeypatch, callback=inspect)
    assert _run(X, y, parts, weights=weights).converged


def _six_dlpack_owners():
    class ExplicitOwner:
        pass

    arrays = [np.arange(4, dtype=np.float32) + i for i in range(6)]
    refs = [weakref.ref(array) for array in arrays]
    capsules = tuple(array.__dlpack__() for array in arrays)
    originals = [ExplicitOwner() for _ in range(6)]
    refs.extend(weakref.ref(original) for original in originals)
    return (capsules, *originals), refs


@pytest.mark.parametrize("wait_failures,status_failures", [(2, 0), (1, 2)])
def test_submitted_owners_survive_repeated_wait_or_status_failures(
    wait_failures, status_failures
):
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    loop._reap_completed()
    before = loop._test_retained_count()
    capsules, refs = _six_dlpack_owners()
    token = loop._test_begin_retention(capsules)
    del capsules
    try:
        assert loop._test_retained_count() == before + 1
        message = loop._test_fail_waits(token, wait_failures, status_failures)
        assert "simulated first wait failure" in message
        gc.collect()
        assert all(ref() is not None for ref in refs)
        assert loop._test_retained_count() == before + 1
        # Completion can arrive after the caller has already handled the error.
        loop._test_set_terminal(token)
        loop._reap_completed()
        gc.collect()
        assert loop._test_retained_count() == before
        assert all(ref() is None for ref in refs)
    finally:
        loop._test_set_terminal(token)
        loop._reap_completed()


def test_concurrent_submissions_reap_independently_after_delayed_completion():
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    loop._reap_completed()
    before = loop._test_retained_count()

    def submit(_):
        capsules, refs = _six_dlpack_owners()
        token = loop._test_begin_retention(capsules)
        del capsules
        loop._test_fail_waits(token, 2, 0)
        return token, refs

    with ThreadPoolExecutor(max_workers=4) as executor:
        submissions = list(executor.map(submit, range(4)))
    try:
        gc.collect()
        assert loop._test_retained_count() == before + 4
        assert all(ref() is not None for _, refs in submissions for ref in refs)
        for token, _ in submissions[:2]:
            loop._test_set_terminal(token)
        loop._reap_completed()
        gc.collect()
        assert loop._test_retained_count() == before + 2
        assert all(ref() is None for _, refs in submissions[:2] for ref in refs)
        assert all(ref() is not None for _, refs in submissions[2:] for ref in refs)
    finally:
        for token, _ in submissions:
            loop._test_set_terminal(token)
        loop._reap_completed()
    gc.collect()
    assert loop._test_retained_count() == before
    assert all(ref() is None for _, refs in submissions for ref in refs)


def test_separate_module_instances_have_independent_retention_registries():
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    spec = importlib.util.spec_from_file_location("_metal_loop", loop.__file__)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    loop._reap_completed()
    before = loop._test_retained_count()
    assert other._test_retained_count() == 0
    token = loop._test_begin_retention(tuple(object() for _ in range(6)))
    try:
        loop._test_fail_waits(token, 2, 0)
        assert loop._test_retained_count() == before + 1
        other._reap_completed()
        assert other._test_retained_count() == 0
        assert loop._test_retained_count() == before + 1
    finally:
        loop._test_set_terminal(token)
        loop._reap_completed()
    assert loop._test_retained_count() == before


def test_subinterpreter_teardown_cannot_reap_another_interpreters_owners():
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    if not any(importlib.util.find_spec(name) for name in ("_interpreters", "_xxsubinterpreters")):
        pytest.skip("this Python has no subinterpreter module")
    script = r'''
import importlib.util
try:
    import _interpreters as si
    def create(): return si.create("isolated")
except ImportError:
    import _xxsubinterpreters as si
    def create(): return si.create()
path = __MODULE_PATH__
def load():
    spec = importlib.util.spec_from_file_location("_metal_loop", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
main = load()
assert main._test_retained_count() == 0
first = create()
second = create()
source = """
import importlib.util
spec = importlib.util.spec_from_file_location('_metal_loop', %r)
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)
assert loop._test_retained_count() == 0
""" % path
assert si.run_string(first, source + """
token = loop._test_begin_retention(tuple(object() for _ in range(6)))
loop._test_fail_waits(token, 2, 0)
assert loop._test_retained_count() == 1
""") is None
assert si.run_string(second, source) is None
si.destroy(first)
assert si.run_string(second, "loop._reap_completed(); assert loop._test_retained_count() == 0") is None
si.destroy(second)
main._reap_completed()
assert main._test_retained_count() == 0
'''.replace("__MODULE_PATH__", repr(str(Path(loop.__file__).resolve())))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_abi3_none_returns_keep_reference_counts_balanced():
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    script = r'''
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('_metal_loop', __MODULE_PATH__)
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)
token = loop._test_begin_retention(tuple(object() for _ in range(6)))
loop._test_fail_waits(token, 2, 0)
loop._test_set_terminal(token)
loop._reap_completed()
before = sys.getrefcount(None)
for _ in range(1000):
    loop._test_set_terminal(token)
    loop._reap_completed()
after = sys.getrefcount(None)
assert after >= before - 10, (before, after)
'''.replace("__MODULE_PATH__", repr(str(Path(loop.__file__).resolve())))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.version_info < (3, 12), reason="per-interpreter GIL requires Python 3.12")
def test_subinterpreter_registries_are_safe_during_concurrent_execution():
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    if not any(importlib.util.find_spec(name) for name in ("_interpreters", "_xxsubinterpreters")):
        pytest.skip("this Python has no subinterpreter module")
    script = r'''
import threading
try:
    import _interpreters as si
    def create(): return si.create("isolated")
except ImportError:
    import _xxsubinterpreters as si
    def create(): return si.create()
interpreters = [create() for _ in range(2)]
source = """
import importlib.util
spec = importlib.util.spec_from_file_location('_metal_loop', %r)
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)
for _ in range(50):
    token = loop._test_begin_retention(tuple(object() for _ in range(6)))
    loop._test_fail_waits(token, 2, 0)
    assert loop._test_retained_count() == 1
    loop._test_set_terminal(token)
    loop._reap_completed()
    assert loop._test_retained_count() == 0
""" % __MODULE_PATH__
barrier = threading.Barrier(3)
errors = []
def execute(interpreter):
    try:
        barrier.wait()
        si.run_string(interpreter, source)
    except BaseException as error:
        errors.append(repr(error))
threads = [threading.Thread(target=execute, args=(interpreter,)) for interpreter in interpreters]
for thread in threads: thread.start()
barrier.wait()
for thread in threads: thread.join()
for interpreter in interpreters: si.destroy(interpreter)
assert not errors, errors
'''.replace("__MODULE_PATH__", repr(str(Path(loop.__file__).resolve())))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="requires a macOS Python framework")
def test_runtime_reinitialization_never_reaps_previous_interpreter_owners(tmp_path):
    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("optional native command extension is not built")
    clang = shutil.which("clang")
    prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
    framework = sysconfig.get_config_var("PYTHONFRAMEWORK")
    include = sysconfig.get_paths().get("include")
    if not all((clang, prefix, framework, include)):
        pytest.skip("a C compiler and Python framework headers are required")
    source = tmp_path / "reinitialize.c"
    executable = tmp_path / "reinitialize"
    marker = tmp_path / "released.txt"
    source.write_text(
        "#include <Python.h>\n"
        "int main(int argc, char **argv) {\n"
        "  for (int i = 1; i < argc; ++i) {\n"
        "    Py_Initialize();\n"
        "    if (PyRun_SimpleString(argv[i]) != 0) return 2;\n"
        "    if (Py_FinalizeEx() < 0) return 3;\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    )
    built = subprocess.run(
        [clang, "-I", include, "-F", prefix, "-framework", framework, str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    load = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('_metal_loop', {str(Path(loop.__file__).resolve())!r})\n"
        "loop = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(loop)\n"
    )
    first = load + (
        "class Owner:\n"
        f"    def __del__(self): open({str(marker)!r}, 'a').write('released\\n')\n"
        "token = loop._test_begin_retention(tuple(Owner() for _ in range(6)))\n"
        "loop._test_fail_waits(token, 2, 0)\n"
        "assert loop._test_retained_count() == 1\n"
    )
    second = load + (
        "assert loop._test_retained_count() == 0\n"
        "loop._reap_completed()\n"
        "assert loop._test_retained_count() == 0\n"
    )
    result = subprocess.run([str(executable), first, second], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists(), "a later interpreter released an earlier interpreter's owners"


def test_minimize_returns_native_success_and_declines_to_existing_solver(cpu, monkeypatch):
    X, y, parts = _inputs()
    expected = LBFGSState(mx.zeros((1, X.shape[1] + 1)), 0.0, 0.0, 0, 1, True, "gtol")
    monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: expected)
    result = native_logistic.minimize_binary(
        X, y, None, float(X.shape[0]), 0.1, True, LBFGSConfig(), _objective(),
        initial_parts=parts,
    )
    assert result is expected

    monkeypatch.setattr(native_command, "run_native_loop", lambda *a, **kw: None)

    def fallback_reached(*args):
        raise RuntimeError("existing solver reached")

    monkeypatch.setattr(native_logistic, "initial_from_parts", fallback_reached)
    with pytest.raises(RuntimeError, match="existing solver reached"):
        native_logistic.minimize_binary(
            X, y, None, float(X.shape[0]), 0.1, True, LBFGSConfig(), _objective(),
            initial_parts=parts,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [0, 7, 8, 9, 10])
def test_native_chunk_boundaries_honor_iteration_budget_and_fallback(monkeypatch, budget):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    from mlxlearn.linear_model import LogisticRegression

    original = native_command._metal_loop
    observed = []

    def run(*args):
        result = original.run(*args)
        observed.append(result[1:])
        return result

    monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
    rng = np.random.default_rng(41)
    X = rng.normal(size=(1000, 519)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
    weights = rng.uniform(0.25, 3.0, len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with tuning_context(force_backend="mlx"):
            model = LogisticRegression(C=1.0, tol=1e-10, max_iter=budget).fit(
                X, y, sample_weight=weights
            )
    assert model.execution_backend_ == "mlx"
    assert 0 <= model.n_iter_[0] <= budget
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    if budget == 0:
        assert not observed
        assert model.n_iter_[0] == 0
    else:
        assert len(observed) == 1
        native_iterations, status, *_ = observed[0]
        assert 0 <= native_iterations <= budget
        assert status != 1  # No direct certification: caller uses the existing fallback.


@pytest.mark.gpu
def test_native_weighted_progress_is_directly_certified(monkeypatch):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    from mlxlearn.linear_model import LogisticRegression

    extension = native_command._metal_loop
    observed = []
    direct = []
    original_objective = logistic_solver._native_binary_objective

    def run(*args):
        result = extension.run(*args)
        observed.append(result[1:])
        return result

    def objective(*args, **kwargs):
        fun = original_objective(*args, **kwargs)

        def checked(theta):
            direct.append(np.asarray(theta).copy())
            return fun(theta)

        return checked

    monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
    monkeypatch.setattr(logistic_solver, "_native_binary_objective", objective)
    rng = np.random.default_rng(41)
    X = rng.normal(size=(8000, 519)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
    weights = rng.uniform(0.25, 3.0, len(X))
    with tuning_context(force_backend="mlx"):
        model = LogisticRegression(C=1.0, tol=1e-4, max_iter=200).fit(X, y, sample_weight=weights)
    assert len(observed) == 1 and 9 < observed[0][0] <= 200 and observed[0][1] == 1
    assert model.n_iter_[0] == observed[0][0] and direct
    raw = X.astype(np.float64) @ model.coef_[0] + model.intercept_[0]
    residual = (expit(raw) - y) * weights
    denominator = weights.sum(dtype=np.float64)
    gradient = residual @ X / denominator + model.coef_[0] / denominator
    gradient = np.append(gradient, residual.sum() / denominator)
    assert np.max(np.abs(gradient)) <= 1e-4
    loss = np.dot(weights, np.logaddexp(0, raw) - y * raw) / denominator
    objective = loss + 0.5 * np.dot(model.coef_[0], model.coef_[0]) / denominator
    _, _, _, last_difference, previous_difference = observed[0]
    assert 0.5 * (abs(last_difference) + abs(previous_difference)) <= 0.025 * 1e-4 * objective

    observed.clear()
    with warnings.catch_warnings(record=True) as caught, tuning_context(force_backend="mlx"):
        warnings.simplefilter("always", ConvergenceWarning)
        limited = LogisticRegression(C=1.0, tol=1e-4, max_iter=9).fit(X, y, sample_weight=weights)
    assert len(observed) == 1 and observed[0][0] == limited.n_iter_[0] == 9
    assert any(issubclass(item.category, ConvergenceWarning) for item in caught)


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [1, 8, 9])
def test_native_initial_optimum_stops_without_a_step(monkeypatch, budget):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    from mlxlearn.linear_model import LogisticRegression

    extension = native_command._metal_loop
    observed = []

    def run(*args):
        result = extension.run(*args)
        observed.append(result[1:])
        return result

    monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
    X = np.zeros((512, 519), dtype=np.float32)
    y = np.arange(len(X)) % 2
    with tuning_context(force_backend="mlx"):
        model = LogisticRegression(tol=1e-4, max_iter=budget).fit(X, y)
    assert len(observed) == 1 and observed[0][0] == 0 and observed[0][1] == 1
    assert model.n_iter_[0] == 0
    np.testing.assert_array_equal(model.coef_, 0.0)
    np.testing.assert_array_equal(model.intercept_, 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("width", [519, 1025, 2049])
@pytest.mark.parametrize("weighted", [False, True])
def test_real_loop_fit_tails_ownership_and_independent_gradient(monkeypatch, weighted, width):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    if native_command._metal_loop is None:
        pytest.skip("optional native command extension is not built")
    from mlxlearn.linear_model import LogisticRegression

    # Exercise prepared input and real command execution with tail blocks.
    seen = []
    original = native_command.run_native_loop

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result.state if isinstance(result, native_command.NativeAttempt) else result)
        return result

    monkeypatch.setattr(native_command, "run_native_loop", observe)
    rng = np.random.default_rng(139)
    X = rng.normal(scale=0.1, size=(259, width)).astype(np.float32) + np.float32(0.25)
    y = rng.integers(0, 2, len(X))
    weights = np.linspace(0, 2, len(X), dtype=np.float32) if weighted else None
    before = X.copy()
    X.flags.writeable = False
    with tuning_context(force_backend="mlx"):
        model = LogisticRegression(C=0.1, tol=1e-5, max_iter=200).fit(X, y, sample_weight=weights)
    assert seen and seen[-1] is not None and seen[-1].converged
    assert model.execution_backend_ == "mlx"
    np.testing.assert_array_equal(X, before)
    raw = X.astype(np.float64) @ model.coef_[0] + model.intercept_[0]
    residual = expit(raw) - y
    denominator = len(X)
    if weighted:
        residual *= weights
        denominator = weights.sum(dtype=np.float64)
    grad = residual @ X / denominator + model.coef_[0] / (0.1 * denominator)
    grad = np.append(grad, residual.sum() / denominator)
    assert np.max(np.abs(grad)) <= 1.05e-5
    prediction = model.predict_proba(before[:10])
    X.flags.writeable = True
    X.fill(0)
    np.testing.assert_array_equal(model.predict_proba(before[:10]), prediction)
