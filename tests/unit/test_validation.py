"""Input validation, capability checks, and the float32 range helpers.

The capability checks run on input *as the caller passed it* — the only window in
which patched mode can still fall back — so they have to read everything
scikit-learn reads, and fail only through the ``CapabilityError`` family. Most of
what is asserted here is one of those two halves: an input scikit-learn accepts
must not crash the check, and an input mlxlearn cannot serve must be a capability
error rather than scikit-learn's generic one.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tracemalloc
import warnings

import numpy as np
import pytest
import scipy.sparse
from sklearn.base import BaseEstimator

from mlxlearn._common.arrays import (
    FLOAT32_MAX,
    INDEX_DTYPE,
    MLX_FLOAT,
    ValueRange,
    as_float32,
    check_squared_float32_range,
    finite_range,
    is_mlx_array,
    sync,
    to_mlx,
    to_numpy,
)
from mlxlearn._common.config import tuning_context
from mlxlearn._common.diagnostics import get_backend_diagnostics
from mlxlearn._common.validation import (
    check_float32_range,
    check_sample_weight,
    count_labels,
    is_sparse,
    require_dense,
    require_nonnegative_weights,
    validate_float32_input,
)
from mlxlearn.exceptions import SparseInputError, UnsupportedInputError
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn.patching import _estimators as patched
from mlxlearn.svm import SVC

LAYER1 = [KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, LogisticRegression, SVC]
LAYER2 = [
    patched.KNeighborsClassifier,
    patched.KNeighborsRegressor,
    patched.NearestNeighbors,
    patched.LogisticRegression,
    patched.SVC,
]

#: Finite in float64, infinite in float32 — and small enough that stock
#: scikit-learn still fits every estimator on it, so a fallback can succeed.
OVERFLOW = 1e39


def _ids(cls):
    return f"{'L2' if cls._mlxlearn_allow_fallback else 'L1'}-{cls.__name__}"


def _output(model, X):
    if hasattr(model, "predict"):
        return model.predict(X)
    return model.kneighbors(X)[1]


@pytest.fixture
def data(rng):
    X = rng.normal(size=(240, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _overflow_events():
    return [e for e in get_backend_diagnostics() if e.reason == "float32-overflow"]


def _layer2_fit_headed_for_mlx(cls, X, y):
    """Fit a patched estimator on a problem it would otherwise run on MLX.

    ``data`` is below every crossover, and patched mode hands such fits to
    scikit-learn on size alone, before any data scan — by design, so that
    handing work away costs no more than stock scikit-learn. Forcing the MLX
    backend takes the size rule out, which is where the range check has to
    turn an overflow into a fallback; a capability mismatch still outranks the
    forced backend.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tuning_context(force_backend="mlx"):
            return cls().fit(X, y)


# ----------------------------------------------------------------------------------
# non-NumPy array-likes
# ----------------------------------------------------------------------------------


class _ForeignDtype:
    """A dtype with none of NumPy's attributes, like ``mlx.core.Dtype`` or ``torch.dtype``."""


class _ForeignArray:
    """The minimum scikit-learn accepts: ``shape`` and ``__array__``, with a foreign ``dtype``."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = _ForeignDtype()

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self._data if dtype is None else self._data.astype(dtype, copy=False)
        return out.copy() if copy else out


def _mx(X, dtype="float32"):
    import mlx.core as mx

    return mx.array(np.asarray(X, dtype=dtype), dtype=getattr(mx, dtype))


def _torch(X, dtype):
    torch = pytest.importorskip("torch")
    return torch.tensor(np.asarray(X), dtype=getattr(torch, dtype))


#: Each turns a float64 ndarray into another container scikit-learn accepts.
CONTAINERS = {
    "mx-float32": lambda X: _mx(X),
    "torch-float32": lambda X: _torch(X, "float32"),
    "torch-float64": lambda X: _torch(X, "float64"),
    "foreign-dtype": _ForeignArray,
    "list": lambda X: np.asarray(X).tolist(),
}

#: The containers that can carry a float64 value past float32's limit.
WIDE_CONTAINERS = {
    "mx-float64": lambda X: _mx(X, "float64"),
    "torch-float64": lambda X: _torch(X, "float64"),
    "foreign-dtype": _ForeignArray,
    "list": lambda X: np.asarray(X).tolist(),
}


@pytest.mark.parametrize("make", CONTAINERS.values(), ids=CONTAINERS.keys())
def test_range_check_reads_non_numpy_array_likes(make):
    """``mlx.core.Dtype`` has no ``.kind``; reading it crashed with AttributeError."""
    X = np.arange(12.0).reshape(4, 3)
    value_range = check_float32_range(make(X), need_range=True)
    assert value_range == ValueRange(0.0, 11.0)


@pytest.mark.parametrize("make", WIDE_CONTAINERS.values(), ids=WIDE_CONTAINERS.keys())
def test_range_check_catches_overflow_through_non_numpy_containers(make):
    X = np.ones((4, 3))
    X[2, 1] = 1e300
    with pytest.raises(UnsupportedInputError) as excinfo:
        check_float32_range(make(X))
    assert excinfo.value.reason == "float32-overflow"


@pytest.mark.parametrize("container", CONTAINERS, ids=str)
@pytest.mark.parametrize("cls", LAYER1, ids=_ids)
def test_layer1_fits_and_predicts_on_non_numpy_array_likes(cls, container, data):
    X, y = data
    X = X.astype(np.float32).astype(np.float64)  # every container holds the same values
    make = CONTAINERS[container]

    reference = cls().fit(X, y)
    model = cls().fit(make(X), y)

    np.testing.assert_array_equal(_output(model, make(X)), _output(reference, X))


@pytest.mark.parametrize("container", CONTAINERS, ids=str)
@pytest.mark.parametrize("cls", LAYER2, ids=_ids)
def test_layer2_serves_non_numpy_array_likes_on_mlx(cls, container, data):
    """Patched mode must not break code that works on stock scikit-learn."""
    X, y = data
    X = X.astype(np.float32).astype(np.float64)
    make = CONTAINERS[container]

    with tuning_context(force_backend="mlx"):
        reference = cls().fit(X, y)
        model = cls().fit(make(X), y)

    assert model.execution_backend_ == "mlx"
    np.testing.assert_array_equal(_output(model, make(X)), _output(reference, X))


@pytest.mark.parametrize("cls", LAYER1, ids=_ids)
def test_layer1_rejects_overflow_in_a_non_numpy_container(cls, data):
    X, y = data
    X = X.copy()
    X[0, 0] = OVERFLOW
    with pytest.raises(UnsupportedInputError) as excinfo:
        cls().fit(_ForeignArray(X), y)
    assert excinfo.value.reason == "float32-overflow"


@pytest.mark.parametrize("cls", LAYER2, ids=_ids)
def test_layer2_falls_back_on_overflow_in_a_non_numpy_container(cls, data):
    X, y = data
    X = X.copy()
    X[0, 0] = OVERFLOW
    model = _layer2_fit_headed_for_mlx(cls, _ForeignArray(X), y)
    assert model.execution_backend_ == "sklearn"
    assert _overflow_events()


# ----------------------------------------------------------------------------------
# object dtype and mixed-dtype frames
# ----------------------------------------------------------------------------------


def test_range_check_scans_object_arrays():
    """Object dtype used to return early, so 1e39 in an object array went unchecked."""
    X = np.ones((3, 2), dtype=object)
    X[1, 1] = OVERFLOW
    with pytest.raises(UnsupportedInputError, match="float32"):
        check_float32_range(X)

    X[1, 1] = -7.0
    assert check_float32_range(X) == ValueRange(-7.0, 1.0)


def test_range_check_leaves_unreadable_object_data_to_validation():
    """A capability check never invents a validation error; scikit-learn's own follows."""
    X = np.array([[1.0, "a"], [2.0, "b"]], dtype=object)
    assert check_float32_range(X) is None
    assert check_float32_range([[1.0, 2.0], [3.0]]) is None  # ragged


@pytest.mark.parametrize("cls", LAYER1, ids=_ids)
def test_layer1_object_array_overflow_is_a_capability_error(cls, data):
    """Layer 1 SVC used to fit silently on inf here, returning a NaN-intercept model."""
    X, y = data
    X = X.astype(object)
    X[3, 2] = OVERFLOW
    with pytest.raises(UnsupportedInputError) as excinfo:
        cls().fit(X, y)
    assert excinfo.value.reason == "float32-overflow"


@pytest.mark.parametrize("cls", LAYER2, ids=_ids)
def test_layer2_object_array_overflow_falls_back(cls, data):
    X, y = data
    X = X.astype(object)
    X[3, 2] = OVERFLOW
    model = _layer2_fit_headed_for_mlx(cls, X, y)
    assert model.execution_backend_ == "sklearn"
    assert _overflow_events()


def _mixed_frame(X, pd):
    frame = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
    frame["flag"] = X[:, 0] > 0
    return frame


def test_range_check_reads_mixed_dtype_frames_column_by_column():
    pd = pytest.importorskip("pandas")
    X = np.arange(12.0).reshape(4, 3)
    frame = _mixed_frame(X, pd)
    assert np.asarray(frame).dtype == object  # what the old check saw, and skipped
    assert check_float32_range(frame) == ValueRange(0.0, 11.0)

    frame.loc[2, "f1"] = OVERFLOW
    with pytest.raises(UnsupportedInputError, match="float32"):
        check_float32_range(frame)


def test_range_check_reads_nullable_extension_columns():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "a": pd.array([1.0, None, OVERFLOW], dtype="Float64"),
            "b": pd.array([1, 2, 3], dtype="Int64"),
        }
    )
    with pytest.raises(UnsupportedInputError, match="float32"):
        check_float32_range(frame)

    frame["a"] = pd.array([1.0, None, -4.0], dtype="Float64")
    assert check_float32_range(frame) == ValueRange(-4.0, 1.0)  # Int64 cannot overflow
    assert check_float32_range(frame, need_range=True) == ValueRange(-4.0, 3.0)


def test_range_check_skips_string_columns():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
    assert check_float32_range(frame) == ValueRange(1.0, 2.0)

    # An object column of numbers is still read when a string column cannot be.
    frame["o"] = pd.Series([OVERFLOW, 1.0], dtype=object)
    with pytest.raises(UnsupportedInputError):
        check_float32_range(frame)


def test_range_check_reads_frames_with_duplicate_column_names():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[1.0, -2.0, True], [OVERFLOW, 0.0, False]], columns=["a", "a", "b"])
    frame["b"] = frame["b"].astype(bool)
    with pytest.raises(UnsupportedInputError):
        check_float32_range(frame)


@pytest.mark.parametrize("cls", LAYER2, ids=_ids)
def test_layer2_mixed_frame_overflow_falls_back(cls, data):
    pd = pytest.importorskip("pandas")
    X, y = data
    frame = _mixed_frame(X, pd)
    frame.loc[5, "f2"] = OVERFLOW
    model = _layer2_fit_headed_for_mlx(cls, frame, y)
    assert model.execution_backend_ == "sklearn"
    assert _overflow_events()


@pytest.mark.parametrize("cls", LAYER2, ids=_ids)
def test_layer2_below_the_crossover_hands_overflow_to_sklearn_by_size(cls, data):
    """No scan at all: the fit is scikit-learn's on size, and it serves the overflow."""
    X, y = data
    X = X.astype(object)
    X[3, 2] = OVERFLOW
    with warnings.catch_warnings():
        # scikit-learn's own lbfgs reports 1e39 as ABNORMAL; not what is tested.
        warnings.simplefilter("ignore")
        model = cls().fit(X, y)
    assert model.execution_backend_ == "sklearn"
    assert model.fit_event_.reason == "below-crossover"
    assert not _overflow_events()


# ----------------------------------------------------------------------------------
# sparse detection
# ----------------------------------------------------------------------------------


def _sparse_frame(X, pd):
    return pd.DataFrame(
        {f"f{j}": pd.arrays.SparseArray(X[:, j], fill_value=0.0) for j in range(X.shape[1])}
    )


def test_is_sparse_follows_scikit_learns_pandas_rule():
    """Only an all-sparse frame is sparse; scikit-learn densifies a partly sparse one."""
    pd = pytest.importorskip("pandas")
    X = np.where(np.arange(12.0).reshape(4, 3) % 2 == 0, 0.0, 1.0)
    full = _sparse_frame(X, pd)
    mixed = full.copy()
    mixed["dense"] = X[:, 0]

    assert is_sparse(full)
    assert not is_sparse(mixed)
    assert not is_sparse(pd.DataFrame(X))
    assert not is_sparse(full["f0"])  # 1-D: validation rejects it on shape
    assert not is_sparse(pd.DataFrame())
    assert is_sparse(scipy.sparse.csr_matrix(X))
    assert not is_sparse(X)


def test_sparse_frame_message_names_the_pandas_conversion():
    pd = pytest.importorskip("pandas")
    full = _sparse_frame(np.eye(3), pd)
    with pytest.raises(SparseInputError, match=r"X\.sparse\.to_dense\(\)"):
        require_dense(full, estimator="SVC")
    with pytest.raises(SparseInputError, match=r"X\.toarray\(\)"):
        require_dense(scipy.sparse.csr_matrix(np.eye(3)), estimator="SVC")


@pytest.mark.parametrize("cls", LAYER1, ids=_ids)
def test_layer1_sparse_frame_is_a_sparse_input_error(cls, data):
    pd = pytest.importorskip("pandas")
    X, y = data
    X = np.where(np.abs(X) < 0.5, 0.0, X)
    with pytest.raises(SparseInputError):
        cls().fit(_sparse_frame(X, pd), y)


@pytest.mark.parametrize(
    "cls",
    [patched.KNeighborsClassifier, patched.NearestNeighbors, patched.LogisticRegression],
    ids=_ids,
)
def test_layer2_sparse_frame_falls_back_and_matches_stock(cls, data):
    """Stock SVC itself cannot fit an all-sparse frame, so it is not promised here."""
    pd = pytest.importorskip("pandas")
    X, y = data
    X = np.where(np.abs(X) < 0.5, 0.0, X)
    frame = _sparse_frame(X, pd)
    stock = next(c for c in cls.__mro__ if c.__module__.startswith("sklearn."))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls().fit(frame, y)
        expected = _output(stock().fit(frame, y), frame)
        observed = _output(model, frame)
    assert model.execution_backend_ == "sklearn"
    np.testing.assert_array_equal(observed, expected)


def test_partly_sparse_frame_is_still_served_by_mlx(data):
    pd = pytest.importorskip("pandas")
    X, y = data
    frame = _sparse_frame(X, pd)
    frame["dense"] = X[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scikit-learn's own densification warning
        with tuning_context(force_backend="mlx"):
            model = patched.KNeighborsClassifier().fit(frame, y)
    assert model.execution_backend_ == "mlx"


def test_validation_does_not_need_pandas():
    """``is_sparse`` and the range check import pandas only for pandas objects."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["pandas"] = None  # any `import pandas` now raises ImportError
        import numpy as np
        from mlxlearn._common.validation import check_float32_range, is_sparse, require_dense
        X = np.ones((3, 2))
        assert not is_sparse(X)
        require_dense(X, estimator="Probe")
        assert check_float32_range(X) is not None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------------
# check_float32_range: verdicts, range, cost
# ----------------------------------------------------------------------------------

_BELOW = float(np.nextafter(np.float32(FLOAT32_MAX), np.float32(0)))
_ABOVE = float(np.nextafter(FLOAT32_MAX, np.inf))

RANGE_CASES = {
    "normal": ([[1.0, -2.0], [3.0, 0.5]], ValueRange(-2.0, 3.0)),
    "positive-overflow": ([[1.0, 1e300]], "raise"),
    "negative-overflow": ([[-1e300, 1.0]], "raise"),
    "nan": ([[1.0, np.nan]], ValueRange(1.0, 1.0)),
    "inf": ([[np.inf, 2.0]], ValueRange(2.0, 2.0)),
    "inf-and-overflow": ([[np.inf, 1e300]], "raise"),
    "nan-and-overflow": ([[np.nan, -1e300]], "raise"),
    "all-nan": ([[np.nan, np.nan]], None),
    "float32-max": ([[FLOAT32_MAX, -FLOAT32_MAX]], ValueRange(-FLOAT32_MAX, FLOAT32_MAX)),
    "just-above-float32-max": ([[_ABOVE]], "raise"),
    "just-below-float32-max": ([[_BELOW]], ValueRange(_BELOW, _BELOW)),
    "empty": (np.empty((0, 3)), None),
}


@pytest.mark.parametrize("values, expected", RANGE_CASES.values(), ids=RANGE_CASES.keys())
@pytest.mark.parametrize("order", ["C", "F"])
def test_range_check_verdicts(values, expected, order):
    X = np.asarray(values, dtype=np.float64, order=order)
    if expected == "raise":
        with pytest.raises(UnsupportedInputError) as excinfo:
            check_float32_range(X)
        assert excinfo.value.reason == "float32-overflow"
        assert "float32" in str(excinfo.value)
    else:
        assert check_float32_range(X) == expected


def test_range_check_skips_what_cannot_overflow_unless_asked():
    for X in (np.ones((3, 2), np.float32), np.ones((3, 2), np.int64), np.ones((3, 2), bool)):
        assert check_float32_range(X) is None
        assert check_float32_range(X, need_range=True) == ValueRange(1.0, 1.0)


def test_range_check_allocates_nothing_for_finite_input(rng):
    """The old check built a mask, a gathered copy and an absolute value: ~2x the input."""
    X = rng.normal(size=(2000, 500))  # 8 MB
    check_float32_range(X)  # warm up any one-time allocation
    tracemalloc.start()
    try:
        check_float32_range(X)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < X.nbytes / 100, f"peak {peak} bytes for a {X.nbytes}-byte input"


def test_finite_range():
    assert finite_range(np.array([3, -1, 2])) == ValueRange(-1.0, 3.0)
    assert finite_range(np.array([-np.inf, 1.0, np.nan])) == ValueRange(1.0, 1.0)
    assert finite_range(np.array([np.inf])) is None
    assert finite_range(np.array([], dtype=np.float64)) is None
    assert finite_range(np.array(["a"])) is None
    assert finite_range(np.array([1 + 1j])) is None


# ----------------------------------------------------------------------------------
# the squared-range guard
# ----------------------------------------------------------------------------------


def test_value_range_magnitude_and_span():
    value_range = ValueRange(-3.0, 2.0)
    assert value_range.magnitude == 3.0
    assert value_range.span == 5.0


@pytest.mark.parametrize(
    "magnitude, n_terms",
    [(0.0, 10), (1.0, 1000), (1e18, 10), (1e-15, 10), (FLOAT32_MAX**0.5 / 4, 1)],
)
def test_squared_range_accepts_representable_squares(magnitude, n_terms):
    check_squared_float32_range(magnitude, n_terms=n_terms)


@pytest.mark.parametrize(
    "magnitude, n_terms, reason",
    [
        (1e19, 10, "float32-overflow"),  # fits in float32; ten squares do not
        (1e19, 1, "float32-overflow"),  # with the four-fold Gram-form headroom
        (np.inf, 1, "float32-overflow"),
        (1e-17, 10, "float32-underflow"),  # squares within 24 bits of subnormal
        (1e-20, 1, "float32-underflow"),
        (1e-200, 1, "float32-underflow"),  # squares to exactly 0.0 even in float64
    ],
)
def test_squared_range_rejects_squares_float32_cannot_hold(magnitude, n_terms, reason):
    with pytest.raises(UnsupportedInputError) as excinfo:
        check_squared_float32_range(magnitude, n_terms=n_terms, name="Q")
    assert excinfo.value.reason == reason
    assert str(excinfo.value).startswith("Q ")


def test_range_helpers_are_importable_without_scikit_learn():
    """``_kernels`` may call them, and a kernel module must not import scikit-learn.

    Also pins that ``is_mlx_array`` does not import MLX to classify an mlxlearn
    object: a bare ``startswith("mlx")`` matched ``mlxlearn.*`` too.
    """
    script = textwrap.dedent(
        """
        import sys
        from mlxlearn._common.arrays import check_squared_float32_range, finite_range, is_mlx_array
        from mlxlearn.exceptions import MLXLearnError
        import numpy as np
        finite_range(np.ones(3))
        check_squared_float32_range(1.0, n_terms=3)
        assert not is_mlx_array(MLXLearnError("x"))
        assert "sklearn" not in sys.modules, "arrays imported scikit-learn"
        assert "mlx.core" not in sys.modules, "is_mlx_array imported MLX"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------------
# validate_float32_input
# ----------------------------------------------------------------------------------


class _Probe(BaseEstimator):
    def fit(self, X, y=None):
        validate_float32_input(self, X, operation="fit")
        return self


def test_validate_float32_input_float32_path(rng):
    X = rng.normal(size=(20, 3))
    probe = _Probe().fit(X)
    assert probe.n_features_in_ == 3

    out = validate_float32_input(probe, X[:5])
    assert out.dtype == np.float32 and out.flags.c_contiguous

    # Inference must not reset n_features_in_ by default.
    with pytest.raises(ValueError, match="features"):
        validate_float32_input(probe, X[:5, :2])

    X_bad = X.copy()
    X_bad[0, 0] = 1e300
    with pytest.raises(UnsupportedInputError) as excinfo:
        validate_float32_input(probe, X_bad)
    assert excinfo.value.reason == "float32-overflow"


def test_validate_float32_input_float64_path_checks_the_validated_array(rng):
    """SVC validates in float64 and casts later; inference used to cast 1e39 to inf."""
    X = rng.normal(size=(20, 3))
    probe = _Probe().fit(X)

    out = validate_float32_input(probe, X, dtype=np.float64)
    assert out.dtype == np.float64

    X_bad = X.astype(object)
    X_bad[4, 1] = OVERFLOW
    with pytest.raises(UnsupportedInputError):
        validate_float32_input(probe, X_bad, dtype=np.float64)


def test_validate_float32_input_names_the_operation_for_sparse_input(rng):
    probe = _Probe().fit(rng.normal(size=(5, 3)))
    with pytest.raises(SparseInputError, match=r"_Probe\.decision_function"):
        validate_float32_input(
            probe, scipy.sparse.csr_matrix(np.eye(3)), operation="decision_function"
        )


def test_validate_float32_input_returns_y_when_given(rng):
    X = rng.normal(size=(8, 3))
    y = np.arange(8)
    X_out, y_out = validate_float32_input(_Probe(), X, y, operation="fit")
    assert X_out.dtype == np.float32
    np.testing.assert_array_equal(y_out, y)


# ----------------------------------------------------------------------------------
# sample weights
# ----------------------------------------------------------------------------------


def test_require_nonnegative_weights():
    require_nonnegative_weights(None, estimator="LogisticRegression")
    require_nonnegative_weights([1.0, 0.0, 2.0], estimator="LogisticRegression")
    # Malformed weights are check_sample_weight's to reject, after selection.
    require_nonnegative_weights("abc", estimator="LogisticRegression")
    require_nonnegative_weights([[1.0], [2.0, 3.0]], estimator="LogisticRegression")

    for weights in ([1.0, -0.5], -1.0, np.array([-1])):
        with pytest.raises(UnsupportedInputError, match="LogisticRegression") as excinfo:
            require_nonnegative_weights(weights, estimator="LogisticRegression")
        assert excinfo.value.reason == "negative-sample-weight"


def _labels(kind, values):
    if kind == "numpy":
        return np.asarray(values)
    if kind == "object":
        return np.asarray(values, dtype=object)
    if kind == "column":
        return np.asarray(values).reshape(-1, 1)
    if kind == "list":
        return list(values)
    pd = pytest.importorskip("pandas")
    if kind == "series":
        return pd.Series(values)
    if kind == "string":
        return pd.Series(values, dtype="string")
    return pd.Series(values, dtype="category")


@pytest.mark.parametrize(
    "kind", ["numpy", "object", "column", "list", "series", "string", "category"]
)
@pytest.mark.parametrize(
    "values, expected",
    [
        (["cat"] * 5, 1),
        (["cat", "dog", "cat", "dog", "dog"], 2),
        (["cat", "dog", "cat", "bird", "dog", "fish"], 4),
        ([3, 3, 1, 3], 2),
        ([0.5, 1.5, 2.5, 0.5], 3),
    ],
    ids=["one", "two", "four", "two-ints", "three-floats"],
)
def test_count_labels_counts_every_label_type(kind, values, expected):
    """A pandas Series of class names arrives as an object array. Read as binary, it
    sent a five-class LogisticRegression fit to the binary rule, so where the fit ran
    depended on whether its labels were integers or strings."""
    assert count_labels(_labels(kind, values)) == expected


@pytest.mark.parametrize(
    "y",
    [
        np.ones((4, 2)),  # not a single column
        np.empty((0, 2)),
        np.array([{1}, {2}, {3}], dtype=object),  # unhashable, and not comparable
        np.array([1 + 1j, 2 + 0j]),  # complex: not a label type scikit-learn takes
    ],
    ids=["two-columns", "empty-2d", "unhashable", "complex"],
)
def test_count_labels_gives_up_on_what_it_cannot_read(y):
    assert count_labels(y) is None


def test_count_labels_reads_binary_labels_without_sorting(monkeypatch):
    """Binary is the common case, and it is decided in two vectorized passes."""
    labels = np.array(["no", "yes"] * 50_000, dtype=object)
    monkeypatch.setattr(np, "unique", lambda *a, **k: pytest.fail("sorted binary labels"))
    assert count_labels(labels) == 2


def test_check_sample_weight():
    X = np.zeros((3, 2))
    with pytest.raises(UnsupportedInputError) as excinfo:
        check_sample_weight([1.0, -1.0, 2.0], X)
    assert excinfo.value.reason == "negative-sample-weight"

    weights = check_sample_weight([1.0, 0.0, 2.0], X, dtype=np.float64)
    np.testing.assert_array_equal(weights, [1.0, 0.0, 2.0])
    assert check_sample_weight(1.0, X) is None
    with pytest.raises(ValueError, match="NaN"):
        check_sample_weight([1.0, np.nan, 2.0], X)


# ----------------------------------------------------------------------------------
# arrays: copies, views, and the MLX boundary
# ----------------------------------------------------------------------------------


class _Wrapper:
    """An array-like whose ``__array__`` hands out its own buffer unless asked to copy."""

    def __init__(self, data):
        self._data = data

    def __array__(self, dtype=None, copy=None):
        return self._data.copy() if copy else self._data


class _Subclass(np.ndarray):
    pass


def _copy_sources(tmp_path):
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    memmap = np.memmap(tmp_path / "m.dat", dtype=np.float32, mode="w+", shape=(2, 3))
    memmap[:] = base
    sources = {
        "ndarray": base.copy(),
        "matrix": np.asmatrix(base.copy()),
        "memmap": memmap,
        "subclass": base.copy().view(_Subclass),
        "wrapper": _Wrapper(base.copy()),
        "memoryview": memoryview(base.copy()),
        "mx": _mx(base),
    }
    try:
        import pandas as pd
    except ImportError:
        pass
    else:
        sources["pandas"] = pd.DataFrame(base.copy())
    return sources


@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
def test_as_float32_copy_never_shares_memory(tmp_path):
    """``out is array`` missed every input whose conversion is a new view of the same buffer."""
    for name, source in _copy_sources(tmp_path).items():
        out = as_float32(source, copy=True)
        assert type(out) is np.ndarray, name
        assert out.flags.writeable and out.flags.c_contiguous, name
        host = np.asarray(source._data if name == "wrapper" else source)
        assert not np.shares_memory(out, host), name
        out[0, 0] = 99.0
        assert float(np.asarray(host)[0, 0]) == 0.0, name


def test_as_float32_without_copy_does_not_copy():
    X = np.ones((4, 3), dtype=np.float32)
    assert as_float32(X) is X
    assert as_float32(np.ones((4, 3))).dtype == np.float32


def test_to_numpy_view_of_an_mx_array_is_read_only():
    """A writable view let a write change the mx.array, and pending graphs built on it."""
    import mlx.core as mx

    a = mx.arange(3, dtype=mx.float32)
    doubled = a * 2
    view = to_numpy(a)
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = 5.0
    mx.eval(doubled)
    np.testing.assert_array_equal(np.asarray(doubled), [0.0, 2.0, 4.0])

    widened = to_numpy(a, dtype=np.float64)
    assert widened.dtype == np.float64 and widened.flags.writeable

    host = np.ones(3)
    assert to_numpy(host) is host


def test_to_mlx_round_trip_and_ownership():
    import mlx.core as mx

    host = np.arange(6.0).reshape(2, 3)
    device = to_mlx(host, device="cpu")
    assert device.dtype == mx.float32  # the float32 policy, applied on the host
    host[0, 0] = 99.0  # mx.array copied the host buffer
    np.testing.assert_array_equal(to_numpy(device), np.arange(6.0).reshape(2, 3))

    assert to_mlx(device, device="cpu") is device
    ints = to_mlx(device, dtype=mx.int32, device="cpu")
    assert ints.dtype == mx.int32
    assert to_mlx([1, 2], dtype=mx.int32, device="cpu").dtype == mx.int32


def test_sync_and_mlx_float():
    import mlx.core as mx

    assert MLX_FLOAT() == mx.float32
    pending = mx.ones(4) + 1
    sync(pending, np.ones(2), None)  # non-MLX values are ignored
    assert is_mlx_array(pending)
    assert not is_mlx_array(np.ones(2))
    assert not is_mlx_array([1.0])
    assert INDEX_DTYPE is np.int64
