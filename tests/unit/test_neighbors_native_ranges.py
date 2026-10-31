"""Native range metadata preserves finite float32 extrema and storage safety."""

import numpy as np
import pytest

from mlxlearn._kernels import _neighbors_cpu_native as native


@pytest.mark.parametrize("features", [1, 4, 30, 32, 65])
def test_native_feature_ranges_match_original_float32_extrema(features):
    rng = np.random.default_rng(10280 + features)
    bits = rng.integers(0, 2**32, size=(37, features), dtype=np.uint32)
    bits = (bits & np.uint32(0x807fffff)) | (
        rng.integers(0, 255, size=bits.shape, dtype=np.uint32) << np.uint32(23)
    )
    train = bits.view(np.float32)
    before = train.view(np.uint32).copy()
    train.setflags(write=False)
    result = np.empty((1, features), np.float64)
    assert native.feature_ranges(train, result) is None
    original = train.astype(np.float64)
    np.testing.assert_array_equal(result[0], original.max(axis=0) - original.min(axis=0))
    np.testing.assert_array_equal(train.view(np.uint32), before)


def test_native_feature_ranges_preserve_subnormals_signed_zero_and_extreme_range():
    train = np.zeros((3, 7), np.float32)
    train[:, 0] = [0, 2**-149, 2**-148]
    train[:, 1] = [-2**-149, 0, 2**-149]
    train[:, 2] = [-0.0, 0.0, -0.0]
    train[:, 3] = [-np.finfo(np.float32).max, 0, np.finfo(np.float32).max]
    train[:, 4] = [1, 1, 1]
    train[:, 5] = [-2**-148, -2**-149, -0.0]
    train[:, 6] = [np.float32(2**-126), np.nextafter(np.float32(2**-126), np.float32(np.inf)), 0]
    output = np.empty((1, 7), np.float64)
    native.feature_ranges(train, output)
    expected = [[2**-148, 2**-148, 0, 2 * float(np.finfo(np.float32).max), 0,
                 2**-148, 2**-126 + 2**-149]]
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [0, 4])
def test_native_feature_ranges_reject_nonfinite_vector_and_scalar_lanes(bad, position):
    train = np.zeros((3, 5), np.float32)
    train[-1, position] = bad
    output = np.full((1, 5), 137, np.float64)
    with pytest.raises(ValueError, match="finite"):
        native.feature_ranges(train, output)
    np.testing.assert_array_equal(output, 137)


@pytest.mark.parametrize("kind", ["input_dtype", "output_dtype", "empty_train", "empty_features",
                                  "output_shape", "readonly_output", "input_strides",
                                  "unaligned_input", "unaligned_output", "overlap"])
def test_native_feature_ranges_validate_buffers(kind):
    train = np.zeros((16, 8), np.float32)
    output = np.empty((1, 8), np.float64)
    if kind == "input_dtype":
        train = train.astype(np.float64)
    elif kind == "output_dtype":
        output = output.astype(np.float32)
    elif kind == "empty_train":
        train = train[:0]
    elif kind == "empty_features":
        train, output = np.empty((16, 0), np.float32), np.empty((1, 0), np.float64)
    elif kind == "output_shape":
        output = np.empty((8, 1), np.float64)
    elif kind == "readonly_output":
        output.setflags(write=False)
    elif kind == "input_strides":
        train = train[::2]
    elif kind == "unaligned_input":
        train = np.ndarray(train.shape, train.dtype, buffer=bytearray(train.nbytes + 1), offset=1)
        train.fill(0)
    elif kind == "unaligned_output":
        output = np.ndarray(output.shape, output.dtype, buffer=bytearray(output.nbytes + 1), offset=1)
    else:
        output = np.ndarray(output.shape, output.dtype, buffer=train)
    with pytest.raises((ValueError, BufferError)):
        native.feature_ranges(train, output)
