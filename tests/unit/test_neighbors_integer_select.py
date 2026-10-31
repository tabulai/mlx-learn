"""Certified integer Gram scores retain canonical order in native selection."""

import numpy as np
import pytest

from mlxlearn._kernels import _neighbors_cpu_native as native
from mlxlearn._kernels import neighbors_cpu
from tests.unit.test_neighbors_cpu_exact import exact_oracle


def _inputs(train, queries, k):
    return [queries @ train.T,
            np.einsum("ij,ij->i", train, train).reshape(1, -1),
            np.einsum("ij,ij->i", queries, queries).reshape(-1, 1),
            k, False, 0,
            np.empty((len(queries), k), np.float64),
            np.empty((len(queries), k), np.int64)]


@pytest.mark.parametrize("k", [1, 5, 23])
@pytest.mark.parametrize("return_squared", [False, True])
def test_native_integer_selector_is_stable_at_exact_ties(k, return_squared):
    train = np.zeros((23, 64), np.float32)
    train[:4, 0] = [1, -1, 2, -2]
    train[4:8, 1] = [1, -1, 2, -2]
    queries = np.zeros((3, 64), np.float32)
    queries[1, 0], queries[2, 1] = 3, -3
    args = _inputs(train, queries, k)
    if not return_squared:
        args[6] = None
    before = [array.copy() for array in args[:3]]
    for array in args[:3]:
        array.setflags(write=False)
    assert native.select_integer_gram(*args) is None
    expected = exact_oracle(train, queries, k)
    np.testing.assert_array_equal(args[7], expected)
    if return_squared:
        delta = train[expected].astype(np.float64) - queries.astype(np.float64)[:, None]
        np.testing.assert_array_equal(args[6], np.sum(delta * delta, axis=2))
    for actual, expected_array in zip(args[:3], before, strict=True):
        np.testing.assert_array_equal(actual, expected_array)


def test_native_integer_selector_handles_inclusive_intermediate_bounds():
    train = np.array([[256] * 64, [-256] * 64, [0] * 64, [128] * 64], np.float32)
    queries = train[[0, 1]].copy()
    args = _inputs(train, queries, len(train))
    native.select_integer_gram(*args)
    expected = exact_oracle(train, queries, len(train))
    np.testing.assert_array_equal(args[7], expected)
    delta = train[expected].astype(np.float64) - queries.astype(np.float64)[:, None]
    np.testing.assert_array_equal(args[6], np.sum(delta * delta, axis=2))
    assert np.max(args[6]) == 2**24


@pytest.mark.parametrize("count", [15, 16, 17, 31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_native_integer_selector_keeps_late_winners_and_ties_across_wide_blocks(count, exclude_self):
    train = np.zeros((count, 4), np.float32)
    train[:, 0] = 5
    train[::3, 0] = 2
    train[-1, 0] = 1
    queries = np.zeros((3, 4), np.float32)
    args = _inputs(train, queries, 5)
    args[4:6] = exclude_self, 2
    native.select_integer_gram(*args)
    expected = exact_oracle(train, queries, len(train))
    if exclude_self:
        expected = np.stack([row[row != i + 2][:5] for i, row in enumerate(expected)])
    else:
        expected = expected[:, :5]
    np.testing.assert_array_equal(args[7], expected)


@pytest.mark.parametrize("position", [12, 15, 16, 31, 32, 60, 63, 64])
@pytest.mark.parametrize("bad", [np.nan, np.float32(2**-149), -np.float32(2**-149), 0.5])
def test_native_integer_selector_validates_later_skippable_blocks_before_writing(position, bad):
    args = _inputs(np.zeros((65, 4), np.float32), np.zeros((1, 4), np.float32), 5)
    args[0][0, position] = bad
    args[6].fill(-77)
    args[7].fill(-91)
    with pytest.raises(ValueError):
        native.select_integer_gram(*args)
    np.testing.assert_array_equal(args[6], -77)
    np.testing.assert_array_equal(args[7], -91)


@pytest.mark.parametrize("position", [0, 15, 16, 63, 64])
@pytest.mark.parametrize("bad", [np.nan, np.inf, 2**22 + 1, np.float32(2**-149), -0.5])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_integer_five_stages_all_rows_until_every_dot_is_valid(position, bad, exclude_self):
    args = _inputs(np.zeros((65, 4), np.float32), np.zeros((3, 4), np.float32), 5)
    args[4:6] = exclude_self, 0
    args[0][-1, position] = bad
    args[6].fill(-77)
    args[7].fill(-91)
    with pytest.raises(ValueError, match="integral and bounded"):
        native.select_integer_gram(*args)
    np.testing.assert_array_equal(args[6], -77)
    np.testing.assert_array_equal(args[7], -91)


@pytest.mark.parametrize("count,position", [(65, 2), (21, 19), (23, 21)])
@pytest.mark.parametrize("bad", [np.nan, np.float32(2**-149)])
@pytest.mark.parametrize("return_squared", [False, True])
def test_integer_five_checks_excluded_ids_and_vector_tails(count, position, bad, return_squared):
    args = _inputs(np.zeros((count, 4), np.float32), np.zeros((3, 4), np.float32), 5)
    args[4:6] = True, 0
    # The final row excludes ID 2. Counts21/23 also exercise the 4-wide
    # remainder and a scalar tail after it, rather than only 16-wide blocks.
    args[0][-1, position] = bad
    args[6].fill(-77)
    args[7].fill(-91)
    if not return_squared:
        args[6] = None
    with pytest.raises(ValueError, match="integral and bounded"):
        native.select_integer_gram(*args)
    if return_squared:
        np.testing.assert_array_equal(args[6], -77)
    np.testing.assert_array_equal(args[7], -91)


@pytest.mark.parametrize("bad", [np.nan, 0.5, np.float32(2**-149)])
def test_integer_five_keeps_domain_error_priority_after_negative_squared(bad):
    args = _inputs(np.zeros((65, 4), np.float32), np.zeros((3, 4), np.float32), 5)
    args[0][0, 0] = 1  # Valid integer input, but an inconsistent negative score.
    args[0][-1, -1] = bad
    args[6].fill(-77)
    args[7].fill(-91)
    with pytest.raises(ValueError, match="integral and bounded"):
        native.select_integer_gram(*args)
    np.testing.assert_array_equal(args[6], -77)
    np.testing.assert_array_equal(args[7], -91)


def test_integer_five_preserves_outputs_on_late_negative_squared():
    args = _inputs(np.zeros((65, 4), np.float32), np.zeros((3, 4), np.float32), 5)
    args[0][-1, -1] = 1
    args[6].fill(-77)
    args[7].fill(-91)
    with pytest.raises(ValueError, match="negative squared"):
        native.select_integer_gram(*args)
    np.testing.assert_array_equal(args[6], -77)
    np.testing.assert_array_equal(args[7], -91)


@pytest.mark.parametrize("count,exclude_self", [(5, False), (6, False), (6, True)])
def test_native_integer_five_fills_all_slots_at_minimum_reference_count(count, exclude_self):
    train = np.zeros((count, 4), np.float32)
    train[:, 0] = np.arange(count)[::-1]
    queries = train[-1:].copy()
    args = _inputs(train, queries, 5)
    args[4:6] = exclude_self, count - 1
    native.select_integer_gram(*args)
    expected = exact_oracle(train, queries, count)[0]
    if exclude_self:
        expected = expected[expected != count - 1]
    np.testing.assert_array_equal(args[7][0], expected[:5])


@pytest.mark.parametrize("offset", [0, 3, 17])
def test_native_integer_selector_excludes_identity_at_vector_and_row_tails(offset):
    train = np.zeros((23, 64), np.float32)
    train[:, :2] = np.arange(46).reshape(23, 2) % 7
    train[1:5] = train[0]
    queries = train[offset:offset + 6].copy()
    args = _inputs(train, queries, len(train) - 1)
    args[4:6] = True, offset
    native.select_integer_gram(*args)
    full = exact_oracle(train, queries, len(train))
    expected = np.stack([row[row != offset + i] for i, row in enumerate(full)])
    np.testing.assert_array_equal(args[7], expected)


def test_integer_cpu_index_uses_native_selector_across_query_tiles_and_rechecks_queries(monkeypatch):
    rng = np.random.default_rng(10201)
    train = rng.integers(-16, 17, size=(512, 64)).astype(np.float32)
    queries = train[:17].copy()
    actual_selector = neighbors_cpu._native_integer_select
    assert actual_selector is not None
    calls = []

    def record(*args):
        calls.append((args[0].shape, args[5]))
        return actual_selector(*args)

    monkeypatch.setattr(neighbors_cpu, "_native_integer_select", record)
    index = neighbors_cpu.CPUIndex(train)
    actual = index.query(queries, 5, exclude_self=True, query_block_size=7)
    np.testing.assert_array_equal(actual[1], exact_oracle(train, queries, 5, exclude_self=True))
    assert calls == [((7, 512), 0), ((7, 512), 7), ((3, 512), 14)]
    queries[0, 0] += np.float32(0.5)
    result = index.query(queries, 5)
    np.testing.assert_array_equal(result[1], exact_oracle(train, queries, 5))
    assert len(calls) == 3
    monkeypatch.setattr(neighbors_cpu, "_native_integer_select", None)
    fallback = index.query(train[:17], 5, exclude_self=True, query_block_size=7)
    np.testing.assert_array_equal(fallback[0], actual[0])
    np.testing.assert_array_equal(fallback[1], actual[1])


@pytest.mark.parametrize("argument", [0, 1, 2])
@pytest.mark.parametrize("position", [0, -1])
@pytest.mark.parametrize("bad", [0.5, -0.5, 2**22 + 1, np.inf, -np.inf, np.nan,
                                 np.float32(2**-149), -np.float32(2**-149)])
def test_native_integer_selector_rejects_nonintegral_nonfinite_or_out_of_bound_inputs(argument, position, bad):
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    # Exercise both a vectorized lane and the scalar input-validation tail.
    args[argument].flat[position] = bad
    with pytest.raises(ValueError):
        native.select_integer_gram(*args)


def test_native_integer_selector_rejects_inconsistent_negative_squared_values():
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    args[0][0, 0] = 1
    with pytest.raises(ValueError, match="negative squared"):
        native.select_integer_gram(*args)


@pytest.mark.parametrize("argument", [1, 2])
def test_native_integer_selector_rejects_negative_norms(argument):
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    args[argument].flat[0] = -1
    with pytest.raises(ValueError):
        native.select_integer_gram(*args)


@pytest.mark.parametrize("argument,kind", [
    (argument, kind) for argument in [0, 1, 2, 6, 7]
    for kind in ["dtype", "shape", "unaligned", "readonly"]
    if kind != "readonly" or argument >= 6
])
def test_native_integer_selector_checks_buffers(argument, kind):
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    array = args[argument]
    if kind == "dtype":
        args[argument] = array.astype(np.float64 if array.dtype == np.float32 else np.float32)
    elif kind == "shape":
        args[argument] = array.reshape(-1)
    elif kind == "unaligned":
        unaligned = np.ndarray(array.shape, array.dtype, buffer=bytearray(array.nbytes + 1), offset=1)
        unaligned[:] = array
        args[argument] = unaligned
    else:
        array.setflags(write=False)
    with pytest.raises((TypeError, ValueError, BufferError)):
        native.select_integer_gram(*args)


@pytest.mark.parametrize("input_argument", [0, 1, 2])
@pytest.mark.parametrize("output_argument", [6, 7])
def test_native_integer_selector_rejects_output_input_overlap(input_argument, output_argument):
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    source, output = args[input_argument], args[output_argument]
    storage = np.zeros(max(source.nbytes, output.nbytes), np.uint8)
    args[input_argument] = np.ndarray(source.shape, source.dtype, buffer=storage)
    args[output_argument] = np.ndarray(output.shape, output.dtype, buffer=storage)
    with pytest.raises(ValueError, match="overlap"):
        native.select_integer_gram(*args)


def test_native_integer_selector_accepts_signed_zero_and_empty_queries():
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    for array in args[:3]:
        array[:] = -0.0
    native.select_integer_gram(*args)
    np.testing.assert_array_equal(args[7], np.tile(np.arange(3), (5, 1)))
    native.select_integer_gram(*_inputs(np.zeros((7, 4), np.float32), np.zeros((0, 4), np.float32), 3))


@pytest.mark.parametrize("value", [np.float32(2**-149), -np.float32(2**-149), np.float32(0.5)])
@pytest.mark.parametrize("location", ["training_sample", "training_tail", "query"])
def test_integer_certificate_rejects_sparse_small_noninteger_bits(value, location):
    train = np.zeros((512, 64), np.float32)
    train[:, 0] = 1
    queries = train[:16].copy()
    if location == "training_sample":
        train[0, -1] = value
    elif location == "training_tail":
        train[-1, -1] = value
    else:
        queries[-1, -1] = value
    index = neighbors_cpu.CPUIndex(train)
    assert not index._integer_query(queries)
    assert not index.sparse_integer_batch(queries)


def test_integer_certificate_accepts_signed_zero_bits():
    train = np.full((512, 64), -0.0, np.float32)
    index = neighbors_cpu.CPUIndex(train)
    assert index._integer_query(train[:16])
    assert index.sparse_integer_batch(train[:16])


@pytest.mark.parametrize("kind", ["zero_k", "large_k", "self_large_k", "negative_offset",
                                  "self_past_end", "train_norm_shape", "query_norm_shape",
                                  "indices_shape", "squared_shape", "output_overlap"])
def test_native_integer_selector_rejects_invalid_dimensions_and_output_overlap(kind):
    args = _inputs(np.zeros((7, 4), np.float32), np.zeros((5, 4), np.float32), 3)
    if kind == "zero_k":
        args[3] = 0
    elif kind == "large_k":
        args[3] = 8
    elif kind == "self_large_k":
        args[3], args[4] = 7, True
    elif kind == "negative_offset":
        args[5] = -1
    elif kind == "self_past_end":
        args[4:6] = True, 3
    elif kind == "train_norm_shape":
        args[1] = args[1].T.copy()
    elif kind == "query_norm_shape":
        args[2] = args[2].T.copy()
    elif kind == "indices_shape":
        args[7] = np.empty((5, 4), np.int64)
    elif kind == "squared_shape":
        args[6] = np.empty((4, 3), np.float64)
    else:
        args[7] = args[6].view(np.int64)
    with pytest.raises(ValueError):
        native.select_integer_gram(*args)


def test_cpu_integer_selector_does_not_bypass_explicit_training_tiles(monkeypatch):
    train = np.random.default_rng(802).integers(-16, 17, size=(512, 64)).astype(np.float32)
    queries = train[:17].copy()

    def forbidden(*args):
        raise AssertionError("the full-width native selector must not bypass explicit training tiles")

    monkeypatch.setattr(neighbors_cpu, "_native_integer_select", forbidden)
    result = neighbors_cpu.CPUIndex(train).query(queries, 5, train_block_size=127)
    np.testing.assert_array_equal(result[1], exact_oracle(train, queries, 5))
