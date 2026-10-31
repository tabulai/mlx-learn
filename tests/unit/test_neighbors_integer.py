"""Integer screening preserves original neighbors and owned public state."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn.neighbors import NearestNeighbors
from mlxlearn_device import _neighbors_integer as integer
from mlxlearn_device import fit_neighbors
from mlxlearn_device._neighbors_fused import _precise_merge_kernel, _scan_kernel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
]


@pytest.fixture
def supported():
    if not integer.is_supported():
        pytest.skip("requires integer tensor operations")


def original_neighbors(train, queries, k, *, exclude_self=False, query_offset=0):
    """Bypass dispatch and compare against the original guarded Kahan scan."""
    n, d = train.shape
    nq, parts = queries.shape[0], 7
    partial = _scan_kernel()(
        inputs=[train, queries, mx.array([n, nq, d, parts, query_offset], mx.int32)],
        template=[("K", k), ("D", d), ("EXCLUDE_SELF", exclude_self)],
        grid=(((nq * parts * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, parts, k)] * 2,
        output_dtypes=[mx.float32, mx.int32],
    )
    return tuple(_precise_merge_kernel()(
        inputs=[*partial, train, queries, mx.array([parts, nq, n], mx.int32)],
        template=[("K", k), ("D", d)],
        grid=(((nq * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, k)] * 2,
        output_dtypes=[mx.float32, mx.int32],
    ))


def assert_exact(actual, expected):
    mx.eval(*actual, *expected)
    for value, reference in zip(actual[:2], expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("k", [1, 5, 8])
@pytest.mark.parametrize("case", [
    "normal", "duplicate", "zero", "quantization_ties", "small_scale_edge",
    "large_scale_edge", "subnormal", "overflow", "mixed_scale", "small_terms",
])
def test_integer_adversarial_distances_match_original(supported, case, k):
    rng = np.random.default_rng(629)
    # Each of 64 partitions contains at least one complete 64-reference tile.
    train = rng.normal(size=(4099, 32)).astype(np.float32)
    query = rng.normal(size=(9, 32)).astype(np.float32)
    if case == "duplicate":
        train[:30] = train[31]
        query[:5] = train[31]
    elif case == "zero":
        train[:] = 0
        query[:] = 0
    elif case == "quantization_ties":
        train[:] = 1
        train[:, 0] = (np.arange(len(train), dtype=np.float32) - 2049) / np.float32(127)
        query[:] = train[2047]
    elif case == "small_scale_edge":
        train *= np.float32(127 * 2**-32)
        query *= np.float32(127 * 2**-32)
    elif case == "large_scale_edge":
        train *= np.float32(127 * 2**32)
        query *= np.float32(127 * 2**32)
    elif case == "subnormal":
        train *= np.float32(2**-140)
        query *= np.float32(2**-140)
    elif case == "overflow":
        train[:] = np.float32(1e30)
        query[:] = np.float32(-1e30)
    elif case == "mixed_scale":
        train[:, 0] = np.float32(1e20)
        query[:, 0] = np.float32(1e20)
        train[5, 1] = np.float32(-1e30)
    elif case == "small_terms":
        train[:] = 30000
        train[0] = [10000] + [1] * 31
        train[1] = [10000, 4] + [0] * 30
        query[:] = 0
    X, Q = mx.array(train), mx.array(query)
    prepared = integer.prepare_integer(X)
    actual = integer.integer_neighbors(X, Q, k, prepared)
    assert_exact(actual, original_neighbors(X, Q, k))
    actual = integer.integer_neighbors(
        X, X[13:22], k, prepared, exclude_self=True, query_offset=13,
    )
    assert_exact(actual, original_neighbors(
        X, X[13:22], k, exclude_self=True, query_offset=13,
    ))


@pytest.mark.parametrize("n,nq", [(9, 1), (65, 7), (1027, 17), (16385, 65), (4099, 257)])
def test_integer_reference_and_query_tails(supported, n, nq):
    rng = np.random.default_rng(630 + n)
    X = mx.array(rng.normal(size=(n, 32)).astype(np.float32))
    Q = mx.array(rng.normal(size=(nq, 32)).astype(np.float32))
    actual = integer.integer_neighbors(X, Q, 8, integer.prepare_integer(X))
    assert_exact(actual, original_neighbors(X, Q, 8))


def test_integer_owned_state_and_outstanding_query_graphs(supported):
    rng = np.random.default_rng(51)
    host = rng.normal(size=(50000, 32)).astype(np.float32)
    baseline, source = mx.array(host.copy()), mx.array(host.copy())
    index = fit_neighbors(source)
    source[:] = 777
    Q = mx.array(rng.normal(size=(129, 32)).astype(np.float32))
    assert_exact(index.query(Q, 5, _squared=True), original_neighbors(baseline, Q, 5))
    prepared = index._integer
    assert prepared is not None and index._compact is None
    Q[:] = Q + np.float32(0.5)
    assert_exact(index.query(Q, 5, _squared=True), original_neighbors(baseline, Q, 5))
    assert index._integer is prepared
    A = mx.array(rng.normal(size=(65, 32)).astype(np.float32))
    B = mx.array(rng.normal(size=(64, 32)).astype(np.float32))
    out_a, out_b = index.query(A, 5, _squared=True), index.query(B, 5, _squared=True)
    assert_exact(out_b, original_neighbors(baseline, B, 5))
    assert_exact(out_a, original_neighbors(baseline, A, 5))
    assert index._integer is prepared
    omitted, indices = index.query(A, 5, return_distance=False)
    assert omitted is None
    np.testing.assert_array_equal(indices, original_neighbors(baseline, A, 5)[1])


def test_integer_estimator_refit_replaces_preparation(supported):
    rng = np.random.default_rng(51)
    train = rng.normal(size=(50000, 32)).astype(np.float32)
    query = rng.normal(size=(65, 32)).astype(np.float32)
    with tuning_context(force_backend="mlx"):
        model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(train)
        first = model.kneighbors(query)
        prepared = model._index_.device_index._integer
        assert prepared is not None
        train[:] = 0
        for value, reference in zip(model.kneighbors(query), first, strict=True):
            np.testing.assert_array_equal(value, reference)
        model.fit(train)
        assert model._index_.device_index._integer is None
        _, indices = model.kneighbors(query)
        np.testing.assert_array_equal(indices, np.tile(np.arange(5), (len(query), 1)))
        assert model._index_.device_index._integer is not prepared


def test_integer_unavailable_and_small_blocks_keep_existing_paths(monkeypatch):
    rng = np.random.default_rng(632)
    X = mx.array(rng.normal(size=(32768, 32)).astype(np.float32))
    Q = mx.array(rng.normal(size=(65, 32)).astype(np.float32))
    expected = original_neighbors(X, Q, 5)
    index = fit_neighbors(X)
    monkeypatch.setattr(integer, "is_supported", lambda: False)
    assert_exact(index.query(Q, 5, _squared=True), expected)
    assert index._integer is None and index._compact is not None
    small = fit_neighbors(X)
    assert_exact(small.query(Q, 5, query_block_size=32, _squared=True), expected)
    assert small._integer is None and small._compact is None


def test_integer_old_os_does_not_attempt_compilation(monkeypatch):
    integer.is_supported.cache_clear()
    monkeypatch.setattr(integer.platform, "mac_ver", lambda: ("25.9", ("", "", ""), ""))

    def forbidden(*args, **kwargs):
        raise AssertionError("old OS attempted integer tensor compilation")

    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    try:
        assert integer.is_supported() is False
    finally:
        integer.is_supported.cache_clear()


def test_integer_runtime_evaluation_failure_is_cached(monkeypatch):
    integer.is_supported.cache_clear()
    monkeypatch.setattr(integer.platform, "mac_ver", lambda: ("26.0", ("", "", ""), ""))
    calls = []

    def factory(**kwargs):
        return lambda **kwargs: [mx.zeros((32,), mx.int32)]

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("integer tensor operation unavailable at evaluation")

    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    monkeypatch.setattr(mx, "eval", unavailable)
    try:
        assert integer.is_supported() is False
        assert integer.is_supported() is False
        assert len(calls) == 1
    finally:
        integer.is_supported.cache_clear()


def test_integer_supported_runtime_probe_is_cached(supported):
    before = integer.is_supported.cache_info()
    assert integer.is_supported()
    after = integer.is_supported.cache_info()
    assert after.misses == before.misses
    assert after.hits == before.hits + 1


@pytest.mark.parametrize("centered", [False, True])
def test_integer_all_pair_bounds_are_conservative_and_nonvacuous(supported, centered):
    """Expose raw filter bounds before selection and compare every original pair."""
    from mlxlearn_device._neighbors_fused import refine_neighbors

    source = integer._SCAN
    fallback = "if(!metal::simd_all(valid_layout && peer_count==4)) {"
    assert fallback in source
    source = source.replace(fallback, fallback + """
    for(int reference=begin+int(lane);reference<end;reference+=32)
        for(int q=0;q<8;++q)if(query_base+q<nq)
            bounds[size_t(query_base+q)*n+reference]=-INFINITY;
""")
    marker = "        const uint bits=as_type<uint>(lower);"
    assert marker in source
    source = source.replace(marker, "        bounds[size_t(query)*n+reference]=lower;\n" + marker)
    kernel = mx.fast.metal_kernel(
        name=f"test_integer_all_pair_bounds_{int(centered)}",
        input_names=["packed_train", "train_biases", "train_scales", "queries", "center", "shape"],
        output_names=["out_distances", "out_indices", "bounds"],
        source=source,
        header=integer._HEADER + integer._CENTER_HEADER,
    )
    rng = np.random.default_rng(633)
    train = rng.normal(size=(259, 32)).astype(np.float32)
    train[64:96] *= np.float32(127 * 2**-32)
    train[96:128] *= np.float32(127 * 2**32)
    train[128:160] *= np.float32(2**-140)
    train[160:192] -= np.float32(1e6)
    train[192:224] *= np.float32(1e30)
    center = np.full(32, 10_000, np.float32) if centered else np.zeros(32, np.float32)
    if centered:
        center[::2] *= -1
        # Ordinary rows satisfy the exact-translation guard. Retain all original
        # scale/subnormal/overflow fixtures as unsafe untranslated rows.
        train[:64] += center
        train[224:] += center
    query = np.vstack([train[::17], center[None, :]])
    X, Q = mx.array(train), mx.array(query)
    C = mx.array(center)
    preparation = integer.prepare_integer(X, **({"center": C} if centered else {}))
    n, nq, parts, carry = len(train), len(query), 1, 16
    count = (nq + 7) // 8
    _, _, lower = kernel(
        inputs=[*preparation, Q, C, mx.array([n, nq, parts, 0], mx.int32)],
        template=[("D", 32), ("C", carry), ("BITS", (n - 1).bit_length()),
                  ("EXCLUDE_SELF", False), ("CENTERED", centered)],
        grid=(((count * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, parts, carry)] * 2 + [(nq, n)],
        output_dtypes=[mx.float32, mx.int32, mx.float32],
    )
    exact = refine_neighbors(X, Q, mx.broadcast_to(mx.arange(n, dtype=mx.int32), (nq, n)))
    mx.eval(lower, exact)
    bounds, reference = np.asarray(lower), np.asarray(exact)
    assert not np.isnan(bounds).any()
    assert np.all(bounds <= reference)
    assert np.count_nonzero(np.isfinite(bounds) & (bounds > 0)) >= 64
    if centered:
        assert np.isneginf(bounds[:, 64:224]).all()
    # Ordinary rows should exercise a useful certificate, not only -inf fallback.
    assert np.median(bounds[-1, :64] / reference[-1, :64]) > 0.95


@pytest.mark.parametrize("nq", [65, 129])
def test_integer_large_query_tails_actually_certify(supported, nq):
    rng = np.random.default_rng(51)
    X = mx.array(rng.normal(size=(50000, 32)).astype(np.float32))
    Q = mx.array(rng.normal(size=(nq, 32)).astype(np.float32))
    distances, indices, certified = integer.integer_neighbors(
        X, Q, 5, integer.prepare_integer(X), return_certificate=True,
    )
    assert_exact((distances, indices), original_neighbors(X, Q, 5))
    assert bool(mx.all(certified))


def test_integer_unavailable_platform_bypasses_compilation(monkeypatch):
    integer.is_supported.cache_clear()
    monkeypatch.setattr(integer.sys, "platform", "linux")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported platform attempted integer tensor compilation")

    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    try:
        assert integer.is_supported() is False
    finally:
        integer.is_supported.cache_clear()
