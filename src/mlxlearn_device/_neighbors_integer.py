"""Certified 32-feature neighbors using exact integer matrix products.

Only the search filter quantizes coordinates. Metal Performance Primitives
computes exact bounded int8-by-int8 products into int32; measured quantization
errors give conservative lower bounds. Original FP32 compensated distances,
index tie ordering, self exclusion, and uncertain-query fallback are shared with
the compact neighbor implementation. No query result is cached.

The dispatch layer checks capability and shape eligibility before calling here.
"""

import platform
import sys
from functools import lru_cache

import mlx.core as mx

from ._neighbors_compact import _finish_kernel

_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
"""

_CENTER_HEADER = r"""
// Only use translations whose subtraction is exact by Sterbenz's lemma.
// The exponent guard also keeps nonzero differences above the Metal FTZ range.
// A failed guard disables this row's certificate, not its original search.
inline bool exact_neighbor_translation(float value, float center) {
    const float magnitude=metal::abs(value), anchor=metal::abs(center);
    const bool normal=magnitude>=0x1p-30f && magnitude<=0x1p30f;
    if(center==0.0f) return value==0.0f || normal;
    return normal && anchor>=0x1p-30f && anchor<=0x1p30f &&
           value*center>0.0f && magnitude>=0.5f*anchor && magnitude<=2.0f*anchor;
}
"""

_PREPARE = r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const int row=int(thread_position_in_grid.x);
if(row>=shape[0]) return;
float4 maximum=float4(0.0f);
bool centered_safe=true;
for(int f=0;f<D;f+=4) {
 float4 x=float4(*reinterpret_cast<device const packed_float4*>(values+size_t(row)*D+f));
 if(CENTERED) {
  const float4 c=float4(*reinterpret_cast<device const packed_float4*>(center+f));
  centered_safe=centered_safe && exact_neighbor_translation(x.x,c.x) &&
    exact_neighbor_translation(x.y,c.y) && exact_neighbor_translation(x.z,c.z) &&
    exact_neighbor_translation(x.w,c.w);
  x=centered_safe?x-c:float4(0.0f);
 }
 maximum=metal::max(maximum,metal::abs(x));
}
const float peak=metal::max(metal::max(maximum.x,maximum.y),metal::max(maximum.z,maximum.w));
const float scale=metal::max(0x1p-32f,metal::min(0x1p32f,peak/127.0f));
scales[row]=scale;
float4 error=float4(0.0f);
int4 norm=int4(0);
for(int f=0;f<D;f+=4) {
 float4 x=float4(*reinterpret_cast<device const packed_float4*>(values+size_t(row)*D+f));
 if(CENTERED) x=centered_safe?
   x-float4(*reinterpret_cast<device const packed_float4*>(center+f)):float4(0.0f);
 const char4 code=char4(metal::rint(metal::clamp(x/scale,float4(-127.0f),float4(127.0f))));
 *reinterpret_cast<device char4*>(packed+size_t(row)*D+f)=code;
 const float4 delta=x-float4(code)*scale;
 error=metal::fma(delta,delta,error);
 const int4 integers=int4(code);
 norm+=integers*integers;
}
const float measured=(error.x+error.y)+(error.z+error.w);
const int integer_norm=(norm.x+norm.y)+(norm.z+norm.w);
const float quant_norm=float(integer_norm)*(scale*scale);
const bool safe=centered_safe && metal::isfinite(measured) && measured<=0x1p110f;
// The reserved norm coefficient covers decoding, dot reconstruction, and
// final-expression rounding. Invalid rows force the original exact fallback.
// Expanding the paired biases gives .9842 times the quantized squared distance
// minus 128 times measured error, with a 1e-4 times norm-sum rounding reserve.
// The independently bounded rounding allowance is below 1.41e-6 times that
// norm sum, including arbitrary clamped scales and rounded decoded coordinates.
biases[row]=safe?0.9841f*quant_norm-128.0f*measured:-INFINITY;
"""

_SCAN = r"""
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const uint lane=thread_index_in_simdgroup;
const int n=shape[0],nq=shape[1],parts=shape[2];
const int task=int(thread_position_in_grid.x/32);
const int query_base=(task/parts)*8,part=task%parts;
if(query_base>=nq) return;
const int begin=int((size_t(part)*n)/parts),end=int((size_t(part+1)*n)/parts);
const uint mask=(1u<<BITS)-1u;
constexpr auto desc=tensor_ops::matmul2d_descriptor(8,64,D,false,true);
tensor_ops::matmul2d<desc,execution_simdgroup> op;
threadgroup int8_t query_cache[4*8*D];
threadgroup int8_t* cache=query_cache+(int(thread_index_in_threadgroup)/32)*8*D;
// Shared query storage always contains eight complete rows, including zeros
// for a final partial batch. Keep those rows visible to tensor ownership;
// output and candidate guards independently exclude queries beyond nq.
auto qa=tensor(cache,dextents<int,2>{D,8},array<int,2>{1,D});
auto example=tensor(const_cast<device int8_t*>(packed_train)+size_t(begin)*D,
    dextents<int,2>{D,metal::min(64,end-begin)},array<int,2>{1,D});
auto result=op.template get_destination_cooperative_tensor<decltype(qa),decltype(example),int>();
const bool first_valid=result.is_valid_element(ushort(0));
const int local_query=first_valid?result.get_multidimensional_index(ushort(0))[1]:-1;
const int query=query_base+local_query;
bool valid_layout=first_valid && local_query>=0 && local_query<8;
for(int i=0;i<result.get_capacity();++i)
    if(result.is_valid_element(i) && result.get_multidimensional_index(i)[1]!=local_query) valid_layout=false;
uint peer_mask=0;
for(int row=0;row<8;++row) {
    const uint mask=uint(ulong(metal::simd_ballot(local_query==row)));
    if(local_query==row) peer_mask=mask;
}
const int peer_count=metal::popcount(peer_mask);
int peers[4];
for(int i=0;i<4;++i) {peers[i]=int(metal::ctz(peer_mask));peer_mask&=peer_mask-1u;}

// The coordinate API determines ownership. A future device with a different
// ownership structure takes the unchanged exact fallback through unsafe lists.
if(!metal::simd_all(valid_layout && peer_count==4)) {
    if(lane<8 && query_base+int(lane)<nq) for(int slot=0;slot<C;++slot) {
        const size_t address=(size_t(query_base+int(lane))*parts+part)*C+slot;
        out_distances[address]=-INFINITY;out_indices[address]=2147483647;
    }
    return;
}
float qs=1.0f,qb=-INFINITY;
for(int row=0;row<8;++row) {
    const int qr=query_base+row;
    const float original=qr<nq?queries[size_t(qr)*D+lane]:0.0f;
    const bool centered_safe=!CENTERED || metal::simd_all(exact_neighbor_translation(original,center[lane]));
    const float value=CENTERED?(centered_safe?original-center[lane]:0.0f):original;
    const float peak=metal::simd_max(metal::abs(value));
    const float scale=metal::max(0x1p-32f,metal::min(0x1p32f,peak/127.0f));
    const int code=int(metal::rint(metal::clamp(value/scale,-127.0f,127.0f)));
    cache[row*D+lane]=int8_t(code);
    const float delta=value-float(code)*scale;
    const float measured=metal::simd_sum(delta*delta);
    const int integer_norm=metal::simd_sum(code*code);
    const float quant_norm=float(integer_norm)*(scale*scale);
    const bool safe=centered_safe && metal::isfinite(measured)&&measured<=0x1p110f;
    if(row==local_query) {qs=scale;qb=safe?0.9841f*quant_norm-128.0f*measured:-INFINITY;}
}
metal::simdgroup_barrier(mem_flags::mem_threadgroup);
const float qfactor=(2.0f*0.9842f)*qs;
uint key0=0xffffffffu;
uint key1=0xffffffffu;
uint key2=0xffffffffu;
uint key3=0xffffffffu;
uint key4=0xffffffffu;
for(int base=begin;base<end;base+=64) {
    auto tb=tensor(const_cast<device int8_t*>(packed_train)+size_t(base)*D,
        dextents<int,2>{D,metal::min(64,end-base)},array<int,2>{1,D});
    op.run(qa,tb,result);
    for(int i=0;i<result.get_capacity();++i) {
        if(!result.is_valid_element(i)) continue;
        const int reference=base+result.get_multidimensional_index(i)[0];
        if(reference>=end || query>=nq || (EXCLUDE_SELF && reference==query+shape[3])) continue;
        const float ts=train_scales[reference];
        const float cross=float(result[i])*(qfactor*ts);
        const float lower=(qb+train_biases[reference])-cross-1e-30f;
        const uint bits=as_type<uint>(lower);
        const uint ordered=(bits&0x80000000u)?~bits:(bits^0x80000000u);
        const uint prefix=metal::isfinite(lower)?(ordered&~mask):0u;
        const uint key=prefix|uint(reference-begin);
key4=metal::max(key3,metal::min(key4,key));
key3=metal::max(key2,metal::min(key3,key));
key2=metal::max(key1,metal::min(key2,key));
key1=metal::max(key0,metal::min(key1,key));
key0=metal::min(key0,key);
    }
}
uint barrier=0xffffffffu;
const uint local_barrier=key4&~mask;

for(int p=0;p<4;++p) barrier=metal::min(barrier,metal::simd_shuffle(local_barrier,peers[p]));
key4=0xffffffffu;

for(int slot=0;slot<C;++slot) {
    uint best=0xffffffffu;
    for(int p=0;p<4;++p) best=metal::min(best,metal::simd_shuffle(key0,peers[p]));
    const uint ordered=best&~mask;
    const bool inject=barrier<ordered;
    const uint selected=inject?barrier:ordered;
    const uint bits=(selected&0x80000000u)?(selected^0x80000000u):~selected;
    const float lower=selected==(0xffffffffu&~mask)?INFINITY:
        (selected<0x00800000u?-INFINITY:as_type<float>(bits));
    if(int(lane)==peers[slot%4] && query<nq) {
        const size_t address=(size_t(query)*parts+part)*C+slot;
        out_distances[address]=lower;
        out_indices[address]=inject||best==0xffffffffu?2147483647:begin+int(best&mask);
    }
    if(inject) barrier=0xffffffffu;
    else if(key0==best) {key0=key1; key1=key2; key2=key3; key3=key4;key4=0xffffffffu;}
}
"""

_SUPPORT_PROBE = r"""
constexpr auto descriptor = tensor_ops::matmul2d_descriptor(8, 32, 32, false, true);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
auto left = tensor(const_cast<device int8_t*>(values),
                   extents<int, 32, 8>{}, array<int, 2>{1, 32});
auto right = tensor(const_cast<device int8_t*>(values) + 8 * 32,
                    extents<int, 32, 32>{}, array<int, 2>{1, 32});
auto result = operation.get_destination_cooperative_tensor<
    decltype(left), decltype(right), int>();
operation.run(left, right, result);
output[thread_index_in_simdgroup] = result[ushort(0)];
"""


@lru_cache(maxsize=1)
def is_supported():
    """Probe Metal 4 integer tensor operations once, only on eligible hosts.

    A runtime probe also handles older MLX builds selecting an older Metal
    language or SDK. It runs on the first eligible query and caches the device
    capability. Unsupported hosts retain the compact FP32 filter.
    """
    if sys.platform != "darwin":
        return False
    try:
        if int(platform.mac_ver()[0].split(".")[0]) < 26:
            return False
    except ValueError:
        return False
    try:
        probe = mx.fast.metal_kernel(
            name="mlxlearn_integer_neighbors_support_probe",
            input_names=["values"],
            output_names=["output"],
            source=_SUPPORT_PROBE,
            header=_HEADER,
        )
        output = probe(
            inputs=[mx.ones((40, 32), mx.int8)],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(32,)],
            output_dtypes=[mx.int32],
            stream=mx.gpu,
        )
        mx.eval(*output)
        return all(value == 32 for value in output[0].tolist())
    except RuntimeError:
        return False


@lru_cache(maxsize=1)
def _prepare_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_integer_neighbors_prepare",
        input_names=["values", "center", "shape"],
        output_names=["packed", "biases", "scales"],
        source=_PREPARE,
        header=_CENTER_HEADER,
    )


@lru_cache(maxsize=1)
def _scan_kernel():
    return mx.fast.metal_kernel(
        name="mlxlearn_integer_neighbors_scan",
        input_names=["packed_train", "train_biases", "train_scales", "queries", "center", "shape"],
        output_names=["out_distances", "out_indices"],
        source=_SCAN,
        header=_HEADER + _CENTER_HEADER,
    )


def prepare_integer(train, *, center=None):
    """Build lazy owned int8 training codes, conservative biases, and scales.

    The internal caller supplies finite float32 coordinates, exactly 32 features,
    and 1..2**21 rows. Quantization scales are clamped to [2**-32, 2**32]; actual
    decoded errors account for clipping. An optional shared center is used only
    for rows passing an exact-subtraction guard. Unsafe rows produce negative-
    infinity biases so the certificate cannot discard them. Query screening
    must receive the same center; original coordinates still drive refinement.
    """
    if train.ndim != 2 or train.shape[1] != 32 or train.dtype != mx.float32:
        raise ValueError("integer neighbor preparation requires a float32 matrix with 32 features")
    n = train.shape[0]
    if not 1 <= n <= 2**21:
        raise ValueError("integer neighbor preparation requires 1..2**21 training rows")
    _validate_center(center)
    return tuple(
        _prepare_kernel()(
            inputs=[train, train if center is None else center, mx.array([n], mx.int32)],
            template=[("D", 32), ("CENTERED", center is not None)],
            grid=(((n + 127) // 128) * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n, 32), (n,), (n,)],
            output_dtypes=[mx.int8, mx.float32, mx.float32],
            stream=mx.gpu,
        )
    )


def _validate_center(center):
    if center is not None and (not isinstance(center, mx.array)
                               or center.shape != (32,) or center.dtype != mx.float32):
        raise ValueError("integer neighbor center requires a float32 vector with 32 features")


def integer_neighbors(
    train,
    queries,
    k,
    prepared,
    *,
    exclude_self=False,
    query_offset=0,
    return_certificate=False,
    center=None,
):
    """Return original FP32 squared distances and indices using a certified filter.

    Caller contract: finite float32 matrices with 32 features, 1<=k<=8,
    N<=2**21, and a valid self-query offset. ``prepared`` comes from this exact
    owned ``train`` array and the same optional ``center``. Centering never
    changes the original coordinates used in final comparisons and distances.
    The query codes live only in per-SIMD shared memory;
    all lanes participate even in query/reference tails. Tensor ownership is
    obtained through documented coordinates and ballots. An unfamiliar ownership
    arrangement emits unsafe lists and takes the existing complete exact scan.
    """
    if (
        train.ndim != 2
        or queries.ndim != 2
        or train.shape[1] != 32
        or queries.shape[1] != 32
        or train.dtype != mx.float32
        or queries.dtype != mx.float32
    ):
        raise ValueError("integer neighbors requires float32 matrices with 32 features")
    n, nq = train.shape[0], queries.shape[0]
    if not 1 <= n <= 2**21 or not 1 <= k <= 8:
        raise ValueError("integer neighbors requires 1..2**21 training rows and 1<=k<=8")
    _validate_center(center)
    if nq == 0:
        result = (mx.zeros((0, k), mx.float32), mx.zeros((0, k), mx.int32))
        return (*result, mx.zeros((0,), mx.bool_)) if return_certificate else result
    parts = min(64, n)
    # Translated rows can have tighter kth gaps after float32 coordinate
    # quantization. A wider shortlist avoids needless complete scans; the
    # unchanged omitted-candidate bound remains the acceptance criterion.
    carry = int(k) + (19 if center is not None else 11)
    # The inherited exact finish refines at most one candidate per SIMD lane.
    # carry <= 32 is therefore a correctness requirement, not only a tuning knob.
    task_count = ((nq + 7) // 8) * parts
    bits = max(1, ((n + parts - 1) // parts - 1).bit_length())
    shape = mx.array([n, nq, parts, query_offset], mx.int32)
    partial_distances, partial_indices = _scan_kernel()(
        inputs=[*prepared, queries, train if center is None else center, shape],
        template=[("D", 32), ("C", carry), ("BITS", bits), ("EXCLUDE_SELF", bool(exclude_self)),
                  ("CENTERED", center is not None)],
        grid=(((task_count * 32 + 127) // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nq, parts, carry)] * 2,
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    result = tuple(
        _finish_kernel()(
            inputs=[partial_distances, partial_indices, train, queries, shape],
            template=[
                ("D", 32),
                ("C", carry),
                ("K", int(k)),
                ("LISTS", (parts + 31) // 32),
                ("EXCLUDE_SELF", bool(exclude_self)),
            ],
            grid=(((nq * 32 + 127) // 128) * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(nq, k), (nq, k), (nq,)],
            output_dtypes=[mx.float32, mx.int32, mx.bool_],
            stream=mx.gpu,
        )
    )
    return result if return_certificate else result[:2]
