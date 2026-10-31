# Exactness of the compact KNN filter

The compact path uses half-precision training storage to find a small candidate
set. Retained distances use original-coordinate compensated float32 arithmetic.
The original lower-bound derivation below targets that Kahan value. Production
selection now resolves overlapping rounding intervals using exact original
coordinate bits and converts omission bounds to real-distance endpoints before
acceptance. See the [precision-ordering argument](knn-precision-ordering.md).
Reported distances retain float32 precision; only true distance ties use index
order. Uncertain queries execute the complete scan with the same comparison.

This document covers
[`_neighbors_compact.py`](../src/mlxlearn_device/_neighbors_compact.py), including
its quantization error, packed keys, short local lists, and final certificate.
It uses scalar FP32 subtraction, multiplication, addition and explicit `fma`.
It does not assume an error bound for a SIMD matrix multiplication.

## Arithmetic and storage requirements

The internal caller supplies finite FP32 coordinates, at most 32 features, and a
feature count divisible by four. The current compact helper supports `k<=8` and
at most `2**21` training rows. The kernel disables implicit contraction and
reassociation, while using explicit `fma` for the inexpensive distance filter.
Custom kernels use MLX's safe math compilation. The relevant compiler setup is
visible in the [MLX 0.29 implementation](https://github.com/ml-explore/mlx/blob/v0.29.0/mlx/backend/metal/device.cpp)
and [MLX 0.32.2 implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp).
The latter's [compile-options definition](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/metal_kernel.h)
defaults to safe math; MLX 0.29 disables fast math when building the library.

The numerical model includes both permitted rounding directions and FP32
flush-to-zero behavior, using

```
u   = 2**-23
eta = 2**-126
```

These deliberately conservative constants cover correctly rounded scalar
operations and absolute errors from flushing. See sections 8.1–8.6 of the
[Metal Shading Language Specification, June 4, 2026](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Half storage does not introduce an ambiguous subnormal value. Section 8.6
requires lossless half-to-float conversion, ties-to-even float-to-half rounding,
and preservation of half subnormals produced by that conversion. Preparation
converts to half, stores those bits, and converts the same half value back to
FP32 when measuring its error. The scan loads the stored half bits and converts
them to FP32. Neither stage performs half arithmetic. Thus both stages use the
same represented value, including at the half subnormal boundary; an explicit
flush-to-zero modification is unnecessary. This conversion guarantee is also
specified independently of fast-math settings in the same specification.

## Quantization error and the lower bound

For one original training vector `t`, its stored half vector converted back to
FP32 `h`, and an original FP32 query `q`, define exact real squared norms

```
A = ||q - h||²
E = ||t - h||²
T = ||q - t||².
```

Preparation computes `Ec`, an FP32 FMA accumulation of the squared rounded
`t-h` differences. The scan computes `G` in the same way from rounded `q-h`
differences. `K` denotes the original guarded Kahan result on `q-t`.

Before ranking a pair, the scan requires both `G` and `Ec` to be finite and no
greater than `2**110`. These are sums of nonnegative terms. A failed guard gives
the pair a lower bound of negative infinity. In particular, a training value
that cannot be represented as finite half storage cannot pass this guard.

For at most 32 features, a conservative scalar error analysis gives

```
(1 - 64u) A - 4096 eta <= G  <= (1 + 64u) A + 4096 eta
(1 - 64u) E - 4096 eta <= Ec <= (1 + 64u) E + 4096 eta.
```

One way to obtain these loose bounds is to first bound each squared rounded
subtraction by `(1 +/- 4u)` times its exact square, with an absolute allowance of
`32 eta`. Nonnegative FMA chains and their final reductions add at most 32
rounding steps along a term's path. The constants above exceed the combined
relative and absolute allowances. They cover both the single distance chain and
the four preparation chains followed by their reduction.

The guards and the lower inequalities imply `A,E < 2**111`. Since
`T <= 2A + 2E`, the original distance has `T < 2**113`, leaving ample headroom
for its compensated FP32 intermediates. Including the original subtraction and
product rounding, its accumulation satisfies

```
K >= (1 - 512u) T - 4096 eta.
```

For completeness, a direct compensated-sum bound can be obtained by writing its
rounded squared terms as `p_i`, their real sum as `S`, and `M=S+4096 eta`.
Induction for at most 32 steps bounds the partial accumulated sum by `2M` and
the compensation magnitude by `8uM+16 eta`. The rounding errors in the adjusted
term, sum, and intermediate subtraction are respectively bounded by
`2uM+4 eta`, `4uM+4 eta`, and `3uM+4 eta`. Accumulated sum error is then at most
`448uS+769 eta`. Adding the squared-product and original subtraction allowances
gives the wider `512u` and `4096 eta` bound used above. The guard-derived range
ensures these induction bounds do not encounter overflow.

Young's inequality with `a=1/2000` gives

```
T >= (1-a) A - (1/a-1) E
  = .9995 A - 1999 E.
```

Let `b=1-512u`. Substitution yields

```
K >= b * [ .9995 * (G - 4096 eta) / (1 + 64u)
         - 1999 * (Ec + 4096 eta) / (1 - 64u) ] - 4096 eta
  >= .9994 G - 2000 Ec - 1e7 eta.
```

The unrounded coefficients in the first line are approximately
`.9994313703` and `1998.8932411`; its absolute allowance is less than
`8.2e6 eta`. The implemented, deliberately wider expression is

```
L = fl(fl(fl(.9993 * G) - fl(2050 * Ec)) - 1e-30).
```

The source comment describes the equivalent choice `a=2**-11`; that gives a
penalty coefficient of 2047 before arithmetic allowances. The `a=1/2000`
derivation above also proves the implemented coefficients and leaves a larger
margin on the quantization-error term.

Allowing upward rounding of each operation and the decimal literals gives the
conservative upper bound

```
L <= .999302 G - 2049.998 Ec - 9.99999e-31 + 32 eta.
```

Because `G,Ec>=0`, this is below `.9994G-2000Ec-1e7 eta`. The absolute slack is
also sufficient: `1e7 eta` is approximately `1.18e-31`. Consequently `L<=K`,
including tiny and zero distances. Unsafe pairs use `-inf`, which is always a
valid lower bound. The original scan handles their actual distances if needed.

## Packed keys preserve a lower bound

The scan maps each finite float lower bound to monotonically ordered unsigned
bits: invert all bits for a negative float, or flip the sign bit for a
nonnegative float. Clearing the lowest eight ordered bits rounds its represented
value downward. The remaining 24 bits encode the distance bucket; the low eight
bits encode the reference's position within its lane.

Partition selection guarantees at most 256 references per lane, so the position
fits without wrapping. Decoding reconstructs the reference index from its
partition start, lane, and position. The all-ones key is reserved for an empty
slot, and unsafe pairs decode to `-inf`. If an operation flushes a decoded FP32
subnormal, a positive bound can only decrease to zero; a negative bound remains
nonpositive and therefore cannot exceed a squared distance.

This quantization can admit extra candidates or trigger a fallback. It cannot
exclude a closer point on the strength of a bound rounded upward. Final distance
and index ordering never uses the packed bucket as the returned distance.

## Two candidates per lane and omitted-point barriers

Each lane retains its first two real candidates and the next key. The decoded
third key bounds every candidate omitted by that lane. Taking the minimum of
these third-key bounds produces a partition's omitted-point barrier.

The partition emits a sorted list containing real candidates and, when needed,
a sentinel-index entry carrying that barrier. A barrier is inserted before a
strictly larger real lower bound. Equal bounds are safe in either order: the
final certificate requires strict separation. If a partition emits only its
first `C` entries, its final emitted bound also bounds entries beyond that
truncation.

The finish kernel merges these partition lists, selecting `C=k+4` entries. It
tracks the minimum sentinel bound encountered and the last selected lower
bound. Their minimum, `B`, bounds every training point absent from the real
candidate set. Equal sentinel entries can be consumed together without losing
their minimum bound; a cursor advances only when both its head value and index
match the winning head. An unrelated, larger sentinel is never consumed early.

## Final certificate and fallback

The finish kernel computes compensated distances for retained candidates and
orders them with the adaptive exact comparison. Let `R` be the selected kth
point's computed distance and `B` the existing lower bound on every omitted
point's Kahan result. Acceptance requires at least k real candidates, finite R,
and strict separation of the converted endpoints:

```text
real_lower(B) > real_upper(R).
```

The endpoint functions and their error allowances are described in the
[precision-ordering argument](knn-precision-ordering.md). Every omitted point's
real distance is at least `real_lower(B)`, so none can beat or truly tie the kth
winner. In particular, equal rounded float32 values no longer justify an index
tie break or candidate omission.

Insufficient separation, unsafe values, too few candidates or an infinite
cutoff trigger a complete scan. Its local selection and partition merging use
the same exact ambiguous comparisons and original self-exclusion offset.
Only fitted training preparation is cached; results are not cached.

## Independent checks

[`test_neighbors_compact.py`](../tests/unit/test_neighbors_compact.py) invokes
the complete scan and precise partition merge directly as a cross-check, bypassing the compact dispatch.
Its 52 tests cover random and 50,000-row inputs, odd query batches, ties and
self-exclusion, multiple winners in one lane, half rounding boundaries and
subnormals, FP32 subnormals and overflow, large offsets, small terms alongside a
large squared term, noncontiguous and unaligned views, and packed position 255.
The lower-bound tests expose every pair in four adversarial fixtures and check
3,652 actual decoded bounds against the original Kahan distances. A fresh process
also runs the compact path while imports of NumPy and scikit-learn are blocked.
These tests supplement the argument; passing samples alone would not prove the
certificate for other finite inputs.
