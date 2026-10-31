# Exactness of the integer-matrix KNN filter

[`_neighbors_integer.py`](../src/mlxlearn_device/_neighbors_integer.py) uses
quantized coordinates only to choose candidates. Every returned squared
distance uses original-coordinate compensated float32 arithmetic. Ordering now
resolves overlapping rounding intervals from the original coordinate bits;
only true distance ties use index order. The filter bounds below still target
the Kahan value, and production acceptance converts those bounds to
real-distance endpoints. See the [precision-ordering argument](knn-precision-ordering.md).
Uncertified rows run the complete scan with the same adaptive comparison.

The integer path reuses the compact implementation's finish kernel. Its
[final certificate and fallback proof](knn-compact-certificate.md#final-certificate-and-fallback)
therefore applies after establishing the lower bounds and omission barriers
described below.

## Scope and supported primitive

The internal helper requires finite FP32 matrices with exactly 32 features,
`1 <= k <= 8`, at most `2**21` training rows, and prepared arrays belonging to
the same owned training matrix. The selected configuration uses up to 64
reference partitions, an 8-by-64 output tile per SIMD group, 128 threads per
threadgroup, five local keys per lane, and `C=k+11` entries per partial list.
The optional centered route retains `C=k+19`, at most 27 entries for its supported
`k<=8`, within the finish kernel's 32-lane refinement capacity. A larger list
changes filtering efficiency only; the same omitted-point bound must certify it.
The dispatch layer also checks workload eligibility and runtime capability.

Production dispatch additionally estimates feature-variation concentration from
eight row differences spanning 16 evenly spaced rows of the owned training
matrix. If the estimated participation ratio is below 16, it retains the
compact filter instead of the integer filter. The decision is cached per fit;
original coordinates and distance arithmetic are unchanged. This bounded host
inspection is a performance heuristic, not part of either numerical
certificate: sampling can miss heterogeneous training data, and either selected
filter must still certify its candidates or execute its complete fallback.
Unsupported buffer inspection also retains compact search. See the
[unequal-scale regression fix](knn-scale-regression-fix-2026-10-29.md).

The matrix operation has signed 8-bit operands and a signed 32-bit result.
This combination is specified in table 7.3 of the
[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
section 7.2.1. The descriptor uses ordinary multiplication, with relaxed
precision disabled; it does not accumulate earlier reference tiles into the
current result. This is an integer product, so the proof does not assume an
accuracy bound for a floating-point matrix multiplication.

The scalar stages disable implicit contraction and reassociation. They use
the same conservative FP32 model as the
[compact certificate](knn-compact-certificate.md#arithmetic-and-storage-requirements):

```
u   = 2**-23
eta = 2**-126
```

The full epsilon `u` covers the permitted rounding directions; absolute
allowances cover FP32 flush-to-zero. The arithmetic model follows sections
8.1–8.6 of the Metal specification.

## Codes, scales, and measured errors

For each original row `x`, preparation computes

```
scale = clamp(max(abs(x)) / 127, 2**-32, 2**32)
code  = rint(clamp(x / scale, -127, 127))
```

The scale is an ordinary FP32 value, not necessarily a power of two. The
integer codes are stored exactly. Define two different reconstructions:

```
h_i = code_i * scale            # exact real product
r_i = fl(float(code_i) * scale) # represented FP32 decoded coordinate
```

The measured error uses `x-r`, including any clipping caused by the scale
clamp. Training preparation uses four positive eight-term FMA chains and a
final reduction. Query preparation uses one squared difference per SIMD lane
and a 32-lane sum. These reductions need not produce identical error bits;
the proof covers either order.

Because the scale is bounded, every nonzero reconstruction product is normal
and finite, and `|r_i-h_i| <= u|h_i|`. Each integer norm and dot product has
absolute value at most

```
32 * 127**2 = 516128 < 2**19.
```

Thus the integer operation cannot overflow, and conversion of its result to
FP32 is exact. Multiplication by an arbitrary scale still rounds; that
rounding is included below. The scale clamp keeps norm and cross-term
intermediates finite and normal whenever they are nonzero.

A row whose measured error is nonfinite or exceeds `2**110` receives a
negative-infinity bias. It cannot be discarded on the strength of the filter.
Input validation remains responsible for rejecting NaN and infinity in the
public input.

## From reconstructed distance to original FP32 distance

For a query and training row, define real quantities

```
H = ||h_q-h_t||²
N = ||h_q||² + ||h_t||²
T = ||q-t||²
E_r = ||q-r_q||² + ||t-r_t||²
E_h = ||q-h_q||² + ||t-h_t||².
```

Let `E` be the exact real sum of the two stored FP32 error values. Taking
`zeta=1/1024`, Young's inequality and the reconstruction rounding give

```
E_h <= (1+zeta) E_r + (1+1/zeta) u² N.
```

The positive error sums satisfy the deliberately loose envelope

```
E >= .99998 E_r - 16384 eta.
```

This follows from the compact certificate's per-row bound
`(1-64u) E_true - 4096 eta`. A query term encounters a rounded subtraction,
a square, and at most 31 additions along any reduction path. Its relative
allowance is within `64u`, including a subtraction-underflow allowance.
The combined bound above also leaves room for an optional rounded sum of the
two errors. It does not require a balanced SIMD reduction tree.

### Optional exact translation for a shared offset

Training preparation and query screening can use a common float32 center `c`.
The fitted index caches the training mean as its center for the offset route;
the internal helpers require the same center in preparation and every query.
This is a translation of the filter coordinates, not feature normalization.
Original training and query arrays are retained unchanged for final distance
calculation, exact ordering, self-exclusion and complete fallback.

Every coordinate is checked before a translated row can receive a finite bias.
For a nonzero center coordinate, the original value and center must have the
same sign, magnitudes in `[2^-30, 2^30]`, and magnitude ratio in `[1/2, 2]`.
Sterbenz's lemma then makes the float32 subtraction `x-c` exact. Any nonzero
difference between permitted input values is at least `2^-53`, their smallest
possible spacing, and so cannot flush to zero on Metal. The subtraction also
cannot overflow in this range. The guards themselves operate within normal,
finite ranges; their factor-of-two multiplications are exact.

A zero center coordinate is an identity translation. Its original value must
be zero or have magnitude in `[2^-30, 2^30]`; subnormal input is deliberately
ineligible. Thus an accepted translated row represents exactly the mathematical
vector `x-c`, including zero coordinates. For two accepted rows,
`||(q-c)-(t-c)||² = ||q-t||²` exactly as a real identity. The existing
quantization-error and original-distance inequalities therefore apply with
translated filter coordinates and the same original-distance `T`; no coefficient
or tolerance is weakened. The original-coordinate Kahan envelope is still used
at the final acceptance step.

If any coordinate fails its translation guard, its row bias is negative
infinity. It cannot be discarded on the strength of the filter. The finite
code/scale bounds keep reconstructed cross terms finite, so a pair with an
unsafe row has a negative-infinity lower bound. That bound is retained through
selection and omission barriers; it either brings the unsafe row into exact
refinement or prevents certification and triggers the existing complete scan.
Every query is guarded anew, including queries unlike the fitted training data.
Uncentered calls retain their original coefficients, list capacity and code path.

The [centered tests](../tests/unit/test_neighbors_integer_centered.py) compare
against direct float64 and Python-integer oracles on positive, negative and
mixed-sign offsets; rounded unequal distances and genuine ties; out-of-range
queries and training outliers; subnormals, overflow, and guard boundaries; and
self-exclusion with reference/query tails. The
[all-pair bound test](../tests/unit/test_neighbors_integer.py) exposes every
translated and uncentered filter bound before selection, checks it against
original-coordinate distances, and requires finite useful bounds for ordinary
rows while unsafe centered rows remain negative infinity.

Use the already established original-distance bound from the compact proof,
with `b=1-512u`:

```
K >= b T - 4096 eta,
```

where `K` is the returned original Kahan squared distance. The error guards
and bounded decoded coordinates leave sufficient FP32 range for that
accumulation. Young's inequality with `alpha=1/64` gives

```
T >= (63/64) H - 126 E_h.
```

Substituting the three inequalities yields

```
K >= A H - B E - R N - J eta

A = b * 63/64                         = .9843149185180664
B = 126*b*(1+zeta)/.99998              = 126.1178712925531
R = 126*b*(1+1/zeta)*u²                = 1.835219866700e-9
J = 4096 + 16384*B                    < 2070412.
```

In particular, `J*eta < 2.434e-32`. These looser constants suffice for the
production expression; no tighter Kahan estimate is needed.

## Prepared signed biases and rounding reserve

Let `a` and `c` denote the actual FP32 literals `.9842f` and `.9841f`, and
let `beta=128`. Per row the source computes

```
P    = fl(float(integer_norm) * fl(scale*scale))
bias = fl(fl(c*P) - fl(beta*measured_error)).
```

The query's `qfactor=fl((2*a)*query_scale)` is reused across references. Each
pair then computes

```
cross = fl(float(integer_dot) * fl(qfactor*train_scale))
L     = fl(fl(fl(query_bias+train_bias) - cross) - 1e-30f).
```

In exact arithmetic this is `a*H-beta*E-(a-c)*N-1e-30`. The coefficient
difference reserves approximately `1e-4*N` for rounding and for the distinction
between real and represented decoded coordinates. Computing this row term once
avoids loading and reconstructing the norm and error separately for every pair.

The following bound allows signed, including negative, row biases. Write
`rho=1+u`. The norm side of a row bias has at most four rounding stages; the
error side has at most two. Adding the two biases raises those counts to five
and three. The cross term has three rounded products. Since the absolute real
cross term is at most `N`, the final two subtractions give

```
L <= a H - S N - F E - (1-u)*tau + 1000 eta

S = a-c-c*(rho**7-1)-a*(rho**5-1)
F = beta*(2-rho**5)
tau = exact real value of the FP32 literal 1e-30f.
```

The additive allowance includes flushing a tiny error value before its
multiplication by 128. It is not necessary to assume positive subtraction
intermediates. Using the actual binary literals:

```
a                                  = .9842000007629395 < A
c*(rho**7-1)+a*(rho**5-1)            < 1.408e-6
S                                  > 9.8608e-5         > R
F                                  > 127.9999         > B
(1-u)*tau                          > (J+1000)*eta.
```

Since `H,N,E` are nonnegative, these comparisons prove `L <= K` for every
safe pair. They also handle cancellation between the two biases and the
cross term. If either row is unsafe, its bias is `-inf`, the cross term stays
finite under the scale/code bounds, and the pair's lower bound is `-inf`.

## Cooperative ownership, padding, and memory lifetime

Training codes, biases, and scales are owned MLX outputs derived from the
index's owned FP32 training buffer. The original coordinates remain available
for refinement. There is no borrowed NumPy buffer and no cached query result.

Each SIMD group writes its own 8-by-32 query-code region in threadgroup memory.
Every lane participates in all eight query preparations. Queries beyond the
last real row contribute zeros; their cache slots are still written. The query
tensor exposes all eight initialized rows, preserving a full ownership layout
on query tails. A uniform `simdgroup_barrier(mem_threadgroup)` precedes tensor
reads. Different SIMD groups use disjoint regions. Apple's
[synchronization guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
explains the need for a memory barrier even within a SIMD group.

The kernel obtains coordinates through `get_multidimensional_index` and uses
ballots to identify the four lanes owning one query's outputs. It verifies that
each lane's valid elements belong to one query and that each peer set has four
members. An unfamiliar layout emits unsafe lists and takes the exact fallback;
it does not guess the lane layout. The execution-scope requirements and
cooperative-tensor model are documented in sections 7.1–7.2 of the Metal
specification and sections 2.1 and 3.1 of Apple's
[MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).

All threads in a participating SIMD group reach `op.run`. Global query reads
are guarded before loading, candidate/output accesses require `query<nq`, and
reference accesses require `reference<partition_end`. Training tensors carry
the remaining tile extent. Self queries exclude `reference==query+query_offset`
before inserting a candidate. The complete fallback uses the same exclusion.

## Packed keys and omitted-point barriers

A key combines a downward-rounded distance prefix with the reference's offset
from its partition start. The offset occupies
`BITS=ceil(log2(max_partition_size))`, with at least one bit. Production uses
`parts=min(64,n)`, which gives `BITS<=15` over its supported range. Every relative index fits without
wrapping. The ordered-float bit mapping and downward truncation follow the
[compact key argument](knn-compact-certificate.md#packed-keys-preserve-a-lower-bound),
but the integer path stores a partition-relative index instead of a lane-relative
position.

Each lane keeps its smallest five keys. Four become real candidates; the fifth
is a lower bound for everything that lane omitted. The minimum fifth-key bound
over the four query peers becomes the partition barrier. The emitted sorted
list interleaves real candidates with a sentinel carrying that barrier. Clearing
low ordered bits can admit extra candidates or cause fallback; it cannot make
an omitted-point lower bound larger.

The inherited finish selects `C=k+11` entries across partition lists, records
encountered barriers and the final selected lower bound, and refines every real
candidate with the original arithmetic. Here `12<=C<=19`, within the correctness
limit of 32 candidates: the finish assigns at most one candidate to each lane.
Increasing `C` beyond 32 without changing the finish would be unsound.

Let `B_omitted` be the resulting lower bound on every unrefined point and `R_k`
the exact kth retained distance. Acceptance requires at least `k` real
candidates, a finite cutoff, and `B_omitted > R_k`. The strict comparison also
protects equal-distance index ties. Otherwise the complete original scan
replaces the tentative output. The
[shared omission and finish proof](knn-compact-certificate.md#two-candidates-per-lane-and-omitted-point-barriers)
does not depend on how many real keys each local lane kept.

## Capability checks and verification

The cached support check first requires macOS 26 or later, then compiles and
executes a small integer TensorOp through the active MLX/Metal toolchain. An
all-ones length-32 product must return 32 for every checked lane. A compilation
or runtime failure, or a wrong result, leaves the existing compact path in use.
The OS check alone is not treated as proof of hardware/toolchain support. The
probe runs on the first eligible query and is included in that first-call
observation; it is not repeated in retained timing rounds.

[`test_neighbors_integer_bounds.py`](../tests/unit/test_neighbors_integer_bounds.py)
checks the coefficient comparisons with exact rational arithmetic and reads the
production literals without importing MLX. Device tests compare candidate and
fallback outputs against the original direct Kahan scan. The tests supplement
the proof: they do not turn sampled agreement into a guarantee for untested
inputs or hardware. Speed claims remain separate from this correctness argument.

## CPU bounded-integer selection

The CPU path in [`neighbors_cpu.py`](../src/mlxlearn/_kernels/neighbors_cpu.py)
uses a separate certificate on original integral float32 coordinates. With
feature count `D` and maximum absolute coordinate `M`, it requires
`4*D*M*M <= 2**24`. Each norm and absolute dot product is therefore at most
`2**22`; all products, accumulation partials, and the squared-distance expression
`(-2*dot + train_norm) + query_norm` are exactly representable integers in
float32. Original coordinate bits are checked, so a subnormal cannot be
accepted as integer zero under flush-to-zero arithmetic.

For `k=5` only, the
[native selector](../src/mlxlearn/_kernels/_neighbors_cpu_native.cpp) now validates
each dot while computing and selecting its score, eliminating a separate pass
over the dot matrix. Norms retain their complete validation before selection.
Every dot lane is checked, including self-excluded references and blocks whose
scores cannot enter the selected five. Malformed vector lanes are masked before
arithmetic, and their failed checks are retained across all rows. The 16-wide,
4-wide, and scalar paths use the same domain restrictions.

Each query's five candidates are staged in private storage. Output buffers are
written only after every row passes both domain and nonnegative-score checks;
malformed input takes precedence over an inconsistent negative score. Late
failures therefore leave all output rows unchanged. References still enter in
ascending training-index order, with equal scores placed after earlier IDs,
preserving canonical ties at the retained boundary. General `k` selection,
coordinate certificates, dispatch, and prediction weighting are unchanged.
The [selector tests](../tests/unit/test_neighbors_integer_select.py) cover exact
ties, block tails, self exclusion, late malformed rows, error precedence, and
unchanged outputs on failure.
