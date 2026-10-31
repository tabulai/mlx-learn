# Original-coordinate kNN ordering

GPU searches now separate the precision of a reported distance from the precision
needed to choose a neighbor. A float32 squared distance may coalesce two unequal
values. Comparing their indices at that point can change membership at the kth
boundary, not just the displayed order. The new comparison in
[`_neighbors_precision.py`](../src/mlxlearn_device/_neighbors_precision.py) uses
original float32 coordinate bits whenever ordinary rounding intervals overlap.
Reported squared distances retain the compensated float32 computation.

## The inexpensive comparison

Let `T` be the real squared distance on the original float32 inputs and `K` its
guarded Kahan result. For up to 4096 features, use `u=2^-24` (unit roundoff).
In the normal range, subtraction and multiplication together perturb each
nonnegative squared term by less than `4u` relatively.

The Kahan accumulation bound can be obtained without treating every partial sum
rounding error as independent accumulated error. Write its four local rounding
errors as `e1` for adjusted-term subtraction, `e2` for addition, `e3` for the
subtraction of successive sums, and `e4` for the compensation subtraction.
Then `c_i=e2_i+e3_i+e4_i`, and the accumulated error telescopes to
`e2_n + sum(e1_i) - sum(e3_i+e4_i, i<n)`.

For nonnegative terms with sum `S` and `D<=4096`, induction gives partial
sum error at most `7*D*u*S` and `|c_i|<=4uS`. The adjusted-term errors sum
to at most `uS+4Du²S`; successive-sum subtraction errors sum to at most
`uS+(6D+1)u²S+4Du³S`; compensation-subtraction errors sum to at most
`3Du²S+7Du³S`. The final addition error is at most `2uS`. Their total is
`4uS+(13D+1)u²S+11Du³S`, below `4.004uS` for these dimensions.
Including squared-term errors gives a bound below `9uT`.
This telescoping reasoning is consistent with the compensated-summation analysis
of [Hallman and Ipsen](https://arxiv.org/abs/2107.01604); the constants here are
deliberately wider and include the original subtraction and product steps.

The implementation widens this to `12u`, including inversion and endpoint
rounding, and adds `D*2^-100` on each side for float32 flushing and underflow.
It uses these inexpensive bounds only for nonnegative finite computed distances
at most `2^110`. Overflow, nonfinite values and unsupported ranges use the exact
comparison. Dimensions over 4096 always use exact comparison. Floating-point
contraction and reassociation are disabled for the compensated arithmetic.

Disjoint intervals settle an ordering immediately. Overlapping intervals do
not constitute a tie: they request more precise comparison below. A SIMD minimum
first selects a provisional winner and checks other possible winners. Each
replacement strictly precedes the previous winner under the exact total order,
so the comparison terminates after at most the number of active lanes.

## A second precision filter

For `D<=4096` and original nonzero coordinates in `[2^-30, 2^30]` in magnitude,
TwoSum retains the residual of each subtraction. An explicit FMA recovers the
rounded square's residual, and two float32 components accumulate the squared
distance. The serial comparator uses this representation only for an ambiguous
pair. Wide-feature and large-list candidate refinement computes it once per candidate
with 32 GPU lanes, then reuses the high and low components while ranking. Large
lists first sort the high components and verify adjacent precise comparisons;
only a misordered row needs a full candidate ranking.

These are the error-free transformation primitives described by
[Ogita, Rump and Oishi](https://ogilab.w.waseda.jp/ogita/math/doc/2005_OgRuOi.pdf).
The complete accumulation is **bounded**, not assumed exact. With `u=2^-24`,
its ordering filter uses `512*D*u²*(high_a+high_b) + D*2^-90`.
The product residual's local error is of order `u²` times the squared term;
renormalization bounds the low component by a small multiple of `u` times the
partial sum. Applying these local bounds over the `D` terms and five reduction
levels is bounded by `64*(D+5)*u²` times the sum, below the filter's coefficient.
The wider comparison allowance includes final subtraction and endpoint rounding.
The coordinate guard prevents overflow; its absolute floor also exceeds the
accumulation of potentially flushed residuals. Out-of-range coordinates disable
this filter. A wide candidate's reported high component is a compensated float32
distance and satisfies the same outer `12u` interval.

Only a difference strictly outside the allowance resolves an order. An
unresolved difference, including an apparent zero, goes to exact integer
comparison. Thus this optimization cannot turn a close unequal pair into an
index tie.

## Exact comparisons

Every finite float32 number is a signed integer times `2^-149`. Decode its
mantissa and exponent without floating-point arithmetic, including subnormal
bits. To compare references `a` and `b` for query `q`, accumulate

```text
||q-a||² - ||q-b||² = sum(a² - b² - 2qa + 2qb).
```

Each term is a shifted product of at most 24-bit mantissas. Nineteen 32-bit
limbs hold the signed two's-complement difference, including every intermediate,
for any positive int32 feature count. The sign determines order; only an exact
zero permits the training-index tie breaker. If all coordinates share one
exponent and `D<=4096`, the same calculation fits signed int64 and avoids the
general limb accumulator. A nonzero-coordinate exponent span at most 30 permits
a rebased four-limb accumulator when `D<=4096`: its intermediate magnitude is
below `2^123`, within signed 128 bits. Zero, negative coordinates, mixed exponents, squared
underflow and squared overflow require no special numerical approximation.

Exact fallback insertion, precise partition merging and retained-candidate
ranking use this comparison. Large fallback lists use bounded output storage
to avoid pathological Metal compilation of large private arrays.

## Certifying that no omitted point can win

Direct scans first use the established float32 ordering and track a lower
barrier for every omitted candidate. A partition merge retains an extra point
for small k, so a close cutoff normally requires only precise shortlist sorting.
Separated output intervals certify order and membership. If only order is
uncertain, the finish kernel sorts the shortlist precisely; if membership cannot
be certified, it scans all original references. Neither step requires a host
decision or a query-result cache.

The compact and integer filters still provide a lower bound `B` on the original
Kahan result for every omitted point. The finish kernel converts it to the
corresponding real-distance lower endpoint and compares it strictly against the
selected kth point's upper endpoint. Thus a rounded tie at the cutoff cannot
hide an omitted closer point. Insufficient separation triggers a complete
original-coordinate scan with exact ambiguous comparisons.

The general GPU path now screens with a conservative Gram lower bound, retaining
`k+8` candidates and one extra lower-bound candidate to bound omissions.
For 33–256 features with k<=8, the [fused shortlist path](knn-128-feature-fix-2026-10-30.md)
retains `k+2` candidates instead. One threadgroup per query refines these
candidates, orders them with the same exact-capable comparison, and computes
the same omission certificate. Its two barriers synchronize candidate values
and the kth cutoff. A prior failed tile certificate remains false; a later
exact merge cannot erase that obligation. The final complete-search fallback
remains in place for every uncertified query.
The extra capacity is a performance choice: correctness comes from the bound
and complete fallback, not a fixed shortlist margin. Its error allowance is
`(2D+16)*2^-23` times the sum of centered squared norms, plus `D*1e-30`.
The allowance covers centered subtraction, norm/dot accumulation, and final
Gram subtraction. Screening is disabled outside a finite norm sum at most
`2^108`, or when the relative allowance reaches `1/8`. For this general-path upper
endpoint, the same Kahan induction extends to `D<2^19`: `7Du<7/32`, so
partial sums remain below `1.25S`, and the accumulated error bound is below
`4.407uS`. Including squared-term rounding still gives less than `9uT`.
The Gram screen’s `1/8` cutoff implies this dimension restriction; its absolute
floor `D*1e-30` is wider than `D*2^-100`. The direct comparison helpers retain
their more restrictive `D<=4096` fast-filter guard. Directly refined winners
must strictly precede the omitted bound after their own rounding allowance.
Uncertified rows execute a complete scan on-device; there is no host decision or
unconditional query transfer. Self identities are masked before selection.

CPU search instead evaluates bounded tiles of direct float64 differences before
selection, using MLX CPU operations and stable distance/index ordering. This
removes float32 Gram candidate loss and ranks at float64 precision; it is not an
arbitrary-precision CPU implementation. CPU results retain the native float32
distance dtype after selection. Native fit uses a bitwise owned copy, preserving
subnormal coordinates that an arithmetic copy can flush. NumPy is not a search
dependency.

The [independent tests](../tests/unit/test_neighbors_precision.py) use exhaustive
direct float64 oracles for the reported failures and a Python integer oracle
for GPU comparisons across the full finite float32 exponent range, plus close
pairs at moderate scales with 8, 32, 128 and 1024 features. Tests also
exercise kth-boundary membership, predictions, true ties, self-exclusion,
single-row tile tails, and forced general-path cancellation.
