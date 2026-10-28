# One fused line-search pass and one input pipeline for every feature count

Before this change the native binary L-BFGS step used two different forward
paths, and the estimator prepared GPU input two different ways. Fits with 512 to 4096 features ran the fused speculative kernel: one
pass over `X` produced the directional projections, every backtracking
candidate's loss change and the first candidate's sample gradient, so a
first-candidate acceptance needed no second read of the design. Fits with
fewer than 512 features ran an older pair of kernels: a line-search pass that
gave each 256-thread threadgroup eight rows (one SIMD group per row, one float
per lane) and wrote a `[candidates, n/8]` partial array that `mx.sum` read
back, followed by an unconditional backward pass over `X` for the gradient.
On 500,000 × 32 the line-search pass streamed the design at about 58 GB/s
while the backward pass streamed the same bytes at about 194 GB/s, and every
accepted iteration read `X` twice.

## What changed

- `native_geometry.speculative_geometry(n, d)` is the single launch-geometry
  table. The MLX solver step, the native command loop and the
  `mlxlearn_device` bridge all use it; its tuples for 512 to 4096 features
  are byte-identical to the three tables it replaces, so the compiled native
  command library does not change.
- `native_logistic.step` always calls `forward_gradient`; the `512 <= d`
  branch, the narrow line-search kernel (`_forward_search`) and the
  unconditional backward pass are gone. Between 129 and 511 features the
  wide kernel runs with one feature group and eight SIMD groups per
  threadgroup (32 rows per SIMD group on large designs, four on small ones).
- Designs of at most 128 features use two further kernels with the same
  contract. In both, a SIMD group carries four rows per batch and every lane
  evaluates one line-search candidate of its row, so all 32 lanes work on
  the transcendental part instead of eight. Up to 32 features (`_cluster`),
  eight lanes share a row outright (lane `8r+q` holds quad `q` of row `r`)
  and the dot product closes with three shuffles inside the eight-lane
  cluster; between 33 and 128 features (`_rows`) every lane holds one quad
  of each of the four rows and the dot products close with `simd_sum`.
  Above 128 features the register copy of four rows stops paying for itself
  (at 511 features it is slower than the one-row layout) while the wide
  kernel already streams at the memory ceiling, so the wide kernel serves
  129 to 4096 features.
- A converged or failed status skips the pass in every kernel; the narrow
  kernels also return early after zeroing their outputs, so the eighth no-op
  step of a seven-iteration fit no longer streams the design.
- The `mlxlearn_device` bridge loads the geometry module by path, as it does
  the input kernels, so it still never imports `mlxlearn`. Its one-pass
  Hessian-vector kernel keeps a separate table that coincides with this one
  only from 512 features up.
- `minimize_binary` now states its width contract up front (1 to 4096
  features, the range `_use_native_solver` has always enforced for the
  estimator) instead of relying on the old kernel accepting any width.

## One input pipeline

The estimator used to prepare GPU input two ways. Intercept fits of 512 to
4,096 features with at least four million elements went through
`prepare_centered`: a statistics pass computes the per-column mean, and one
fused pass writes the owned centered copy, checks every original value for
finiteness and accumulates the binary solver's initial gradient and
curvature partials. Every other GPU fit went through an older three-stage
pipeline: `copy_and_check` (one read, one write), `_center_gpu` (three more
passes and two full-size temporaries) and a separate `_initial` pass, so a
500,000 × 32 design was read or written eight times before the first
iteration. The two computed the same quantities with the same kernels for
the copy and the partials; the difference was a route gate and the
reduction order of the mean.

Now every GPU fit goes through `prepare_centered`:

- `prepare_centered(X, y, sw, center=..., intercept=...)` serves fits
  without an intercept (the copy holds the original values, the returned
  center is `None`, the partials have no intercept column) and designs of
  any size. Finiteness moved from the statistics pass into the copy kernels,
  so an uncentered copy is checked the same way a centered one is.
- The estimator's `_prepare_centered_logistic_input` context manager owns
  the borrowed input for the whole fit and drains the device before
  releasing it; the former `_upload_logistic_input`, which dropped its
  NumPy-backed wrapper the moment the copy returned, is gone together with
  `copy_and_check`, `_center_gpu`, the `_initial` kernel and the
  four-million-element threshold. `solve_logistic`'s own fallback for
  callers that pass no device input, and `native_logistic.initial` for
  callers that pass no partials, use the same kernel.
- Fits without an intercept keep their previous stopping threshold: the
  scale-skew check still reads the curvature diagonal only for centered
  designs (a test pins this). Intercept fits of 512 to 4,096 features
  below four million elements, which only `force_backend="mlx"` reaches,
  now carry prepared statistics like larger ones and therefore take the
  native command loop, the scale-skew threshold and, when requested, the
  Hessian sketch, where they previously ran the MLX loop at the plain
  tolerance.
- What is and is not bitwise identical to the deleted route: the owned
  copy of an uncentered design and the partials computed from any given
  copy are (same kernels, same accumulation order; a review probe
  confirmed both at widths 1 to 1,024 and 1 to 4,100 rows, and that
  no-intercept fits reproduce the previous coefficients bit for bit). The
  per-column mean is not: `_center_gpu` reduced it with a compiled `mx.sum`
  and the statistics kernel reduces it in 256-row blocks, and on ordinary
  data the two disagree by one float32 ulp in a fraction of the columns
  (both within about 1e-5 of the float64 mean). Intercept fits of fewer
  than 512 features or four million elements therefore adopt the former
  wide route's mean and can move at float32 rounding level: on 2,000 × 32
  to 300,000 × 32 probes the coefficients changed by at most 1e-6 and the
  intercepts by at most 1e-5, with identical iteration counts and
  deterministic reruns. Fits that already used the fused pass (512 to
  4,096 features, four million elements or more) are bitwise unchanged.
- One more route detail: a multinomial fit on float64 input used to run no
  Metal kernel at all (its private float32 conversion was wrapped zero-copy
  and centered by compiled MLX primitives); it now compiles the same
  shape-specialized copy kernels every other route already compiled.
- The rounding-level change of the mean surfaced two things in the generic
  L-BFGS solvers that serve multinomial, CPU and generic weighted fits. On
  the continuous-integration GPU a six-class parity fixture that used to
  run to its 100-iteration budget stopped at iteration 91 with a failed
  line search. Measured on the M4 Max, that fixture's float32 objective
  reaches its resolution floor after about 70 iterations (the remaining
  decreases are fractions of a float32 ulp), so whether a late search still
  finds a decrease is rounding luck that differs between GPUs; a stop
  there is the documented "line-search failure as a stop reason" case, and
  the parity test's precondition now asks that neither solver reached the
  tolerance instead of demanding exactly 100 iterations. Separately, both
  solvers now clear a quasi-Newton history and retry once along the
  gradient before reporting `line-search-failure` (on the CI GPU that
  carried the fixture from iteration 91 to 94); a run that never fails a
  search is unchanged, and a plateau still ends with the same stop reason.

Input preparation at 500,000 × 32 drops from about 1.98 ms to 1.12 ms per
fit and at 500,000 × 128 from 6.96 ms to 3.98 ms (fresh host buffers,
measured on the previous kernels); the solver's work is unchanged. One
memory detail changes: a float64 input is converted to a private float32
copy, and a fit without an intercept used to run the solver directly on a
zero-copy view of that private array. It now gets the same owned device
copy as every other input, so its peak memory during `fit` grows by one
float32 design (intercept fits already held a centered copy before).

## Per-pass cost of one line-search step

Apple M4 Max, MLX 0.32.2, 500,000 rows, device-resident float32 design,
eight candidates, medians of five interleaved rounds of eight launches,
`mx.eval` brackets. "Previous" is the kernel pair the width used before
(`_forward_search` + `mx.sum` + unconditional `bwd` below 512 features);
"now" is the kernel the width uses after this change.

| Features | Previous, ms | Now, ms | Now, GB/s | Kernel now |
| ---: | ---: | ---: | ---: | --- |
| 20 | 1.25 | 0.32 | 126 | `_cluster` |
| 32 | 1.25 | 0.31 | 209 | `_cluster` |
| 48 | 1.3 | 0.43 | 223 | `_rows` |
| 64 | 1.3 | 0.46 | 281 | `_rows` |
| 96 | 1.5 | 0.57 | 339 | `_rows` |
| 128 | 1.67 | 0.74 | 346 | `_rows` |
| 300 | – | 1.42 | 424 | `_speculative`, one feature group |
| 511 | – | 2.24 | 457 | `_speculative`, one feature group |

The previous per-iteration figures combine the line-search pass, its host
reduction and the backward pass (1.25 ms at 32 features, 1.67 ms at 128);
the two widths with dashes were not timed on the previous kernels. The
four-rows layout was also tried at 160 to 511 features: it ties the one-row
layout up to 256 features and loses above (2.57 ms against 2.24 ms at 511),
which sets the 128-feature switch.

## Complete fresh-fit timings

Same harness, settings and data as the [earlier recheck](tall-logistic-recheck-2026-10-27.md):
Apple M4 Max, MLX 0.32.2, scikit-learn 1.9.0, float32 input, seed 41,
`C=1`, `tol=1e-4`, `max_iter=200`, with an intercept; one warmup and seven
measured complete fresh fits per implementation in the original rotating
order, no thread limit; medians in milliseconds.

| Dataset | Earlier recheck | Kernels only | Kernels and input pipeline | Stock sklearn, rerun | sklearn / now | Fused control, rerun |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500,000 × 32 | 13.864 | 6.604 | **6.302** | 74.591 | **11.8×** | 82.943 |
| 500,000 × 128 | 23.957 | 14.849 | **13.202** | 76.874 | **5.8×** | 86.168 |
| 60,000 × 1,024 | 13.191 | 12.872 | 13.009 | 35.525 | 2.7× | 28.889 |

The third dataset uses neither the new kernels nor a new input route (1,024
features already took the fused centering pass, the wide kernel and the
native command loop); its timing moved within run-to-run variability. Only
the "kernels and input pipeline" column is backed by a recorded results
file; the "kernels only" column is the same harness run earlier in the
session, before the input-pipeline change, and is quoted from that session
(its results file was superseded). Harness medians move by up to about
0.5 ms between runs of the same tree in one session, and sklearn's own
medians by a few milliseconds (earlier runs of the final tree gave 6.07 and
6.53 ms against 72.7 and 71.6 ms for sklearn); the recorded run is the one
quoted. The
earlier recheck's native times are the published figures, not a
re-execution in this session. An interleaved same-session comparison of the
previous `main` against the final tree, also unrecorded, gave
13.80 → 6.25 ms, 22.22 → 12.69 ms and, for 250,000 × 32, 8.08 → 4.87 ms
(nine fits each).

All native fits used seven accepted iterations, as before. Every measured
fit passed the independent float64 audit: relative objective gaps to the
tight reference of 2.15e-7, 2.40e-7 and 1.96e-6 (identical to the earlier
recheck), maximum float64 gradients 5.16e-5, 5.37e-5 and 3.94e-5, prediction
agreement with the tight reference of at least 99.998 %, 99.998 % and
99.968 %. Source hashes matched before and after the run, and the recorded
hashes are those of the shipped solver sources.

```bash
python -m benchmarks.tall_logistic_recheck --repeats 7 --warmup 1 \
  --output benchmarks/results/tall-logistic-recheck-fused-narrow-2026-10-27.json
```

The [full results](../benchmarks/results/tall-logistic-recheck-fused-narrow-2026-10-27.json)
include every timed sample, fit state, warning, independent score, fixture
hash, environment and source hash.

## Tests

The full suite passes 1,538 tests with both changes (361 of them new or
replaced). For the input pipeline: both copy kernels preserve every bit
(including a negative zero) at tile boundaries from 1 to 2,049 elements;
the last element of an incomplete tile trips the non-finite check in every
center/labels mode; the partials match a float64 reference with and
without centering at widths 1 to 1,024; the solver-side fallback and the
prepared route produce bitwise-identical partials from one kernel; the
estimator's no-intercept route yields an uncentered copy with no intercept
column; the scale-skew threshold receives statistics only for centered
designs; and the centering tests for extreme finite constants and large
offsets now run against `prepare_centered`.
`tests/unit/test_native_logistic_narrow.py`:

- the geometry table pins the wide tuples and the narrow rules;
- the fused pass is checked against a float64 reference for projections,
  all candidate loss changes and the first-candidate gradient at widths
  {1, 7, 20, 31, 32, 33, 37, 128, 300, 511} × row counts {67, 259, 16385,
  16391} × intercept × {8 candidates at unit step, 24 candidates in a resumed
  batch}, with zero and geometrically spread sample weights;
- a converged status leaves the state bit-identical through a full step;
- the narrow analogue of the accepted-candidate contract: the speculative
  gradient is used only when the first candidate is accepted, otherwise the
  backward pass must match a fresh native evaluation exactly;
- end-to-end fits at widths {1, 20, 300}, weighted and unweighted, with and
  without an intercept, certified by a float64 gradient at the requested
  tolerance; a rank-one width-300 design that exhausts the first eight
  candidates and resumes with 24; an overflowing design that fails honestly.

One portability detail surfaced during the work: MLX binds a one-element
input in the `constant` address space, where a `packed_float4` pointer cast
does not compile. The narrow kernels therefore load their quads as four
scalar reads, which the compiler vectorizes; the per-pass cost is the same.
