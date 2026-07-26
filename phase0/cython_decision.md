# Compiled-extension go/no-go

**Decision: NO-GO for `0.1.0a1`. mlxlearn ships as a pure-Python wheel.**

Revisit when `TSNE` lands in 0.2.x, and only with fresh measurements.

## The question

The ancestor carried a Cython extension used in exactly two places: the cached-row SMO
path in `mpsbackend/svm/svm.py`, and Barnes-Hut t-SNE. The dense-kernel SVC path never
touched it, and a pure-Python fallback already existed behind
`SKMPS_SVC_SMO_USE_NATIVE_CORE=0`.

Shipping the extension would cost five ABI wheels (cp310–cp314 via cibuildwheel), a build
toolchain in CI, a source-distribution build path that needs a compiler on the user's
machine, and a class of install failures that pure-Python wheels do not have. The prior
microbenchmark suggesting the extension was worth it (≈41 ms native vs ≈61 ms pure) was
measured on the SMO inner loop in isolation, which is not what a user experiences.

## What was measured

`SVC(kernel="rbf", C=1.0).fit(X, y)` end to end on the frozen ancestor at `720053b`,
median of 3 runs, with the compiled core enabled and disabled. Same process, same data,
same machine (Apple M4 Max, 128 GB, macOS 25.2.0, Python 3.13.3).

| Problem | compiled core | pure Python | pure / compiled |
|---|---|---|---|
| 2 000 × 16 | 0.035 s | 0.044 s | **1.26×** |
| 6 000 × 16 | 0.323 s | 0.376 s | **1.16×** |
| 12 000 × 32 | 3.681 s | 3.667 s | **1.00×** |

Reproduce with `tools/measure_native_core.py`.

## Reading the numbers

**The advantage shrinks to nothing as the problem grows.** At 2 000 samples the compiled
core is worth 26%; at 12 000 it is worth zero — the two are within noise of each other.
That is the expected shape, and it is decisive: SMO's cost is split between the
bookkeeping loop (working-set selection, α updates, the gradient array) and kernel
evaluation. The extension accelerates only the bookkeeping, and kernel evaluation grows
faster. The larger the problem — which is to say, everywhere mlxlearn is supposed to
matter — the less the extension buys.

**The regime where it helps is the regime mlxlearn dispatches away.** Below the measured
crossover (`svc_min_samples = 2048`), patched mode hands the work to scikit-learn
precisely because MLX has nothing to offer at that size. Paying five ABI wheels to
accelerate a code path that patched users do not execute is a bad trade in both
directions.

**The earlier microbenchmark measured the wrong thing.** ≈41 ms vs ≈61 ms in the SMO inner
loop is a real 1.5×, and it is also 20 ms out of a fit that takes 3.7 seconds. An
isolated component measurement is not evidence about a product.

## The decision, and what it costs

`0.1.0a1` ships one `py3-none-any` wheel. CI asserts this — the `build` job fails if the
wheel name is not `-py3-none-any.whl`, so a compiled extension cannot be reintroduced
without someone deliberately changing that assertion.

The cost of the no-go is up to 26% on SVC fits of a few thousand samples. That is
acknowledged rather than hidden: `docs/benchmarks.md` reports SVC timings at those sizes,
and the crossover mechanism already routes them to scikit-learn under `patch_sklearn()`.

## What would reverse it

Any one of these, with measurements attached:

1. **t-SNE.** Barnes-Hut is the other extension consumer, and its tree traversal is not a
   bookkeeping loop that vanishes into the kernel cost — it *is* the cost. When `TSNE`
   lands in 0.2.x, measure it again. This is the likeliest reason to revisit.
2. A demonstrated ≥1.5× end-to-end gain **at or above** an algorithm's published
   crossover, not below it.
3. A Metal kernel that needs a C shim MLX cannot express. (An MLX custom Metal kernel is
   not a Python extension module and does not imply an ABI matrix.)

If it is ever reversed: build with cibuildwheel, never commit a `.so` or a generated `.c`
— `tools/compliance.py` fails on either — and keep the pure-Python fallback working and
tested, so an install without a matching wheel degrades in speed rather than failing.
