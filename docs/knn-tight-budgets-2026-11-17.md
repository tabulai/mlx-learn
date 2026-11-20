# Exact kNN queries at tighter workspace budgets

At 256 MiB, querying 512 points against 1,048,576 training points with 96
features and `k=10` took **39.56 ms**, down from **66.82 ms** on main.
The resident API's peak active MLX memory fell from **1,040.7 to 838.1 MiB**.
Every neighbor index and distance bit matched main across all 62 comparisons.

The M4 general GPU search can trade training columns for query rows while
keeping the same 16-byte workspace allowance per query/training pair. It keeps
the smallest screening bounds from each training tile, merges them in training
index order on ties, and refines and certifies the resulting global shortlist
once per query block. Completing each intermediate block releases its Gram
workspace before the next block runs.

The planner considers this tile at 64–512 MiB for at least 524,288 training
rows, at least 32 queries, 33–256 features and up to 32 neighbors. It selects
at most 128 query rows and at least 65,536 training columns. Explicit tile
sizes and a custom query-row alignment take precedence. The original GPU
selection rules still determine whether an index uses the general search.

There is a numerical reason to keep some older tile shapes. When MLX's sort
width limit already divides the reference data into tiles, a local tie may
trigger an exact scan. That scan can round the same neighbor's float32 distance
one bit differently from shortlist refinement. The planner checks the
original shape of both a full query block and the final partial block before
changing how the search certifies its results.

## Results

For the million-row Gaussian workload above, times are median milliseconds.
Memory columns are MiB for the resident device API. Each pair goes from main
(`6d871cc`) to the new planner (`8712ec2`).

| Budget | Device query (ms) | Speedup | NumPy query (ms) | Peak query increment (MiB) | Peak incl. resident index (MiB) |
|---:|---:|---:|---:|---:|---:|
| 64 MiB | 203.71 → 58.46 | 3.48× | 206.13 → 61.30 | 98.3 → 17.0 | 870.5 → 789.2 |
| 128 MiB | 112.57 → 43.63 | 2.58× | 111.88 → 43.85 | 134.6 → 33.4 | 906.8 → 805.6 |
| 256 MiB | 66.82 → 39.56 | 1.69× | 67.14 → 38.79 | 268.5 → 65.9 | 1,040.7 → 838.1 |
| 512 MiB | 44.31 → 35.01 | 1.27× | 44.00 → 35.06 | 536.8 → 130.9 | 1,309.0 → 903.1 |
| 1 GiB control | 35.40 → 35.78 | 0.99× | 35.35 → 35.30 | 1,049.0 → 1,049.0 | 1,821.2 → 1,821.2 |

At 256 MiB, main queried 16 rows against all 1,048,576 training rows at
once. The new planner used 128 query rows and 131,072 training columns.
Other 256 MiB resident queries illustrate how the gain depends on the workload.

| Training shape | Queries | k | Distribution | Device query (ms) | Peak query increment (MiB) |
|---|---:|---:|---|---:|---:|
| 1,048,576 × 96 | 512 | 5 | Gaussian | 66.67 → 38.72 | 792.4 → 65.4 |
| 1,048,576 × 96 | 512 | 32 | Gaussian | 68.38 → 42.84 | 269.0 → 67.1 |
| 1,048,576 × 96 | 128 | 10 | Tied integers | 29.07 → 11.90 | 393.4 → 65.8 |
| 1,048,583 × 101 | 257 | 17 | Gaussian, partial final tiles | 41.58 → 25.59 | 252.3 → 65.7 |

Across both APIs, all 34 comparisons that used a different plan ran faster,
by **1.17–3.93×**, and cut the peak query increment by **51–92%**. The other
28 comparisons exercised the original routes. Their timings varied in both
directions. At 128 MiB the 2,048-row device control took 0.94 → 1.22 ms, and
at 512 MiB the 262,144-row device control took 8.30 → 9.13 ms. The latter
workload used 262.9 MiB of additional active memory in both builds at the
128 MiB setting. The 1 GiB limit is the default on the benchmark machine.

Across all **1,116** warmup and timed queries, the two revisions returned
bit-identical distance buffers and identical neighbor indices, shapes, and
dtypes for each API. Cases included tied data, uneven final tiles, and a
2,097,153-row regression where changing the certificate would change one
float32 bit. Independent float64 calculations also checked the first three
queries of every fixture for exact indices and distances within `1e-6`
absolute and relative tolerance.

The repository test suite and kNN benchmark-harness tests passed together
with 4,387 passes and eight skips. Seven skips require pandas, and one requires
a C compiler with Python framework headers.

## Measurement

The benchmark ran on an Apple M4 Max with 128 GiB, macOS 26.6.2, Python
3.14.2, MLX 0.32.4 and NumPy 2.5.4. Each revision ran in an isolated worker
with its own frozen source and fitted index. Both workers received the same
fixture and the same minimum and maximum workspace setting. After two warmups,
seven paired rounds alternated execution order from a randomized starting
order. All queries explicitly selected GPU execution. Device timings include
evaluation, and NumPy timings include output conversion. Fitting and oracle
checks occur outside the timed interval.

Peak memory is the maximum active allocation reported by MLX over the seven
measured queries. The query increment subtracts active bytes immediately
before each query. Total active memory includes the resident index and its
prepared training buffers. These counters exclude the allocator cache and
host NumPy allocations. The configured byte budget sizes the distance tiles.

## Reproduction

From this checkout, run the following command with the repository's benchmark
dependencies installed. The raw report records output hashes, actual plans,
source revisions, timings and memory counters for both APIs.

```sh
python -m benchmarks.knn_planner_budgets \
  --baseline 6d871ccd93bb8e6278e2683e8d449084c0c33c03 \
  --candidate 8712ec24e8d31862a60fb9a5e33ffab0d62d6e1c \
  --include-64-controls --output /tmp/knn-tight-budgets.json
```

[Benchmark driver](../benchmarks/knn_planner_budgets.py) ·
[All measurements](../benchmarks/results/knn-tight-budgets-2026-11-17.json)
