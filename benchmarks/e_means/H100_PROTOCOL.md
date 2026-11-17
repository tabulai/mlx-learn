# H100 protocol audit and Metal rerun contract

[h100_protocol.json](h100_protocol.json) records the 17 smaller configurations,
sources, precision semantics, timing boundaries, selected seeds, and unresolved
differences. It does not claim an exact reproduction of every published H100
experiment. The five 10M configurations are outside this rerun's scope.

The [machine-readable interpretation addendum](h100_protocol_addendum.json)
applies to the frozen manifest SHA-256
`90c885b91fe5fba59590ebe5872d9d751c80dd2e6e9b732c8907df100da91b28`.
It clarifies that the local timer includes prepared-input signature checks and
state initialization, while full input/initial-centroid validation and recipe
preparation are untimed. It also distinguishes fixed published aggregate budgets
from unreleased per-seed H100 stopping epochs. No configuration values, measured
sources, or measurements change; the frozen manifest remains unchanged.

## Confirmed precision and finalization

The reference is Faiss commit
[`ebab8f784c1d`](https://github.com/facebookresearch/faiss/tree/ebab8f784c1d177129602269d5ad966c525fbdcb/contrib/e_means).
Its [H100 driver](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means/paper_xb.py#L173)
uses GPU-resident FP32 input with `CastQuantizer(d, torch.bfloat16)`, without
centering. Rows are rounded to BF16 per batch, then widened for **FP32 batch
statistics**. This is more than BF16 matrix multiplication on otherwise original
FP32 data. Optimizer state, explicitly supplied initial centroids, and output
centroids remain FP32. Assignment uses BF16 row/centroid operands, FP32
intermediates, and FP32 norms computed from the **rounded** centroid mirror.
The alternative preconverted-BF16 input route is distinct and should be labeled
if benchmarked; this contract selects the resident-FP32 cast route.

The reference [finalizer](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/contrib/e_means/e_means_torch.py#L656)
returns unweighted means of final-pass assignments, retaining the optimizer
position for unassigned clusters. There is no new assignment pass. Paper
Section 3.2, page 4, explicitly makes this the default throughout experiments;
the [author's description](https://e-means.github.io/) also specifies last-epoch
means. Figure 14's “e-means (EMA)” legend describes the algorithm and does not
establish a raw-EMA output. Use **`finalization="last_epoch"`**.

Upstream also repairs splits that collapse to identical BF16 operands by moving
the child's first coordinate to a distinct BF16 neighbor. This representation
rule, as well as rounding of statistics, matters for fidelity. GPU reductions,
random streams, and numerical behavior can still differ across implementations.

## Timing and output workload

The released [timing function](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means/paper_xb.py#L140)
defines these boundaries:

1. In a fresh process, load and upload input and initial centroids before timing.
2. Make one complete, identical, untimed warm-up fit.
3. Before each timed fit, discard prior state, collect garbage, clear the
   allocator cache, and synchronize while retaining input and initial centroids.
4. Time construction of a new model and its complete fit, followed by device
   synchronization. **Epoch shuffling remains inside the timed fit.** Include
   batch casts, updates, revival, finalization, online objectives/history, and
   capture of labels from the final pass. The upstream labels are int64; there
   is no extra final nearest-centroid pass.
5. If the first timed fit takes less than 2 seconds, perform two more and report
   the median of all three. Otherwise report that one fit. Copy and score the
   **first timed fit's** centroids after its timer stops.

Do not include upload, initial-row sampling, warm-up, cache clearing before the
call, host centroid copy, or quality scoring in the fit timer. Preserve every
call sample and per-seed result. This is warmed fitting on resident data, not
cold-start or end-to-end latency.

Upstream accumulates online objective and pass counts in FP64 and reads epoch
totals to the host within `end_epoch()`, even with verbose output disabled.
Deferring those reads is an implementation optimization, not identical
synchronization behavior. If Metal uses FP32 history accumulation because its
GPU lacks FP64, disclose that difference; do not omit the online work. Its
history does not replace the shared post-fit FP64 quality evaluation.

## Epoch budgets and seeds

The attached paper calls the core crossing table **Table 4, page 8**. The pinned
repository calls this “Table 3.” Four 1M budgets are published aggregate H100
crossing budgets; the other two core configurations use a supplemental
three-epoch budget. **This rerun applies each table-level budget to every local
seed. It does not reproduce the reference's per-seed stopping search.**

| Configuration | Epochs | Seed IDs | Evidence status |
|---|---:|---|---|
| DEEP 1M | 2.0 | 10–19 | Table 4 aggregate GPU crossing |
| SPACEV 1M | 2.4 | 10–19 | Table 4 aggregate GPU crossing |
| SimSearchNet++ 1M | 1.4 | 10–19 | Table 4 aggregate GPU crossing |
| DINO10B 1M | 2.0 | 10–19 | Table 4 aggregate GPU crossing |
| Core ImageNet-CLIP 1M | 3.0 | 10–19 | Supplemental; H100 crossing/time unavailable |
| SIFT 1M | 3.0 | 10–19 | Supplemental; H100 crossing/time unavailable |
| VIBE AG-News | 2.2 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE arXiv | 2.4 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE CC-News | 2.0 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE CelebA | 1.8 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE CodeSearchNet | 2.2 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE GloVe-200 | 1.6 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE GooAQ | 2.6 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE ImageNet-CLIP | 1.8 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE Landmark-DINO | 1.8 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE Landmark-Nomic | 2.2 | 10–29 | Table 11 listed budget; GPU association unverified |
| VIBE SimpleWiki | 2.2 | 10–29 | Table 11 listed budget; GPU association unverified |

The reference [crossing search and aggregation](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means/paper_bench.py#L113)
evaluate each seed independently against that seed's Flash-KMeans@25 MSE, in
0.2-epoch increments. The table reports the upper median crossing budget and,
separately, the median of per-seed speedups. The individual crossing budgets are
not shipped in `paper_targets.json`. Its per-seed Flash target errors do not
reveal how many epochs the H100 e-means fit needed. Thus a verified table budget
is not a verified H100 budget for every seed. The README's statement that the
two precision routes cross at the same budget on every seed concerns agreement
between routes within a seed, not a common budget across all seeds.

Table 11, page 23, reports one e-means epoch row without labeling its backend
or specifying how that epoch row is aggregated. It supplies dataset-level
budgets, not individual seed budgets. These are likewise fixed local budgets
in this rerun; no per-seed H100 stopping sequence is claimed.
It reports 20 seeds but not their IDs: **10–29 is a chosen local seed set, not
verified paper IDs**. The author's blog instead shows different VIBE budgets
over five seeds; those are not substituted for the paper's twenty-seed timings.
The JSON preserves both sources explicitly. The released core reproduction
code confirms its seed IDs as 10–19. All selected configurations use the
throughput batch rule under that driver's `n <= 2,000,000` policy; the separate
VIBE benchmark implementation was not found.

## Limits on matching the H100 results

No NVIDIA H100/CUDA device is available locally. CUDA `torch.randperm` selects
the reference initial rows and epoch orders; using the same integer seed with
NumPy or MLX does not reproduce these streams. Record the actual local
initialization hash, shuffling policy, and donor RNG policy. Matching workload
and representation does not imply bitwise equality.

The VIBE sources are the same public corpora, but the original holdout indices
and per-seed targets were not released. Our deterministic duplicate-disjoint
holdouts and resulting exact training sizes differ. The released warm-up/call
protocol is verified for the core reproduction code; no VIBE GPU script or raw
per-seed timing/crossing file was found in the pinned or current Faiss benchmark
directory or the author's site. Table 11's H100 times are rounded to 0.01 seconds.

`paper_targets.json` supplies Flash-KMeans/Lloyd baseline evidence, **not H100
e-means timings**. Ratios against published times must retain these limitations;
they are descriptive comparisons across hardware and partially different
experimental conditions, not proof of a completely matched VIBE reproduction.
