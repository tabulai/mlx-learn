# Auditing the M4 Max / published H100 timing gap

This is the initial workload/hardware audit. Completed sampler controls,
telemetry cleanup, memory experiments and the final optimization results are in
the [final investigation report](../../../../docs/e-means-gap-investigation-2026-11-14.md).

This is a read-only analysis of the complete 560-record resident BF16 campaign, the attached paper, the pinned Faiss implementation, official hardware sources, and separately recorded fresh diagnostics. No benchmark or runtime code was changed by this analysis. The campaign's approximately **33–68×** VIBE time ratios are historical measurements under that campaign's conditions, not an established hardware or implementation limit.

The useful assignment work is consistent with plausible H100 BF16 throughput. However, newer unchanged-code measurements are materially faster, and the H100 epoch/work mapping is incomplete. Neither a hardware specification ratio nor a FLOP calculation identifies how much of the observed gap comes from hardware, kernel efficiency, runtime overhead, measurement overhead, or changing run conditions.

## Work actually performed

For each Metal result, the audit checks that `iteration_history[*].rows` agrees with `recipe_resolved.epoch_rows`, and that the final step count agrees with `total_steps`. It then counts **2 × processed rows × k × d** useful assignment FLOPs: one multiply and one add per point–centroid coordinate. The full tables and evidence hashes are in [workload_analysis.json](workload_analysis.json) and [workload_table.md](workload_table.md); regenerate with `python benchmarks/e_means/results/gap_investigation/analyze.py`.

This convention excludes padded matrix arithmetic, reductions, norms, direct residual calculations, statistics, updates, revival, permutations and host work. Dividing it by the **entire fit's** median time is a fit-equivalent useful rate, not a hardware instruction count or an isolated GEMM measurement. Tails are kept. For a budget above one epoch, the fractional pass precedes the final full closing pass: AG-News at T=2.2 processes `[749088, 149818, 749088]`, totaling 1,647,994 rows, rather than three full passes. It still creates three full-length permutations. Closing means and captured labels do not trigger another assignment pass. [Reference epoch policy](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/contrib/e_means/base.py#L156).

Across the eleven VIBE configurations, the historical Metal rates are **7.47–9.02 useful TFLOP/s**. Dividing the same local work by Table 11's H100 times gives **246–595 TFLOP/s**, or about **25–60%** of the SXM dense BF16 peak. These H100 rates are **conditional calculations**, not measured H100 utilization. Using the paper's rounded nominal N values instead yields roughly 250–605 TFLOP/s, also physically plausible. The local workload has not been shown to equal each original H100 workload.

Examples:

| Dataset | Useful assignment work | Historical Metal median | Fit-equivalent rate | Published H100 | Conditional H100 rate |
|---|---:|---:|---:|---:|---:|
| AG-News | 55.30 TFLOP | 6.763 s | 8.18 TFLOP/s | 0.10 s | 553 TFLOP/s |
| GloVe-200 | 12.29 TFLOP | 1.645 s | 7.47 TFLOP/s | 0.05 s | 246 TFLOP/s |
| GooAQ | 95.20 TFLOP | 10.820 s | 8.80 TFLOP/s | 0.16 s | 595 TFLOP/s |
| SimpleWiki | 26.62 TFLOP | 3.399 s | 7.83 TFLOP/s | 0.06 s | 444 TFLOP/s |

The rounded H100 times are too coarse for precise ratios: assuming ordinary rounding to 0.01 s, a displayed 0.03 s could represent 0.025–0.035 s. The JSON preserves this rounding-only sensitivity separately from the much larger protocol uncertainty. No absolute H100 e-means time is available for the six core configurations; Flash-KMeans times are not e-means times.

## Hardware evidence and limits

NVIDIA's final architecture whitepaper gives H100 SXM **989.4 dense BF16 TFLOP/s with FP32 accumulation**, versus **1,978.9 with structured sparsity**. The PCIe 80GB variant gives 756 dense TFLOP/s. Our dense clustering workload does not invoke a structured-sparse matrix path, so the doubled sparse figure is inapplicable. The product page lists SXM bandwidth as 3.35 TB/s; NVIDIA's PCIe specification lists 2.0 TB/s. [Final architecture specifications, Tables 1 and 3](https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf), [SXM product specifications](https://www.nvidia.com/en-us/data-center/h100/), [PCIe specification, Table 2](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf).

The paper states only H100 80GB. A pinned reproduction log identifies **H100 80GB HBM3**, which supports an SXM-class inference; it does not prove the exact SKU, clock, power limit or partition used for every Table 11 run. H100 NVL is a different 94GB product and should not be silently substituted. [Paper §4.1](https://arxiv.org/abs/2611.09501v1), [pinned reproduction log](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means/README.md#L87).

The local GPU is a **40-core M4 Max**, confirmed by `system_profiler`, with 128 GiB recorded by the benchmark. Apple specifies **546 GB/s** for this variant. Thus nominal bandwidth differs by about **6.14×** versus SXM, or **3.66×** versus PCIe—not 33–68×. The inspected official Apple specifications do not supply a dense BF16 GPU TFLOP/s rating. Neural Engine TOPS are a different processor/metric and cannot fill that gap. [Apple technical specifications](https://support.apple.com/en-us/121553), [Apple M4 Max announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/).

Hardware matrix execution differs: H100 has dedicated Tensor Cores. Apple's description of dedicated GPU Neural Accelerators concerns the later M5, not this M4 Max. This supports avoiding an assumption of equal BF16 matrix capability, but is not a quantitative attribution of our measured gap. [Apple's MLX/M5 matrix-accelerator description](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

## Measured arithmetic proxies and changing run conditions

The new [roofline diagnostic](roofline.json) at N=8192, k=16384 measures production fused assignment at **14.07 TFLOP/s** for d=1024 and **14.16 TFLOP/s** for d=3072. Bare BF16-output matrix multiplication reaches **14.69 and 14.84 TFLOP/s**, respectively. Those bare products omit norms/argmin/residuals and return a different precision, so they are **not valid replacement assignment implementations**. They are attained arithmetic-capacity proxies under those shapes and conditions, not official Apple peaks or proven upper bounds. Fused assignment is within about 4–5% of their time on these two shapes; this does not guarantee that every end-to-end workload is similarly close to its best implementation.

H100's 989.4 TFLOP/s theoretical peak divided by the roughly 14 TFLOP/s local attained proxy is about 71×. This mixes a specification with a measurement and is **not a predicted speedup**. It only makes a many-fold compute-throughput difference unsurprising; it does not prove that the entire observed gap is inevitable.

The fresh [unchanged-runtime stage diagnostic](stages-before.json) reports AG-News **4.126 s** and GloVe **1.100 s**, corresponding to contextual ratios of **41.3× and 22.0×**, versus the historical all-seed medians 6.763 s and 1.645 s. Runtime hashes match the campaign. The fresh diagnostic omits external memory sampling; process/controller details and run conditions differ. **The cause of this timing shift has not been isolated**, so these measurements must not replace or be pooled into the historical campaign. Controlled interleaving with and without sampling is needed before assigning causality.

In the fresh serialized profiles, assignment occupies about **92.2%** of AG-News and **89.9%** of GloVe time. The remaining stages include separately materialized gather/cast work, statistics, EMA, revival and unclassified overhead. Instrumentation serializes otherwise asynchronous execution, so these are diagnostic stage shares, not hardware-counter measurements. “Other” time is not a direct measurement of CPU permutation alone. The older DINO profile likewise attributed roughly 95.6% to assignment. It would be unsupported to blame the wide-dimension gap primarily on Python or shuffling without further evidence.

## Remaining work and protocol differences

- **No hidden approximate-assignment advantage was found.** Paper Appendix J says every point is compared with every centroid; pruning/IVF belongs to the separate large-scale experiments. VIBE uses one H100, BF16 e-means, and k=8192 or 16384. Local exact assignment has the same leading O(rows × k × d) work; no extra final reassignment is timed. The reference's default `max_points_per_centroid=None` does not silently subsample these workloads.
- **Execution organization differs.** The pinned CUDA implementation can replay full-batch update work through CUDA graphs, generates permutations on the GPU, and uses Hopper-tuned CuTe kernels. Metal constructs per-batch MLX work and creates NumPy permutations on the CPU before making them available to the device. Kernel tiling, scheduling, reductions and cache reuse differ. These are concrete candidates for controlled profiling, not measured explanations of a particular fraction of the gap. The original VIBE execution source is unavailable; graph replay in the pinned source does not prove it was used for every published timing. [CUDA replay and train loop](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/contrib/e_means/e_means_torch.py#L727).
- **Equal quality-target work is not established.** Four core epoch values are aggregate per-seed crossings, held fixed locally; Table 11's epoch row has unverified GPU association and aggregation. VIBE original holdout indices and per-seed stopping budgets are unavailable. Local seed IDs and initial samples differ from CUDA's streams. Counts, revival trajectories and reductions can therefore differ. There is no defensible per-seed equal-quality H100 speedup here.
- **Timing is close but not identical.** Local input scans and recipe preparation occur outside timing; upstream performs lighter checks inside. Both retain shuffling, state/update work, closing means and online labels, while upload/scoring are outside. Adaptive repeats are per-fit medians, not a division by the number of repetitions. The local sampler and execution controller are additional differences requiring measurement. [Reference timing protocol](https://github.com/facebookresearch/faiss/blob/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means/paper_xb.py), [local interpretation addendum](../../h100_protocol_addendum.json).

The evidence supports **plausible H100 timings, a substantial matrix-throughput advantage, and remaining implementation/measurement uncertainty**. It does not support claiming that bandwidth alone, host overhead alone, or an immutable hardware limit accounts for the reported historical ratio.
