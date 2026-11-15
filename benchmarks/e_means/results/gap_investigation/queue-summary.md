# Periodic queue synchronization ablation

The production implementation was unchanged. Each period had one full warmup and three interleaved, cache-cleared fits on resident FP32 input, seed 10, with the same BF16 Faiss recipe and online final-pass labels. N=0 passes all submissions to `async_eval`; N>0 replaces each Nth batch submission with `eval` on the identical pending arrays. Memory here is the MLX active allocation peak, including resident inputs, not process physical footprint.

| Dataset | Period N | Median seconds | Time change vs N=0 | Peak GiB | Peak change | Time range (s) |
|---|---:|---:|---:|---:|---:|---|
| spacev-1M | 0 | 1.142 | +0.0% | 1.897 | +0.0% | 1.028–1.186 |
| spacev-1M | 2 | 1.172 | +2.7% | 0.760 | -59.9% | 0.987–1.178 |
| spacev-1M | 4 | 1.204 | +5.4% | 1.079 | -43.1% | 1.119–1.247 |
| spacev-1M | 8 | 1.131 | -0.9% | 1.660 | -12.5% | 1.105–1.209 |
| vibe-agnews | 0 | 5.502 | +0.0% | 4.179 | +0.0% | 4.863–5.782 |
| vibe-agnews | 2 | 5.361 | -2.6% | 3.975 | -4.9% | 5.313–5.626 |
| vibe-agnews | 4 | 5.266 | -4.3% | 4.177 | -0.0% | 5.255–5.591 |
| vibe-agnews | 8 | 5.485 | -0.3% | 4.178 | -0.0% | 5.280–5.614 |

All 16 fits per dataset—including warmups—produced the same centroid hash and the same history hash after removing elapsed-time fields. Allocation peaks were identical across the three measured repeats for each period.

SPACEV has a clear memory tradeoff: synchronizing every two batches reduces peak active allocation by 59.9%. AGNews falls by 4.9% with that period. Median timing differences do not establish a reliable speed gain: sample ranges overlap, individual paired comparisons change direction, and runtime drifts across the run. Keep the current production scheduling for the speed-focused implementation; periodic synchronization remains an experimental memory-control option.

The result shows that pending-work scheduling materially affects current allocator peaks. It does not alone prove why the earlier fused-gather implementation increased peaks relative to the frozen implementation. The original campaign and raw ablation results remain preserved.
