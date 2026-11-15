# e-means on Apple Silicon: benchmark protocol

`run.py` downloads the public corpora used by [the e-means paper](https://arxiv.org/abs/2611.09501),
checks their provenance, and compares the Metal implementation with full-batch
Metal Lloyd, the upstream Faiss CPU e-means implementation, and Faiss CPU Lloyd.
Raw results record actual dimensions, initialization hashes, source hashes,
timing samples, resolved recipes, memory, hardware, versions, and test MSE.

## Reproduce

Use the repository's Python environment with `faiss-cpu` and `h5py` installed.
The reference is pinned to Faiss commit
`ebab8f784c1d177129602269d5ad966c525fbdcb`. Its original MIT-licensed source is
downloaded separately, not copied into this package.

```bash
uv pip install --python .venv/bin/python faiss-cpu h5py
git clone --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss-e-means
git -C /tmp/faiss-e-means sparse-checkout set contrib/e_means benchs/e_means
git -C /tmp/faiss-e-means checkout ebab8f784c1d177129602269d5ad966c525fbdcb

.venv/bin/python benchmarks/e_means/run.py download --suite core1m --faiss-source /tmp/faiss-e-means
.venv/bin/python benchmarks/e_means/run.py run --suite core1m --faiss-source /tmp/faiss-e-means \
  --methods metal metal-lloyd --seeds 10 11 12 --epochs 3 --repeats 3
.venv/bin/python benchmarks/e_means/run.py summary
```

For the recorded campaign's exact package versions, install
`benchmarks/e_means/requirements.txt`. Every result also records Python and
package versions. Resume keys include the source hashes and versions; a changed
implementation gets a new result file. Workers snapshot provenance before
training and refuse to save if files or package versions change during the run.
The summary pairs only matching source, software, hardware, training/test
fingerprints, precision, and initial centroids.

The default data cache is `~/.cache/mlxlearn/e_means`, outside the source tree.
`--cache` overrides it. `--dry-run` on `download` prints byte ranges without
downloading. Completed ranges resume using the upstream downloader. File sizes
and SHA-256 hashes in `vibe_manifest.json` pin the VIBE files to revision
`07b387891a221b7b073b83d2f752b76462e5fa03`.

Other campaigns:

```bash
# Five ten-million-row configurations; 65,536 clusters by default.
.venv/bin/python benchmarks/e_means/run.py download --suite core10m
.venv/bin/python benchmarks/e_means/run.py run --suite core10m --methods metal metal-lloyd --seeds 10

# All eleven Appendix D.3 VIBE corpora, with new reproducible holdouts.
.venv/bin/python benchmarks/e_means/run.py download --suite vibe
.venv/bin/python benchmarks/e_means/run.py run --suite vibe --methods metal metal-lloyd --seeds 10

# Current upstream CPU reference, same initial centroids, with its recipe.
.venv/bin/python benchmarks/e_means/run.py run --suite core1m \
  --methods metal faiss-emeans --recipe faiss --seeds 10 --repeats 1

# Convergence budgets are independent fits with their own cosine schedules.
.venv/bin/python benchmarks/e_means/run.py run --datasets deep1b-1M \
  --methods metal --epochs 1 1.5 2 2.5 3 5 --seeds 10
```

Pass `--faiss-source` if the reference checkout is not `tmp/e_means/faiss`.
`--rows` explicitly requests a smaller subset; it is recorded in every result
and disables comparison with the full-data paper targets. Use `--k` to override
cluster count. `--precision float16` requests approximate assignment explicitly;
the default is float32. Neither option is silently used to reduce the benchmark.
Every configuration runs in a fresh subprocess, with a default two-hour timeout.
Existing result files are skipped; `--force` repeats them.

## Datasets and splits

| Suite/name | Dimension | Training rows | Default clusters | Evaluation |
|---|---:|---:|---:|---|
| `deep1b-1M`, `deep1b-10M` | 96 | 1M / 10M | 16,384 / 65,536 | Official next-10K split, 5K test |
| `spacev-1M`, `spacev-10M` | 100 | 1M / 10M | 16,384 / 65,536 | Official held-out indices, 10K test |
| `fb-ssnpp-1M`, `fb-ssnpp-10M` | 256 | 1M / 10M | 16,384 / 65,536 | Official held-out indices, 10K test |
| `dino10b-1M-div4096`, `dino10b-10M-div4096` | 1,024 | 1M / 10M | 16,384 / 65,536 | Official queries, 50K test |
| `imagenet-clip-512-normalized` | 512 | 1M | 16,384 | Official million-row subset, 10K test |
| `sift1b-1M-div512`, `sift1b-10M-div512` | 128 | 1M / 10M | 16,384 / 65,536 | Next-10K split, 5K test |

The upstream `paper_data.py` and `paper_splits/fingerprints.json` verify both
evaluation halves byte for byte and 1,000 evenly spaced training rows. This is
the exact released paper split for all core configurations above except SIFT1M:
the release has fingerprints for SIFT10M/100M only. SIFT1M uses the same corpus,
prefix rule, rescaling, and seeded split, and is marked unverified against a
paper fingerprint. SIFT coordinates are divided by 512 and DINO by 4096, as in
the released reproduction scripts. These exact power-of-two rescalings change
reported MSE units; raw-coordinate MSE is multiplied by the divisor squared.

The paper's Table 6 describes held-out sets generally. The release's executable
split definition is more specific: DEEP/SIFT1M/10M evaluation is the next 10K
database rows, and only half is scored. The harness follows the released code.

The `vibe` suite includes GooAQ (768D), CodeSearchNet (768D), arXiv (768D),
ImageNet-CLIP (512D), GloVe (200D), AG-News (1,024D), Landmark-DINO (768D),
Landmark-Nomic (768D), CC-News (768D), SimpleWiki (3,072D), and CelebA (2,048D).
It downloads the full public HDF5 files (30.72 GB total) and uses their `train`
corpora. These eleven Appendix D.3 holdout lists are **not supplied** in the
Faiss release. Therefore this suite constructs its own seeded 20K holdout,
removes all exact duplicate vectors from training, and scores the second 10K
half. Split indices and hashes are saved in the external cache. Cluster count
is the power of two nearest to `16 * sqrt(n)`. Results on these new splits are
clearly marked and cannot be claimed as an exact reproduction of Table 11.
The VIBE arrays are used as stored; all methods optimize squared Euclidean loss.

The paper additionally studies 100M, 1B and 10B rows, synthetic distributions,
multi-GPU sharding, and IVF approximate search. Those scales are not implied by
the `core1m`, `core10m`, or `vibe` campaigns. Explicit dataset names from the
upstream loader can be passed for its 100M configurations. Full DINO10B alone
occupies approximately 10.28 TB before working storage, and its 16,777,216 by
1,024 float32 centroids occupy 64 GiB per state array. This local benchmark is
not a replication of the paper's distributed H100 experiment.

## Timing and numerical comparisons

- Initial centroids are byte-identical across methods for each dataset and seed:
  NumPy `RandomState(seed).permutation(n)[:k]`. The paper samples them with a CUDA
  RNG, so our initialization differs from the original even when seed numbers match.
- Each timed configuration has one full untimed warm-up, then the requested
  number of complete fits. Metal is synchronized immediately before and after
  each timer. Results retain every duration and report the median.
- Timers include validation, upload from NumPy input, shuffling, training,
  finalization, and final centroid copies. Loading, initial-row sampling,
  warm-up, final label assignment, and scoring are outside the timer. This is
  an end-to-end fit measurement, broader than the paper's resident-GPU timer.
- Metal Lloyd uses the same assignment and accumulation kernels, changes
  centroids once per complete pass, retains empty centers, and forces all 25
  iterations. It never subsamples training vectors. Faiss Lloyd uses its own
  empty-center policy and also disables automatic subsampling.
- The CPU reference is the actual pinned upstream `NumpyEmeans`. Its defaults
  differ from Algorithm 1; compare it with Metal `--recipe faiss` to check
  implementation parity. The default Metal `--recipe paper` evaluates the
  published recipe. Resolved recipes are included in JSON.
- Test scoring is shared across all methods: exact Faiss `IndexFlatL2` search
  in float32, then direct residual recomputation in float64. This avoids
  cancellation in norm-expansion distance values. The training MSE is explicitly
  a deterministic sample (default 10K rows), not the full training objective.
- Paper Flash-KMeans25 test-MSE targets can be compared only when n, k and exact
  test split match. They are useful quality references with different random
  starts and precision. The harness never divides Apple timings by H100 timings.
- Fixed-budget speedup is meaningful alongside the measured MSE difference.
  A faster run with worse MSE is not a demonstrated equal-quality speedup.
  Find an equal-quality crossing by running separate epoch budgets against a
  locally measured Lloyd25 target, as in the convergence command above.

Data sources and attribution: [Faiss e-means reproduction code](https://github.com/facebookresearch/faiss/tree/ebab8f784c1d177129602269d5ad966c525fbdcb/benchs/e_means),
[BigANN data](https://big-ann-benchmarks.com/),
[VIBE](https://huggingface.co/datasets/vector-index-bench/vibe) (CC BY 4.0), and
[DINO10B](https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/) (CC BY 4.0;
Bessa et al., 2026). See the upstream dataset sources for individual corpus terms.

## Direct Faiss e-means comparison with process memory

`compare.py` runs the unmodified pinned `NumpyEmeans` against Metal using
`recipe="faiss"` for both: three epochs, float32, identical initial centroids,
the same training and evaluation arrays, and last-epoch finalization. The
reference's `faiss.knn` path uses CPU search with Apple Accelerate and OpenMP;
this is not a comparison with its CUDA implementation on an NVIDIA GPU.

```bash
.venv/bin/python benchmarks/e_means/compare.py --suite core1m --threads 8
.venv/bin/python benchmarks/e_means/compare.py --suite vibe --threads 8
.venv/bin/python benchmarks/e_means/comparison_report.py
```

Each measurement is one first fit in a fresh process. Compilation during fit
is included; data loading, initial centroid selection and common quality
scoring are excluded. `--repeats` runs additional fresh-process trials, rather
than warmed fits. The default seed is 10 and default repeats is one. Runtime
and memory therefore describe the same fit. The recorded thread calibration
selected eight OpenMP threads with Accelerate's default thread policy; the
benchmark does not change the upstream algorithm or assignment implementation.

The parent samples process physical footprint and RSS through macOS
`proc_pid_rusage` every 10 ms between explicit fit handshakes. Results retain
the prefit baseline, sampled fit peak, increase above baseline, sample count,
largest observed sampling gap, and collection failures. Peaks can miss shorter
transients. The lifetime high-water mark is diagnostic only: loading and
preparation can raise it before the fit. Physical footprint and RSS are
different accounting metrics; neither should be added to MLX's separately
reported active-allocation peak. In particular, clean file-backed input pages
may count toward RSS without being charged to physical footprint.

New results live in `results/faiss_comparison`, leaving the earlier Lloyd and
CPU-reference campaign intact. Source hashes cover the comparison harness,
memory sampler, dataset loader and implementation. The runner rejects changed
sources or software versions; the report pairs only compatible measurements.
The old cold-fit runner also accepts `--suite core10m`. For the new warmed,
paper-scale comparison, use the separate protocol below; its results are not
interchangeable with these historical cold-fit measurements.

## Five 10M configurations: matched Faiss comparison

The [large-run manifest](large_protocol.json) fixes all five official 10M
datasets at 65,536 clusters and explicitly selects the paper driver's
`quality` batch law. DEEP/SPACEV use 2 epochs, SimSearchNet++ 1.6, DINO 1.4,
and SIFT 3. The first four are Table 4 aggregate crossing budgets applied as
fixed local budgets; SIFT 3 is a local choice. Seed 10 is used throughout.

```bash
caffeinate -i .venv/bin/python benchmarks/e_means/large_compare.py
.venv/bin/python benchmarks/e_means/large_report.py --require-complete
# After every timed worker has finished:
uv run --isolated --no-project --with matplotlib --python .venv/bin/python benchmarks/e_means/large_plot.py
.venv/bin/python benchmarks/e_means/large_report.py --require-complete
```

`metal-fp32` and unmodified upstream `faiss-fp32` are the matched-precision
pair. `metal-bf16` separately measures the native BF16 Metal kernels; it is
not a precision-matched CPU comparison. All modes share initial FP32 centers,
epoch budgets, statistical batch sizes, revival constants, final-pass labels,
and last-epoch means. No extra assignment pass is added after finalization.

Every mode runs in a fresh process with resident input, one complete warm-up,
then one timed fit (median of three only if the first is under two seconds).
The timer includes new model construction, shuffling, training and finalization,
and GPU synchronization where applicable. Upload/preparation, output export
and shared quality scoring are outside. Metal releases its host input copy
before warm-up, retaining only the fixed 10,000-row quality sample. Memory is
the sampled timed-fit process footprint/RSS including resident input; MLX active
memory is recorded separately. Preparation peaks are not fit peaks.

Results and logs are saved under `results/large_comparison`; subsets supplied
with `--rows`/`--k` are diagnostic and cannot enter the full-scale report.
Sources, manifest, environment, data fingerprints and initialization are
verified before pairing. The optimized runtime is also archived in
`reference/metal-v3-source.zip`; `reference/large-comparison-source.zip`
preserves the exact shared benchmark sources and protocol. See the
[10M comparison report](../../docs/e-means-10m-comparison-2026-11-14.md).

## Resident BF16 optimization and H100 context

The separate [protocol audit](H100_PROTOCOL.md) specifies 17 configurations,
published epoch budgets where identifiable, BF16 assignment and statistics,
float32 state, last-epoch means, and online final-pass labels. Timing follows
the released H100 driver: resident inputs/initialization, one complete warm-up,
then one timed fit or the median of three if the first takes less than two
seconds. Shuffling, BF16 casting, state initialization, diagnostics and closing
means remain timed. Full validation/upload and host output copies are outside.

```bash
.venv/bin/python benchmarks/e_means/h100.py --seeds 10  # 17-configuration pilot
.venv/bin/python benchmarks/e_means/h100.py             # all seeds, both modes; resumes
uv run --isolated --no-project --with matplotlib --python .venv/bin/python benchmarks/e_means/h100_report.py --plots
```

The baseline widens BF16-rounded operands before float32 GEMM; the optimized
path uses native BF16 Metal tiles with float32 accumulation and register argmin.
Both modes use identical data, initialization, precision, epoch budget and
timing. Each dataset/seed/mode runs in a fresh process. Results include each
timed call, common float32/float64 quality scoring, external process footprint
and RSS, and MLX active-allocation peaks. The report pairs only compatible
records and records seed coverage explicitly. Plotting runs after measurement.

No H100 run is performed locally. Table 11 times provide context; VIBE's exact
GPU epoch budgets, seed IDs and holdout indices are not verified. SIFT1M and
core ImageNet T3 runs are supplemental. The four-file original Metal snapshot
is retained in `reference/metal-v1-source.zip`; old result files remain intact.
`profile.py` records synchronized per-stage costs and ordinary resident fit
times; `results/optimization` also holds kernel sweeps and validation evidence.

## Investigating the remaining H100 gap

The [gap investigation report](../../docs/e-means-gap-investigation-2026-11-14.md)
separates published hardware specifications, measured local matrix throughput,
serialized stage attribution, and fresh paired full fits. Published H100 times
remain contextual because exact per-seed work and VIBE splits are unavailable.

```bash
.venv/bin/python benchmarks/e_means/gap_bench.py
.venv/bin/python benchmarks/e_means/gap_report.py
uv run --isolated --no-project --with matplotlib --python .venv/bin/python benchmarks/e_means/gap_report.py --plots
```

The default comparison covers all 17 smaller configurations and seeds 10–12,
using `assignment="auto"` for both variants. `previous` loads verified source
bytes from `reference/metal-v2-source.zip`; `optimized` loads the current tree.
Every worker records actual imported paths/hashes. The unchanged `h100.py`
worker/controller supplies warm-up, adaptive repetitions, memory windows and
quality evaluation. Shared inputs, initialization, recipes, software and
hardware must match; only the declared runtime sources may differ. Results go
to `results/gap_protocol`, and historical results remain untouched.

`gap_profile.py` records normal resident fits and separately synchronized stage
diagnostics. The [experiment guide](experiments/README.md) documents kernel,
matrix-throughput and dispatch/copy probes; their raw evidence is retained in
`results/gap_investigation`. BF16-output or FP16 matrix products are throughput
diagnostics only and are never substituted for valid FP32 assignment scores.

## Lloyd KMeans versus e-means on five 10M datasets

The [four-method comparison](../../docs/k-means-10m-comparison-2026-11-15.md)
uses Metal KMeans, unmodified sklearn Lloyd KMeans, Metal EMeans and original
Faiss NumpyEmeans. All use FP32 data, seed-10 explicit initialization and 65,536
centroids. Training-pass caps are 2/2/2/2/3 for DEEP/SPACEV/SimSearchNet++/DINO/SIFT;
the complete timed output includes fresh final labels and inertia. This differs
from the earlier fractional-budget, online-label-only e-means comparison, so all
four methods are measured anew.

```sh
caffeinate -i .venv/bin/python benchmarks/e_means/kmeans_compare.py
.venv/bin/python benchmarks/e_means/kmeans_report.py --require-complete
# Run plotting after every timed worker has finished.
uv run --isolated --no-project --with matplotlib --python .venv/bin/python benchmarks/e_means/kmeans_plot.py
.venv/bin/python benchmarks/e_means/kmeans_report.py --require-complete
```

The frozen contract is `kmeans_protocol.json`; runtime and harness sources are
archived in `reference/kmeans-comparison-source.zip`. Raw records and CSVs live
in `results/kmeans_comparison`, with preflight and independent audit evidence in
`results/kmeans_validation`. Diagnostic subset runs are stored separately in
`results/kmeans_smoke` and `results/kmeans_scale_smoke`.

Sklearn uses `copy_x=False` to avoid an extra full input allocation. The driver
reloads exact cached input outside timing after warmup and between repeated fits,
because sklearn's subtract/restore centering can change FP32 bits. Its native
empty-cluster relocation may still allocate multiple input-sized temporaries.
Metal retains resident FP32 device input and releases the full NumPy reference.
The primary memory statistic is the sampled fit-window process physical
footprint; RSS, preparation/lifetime peaks and MLX allocator peaks are separate.

Equal pass budgets do not imply equal clustering quality: Lloyd holds centroids
fixed during each pass, while e-means updates them between mini-batches. Both
KMeans implementations can stop early on strict convergence even with `tol=0`.
The report records actual passes and scores every result with the same held-out
FP32 assignment / FP64 residual-MSE routine. See [the KMeans API guide](../../docs/k-means.md)
for supported estimator parameters and resident training methods.
