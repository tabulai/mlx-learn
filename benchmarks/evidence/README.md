# KNN benchmark evidence

The [artifact catalog](knn-evidence-2026-10-31.md) lists immutable KNN benchmark
results larger than 1 MiB. Those files are distributed in lossless compressed
assets attached to the
[`knn-benchmark-evidence-2026-10-31` release](https://github.com/tabulai/mlx-learn/releases/tag/knn-benchmark-evidence-2026-10-31).
Small result files, benchmark drivers, tests, reports, source snapshots and native
runtime snapshots remain in Git. Original large result files remain unchanged in
the producing workspace and are ignored individually by Git.

Storage was changed for publication because several original JSON files exceed
GitHub's ordinary Git file limit. No timings, warmup audits, numerical failures,
unsuccessful candidates or earlier benchmark results were discarded. The catalog
records every archived original path, byte count and SHA256, plus the download
size and SHA256 of each bundle. All original paths can be restored; the original
execution records and their hashes still apply after restoration.

## Restore evidence

Run from the repository root using Python 3.10 or newer. The tool uses only the
standard library and does not import NumPy, scikit-learn or MLX.

Restore one result (its complete containing bundle must be downloaded and checked):

```bash
python3 tools/knn_evidence.py restore \
  --bundle-dir /tmp/mlxlearn-knn-evidence --download \
  --path benchmarks/results/knn-digits-final-2026-10-30-control.json
```

Restore every archived result:

```bash
python3 tools/knn_evidence.py restore \
  --bundle-dir /tmp/mlxlearn-knn-evidence --download --all
```

The complete original results occupy approximately 11.3 GB. Restoration stages
selected files in a temporary directory inside the checkout, verifies every
member of every selected bundle, then installs them at their original paths.
Staged files and their installed paths share storage through hard links during
installation. The temporary directory is removed when the command finishes.

If bundles were downloaded manually, omit `--download` and point `--bundle-dir`
at the directory containing the `.tar.gz` assets. Verify all bundle bytes and
member hashes without restoring results:

```bash
python3 tools/knn_evidence.py verify \
  --bundle-dir /tmp/mlxlearn-knn-evidence --all
```

Restoration is idempotent: an existing file with matching size and SHA256 is left
unchanged. A different existing file causes an error and is never overwritten.
Absolute paths, traversal, symlinks, special files, duplicate or unexpected
members, missing members and hash mismatches are rejected. No original result
path is written until every selected bundle and all of its members pass. The
catalog is trusted repository metadata; obtain it from the same reviewed commit
as the restore tool.

## Reproduce publication bundles

The original results must already exist in the checkout. Use new output paths:
the packer refuses to replace existing bundles or catalogs.

```bash
python3 tools/knn_evidence.py package \
  --output-dir /tmp/mlxlearn-knn-publication \
  --catalog benchmarks/evidence/knn-evidence-2026-10-31.json \
  --markdown benchmarks/evidence/knn-evidence-2026-10-31.md
```

Selection includes only regular `benchmarks/results/knn*` files larger than
1,048,576 bytes. Files are sorted by repository-relative path and grouped with at
most 1 GiB of original member bytes per bundle. Tar headers have fixed ownership,
permissions and timestamps; gzip has no original filename and a zero timestamp.
The same inputs and compression implementation produce the same bytes. Source
size, identity, modification time and content hashes are checked during packing.
Compression does not modify any source result.

Source and runtime snapshots in `benchmarks/reference/` serve a different purpose
and are not duplicated here. Reproducing native CPU timings requires building the
archived C++ source or restoring a matching archived native extension with the
exact Python ABI, operating system and architecture recorded by its runtime
manifest. A source-only portable fallback is not the measured native runtime.

The evidence release is a data-only prerelease targeting the KNN implementation
commit [`94ab00b`](https://github.com/tabulai/mlx-learn/commit/94ab00b), whose
63 measured package source hashes match the final benchmark source snapshot.
Merged `main` also contains upstream logistic-regression updates. Use the frozen
source, harness and runtime ZIPs in `benchmarks/reference/` to reproduce the exact
measured configuration; a later checkout of `main` can contain additional work.
