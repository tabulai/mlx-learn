# LR, KNN and SVC benchmark evidence

The full refresh results are described in the [benchmark report](../../docs/sklearn-refresh-2026-10-31.md). Results larger than 1 MiB are stored losslessly in a [release asset](https://github.com/tabulai/mlx-learn/releases/tag/sklearn-refresh-evidence-2026-10-31); smaller results remain in Git. The [catalog](sklearn-refresh-evidence-2026-10-31.json) records each original path, size and SHA-256, plus the bundle's size and SHA-256. This includes superseded initial LR runs, which are retained for audit and excluded from the report's statistics.

From the repository root, use Python 3.10 or later to download, verify and restore all original files:

```bash
python3 tools/sklearn_refresh_evidence.py restore --bundle-dir /tmp/mlxlearn-sklearn-refresh --download
```

To verify an already downloaded asset without restoring files, run:

```bash
python3 tools/sklearn_refresh_evidence.py verify --bundle-dir /tmp/mlxlearn-sklearn-refresh
```

Restoration checks the bundle and every member against the catalog before installing files. Existing matching files are retained; differing files, unsafe paths, links and unexpected archive members are rejected. Restored results are ignored by Git. The source and harness snapshots remain in `benchmarks/reference/`.
