#!/usr/bin/env python3
"""Verify or restore the release-hosted sklearn refresh evidence.

The existing evidence reader provides the hash-verified, staged restoration;
this entry point restricts it to the refresh inventory's two directories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import knn_evidence as evidence

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / 'benchmarks/evidence/sklearn-refresh-evidence-2026-10-31.json'


def refresh_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or '\\' in name:
        raise ValueError(f'Unsafe evidence path: {name!r}')
    path = PurePosixPath(name)
    prefix = ('benchmarks', 'results', 'sklearn-refresh-2026-10-31')
    directories = path.parts[3:-1]
    if (len(path.parts) not in (4, 5) or path.as_posix() != name or path.parts[:3] != prefix
            or directories not in ((), ('initial-lr-gradient-audit',))
            or not evidence.SAFE_NAME.fullmatch(path.name)):
        raise ValueError(f'Unsafe evidence path: {name!r}')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'restore'))
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    # Scope the established verifier to this release; the process exits afterward.
    evidence.evidence_path = refresh_path
    catalog = json.loads(CATALOG.read_text())
    result = evidence.restore(catalog, args.root, args.bundle_dir, download=args.download,
                              verify_only=args.action == 'verify')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
