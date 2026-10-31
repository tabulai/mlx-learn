"""Freeze the exact benchmark source inventory without importing MLX.

Archive entries are sorted, uncompressed and given fixed timestamps/modes, so
identical source bytes produce an identical ZIP independent of file mtimes or
compression-library versions. Existing evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import zipfile
from pathlib import Path

from .knn_pipeline import ROOT, sha256, source_hashes


def verify_source(archive, source_root):
    archive, source_root = Path(archive), Path(source_root)
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    data = archive.read_bytes()
    if sha256(data) != manifest["archive"]["sha256"]:
        raise ValueError("Archive differs from its manifest")
    if source_hashes(source_root) != manifest["files"]:
        raise ValueError("Current source differs from its frozen manifest")
    with zipfile.ZipFile(io.BytesIO(data)) as saved:
        names = saved.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("Archive inventory differs from its manifest")
        for name in names:
            if sha256(saved.read(name)) != manifest["files"][name]:
                raise ValueError(f"Archive member differs from its manifest: {name}")
    return manifest


def freeze_source(source_root, archive, *, source_state, revision):
    source_root, archive = Path(source_root).resolve(), Path(archive).resolve()
    if archive.suffix != ".zip":
        raise ValueError("Source archive must have a .zip suffix")
    manifest_path = archive.with_suffix(".manifest.json")
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to replace a source archive or manifest; choose a new name")
    before = source_hashes(source_root)
    if not before:
        raise ValueError("Source inventory is empty")
    captured = {name: (source_root / name).read_bytes() for name in sorted(before)}
    if {name: sha256(data) for name, data in captured.items()} != before:
        raise RuntimeError("Source changed while its bytes were being captured")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as saved:
        for name, data in captured.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_STORED
            saved.writestr(entry, data)
    if source_hashes(source_root) != before:
        raise RuntimeError("Source changed while its archive was being assembled")
    data = buffer.getvalue()
    stored_path = str(archive.relative_to(ROOT) if archive.is_relative_to(ROOT) else archive)
    manifest = {
        "revision": revision, "source_state": source_state,
        "freeze_format": "sorted ZIP_STORED; fixed 1980 timestamp; regular files mode 0644",
        "files": before, "archive": {"path": stored_path, "sha256": sha256(data)},
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with archive.open("xb") as output:
            created.append(archive)
            output.write(data)
        with manifest_path.open("x") as output:
            created.append(manifest_path)
            output.write(json.dumps(manifest, indent=2) + "\n")
        verify_source(archive, source_root)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "src")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-state", help="Concrete description of the working-tree implementation")
    parser.add_argument("--verify", action="store_true", help="Verify existing archive, manifest and current source")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_source(args.output, args.source_root)
    else:
        if not args.source_state:
            parser.error("--source-state is required when freezing source")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = freeze_source(args.source_root, args.output,
                                 source_state=args.source_state, revision=revision)
    print(json.dumps({"archive": manifest["archive"], "file_count": len(manifest["files"]),
                      "verified_current_source": True}, sort_keys=True))


if __name__ == "__main__":
    main()
