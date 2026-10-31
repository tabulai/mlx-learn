#!/usr/bin/env python3
"""Package and safely restore immutable KNN benchmark evidence using only the stdlib."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

CHUNK = 1024 * 1024
DEFAULT_TAG = "knn-benchmark-evidence-2026-10-31"
DEFAULT_CATALOG = "benchmarks/evidence/knn-evidence-2026-10-31.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            value.update(block)
    return value.hexdigest()


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def evidence_path(name: str) -> PurePosixPath:
    """Restrict every restored artifact to one canonical KNN result file."""
    if not isinstance(name, str) or "\\" in name:
        raise ValueError(f"Unsafe evidence path: {name!r}")
    path = PurePosixPath(name)
    if (len(path.parts) != 3 or path.parts[:2] != ("benchmarks", "results")
            or path.as_posix() != name or not SAFE_NAME.fullmatch(path.name)
            or not path.name.startswith(("knn-", "knn_"))):
        raise ValueError(f"Unsafe evidence path: {name!r}")
    return path


def anchor(name: str) -> str:
    return "artifact-" + hashlib.sha256(name.encode()).hexdigest()[:20]


def _size(value: object) -> bool:
    return type(value) is int and value >= 0


def _release_base(repository: str, tag: str) -> str:
    parts = repository.split("/")
    if len(parts) != 2 or not all(SAFE_NAME.fullmatch(part) for part in parts):
        raise ValueError("Invalid GitHub repository")
    if not SAFE_NAME.fullmatch(tag):
        raise ValueError("Invalid release tag")
    return f"https://github.com/{repository}/releases/download/{tag}"


def validate_catalog(catalog: dict) -> dict[str, dict]:
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported evidence catalog schema")
    base = _release_base(catalog["repository"], catalog["release_tag"])
    bundles = {}
    for bundle in catalog["bundles"]:
        name = bundle["name"]
        if (not isinstance(name, str) or not SAFE_NAME.fullmatch(name)
                or not name.endswith(".tar.gz") or name in bundles):
            raise ValueError(f"Unsafe or duplicate bundle: {name!r}")
        if (not _size(bundle["size"]) or not SHA256.fullmatch(bundle["sha256"])
                or bundle["url"] != f"{base}/{name}"):
            raise ValueError(f"Invalid bundle metadata: {name}")
        bundles[name] = bundle
    artifacts = {}
    for artifact in catalog["artifacts"]:
        name = artifact["path"]
        evidence_path(name)
        if name in artifacts:
            raise ValueError(f"Duplicate evidence path: {name}")
        if (artifact["bundle"] not in bundles or not _size(artifact["size"])
                or not SHA256.fullmatch(artifact["sha256"])
                or artifact["anchor"] != anchor(name)):
            raise ValueError(f"Invalid evidence metadata: {name}")
        artifacts[name] = artifact
    if not artifacts or not bundles:
        raise ValueError("Empty evidence catalog")
    for name, bundle in bundles.items():
        members = [a for a in artifacts.values() if a["bundle"] == name]
        if (len(members) != bundle["file_count"] or not members
                or sum(a["size"] for a in members) != bundle["uncompressed_member_bytes"]):
            raise ValueError(f"Bundle inventory mismatch: {name}")
    if (catalog["file_count"] != len(artifacts)
            or catalog["original_bytes"] != sum(a["size"] for a in artifacts.values())
            or catalog["compressed_bytes"] != sum(b["size"] for b in bundles.values())):
        raise ValueError("Catalog totals do not match inventory")
    return artifacts


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hash.update(block)
        return block


def package(root: Path, output: Path, *, repository="tabulai/mlx-learn", tag=DEFAULT_TAG,
            threshold=CHUNK, bundle_bytes=1024**3, compression_level=6) -> dict:
    """Copy source bytes into deterministic, bounded bundles; never modify originals."""
    root = root.resolve()
    base = _release_base(repository, tag)
    if threshold < 0 or bundle_bytes < 1 or not 0 <= compression_level <= 9:
        raise ValueError("Invalid packaging limit")
    artifacts = []
    for path in sorted((root / "benchmarks/results").glob("knn*")):
        if path.is_symlink():
            raise ValueError(f"Refusing source symlink: {path}")
        if not path.is_file() or path.stat().st_size <= threshold:
            continue
        relative = path.relative_to(root).as_posix()
        evidence_path(relative)
        before = _signature(path)
        sha = digest(path)
        if _signature(path) != before:
            raise ValueError(f"Source changed while hashing: {path}")
        if before[2] > bundle_bytes:
            raise ValueError(f"Source exceeds per-bundle limit: {path}")
        artifacts.append({"path": relative, "size": before[2],
                          "sha256": sha, "anchor": anchor(relative)})
    if not artifacts:
        raise ValueError("No KNN results exceed the threshold")
    groups = []
    for artifact in artifacts:
        if not groups or sum(a["size"] for a in groups[-1]) + artifact["size"] > bundle_bytes:
            groups.append([])
        groups[-1].append(artifact)
    names = [f"{tag}-{i:03d}.tar.gz" for i in range(1, len(groups) + 1)]
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError(output / name)
    bundles = []
    for name, members in zip(names, groups, strict=True):
        destination = output / name
        with destination.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                               compresslevel=compression_level) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for artifact in members:
                        source = root / artifact["path"]
                        before = _signature(source)
                        if source.is_symlink() or before[2] != artifact["size"]:
                            raise ValueError(f"Source changed before archiving: {source}")
                        info = tarfile.TarInfo(artifact["path"])
                        info.size = artifact["size"]
                        info.mode = 0o644
                        info.mtime = info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with source.open("rb") as stream:
                            reader = _HashingReader(stream)
                            archive.addfile(info, reader)
                        if reader.hash.hexdigest() != artifact["sha256"] or _signature(source) != before:
                            raise ValueError(f"Source changed while archiving: {source}")
                        artifact["bundle"] = name
        if destination.stat().st_size >= 2 * 1024**3:
            raise ValueError(f"Bundle exceeds release-asset limit: {destination}")
        bundle = {"name": name, "size": destination.stat().st_size,
                  "sha256": digest(destination), "url": f"{base}/{name}",
                  "file_count": len(members),
                  "uncompressed_member_bytes": sum(a["size"] for a in members)}
        bundles.append(bundle)
        print(json.dumps({"packaged": name, "files": len(members),
                          "original_bytes": bundle["uncompressed_member_bytes"],
                          "compressed_bytes": bundle["size"]}), flush=True)
    catalog = {"schema_version": 1, "repository": repository, "release_tag": tag,
               "selection": {"directory": "benchmarks/results", "prefix": "knn",
                             "size_strictly_greater_than": threshold},
               "format": {"tar": "ustar", "gzip_mtime": 0, "compression_level": compression_level,
                          "maximum_uncompressed_member_bytes_per_bundle": bundle_bytes},
               "file_count": len(artifacts), "original_bytes": sum(a["size"] for a in artifacts),
               "compressed_bytes": sum(b["size"] for b in bundles),
               "bundles": bundles, "artifacts": artifacts}
    validate_catalog(catalog)
    return catalog


def markdown_catalog(catalog: dict) -> str:
    lines = ["# KNN benchmark evidence — 2026-10-31", "",
             "Large immutable result files are distributed as lossless release assets. "
             "The original JSON bytes, all measurements, failed observations and audit records are preserved. "
             "This catalog changes storage only; it does not replace any measurements.", "",
             f"**{catalog['file_count']} files; {catalog['original_bytes']:,} original bytes; "
             f"{catalog['compressed_bytes']:,} compressed bytes.**", "",
             "See [restoration instructions](README.md) and the "
             "[machine-readable SHA256 catalog](knn-evidence-2026-10-31.json). "
             "Restoration recreates each original repository-relative path.", "",
             "## Release bundles", "",
             "| Bundle | Files | Download bytes | SHA256 |", "| --- | ---: | ---: | --- |"]
    for bundle in catalog["bundles"]:
        lines.append(f"| [{bundle['name']}]({bundle['url']}) | {bundle['file_count']} | "
                     f"{bundle['size']:,} | `{bundle['sha256']}` |")
    lines += ["", "## Original artifacts", "",
              "Each download contains the complete bundle shown; use the restore tool to select one original file.", "",
              "| Original artifact | Original bytes | SHA256 | Bundle |", "| --- | ---: | --- | --- |"]
    by_name = {b["name"]: b for b in catalog["bundles"]}
    for artifact in catalog["artifacts"]:
        bundle = by_name[artifact["bundle"]]
        lines.append(f"| <a id=\"{artifact['anchor']}\"></a>`{artifact['path']}` | "
                     f"{artifact['size']:,} | `{artifact['sha256']}` | "
                     f"[{bundle['name']}]({bundle['url']}) |")
    return "\n".join(lines) + "\n"


def _bundle_path(bundle: dict, directory: Path, download: bool) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / bundle["name"]
    if path.is_symlink():
        raise ValueError(f"Refusing bundle symlink: {path}")
    if not path.exists() and download:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urllib.request.urlopen(bundle["url"], timeout=60) as response:
                    copied = 0
                    for block in iter(lambda: response.read(CHUNK), b""):
                        copied += len(block)
                        if copied > bundle["size"]:
                            raise ValueError("Download exceeds declared bundle size")
                        stream.write(block)
                stream.flush()
                if temporary.stat().st_size != bundle["size"] or digest(temporary) != bundle["sha256"]:
                    raise ValueError("Downloaded bundle size/SHA256 mismatch")
                os.link(temporary, path)  # Exclusive installation: never overwrite a cache entry.
            finally:
                temporary.unlink(missing_ok=True)
    if not path.is_file() or path.stat().st_size != bundle["size"] or digest(path) != bundle["sha256"]:
        raise ValueError(f"Missing bundle or size/SHA256 mismatch: {path}")
    return path


def _target(root: Path, name: str) -> Path:
    target = root
    for part in evidence_path(name).parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Refusing destination symlink: {target}")
    return target


def _scan_bundle(path: Path, expected: dict[str, dict], stage: Path | None,
                 requested: set[str]) -> dict[str, Path]:
    seen = set()
    staged = {}
    with tarfile.open(path, "r:gz", ignore_zeros=True) as archive:
        for member in archive:
            evidence_path(member.name)
            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            if (member.name not in expected or not member.isreg() or member.linkname
                    or member.pax_headers or member.size != expected[member.name]["size"]):
                raise ValueError(f"Unexpected member/type/size: {member.name}")
            value = hashlib.sha256()
            output = None
            try:
                if member.name in requested and stage is not None:
                    temporary = stage / anchor(member.name)
                    output = temporary.open("xb")
                    staged[member.name] = temporary
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable member: {member.name}")
                with source:
                    for block in iter(lambda stream=source: stream.read(CHUNK), b""):
                        value.update(block)
                        if output is not None:
                            output.write(block)
            finally:
                if output is not None:
                    output.close()
            if value.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"Member SHA256 mismatch: {member.name}")
    if seen != set(expected):
        raise ValueError(f"Missing archive members: {sorted(set(expected) - seen)}")
    return staged


def restore(catalog: dict, root: Path, bundle_dir: Path, *, paths=None, download=False,
            verify_only=False) -> dict:
    """Validate every member of selected bundles before installing any result file."""
    artifacts = validate_catalog(catalog)
    requested = list(artifacts) if paths is None else list(paths)
    if len(set(requested)) != len(requested) or any(p not in artifacts for p in requested):
        raise ValueError("Unknown or duplicate requested evidence path")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Restoration root must already exist")
    needed = {artifacts[p]["bundle"] for p in requested}
    pending = set(requested)
    if not verify_only:
        for name in requested:
            target = _target(root, name)
            if target.exists():
                if (not target.is_file() or target.stat().st_size != artifacts[name]["size"]
                        or digest(target) != artifacts[name]["sha256"]):
                    raise ValueError(f"Refusing to overwrite differing destination: {target}")
                pending.remove(name)
    staged = {}
    # Staging beside the checkout permits exclusive hard-link installation without copying again.
    with tempfile.TemporaryDirectory(prefix=".knn-evidence-", dir=root) as temporary:
        for bundle in catalog["bundles"]:
            if bundle["name"] not in needed:
                continue
            path = _bundle_path(bundle, bundle_dir, download)
            expected = {p: a for p, a in artifacts.items() if a["bundle"] == bundle["name"]}
            staged.update(_scan_bundle(path, expected, None if verify_only else Path(temporary), pending))
        # No original path is written until every selected archive and every member has passed.
        if not verify_only:
            for name, source in staged.items():
                target = _target(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target = _target(root, name)
                try:
                    os.link(source, target)
                except FileExistsError:
                    if (not target.is_file() or target.is_symlink()
                            or target.stat().st_size != artifacts[name]["size"]
                            or digest(target) != artifacts[name]["sha256"]):
                        raise ValueError(f"Destination appeared with differing content: {target}") from None
    return {"verified_bundles": len(needed), "requested_files": len(requested),
            "restored_files": 0 if verify_only else len(staged),
            "already_present": 0 if verify_only else len(requested) - len(pending),
            "verify_only": verify_only}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package", help="Create new deterministic release bundles; refuse overwrites")
    pack.add_argument("--root", type=Path, default=Path.cwd())
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--catalog", type=Path, default=Path(DEFAULT_CATALOG))
    pack.add_argument("--markdown", type=Path, default=Path(DEFAULT_CATALOG).with_suffix(".md"))
    pack.add_argument("--repository", default="tabulai/mlx-learn")
    pack.add_argument("--tag", default=DEFAULT_TAG)
    pack.add_argument("--threshold", type=int, default=CHUNK)
    pack.add_argument("--bundle-bytes", type=int, default=1024**3)
    pack.add_argument("--compression-level", type=int, default=6)
    for verb in ("restore", "verify"):
        restore_parser = sub.add_parser(verb, help=f"{verb.title()} archived original result bytes")
        restore_parser.add_argument("--root", type=Path, default=Path.cwd())
        restore_parser.add_argument("--catalog", type=Path, default=Path(DEFAULT_CATALOG))
        restore_parser.add_argument("--bundle-dir", type=Path, required=True)
        restore_parser.add_argument("--download", action="store_true", help="Download missing bundles from catalog release URLs")
        selection = restore_parser.add_mutually_exclusive_group(required=True)
        selection.add_argument("--all", action="store_true")
        selection.add_argument("--path", action="append")
    args = parser.parse_args(argv)
    if args.command == "package":
        for destination in (args.catalog, args.markdown):
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
        catalog = package(args.root, args.output_dir, repository=args.repository, tag=args.tag,
                          threshold=args.threshold, bundle_bytes=args.bundle_bytes,
                          compression_level=args.compression_level)
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        with args.catalog.open("x") as stream:
            json.dump(catalog, stream, indent=2)
            stream.write("\n")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        with args.markdown.open("x") as stream:
            stream.write(markdown_catalog(catalog))
        print(json.dumps({key: catalog[key] for key in ("file_count", "original_bytes", "compressed_bytes")}))
    else:
        catalog = json.loads(args.catalog.read_text())
        print(json.dumps(restore(catalog, args.root, args.bundle_dir, paths=args.path,
                                 download=args.download, verify_only=args.command == "verify")))


if __name__ == "__main__":
    main()
