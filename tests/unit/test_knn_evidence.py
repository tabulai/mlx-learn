"""Lossless publication and fail-closed restoration, using tiny synthetic archives."""

import copy
import gzip
import hashlib
import io
import json
import tarfile

import pytest
from tools import knn_evidence as evidence


@pytest.fixture
def packaged(tmp_path):
    root = tmp_path / "source"
    results = root / "benchmarks/results"
    results.mkdir(parents=True)
    originals = {"knn-alpha.json": b'{"samples": [1,2,3]}\n',
                 "knn-beta.json": b'{"failure": true}\n'}
    for name, data in originals.items():
        (results / name).write_bytes(data)
    (results / "logistic-other.json").write_bytes(b"unrelated" * 100)
    (results / "knn-small.log").write_bytes(b"x")
    bundles = tmp_path / "bundles"
    catalog = evidence.package(root, bundles, threshold=1, bundle_bytes=1000)
    destination = tmp_path / "restored"
    destination.mkdir()
    return root, bundles, catalog, destination, originals


def test_deterministic_selection_and_lossless_idempotent_restore(packaged, tmp_path):
    root, bundles, catalog, destination, originals = packaged
    second = evidence.package(root, tmp_path / "second", threshold=1, bundle_bytes=1000)
    assert catalog == second
    assert catalog["file_count"] == 2
    assert catalog["original_bytes"] == sum(map(len, originals.values()))
    assert (bundles / catalog["bundles"][0]["name"]).read_bytes() == (
        tmp_path / "second" / catalog["bundles"][0]["name"]
    ).read_bytes()
    assert evidence.restore(catalog, destination, bundles)["restored_files"] == 2
    assert evidence.restore(catalog, destination, bundles)["already_present"] == 2
    for name, data in originals.items():
        assert (destination / "benchmarks/results" / name).read_bytes() == data
        assert (root / "benchmarks/results" / name).read_bytes() == data
    assert not (destination / "benchmarks/results/logistic-other.json").exists()
    with pytest.raises(FileExistsError):
        evidence.package(root, bundles, threshold=1)


def test_verify_and_selected_restore(packaged):
    _, bundles, catalog, destination, _ = packaged
    result = evidence.restore(catalog, destination, bundles, verify_only=True)
    assert result["restored_files"] == 0
    assert not (destination / "benchmarks").exists()
    selected = catalog["artifacts"][0]["path"]
    result = evidence.restore(catalog, destination, bundles, paths=[selected])
    assert result["restored_files"] == 1
    assert (destination / selected).exists()
    assert not (destination / catalog["artifacts"][1]["path"]).exists()


def test_differing_existing_file_is_preserved(packaged):
    _, bundles, catalog, destination, _ = packaged
    target = destination / catalog["artifacts"][0]["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user data")
    with pytest.raises(ValueError, match="overwrite differing"):
        evidence.restore(catalog, destination, bundles)
    assert target.read_bytes() == b"user data"


def test_corrupt_bundle_is_rejected_before_installing(packaged):
    _, bundles, catalog, destination, _ = packaged
    path = bundles / catalog["bundles"][0]["name"]
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evidence.restore(catalog, destination, bundles)
    assert not (destination / "benchmarks").exists()


def rewrite_archive(bundles, catalog, members):
    """Update outer digest so tests reach validation of malicious inner members."""
    bundle = catalog["bundles"][0]
    path = bundles / bundle["name"]
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for name, payload, kind in members:
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    if kind == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../../outside"
                        member.size = 0
                    archive.addfile(member, io.BytesIO(payload))
    bundle["size"] = path.stat().st_size
    bundle["sha256"] = evidence.digest(path)
    catalog["compressed_bytes"] = sum(b["size"] for b in catalog["bundles"])


@pytest.mark.parametrize("variant", ["traversal", "absolute", "duplicate", "unexpected", "symlink", "missing", "wrong_hash"])
def test_malicious_members_never_write_original_paths(packaged, variant):
    root, bundles, catalog, destination, _ = packaged
    members = [(a["path"], (root / a["path"]).read_bytes(), "file") for a in catalog["artifacts"]]
    if variant == "traversal":
        members.append(("benchmarks/results/../../escaped", b"bad", "file"))
    elif variant == "absolute":
        members.append(("/tmp/knn-escaped", b"bad", "file"))
    elif variant == "duplicate":
        members.append(members[0])
    elif variant == "unexpected":
        members.append(("benchmarks/results/knn-unexpected.json", b"bad", "file"))
    elif variant == "symlink":
        members[0] = (members[0][0], b"", "symlink")
    elif variant == "missing":
        members.pop()
    else:
        name, data, kind = members[-1]
        members[-1] = (name, b"x" * len(data), kind)
    rewrite_archive(bundles, catalog, members)
    with pytest.raises(ValueError):
        evidence.restore(catalog, destination, bundles)
    assert not (destination / "benchmarks").exists()
    assert list(destination.iterdir()) == []


def test_destination_and_bundle_symlinks_rejected(packaged, tmp_path):
    _, bundles, catalog, destination, _ = packaged
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "benchmarks").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="destination symlink"):
        evidence.restore(catalog, destination, bundles)
    assert list(outside.iterdir()) == []
    (destination / "benchmarks").unlink()
    bundle = bundles / catalog["bundles"][0]["name"]
    moved = bundles / "moved.tar.gz"
    bundle.rename(moved)
    bundle.symlink_to(moved)
    with pytest.raises(ValueError, match="bundle symlink"):
        evidence.restore(catalog, destination, bundles)


@pytest.mark.parametrize("variant", ["duplicate", "unsafe", "wrong_total", "wrong_url"])
def test_invalid_catalog_rejected(packaged, variant):
    _, _, original, _, _ = packaged
    catalog = copy.deepcopy(original)
    if variant == "duplicate":
        catalog["artifacts"].append(catalog["artifacts"][0])
    elif variant == "unsafe":
        catalog["artifacts"][0]["path"] = "../outside"
    elif variant == "wrong_total":
        catalog["original_bytes"] += 1
    else:
        catalog["bundles"][0]["url"] = "https://example.com/untrusted"
    with pytest.raises(ValueError):
        evidence.validate_catalog(catalog)


def test_cli_metadata_and_markdown_anchors(tmp_path):
    root = tmp_path / "root"
    (root / "benchmarks/results").mkdir(parents=True)
    (root / "benchmarks/results/knn-example.json").write_bytes(b"example")
    catalog_path = tmp_path / "catalog.json"
    markdown_path = tmp_path / "catalog.md"
    args = ["package", "--root", str(root), "--output-dir", str(tmp_path / "bundles"),
            "--catalog", str(catalog_path), "--markdown", str(markdown_path), "--threshold", "1"]
    evidence.main(args)
    catalog = json.loads(catalog_path.read_text())
    item = catalog["artifacts"][0]
    assert f'id="{item["anchor"]}"' in markdown_path.read_text()
    assert item["sha256"] == hashlib.sha256(b"example").hexdigest()
    with pytest.raises(FileExistsError):
        evidence.main(args)


def test_all_archives_validate_before_any_install(packaged, tmp_path):
    root, _, _, destination, _ = packaged
    bundles = tmp_path / "split"
    catalog = evidence.package(root, bundles, threshold=1, bundle_bytes=21)
    assert len(catalog["bundles"]) == 2
    last = bundles / catalog["bundles"][-1]["name"]
    last.write_bytes(b"broken")
    with pytest.raises(ValueError):
        evidence.restore(catalog, destination, bundles)
    assert not (destination / "benchmarks").exists()
