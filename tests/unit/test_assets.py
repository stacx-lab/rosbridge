"""Cache integrity, concurrency, source-independent reuse, and ABC mounting."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
import zipfile

import pytest

from rosbridge.assets import AssetStore, asset_root, link_asset
from tests.asset_helpers import package, ensure


def test_asset_root_override(monkeypatch, tmp_path):
    monkeypatch.delenv("ROSBRIDGE_ASSET_ROOT", raising=False)
    monkeypatch.setattr("rosbridge.assets.cache.project_root", lambda: tmp_path)
    assert asset_root() == tmp_path / ".runtime/assets"
    monkeypatch.setenv("ROSBRIDGE_ASSET_ROOT", str(tmp_path / "disk"))
    assert asset_root() == tmp_path / "disk"


def test_checksum_failure_never_publishes_and_can_retry(tmp_path):
    pkg = package(tmp_path)
    store = AssetStore(tmp_path / "cache")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure(store, {**pkg, "sha256": "0" * 64})
    assert not list(store.root.rglob("complete.json"))
    assert (ensure(store, pkg) / "robot/mesh.obj").read_bytes() == b"mesh bytes"


def test_concurrent_requests_download_once(tmp_path, monkeypatch):
    pkg = package(tmp_path)
    store = AssetStore(tmp_path / "cache")
    original = urllib.request.urlopen
    calls = []

    def download(*args, **kwargs):
        calls.append(args[0].full_url)
        return original(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: ensure(store, pkg), range(4)))
    assert len(calls) == 1 and len(set(paths)) == 1


def test_incomplete_cache_is_repaired(tmp_path):
    pkg = package(tmp_path)
    store = AssetStore(tmp_path / "cache")
    data = ensure(store, pkg)
    (data / "robot/mesh.obj").unlink()
    assert (ensure(store, pkg) / "robot/mesh.obj").read_bytes() == b"mesh bytes"


def test_versions_coexist(tmp_path):
    store = AssetStore(tmp_path / "cache")
    first = ensure(store, package(tmp_path))
    second = ensure(store, package(tmp_path, contents=b"new mesh"))
    assert first != second
    assert (first / "robot/mesh.obj").read_bytes() == b"mesh bytes"
    assert (second / "robot/mesh.obj").read_bytes() == b"new mesh"


def test_links_replace_links_but_never_local_contents(tmp_path):
    store = AssetStore(tmp_path / "cache")
    source = ensure(store, package(tmp_path)) / "robot"
    linked = tmp_path / "runtime/assets/robot"
    link_asset(source, linked)
    link_asset(source, linked)
    assert linked.is_symlink() and linked.resolve() == source
    local = tmp_path / "runtime/assets/local"
    shutil.copytree(source, local)
    with pytest.raises(ValueError, match="occupied"):
        link_asset(source, local)
    assert not local.is_symlink()


def test_unpinned_zip_records_observed_hash(tmp_path):
    archive = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("objects/mesh.obj", b"mesh")
    store = AssetStore(tmp_path / "cache")
    data = store.ensure("abc", "external_objects", archive.as_uri())
    marker = json.loads((data.parent / "complete.json").read_text())
    assert not marker["verified"]
    assert marker["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (data / "objects/mesh.obj").read_bytes() == b"mesh"


@pytest.mark.parametrize("archive_type", ["tar", "zip"])
def test_archive_cannot_escape_cache(tmp_path, archive_type):
    archive = tmp_path / "bad.archive"
    if archive_type == "tar":
        with tarfile.open(archive, "w") as bundle:
            info = tarfile.TarInfo("../../escape")
            info.size = 1
            bundle.addfile(info, io.BytesIO(b"x"))
    else:
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../../escape", b"x")
    store = AssetStore(tmp_path / "cache")
    with pytest.raises((tarfile.FilterError, ValueError)):
        store.ensure("abc", "bad", archive.as_uri())
    assert not list(store.root.rglob("complete.json"))
    assert not list(store.root.rglob("escape"))


def test_adoption_moves_a_verified_seed_without_copying(tmp_path):
    seed = tmp_path / "seed/pack"
    (seed / "nested").mkdir(parents=True)
    (seed / "nested/a.bin").write_bytes(b"abc")
    inode = (seed / "nested/a.bin").stat().st_ino
    store = AssetStore(tmp_path / "store")

    def reject(path):
        raise ValueError("checksum mismatch")
    with pytest.raises(ValueError, match="mismatch"):
        store.adopt("sim", "pack", "rev1", seed, verify=reject)
    assert (seed / "nested/a.bin").is_file() and store.cached("sim", "pack", "rev1") is None
    checked = []
    data = store.adopt("sim", "pack", "rev1", seed, verify=checked.append, subdirectory="pack")
    assert checked == [seed] and not seed.exists()
    assert (data / "pack/nested/a.bin").stat().st_ino == inode
    assert store.cached("sim", "pack", "rev1") == data
    # A complete package is reused; the seed is no longer needed.
    assert store.adopt("sim", "pack", "rev1", seed, verify=reject) == data
    with pytest.raises(ValueError, match="not a directory"):
        store.adopt("sim", "other", "rev1", seed, verify=checked.append)
