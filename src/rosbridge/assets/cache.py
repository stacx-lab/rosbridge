"""Shared asset cache. Standard library only; no ROS or simulator dependencies."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import uuid
import zipfile

from rosbridge.project import project_root


def seed_root():
    """An existing, separately verified asset depot to adopt from instead of downloading."""
    seed = os.environ.get("ROSBRIDGE_ASSET_SEED")
    return Path(seed).expanduser().resolve() if seed else None


def asset_root():
    override = os.environ.get("ROSBRIDGE_ASSET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / ".runtime/assets"


@contextmanager
def file_lock(path, *, shared=False, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(handle, flags | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe asset path: {name}")
    return path


def _inventory(root):
    """File sizes detect incomplete caches cheaply, without rehashing large packs."""
    return {str(p.relative_to(root)): p.stat().st_size
            for p in root.rglob("*") if p.is_file()}


def _component(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError(f"invalid asset identifier: {value}")
    return value


class AssetStore:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root is not None else asset_root()

    def cached(self, simulator, name, key):
        """A complete package's data directory, or None."""
        destination = self.root / _component(simulator) / _component(name) / _component(key)
        try:
            saved = json.loads((destination / "complete.json").read_text())
            if saved["key"] == key and saved["files"] == _inventory(destination / "data"):
                return destination / "data"
        except (OSError, ValueError, KeyError):
            pass
        return None

    def adopt(self, simulator, name, key, source, *, verify, subdirectory=None, url=None):
        """Move an unpacked seed into the store; never download or copy.

        verify(source) must raise unless every file matches the pin. The seed
        is renamed, so it must share the store's filesystem; on failure it is
        moved back.
        """
        parent = self.root / _component(simulator) / _component(name)
        destination = parent / _component(key)
        source = Path(source)
        with file_lock(parent / f".{key}.lock"):
            data = self.cached(simulator, name, key)
            if data is not None:
                print(f"[assets] cached {simulator}/{name}", flush=True)
                return data
            if not source.is_dir() or source.is_symlink():
                raise ValueError(f"asset seed is not a directory: {source}")
            print(f"[assets] verifying seed {source}", flush=True)
            verify(source)
            staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=parent))
            target = staging / "data" / (_relative(subdirectory) if subdirectory else "")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, target)
            try:
                files = _inventory(staging / "data")
                if not files:
                    raise ValueError(f"asset seed {source} contains no files")
                (staging / "complete.json").write_text(json.dumps({
                    "key": key, "url": url, "verified": True, "adopted_from": str(source),
                    "files": files}, indent=2) + "\n")
                if destination.exists():
                    shutil.rmtree(destination)
                staging.rename(destination)
            except BaseException:
                os.rename(target, source)
                shutil.rmtree(staging, ignore_errors=True)
                raise
            print(f"[assets] adopted {simulator}/{name} from {source}", flush=True)
            return destination / "data"

    def ensure(self, simulator, name, url, *, checksum=None, size=None, contains=(".",)):
        """Return an unpacked package, publishing it only after successful extraction.

        Without an upstream checksum, key by URL and record the observed digest;
        this provides reuse, not verification against a trusted expected digest.
        """
        for component in (simulator, name):
            if not component or Path(component).name != component or component in (".", ".."):
                raise ValueError(f"invalid asset identifier: {component}")
        if checksum is not None and (len(checksum) != 64
                                     or any(c not in "0123456789abcdef" for c in checksum)):
            raise ValueError("asset checksum must be a lowercase SHA-256 digest")
        entries = [_relative(entry) for entry in contains]
        key = checksum or "url-" + hashlib.sha256(url.encode()).hexdigest()
        parent = self.root / simulator / name
        destination = parent / key
        with file_lock(parent / f".{key}.lock"):
            marker = destination / "complete.json"
            if marker.exists():
                try:
                    saved = json.loads(marker.read_text())
                    data = destination / "data"
                    if (saved["key"] == key and saved["files"] == _inventory(data)
                            and all((data / entry).exists() for entry in entries)):
                        print(f"[assets] cached {simulator}/{name}", flush=True)
                        return data
                except (OSError, ValueError, KeyError):
                    pass
            with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=parent) as temporary:
                staging = Path(temporary)
                archive = staging / "download"
                print(f"[assets] downloading {simulator}/{name}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "rosbridge-assets/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
                observed = sha256(archive)
                if checksum is not None and observed != checksum:
                    raise ValueError(f"SHA-256 mismatch for {name}")
                if size is not None and archive.stat().st_size != size:
                    raise ValueError(f"download size mismatch for {name}")
                data = staging / "data"
                data.mkdir()
                if zipfile.is_zipfile(archive):
                    with zipfile.ZipFile(archive) as bundle:
                        for member in bundle.namelist():
                            _relative(member)
                        bundle.extractall(data)
                else:
                    with tarfile.open(archive) as bundle:
                        bundle.extractall(data, filter="data")
                if not all((data / entry).exists() for entry in entries):
                    raise ValueError(f"package {name} is missing declared contents")
                files = _inventory(data)
                if not files:
                    raise ValueError(f"package {name} contains no files")
                archive.unlink()
                (staging / "complete.json").write_text(json.dumps({
                    "key": key, "url": url, "sha256": observed,
                    "verified": checksum is not None, "files": files,
                }, indent=2) + "\n")
                if destination.exists():
                    shutil.rmtree(destination)
                staging.rename(destination)
            return destination / "data"


def link_asset(source, destination):
    """Atomically link a cached asset into a prepared tree."""
    source, destination = Path(source).absolute(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() and not destination.is_symlink():
        raise ValueError(f"asset path is occupied by local contents: {destination}; "
                         "move it aside before preparing again")
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}")
    try:
        temporary.symlink_to(source, target_is_directory=source.is_dir())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
