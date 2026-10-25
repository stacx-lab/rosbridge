"""Pinned RoboTwin 2.0 embodiment and object packs, mounted into the generated tree.

Only the clean-background (demo_clean) scope is prepared; randomized background
textures are excluded, as in RoboEnv. With ROSBRIDGE_ASSET_SEED, an already
extracted, verified RoboEnv cache is adopted instead of downloading.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import zipfile
import zlib

from rosbridge.assets import AssetStore, link_asset, seed_root
from rosbridge.storage import write_json

REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
URL = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/{revision}/{name}.zip"
ARCHIVES = {
    "embodiments": (219859313, "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"),
    "objects": (3737778549, "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"),
}


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _crc32(path):
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            value = zlib.crc32(chunk, value)
    return value


def members(bundle, name):
    """Files of a pinned archive below its single root directory, without macOS metadata."""
    for item in bundle.infolist():
        parts = PurePosixPath(item.filename).parts
        if not parts or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store" or item.is_dir():
            continue
        if parts[0] != name or ".." in parts:
            raise ValueError(f"unexpected entry in {name}.zip: {item.filename}")
        yield item, Path(*parts[1:])


def verify_seed(archive, name, size, digest):
    """Check a seed's archive against the pin, then its extracted files against the archive."""
    def verify(directory):
        if archive.stat().st_size != size or _sha256(archive) != digest:
            raise ValueError(f"{archive} differs from the pinned {name} archive")
        with zipfile.ZipFile(archive) as bundle:
            expected = {}
            for item, relative in members(bundle, name):
                expected[relative] = item
            found = {path.relative_to(directory) for path in directory.rglob("*") if path.is_file()}
            if found != set(expected):
                raise ValueError(f"{directory} does not contain exactly the files of {name}.zip")
            for relative, item in expected.items():
                path = directory / relative
                if path.stat().st_size != item.file_size or _crc32(path) != item.CRC:
                    raise ValueError(f"{path} differs from {name}.zip")
    return verify


def package(store, name):
    """The verified pack's data directory, adopting a seed or downloading as needed."""
    size, digest = ARCHIVES[name]
    data = store.cached("robotwin", name, digest)
    if data is not None:
        return data
    url = URL.format(revision=REVISION, name=name)
    seed = seed_root()
    if seed is not None:
        source = seed / "robotwin" / f"{name}-{digest[:12]}"
        archive = seed / "robotwin" / f"{name}.zip"
        if not archive.is_file():
            raise ValueError(f"asset seed lacks {archive}; seeded preparation never downloads")
        return store.adopt("robotwin", name, digest, source, url=url, subdirectory=name,
                           verify=verify_seed(archive, name, size, digest))
    return store.ensure("robotwin", name, url, checksum=digest, size=size, contains=(name,))


def prepare(runtime, task, store=None):
    runtime, store = Path(runtime), store or AssetStore()
    mounted = []
    for name in ARCHIVES:
        destination = runtime / "assets" / name
        link_asset(package(store, name) / name, destination)
        mounted.append(str(destination.relative_to(runtime)))
    marker = runtime / ".asset_tasks.json"
    tasks = json.loads(marker.read_text()) if marker.exists() else {}
    tasks[task] = mounted
    write_json(marker, tasks)
    check_task(runtime, task)
    print(f"[assets] {task} ready; cache: {store.root}", flush=True)


def check_task(runtime, task):
    """Fail before constructing a scene; never download during a reset."""
    runtime = Path(runtime)
    try:
        paths = json.loads((runtime / ".asset_tasks.json").read_text())[task]
        if paths and all((runtime / path).is_dir() for path in paths):
            embodiment = runtime / "assets/embodiments/aloha-agilex/config.yml"
            if embodiment.is_file() and (runtime / f"envs/{task}.py").is_file():
                return
    except (OSError, ValueError, KeyError):
        pass
    raise RuntimeError(f"RoboTwin assets for {task!r} are not prepared. Run: "
                       f"rosbridge prepare robotwin --task {shlex.quote(task)}")
