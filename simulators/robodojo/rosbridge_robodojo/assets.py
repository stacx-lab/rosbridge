"""Pinned public RoboDojo assets (robots, objects, saved evaluation layouts), verified per file.

The Hugging Face dataset revision and its file listing pin every asset: LFS files
by SHA-256, others by git blob SHA-1. With ROSBRIDGE_ASSET_SEED, RoboEnv's
verified depot is adopted by rename; otherwise files are downloaded once.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

from rosbridge.assets import AssetStore, file_lock, link_asset, seed_root
from rosbridge.storage import write_json

REPOSITORY = "RoboDojo-Benchmark/RoboDojo"
REVISION = "0a709fe7863f6c869a8d141676585a883fce4270"
LISTING = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
FILE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{{name}}"
LAYOUTS = "Assets/Eval_Layout/RoboDojo/arx_x5"


def relative(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or ".." in path.parts or "\\" in name
            or path.parts[0] != "Assets"):
        raise ValueError(f"unsafe RoboDojo asset path: {name}")
    return Path(*path.parts[1:])


def verified(path, entry):
    """A file matches its pinned LFS SHA-256 or git blob SHA-1."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        return False
    lfs = entry.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == (lfs["sha256"] if lfs else entry["blobId"])


def listing(store):
    """The pinned dataset's Assets entries; the listing is kept beside the depot."""
    path = store.root / "robodojo" / "source.json"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = seed_root()
        if seed is not None and (seed / "robodojo/source.json").is_file():
            shutil.copyfile(seed / "robodojo/source.json", path.with_suffix(".tmp"))
        else:
            with urlopen(LISTING, timeout=90) as response:
                path.with_suffix(".tmp").write_bytes(response.read())
        path.with_suffix(".tmp").replace(path)
    source = json.loads(path.read_text())
    if source["sha"] != REVISION or source.get("gated") or source.get("private"):
        raise ValueError("expected the pinned public, ungated RoboDojo dataset listing")
    entries = [entry for entry in source["siblings"] if entry["rfilename"].startswith("Assets/")]
    if not entries or len({entry["rfilename"] for entry in entries}) != len(entries):
        raise ValueError("empty or duplicated RoboDojo asset listing")
    for entry in entries:
        relative(entry["rfilename"])
    return entries


def verify_tree(entries, workers=8):
    """Every pinned file is present and verified, and nothing else is."""
    def verify(directory):
        expected = {relative(entry["rfilename"]) for entry in entries}
        found = {path.relative_to(directory) for path in directory.rglob("*")
                 if path.is_file() or path.is_symlink()}
        if found != expected:
            raise ValueError(f"{directory} differs from the pinned listing: "
                             f"{len(expected - found)} missing, {len(found - expected)} extra")
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            checks = pool.map(lambda e: (e, verified(directory / relative(e["rfilename"]), e)), entries)
            for count, (entry, ok) in enumerate(checks, 1):
                if not ok:
                    raise ValueError(f"asset differs from its pin: {entry['rfilename']}")
                if count % 2500 == 0 or count == len(entries):
                    print(f"[assets] RoboDojo verified {count}/{len(entries)}", flush=True)
    return verify


class Pacer:
    """Shared request spacing and server-directed cooldown across download threads."""

    def __init__(self):
        self.lock, self.next = threading.Lock(), 0.

    def wait(self):
        with self.lock:
            delay = max(0., self.next - time.monotonic())
            self.next = time.monotonic() + delay + .15
        time.sleep(delay)

    def cooldown(self, headers):
        retry = headers.get("retry-after", "")
        with self.lock:
            self.next = max(self.next, time.monotonic() + min(float(retry) if retry.isdigit() else 60., 600.))


def download(directory, entry, pacer):
    path = directory / relative(entry["rfilename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        pacer.wait()
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as out:
                pending = Path(out.name)
                with urlopen(FILE.format(name=quote(entry["rfilename"])), timeout=90) as response:
                    shutil.copyfileobj(response, out, length=1 << 20)
            if not verified(pending, entry):
                pending.unlink()
                raise ValueError(f"downloaded asset failed verification: {entry['rfilename']}")
            pending.replace(path)
            return
        except HTTPError as error:
            if error.code == 429:
                pacer.cooldown(error.headers)
                continue
            if error.code < 500 or attempt == 5:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)
    raise TimeoutError(f"could not acquire {entry['rfilename']}")


def depot(store):
    """The verified Assets directory, adopting a seed or downloading as needed."""
    entries = listing(store)
    data = store.cached("robodojo", "assets", REVISION)
    if data is not None:
        return data / "Assets"
    seed = seed_root()
    if seed is not None:
        source = seed / "robodojo" / REVISION / "Assets"
        return store.adopt("robodojo", "assets", REVISION, source, verify=verify_tree(entries),
                           subdirectory="Assets", url=LISTING) / "Assets"
    parent = store.root / "robodojo" / "assets"
    staging = parent / f".{REVISION}-download/Assets"
    with file_lock(parent / f".{REVISION}-download.lock"):
        missing = [e for e in entries if not verified(staging / relative(e["rfilename"]), e)]
        if shutil.disk_usage(parent).free < sum(e["size"] for e in missing) + (2 << 30):
            raise ValueError("insufficient space for RoboDojo assets; set ROSBRIDGE_ASSET_ROOT")
        print(f"[assets] downloading {len(missing)} RoboDojo files", flush=True)
        pacer = Pacer()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda entry: download(staging, entry, pacer), missing))
    # Publishing reuses adoption: the staged files are verified once more and renamed.
    return store.adopt("robodojo", "assets", REVISION, staging, verify=verify_tree(entries),
                       subdirectory="Assets", url=LISTING) / "Assets"


def prepare(runtime, task, store=None):
    runtime, store = Path(runtime), store or AssetStore()
    link_asset(depot(store), runtime / "Assets")
    marker = runtime / ".asset_tasks.json"
    tasks = json.loads(marker.read_text()) if marker.exists() else {}
    tasks[task] = ["Assets"]
    write_json(marker, tasks)
    check_task(runtime, task)
    print(f"[assets] {task} ready; cache: {store.root}", flush=True)


def check_task(runtime, task):
    """Fail before booting Isaac Sim; never download during a reset."""
    runtime = Path(runtime)
    try:
        json.loads((runtime / ".asset_tasks.json").read_text())[task]
        layouts = runtime / LAYOUTS
        if (runtime / "Assets/Robots/x5/robot_config.yml").is_file() and any(
                layouts.glob(f"*/{task}_*.json")):
            return
    except (OSError, ValueError, KeyError):
        pass
    raise RuntimeError(f"RoboDojo assets for {task!r} are not prepared. Run: "
                       f"rosbridge prepare robodojo --task {shlex.quote(task)}")
