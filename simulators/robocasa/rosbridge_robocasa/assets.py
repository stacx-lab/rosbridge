"""RoboCasa's kitchen asset archives, cached once and overlaid onto the generated tree.

Every task samples kitchens from the fixture, texture and generative-texture archives and
objects from the objaverse, lightwheel and aigen registries, so all six archives are
prepared together. Upstream publishes stable Box URLs without checksums; ABC mounts two of
the same packs from this cache. With ROSBRIDGE_ASSET_SEED, extracted archives
(<seed>/robocasa/<root>/) are verified against their per-file sha256 receipts
(<seed>/robocasa/<name>.json, as RoboEnv records them) and adopted instead of downloading.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex

from rosbridge.assets import AssetStore, link_asset, seed_root
from rosbridge.assets.cache import sha256
from rosbridge.storage import write_json

URL = "https://utexas.box.com/shared/static/{pack}.zip"
# name: (Box pack ID, archive root directory, its parent below robocasa/models/assets)
ARCHIVES = {
    "textures": ("4i85ileasdvstmlln5sbvzptz7keuoy1", "textures", "."),
    "generative_textures": ("ebaad09k82tmfmlq6ohdkmrh8izl9vn5", "generative_textures", "."),
    "fixtures_lightwheel": ("idbncsadpnaz1jfl4i6m8qejawk7p9pi", "fixtures", "."),
    "objaverse": ("03eionyo8fk3a9dsksq9jb8du5lqfw8h", "objaverse", "objects"),
    "aigen_objs": ("nwi1vrn5pgbo95kushkasa3nx1i012ff", "aigen_objs", "objects"),
    "objects_lightwheel": ("vckqvvkh1z8t69k8qcpcmee6k66stii4", "lightwheel", "objects"),
}
MARKER = ".asset_links.json"


def url(name):
    return URL.format(pack=ARCHIVES[name][0])


def verify_receipt(files, root):
    """Require exactly the receipt's files below root, each with its recorded sha256."""
    expected = {entry["path"]: entry["sha256"] for entry in files}

    def verify(directory):
        found = {f"{root}/{path.relative_to(directory)}" for path in directory.rglob("*")
                 if path.is_file()}
        if found != set(expected):
            raise ValueError(f"{directory} does not contain exactly the receipt's {root} files")
        paths = list(expected)
        with ThreadPoolExecutor(8) as pool:
            digests = pool.map(lambda path: sha256(directory.parent / path), paths)
            for path, digest in zip(paths, digests):
                if digest != expected[path]:
                    raise ValueError(f"{directory.parent / path} differs from its receipt")
    return verify


def package(store, name):
    """The archive's data directory, adopting a seed or downloading as needed."""
    root, address = ARCHIVES[name][1], url(name)
    key = "url-" + hashlib.sha256(address.encode()).hexdigest()  # AssetStore.ensure's key
    data = store.cached("robocasa", name, key)
    if data is not None:
        return data
    seed = seed_root()
    if seed is not None:
        receipt = seed / "robocasa" / f"{name}.json"
        if not receipt.is_file():
            raise ValueError(f"asset seed lacks {receipt}; seeded preparation never downloads")
        receipt = json.loads(receipt.read_text())
        if receipt["url"] != address:
            raise ValueError(f"seed receipt for {name} names another archive: {receipt['url']}")
        return store.adopt("robocasa", name, key, seed / "robocasa" / root, subdirectory=root,
                           url=address, verify=verify_receipt(receipt["files"], root))
    return store.ensure("robocasa", name, address)


def overlay(source, destination, mounted):
    """Link source at destination, descending into directories the upstream tree has.

    Upstream tracks some fixture files that an archive repeats; its own copies stay.
    """
    if destination.is_dir() and not destination.is_symlink():
        for child in sorted(source.iterdir()):
            overlay(child, destination / child.name, mounted)
    elif destination.is_symlink() or not destination.exists():
        link_asset(source, destination)
        mounted.append(destination)


def prepare(runtime, task, store=None):
    runtime = Path(runtime)
    try:
        return check_task(runtime, task)  # one preparation serves every task
    except RuntimeError:
        pass
    store = store or AssetStore()
    assets, mounted = runtime / "robocasa/models/assets", []
    for name, (_, root, parent) in ARCHIVES.items():
        overlay(package(store, name) / root, assets / parent / root, mounted)
    # Both upstreams warn on every import until a private macro module exists.
    for source in ("robocasa", "robosuite/robosuite"):
        (runtime / source / "macros_private.py").touch()
    write_json(runtime / MARKER, sorted(str(path.relative_to(runtime)) for path in mounted))
    check_task(runtime, task)
    print(f"[assets] RoboCasa kitchen assets ready; cache: {store.root}", flush=True)


def check_task(runtime, task):
    """Fail before constructing an environment; never download during a reset."""
    runtime = Path(runtime)
    try:
        mounted = json.loads((runtime / MARKER).read_text())
        if mounted and all((runtime / path).exists() for path in mounted):
            return
    except (OSError, ValueError):
        pass
    raise RuntimeError("RoboCasa assets are not prepared. Run: "
                       f"rosbridge prepare robocasa --task {shlex.quote(task)}")
