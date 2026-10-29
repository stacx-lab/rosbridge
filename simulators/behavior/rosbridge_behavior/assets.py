"""Pinned BEHAVIOR data bundles and the dataset key, linked into each simulator's data root.

The three archives of behavior-1k/zipped-datasets at a pinned revision are verified by their
SHA-256 and size. With ROSBRIDGE_ASSET_SEED, RoboEnv's extracted depot
(<seed>/behavior/datasets/<dataset>) is verified file by file against RoboEnv's receipts
(<seed>/behavior/<dataset>.json) and adopted instead of downloading. The BEHAVIOR Data Bundle
is encrypted; OmniGibson decrypts it with a key that requires accepting its license.
behavior-g1 adds the G1 with Sharpa hands, imported once per source and adapter revision.
Every activity shares the data, so one preparation serves every task until a pin changes.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import yaml

from rosbridge.assets import AssetStore, file_lock, link_asset, seed_root
from rosbridge.assets.cache import sha256
from rosbridge.storage import write_json

REVISION = "9f0d57d465726976ed98138d3f8b8ca3e2186775"
URL = f"https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/{REVISION}/{{file}}"
# dataset: archive, bytes, sha256, VERSION
ARCHIVES = {
    "behavior-1k-assets": (
        "behavior-1k-assets-3.9.0.zip", 31457673073,
        "09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e", "3.9.0"),
    "omnigibson-robot-assets": (
        "omnigibson-robot-assets-3.8.2.zip", 641004353,
        "3d813b2181e0581cf2300a40892de70f8475fe59346ceeea4fb9bf7ff21ce126", "3.8.2"),
    "2026-challenge-task-instances": (
        "2026-challenge-task-instances.zip", 108543852,
        "d4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81", None),
}
LICENSE = "ROSBRIDGE_BEHAVIOR_ACCEPT_LICENSE"
KEY = "omnigibson.key"
INSTANCES = 20  # public test instances per activity
MARKER = ".assets.json"  # the pins a preparation linked into the data root


def accept_license():
    if os.environ.get(LICENSE, "").upper() != "YES":
        raise ValueError(
            f"BEHAVIOR's data is licensed under the BEHAVIOR Data Bundle End User License Agreement "
            f"(https://behavior.stanford.edu/); set {LICENSE}=YES to accept it and prepare.")


def verify_receipt(files):
    """Require exactly the receipt's files below a dataset, each with its recorded sha256."""
    def verify(directory):
        found = {str(path.relative_to(directory)) for path in directory.rglob("*")
                 if path.is_file() or path.is_symlink()}
        if found != set(files):
            raise ValueError(f"{directory} does not contain exactly its receipt's files: "
                             f"{len(set(files) - found)} missing, {len(found - set(files))} extra")
        with ThreadPoolExecutor(8) as pool:
            for name, digest in zip(files, pool.map(lambda name: sha256(directory / name), files)):
                if digest != files[name]:
                    raise ValueError(f"{directory / name} differs from its receipt")
    return verify


def dataset(store, name):
    """An extracted dataset, adopting a seed or downloading its archive as needed."""
    file, size, checksum, version = ARCHIVES[name]
    data = store.cached("behavior", name, checksum)
    seed = seed_root()
    if data is None and seed is not None:
        receipt = seed / "behavior" / f"{name}.json"
        if not receipt.is_file():
            raise ValueError(f"asset seed lacks {receipt}; seeded preparation never downloads")
        receipt = json.loads(receipt.read_text())
        if (receipt["revision"], receipt["archive"]["sha256"]) != (REVISION, checksum):
            raise ValueError(f"seed receipt for {name} names another archive")
        data = store.adopt("behavior", name, checksum, seed / "behavior/datasets" / name,
                           verify=verify_receipt(receipt["files"]), url=URL.format(file=file))
    elif data is None:
        data = store.ensure("behavior", name, URL.format(file=file), checksum=checksum, size=size)
    if version is not None and (data / "VERSION").read_text().strip() != version:
        raise ValueError(f"{name} is not version {version}")
    return data


def key(store):
    """The dataset key: copied from a seed, or installed by OmniGibson's own download."""
    path = store.root / "behavior" / KEY
    with file_lock(store.root / "behavior" / f".{KEY}.lock"):
        if path.is_file():
            return path
        seed = seed_root()
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            pending = Path(temporary) / KEY
            if seed is not None:
                shutil.copyfile(seed / "behavior/datasets" / KEY, pending)
            else:
                from omnigibson.macros import gm
                from omnigibson.utils.asset_utils import download_key

                gm.DATA_PATH = temporary
                download_key()
            pending.chmod(0o600)
            pending.replace(path)
    return path


def link_data(runtime, store, robots=None):
    """The data root OmniGibson reads (environment.data_root): datasets and the key. Extra
    robot models join the robot assets in an overlay of links."""
    accept_license()
    root = Path(runtime) / "datasets"
    for name in ARCHIVES:
        data = dataset(store, name)
        if name != "omnigibson-robot-assets" or robots is None:
            link_asset(data, root / name)
            continue
        overlay = root / name
        if overlay.is_symlink():
            overlay.unlink()
        for entry in data.iterdir():
            if entry.name != "models":
                link_asset(entry, overlay / entry.name)
        for model in (*(data / "models").iterdir(), *robots):
            link_asset(model, overlay / "models" / model.name)
    link_asset(key(store), root / KEY)
    return root


def model_key(runtime):
    """The G1 import's cache key: the prepared sources and the importer."""
    from . import g1_model

    digest = hashlib.sha256((Path(runtime) / ".prepared.json").read_bytes())
    digest.update(Path(g1_model.__file__).read_bytes())
    return digest.hexdigest()


def pins(runtime, g1=False):
    """What a prepared data root links: the pinned archives and, for behavior-g1, the G1 import."""
    archives = {name: checksum for name, (_, _, checksum, _) in ARCHIVES.items()}
    return {**archives, "g1": model_key(runtime)} if g1 else archives


def prepare(runtime, task, store=None):
    try:
        return check_task(runtime, task)  # one preparation serves every task
    except RuntimeError:
        pass
    store = store or AssetStore()
    link_data(runtime, store)
    write_json(Path(runtime) / MARKER, pins(runtime))
    check_task(runtime, task)
    print(f"[assets] {task} ready; cache: {store.root}", flush=True)


def g1_model(runtime, store):
    """The imported G1 with Sharpa hands, keyed by the prepared sources and the importer."""
    from . import g1_model as model

    runtime, key = Path(runtime), model_key(runtime)
    data = store.cached("behavior", model.MODEL, key)
    if data is not None:
        return data / model.MODEL
    parent = store.root / "behavior" / model.MODEL
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=parent) as temporary:
        staging = Path(temporary)
        for entry in (runtime / "datasets").iterdir():
            (staging / "data").mkdir(exist_ok=True)
            (staging / "data" / entry.name).symlink_to(entry.resolve())
        print("[assets] importing the G1 with Sharpa hands into OmniGibson", flush=True)
        subprocess.run([sys.executable, "-m", "rosbridge_behavior.g1_model", str(runtime),
                        str(staging / "data"), str(staging / "model" / model.MODEL)], check=True)

        def verify(directory):
            for name in (f"usd/{model.MODEL}.usda", f"{model.MODEL}.yaml", "drives.json"):
                if not (directory / model.MODEL / name).is_file():
                    raise ValueError(f"the G1 import produced no {name}")
        return store.adopt("behavior", model.MODEL, key, staging / "model", verify=verify) / model.MODEL


def prepare_g1(runtime, task, store=None):
    try:
        return check_g1_task(runtime, task)  # one preparation serves every task
    except RuntimeError:
        pass
    store = store or AssetStore()
    link_data(runtime, store, robots=[])  # the import reads the data root
    link_data(runtime, store, robots=[g1_model(runtime, store)])
    write_json(Path(runtime) / MARKER, pins(runtime, g1=True))
    check_g1_task(runtime, task)
    print(f"[assets] {task} ready with the G1; cache: {store.root}", flush=True)


def check_g1_task(runtime, task):
    from .g1_model import MODEL

    check_task(runtime, task, g1=True)
    model = Path(runtime) / "datasets/omnigibson-robot-assets/models" / MODEL
    if not all((model / name).is_file() for name in (f"usd/{MODEL}.usda", f"{MODEL}.yaml", "drives.json")):
        raise RuntimeError(f"the G1 model is not prepared. Run: {LICENSE}=YES rosbridge prepare "
                           f"behavior-g1 --task {shlex.quote(task)}")


def check_task(runtime, task, g1=False):
    """Fail before booting Isaac Sim; never download during a reset."""
    root = Path(runtime) / "datasets"
    try:
        if json.loads((Path(runtime) / MARKER).read_text()) != pins(runtime, g1):
            raise ValueError("the data root links other pins")
        instances = root / "2026-challenge-task-instances"
        metadata = yaml.safe_load((instances / "metadata/available_tasks.yaml").read_text())
        scene = metadata[task][0]["scene_model"]
        found = list((instances / "scene_test/public" / scene / "json" /
                      f"{scene}_task_{task}_instances").glob("*-tro_state.json"))
        if (len(found) == INSTANCES and (root / KEY).is_file()
                and all((root / name / "VERSION").is_file() or not version
                        for name, (*_, version) in ARCHIVES.items())):
            return
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        pass
    simulator = Path(runtime).name
    raise RuntimeError(f"BEHAVIOR assets for {task!r} are not prepared. Run: "
                       f"{LICENSE}=YES rosbridge prepare {simulator} --task {shlex.quote(task)}")
