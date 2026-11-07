"""Pinned BEHAVIOR data bundles and the dataset key, linked into each simulator's data root.

The three archives of behavior-1k/zipped-datasets at a pinned revision are verified by their
SHA-256 and size. With ROSBRIDGE_ASSET_SEED, RoboEnv's extracted depot
(<seed>/behavior/datasets/<dataset>) is verified file by file against RoboEnv's receipts
(<seed>/behavior/<dataset>.json) and adopted instead of downloading. The BEHAVIOR Data Bundle
is encrypted; OmniGibson decrypts it with a key that requires accepting its license.
behavior-g1 adds the G1 with Sharpa hands and behavior-dex the R1Pro with Sharpa hands, each
imported once per source and adapter revision. Every activity shares the data, so one
preparation serves every task until a pin changes.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
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

                with gm.unlocked():  # importing OmniGibson already read DATA_PATH; the key goes to a temporary root
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


# Extra robot models: the pin naming each in a data root's marker, and its importer module.
MODELS = {"g1": "g1_model", "dex": "dex_model"}


def model_key(runtime, pin):
    """A model import's cache key: the prepared sources, the robot assets it may build on, the
    importer and the Sharpa helpers it shares."""
    from . import sharpa

    importer = importlib.import_module(f"{__package__}.{MODELS[pin]}")
    digest = hashlib.sha256((Path(runtime) / ".prepared.json").read_bytes())
    digest.update(ARCHIVES["omnigibson-robot-assets"][2].encode())
    for module in (importer, sharpa):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def crate_key(runtime):
    """The G1 crate's cache key: the G1 import's key and the converter."""
    from . import g1_crate

    digest = hashlib.sha256(model_key(runtime, "g1").encode())
    digest.update(Path(g1_crate.__file__).read_bytes())
    return digest.hexdigest()


def pins(runtime, model=None):
    """What a prepared data root links: the pinned archives and an extra robot model's import
    (the G1's as a crate)."""
    archives = {name: checksum for name, (_, _, checksum, _) in ARCHIVES.items()}
    if not model:
        return archives
    return {**archives, model: crate_key(runtime) if model == "g1" else model_key(runtime, model)}


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


def import_model(runtime, store, pin):
    """An extra robot model imported into OmniGibson, keyed by model_key."""
    model = importlib.import_module(f"{__package__}.{MODELS[pin]}")
    runtime, key = Path(runtime), model_key(runtime, pin)
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
        print(f"[assets] importing {model.TITLE} into OmniGibson", flush=True)
        subprocess.run([sys.executable, "-m", model.__name__, str(runtime),
                        str(staging / "data"), str(staging / "model" / model.MODEL)], check=True)

        def verify(directory):
            for name in model_files(model.MODEL):
                if not (directory / model.MODEL / name).is_file():
                    raise ValueError(f"the {model.TITLE} import produced no {name}")
        return store.adopt("behavior", model.MODEL, key, staging / "model", verify=verify) / model.MODEL


def model_files(name):
    return f"usd/{name}.usda", f"{name}.yaml", "drives.json"


def linked_files(pin):
    """The files of the model a prepared data root links; the G1's USD is its crate."""
    model = importlib.import_module(f"{__package__}.{MODELS[pin]}")
    if pin == "g1":
        from .g1_crate import USD
        return USD, f"{model.MODEL}.yaml", "drives.json"
    return model_files(model.MODEL)


def g1_crate(runtime, store, text):
    """The imported G1 (text) with its USD as a binary crate (g1_crate.py), keyed by the import and the converter."""
    from . import g1_crate as crate

    key = crate_key(runtime)
    data = store.cached("behavior", crate.MODEL, key)
    if data is not None:
        return data / crate.MODEL
    parent = store.root / "behavior" / crate.MODEL
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crate-", dir=parent) as temporary:
        staging = Path(temporary) / "model"
        print("[assets] storing the G1's USD as a binary crate", flush=True)
        subprocess.run([sys.executable, "-m", "rosbridge_behavior.g1_crate", str(text), str(staging / crate.MODEL)],
                       check=True)

        def verify(directory):
            model = directory / crate.MODEL
            for name in (crate.USD, f"{crate.MODEL}.yaml", "drives.json"):
                if not (model / name).is_file():
                    raise ValueError(f"the G1 crate has no {name}")
            if yaml.safe_load((model / f"{crate.MODEL}.yaml").read_text()).get("usd_path") != \
                    f"models/{crate.MODEL}/{crate.USD}":
                raise ValueError("the G1 crate's definition does not name its USD")
        return store.adopt("behavior", crate.MODEL, key, staging, verify=verify) / crate.MODEL


def prepare_model(runtime, task, store, pin):
    check = {"g1": check_g1_task, "dex": check_dex_task}[pin]
    try:
        return check(runtime, task)  # one preparation serves every task
    except RuntimeError:
        pass
    store = store or AssetStore()
    link_data(runtime, store, robots=[])  # the import reads the data root
    model = import_model(runtime, store, pin)
    if pin == "g1":
        model = g1_crate(runtime, store, model)  # boots open the crate, not the text import
    link_data(runtime, store, robots=[model])
    write_json(Path(runtime) / MARKER, pins(runtime, pin))
    check(runtime, task)
    title = importlib.import_module(f"{__package__}.{MODELS[pin]}").TITLE
    print(f"[assets] {task} ready with {title}; cache: {store.root}", flush=True)


def prepare_g1(runtime, task, store=None):
    prepare_model(runtime, task, store, "g1")


def prepare_dex(runtime, task, store=None):
    prepare_model(runtime, task, store, "dex")


def check_model_task(runtime, task, pin):
    model = importlib.import_module(f"{__package__}.{MODELS[pin]}")
    check_task(runtime, task, pin)
    directory = Path(runtime) / "datasets/omnigibson-robot-assets/models" / model.MODEL
    if not all((directory / name).is_file() for name in linked_files(pin)):
        raise RuntimeError(f"{model.TITLE} is not prepared. Run: {LICENSE}=YES rosbridge prepare "
                           f"{Path(runtime).name} --task {shlex.quote(task)}")


def check_g1_task(runtime, task):
    check_model_task(runtime, task, "g1")


def check_dex_task(runtime, task):
    check_model_task(runtime, task, "dex")


def check_task(runtime, task, model=None):
    """Fail before booting Isaac Sim; never download during a reset."""
    root = Path(runtime) / "datasets"
    try:
        if json.loads((Path(runtime) / MARKER).read_text()) != pins(runtime, model):
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
