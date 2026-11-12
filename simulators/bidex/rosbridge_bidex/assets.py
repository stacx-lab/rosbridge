"""BiDex-100's assets, mounted into the generated tree as upstream's setup_assets_links.sh links them.

The packs are LFS objects of the pinned Hugging Face dataset revision; eval_dataset.h5 (the ID
robot poses and horizons) is the Git LFS object the pinned source exports a pointer for. RoboCasa
objects and textures are the robocasa simulator's cached archives, identical to BiDex's
robocasa_assets pack in every file BiDex uses. Upstream writes temporary MJCF beside each object
it loads and beside the OOD scene, so mounts are real directories down to object depth with links
below: the shared caches are never written.
"""
import json
from pathlib import Path
import re
import shlex
import sys

from rosbridge.assets import AssetStore, link_asset
from rosbridge.assets.cache import sha256
from rosbridge.project import project_root
from rosbridge.storage import write_json

DATASET = ("https://huggingface.co/datasets/Robbyant-Research/BiDex-100/resolve/"
           "ca846b94d51302387cc6a32e124b1e38834d1974/assets/{name}/{name}.tar.gz")
LFS = "https://media.githubusercontent.com/media/robbyant-research/BiDex-100/{revision}/{path}"
# name: (sha256, size)
PACKS = {
    "libero": ("ee2191e18aee4d25f424bd1c85d3d980384d9106bce89c3c8d799647312ea860", 289121848),
    "articraft_objects": ("389101a3dac6f958956550f0eea0f74ee8656704fde3ec4e6ccee0effbe85b09", 30662716),
    "assembly_objects": ("4b78311782929d916f0e9b6df0e5d1bc2697b1405e4eec2acb5fa4894c5e1327", 20963),
    "astribot_descriptions": ("af4dd6067a758049f1b6e05cf9a7bd50be96e9cd690ec47b85b953d0098f38f2", 93444195),
}
EVAL = "bidex/eval/eval_dataset.h5", "7903a930c6596e98bb2d05ecb49a1b77341051900861e19f646cdfd512746b0a", 12364733
MARKER = ".asset_links.json"


def mirror(source, destination, depth):
    """Real directories depth levels deep, with links to everything below them."""
    if depth and source.is_dir() and not source.is_symlink():
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir()):
            mirror(child, destination / child.name, depth - 1)
    else:
        link_asset(source, destination)


def robocasa(store, name):
    """An archive as the robocasa simulator prepares it: its cache, a seed or a download."""
    sys.path.append(str(project_root() / "simulators/robocasa"))
    from rosbridge_robocasa.assets import ARCHIVES, package
    return package(store, name) / ARCHIVES[name][1]


def aliases(runtime):
    """setup_assets_links.sh's role names for one canonical model: (alias, canonical)."""
    objects = runtime / "bidex/libero/objects"
    script = (runtime / "scripts/setup_assets_links.sh").read_text()
    table = re.search(r"<<'ASSEMBLY_ALIASES'\n(.*?)\nASSEMBLY_ALIASES", script, re.S).group(1)
    for line in table.split("\n"):
        canonical, *names = line.split()
        for name in names:
            yield (objects / "assembly_objects" / name / f"{name}.xml",
                   objects / "assembly_objects" / canonical / f"{canonical}.xml")
    for side in ("left", "mid", "right"):
        yield (objects / f"dexjoco_objects/threading_rod_{side}/threading_rod_{side}.xml",
               objects / "dexjoco_objects/threading_rod/threading_rod.xml")
    baskets = runtime / "bidex/robocasa/assets/objects/lightwheel/tiered_basket"
    for suffix in "bcd":
        yield baskets / f"TieredBasket007_{suffix}", baskets / "TieredBasket007"


def prepare(runtime, task, store=None):
    runtime = Path(runtime)
    try:
        return check_task(runtime, task)  # one preparation serves every task
    except RuntimeError:
        pass
    store = store or AssetStore()
    pack = {name: store.ensure("bidex", name, DATASET.format(name=name), checksum=digest, size=size)
            for name, (digest, size) in PACKS.items()}
    meshes = pack["astribot_descriptions"] / "urdf/astribot_s1_urdf/meshes"
    mounts = [
        (pack["libero"], "bidex/libero", 4),
        (pack["articraft_objects"], "bidex/libero/objects/articraft_objects", 2),
        (pack["assembly_objects"], "bidex/libero/objects/assembly_objects", 2),
        (robocasa(store, "objaverse"), "bidex/robocasa/assets/objects/objaverse", 3),
        (robocasa(store, "objects_lightwheel"), "bidex/robocasa/assets/objects/lightwheel", 3),
        (robocasa(store, "textures"), "bidex/robocasa/assets/textures", 0),
        (robocasa(store, "generative_textures"), "bidex/robocasa/assets/generative_textures", 0),
        (meshes, "bidex/robots/assembly/astribot_s1/meshes", 0),
        (meshes, "bidex/robots/assembly/astribot_s1/shared/meshes", 0),
    ]
    for source, destination, depth in mounts:
        mirror(source, runtime / destination, depth)
    for alias, canonical in aliases(runtime):
        link_asset(canonical, alias)
    # git archive exports the LFS object where git-lfs is installed, and otherwise its pointer.
    path, digest, size = EVAL
    target = runtime / path
    if target.is_symlink() or sha256(target) != digest:
        if not target.is_symlink():
            if f"oid sha256:{digest}".encode() not in target.read_bytes():
                raise ValueError(f"{target} is neither the pinned evaluation data nor its pointer")
            target.unlink()
        revision = json.loads((runtime / ".prepared.json").read_text())["revision"]
        data = store.ensure("bidex", "eval_dataset", LFS.format(revision=revision, path=path),
                            checksum=digest, size=size, contains=["eval_dataset.h5"])
        link_asset(data / "eval_dataset.h5", target)
    mounted = [destination for _, destination, _ in mounts]
    mounted += [str(alias.relative_to(runtime)) for alias, _ in aliases(runtime)] + [path]
    write_json(runtime / MARKER, mounted)
    check_task(runtime, task)
    print(f"[assets] BiDex-100 assets ready; cache: {store.root}", flush=True)


def check_task(runtime, task):
    """Fail before constructing an environment; never download during a reset."""
    runtime = Path(runtime)
    try:
        mounted = json.loads((runtime / MARKER).read_text())
        if mounted and all((runtime / path).exists() for path in mounted):
            return
    except (OSError, ValueError):
        pass
    raise RuntimeError("BiDex-100 assets are not prepared. Run: "
                       f"rosbridge prepare bidex --task {shlex.quote(task)}")
