"""HandsOn bundles its robot, task and rendering assets in the pinned source; nothing is downloaded.

Upstream archives its Core task surfaces under content hashes. Three archived MDL files, Isaac Sim's
OmniGlass.mdl (byte for byte), an older OmniPBR.mdl and a LabelFilm material, import Isaac Sim's
OmniGlass_Opacity, OmniPBR_ClearCoat and OmniPBRBase modules by relative name. MDL 1.7 resolves such
imports only beside the importing file, where the archive has none, so those materials fail to
compile and render as an opaque fallback. Preparation links the three modules there from the pinned
Isaac Sim, which owns them as it owns the OmniPBR and OmniSurface modules upstream leaves to it.
"""
import hashlib
from pathlib import Path

SURFACES = "handson/handson/assets/task_surfaces/objects"  # below the generated runtime
MDL_MODULES = ("OmniGlass_Opacity.mdl", "OmniPBR_ClearCoat.mdl", "OmniPBRBase.mdl")
# The archived OmniGlass.mdl; the linked modules are its siblings in the Isaac Sim that matches it.
ARCHIVED_OMNIGLASS = "c3f9555fbf8db2c4d47ac25501afaf23720260255381bde3df003fecf01869af"


def isaac_mdl_base():
    """Isaac Sim's core MDL modules in this interpreter's environment, found without importing isaacsim."""
    import importlib.util

    spec = importlib.util.find_spec("isaacsim")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("Isaac Sim is not installed in this interpreter")
    base = Path(next(iter(spec.submodule_search_locations))) / "kit/mdl/core/Base"
    if hashlib.sha256((base / "OmniGlass.mdl").read_bytes()).hexdigest() != ARCHIVED_OMNIGLASS:
        raise ValueError(f"{base} is not the Isaac Sim whose OmniGlass.mdl upstream archived")
    return base


def link_mdl_modules(runtime, base=None):
    from rosbridge.assets import link_asset

    base = base or isaac_mdl_base()
    for name in MDL_MODULES:
        link_asset(base / name, Path(runtime) / SURFACES / name)


def check_mdl_modules(runtime, base=None):
    base = base or isaac_mdl_base()
    for name in MDL_MODULES:
        path = Path(runtime) / SURFACES / name
        if not path.is_symlink() or path.resolve() != (base / name).resolve():
            raise ValueError(f"{path} does not link Isaac Sim's {name}")


def prepare(runtime, task):
    link_mdl_modules(runtime)
    check_task(runtime, task)


def check_task(runtime, task):
    """The task has an official evaluation budget, the bundled assets match upstream's SHA-256 pins
    and the archived materials find Isaac Sim's modules."""
    import json
    import sys

    try:
        source = Path(runtime) / "handson"
        if str(source) not in sys.path:  # its editable install loads only in the next interpreter
            sys.path.append(str(source))
        budgets = json.loads((source / "handson/evaluation/budgets.json").read_text())
        if task not in budgets:
            raise ValueError(f"{task} has no evaluation budget")
        from handson.robots.assets import require_benchmark_assets
        require_benchmark_assets()
        check_mdl_modules(runtime)
    except Exception as error:
        raise RuntimeError(f"HandsOn assets unavailable; run rosbridge prepare handson --task {task}") from error
