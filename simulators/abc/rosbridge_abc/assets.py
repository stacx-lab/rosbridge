"""Resolve upstream ABC asset packages and mount the shared cache into its tree."""
import importlib.util
import json
import shlex
import sys

from rosbridge.assets import AssetStore, link_asset
from rosbridge.storage import write_json


def _upstream_prepare(runtime):
    spec = importlib.util.spec_from_file_location("_rosbridge_abc_prepare", runtime / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass annotations require a registered module
    spec.loader.exec_module(module)
    return module


def prepare(runtime, task, store=None):
    upstream = _upstream_prepare(runtime)
    packages = upstream.load_sim_manifest()
    wanted = upstream.sim_task_packages([task])
    store = store or AssetStore()
    mounted = []
    for name, package in packages.items():
        if name not in wanted:
            continue
        data = store.ensure("abc", name, package["uri"], checksum=package["sha256"],
                            size=package["size_bytes"], contains=package["contains"])
        for entry in package["contains"]:
            destination = runtime / package["extract_dir"] / entry
            if name == upstream.ROBOCASA_PACKAGE:
                # The small manifest package has scripts; external object packs
                # mount alongside them, without writing into the cached package.
                destination.mkdir(parents=True, exist_ok=True)
                for child in (data / entry).iterdir():
                    link_asset(child, destination / child.name)
                    mounted.append(str((destination / child.name).relative_to(runtime)))
            else:
                link_asset(data / entry, destination)
                mounted.append(str(destination.relative_to(runtime)))
    if upstream.ROBOCASA_PACKAGE in wanted:
        for name, pack_id in upstream.ROBOCASA_PACKS:
            # Upstream provides stable URLs but no expected checksums for these zips. They
            # are RoboCasa's own packs, so they share the robocasa simulator's cache.
            data = store.ensure("robocasa", name, upstream.ROBOCASA_URL.format(pack_id=pack_id))
            destination = runtime / "abc_sim/models/assets_robocasa" / name
            link_asset(data, destination)
            mounted.append(str(destination.relative_to(runtime)))
    marker = runtime / ".asset_tasks.json"
    tasks = json.loads(marker.read_text()) if marker.exists() else {}
    tasks[upstream.resolve_sim_task(task)] = mounted
    write_json(marker, tasks)
    print(f"[assets] {task} ready; cache: {store.root}", flush=True)


def require_assets(runtime, task, scene):
    """Fail before constructing an environment; never download during a reset."""
    marker = runtime / ".asset_tasks.json"
    try:
        paths = json.loads(marker.read_text())[scene]
        if paths and all((runtime / path).exists() for path in paths):
            return
    except (OSError, ValueError, KeyError):
        pass
    raise RuntimeError(f"Assets for {task!r} are not prepared. Run: "
                       f"rosbridge prepare abc --task {shlex.quote(task)}")


def check_task(runtime, task):
    scene = _upstream_prepare(runtime).resolve_sim_task(task)
    require_assets(runtime, task, scene)
