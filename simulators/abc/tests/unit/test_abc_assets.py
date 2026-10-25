"""ABC task resolution and asset mounting without native imports."""
import shutil
import ast
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile
from types import SimpleNamespace
import pytest
from rosbridge_abc import assets as abc_assets
from rosbridge.assets import AssetStore
from rosbridge.catalog import tasks
from tests.asset_helpers import package, ensure


def test_patched_upstream_asset_map_covers_the_whole_catalog(tmp_path):
    root = Path(__file__).resolve().parents[4]
    upstream = root / "third_party/abc"
    shutil.copyfile(upstream / "prepare.py", tmp_path / "prepare.py")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", str(root / "patches/abc/0001-complete-task-asset-map.patch")],
                   cwd=tmp_path, check=True)
    assignments = ast.parse((tmp_path / "prepare.py").read_text()).body
    mapping = next(ast.literal_eval(node.value) for node in assignments
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "SIM_TASK_PACKAGES"
                       for target in node.targets))
    specs = {}
    for node in ast.walk(ast.parse((upstream / "abc_sim/task_specs.py").read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SimTaskSpec":
            values = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                      if kw.arg in ("name", "env_task")}
            specs[values["name"]] = values["env_task"]
    packages = {item["name"] for item in json.loads(
        (upstream / "abc_sim/models/assets_manifest.json").read_text())["packages"]}
    for task in tasks(root, "abc").values():
        assert set(mapping[specs[task["native_task"]]]) <= packages
    assert mapping["drawer"] == ("drawer", "marker")
    assert mapping["building_blocks"] == ("building_blocks",)

def test_abc_resolves_task_dependencies_and_reuses_after_runtime_deletion(tmp_path, monkeypatch):
    packages = {name: package(tmp_path, name) for name in ("robot", "bottles", "tray")}
    tasks = {"bottle_task": {"robot", "bottles"}, "tray_task": {"robot", "tray"}}
    catalog = SimpleNamespace(
        load_sim_manifest=lambda: packages,
        sim_task_packages=lambda names: tasks[names[0]],
        resolve_sim_task=lambda task: task,
        ROBOCASA_PACKAGE="assets_robocasa",
    )
    monkeypatch.setattr(abc_assets, "_upstream_prepare", lambda _: catalog)
    store = AssetStore(tmp_path / "cache")
    runtime = tmp_path / "runtime"
    abc_assets.prepare(runtime, "bottle_task", store)
    assert not (store.root / "abc/tray").exists()
    robot = (runtime / "abc_sim/models/assets/robot").resolve()
    abc_assets.prepare(runtime, "tray_task", store)
    assert (runtime / "abc_sim/models/assets/robot").resolve() == robot
    abc_assets.require_assets(runtime, "bottle_task", "bottle_task")
    shutil.rmtree(runtime)
    # Simulates a fresh source export and an offline machine.
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("redownload"))
    abc_assets.prepare(runtime, "bottle_task", store)
    assert (runtime / "abc_sim/models/assets/robot").resolve() == robot
    abc_assets.require_assets(runtime, "bottle_task", "bottle_task")
    with pytest.raises(RuntimeError, match="prepare abc --task tray_task"):
        abc_assets.require_assets(runtime, "tray_task", "tray_task")
    (runtime / "abc_sim/models/assets/bottles").unlink()
    with pytest.raises(RuntimeError, match="prepare abc --task bottle_task"):
        abc_assets.require_assets(runtime, "bottle_task", "bottle_task")


def test_abc_external_packs_do_not_modify_cached_metadata(tmp_path, monkeypatch):
    metadata = package(tmp_path, "assets_robocasa")
    metadata["extract_dir"] = "abc_sim/models"
    archive = tmp_path / "123.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("object/model.xml", "<mujoco/>")
    catalog = SimpleNamespace(
        load_sim_manifest=lambda: {"assets_robocasa": metadata},
        sim_task_packages=lambda _: {"assets_robocasa"},
        resolve_sim_task=lambda _: "inhand_transfer",
        ROBOCASA_PACKAGE="assets_robocasa", ROBOCASA_PACKS=(("objects", "123"),),
        ROBOCASA_URL=archive.as_uri().replace("123.zip", "{pack_id}.zip"),
    )
    monkeypatch.setattr(abc_assets, "_upstream_prepare", lambda _: catalog)
    store = AssetStore(tmp_path / "cache")
    runtime = tmp_path / "runtime"
    abc_assets.prepare(runtime, "transfer", store)
    metadata_cache = ensure(store, metadata) / "assets_robocasa"
    assert not (metadata_cache / "objects").exists()
    mounted = runtime / "abc_sim/models/assets_robocasa"
    assert (mounted / "mesh.obj").is_symlink()
    assert (mounted / "objects").is_symlink()
    assert (mounted / "objects/object/model.xml").read_text() == "<mujoco/>"
    assert (store.root / "robocasa/objects").is_dir()  # shared with the robocasa simulator
    abc_assets.require_assets(runtime, "transfer", "inhand_transfer")

