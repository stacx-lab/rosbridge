import hashlib
import json

import pytest

from rosbridge.assets import AssetStore
from rosbridge.catalog import resolve, simulator, tasks
from rosbridge_robodojo import assets


def test_catalog_pairs_the_base_tasks_with_the_dual_x5_rig():
    entries = list(tasks(name="robodojo").values())
    assert len(entries) == 42 and not any(t["native_task"].endswith("_random") for t in entries)
    for task in entries:
        assert (task["robot"], task["env_cfg_type"]) == ("robodojo/robodojo-arx-x5", "arx_x5")
        assert resolve(task["id"])[2] == task
    assert resolve("robodojo")[2]["native_task"] == "pack_objects_into_box"
    assert resolve("play_Xylophone")[2]["id"] == "robodojo/play-xylophone"


def test_manifest_runs_isaac_in_a_native_worker_with_pinned_forks():
    _, _, config = simulator("robodojo")
    assert (config["environment"], config["native_environment"]) == ("robodojo", "robodojo-native")
    assert config["native_backend"] == "rosbridge_robodojo.backend:RoboDojoSim"
    assert [(n["path"], len(n["source_revision"])) for n in config["nested_sources"]] == [
        ("third_party/IsaacLab", 40), ("third_party/curobo", 40)]


def entry(name, data, lfs):
    value = {"rfilename": name, "size": len(data)}
    if lfs:
        value["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    else:
        value["blobId"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return value


def test_files_verify_by_lfs_sha256_or_git_blob_and_seeds_must_match_exactly(tmp_path):
    files = {"Assets/Robots/x5/robot_config.yml": (b"urdf_path: x\n", False),
             "Assets/Object/mesh.usd": (b"\x00usd", True)}
    entries = [entry(name, data, lfs) for name, (data, lfs) in files.items()]
    seed = tmp_path / "seed"
    for name, (data, _) in files.items():
        path = seed / assets.relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert all(assets.verified(seed / assets.relative(e["rfilename"]), e) for e in entries)
    (seed / "Object/mesh.usd").write_bytes(b"\x01usd")
    store = AssetStore(tmp_path / "store")
    with pytest.raises(ValueError, match="differs from its pin"):
        store.adopt("robodojo", "assets", "rev", seed, verify=assets.verify_tree(entries))
    (seed / "Object/mesh.usd").write_bytes(b"\x00usd")
    (seed / "stray.txt").write_text("unpinned")
    with pytest.raises(ValueError, match="1 extra"):
        store.adopt("robodojo", "assets", "rev", seed, verify=assets.verify_tree(entries))
    (seed / "stray.txt").unlink()
    data = store.adopt("robodojo", "assets", "rev", seed, verify=assets.verify_tree(entries),
                       subdirectory="Assets")
    assert (data / "Assets/Robots/x5/robot_config.yml").is_file()
    for unsafe in ("../x", "Other/file", "Assets/../../x"):
        with pytest.raises(ValueError, match="unsafe"):
            assets.relative(unsafe)
    assert json.dumps(entries)
