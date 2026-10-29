import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rosbridge.assets import AssetStore
from rosbridge.catalog import resolve, simulator, tasks
from rosbridge_behavior import assets
from rosbridge_behavior.backend import BehaviorSim
from rosbridge_behavior.r1pro import R1Pro


def test_catalog_holds_the_100_challenge_activities():
    entries = list(tasks(name="behavior").values())
    assert len(entries) == 100 and {t["robot"] for t in entries} == {"behavior/behavior-r1pro"}
    for task in entries:
        assert task["id"] == "behavior/" + task["native_task"].lower().replace("_", "-")
        assert resolve(task["id"])[2] == task
    assert resolve("behavior")[2]["native_task"] == "turning_on_radio"
    assert resolve("behavior/putting_away_Halloween_decorations")[2]["id"] == \
        "behavior/putting-away-halloween-decorations"
    with pytest.raises(ValueError, match="Ambiguous"):  # behavior-g1 runs the same activities
        resolve("turning_on_radio")


def test_manifest_runs_omnigibson_in_a_native_worker():
    _, _, config = simulator("behavior")
    assert (config["environment"], config["native_environment"]) == ("behavior", "behavior-native")
    assert config["native_backend"] == "rosbridge_behavior.backend:R1ProSim"
    assert config["install"] == ["bddl3", "OmniGibson"]


def seed_dataset(seed, name, files, checksum=None):
    for path, data in files.items():
        (seed / "behavior/datasets" / name / path).parent.mkdir(parents=True, exist_ok=True)
        (seed / "behavior/datasets" / name / path).write_bytes(data)
    (seed / "behavior" / f"{name}.json").write_text(json.dumps({
        "revision": assets.REVISION, "archive": {"sha256": checksum or assets.ARCHIVES[name][2]},
        "files": {path: hashlib.sha256(data).hexdigest() for path, data in files.items()}}))


def instance_files(task="turning_on_radio", scene="house_double_floor_lower", count=assets.INSTANCES):
    folder = f"scene_test/public/{scene}/json/{scene}_task_{task}_instances"
    files = {f"{folder}/{scene}_task_{task}_0_{301 + i}_template-tro_state.json": b"{}"
             for i in range(count)}
    files["metadata/available_tasks.yaml"] = f"{task}:\n  0:\n    scene_model: {scene}\n".encode()
    return files


def test_seeded_prepare_adopts_receipt_verified_datasets_and_needs_the_license(tmp_path, monkeypatch):
    seed, runtime = tmp_path / "seed", tmp_path / "runtime/behavior"
    datasets = {"behavior-1k-assets": {"VERSION": b"3.9.0\n", "objects/a.usd": b"encrypted"},
                "omnigibson-robot-assets": {"VERSION": b"3.8.2\n", "models/r1pro/r1pro.yaml": b"x"},
                "2026-challenge-task-instances": instance_files()}
    for name, files in datasets.items():
        seed_dataset(seed, name, files)
    (seed / "behavior/datasets/omnigibson.key").write_bytes(b"key")
    monkeypatch.setenv("ROSBRIDGE_ASSET_SEED", str(seed))
    monkeypatch.delenv(assets.LICENSE, raising=False)
    store = AssetStore(tmp_path / "store")
    with pytest.raises(RuntimeError, match="not prepared"):
        assets.check_task(runtime, "turning_on_radio")
    with pytest.raises(ValueError, match="License Agreement"):
        assets.prepare(runtime, "turning_on_radio", store)
    monkeypatch.setenv(assets.LICENSE, "yes")
    (seed / "behavior/datasets/behavior-1k-assets/objects/a.usd").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its receipt"):
        assets.prepare(runtime, "turning_on_radio", store)
    (seed / "behavior/datasets/behavior-1k-assets/objects/a.usd").write_bytes(b"encrypted")
    assets.prepare(runtime, "turning_on_radio", store)
    root = runtime / "datasets"
    assert not (seed / "behavior/datasets/behavior-1k-assets").exists()
    assert (root / "behavior-1k-assets/objects/a.usd").read_bytes() == b"encrypted"
    assert (root / "omnigibson.key").stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(assets, "link_data", None)
    assets.prepare(runtime, "turning_on_radio", store)  # one preparation serves every task
    with pytest.raises(RuntimeError, match="not prepared"):
        assets.check_task(runtime, "picking_up_trash")
    *pinned, version = assets.ARCHIVES["omnigibson-robot-assets"]
    monkeypatch.setitem(assets.ARCHIVES, "omnigibson-robot-assets", (*pinned[:2], "0" * 64, version))
    with pytest.raises(RuntimeError, match="not prepared"):  # a new pin needs a new preparation
        assets.check_task(runtime, "turning_on_radio")


def test_seed_receipts_must_name_the_pinned_archive_and_every_file(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed_dataset(seed, "omnigibson-robot-assets", {"VERSION": b"3.8.2\n"}, checksum="0" * 64)
    monkeypatch.setenv("ROSBRIDGE_ASSET_SEED", str(seed))
    store = AssetStore(tmp_path / "store")
    with pytest.raises(ValueError, match="another archive"):
        assets.dataset(store, "omnigibson-robot-assets")
    seed_dataset(seed, "omnigibson-robot-assets", {"VERSION": b"3.8.2\n"})
    (seed / "behavior/datasets/omnigibson-robot-assets/stray").write_text("unpinned")
    with pytest.raises(ValueError, match="1 extra"):
        assets.dataset(store, "omnigibson-robot-assets")
    with pytest.raises(ValueError, match="never downloads"):
        assets.dataset(store, "behavior-1k-assets")


def fake_r1pro():
    """OmniGibson's R1Pro controller layout: base 3, trunk 4, arms 7, smooth grippers 1."""
    dofs = {"base": [0, 1, 5], "trunk": [6, 7, 8, 9], "arm_left": list(range(10, 17)),
            "gripper_left": [24, 25], "arm_right": list(range(17, 24)), "gripper_right": [26, 27]}
    actions, start = {}, 0
    for part in dofs:
        width = 3 if part == "base" else 1 if part.startswith("gripper") else len(dofs[part])
        actions[part], start = list(range(start, start + width)), start + width
    lower, upper = -np.ones(28), np.ones(28)
    lower[24:], upper[24:] = 0., .05
    return SimpleNamespace(controller_order=list(dofs), controller_joint_idx=dofs,
                           controller_action_idx=actions, joint_lower_limits=lower,
                           joint_upper_limits=upper, dof_names_ordered=[f"j{i}" for i in range(28)])


def test_r1pro_channels_follow_the_native_action_and_grippers_report_the_smooth_command():
    embodiment, robot = R1Pro.__new__(R1Pro), fake_r1pro()
    layout = embodiment.layout(robot)
    assert len(layout.joint_names) == 23 and layout.joint_names[14] == "left_gripper"
    groups = {g.name: g for g in layout.groups}
    assert groups["base"].velocity and (groups["base"].lower, groups["base"].upper) == ((-1.,) * 3, (1.,) * 3)
    assert groups["right_gripper"].scalar and groups["right_gripper"].indices == (22,)
    q = np.zeros(28)
    q[24:26], q[26:28] = .05, (0., .01)
    robot.get_joint_positions, robot.get_joint_velocities = lambda: q, lambda: np.full(28, .025)
    episode = SimpleNamespace(robot=robot, odometry=lambda: ((1., 2., .5), (.1, 0., .2)))
    positions, velocities = embodiment.state(episode)
    assert positions[[0, 1, 2]].tolist() == [1., 2., .5] and velocities[2] == .2
    assert positions[14] == pytest.approx(1.) and positions[22] == pytest.approx(-.8)
    assert velocities[14] == pytest.approx(1.)


def test_a_following_overview_turns_after_the_robot_the_short_way():
    sim = BehaviorSim.__new__(BehaviorSim)
    sim.period, sim.steps, sim._followed = .1, 0, None

    def heading(degrees):
        half = np.deg2rad(degrees) / 2
        return np.rad2deg(sim._heading([0., 0., np.sin(half), np.cos(half)]))
    assert heading(170) == pytest.approx(170)  # the first view faces the robot's heading
    sim.steps = 1
    assert 170 < heading(-170) < 172  # across ±180°, a tenth of a second into a one-second lag
    sim.steps = 10
    assert heading(-170) == pytest.approx(-170)  # after a gap, as after a cut, it starts over
