from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rosbridge.catalog.robots import list_robots, setup


PROJECT = Path(__file__).resolve().parents[2]
# The operations rosbridge serves for each Kinex device type.
OPERATIONS = {
    "joint": {"joint_state": "sensor_msgs/msg/JointState",
              "joint_target": "trajectory_msgs/msg/JointTrajectory", "stop": "std_srvs/srv/Trigger"},
    "gripper": {"joint_state": "sensor_msgs/msg/JointState", "target": "std_msgs/msg/Float64",
                "stop": "std_srvs/srv/Trigger"},
    "locomotion": {"velocity": "geometry_msgs/msg/Twist", "odometry": "nav_msgs/msg/Odometry",
                   "stop": "std_srvs/srv/Trigger"},
    "rgb": {"image": "sensor_msgs/msg/Image", "camera_info": "sensor_msgs/msg/CameraInfo",
            "capture": "kinex_interfaces/srv/Capture"},
    "rgbd": {"image": "sensor_msgs/msg/Image", "depth": "sensor_msgs/msg/Image",
             "camera_info": "sensor_msgs/msg/CameraInfo",
             "capture": "kinex_interfaces/srv/CaptureRGBD"},
}
WORLD = {"time": ("/clock", "rosgraph_msgs/msg/Clock"),
         "wait": ("/world/wait", "kinex_interfaces/srv/Wait"),
         "stop": ("/world/stop", "std_srvs/srv/Trigger"),
         "reset": ("/world/reset", "std_srvs/srv/Trigger")}


def manifest(simulator):
    import tomllib
    return tomllib.loads((PROJECT / "simulators" / simulator / "simulator.toml").read_text())


def export(identifier, directory, **options):
    with patch.dict("sys.modules", {"rclpy": None, "mujoco": None}):
        result = setup(identifier, directory, project=PROJECT, **options)
    root = Path(directory) / ".kinex"
    profiles = {path.parent.name: yaml.safe_load(path.read_text())
                for path in root.glob("devices/*/profile.yaml")}
    return result, root, profiles


@pytest.mark.parametrize("identifier", list_robots(PROJECT))
def test_every_rig_exports_unloaded_profiles_on_rosbridge_endpoints(tmp_path, identifier):
    import json
    result, root, profiles = export(identifier, tmp_path, domain_id=83)
    assert not result["loaded"]
    assert json.loads((root / "kinex.json").read_text())["python"] == {"provider": "ros-jazzy"}
    kinds = manifest(identifier.split("/")[0])["devices"]
    assert kinds | {"world": "world"} == {
        name: profile["type"] for name, profile in profiles.items()}
    for name, profile in profiles.items():
        assert profile["name"] == name and profile["loaded"] is False
        assert profile["runtime"]["domain_id"] == 83
        expected = WORLD if name == "world" else {
            op: (f"/{name}/{op}", kind) for op, kind in OPERATIONS[profile["type"]].items()}
        assert {op: (spec["endpoint"], spec["type"]) for op, spec in profile["operations"].items()} == expected
        assert all(isinstance(op["description"], str) and op["description"]
                   for op in profile["operations"].values())
        assert all("parent" in frame for frame in profile["frames"].values())
        # Profiles borrow rosbridge's endpoints; only URDF link publishers run in Kinex.
        nodes = ([{"entry": "links.py", "factory": "create"}]
                 if profile["type"] in ("joint", "gripper") else [])
        assert profile["runtime"]["nodes"] == nodes
        for node in nodes:
            exported = root / "devices" / name / node["entry"]
            assert not exported.is_symlink() and "LinkPublisher(profile, context" in exported.read_text()
            assert (root / "devices" / name / profile["model"]["path"]).is_file()
    assert not any(root.rglob("*.pyc")) and not any(path.is_symlink() for path in root.rglob("*"))
    with pytest.raises(ValueError, match="already contains"):
        setup(identifier, tmp_path, project=PROJECT)
    with pytest.raises(ValueError, match="domain ID"):
        setup(identifier, tmp_path / "other", project=PROJECT, domain_id=-1)


def test_native_gripper_commands_drive_their_fingers_through_mimics(tmp_path):
    """Native gripper commands (normalized 0..1 openings, RoboCasa's integrated -1..1) live
    on a virtual joint spanning open to closed; the URDF fingers mimic it."""
    import xml.etree.ElementTree as ET
    for identifier in list_robots(PROJECT):
        _, root, profiles = export(identifier, tmp_path / identifier.replace("/", "-"))
        for name, profile in profiles.items():
            grip = profile.get("grip")
            if profile["type"] != "gripper" or (grip["open"], grip["closed"]) not in ((1., 0.), (-1., 1.)):
                continue
            robot = ET.parse(root / "devices" / name / profile["model"]["path"]).getroot()
            joints = {joint.get("name"): joint for joint in robot.iter("joint")}
            drive = joints[grip["joint"]]
            limit = drive.find("limit")
            assert drive.find("mimic") is None
            assert (float(limit.get("lower")), float(limit.get("upper"))) == (
                min(grip["open"], grip["closed"]), max(grip["open"], grip["closed"]))
            followers = [j for j in joints.values()
                         if j.find("mimic") is not None and j.find("mimic").get("joint") == grip["joint"]]
            assert len(followers) == 2 and all(j.get("type") == "prismatic" for j in followers)


@pytest.mark.parametrize("variant,base", [
    ("tabletop", [-.66, 0, .912]), ("kitchen", [-.66, 0, .912]),
    ("living-room", [-.51, 0, .42]), ("study", [-.75, 0, .912]),
    ("floor", [-.6, 0, 0]),
])
def test_rigs_carry_their_calibration_in_profiles(tmp_path, variant, base):
    project = Path(__file__).resolve().parents[2]
    identifier = f"libero/libero-panda-{variant}"
    with patch.dict("sys.modules", {"rclpy": None, "mujoco": None}):
        setup(identifier, tmp_path, project=project)
    root = tmp_path / ".kinex"
    assert yaml.safe_load((root / "workspace.yaml").read_text())["identity"]["id"] == identifier.split("/")[1]
    arm = yaml.safe_load((root / "devices/arm/profile.yaml").read_text())
    assert arm["frames"]["base"]["translation_m"] == base
    assert arm["frames"]["base"]["parent"] == "world:base"
    camera = yaml.safe_load((root / "devices/camera_agentview/profile.yaml").read_text())
    assert camera["frames"]["base"]["parent"] == "world:base"
    assert (root / "devices/arm/arm.urdf").is_file()


def test_every_task_names_an_exportable_rig_of_its_simulator():
    from rosbridge.catalog import tasks
    robots = list_robots(PROJECT)
    for task in tasks(PROJECT).values():
        assert task["robot"] in robots and task["robot"].split("/")[0] == task["simulator"], task["id"]


def test_every_libero_task_names_an_exportable_rig():
    from collections import Counter
    from rosbridge.catalog import tasks
    project = Path(__file__).resolve().parents[2]
    counts = Counter(task["robot"] for task in tasks(project, "libero").values())
    assert counts == {f"libero/libero-panda-{name}": count for name, count in
                      (("tabletop", 20), ("floor", 10), ("kitchen", 50),
                       ("living-room", 32), ("study", 18))}
    assert set(counts) <= set(list_robots(project))


def test_cli_filters_tasks_by_rig(capsys):
    import json
    from rosbridge.cli.main import main
    assert main(["tasks", "list", "--simulator", "libero", "--robot",
                 "libero/libero-panda-floor", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tasks"] == 10
    assert set(result["robots"].values()) == {"libero/libero-panda-floor"}
    assert main(["tasks", "list", "--robot", "libero/missing"]) == 1


def test_kinex_schema_and_readiness_accept_every_rig(tmp_path):
    """Kinex's own profile schema parses every export and its readiness plan completes.

    Kinex resolves URDFs in its own Python environment, a multi-gigabyte Pixi
    project it would otherwise provision in every workspace. Set
    KINEX_PYTHON_PROJECT to one prepared `.kinex/python` directory to share it.
    """
    import json
    import os
    import shutil
    import subprocess
    dist = PROJECT.parent / "kinex2.0/packages/core/dist"
    python = os.environ.get("KINEX_PYTHON_PROJECT")
    if not python or not shutil.which("node") or not (dist / "device/profile/store.js").is_file():
        pytest.skip("needs Node, a built Kinex core next to this checkout and KINEX_PYTHON_PROJECT")
    robots = list_robots(PROJECT)
    workspaces = [tmp_path / robot.replace("/", "-") for robot in robots]
    for robot, workspace in zip(robots, workspaces):
        export(robot, workspace)
        (workspace / ".kinex/python").symlink_to(Path(python).resolve(), target_is_directory=True)
    result = subprocess.run(["node", str(PROJECT / "tests/kinex_profiles.mjs"), str(dist), *map(str, workspaces)],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    for robot, plan in zip(robots, json.loads(result.stdout)):
        assert plan["robot"] == robot.split("/")[1]
        assert {item["status"] for item in plan["items"]} == {"done"}, (robot, plan["items"])
