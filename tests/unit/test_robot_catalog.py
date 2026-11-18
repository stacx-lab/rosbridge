from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rosbridge.catalog.robots import list_robots, setup


PROJECT = Path(__file__).resolve().parents[2]
# The operations rosbridge serves for each device type.
OPERATIONS = {
    "joint": {"joint_state": "sensor_msgs/msg/JointState",
              "joint_target": "trajectory_msgs/msg/JointTrajectory", "stop": "std_srvs/srv/Trigger"},
    "gripper": {"joint_state": "sensor_msgs/msg/JointState", "target": "std_msgs/msg/Float64",
                "stop": "std_srvs/srv/Trigger"},
    "locomotion": {"velocity": "geometry_msgs/msg/Twist", "odometry": "nav_msgs/msg/Odometry",
                   "stop": "std_srvs/srv/Trigger"},
    "rgb": {"image": "sensor_msgs/msg/Image", "camera_info": "sensor_msgs/msg/CameraInfo"},
    "rgbd": {"image": "sensor_msgs/msg/Image", "depth": "sensor_msgs/msg/Image",
             "camera_info": "sensor_msgs/msg/CameraInfo"},
}
# A joint device commanded by the pose of its one tip link serves pose_target in place of joint_target.
POSE = {"joint_state": "sensor_msgs/msg/JointState", "pose_state": "sensor_msgs/msg/MultiDOFJointState",
        "pose_target": "trajectory_msgs/msg/MultiDOFJointTrajectory", "stop": "std_srvs/srv/Trigger"}
# The world node also serves the geometry of a camera rendered through a lens.
GEOMETRY = {op: "custom_interfaces/srv/Json" for op in ("geometry", "unproject", "project")}
WORLD = {"time": ("/clock", "rosgraph_msgs/msg/Clock"),
         "wait": ("/world/wait", "custom_interfaces/srv/Wait"),
         "stop": ("/world/stop", "std_srvs/srv/Trigger")}


def manifest(simulator):
    import tomllib
    return tomllib.loads((PROJECT / "simulators" / simulator / "simulator.toml").read_text())


def task_of(robot):
    """The first catalog task on a robot: setup exports a task's workspace."""
    from rosbridge.catalog import tasks
    return next(task["id"] for task in tasks(PROJECT).values() if task["robot"] == robot)


def export(identifier, directory, **options):
    """A robot's workspace through its first task, without the simulator's task and seed changes."""
    with patch.dict("sys.modules", {"rclpy": None, "mujoco": None}), \
            patch("rosbridge.catalog.robots.changes", lambda *args: {}):
        result = setup(task_of(identifier), directory, project=PROJECT, **options)
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
        operations = POSE if "pose_target" in profile["operations"] else OPERATIONS.get(profile["type"])
        if operations is POSE:
            assert profile["type"] == "joint" and len(profile["model"]["tip_links"]) == 1
        lens = profile["type"] in ("rgb", "rgbd") and profile["color"]["distortion"]["model"] != "none"
        if lens:
            assert profile["color"]["distortion"]["model"] == "opencv_brown_conrady" and profile["color"]["fov"]
            operations = {**operations, **GEOMETRY}
        expected = WORLD if name == "world" else {
            op: (f"/{name}/{op}", kind) for op, kind in operations.items()}
        assert {op: (spec["endpoint"], spec["type"]) for op, spec in profile["operations"].items()} == expected
        assert all(isinstance(op["description"], str) and op["description"]
                   for op in profile["operations"].values())
        assert all("parent" in frame for frame in profile["frames"].values())
        # Every operation is an endpoint of rosbridge's world node; the only device nodes are URDF link publishers.
        nodes = [{"entry": "links.py", "factory": "create"}] if profile["type"] in ("joint", "gripper") else []
        assert profile["runtime"]["nodes"] == nodes
        for node in nodes:
            exported = root / "devices" / name / node["entry"]
            assert not exported.is_symlink() and "LinkPublisher(profile, context" in exported.read_text()
            if "model" in profile:
                assert (root / "devices" / name / profile["model"]["path"]).is_file()
    assert not any(root.rglob("*.pyc")) and not any(path.is_symlink() for path in root.rglob("*"))
    assert result["robot"] == identifier and result["task"] == task_of(identifier)
    with pytest.raises(ValueError, match="already contains"), \
            patch("rosbridge.catalog.robots.changes", lambda *args: {}):
        setup(task_of(identifier), tmp_path, project=PROJECT)
    with pytest.raises(ValueError, match="domain ID"):
        setup(task_of(identifier), tmp_path / "other", project=PROJECT, domain_id=-1)


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
        setup(task_of(identifier), tmp_path, project=project)
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


def test_setup_takes_the_launch_seed_and_names_its_task(tmp_path):
    from rosbridge.catalog import tasks
    seeded = next(task for task in tasks(PROJECT, "robotwin").values() if "seed" in task)
    result = setup(seeded["id"], tmp_path / "default", project=PROJECT)
    assert (result["task"], result["seed"], result["robot"]) == (seeded["id"], seeded["seed"], seeded["robot"])
    assert setup(seeded["id"], tmp_path / "chosen", seed=3, project=PROJECT)["seed"] == 3
    with pytest.raises(ValueError, match="Unknown task"):
        setup("robotwin/no-such-task", tmp_path / "unknown", project=PROJECT)


HANDSON = PROJECT / ".runtime/handson/handson/handson/evaluation/camera_contracts.py"


@pytest.mark.skipif(not HANDSON.is_file(), reason="HandsOn is not prepared")
def test_a_handson_workspace_holds_the_cameras_its_task_and_seed_render(tmp_path):
    rig = PROJECT / "catalog/robots/handson/handson-rokae-wuji/devices"
    nominal = {name: yaml.safe_load((rig / name / "profile.yaml").read_text()) for name in
               ("head_camera", "left_camera", "right_arm", "left_arm")}

    def profiles(task, seed):
        directory = tmp_path / f"{task}-{seed}"
        setup(f"handson/{task}", directory, seed=seed, project=PROJECT)
        return {name: yaml.safe_load((directory / ".kinex/devices" / name / "profile.yaml").read_text())
                for name in nominal}
    first, second = profiles("t24-place-ball-in-bowl", 0), profiles("t24-place-ball-in-bowl", 1)
    # Each seed draws its own mounts, a few millimetres from the nominal ones, on the same parents.
    for camera in ("head_camera", "left_camera"):
        assert first[camera]["frames"] != second[camera]["frames"] != nominal[camera]["frames"]
        assert first[camera]["frames"]["base"]["parent"] == nominal[camera]["frames"]["base"]["parent"]
        offset = [a - b for a, b in zip(first[camera]["frames"]["base"]["translation_m"],
                                        nominal[camera]["frames"]["base"]["translation_m"])]
        assert 0 < max(map(abs, offset)) < .005
        assert first[camera]["color"] == nominal[camera]["color"]
    # The head camera renders through its calibrated lens; the wrist cameras are undistorted pinholes.
    assert first["head_camera"]["color"]["distortion"]["model"] == "opencv_brown_conrady"
    assert first["left_camera"]["color"]["distortion"] == {"model": "none"}
    assert first["right_arm"] == second["right_arm"] == {**nominal["right_arm"], "loaded": False}
    # Upstream holds a single-arm task's left arm and hand at home in some tasks; there they take no commands.
    held = profiles("eval-gen01-cover-lunchbox", 0)
    assert list(held["left_arm"]["operations"]) == ["joint_state", "pose_state"]
    assert held["left_arm"]["operations"]["joint_state"] == nominal["left_arm"]["operations"]["joint_state"]
    assert held["right_arm"]["operations"] == nominal["right_arm"]["operations"]
    assert list(first["left_arm"]["operations"]) == ["joint_state", "pose_target", "pose_state", "stop"]
    # t05 renders its wrists through its own profile.
    assert profiles("t05-fill-cup", 0)["left_camera"]["color"]["projection"]["fx"] == pytest.approx(269.564, abs=1e-3)
    assert first["left_camera"]["color"]["projection"]["fx"] == pytest.approx(272.135, abs=1e-3)


@pytest.mark.skipif(not HANDSON.is_file(), reason="HandsOn is not prepared")
def test_every_handson_task_renders_the_lenses_of_the_catalog_rig():
    """A task's cameras may change their pinhole, never whether they have a lens: the lens decides the geometry
    operations of the rig's profile."""
    from rosbridge.catalog import resolve, tasks
    from rosbridge.catalog.robots import changes
    rig = PROJECT / "catalog/robots/handson/handson-rokae-wuji/devices"
    models = {name: yaml.safe_load((rig / name / "profile.yaml").read_text())["color"]["distortion"]
              for name in ("head_camera", "left_camera", "right_camera")}
    for identifier in tasks(PROJECT, "handson"):
        root, config, task = resolve(identifier, PROJECT)
        by_device = changes(root, config, task, 0)
        assert {name: by_device[name]["color"]["distortion"] for name in models} == models, identifier


def test_a_workspace_hook_changes_profiles_by_json_merge_patch():
    from rosbridge.catalog.robots import merge
    profile = {"operations": {"state": {"type": "a"}, "target": {"type": "b"}}, "frames": {"base": {"parent": None}}}
    merge(profile, {"operations": {"target": None, "state": {"type": "c"}}, "description": "held"})
    assert profile == {"operations": {"state": {"type": "c"}}, "frames": {"base": {"parent": None}},
                       "description": "held"}
