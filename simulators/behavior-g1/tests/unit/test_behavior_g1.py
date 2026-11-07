from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rosbridge.catalog import simulator, tasks
from rosbridge_behavior import g1_model
from rosbridge_behavior.g1_sharpa import LOWER, G1Sharpa, axis_rotation as R, torso_rpy

PROJECT = Path(__file__).resolve().parents[4]


def test_catalog_runs_every_r1pro_activity_on_the_g1():
    entries = list(tasks(name="behavior-g1").values())
    assert {t["robot"] for t in entries} == {"behavior-g1/behavior-g1-sharpa"}
    assert [t["native_task"] for t in entries] == [t["native_task"] for t in tasks(name="behavior").values()]


def test_manifest_shares_the_adapter_and_pins_the_robot_sources():
    _, _, config = simulator("behavior-g1")
    assert config["python_path"] == "simulators/behavior"
    assert config["native_backend"] == "rosbridge_behavior.backend:G1SharpaSim"
    assert (config["environment"], config["native_environment"]) == ("behavior-g1", "behavior-g1-native")
    assert [(n["path"], len(n["source_revision"])) for n in config["nested_sources"]] == [
        ("third_party/unitree_ros", 40), ("third_party/sharpa", 40), ("third_party/decoupled_wbc", 40)]
    series = PROJECT / "patches/behavior-g1/series"
    patches = [line for line in series.read_text().splitlines() if line and not line.startswith("#")]
    shared = [line for line in (PROJECT / "patches/behavior/series").read_text().splitlines()
              if line and not line.startswith("#")]
    assert patches == [f"../behavior/{patch}" for patch in shared] + ["0003-wbc-metric-command.patch"]
    assert all((series.parent / patch).is_file() for patch in patches)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sources(root):
    """A pelvis with two wrists and rubber hands, and one-finger flange-mounted hands."""
    joints = "".join(f'''
  <link name="{side}_wrist_yaw_link"><visual><geometry><mesh filename="meshes/wrist.STL"/></geometry>
    <material name="dark"/></visual></link>
  <link name="{side}_rubber_hand"/>
  <joint name="{side}_wrist_yaw_joint" type="revolute"><parent link="pelvis"/><child link="{side}_wrist_yaw_link"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="22"/></joint>
  <joint name="{side}_hand_palm_joint" type="fixed"><parent link="{side}_wrist_yaw_link"/><child link="{side}_rubber_hand"/>
    <origin xyz="0.0415 0.003 0"/></joint>''' for side in g1_model.SIDES)
    write(root / g1_model.G1 / "g1_29dof_rev_1_0.urdf",
          f'<robot name="g1"><mujoco/><material name="dark"><color rgba="0.2 0.2 0.2 1"/></material>'
          f'<link name="pelvis"/>{joints}</robot>')
    for side in g1_model.SIDES:
        package = f"{side}_sharpa_wave"
        write(root / g1_model.SHARPA / package / f"{package}_with_flange.urdf", f'''<robot name="{package}">
  <link name="{side}_hand_flange"/>
  <link name="{side}_hand_C_MC"><visual><geometry><mesh filename="package://{package}/meshes/palm.STL"/></geometry>
    <material name=""><color rgba="0.79216 0.81961 0.93333 1"/></material></visual></link>
  <link name="{side}_index_PP"/>
  <joint name="{side}_hand_fix_joint" type="fixed"><parent link="{side}_hand_flange"/><child link="{side}_hand_C_MC"/></joint>
  <joint name="{side}_index_MCP_FE" type="revolute"><parent link="{side}_hand_C_MC"/><child link="{side}_index_PP"/>
    <axis xyz="0 0 1"/><limit lower="0" upper="1.5" effort="2" velocity="16"/></joint>
</robot>''')
        write(root / g1_model.SHARPA / package / f"{package}_with_flange.usda", f'''
        over "{side}_index_MCP_FE"
        {{
            float drive:angular:physics:damping = 0.003194
            float drive:angular:physics:stiffness = 0.083078
            float physxJoint:armature = 0.00265
            float physxJoint:jointFriction = 0.104
        }}''')
    write(root / g1_model.WBC / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml",
          f"MOTOR_KP: {list(range(29))}\nMOTOR_KD: {[1] * 29}\n")


def test_composition_mounts_sharpa_hands_on_the_wrists_in_place_of_the_rubber_hands(tmp_path):
    sources(tmp_path)
    robot = ET.fromstring(g1_model.compose(tmp_path))
    names = {element.get("name") for element in robot}
    assert robot.get("name") == "g1_sharpa" and robot.find("mujoco") is None
    assert not names & {"left_rubber_hand", "right_hand_palm_joint"} and "dark" in names
    assert all(Path(mesh.get("filename")).is_absolute() for mesh in robot.iter("mesh"))
    assert str(tmp_path / g1_model.SHARPA / "left_sharpa_wave/meshes/palm.STL") in {
        mesh.get("filename") for mesh in robot.iter("mesh")}
    for side in g1_model.SIDES:
        mount = robot.find(f"joint[@name='{side}_hand_mount_joint']")
        assert (mount.find("parent").get("link"), mount.find("child").get("link")) == (
            f"{side}_wrist_yaw_link", f"{side}_hand_flange")
        roll, pitch, yaw = map(float, mount.find("origin").get("rpy").split())
        np.testing.assert_allclose(R(2, yaw) @ R(1, pitch) @ R(0, roll), g1_model.MOUNTS[side][1], atol=1e-12)
    # Fingers point along the forearm; the palms (+x of the flange) face the midline.
    assert g1_model.MOUNTS["left"][1][:, 0] @ [0, 1, 0] < 0 < g1_model.MOUNTS["right"][1][:, 0] @ [0, 1, 0]


def test_links_wear_their_named_or_inline_urdf_colours(tmp_path):
    sources(tmp_path)
    colors = g1_model.colors(tmp_path)
    assert colors["left_wrist_yaw_link"] == (0.2, 0.2, 0.2)
    assert colors["right_hand_C_MC"] == (0.79216, 0.81961, 0.93333)
    assert "pelvis" not in colors


def test_drives_combine_gr00t_motor_gains_with_sharpas_isaac_calibration(tmp_path):
    sources(tmp_path)
    drives = g1_model.drives(tmp_path)
    assert drives["left_knee_joint"] == {"stiffness": 3., "damping": 1., "armature": g1_model.BODY_ARMATURE}
    assert drives["right_wrist_yaw_joint"]["stiffness"] == 28.
    finger = drives["right_index_MCP_FE"]
    assert finger["stiffness"] == pytest.approx(0.083078 * 180 / np.pi) and finger["armature"] == 0.00265
    assert finger["damping"] == pytest.approx(0.003194 * 180 / np.pi) and finger["friction"] == 0.104
    definition = g1_model.definition(tmp_path)
    assert definition["articulated_trunk"]["trunk_joint_names"] == LOWER
    assert definition["manipulation"]["finger_joint_names"]["left"] == ["left_index_MCP_FE"]


def test_waist_joints_give_the_torso_orientation_gr00t_tracks():
    np.testing.assert_allclose(torso_rpy(0, 0, 0), (0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(torso_rpy(.3, 0, 0), (0, 0, .3), atol=1e-12)
    roll, pitch, yaw = torso_rpy(.4, .2, -.3)
    np.testing.assert_allclose(R(2, yaw) @ R(1, pitch) @ R(0, roll), R(2, .4) @ R(0, .2) @ R(1, -.3), atol=1e-12)


class Policy:
    def __init__(self):
        self.calls = []

    def set_metric_command(self, velocity, height, rpy):
        self.calls.append(("command", list(velocity), height, rpy))

    def set_observation(self, observation):
        self.calls.append(("observation", observation))

    def get_action(self):
        return {"body_action": (np.arange(15) / 100., None, None)}


def test_g1_channels_command_the_lower_body_through_the_policy_and_the_rest_directly():
    hands = {side: [f"{side}_finger{i}" for i in range(22)] for side in g1_model.SIDES}
    names = [*g1_model.BODY, *hands["left"], *hands["right"]][::-1]  # any native dof order
    q = np.arange(len(names), dtype=float)
    robot = SimpleNamespace(
        dof_names_ordered=names, joint_lower_limits=-np.ones(len(names)), joint_upper_limits=np.ones(len(names)),
        controller_order=["trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"],
        get_joint_positions=lambda: q, get_joint_velocities=lambda: -q,
        get_position_orientation=lambda: ([0., 0., .8], [0., 0., np.sin(.25), np.cos(.25)]),
        get_linear_velocity=lambda: [1., 0., 0.], get_angular_velocity=lambda: [0., 0., 2.])
    embodiment = G1Sharpa.__new__(G1Sharpa)
    embodiment.hands, embodiment.wbc, embodiment.policy = hands, {"height_cmd": .74}, Policy()
    embodiment.pose = {"left_knee_joint": .3, "left_shoulder_roll_joint": .2}
    layout = embodiment.layout(robot)
    home = dict(zip(layout.joint_names, embodiment.home))
    assert home["vx"] == 0 and home["left_shoulder_roll_joint"] == .2 and home["right_finger3"] == 0
    assert len(layout.joint_names) == 3 + 3 + 14 + 44
    groups = {g.name: g for g in layout.groups}
    assert groups["base"].velocity and groups["base"].upper == (.5, .5, .5)
    assert [layout.joint_names[i] for i in groups["torso"].indices] == g1_model.WAIST
    episode = SimpleNamespace(robot=robot, odometry=lambda: ((1., 2., .5), (.1, 0., .2)))
    positions, _ = embodiment.state(episode)
    assert positions[3] == names.index("waist_yaw_joint") and positions[:3].tolist() == [1., 2., .5]
    command = np.zeros(len(layout.joint_names))
    command[:3] = .2, 0., .1
    command[list(groups["left_hand"].indices)] = .5
    native = embodiment.action(episode, command)
    assert native.shape == (15 + 7 + 22 + 7 + 22,)
    np.testing.assert_allclose(native[:15], np.arange(15) / 100., atol=1e-7)
    assert (native[22:44] == .5).all() and not native[44:].any()
    (_, velocity, height, rpy), (_, observation) = embodiment.policy.calls
    assert velocity == pytest.approx([.2, 0., .1]) and height == .74 and rpy == (0., 0., 0.)
    assert observation["q"].tolist() == [names.index(joint) for joint in g1_model.BODY]
    np.testing.assert_allclose(observation["floating_base_pose"][3:], [np.cos(.25), 0, 0, np.sin(.25)])
    np.testing.assert_allclose(observation["floating_base_vel"][3:], [0, 0, 2], atol=1e-12)
    np.testing.assert_allclose(observation["floating_base_vel"][:3], [np.cos(.5), -np.sin(.5), 0], atol=1e-12)
