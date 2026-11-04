from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from rosbridge.catalog import simulator, tasks
from rosbridge_behavior import g1_model
from rosbridge_behavior.g1_sharpa import (LOWER, G1Sharpa, attached_object, attachment, axis_rotation as R, digit,
                                        digits, grasp_break_n, tilt, torso_rpy)

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
    assert all((series.parent / patch).is_file() for patch in patches) and len(patches) == 3


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
    """The base twist and the pelvis height go to GR00T's lower-body policy; arm and hand targets go to their drives."""
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
    embodiment.floor = .05
    layout = embodiment.layout(robot)
    home = dict(zip(layout.joint_names, embodiment.home))
    assert home["vx"] == 0 and home["left_shoulder_roll_joint"] == .2 and home["right_finger3"] == 0
    assert home["pelvis_height"] == .74
    assert len(layout.joint_names) == 3 + 1 + 3 + 14 + 44
    groups = {g.name: g for g in layout.groups}
    assert groups["base"].velocity and groups["base"].upper == (.5, .5, .5)
    assert [layout.joint_names[i] for i in groups["torso"].indices] == g1_model.WAIST
    assert groups["pelvis"].indices == (3,) and (groups["pelvis"].lower, groups["pelvis"].upper) == ((.2,), (.78,))
    episode = SimpleNamespace(robot=robot, odometry=lambda: ((1., 2., .5), (.1, 0., .2)))
    positions, _ = embodiment.state(episode)
    assert positions[:3].tolist() == [1., 2., .5] and positions[3] == pytest.approx(.75)  # pelvis 0.8, floor 0.05
    assert positions[4] == names.index("waist_yaw_joint")
    assert layout.joint_state("pelvis")[0] == ["pelvis_height", "pelvis_pitch", "pelvis_roll"]
    assert layout.clip  # a target beyond a limit acts at the limit: a refusal is silent to a topic publisher
    assert len(positions) == len(layout.joint_names) + 2 and positions[-2:].tolist() == pytest.approx([0, 0])
    command = np.zeros(len(layout.joint_names))
    command[:4] = .2, 0., .1, .45
    command[list(groups["left_hand"].indices)] = .5
    command[list(groups["torso"].indices)] = .05, 0., .1  # yaw, roll, pitch: GR00T gets pitch and roll doubled
    native = embodiment.action(episode, command)
    assert native.shape == (15 + 7 + 22 + 7 + 22,)
    np.testing.assert_allclose(native[:15], np.arange(15) / 100., atol=1e-7)
    assert (native[22:44] == .5).all() and not native[44:].any()
    (_, velocity, height, rpy), (_, observation) = embodiment.policy.calls
    assert velocity == pytest.approx([.2, 0., .1]) and height == .45
    np.testing.assert_allclose(rpy, torso_rpy(.05, 0., .2), atol=1e-12)
    assert observation["q"].tolist() == [names.index(joint) for joint in g1_model.BODY]
    np.testing.assert_allclose(observation["floating_base_pose"][3:], [np.cos(.25), 0, 0, np.sin(.25)])
    np.testing.assert_allclose(observation["floating_base_vel"][3:], [0, 0, 2], atol=1e-12)
    np.testing.assert_allclose(observation["floating_base_vel"][:3], [np.cos(.5), -np.sin(.5), 0], atol=1e-12)


def test_the_g1_crate_names_its_usd_and_keeps_the_rest_of_the_definition():
    from rosbridge_behavior import g1_crate

    text = {"raw_controller_order": ["trunk", "arm_left"], "default_joint_pos": [0.0, 0.0],
            "manipulation": {"finger_joint_names": {"left": ["left_index_MCP_FE"]}}}
    crate = g1_crate.definition(text)
    assert next(iter(crate)) == "usd_path" and crate["usd_path"] == "models/g1_sharpa/usd/g1_sharpa.usd"
    assert {key: value for key, value in crate.items() if key != "usd_path"} == text
    assert yaml.safe_load(yaml.safe_dump(crate, sort_keys=False)) == crate
    with pytest.raises(ValueError, match="already names"):
        g1_crate.definition(crate)


def test_the_g1_pin_follows_the_import_and_the_converter(tmp_path):
    from rosbridge_behavior import assets

    (tmp_path / ".prepared.json").write_text('{"revision": "a"}')
    first = assets.crate_key(tmp_path)
    assert first != assets.model_key(tmp_path, "g1") and assets.pins(tmp_path, "g1")["g1"] == first
    (tmp_path / ".prepared.json").write_text('{"revision": "b"}')
    assert assets.crate_key(tmp_path) != first  # a new import makes a new crate


def test_the_pelvis_tilt_is_its_pitch_and_roll_from_the_yaw_only_base_frame():
    """R = Rz(yaw) Ry(pitch) Rx(roll): TF's level base frame, then the measured pelvis_pitch and pelvis_roll."""
    def turn(axis, angle):  # x, y, z, w
        return np.r_[np.eye(3)[axis] * np.sin(angle / 2), np.cos(angle / 2)]

    def product(a, b):
        (x1, y1, z1, w1), (x2, y2, z2, w2) = a, b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

    yaw, pitch, roll = .7, .45, -.03
    (p, r), _ = tilt(product(product(turn(2, yaw), turn(1, pitch)), turn(0, roll)), [0., 0., 0.])
    assert (p, r) == pytest.approx((pitch, roll))
    # Rates: turning about the pitched body's y axis changes pitch alone, about the world z axis yaw alone.
    frame, quaternion = R(2, yaw) @ R(1, pitch), product(turn(2, yaw), turn(1, pitch))
    assert tilt(quaternion, frame @ [0., .2, 0.])[1] == pytest.approx((.2, 0.))
    assert tilt(quaternion, [0., 0., .3])[1] == pytest.approx((0., 0.), abs=1e-12)


PROFILES = PROJECT / "catalog/robots/behavior-g1/behavior-g1-sharpa/devices"


def profile(device):
    return yaml.safe_load((PROFILES / device / "profile.yaml").read_text())


def test_the_g1_status_topics_are_the_string_outputs_its_profiles_declare():
    """Each status text the world publishes is a declared device output (Kinex loads a device only when every declared
    output has a publisher), and every declared String output is published."""
    robot = SimpleNamespace(grasping_mode="physical", _ag_obj_in_hand={"left": None, "right": None})
    embodiment = G1Sharpa.__new__(G1Sharpa)
    embodiment.attachments = 0
    keys = set(embodiment.status(SimpleNamespace(robot=robot)))
    declared = {f"{device}/{name}" for device in (p.name for p in PROFILES.iterdir())
                for name, op in profile(device)["operations"].items() if op["type"] == "std_msgs/msg/String"}
    assert keys == declared


def test_a_hand_reports_what_an_assisted_grasp_holds():
    def robot(mode, joint_type=None, released=False):
        held = object() if joint_type else None
        params = None if released or not joint_type else {"joint_type": joint_type}
        return SimpleNamespace(grasping_mode=mode, _ag_obj_in_hand={"right": held}, _ag_obj_constraint_params={"right": params})

    assert attachment(robot("assisted", "FixedJoint"), "right") == "held"
    assert attachment(robot("assisted", "SphericalJoint"), "right") == "tethered"
    assert attachment(robot("assisted", "FixedJoint", released=True), "right") == "none"
    assert attachment(robot("assisted"), "right") == "none"
    assert attachment(robot("physical"), "right") == "none"


def test_a_hand_names_what_it_is_attached_to():
    def robot(mode, link=None, released=False):
        obj = SimpleNamespace(name="door_tonlzz_0") if link else None
        params = None if released or not link else {"joint_type": "SphericalJoint", "target_obj": obj, "target_link_name": link}
        return SimpleNamespace(grasping_mode=mode, _ag_obj_in_hand={"right": obj}, _ag_obj_constraint_params={"right": params})

    assert attached_object(robot("assisted", "leaf"), "right") == "door_tonlzz_0/leaf"
    assert attached_object(robot("assisted", "leaf", released=True), "right") == "none"
    assert attached_object(robot("assisted"), "right") == "none"
    assert attached_object(robot("physical"), "right") == "none"


def test_an_assisted_grasp_counts_digits_not_finger_links():
    """A fingertip's pad and phalanx, or the thumb's bones, are one digit (such contacts used to attach); the palm
    is none. Every hand joint belongs to a digit, so every finger link does."""
    one = [["/World/robot_g1/right_thumb_MC", "/World/robot_g1/right_thumb_PP", "/World/robot_g1/right_thumb_elastomer"],
           ["right_index_MP", "right_index_DP", "right_index_elastomer"], ["right_thumb_PP", "right_hand_C_MC"]]
    assert [len(digits(links)) for links in one] == [1, 1, 1]
    assert len(digits(["right_thumb_DP", "right_index_elastomer"])) == 2  # a pinch
    assert len(digits(["left_ring_PP", "left_pinky_MP", "left_pinky_MC"])) == 2
    assert digit("right_hand_C_MC") is None and digit("right_eef_link") is None
    for side in ("left", "right"):
        assert all(digit(joint) for joint in profile(f"{side}_hand")["home"])


def test_the_grasp_break_force_is_the_rules_unless_the_environment_sets_one(monkeypatch):
    from rosbridge_behavior import g1_rules
    monkeypatch.delenv("ROSBRIDGE_GRASP_BREAK_N", raising=False)
    assert grasp_break_n() == g1_rules.GRASP_BREAK_N
    for value, expected in (("150", 150.), ("off", None), ("0", None)):
        monkeypatch.setenv("ROSBRIDGE_GRASP_BREAK_N", value)
        assert grasp_break_n() == expected


def test_a_fall_stops_the_legs_until_the_next_episode():
    """1 s of world time in all tilted past 45 degrees (or the pelvis below 0.15 m), counted since the body was last
    within 30 degrees of upright, is a fall: from then on the legs and waist hold the joint positions they had (GR00T
    is no longer consulted), status reads fallen and the episode's counts carry it; a shorter stumble changes
    nothing."""
    from rosbridge_behavior.g1_sharpa import FALL_HOLD_S
    hands = {side: [f"{side}_finger{i}" for i in range(22)] for side in g1_model.SIDES}
    names = [*g1_model.BODY, *hands["left"], *hands["right"]]
    pose = {"quaternion": [0., 0., 0., 1.], "z": .8, "q": np.zeros(len(names))}
    robot = SimpleNamespace(
        dof_names_ordered=names, joint_lower_limits=-np.ones(len(names)), joint_upper_limits=np.ones(len(names)),
        controller_order=["trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"],
        get_joint_positions=lambda: pose["q"], get_joint_velocities=lambda: np.zeros(len(names)),
        get_position_orientation=lambda: ([0., 0., pose["z"]], pose["quaternion"]),
        get_linear_velocity=lambda: [0., 0., 0.], get_angular_velocity=lambda: [0., 0., 0.],
        grasping_mode="physical", _ag_obj_in_hand={"left": None, "right": None})
    embodiment = G1Sharpa.__new__(G1Sharpa)
    embodiment.hands, embodiment.wbc, embodiment.policy, embodiment.pose = hands, {"height_cmd": .74}, Policy(), {}
    embodiment.floor = .05
    layout = embodiment.layout(robot)
    episode = SimpleNamespace(robot=robot)
    command = np.zeros(len(layout.joint_names))
    command[:4] = .3, 0., .2, .74
    lower = len(LOWER)

    def run(seconds, angle=0., z=.8):
        pose["quaternion"], pose["z"] = [0., np.sin(angle / 2), 0., np.cos(angle / 2)], z
        calls = len(embodiment.policy.calls)
        for _ in range(round(seconds * 50)):
            native = embodiment.action(episode, command)
            pose["q"] = pose["q"] + .01  # the body keeps moving
        return native[:lower], len(embodiment.policy.calls) > calls

    embodiment._start_episode()
    gr00t = np.arange(lower) / 100.
    assert np.allclose(run(.5, np.deg2rad(70))[0], gr00t)  # a stumble: GR00T drives the legs
    assert np.allclose(run(.5)[0], gr00t) and not embodiment.fallen  # upright again: the count starts over
    run(.6, np.deg2rad(50)), run(.3, np.deg2rad(40))  # between 30 and 45 degrees the count holds
    assert np.allclose(run(.3, np.deg2rad(50))[0], gr00t) and not embodiment.fallen
    run(.5), run(.5)
    assert np.allclose(run(FALL_HOLD_S - .1, np.deg2rad(65))[0], gr00t)
    held = pose["q"][[names.index(j) for j in g1_model.BODY[:lower]]] + .01 * 4  # positions when the fall registers
    legs, consulted = run(.1, np.deg2rad(65))
    assert embodiment.fallen and np.allclose(legs, held)
    legs, consulted = run(1.)  # upright again: still held, GR00T not consulted
    assert np.allclose(legs, held) and not consulted
    assert embodiment.status(episode)["base/fall"] == "fallen"
    assert embodiment.counts() == {"attachments": 0, "falls": 1, "fall_time_s": pytest.approx(4.2)}
    embodiment._start_episode()
    assert run(.2)[1] and embodiment.status(episode)["base/fall"] == "upright"
    run(FALL_HOLD_S, 0., z=.05 + .14)  # the pelvis 0.14 m above the floor
    assert embodiment.fallen and not run(.1)[1]
