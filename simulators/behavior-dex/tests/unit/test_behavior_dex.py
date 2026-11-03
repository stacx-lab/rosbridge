from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from rosbridge.catalog import simulator, tasks
from rosbridge_behavior import dex_model
from rosbridge_behavior.r1pro_sharpa import PARTS, R1ProSharpa
from rosbridge_behavior.sharpa import SHARPA, SIDES, origin

PROJECT = Path(__file__).resolve().parents[4]
GRIPPER_TOP = .0766


def test_catalog_runs_every_r1pro_activity_with_sharpa_hands():
    entries = list(tasks(name="behavior-dex").values())
    assert {t["robot"] for t in entries} == {"behavior-dex/behavior-r1pro-sharpa"}
    assert [t["native_task"] for t in entries] == [t["native_task"] for t in tasks(name="behavior").values()]
    assert all(t["id"] == "behavior-dex/" + t["native_task"].lower().replace("_", "-") for t in entries)


def test_manifest_shares_the_adapter_and_the_g1s_sharpa_source():
    _, _, config = simulator("behavior-dex")
    assert config["python_path"] == "simulators/behavior"
    assert config["native_backend"] == "rosbridge_behavior.backend:R1ProSharpaSim"
    assert (config["environment"], config["native_environment"]) == ("behavior-dex", "behavior-dex-native")
    sharpa = next(n for n in simulator("behavior-g1")[2]["nested_sources"] if n["path"] == "third_party/sharpa")
    assert config["nested_sources"] == [sharpa]  # one clone, one revision
    series = PROJECT / "patches/behavior-dex/series"
    patches = [line for line in series.read_text().splitlines() if line and not line.startswith("#")]
    assert all((series.parent / patch).is_file() for patch in patches) and len(patches) == 2


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sources(root):
    """An R1Pro torso with two arm_link7s carrying grippers and wrist cameras, and Sharpa hands
    with a thumb and an index finger."""
    links = "".join(f'''
  <link name="{side}_arm_link7"><visual><geometry><mesh filename="meshes/{side}_arm_link7.obj"/></geometry></visual></link>
  <joint name="{side}_arm_joint7" type="revolute"><parent link="torso_link4"/><child link="{side}_arm_link7"/>
    <axis xyz="1 0 0"/><limit lower="-1.5708" upper="1.5708" effort="18" velocity="10"/></joint>
  <link name="{side}_gripper_link"><visual><geometry><mesh filename="meshes/{side}_gripper_link.obj"/></geometry></visual></link>
  <joint name="{side}_gripper_joint" type="fixed"><origin xyz="-0.0295 0 -0.16065" rpy="0 0 0"/>
    <parent link="{side}_arm_link7"/><child link="{side}_gripper_link"/></joint>
  <link name="{side}_realsense_link"/>
  <joint name="{side}_realsense_joint" type="fixed"><origin xyz="0.05051 0.0028934 0.0051317" rpy="-3.1376 -0.43631 3.1399"/>
    <parent link="{side}_gripper_link"/><child link="{side}_realsense_link"/></joint>''' + "".join(f'''
  <link name="{side}_gripper_finger_link{i}"/>
  <joint name="{side}_gripper_finger_joint{i}" type="prismatic"><parent link="{side}_gripper_link"/>
    <child link="{side}_gripper_finger_link{i}"/><axis xyz="0 1 0"/><limit lower="0" upper="0.05" effort="100" velocity=".25"/></joint>'''
        for i in (1, 2)) for side in SIDES)
    urdf = root / dex_model.R1PRO / "urdf"
    write(urdf / "r1pro.urdf", f'<robot name="r1_pro_with_gripper"><link name="torso_link4"/>{links}</robot>')
    for side in SIDES:
        write(urdf / f"meshes/{side}_gripper_link.obj", f"v 0 0 0\nv 0.03 -0.05 {GRIPPER_TOP}\nvn 0 0 1\n")
    write(root / dex_model.R1PRO / "r1pro.yaml", yaml.safe_dump({
        "raw_controller_order": list(PARTS.values()),
        "default_controllers": {"base": "HolonomicBaseJointController", "trunk": "JointController",
                                **{f"{part}_{side}": "X" for part in ("arm", "gripper") for side in SIDES}},
        "disabled_collision_pairs": [["base_link", "wheel_motor_link1"],
                                     ["left_gripper_finger_link1", "left_realsense_link"]],
        "articulated_trunk": {"trunk_joint_names": [f"torso_joint{i}" for i in range(1, 5)]},
        "mobile_manipulation": {"tucked_default_joint_pos": [0.] * 28},
        "manipulation": {"arm_joint_names": {side: [f"{side}_arm_joint{i}" for i in range(1, 8)] for side in SIDES},
                         "finger_link_names": {side: [f"{side}_gripper_finger_link1"] for side in SIDES},
                         "finger_joint_names": {side: [f"{side}_gripper_finger_joint1"] for side in SIDES},
                         "gripper_link_names": {side: [f"{side}_gripper_link"] for side in SIDES}}}))
    for side in SIDES:
        package = f"{side}_sharpa_wave"
        write(root / SHARPA / package / f"{package}_with_flange.urdf", f'''<robot name="{package}">
  <link name="{side}_hand_flange"><visual><geometry><mesh filename="package://{package}/meshes/flange.STL"/></geometry>
    <material name=""><color rgba="0.1 0.1 0.1 1"/></material></visual></link>
  <link name="{side}_hand_C_MC"/>
  <link name="{side}_thumb_CMC_VL"/>
  <link name="{side}_index_PP"/>
  <joint name="{side}_hand_fix_joint" type="fixed"><parent link="{side}_hand_flange"/><child link="{side}_hand_C_MC"/></joint>
  <joint name="{side}_thumb_CMC_FE" type="revolute"><parent link="{side}_hand_C_MC"/><child link="{side}_thumb_CMC_VL"/>
    <axis xyz="0 1 0"/><limit lower="-0.1745" upper="1.9199" effort="2" velocity="16"/></joint>
  <joint name="{side}_index_MCP_FE" type="revolute"><parent link="{side}_hand_C_MC"/><child link="{side}_index_PP"/>
    <axis xyz="0 0 1"/><limit lower="0" upper="1.5" effort="2" velocity="16"/></joint>
</robot>''')
        write(root / SHARPA / package / f"{package}_with_flange.usda", f'''
    over "{side}_hand_C_MC" (
        prepend apiSchemas = ["PhysicsFilteredPairsAPI"]
    )
    {{
        rel physics:filteredPairs = [
            </{package}/{side}_thumb_CMC_VL>,
            </{package}/{side}_index_PP>,
        ]
    }}
    over "{side}_index_PP" (
        prepend apiSchemas = ["PhysicsFilteredPairsAPI"]
    )
    {{
        rel physics:filteredPairs = </{package}/{side}_thumb_CMC_VL>
    }}''')


def test_composition_replaces_each_gripper_with_a_hand_and_moves_the_wrist_camera_behind_it(tmp_path):
    sources(tmp_path)
    robot = ET.fromstring(dex_model.compose(tmp_path))
    names = {element.get("name") for element in robot}
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    assert robot.get("name") == "r1pro_sharpa"
    assert not any("gripper" in name for name in names)
    for side in SIDES:
        camera = joints[f"{side}_realsense_joint"]
        assert camera.find("parent").get("link") == f"{side}_arm_link7"
        mount = joints[f"{side}_hand_mount_joint"]
        # The USD camera sits on the back of the hand: behind the palm (-x of the flange),
        # looking along the fingers (+z) turned toward the palm side.
        view = np.linalg.inv(origin(mount)) @ origin(camera) @ dex_model.CAMERA_IN_LINK
        np.testing.assert_allclose(view[:3, 3], dex_model.CAMERA_POSITION, atol=1e-9)
        forward = -view[:3, 2]
        assert forward[2] > .95 and forward[0] > .1 and view[:3, 1] @ [-1, 0, 0] > .95
        assert (mount.find("parent").get("link"), mount.find("child").get("link")) == (
            f"{side}_arm_link7", f"{side}_hand_flange")
        pose = origin(mount)
        np.testing.assert_allclose(pose[:3, 3], [-.0295, 0, -.16065 + GRIPPER_TOP], atol=1e-9)
        np.testing.assert_allclose(pose[:3, :3], dex_model.MOUNT_ROTATIONS[side], atol=1e-9)
        assert {f"{side}_hand_C_MC", f"{side}_index_PP"} <= names
    # Fingers continue the gripper's approach (-z of arm_link7); palms face the midline.
    rotations = dex_model.MOUNT_ROTATIONS
    assert all((rotations[side][:, 2] == [0, 0, -1]).all() for side in SIDES)
    assert rotations["left"][:, 0] @ [0, 1, 0] < 0 < rotations["right"][:, 0] @ [0, 1, 0]
    assert all(np.isclose(np.linalg.det(rotations[side]), 1) for side in SIDES)
    meshes = {mesh.get("filename") for mesh in robot.iter("mesh")}
    assert str(tmp_path / dex_model.R1PRO / "urdf/meshes/left_arm_link7.obj") in meshes
    assert str(tmp_path / SHARPA / "right_sharpa_wave/meshes/flange.STL") in meshes


def test_the_hands_import_places_each_hand_at_its_arm_link7(tmp_path):
    sources(tmp_path)
    link7 = {"left": np.eye(4), "right": np.eye(4)}
    link7["left"][:3, 3], link7["right"][:3, 3] = (0., .25, .8), (0., -.25, .8)
    robot = ET.fromstring(dex_model.hands(tmp_path, link7))
    assert robot.get("name") == "r1pro_sharpa" and robot.find("link").get("name") == dex_model.ROOT
    for side in SIDES:
        mount = robot.find(f"joint[@name='{side}_hand_mount_joint']")
        assert mount.find("parent").get("link") == dex_model.ROOT
        np.testing.assert_allclose(origin(mount), link7[side] @ dex_model.mount(tmp_path, side), atol=1e-9)


def test_definition_drives_the_hands_by_joint(tmp_path):
    sources(tmp_path)
    robot = dex_model.definition(tmp_path)
    assert robot["default_controllers"]["gripper_left"] == "JointController"
    assert "mobile_manipulation" not in robot
    assert robot["default_joint_pos"] == [0.] * (6 + 4 + 14 + 2 * 2)
    manipulation = robot["manipulation"]
    assert manipulation["finger_joint_names"]["right"] == ["right_thumb_CMC_FE", "right_index_MCP_FE"]
    assert manipulation["finger_link_names"]["left"] == ["left_thumb_CMC_VL", "left_index_PP"]
    assert "left_realsense_link" in manipulation["gripper_link_names"]["left"]
    pairs = robot["disabled_collision_pairs"]
    assert ["base_link", "wheel_motor_link1"] in pairs and not any("gripper" in link for pair in pairs for link in pair)
    assert ["left_realsense_link", "left_thumb_CMC_VL"] in pairs
    assert ["right_arm_link7", "right_hand_C_MC"] in pairs
    # Sharpa's own filtered pairs, which the URDF import drops; the R1Pro keeps self-collisions.
    for side in SIDES:
        assert [f"{side}_hand_C_MC", f"{side}_thumb_CMC_VL"] in pairs and [f"{side}_hand_C_MC", f"{side}_index_PP"] in pairs
        assert [f"{side}_index_PP", f"{side}_thumb_CMC_VL"] in pairs


def fake_r1pro_sharpa(hand=3):
    """OmniGibson's controller layout with joint-controlled hands of `hand` joints each."""
    dofs = {"base": [0, 1, 5], "trunk": [6, 7, 8, 9], "arm_left": list(range(10, 17)),
            "gripper_left": list(range(24, 24 + hand)), "arm_right": list(range(17, 24)),
            "gripper_right": list(range(24 + hand, 24 + 2 * hand))}
    actions, start = {}, 0
    for part in dofs:
        width = 3 if part == "base" else len(dofs[part])
        actions[part], start = list(range(start, start + width)), start + width
    count = 24 + 2 * hand
    return SimpleNamespace(controller_order=list(dofs), controller_joint_idx=dofs, controller_action_idx=actions,
                           joint_lower_limits=-np.ones(count), joint_upper_limits=np.ones(count),
                           dof_names_ordered=[f"j{i}" for i in range(count)])


def test_hands_are_joint_devices_in_native_radians():
    embodiment, robot = R1ProSharpa.__new__(R1ProSharpa), fake_r1pro_sharpa()
    layout = embodiment.layout(robot)
    groups = {g.name: g for g in layout.groups}
    assert list(groups) == list(PARTS) and len(layout.joint_names) == 3 + 4 + 7 + 3 + 7 + 3
    assert not groups["left_hand"].scalar and groups["left_hand"].indices == (14, 15, 16)
    assert [layout.joint_names[i] for i in groups["right_hand"].indices] == ["j27", "j28", "j29"]
    q = np.arange(30, dtype=float)
    robot.get_joint_positions, robot.get_joint_velocities = lambda: q, lambda: -q
    episode = SimpleNamespace(robot=robot, odometry=lambda: ((1., 2., .5), (.1, 0., .2)))
    positions, velocities = embodiment.state(episode)
    assert positions[list(groups["right_hand"].indices)].tolist() == [27., 28., 29.]
    assert velocities[list(groups["left_hand"].indices)].tolist() == [-24., -25., -26.]
    command = np.linspace(-1, 1, len(layout.joint_names))
    np.testing.assert_allclose(embodiment.action(episode, command), command, atol=1e-7)


def test_link_poses_read_a_flat_usda_before_kit_boots(tmp_path):
    usda = tmp_path / "r1pro.usda"
    usda.write_text('''def Xform "r1pro"
{
    def Xform "left_arm_link6" (
    )
    {
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = (9, 9, 9)
    }
    def Xform "left_arm_link7" (
        apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        quatd xformOp:orient = (0.7071067811865476, 0, 0, 0.7071067811865476)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (-0.05, 0.252, 0.8772)
    }
}''')
    pose = dex_model.link_poses(usda, ["left_arm_link7"])["left_arm_link7"]
    np.testing.assert_allclose(pose[:3, 3], [-.05, .252, .8772])
    np.testing.assert_allclose(pose[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)  # 90° about z
