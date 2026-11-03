"""The Unitree G1 with SharpaWave hands as OmniGibson robot model g1_sharpa.

compose() replaces the G1 29-DoF rev 1.0 rubber hands with Sharpa's flange-mounted Wave hands
at the hands' palm mounts. Run as a module, this boots Isaac Sim, converts the composition with
OmniGibson's own robot import functions and writes the model directory: its USD with joint
armature, the hands' textured high-fidelity visuals and the URDF's colours elsewhere, the robot
definition and drives.json, the position-drive gains of every joint.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .sharpa import (SHARPA, SIDES, absolute_meshes, apply_drives, attach, axis_angle,  # noqa: F401
                     hand_drives, hand_joints, high_fidelity_hands, origin, paint)
from .sharpa import colors as link_colors

MODEL = "g1_sharpa"
TITLE = "the G1 with Sharpa hands"
G1 = Path("third_party/unitree_ros/robots/g1_description")
WBC = Path("third_party/decoupled_wbc")
LEGS = [f"{side}_{joint}_joint" for side in SIDES
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARMS = {side: [f"{side}_{joint}_joint" for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                                      "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        for side in SIDES}
# GR00T WholeBodyControl's body order: its lower body (legs, waist), then the arms.
BODY = LEGS + WAIST + ARMS["left"] + ARMS["right"]
# The armature of every body joint in GR00T's MuJoCo G1 (model_data/g1/g1_29dof_with_hand.xml).
BODY_ARMATURE = 0.01
CAMERA = "d435_link"
# The USD camera (-z forward, +y up) on the D435 mount (+x forward, +z up), as x, y, z, w.
CAMERA_ORIENTATION = [0.5, -0.5, -0.5, 0.5]
# Each hand's flange frame (fingers along +z, palm toward +x) in its wrist_yaw_link (forearm
# along +x) at the G1 hand mount: the palm faces the body's midline as the G1's own hands do,
# and the index finger leads on the thumb side.
MOUNTS = {
    "left": ((0.0415, 0.003, 0.), np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])),
    "right": ((0.0415, -0.003, 0.), np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])),
}


def compose(source):
    """The composed URDF text; source is the generated tree holding both robot descriptions."""
    source = Path(source)
    robot = ET.parse(source / G1 / "g1_29dof_rev_1_0.urdf").getroot()
    absolute_meshes(robot, lambda name: source / G1 / name)
    for element in list(robot):
        if element.tag == "mujoco" or element.get("name") in (
                *(f"{side}_hand_palm_joint" for side in SIDES), *(f"{side}_rubber_hand" for side in SIDES)):
            robot.remove(element)
    robot.set("name", MODEL)
    for side in SIDES:
        attach(robot, source, side, f"{side}_wrist_yaw_link", *MOUNTS[side])
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode") + "\n"


def colors(source):
    """Each link's URDF visual colour (rgb) in the composition."""
    return link_colors(ET.fromstring(compose(source)))


def drives(source):
    """Position-drive stiffness and damping per radian, armature and friction of every joint:
    GR00T's G1 motor gains, and Sharpa's Isaac calibration of each hand (Nm/deg in its USD)."""
    source = Path(source)
    config = yaml.safe_load((source / WBC / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml").read_text())
    result = {joint: {"stiffness": float(kp), "damping": float(kd), "armature": BODY_ARMATURE}
              for joint, kp, kd in zip(BODY, config["MOTOR_KP"], config["MOTOR_KD"], strict=True)}
    return {**result, **hand_drives(source)}


def arm_links(source):
    """The links of both arms and hands: everything below the shoulder pitch joints."""
    robot = ET.fromstring(compose(source))
    children = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint.find("child").get("link"))
    links, pending = set(), [f"{side}_shoulder_pitch_link" for side in SIDES]
    while pending:
        link = pending.pop()
        links.add(link)
        pending += children.get(link, [])
    return links | {f"{side}_eef_link" for side in SIDES}


def standing_height(source, angles):
    """The pelvis height at which the G1's lowest foot collision sphere touches the floor, with
    the leg joints at angles (name: radians)."""
    robot = ET.fromstring(compose(source))
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    lowest = np.inf
    for link in (f"{side}_ankle_roll_link" for side in SIDES):
        pose, child = np.eye(4), link
        while child in joints:
            joint = joints[child]
            local = origin(joint)
            if joint.get("type") == "revolute":
                axis = np.array(joint.find("axis").get("xyz").split(), dtype=float)
                local = local @ axis_angle(axis, angles.get(joint.get("name"), 0.))
            pose, child = local @ pose, joint.find("parent").get("link")
        for collision in robot.find(f"link[@name='{link}']").findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is not None:
                center = pose @ origin(collision)[:, 3]
                lowest = min(lowest, center[2] - float(sphere.get("radius")))
    return -lowest


def definition(source):
    """The OmniGibson robot definition: GR00T's lower body as the trunk, two arms and hands."""
    hands = {side: hand_joints(source, side) for side in SIDES}
    return {
        "raw_controller_order": ["trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"],
        # Zero lies within every joint's limits, whatever the native joint order; G1Sharpa sets
        # the standing posture by joint name.
        "default_joint_pos": [0.] * (len(BODY) + sum(len(hands[side][0]) for side in SIDES)),
        "default_controllers": {"trunk": "JointController", **{
            f"{part}_{side}": "JointController" for part in ("arm", "gripper") for side in SIDES}},
        "base_footprint_link_name": "pelvis",
        # A floating base: the legs walk, so there are no base joints.
        "locomotion": {"base_joint_names": [],
                       "floor_touching_base_link_names": [f"{side}_ankle_roll_link" for side in SIDES]},
        "articulated_trunk": {"trunk_joint_names": LEGS + WAIST},
        "manipulation": {
            "n_arms": 2, "arm_names": list(SIDES),
            "arm_link_names": {side: [joint.replace("_joint", "_link") for joint in ARMS[side]]
                               for side in SIDES},
            "arm_joint_names": ARMS,
            "eef_link_names": {side: f"{side}_eef_link" for side in SIDES},
            "finger_link_names": {side: hands[side][1] for side in SIDES},
            "finger_joint_names": {side: hands[side][0] for side in SIDES},
        },
    }


def main():
    """source, staging data root, output model directory: import inside Isaac Sim."""
    source, data, output = map(Path, sys.argv[1:4])
    from rosbridge.core.gpu import single_gpu

    os.environ.update(OMNIGIBSON_DATA_PATH=str(data), OMNIGIBSON_HEADLESS="1", OMNIGIBSON_GPU_ID="0",
                      OMNIGIBSON_RENDER_GPU=str(single_gpu()))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    staging = data / "staging"
    staging.mkdir(parents=True)
    urdf = staging / f"{MODEL}.urdf"
    urdf.write_text(compose(source))
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.examples.robots.import_custom_robot import add_sensor
    from omnigibson.utils.asset_conversion_utils import import_og_asset_from_urdf

    # Upstream import_custom_robot without its canonical-frame shift: Isaac Sim 5.1's URDF
    # importer keeps joints outside the link prims, so that shift would move the pelvis geometry
    # but not its joints. The pelvis stays the root frame; G1Sharpa stands it at its height.
    # Hulls of the visual meshes serve as collision where the URDF has no dedicated proxy.
    _, _, robot = import_og_asset_from_urdf(
        category="robot", model=MODEL, dataset_name="robot", urdf_path=str(urdf),
        collision_method="convex", overwrite=True, use_usda=True)
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()
    for side in SIDES:
        # z out of the fingertips, y across the fingers, between the palm and the fingers.
        add_sensor(stage, robot, "VisualSphere", f"{side}_eef_link", f"{side}_hand_C_MC",
                   [0.02, 0., 0.07], [0., 0., 0., 1.])
    add_sensor(stage, robot, "Camera", CAMERA, None, [0., 0., 0.], CAMERA_ORIENTATION)
    gains = drives(source)
    apply_drives(stage, gains)  # GR00T's and Sharpa's gains are torques per radian
    high_fidelity_hands(stage, robot, source)
    paint(stage, robot, colors(source))
    stage.Save()
    model = data / "robot/objects/robot" / MODEL
    (model / f"{MODEL}.yaml").write_text(yaml.safe_dump(definition(source), sort_keys=False))
    (model / "drives.json").write_text(json.dumps(gains, indent=2) + "\n")
    shutil.move(model, output)
    og.shutdown()


if __name__ == "__main__":
    main()
