"""Measure the G1 with Sharpa hands and write catalog/robots/behavior-g1/behavior-g1-sharpa.

Chains come from the composed G1 and Sharpa URDF (rosbridge_behavior.g1_model); the base
footprint and the head camera mount are measured in the live OmniGibson scene. Runs in the
native environment after `rosbridge prepare behavior-g1`:
    pixi run -e behavior-g1-native env PYTHONPATH=src:simulators/behavior:.runtime/behavior-g1 \
        python tools/robots/build_behavior_g1.py
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulators/behavior"), str(Path(__file__).parent)]

import rigs  # noqa: E402
import sharpa_rig  # noqa: E402
from rosbridge.core.commands import VELOCITY_TIMEOUT_S  # noqa: E402
from rosbridge_behavior import g1_model  # noqa: E402
from rosbridge_behavior.backend import G1SharpaSim  # noqa: E402
from rosbridge_behavior.g1_sharpa import VELOCITY_LIMIT  # noqa: E402


def main():
    sim = G1SharpaSim()
    sim.reset(sim.default_task, 0)
    composed = ET.fromstring(g1_model.compose(ROOT / ".runtime/behavior-g1"))
    source = {joint.get("name"): joint for joint in composed.findall("joint")}
    home = sim.embodiment.pose  # GR00T's standing posture; other joints start at zero
    devices = {
        "world": rigs.world(
            "Simulation clock of the BEHAVIOR house. Furniture, objects and the activity's BDDL goal "
            "are not devices; the robot is not localized in this frame."),
        "base": rigs.locomotion(
            "base",
            "Unitree G1 legs walking on GR00T WholeBodyControl's balance and walking policies, which "
            "own the legs and waist at 50 Hz. Commands are a body twist (vx, vy in m/s, wz in rad/s) "
            f"within GR00T's navigation limits of {VELOCITY_LIMIT}; the gait balances the body while "
            "the arms and hands move. The base frame is the pelvis; odometry starts at the "
            "episode-start pose. A fall is not recovered.",
            {axis: (-VELOCITY_LIMIT, VELOCITY_LIMIT) for axis in ("vx", "vy", "wz")},
            {"parent": None, "translation_m": None, "rotation_xyzw": None},
            timeout_s=VELOCITY_TIMEOUT_S, units="si"),
        "torso": rigs.joint(
            "torso",
            "G1 three-joint waist (yaw, roll, pitch) on the pelvis. Geometry is the sibling "
            "torso.urdf with top (torso_link), where the arms and head camera mount, as its tip "
            "link. Targets set the torso orientation that GR00T's lower-body policy tracks while it "
            "balances, so verify the measured joint_state.",
            g1_model.WAIST, [home.get(joint, 0.) for joint in g1_model.WAIST],
            rigs.frame("base:base", (0, 0, 0), (0, 0, 0, 1)),
            part="waist", tip="top", model="torso.urdf"),
    }
    files = {"devices/torso/torso.urdf": rigs.source_arm(
        source, g1_model.WAIST, "g1_torso",
        "Unitree G1 waist from g1_29dof_rev_1_0.urdf: pelvis..torso_link; top is the torso_link frame.",
        tip="top")}
    for side in g1_model.SIDES:
        arm = g1_model.ARMS[side]
        devices[f"{side}_arm"] = rigs.joint(
            f"{side}_arm",
            f"G1 seven-joint {side} arm on the torso. Geometry is the sibling arm.urdf with flange "
            f"({side}_wrist_yaw_link) as the tip link; measured links are on TF as {side}_arm/<link>, "
            f"and helpers.joint_model('{side}_arm') computes FK/IK from it. Commands are absolute "
            "native joint positions on GR00T's arm gains.",
            arm, [home.get(joint, 0.) for joint in arm], rigs.frame("torso:top", (0, 0, 0), (0, 0, 0, 1)))
        files[f"devices/{side}_arm/arm.urdf"] = rigs.source_arm(
            source, arm, f"g1_{side}_arm",
            f"Unitree G1 {side} arm from g1_29dof_rev_1_0.urdf: torso_link..{side}_wrist_yaw_link; "
            "flange is the wrist_yaw_link frame.")
        devices[f"{side}_hand"], files[f"devices/{side}_hand/hand.urdf"] = sharpa_rig.hand(
            source, side, sim.embodiment.hands[side], home, "on the G1 wrist mount")
    sensor = sim._sensor("head_camera")
    devices["head_camera"] = rigs.camera(
        "head_camera",
        "RGB-D camera on the G1's head D435 mount, looking ahead and down, with the BEHAVIOR R1Pro "
        "head camera's optics. Image and registered depth calibration and the optical-frame mount "
        "are in the profile.",
        sim.intrinsics("head_camera"),
        rigs.pose_frame("torso:top", rigs.relative(sim.link_pose("torso_link"),
                                                   sim.camera_pose("head_camera"))),
        tuple(sensor.clipping_range))
    order = ["world", "base", "torso", "left_arm", "right_arm", "left_hand", "right_hand", "head_camera"]
    rigs.write_rig(ROOT / "catalog/robots/behavior-g1/behavior-g1-sharpa",
                   rigs.workspace("behavior-g1-sharpa", "BEHAVIOR G1 with Sharpa hands", "humanoid",
                                  "Unitree G1 (29 DoF) with SharpaWave hands"),
                   {name: devices[name] for name in order}, files)
    print("rig written", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
