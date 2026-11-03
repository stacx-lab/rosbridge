"""Measure BEHAVIOR's R1Pro and write catalog/robots/behavior/behavior-r1pro.

Chains come from the R1Pro URDF; the grasp frames and camera mounts are measured in the live
OmniGibson scene. Runs in the native environment after `rosbridge prepare behavior`:
    pixi run -e behavior-native env PYTHONPATH=src:simulators/behavior:.runtime/behavior \
        python tools/robots/build_behavior.py
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulators/behavior"), str(Path(__file__).parent)]

import rigs  # noqa: E402
from rosbridge.core.commands import VELOCITY_TIMEOUT_S  # noqa: E402
from rosbridge_behavior.backend import R1ProSim  # noqa: E402

SIDES = ("left", "right")


def mount(sim, parent, child):
    return rigs.relative(sim.link_pose(parent), child)


def body(sim, source, wrist="wrist, looking along the fingers"):
    """The R1Pro's devices and files but for its end effectors: world, base, torso, both arms
    and the three cameras, measured in sim with source the URDF's joints by name; wrist says
    where the wrist cameras sit."""
    robot = sim.episode.robot
    home = dict(zip(robot.dof_names_ordered, robot.reset_joint_pos.tolist()))  # the evaluator's reset posture
    devices = {
        "world": rigs.world(
            "Simulation clock of the BEHAVIOR house. Furniture, objects and the activity's BDDL goal "
            "are not devices; the mobile robot is not localized in this frame. This device exposes "
            "no task instructions or evaluator results."),
        "base": rigs.locomotion(
            "base",
            "R1Pro holonomic wheeled base carrying the torso. Commands are the official BEHAVIOR "
            "evaluator's normalized base action (vx, vy, wz in [-1, 1]), which its controller scales "
            "to 0.75 m/s and 1 rad/s in the base frame. The base frame is base_link at the floor; "
            "odometry starts at the episode-start pose.",
            {axis: (-1., 1.) for axis in ("vx", "vy", "wz")},
            {"parent": None, "translation_m": None, "rotation_xyzw": None},
            timeout_s=VELOCITY_TIMEOUT_S),
        "torso": rigs.joint(
            "torso",
            "R1Pro four-joint torso on the base. Geometry is the sibling torso.urdf with top "
            "(torso_link4), where the arms and head camera mount, as its tip link; commands are "
            "absolute native joint positions in radians.",
            [f"torso_joint{i}" for i in range(1, 5)], [home[f"torso_joint{i}"] for i in range(1, 5)],
            rigs.frame("base:base", (0, 0, 0), (0, 0, 0, 1)), part="torso", tip="top",
            model="torso.urdf"),
    }
    files = {"devices/torso/torso.urdf": rigs.source_arm(
        source, [f"torso_joint{i}" for i in range(1, 5)], "r1pro_torso",
        "BEHAVIOR R1Pro torso from models/r1pro/urdf/r1pro.urdf: base_link..torso_link4; top is "
        "the torso_link4 frame.", tip="top")}
    for side in SIDES:
        joints = [f"{side}_arm_joint{i}" for i in range(1, 8)]
        devices[f"{side}_arm"] = rigs.joint(
            f"{side}_arm",
            f"R1Pro seven-joint {side} arm on the torso. Geometry is the sibling arm.urdf with "
            f"flange ({side}_arm_link7) as the tip link; measured links are on TF as "
            f"{side}_arm/<link>, and helpers.joint_model('{side}_arm') computes FK/IK from it. "
            "Commands are absolute native joint positions, the evaluator's arm action.",
            joints, [home[joint] for joint in joints],
            rigs.pose_frame("torso:top", rigs.urdf_origin(source[f"{side}_arm_base_joint"])))
        files[f"devices/{side}_arm/arm.urdf"] = rigs.source_arm(
            source, joints, f"r1pro_{side}_arm",
            f"BEHAVIOR R1Pro {side} arm from r1pro.urdf: {side}_arm_base_link..{side}_arm_link7; "
            "flange is the link7 frame.")
    for name, parent, frame, where in (
            ("head_camera", "torso:top", "torso_link4", "on the head, looking ahead and down"),
            ("left_camera", "left_arm:flange", "left_arm_link7", f"on the left {wrist}"),
            ("right_camera", "right_arm:flange", "right_arm_link7", f"on the right {wrist}")):
        sensor = sim._sensor(name)
        devices[name] = rigs.camera(
            name,
            f"RGB-D camera {where} (BEHAVIOR's {sensor.name.split(':')[1]} sensor at the evaluator's "
            "full resolution). Image and registered depth calibration and the optical-frame mount "
            "are in the profile.",
            sim.intrinsics(name), rigs.pose_frame(parent, mount(sim, frame, sim.camera_pose(name))),
            tuple(sensor.clipping_range))
    return devices, files


def main():
    sim = R1ProSim()
    sim.reset(sim.default_task, 0)
    source = rigs.urdf_joints(sim.episode.robot.urdf_path)
    link = sim.link_pose
    devices, files = body(sim, source)
    # Both fingers of each gripper share one opening range.
    lower, upper = (float(source["left_gripper_finger_joint1"].find("limit").get(bound))
                    for bound in ("lower", "upper"))
    for side in SIDES:
        flange = link(f"{side}_arm_link7")
        hand = link(f"{side}_gripper_link")
        devices[f"{side}_gripper"] = rigs.gripper(
            f"{side}_gripper",
            f"R1Pro {side} parallel two-finger gripper on the {side}_arm flange. The grip joint "
            f"{side}_gripper is the evaluator's smooth gripper command (-1 closed, 1 open): both "
            f"fingers target {lower} + {(upper - lower) / 2} (v + 1) m. Grasping is assisted: an "
            "object between closing fingers is held by a constraint. The TCP is the native eef "
            "frame between the fingertips, +z approaching. A closed grip alone does not prove a "
            "grasp.",
            f"{side}_gripper",
            "Smooth native gripper command (data) from -1 (closed) to 1 (open), held until "
            "replaced or stopped; out-of-range targets are ignored, so verify joint_state.",
            rigs.pose_frame(f"{side}_arm:flange", rigs.relative(flange, hand)),
            rigs.pose_frame(f"{side}_gripper:base", mount(sim, f"{side}_gripper_link", link(f"{side}_eef_link"))),
            state="The smooth command the mean finger opening reaches, -1 (closed) to 1 (open), "
                  "stamped at acquisition; Kinex publishes the URDF finger links from it.",
            hold="Hold the commanded opening, keeping any grasp.", opened=1.0, closed=-1.0)
        files[f"devices/{side}_gripper/gripper.urdf"] = rigs.source_gripper(
            source, f"{side}_gripper", [f"{side}_gripper_finger_joint{i}" for i in (1, 2)], lower, upper,
            f"r1pro_{side}_gripper",
            f"BEHAVIOR R1Pro {side} gripper on the link7 flange. {side}_gripper is a geometry-less "
            "virtual drive joint carrying the evaluator's smooth command (-1 closed, 1 open); both "
            f"fingers mimic it from {lower} to {upper} m.", limits=(-1, 1))
    order = ["world", "base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper",
             "head_camera", "left_camera", "right_camera"]
    rigs.write_rig(ROOT / "catalog/robots/behavior/behavior-r1pro",
                   rigs.workspace("behavior-r1pro", "BEHAVIOR R1Pro", "mobile",
                                  "Galaxea R1 Pro (wheeled dual-arm mobile manipulator)"),
                   {name: devices[name] for name in order}, files)
    print("rig written", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
