"""Measure the RoboDojo dual ARX X5 rig and write catalog/robots/robodojo/robodojo-arx-x5.

Chains come from the X5 URDF; mounts and cameras are measured in the live Isaac
scene. Runs in the native environment after `rosbridge prepare robodojo`:
    pixi run -e robodojo-native env PYTHONPATH=src:simulators/robodojo:.runtime/robodojo \
        python tools/robots/build_robodojo.py
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulators/robodojo"), str(Path(__file__).parent)]

import rigs  # noqa: E402
from rosbridge_robodojo.backend import RoboDojoSim, runtime  # noqa: E402

# RoboEnv's parallel-gripper geometry: the pad center ahead of link6, in the flange frame.
TCP = (0.15007, -0.000002, -0.00144)


def link(sim, side, name):
    robot = sim._actuators[side][0]
    pose = sim.env.robot_manager.get_link_pose(robot, name, [0])[0]  # x, y, z, qw, qx, qy, qz
    return rigs.transform(pose[:3], rigs.quat_matrix(pose[3:7]))


def main():
    sim = RoboDojoSim()
    sim.reset(sim.default_task, 0)
    x5 = runtime() / "Assets/Robots/x5"
    source = rigs.urdf_joints(x5 / "X5A.urdf")
    devices = {"world": rigs.world(
        "Scene reference and authoritative world clock of the RoboDojo table. Kinex supplies "
        "frame transforms independently. This device exposes no task instructions or "
        "evaluator results.")}
    files = {}
    for side in ("left", "right"):
        robot = sim._actuators[side][0]
        lower, upper = map(float, robot.gripper_scale)
        flange = link(sim, side, "link6")
        devices[f"{side}_arm"] = rigs.joint(
            f"{side}_arm",
            f"ARX X5 six-joint {side} arm of the RoboDojo rig. Geometry is the sibling arm.urdf "
            f"with flange (link6) as the tip link; measured links are on TF as {side}_arm/<link>, "
            f"and helpers.joint_model('{side}_arm') computes FK/IK from it. Commands are "
            "upstream's absolute joint positions ({side}_arm_joint_state of EvalEnv.take_action), "
            "interpolated by the native controller over each 0.04 s frame. Both arms use the "
            "native joint names joint1..joint6.".replace("{side}", side),
            [f"joint{i}" for i in range(1, 7)], [0.0] * 6,
            rigs.pose_frame("world:base", link(sim, side, "base_link")))
        devices[f"{side}_gripper"] = rigs.gripper(
            f"{side}_gripper",
            f"ARX X5 {side} parallel two-finger gripper on the {side}_arm flange. The grip joint "
            f"{side}_gripper is upstream's normalized ee value (0 closed, 1 open): finger targets "
            f"are {lower} + {upper - lower} m per unit, so a closed command squeezes past the "
            "finger stop. The TCP is the pad center, +x approaching. A closed or stalled grip "
            "alone does not prove a successful grasp.",
            f"{side}_gripper",
            f"Native normalized opening (data) from 0 (closed) to 1 (open), the "
            f"{side}_ee_joint_state of upstream's EvalEnv.take_action, which interpolates the "
            "fingers within each frame. A target persists until replaced or stopped; "
            "out-of-range targets are ignored, so verify joint_state.",
            rigs.frame(f"{side}_arm:flange", (0, 0, 0), (0, 0, 0, 1)),
            rigs.frame(f"{side}_gripper:base", TCP, (0, 0, 0, 1)))
        files[f"devices/{side}_arm/arm.urdf"] = rigs.source_arm(
            source, [f"joint{i}" for i in range(1, 7)], f"x5_{side}_arm",
            "RoboDojo ARX X5 arm from Assets/Robots/x5/X5A.urdf: base_link..link6; flange is "
            "the link6 frame.")
        files[f"devices/{side}_gripper/gripper.urdf"] = rigs.source_gripper(
            source, f"{side}_gripper", ["joint7", "joint8"], lower, upper, f"x5_{side}_gripper",
            f"RoboDojo ARX X5 {side} parallel gripper on the link6 flange. {side}_gripper is a "
            "geometry-less virtual drive joint carrying the native normalized opening (0 closed, "
            f"1 open); both fingers mimic it as {lower} + {upper - lower} m per unit, upstream's "
            "gripper_scale. Closed targets lie below the finger stop to squeeze an object.")
        position, quaternion = sim.camera_pose(f"{side}_camera")
        optical = rigs.transform(position, rigs.quat_matrix(quaternion))
        devices[f"{side}_camera"] = rigs.camera(
            f"{side}_camera",
            f"RGB camera (D435 model) on the {side} wrist, rendered by RoboDojo's tiled Isaac "
            f"renderer. Calibration and the optical-frame mount on the {side}_arm flange are in "
            "the profile.", sim.width, sim.height, sim.intrinsics(f"{side}_camera"),
            rigs.pose_frame(f"{side}_arm:flange", rigs.relative(flange, optical)))
    position, quaternion = sim.camera_pose("head_camera")
    devices["head_camera"] = rigs.camera(
        "head_camera",
        "Fixed RGB head camera (Gemini 345Lg model) above the table, rendered by RoboDojo's "
        "tiled Isaac renderer. Calibration and the optical-frame mount are in the profile.",
        sim.width, sim.height, sim.intrinsics("head_camera"),
        rigs.pose_frame("world:base", rigs.transform(position, rigs.quat_matrix(quaternion))))
    order = ["world", "left_arm", "left_gripper", "right_arm", "right_gripper",
             "head_camera", "left_camera", "right_camera"]
    rigs.write_rig(ROOT / "catalog/robots/robodojo/robodojo-arx-x5",
                   rigs.workspace("robodojo-arx-x5", "RoboDojo dual ARX X5", "arm", "ARX X5 (dual)"),
                   {name: devices[name] for name in order}, files)
    print("rig written", flush=True)
    np.set_printoptions(precision=4)
    sim.close()


if __name__ == "__main__":
    main()
