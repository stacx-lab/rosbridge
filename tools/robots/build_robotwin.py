"""Measure the RoboTwin ALOHA AgileX rig and write catalog/robots/robotwin/robotwin-aloha-agilex.

Chains come from the embodiment URDF; mounts, wrist cameras and the TCP are
measured in the live SAPIEN scene. Run after `rosbridge prepare robotwin`:
    pixi run -e robotwin python tools/robots/build_robotwin.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulators/robotwin"), str(Path(__file__).parent)]

import rigs  # noqa: E402
from rosbridge_robotwin.backend import OPTICAL, RoboTwinSim, pose_matrix, runtime  # noqa: E402

PREFIX = {"left": "fl", "right": "fr"}


def arm_urdf(source, side):
    p = PREFIX[side]
    return rigs.source_arm(source, [f"{p}_joint{i}" for i in range(1, 7)], f"aloha_{side}_arm",
                           f"RoboTwin ALOHA AgileX {side} ARX5 arm from arx5_description_isaac.urdf: "
                           f"{p}_base_link..{p}_link6; flange is the {p}_link6 frame.")


def gripper_urdf(source, side, lower, upper):
    p, name = PREFIX[side], f"{side}_gripper"
    return rigs.source_gripper(
        source, name, [f"{p}_joint7", f"{p}_joint8"], lower, upper, f"aloha_{side}_gripper",
        f"RoboTwin ALOHA AgileX {side} parallel gripper on the {p}_link6 flange. {name} is a "
        f"geometry-less virtual drive joint carrying the native normalized opening (0 closed, "
        f"1 open); both fingers mimic it as {lower} + {upper - lower} m per unit, upstream's "
        f"gripper_scale. Closed targets lie below the finger stop to squeeze an object.")


def main():
    sim = RoboTwinSim()
    sim.reset(sim.default_task, 0)
    robot = sim.env.robot
    root = runtime()
    urdf = root / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
    source = rigs.urdf_joints(urdf)
    devices = {"world": rigs.world(
        "Scene reference and authoritative world clock of the RoboTwin 2.0 table. Kinex "
        "supplies frame transforms independently.")}
    files = {}
    sim.env._update_render()
    for side in ("left", "right"):
        p = PREFIX[side]
        entity = getattr(robot, f"{side}_entity")
        flange = pose_matrix(entity.find_link_by_name(f"{p}_link6").get_pose())
        base = pose_matrix(entity.find_link_by_name(f"{p}_base_link").get_pose())
        devices[f"{side}_arm"] = rigs.joint(
            f"{side}_arm",
            f"ARX5 six-joint {side} arm of the RoboTwin ALOHA AgileX. Geometry is the sibling "
            f"arm.urdf with flange ({p}_link6) as the tip link; measured links are on TF as "
            f"{side}_arm/<link>, and helpers.joint_model('{side}_arm') computes FK/IK from it. "
            "Commands are upstream's absolute joint positions (the arm part of its 14-D qpos "
            "action), applied as native drive targets once per 0.04 s frame. The native URDF "
            "limits are +/-10 rad placeholders.",
            [f"{p}_joint{i}" for i in range(1, 7)], [0.0] * 6, rigs.pose_frame("world:base", base))
        lower, upper = map(float, getattr(robot, f"{side}_gripper_scale"))
        tcp = getattr(robot, f"get_{side}_tcp_pose")()
        tcp = rigs.transform(tcp[:3], rigs.quat_matrix(tcp[3:]))
        devices[f"{side}_gripper"] = rigs.gripper(
            f"{side}_gripper",
            f"ALOHA AgileX {side} parallel two-finger gripper on the {side}_arm flange. The grip "
            f"joint {side}_gripper is upstream's normalized gripper value (0 closed, 1 open): "
            f"finger targets are {lower} + {upper - lower} m per unit, so a closed command "
            "squeezes past the finger stop. The TCP is upstream's tcp pose, gripper_bias ahead "
            "of the wrist joint with +x approaching. A closed or stalled grip alone does not "
            "prove a successful grasp.",
            f"{side}_gripper",
            "Native normalized opening (data) from 0 (closed) to 1 (open), the gripper entry of "
            "upstream's 14-D qpos action. The native setter runs on every 250 Hz substep and "
            "moves the drive target at most 0.1 per call toward it. A target persists until "
            "replaced or stopped; out-of-range targets are ignored, so verify joint_state.",
            rigs.frame(f"{side}_arm:flange", (0, 0, 0), (0, 0, 0, 1)),
            rigs.pose_frame(f"{side}_gripper:base", rigs.relative(flange, tcp)))
        files[f"devices/{side}_arm/arm.urdf"] = arm_urdf(source, side)
        files[f"devices/{side}_gripper/gripper.urdf"] = gripper_urdf(source, side, lower, upper)
        camera = sim._camera(f"{side}_camera")
        optical = pose_matrix(camera.get_global_pose())
        optical[:3, :3] = optical[:3, :3] @ OPTICAL
        devices[f"{side}_camera"] = rigs.camera(
            f"{side}_camera",
            f"RGB-D camera (D435 model) on the {side} wrist. Native ray-traced RGB with "
            "pixel-center depth from a raster twin; calibration and the optical-frame mount on "
            f"the {side}_arm flange are in the profile.", sim.intrinsics(f"{side}_camera"),
            rigs.pose_frame(f"{side}_arm:flange", rigs.relative(flange, optical)),
            (camera.get_near(), camera.get_far()))
    camera = sim._camera("head_camera")
    optical = pose_matrix(camera.get_global_pose())
    optical[:3, :3] = optical[:3, :3] @ OPTICAL
    devices["head_camera"] = rigs.camera(
        "head_camera",
        "Fixed RGB-D head camera (D435 model) above the table. Native ray-traced RGB with "
        "pixel-center depth from a raster twin; calibration and the optical-frame mount are in "
        "the profile.", sim.intrinsics("head_camera"),
        rigs.pose_frame("world:base", optical), (camera.get_near(), camera.get_far()))
    order = ["world", "left_arm", "left_gripper", "right_arm", "right_gripper",
             "head_camera", "left_camera", "right_camera"]
    rigs.write_rig(ROOT / "catalog/robots/robotwin/robotwin-aloha-agilex",
                   rigs.workspace("robotwin-aloha-agilex", "RoboTwin ALOHA AgileX", "arm",
                                  "ALOHA AgileX (dual ARX5)"),
                   {name: devices[name] for name in order}, files)
    sim.close()


if __name__ == "__main__":
    main()
