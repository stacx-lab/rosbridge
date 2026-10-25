"""Measure the ABC YAM rig from MuJoCo and write catalog/robots/abc/abc-yam.

Run in the ABC environment after `rosbridge prepare abc`:
    pixi run -e abc python tools/robots/build_abc.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "simulators/abc"), str(ROOT / ".runtime/abc"), str(Path(__file__).parent)]

import mujoco  # noqa: E402
import rigs  # noqa: E402
from rosbridge_abc.backend import GRIPPER_SCALE, OPTICAL, AbcSim  # noqa: E402

HOME = {"left": [-0.20656902, 0.47283894, 0.99431604, -0.7043946, -0.30842298, -0.32864118],
        "right": [0.20160982, 0.39005876, 1.1182956, -0.8726253, 0.13332571, 0.42629892]}


def arm_urdf(model, side):
    return rigs.mujoco_chain(model, [f"{side}_link_{i}" for i in range(1, 7)], f"yam_{side}_arm",
                             f"ABC YAM {side} arm measured from MuJoCo {side}_arm..{side}_link_6; "
                             "flange is the link_6 body frame.")


def gripper_urdf(model, side):
    name = f"{side}_gripper"
    joints = [{"name": name, "type": "revolute", "parent": "base_link", "child": "drive",
               "axis": (0, 0, 1), "limits": (0, 1)}]
    for finger, sign in (("left", 1), ("right", -1)):
        body = model.body(f"{side}_link_{finger}_finger").id
        jid = rigs.body_joint(model, body)
        joints.append({"name": model.joint(jid).name, "type": "prismatic", "parent": "base_link",
                       "child": f"{finger}_finger", "origin": rigs.body_local(model, body),
                       "axis": model.jnt_axis[jid], "limits": model.jnt_range[jid],
                       "mimic": (name, sign * GRIPPER_SCALE, 0)})
    return rigs.urdf(
        f"yam_{side}_gripper",
        f"ABC YAM {side} parallel gripper on the link_6 flange. {name} is a geometry-less "
        f"virtual drive joint carrying the native normalized opening (0 closed, 1 open); "
        f"the sliding fingers mimic it at +/-{GRIPPER_SCALE} m per unit, as upstream's "
        f"position actuator does.",
        ["base_link", "drive", "left_finger", "right_finger"], joints)


def main():
    sim = AbcSim()
    sim.reset(sim.default_task, 0, False)
    model, data = sim.env.model, sim.env.data
    mujoco.mj_forward(model, data)
    near, far = (model.vis.map.znear * model.stat.extent, model.vis.map.zfar * model.stat.extent)

    def optical(camera):
        cid = model.camera(camera).id
        return rigs.transform(data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3) @ OPTICAL)

    devices = {"world": rigs.world(
        "Scene reference and authoritative world clock of the ABC bimanual YAM station. Kinex "
        "supplies frame transforms independently. This device exposes no task instructions or "
        "evaluator results.")}
    files = {}
    for side in ("left", "right"):
        flange = rigs.body_world(data, model.body(f"{side}_link_6").id)
        base = rigs.pose_frame("world:base", rigs.body_world(data, model.body(f"{side}_arm").id))
        devices[f"{side}_arm"] = rigs.joint(
            f"{side}_arm",
            f"YAM six-joint {side} arm of the ABC station. Geometry is the sibling arm.urdf with "
            f"flange (link_6) as the tip link; measured links are on TF as {side}_arm/<link>, and "
            f"helpers.joint_model('{side}_arm') computes FK/IK from it. Commands are the native "
            "absolute joint positions of upstream's 14-D policy action; rosbridge rejects "
            "targets outside the native joint ranges.",
            [f"{side}_joint{i}" for i in range(1, 7)], HOME[side], base)
        site = model.site(f"{side}_grasp_site").id
        tcp = rigs.transform(model.site_pos[site], rigs.quat_matrix(model.site_quat[site]))
        devices[f"{side}_gripper"] = rigs.gripper(
            f"{side}_gripper",
            f"YAM {side} parallel two-finger gripper on the {side}_arm flange. The grip joint "
            f"{side}_gripper is upstream's normalized policy gripper value (0 closed, 1 open); "
            f"each finger slides {GRIPPER_SCALE} m per unit, so the inner pad gap grows "
            "from about 0.8 mm to 95.8 mm. The TCP is the native grasp site between the pads; "
            "+z approaches. A closed or stalled grip alone does not prove a successful grasp.",
            f"{side}_gripper",
            f"Native normalized opening (data) from 0 (closed) to 1 (open), the gripper entry of "
            f"upstream's 14-D policy action; the finger position actuator tracks "
            f"{GRIPPER_SCALE} m per unit. A target persists until replaced or stopped; "
            "out-of-range targets are ignored, so verify joint_state.",
            rigs.frame(f"{side}_arm:flange", (0, 0, 0), (0, 0, 0, 1)),
            rigs.pose_frame(f"{side}_gripper:base", tcp))
        files[f"devices/{side}_arm/arm.urdf"] = arm_urdf(model, side)
        files[f"devices/{side}_gripper/gripper.urdf"] = gripper_urdf(model, side)
        devices[f"camera_{side}"] = rigs.camera(
            f"camera_{side}",
            f"RGB-D camera (D405 model) on the {side} wrist, tilted 50 degrees toward the "
            "fingers. Image and registered depth calibration and the optical-frame mount on the "
            f"{side}_arm flange are in the profile.", sim.width, sim.height,
            sim.intrinsics(side), rigs.pose_frame(f"{side}_arm:flange",
                                                  rigs.relative(flange, optical(side))),
            (near, far))
    devices["camera_top"] = rigs.camera(
        "camera_top",
        "Fixed RGB-D camera (D405 model) on the overhead gate, looking down at the table. Image "
        "and registered depth calibration and the optical-frame mount are in the profile.",
        sim.width, sim.height, sim.intrinsics("top"),
        rigs.pose_frame("world:base", optical("top")), (near, far))
    order = ["world", "left_arm", "left_gripper", "right_arm", "right_gripper",
             "camera_top", "camera_left", "camera_right"]
    rigs.write_rig(ROOT / "catalog/robots/abc/abc-yam",
                   rigs.workspace("abc-yam", "ABC bimanual YAM", "arm", "YAM"),
                   {name: devices[name] for name in order}, files)
    sim.close()


if __name__ == "__main__":
    main()
