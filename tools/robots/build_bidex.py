"""Measure the three BiDex-100 rigs from MuJoCo and write catalog/robots/bidex/<rig>.

Each rig is measured on one of its tasks: the dual FR3v2 on the LIBERO tabletop and on the
dexjoco table, which differ only in the robot base pose, and the Astribot S1, whose torso and head
the 58-D action never moves; their servos hold the initial pose, which the mounts bake in. Run in
the BiDex-100 environment after `rosbridge prepare bidex`:
    pixi run -e bidex python tools/robots/build_bidex.py
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "simulators/bidex"), str(Path(__file__).parent)]

import rigs  # noqa: E402
from rosbridge_bidex.backend import CAMERAS, OPTICAL, BiDexSim  # noqa: E402

SIDES = ("right", "left")
RIGS = {
    "bidex-fr3v2-sharpa-tabletop": ("Generalization/put_apple_in_white_bowl", "on the LIBERO tabletop"),
    "bidex-fr3v2-sharpa-dexjoco": ("Instruction_Following/take_milk_and_lemon_from_shelf_into_tray",
                                   "at the dexjoco table"),
    "bidex-astribot-s1-sharpa": ("Generalization/pick_up_peach", "at the dexjoco table"),
}


def build(rig, task, place):
    from bidex.robots.registry import resolve_init_qpos

    sim = BiDexSim()
    sim.reset(task, 0)
    env, model, data = sim.env, sim.env.model, sim.env.data
    franka = env.robot_type == "dual_fr3v2_sharpa"
    robot = "dual Franka FR3v2" if franka else "Astribot S1"
    home = resolve_init_qpos(env.robot_type)
    near, far = model.vis.map.znear * model.stat.extent, model.vis.map.zfar * model.stat.extent
    groups = {group.name: group for group in sim.layout.groups}

    def joints(device):
        return [sim.layout.joint_names[i] for i in groups[device].indices]

    def world(body):
        return rigs.body_world(data, model.body(body).id)

    def optical(camera):
        cid = model.camera(camera).id
        return rigs.transform(data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3) @ OPTICAL)

    devices = {"world": rigs.world(
        f"Simulation clock of the BiDex-100 scene with the {robot} {place}. Scene objects are not "
        "devices. This device exposes no task instructions or evaluator results.")}
    files, flanges, bases = {}, {}, {}
    for side in SIDES:
        if franka:
            chain = [f"{side}_fr3v2_link{i}" for i in range(1, 9)]
            arm = f"Franka FR3v2 seven-joint {side} arm"
        else:
            chain = [f"astribot_arm_{side}_link_{i}" for i in range(1, 8)] + [f"astribot_arm_{side}_tool_link"]
            arm = f"Astribot S1 seven-joint {side} arm"
        base = model.body(int(model.body_parentid[model.body(chain[0]).id])).name
        bases[side], flanges[side] = world(base), world(chain[-1])
        devices[f"{side}_arm"] = rigs.joint(
            f"{side}_arm",
            f"{arm} of the BiDex-100 station. Geometry is the sibling arm.urdf with flange "
            f"({chain[-1]}, the body that carries the hand and the wrist camera) as the tip link; "
            f"measured links are on TF as {side}_arm/<link>, and helpers.joint_model('{side}_arm') "
            "computes FK/IK from it. Commands are absolute joint positions, BiDex-100's "
            "arm_joint_position, through the native position servos.",
            joints(f"{side}_arm"), [home.get(joint, 0.) for joint in joints(f"{side}_arm")],
            rigs.pose_frame("world:base", bases[side]))
        files[f"devices/{side}_arm/arm.urdf"] = rigs.mujoco_chain(
            model, chain, f"bidex_{side}_arm",
            f"BiDex-100 {arm} from MuJoCo {base}..{chain[-1]}; flange is the {chain[-1]} body frame.")
        hand = f"{side}_hand_C_MC"
        group = groups[f"{side}_hand"]
        devices[f"{side}_hand"] = rigs.joint(
            f"{side}_hand",
            f"Sharpa Wave 22-joint {side} hand on the {side}_arm flange, thumb first, then the index, "
            "middle, ring and little fingers, as BiDex-100's dex_joint_position. Geometry is the "
            f"sibling hand.urdf from the palm ({hand}) to the fingertips, with no single tip link; "
            "commands are absolute joint positions through the native position servos, and "
            "contact is physical. Joint limits are soft, so measured joints can read past their "
            "ranges; clip targets to the URDF limits, since rosbridge refuses one more than "
            "0.001 outside them.",
            joints(f"{side}_hand"), [float(np.clip(0., lo, hi)) for lo, hi in zip(group.lower, group.upper)],
            rigs.pose_frame(f"{side}_arm:flange", rigs.relative(flanges[side], world(hand))),
            part="hand", tip=None, model="hand.urdf")
        files[f"devices/{side}_hand/hand.urdf"] = rigs.mujoco_tree(
            model, hand, f"sharpa_{side}_hand",
            f"BiDex-100 Sharpa Wave {side} hand from MuJoCo: base_link is the {hand} body frame.")
    for name, camera in CAMERAS.items():
        side = name.split("_")[0]
        if side == "head":
            parent, frame = ("right_arm:base_link", bases["right"]) if franka else ("world:base", np.eye(4))
            where = ("above and behind the arms, looking down at the table" if franka else
                     "in the Astribot head, looking down at the table")
        else:
            parent, frame = f"{side}_arm:flange", flanges[side]
            where = f"on the {side} wrist, looking along the hand"
        devices[name] = rigs.camera(
            name,
            f"RGB-D camera {where} (BiDex-100's {camera} camera). Image and registered depth "
            "calibration and the optical-frame mount are in the profile; the depth near plane "
            "scales with each scene's extent.",
            sim.intrinsics(camera), rigs.pose_frame(parent, rigs.relative(frame, optical(camera))),
            (near, far))
    order = ["world", "right_arm", "left_arm", "right_hand", "left_hand", *CAMERAS]
    rigs.write_rig(ROOT / "catalog/robots/bidex" / rig,
                   rigs.workspace(rig, f"BiDex-100 {robot} with Sharpa hands {place}", "arm",
                                  f"{robot} (dual) with Sharpa Wave hands"),
                   {name: devices[name] for name in order}, files)
    sim.close()
    print(f"{rig} written", flush=True)


def main():
    for rig, (task, place) in RIGS.items():
        build(rig, task, place)


if __name__ == "__main__":
    main()
