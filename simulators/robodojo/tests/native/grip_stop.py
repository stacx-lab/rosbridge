"""Run by test_robodojo.py in a process of its own: a gripper stop with an object in the pads, through the command
buffer the world node uses. The toothbrush is held between the open left pads at the home pose while the fingers
close, then the gripper is stopped with its measured opening and the arm holds still for 2 s.
Writes to argv[1], as JSON:
  held     the measured left_gripper opening once the fingers stall on the toothbrush
  target   the left_gripper command the buffer sends after the stop
  offset   the toothbrush's distance from the TCP (m) before the stop and 2 s after it
  drop     how far the toothbrush fell (m) in those 2 s"""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from rosbridge.core.commands import CommandBuffer
from rosbridge_robodojo.backend import RoboDojoSim

TCP = np.array([0.15007, -0.000002, -0.00144])  # the pad centre in the link6 frame, as the rig's profile has it


def numeric(value):
    return value.detach().cpu().numpy().astype(float) if hasattr(value, "detach") else np.asarray(value, float)


def like(template, value):
    if hasattr(template, "detach"):
        import torch
        return torch.tensor(value, dtype=template.dtype, device=template.device)
    return np.asarray(value, dtype=float)


sim = RoboDojoSim()
sim.reset("pack_objects_into_box", 0)
commands = CommandBuffer(sim.layout, sim.state()[0])
step = lambda: sim.step(commands.action(sim.time_s))
commands.set_group("left_gripper", [1.])
for _ in range(10):
    step()
layout = sim.env.scene_manager.layout_manager
brush = next(layout.get_scene_object(0, name) for name, kind in layout.instance_type_by_env[0].items()
             if kind == "rigid" and layout.get_scene_object(0, name).category_name == "electric_toothbrush")
points = brush.sample_mesh_vertices()[0]
position, quaternion = map(numeric, brush.get_world_pose())
body = (points - position) @ Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()
low, high = body.min(0), body.max(0)
order = np.argsort(high - low)  # thinnest across the pads, middle along the approach, longest across both
axes = np.zeros((3, 3))
axes[1, order[0]], axes[0, order[1]], axes[2, order[2]] = 1, 1, 1
if np.linalg.det(axes) < 0:
    axes[2] *= -1
rotation, flange = sim.link_pose("left", "link6")
pose = rotation @ axes
place = rotation @ TCP + flange - pose @ ((high + low) / 2)
x, y, z, w = Rotation.from_matrix(pose).as_quat()
p0, q0 = brush.get_world_pose()
commands.set_group("left_gripper", [0.])
for frame in range(40):
    if frame < 6:  # hold the toothbrush at the pads while the fingers close on it
        brush.set_world_pose(position=like(p0, place), orientation=like(q0, [w, x, y, z]))
        brush.set_linear_velocity(like(p0, [0, 0, 0]))
        brush.set_angular_velocity(like(p0, [0, 0, 0]))
    step()


def offset():
    rotation, flange = sim.link_pose("left", "link6")
    return float(np.linalg.norm(numeric(brush.get_world_pose()[0]) - (rotation @ TCP + flange)))


held, before, height = float(sim.state()[0][6]), offset(), float(numeric(brush.get_world_pose()[0])[2])
commands.stop("left_gripper", [held])  # what the world node's left_gripper/stop does
for _ in range(50):
    step()
result = {"held": held, "target": float(commands.action(sim.time_s)[6]), "offset": [before, offset()],
          "drop": height - float(numeric(brush.get_world_pose()[0])[2])}
with open(sys.argv[1], "w") as file:
    json.dump(result, file)
sim.close()
