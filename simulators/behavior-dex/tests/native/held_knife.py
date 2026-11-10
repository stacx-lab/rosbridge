"""Run by test_slicing.py in a process of its own: halve_an_egg with the carving knife held in the left Sharpa hand by
its fingers' contacts alone. The knife is set into the open hand, handle across the palm and edge down, and held
there while the fingers close; then nothing holds it but the hand. The chopping board, kept in place under the blade
as a table, carries the egg. The arm lowers the hand at 0.02 m/s until the blade is 5 mm above the board, and holds
it there for 2.5 s. Writes to argv[1], as JSON, how the grip held, when the egg was cut, and for how long the knife
touched the halves and when it was active again."""
import json
import sys

import numpy as np

from rosbridge_behavior.backend import R1ProSharpaSim
from rosbridge_behavior.environment import rotation, values

sim = R1ProSharpaSim()
sim.reset("halve_an_egg", 0)
import torch as th  # noqa: E402
from omnigibson.object_states import SlicerActive  # noqa: E402
from omnigibson.utils import transform_utils as T  # noqa: E402
from omnigibson.utils.usd_utils import RigidContactAPI  # noqa: E402

# The left arm's joints 1, 4 and 6 (its pitches) that hold the hand 0.15 m ahead of where it hangs at the evaluator's
# torso posture and this high above it (m), fingers down and thumb forward: solved on the model's URDF
REACH = {.115: (.2254, -1.20297, .97757), .09: (.1684, -1.04153, .87313), .065: (.09701, -.8526, .75559),
         .04: (.00006, -.61119, .61113)}
TOP, SPEED = .115, .02
# Finger flexion (rad) of a fist around the handle, by joint name ending; the thumb swings in front of the palm first
FIST = {"index_MCP_FE": 1.5, "middle_MCP_FE": 1.5, "ring_MCP_FE": 1.5, "pinky_MCP_FE": 1.5, "_PIP": 1.6, "_DIP": 1.2,
        "thumb_CMC_FE": 1.1, "thumb_MCP_FE": .7, "thumb_IP": .8}
# Where the handle's middle goes in the palm's frame: 2.5 cm out of the palm, 8 cm towards the fingertips
GRIP = np.array([.025, 0., .08])

episode = sim.episode
scene, robot, scope, dt = episode.env.scene, episode.robot, episode.env.task.object_scopes[0], sim.period
knife, board, egg = (getattr(scope[name], "wrapped_obj", scope[name]) for name in
                     ("carving_knife.n.01_1", "chopping_board.n.01_1", "hard-boiled_egg.n.01_1"))
groups = {group.name: group for group in sim.layout.groups}
names = list(sim.layout.joint_names)
command = np.array(sim.state()[0], float)
command[list(groups["base"].indices)] = 0.
palm = robot.links["left_hand_C_MC"]
pinned = {}
report = {"blade rule": hasattr(SlicerActive, "disarm")}


def pose(prim):
    position, orientation = prim.get_position_orientation()
    return values(position), rotation(orientation)


def frame(local_a, local_b, world_a, world_b):
    """The rotation that takes the orthogonal local directions a and b to the world directions."""
    local = np.column_stack([local_a, local_b, np.cross(local_a, local_b)])
    world = np.column_stack([world_a, world_b, np.cross(world_a, world_b)])
    return world @ np.linalg.inv(local)


def put(obj, turn, local, world):
    obj.set_position_orientation(position=th.tensor(np.asarray(world) - turn @ local, dtype=th.float32),
                                 orientation=T.mat2quat(th.tensor(turn, dtype=th.float32)))
    obj.keep_still()


def arm(height):
    """Sets the left arm's target for the hand this high above where it hangs."""
    heights = sorted(REACH)
    joints = [np.interp(height, heights, [REACH[h][i] for h in heights]) for i in range(3)]
    target = np.zeros(7)
    target[[0, 3, 5]] = joints
    command[list(groups["left_arm"].indices)] = target


def hand(flexion):
    indices = list(groups["left_hand"].indices)
    target = [next((angle for ending, angle in flexion.items() if names[i].endswith(ending)
                    and ("thumb" in names[i]) == ending.startswith("thumb")), 0.) for i in indices]
    command[indices] = np.clip(target, groups["left_hand"].lower, groups["left_hand"].upper)


def run(steps, each=None):
    for k in range(steps):
        for obj, (position, turn) in pinned.items():
            put(obj, turn, np.zeros(3), position)
        if each is not None:
            each(k)
        sim.step(command)


def blade():
    """The knife's blade box in its own frame: the middle of its edge, its directions and the middle of its handle."""
    origin, turn = pose(knife)
    meta = next(part for part in knife.links.values() if part.is_meta_link and part.meta_link_type == "slicer")
    mesh = next(iter(meta.visual_meshes.values()))
    transform = values(mesh.scaled_transform)
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    centre, axes = turn.T @ (transform[:3, 3] - origin), turn.T @ (transform[:3, :3] / scale)
    half = scale * float(mesh.get_attribute("size")) / 2
    order = np.argsort(-half)
    middle, size = values(knife.base_link_offset) * values(knife.scale), values(knife.native_bbox) * values(knife.scale)
    long = axes[:, order[0]] * np.sign(axes[:, order[0]] @ (centre - middle))
    edge = axes[:, order[1]] * np.sign(axes[:, order[1]] @ (centre - middle))
    end = middle - long * (np.abs(long) @ size / 2)  # the handle's end
    return {"long": long, "edge": edge, "edge_mid": centre + edge * half[order[1]],
            "handle": end + long * ((centre - end) @ long - half[order[0]]) / 2}


def in_palm():
    (origin, turn), (position, held) = pose(palm), pose(knife)
    return turn.T @ (position - origin), turn.T @ held


def edge():
    position, turn = pose(knife)
    return position + turn @ box["edge_mid"]


def halves():
    return [obj for obj in scene.objects if obj.name.startswith(f"half_{name}_")]


box, name = blade(), egg.name
start = command[list(groups["left_arm"].indices)].copy()
arm(TOP)
goal = command[list(groups["left_arm"].indices)].copy()
for k in range(60):
    command[list(groups["left_arm"].indices)] = start + (goal - start) * (k + 1) / 60
    sim.step(command)
# The knife in the palm's frame: its blade out of the thumb's side (-y), its edge towards the fingertips (down here)
turn_in_palm = frame(box["long"], box["edge"], [0., -1., 0.], [0., 0., 1.])


def hold(_):
    origin, turn = pose(palm)
    put(knife, turn @ turn_in_palm, box["handle"], origin + turn @ GRIP)


knife.disable_gravity()
put(knife, np.eye(3), np.zeros(3), pose(palm)[0] + [0., 0., .6])  # out of the way while the thumb swings in
hand({"thumb_CMC_FE": FIST["thumb_CMC_FE"]})
run(20, lambda _: knife.keep_still())
run(10, hold)
hand(FIST)
run(50, hold)
knife.enable_gravity()
gripped = in_palm()
run(45)  # 1.5 s in the hand alone
held = in_palm()
fingers = RigidContactAPI.get_contact_pairs(scene.idx, query_set=[knife], with_set=[robot], current_only=True)


def degrees(cosine):
    return round(float(np.degrees(np.arccos(np.clip(cosine, -1., 1.)))), 1)


report["grip"] = {"slip mm": round(1e3 * float(np.linalg.norm(held[0] - gripped[0])), 1),
                  "turn deg": degrees(((gripped[1].T @ held[1]).trace() - 1) / 2),
                  "links": sorted({path.split("/")[-1] for _, path in fingers}),
                  "edge down deg": degrees(-(pose(knife)[1] @ box["edge"])[2])}
print("grip", json.dumps(report["grip"]), flush=True)

# The table: the board kept in place so that the egg lying on it has its top 2.5 cm under the blade's edge
height, thick = float(values(egg.aabb_extent)[2]), float(np.sort(values(board.aabb_extent))[0])
tip = edge()
top = tip[2] - .025 - height
pinned[board] = (np.array([tip[0], tip[1], top - thick / 2]), pose(board)[1])
run(3)
put(egg, pose(egg)[1], np.zeros(3), [tip[0], tip[1], top + height / 2 + .003])
run(25)
low = max(min(REACH), TOP - (.025 + height - .005))
steps = int((TOP - low) / (SPEED * dt))
result = {"active before": bool(knife.states[SlicerActive].get_value()),
          "whole before": scene.object_registry("name", name) is not None, "cut at": None, "edge speed": 0.,
          "travel mm": round(1e3 * (TOP - low), 1)}
last, touched, was_active = edge(), [], []
for k in range(steps + 75):  # down, then 2.5 s at the bottom
    arm(max(low, TOP - SPEED * dt * (k + 1)))
    run(1)
    now = edge()
    if result["cut at"] is None:
        result["edge speed"] = max(result["edge speed"], round(float(np.linalg.norm(now - last)) / dt, 3))
        if scene.object_registry("name", name) is None:
            result.update({"cut at": k, "edge travel mm": round(1e3 * float(tip[2] - now[2]), 1)})
    last = now
    if result["cut at"] is not None:
        touched.append(any(RigidContactAPI.is_in_contact(scene_idx=scene.idx, query_set=[part], with_set=[knife],
                                                         ignore_set=None, current_only=False) for part in halves()))
        was_active.append(bool(knife.states[SlicerActive].get_value()))
slipped = in_palm()
final = max((k for k, touch in enumerate(touched) if touch), default=None)
again = next((k for k, state in enumerate(was_active) if state), None)
result.update({"cut": result["cut at"] is not None, "halves": len(halves()),
               "seconds touching the halves": round(sum(touched) * dt, 2),
               "seconds after the cut": round(len(touched) * dt, 2),
               "active while touching": any(a and b for a, b in zip(touched, was_active)),
               "steps from the last touch to active": None if final is None or again is None else again - final,
               "slip mm": round(1e3 * float(np.linalg.norm(slipped[0] - held[0])), 1),
               "q_score": float(sim.measures()["q_score"])})
report["press"] = result
print("press", json.dumps(result), flush=True)
with open(sys.argv[1], "w") as file:
    json.dump(report, file, indent=1)
print("REPORT WRITTEN", flush=True)
sim.close()
