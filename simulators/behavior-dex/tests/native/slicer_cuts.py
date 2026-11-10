"""Run by test_slicing.py in a process of its own: slicers and food on the ground of an empty scene, stepped without
rendering as rosbridge steps an episode. A tool floats (no gravity) and is moved by its velocity alone, or falls.
Writes to argv[1], as JSON, what each trial did to the food and to the slicers' SlicerActive state:
  handle / blade   an axe, a carving knife and a table knife lowered onto a log, an egg and a cupcake: handle first,
                   then blade first at several speeds down to a slow press
  second           the table knife that just cut one cupcake lowered blade first onto another while a second table
                   knife floats active; then that second knife
  resting          the carving knife dropped flat onto the two halves of the egg it cut and left there 5 s; then lifted
  pruner           a pruner, its base fixed, closes by its joint's velocity alone on an asparagus stalk in its way
  plain            a cube that is a slicer without a blade meta link, lowered onto an egg"""
import json
import sys
import traceback

import numpy as np

from rosbridge_behavior.environment import boot, rotation, values

boot("behavior-dex")
import omnigibson as og  # noqa: E402
import torch as th  # noqa: E402
from omnigibson.object_states import SlicerActive  # noqa: E402
from omnigibson.utils import transform_utils as T  # noqa: E402
from omnigibson.utils.usd_utils import RigidContactAPI  # noqa: E402

FOOD = {"log": "pepele", "hard_boiled_egg": "xzucin", "cupcake": "mbhweg"}
COUNT = {"log": 4, "hard_boiled_egg": 6, "cupcake": 6}
TOOLS = {"axe": ("axe", "plqyos"), "carver": ("carving_knife", "fqqbop"), "knife": ("table_knife", "jxdfyy"),
         "other": ("table_knife", "jxdfyy")}
PAIRS = {"axe": "log", "carver": "hard_boiled_egg", "knife": "cupcake"}
SPEEDS = (.02, .3, 1.)  # m/s; the first is a slow press
PARK = 1.5  # tools wait this high

objects = [{"type": "DatasetObject", "name": name, "category": category, "model": model,
            "position": [-1. - i, 3., PARK]} for i, (name, (category, model)) in enumerate(TOOLS.items())]
objects.append({"type": "DatasetObject", "name": "pruner", "category": "pruner", "model": "lucqrs",
                "position": [0., 6., 1.], "fixed_base": True})
objects.append({"type": "DatasetObject", "name": "stalk", "category": "asparagus", "model": "npggjn",
                "position": [1., 6., .01]})
objects.append({"type": "PrimitiveObject", "name": "plain", "primitive_type": "Cube", "size": .04,
                "position": [-6., 3., PARK], "abilities": {"slicer": {}}})
for row, (category, model) in enumerate(FOOD.items()):
    objects += [{"type": "DatasetObject", "name": f"{category}_{i}", "category": category, "model": model,
                 "position": [.8 * i, .8 * row, .1]} for i in range(COUNT[category])]
env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": objects})
scene, dt = env.scene, og.sim.get_sim_step_dt()
report = {"blade rule": hasattr(SlicerActive, "disarm"), "errors": {}}


def get(name):
    return scene.object_registry("name", name)


def pose(prim):
    position, orientation = prim.get_position_orientation()
    return values(position), rotation(orientation)


def frame(local_a, local_b, world_a, world_b):
    """The rotation that takes the orthogonal local directions a and b to the world directions."""
    local = np.column_stack([local_a, local_b, np.cross(local_a, local_b)])
    world = np.column_stack([world_a, world_b, np.cross(world_a, world_b)])
    return world @ np.linalg.inv(local)


def put(obj, turn, local, world):
    """Sets obj at rest, turned by `turn`, with its point `local` at the world point."""
    obj.set_position_orientation(position=th.tensor(np.asarray(world) - turn @ local, dtype=th.float32),
                                 orientation=T.mat2quat(th.tensor(turn, dtype=th.float32)))
    obj.keep_still()


def step(obj=None, velocity=(0., 0., 0.)):
    """One step; obj, if given, moves at this velocity without turning."""
    if obj is not None:
        obj.set_linear_velocity(th.tensor(velocity, dtype=th.float32))
        obj.set_angular_velocity(th.zeros(3))
    og.sim.step()


def touching(a, b):
    return bool(RigidContactAPI.is_in_contact(scene_idx=scene.idx, query_set=[a], with_set=[b], ignore_set=None,
                                              current_only=False))


def active(tool):
    return bool(tool.states[SlicerActive].get_value())


def halves(name):
    return [obj for obj in scene.objects if obj.name.startswith(f"half_{name}_")]


def diced():
    return sum(int(system.n_particles) for name, system in scene.active_systems.items() if "diced__" in name)


def top(obj):
    low, high = (values(v) for v in obj.aabb)
    return np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2]])


def blade(tool, link=None):
    """The slicer meta link's box in the frame of `link` (the tool's root link by default) and the tool's directions
    from it: {centre, half, long, edge, flat, edge_mid, handle}. `long` runs along the blade from the handle, `edge`
    points out of the cutting edge, `flat` is the blade's normal, `edge_mid` the middle of the edge and `handle` a
    point of the handle 3 cm from its end, on its edge side."""
    origin, turn = pose(link or tool)
    meta = next(part for part in tool.links.values() if part.is_meta_link and part.meta_link_type == "slicer")
    mesh = next(iter(meta.visual_meshes.values()))
    transform = values(mesh.scaled_transform)
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    centre, axes = turn.T @ (transform[:3, 3] - origin), turn.T @ (transform[:3, :3] / scale)
    half = scale * float(mesh.get_attribute("size")) / 2
    order = np.argsort(-half)
    long, edge = axes[:, order[0]], axes[:, order[1]]
    box = {"centre": centre, "half": half[order]}
    if link is None:
        middle, size = values(tool.base_link_offset) * values(tool.scale), values(tool.native_bbox) * values(tool.scale)
        edge = edge * np.sign(edge @ (centre - middle))
        reach = np.abs(long) @ size / 2  # the tool's half length along the blade
        # The blade lies towards one end of the tool; the handle is the rest
        long = long * np.sign(long @ (centre - middle))
        end = middle - long * reach
        box.update(handle=end + long * .03 + edge * .01, length=2 * reach,
                   blade_from_handle_end=float((centre - end) @ long - half[order[0]]))
    box.update(long=long, edge=edge, flat=np.cross(long, edge), edge_mid=centre + edge * half[order[1]])
    return box


def lower(label, tool, local, name, speed, turn, hold=20, gap=None):
    """The tool, set `gap` above the food with its point `local` over the food's top, moves down at `speed` until the
    food is cut or `hold` steps after it first touched it."""
    food = get(name)
    gap = max(.03, speed * .1) if gap is None else gap
    put(tool, turn, local, top(food) + [0., 0., gap])
    for _ in range(3):
        step(tool)
    before, contact, cut = active(tool), None, None
    for k in range(int(gap / (speed * dt)) + hold + 30):
        step(tool, (0., 0., -speed))
        if get(name) is None:
            cut = k
            break
        if contact is None and touching(food, tool):
            contact = k
        if contact is not None and k - contact >= hold:
            break
    step(tool)
    result = {"speed": speed, "active before": before, "touched at": contact, "cut at": cut, "cut": cut is not None,
              "halves": len(halves(name)), "active after": active(tool)}
    report[label] = result
    print(label, json.dumps(result), flush=True)
    return result


def park(tool, at=None):
    tool.disable_gravity()
    put(tool, np.eye(3), np.zeros(3), pose(tool)[0] * [1., 1., 0.] + [0., 0., PARK] if at is None else at)


def rearm(tool, limit=90):
    """Steps until the floating tool is active again; returns the steps taken (None beyond `limit`)."""
    for k in range(limit + 1):
        if active(tool):
            return k
        step(tool)
    return None


def trial(label, run):
    try:
        run()
    except Exception:
        report["errors"][label] = traceback.format_exc()[-3000:]
        print("ERROR", label, report["errors"][label], flush=True)


tools = {name: get(name) for name in (*TOOLS, "plain", "pruner")}
ACROSS, DOWN = np.array([1., 0., 0.]), np.array([0., 0., -1.])
with og.sim.render_on_step(False):
    for tool in tools.values():
        tool.disable_gravity()
    for _ in range(30):  # the food settles on the ground
        for tool in tools.values():
            tool.keep_still()
        og.sim.step()
    blades = {name: blade(tools[name]) for name in TOOLS}
    report["blades"] = {name: {key: np.round(box[key], 4).tolist() for key in
                               ("centre", "half", "long", "edge", "edge_mid", "handle", "blade_from_handle_end")}
                        for name, box in blades.items()}
    report["start"] = {"active": {name: active(tool) for name, tool in tools.items()},
                       "food": {category: values(get(f"{category}_0").aabb_extent).round(3).tolist()
                                for category in FOOD}}
    print(json.dumps(report["blades"]), json.dumps(report["start"]), flush=True)

    def edge_down(name, along=ACROSS):
        return frame(blades[name]["long"], blades[name]["edge"], along, DOWN)

    # Food lies with its long side along y; a tool comes down across it, along x
    for category in FOOD:
        for i in range(COUNT[category]):
            food = get(f"{category}_{i}")
            size = values(food.native_bbox)
            turn = frame(np.eye(3)[int(np.argmax(size))], np.eye(3)[int(np.argmin(size))], [0., 1., 0.], [0., 0., 1.])
            put(food, turn, np.zeros(3), [.8 * i, .8 * list(FOOD).index(category), float(np.min(size)) / 2 + .01])
    for _ in range(30):
        og.sim.step()

    def handle_then_blade(name):
        tool, box, category = tools[name], blades[name], PAIRS[name]
        result = lower(f"{name} handle", tool, box["handle"], f"{category}_0", .1, edge_down(name))
        park(tool)
        result["rearm steps"] = rearm(tool)
        for i, speed in enumerate(SPEEDS):
            target = f"{category}_0" if i == 0 and not result["cut"] else f"{category}_{i + 1}"
            lower(f"{name} blade {speed}", tool, box["edge_mid"], target, speed, edge_down(name))["target"] = target
            park(tool)
            report[f"{name} blade {speed}"]["rearm steps"] = rearm(tool)

    for name in PAIRS:
        trial(name, lambda name=name: handle_then_blade(name))

    def second():
        """`knife` cuts one cupcake, then meets another within the two seconds while `other` floats active."""
        knife, other, box = tools["knife"], tools["other"], blades["knife"]
        park(other, at=[2., 2., PARK])
        rearm(knife), rearm(other)
        lower("second: first cupcake", knife, box["edge_mid"], "cupcake_4", .2, edge_down("knife"), hold=5)
        step(knife)
        result = lower("second: disarmed knife", knife, box["edge_mid"], "cupcake_5", .2, edge_down("knife"),
                       gap=.015)
        result["other active"] = active(other)
        park(knife)
        if not result["cut"]:
            lower("second: active knife", other, blades["other"]["edge_mid"], "cupcake_5", .2, edge_down("other"))
        park(other)

    trial("second", second)

    def resting():
        """The carving knife cuts an egg, falls flat onto the two halves and lies there; then it is lifted clear."""
        knife, box, name = tools["carver"], blades["carver"], "hard_boiled_egg_4"
        rearm(knife)
        lower("resting: cut", knife, box["edge_mid"], name, .3, edge_down("carver"), hold=5)
        step(knife)
        parts = halves(name)
        tops = [top(part) for part in parts]
        between = np.mean(tops, axis=0)
        apart = (tops[1] - tops[0]) * [1., 1., 0.]
        along = apart / np.linalg.norm(apart) if np.linalg.norm(apart) > 1e-3 else ACROSS
        put(knife, frame(box["long"], box["flat"], along, [0., 0., 1.]), box["centre"],
            [between[0], between[1], max(point[2] for point in tops) + .012])
        knife.enable_gravity()
        touched, was_active, left = [], [], []
        for _ in range(150):  # 5 s lying there
            og.sim.step()
            parts = halves(name)
            touched.append(any(touching(part, knife) for part in parts))
            was_active.append(active(knife))
            left.append((len(parts), diced()))
        knife.disable_gravity()
        put(knife, np.eye(3), np.zeros(3), pose(knife)[0] + [0., 0., .5])
        for _ in range(91):  # lifted clear
            if active(knife):
                break
            step(knife)
            touched.append(any(touching(part, knife) for part in halves(name)))
            was_active.append(active(knife))
        gaps = "".join("x" if touch else " " for touch in touched[:150]).split("x")
        last = max((k for k, touch in enumerate(touched) if touch), default=None)
        again = next((k for k, state in enumerate(was_active) if state), None)
        report["resting"] = result = {
            "seconds touching": round(sum(touched[:150]) * dt, 2), "longest gap s": round(max(map(len, gaps)) * dt, 2),
            "active while lying": any(was_active[:150]), "halves": min(count for count, _ in left),
            "diced": max(count for _, count in left), "halves after": len(halves(name)),
            "steps from the last touch to active": None if last is None or again is None else again - last}
        print("resting", json.dumps(result), flush=True)

    trial("resting", resting)

    def plain():
        tool = tools["plain"]
        rearm(tool)
        lower("plain", tool, np.array([0., 0., -.02]), "hard_boiled_egg_5", .1, np.eye(3))
        park(tool)

    trial("plain", plain)

    def pruner():
        """The pruner's blade link, open, closes by its joint's velocity alone on a stalk that floats in its way."""
        tool, stalk = tools["pruner"], get("stalk")
        joint, link = next(iter(tool.joints.values())), tool.links["blade"]
        closed, opened = float(joint.lower_limit), float(joint.upper_limit)
        local = blade(tool, link)  # the box in the frame of the link that carries it

        def box(q=None):
            """The blade box in the world: centre, edge direction and blade normal; with q, the joint is set there."""
            if q is not None:
                tool.set_joint_positions(th.tensor([q], dtype=th.float32))
                tool.keep_still()
                for _ in range(2):
                    og.sim.step()
            origin, turn = pose(link)
            return origin + turn @ local["centre"], turn @ local["edge"], turn @ local["flat"], turn @ local["long"]

        stalk.disable_gravity()
        size = values(stalk.native_bbox)
        thick = float(np.sort(size)[0])
        for part in (.4, .55, .7):  # how far open the blade is where it will meet the stalk
            centre, edge, flat, long = box(closed + part * (opened - closed))
            # The stalk lies across the blade, through its plane, its thin side towards the edge
            where = centre + edge * (local["half"][1] + thick / 2 + .002)
            turn = frame(np.eye(3)[int(np.argmax(size))], np.eye(3)[int(np.argmin(size))], flat, edge)
            start, edge_open, flat_open, long_open = box(opened)
            put(stalk, turn, np.zeros(3), where)
            for _ in range(3):
                step(stalk if get("stalk") is not None else None)
            if get("stalk") is None or not touching(stalk, tool):
                break
        # How far the stalk is from the box where the open blade holds it: beyond the rule's 5 mm margin
        offset = (where - start) @ np.column_stack([long_open, edge_open, flat_open])
        clear = np.linalg.norm(np.clip(np.abs(offset) - local["half"], 0., None)) - np.sort(size)[1] / 2
        result = {"opening": part, "cut while set": get("stalk") is None, "active before": active(tool),
                  "open box to stalk mm": round(1e3 * float(clear), 1), "cut at": None}
        for k in range(0 if result["cut while set"] else 60):
            tool.set_joint_velocities(th.tensor([-1.]))
            step(stalk)
            if get("stalk") is None:
                result["cut at"] = k
                break
        centre = box()[0]
        result.update({"cut": result["cut at"] is not None, "halves": len(halves("stalk")),
                       "joint": float(tool.get_joint_positions()[0]), "opened": opened,
                       "box moved mm": round(1e3 * float(np.linalg.norm(centre - start)), 1),
                       "active after": active(tool)})
        report["pruner"] = result
        print("pruner", json.dumps(result), flush=True)

    trial("pruner", pruner)

with open(sys.argv[1], "w") as file:
    json.dump(report, file, indent=1)
print("REPORT WRITTEN", flush=True)
og.shutdown()
