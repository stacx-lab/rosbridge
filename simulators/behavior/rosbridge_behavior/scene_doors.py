"""The room doors of a loaded BEHAVIOR scene, read from the scene itself: which way each swings, and which open onto no
floor (a room the activity does not load, or the outside), whose doorways get an invisible collision wall. BEHAVIOR's
evaluation loads only the activity's rooms (load_room_instances), so a door of a loaded room can open onto a room that
is not there: a robot that walks through falls out of the world (three workers did). Per door leaf (the child link of
each revolute or prismatic joint of a door, sliding door or garage door) the leaf is taken closed (its collision points
turned back about the joint by the joint's position), and rays straight down at points on each side of it, 0.3, 0.6 and
1.0 m from the leaf at its middle and a quarter of its width either way, look for floor (a collision surface within
FLOOR_BAND of the leaf's bottom; the door and the robot are ignored). A leaf with floor on one side (FLOOR_MIN_POINTS of
the points) and no floor at any point of the other, nor a lower one (a step), is a boundary door: a static, invisible
collision box fills its doorway, in the plane of the closed leaf and thinner than it, so the robot cannot pass while the
leaf still turns or slides through the box (the pair is filtered); the door itself is untouched. A box is made only when
the robot comes near its doorway, and removed when the next episode starts (Guard).
The world's text gets one line per door with its position in the robot's episode-start frame (base/odom), which way it
opens and whether it is blocked (describe); the facts are written to the run directory (write)."""
import json
import math
import os
from pathlib import Path

import numpy as np

DOOR_CATEGORIES = ("door", "sliding_door", "garage_door")
SIDE_DISTANCES_M = (.3, .6, 1.)  # from the leaf's face
SIDE_FRACTIONS = (-.25, 0., .25)  # of the leaf's width, from its middle
FLOOR_BAND_M = (-.2, .25)  # a floor's top relative to the bottom of the leaf
FLOOR_MIN_POINTS = 5  # of the 9 points on a side, for "floor on this side"
RAY_TOP_M, RAY_BOTTOM_M = 2., -.5  # rays run between these heights above the leaf's bottom
BLOCKER_THICKNESS_M = .01
BLOCKER_ROOT = "/World/rosbridge_scene_edge"
UPRIGHT = .9  # |axis z| of an upright hinge, or at most 1 - UPRIGHT for a horizontal slide
LEAF_MIN_M = (.4, 1.)  # width and height of a door leaf; smaller jointed parts of a door (a handle) are not leaves
# A doorway's wall exists only once the robot's pelvis has come within this distance of the doorway (it walks at most
# ~0.35 m/s): a static collider anywhere in the scene (even with collision off) changes PhysX's pair order, which moved a
# knife resting on a board 15 m away by 1e-6 m at the start of canning_food (12 walls); without walls nothing changes.
GUARD_M = 1.5


def _values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _rotation(quaternion_xyzw):
    x, y, z, w = _values(quaternion_xyzw) / np.linalg.norm(_values(quaternion_xyzw))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def turn(axis, angle):
    """The rotation by angle (rad) about the unit axis."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def leaf(obj, joint):
    """A door joint's world axis and pivot and its child link's collision points (world), from the physics pose and the
    joint's USD frame, as furniture_friction reads them."""
    from pxr import Usd, UsdGeom

    link = obj.links[str(joint.body1).split("/")[-1]]
    position, orientation = (_values(x) for x in link.get_position_orientation())
    usd = np.asarray(UsdGeom.Xformable(link.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = _rotation(orientation) * np.linalg.norm(usd[:3, :3], axis=0), position
    prim = joint.prim
    rotation1 = prim.GetAttribute("physics:localRot1").Get()
    w, xyz = rotation1.GetReal(), np.asarray(rotation1.GetImaginary(), float)
    local = np.eye(3)["XYZ".index(prim.GetAttribute("physics:axis").Get() or "X")]  # USD's default axis is X
    local = local + 2 * w * np.cross(xyz, local) + 2 * np.cross(xyz, np.cross(xyz, local))
    axis = pose[:3, :3] @ local
    axis = axis / np.linalg.norm(axis)
    pivot = pose[:3, :3] @ np.asarray(prim.GetAttribute("physics:localPos1").Get(), float) + position
    cache, points = UsdGeom.XformCache(), []
    for mesh in link.collision_meshes.values():
        if mesh.prim.GetTypeName() != "Mesh":
            continue
        relative = np.asarray(cache.ComputeRelativeTransform(mesh.prim, link.prim)[0]).T
        local_points = np.asarray(UsdGeom.Mesh(mesh.prim).GetPointsAttr().Get(), float)
        points.append(local_points @ relative[:3, :3].T + relative[:3, 3])
    points = np.vstack(points) @ pose[:3, :3].T + pose[:3, 3] if points else None
    return axis, pivot, points


def closed_geometry(points, axis, pivot, position, revolute):
    """The closed leaf's centre, horizontal width direction (unit), face normal (unit), width, thickness, bottom and
    height, from its collision points at joint position `position`."""
    closed = (points - pivot) @ turn(axis, -position).T + pivot if revolute else points - position * axis
    bottom, top = float(closed[:, 2].min()), float(closed[:, 2].max())
    flat = closed[:, :2] - closed[:, :2].mean(0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    along, across = flat @ vt[0], flat @ vt[1]
    middle = closed[:, :2].mean(0) + vt[0] * (along.max() + along.min()) / 2 + vt[1] * (across.max() + across.min()) / 2
    return {"centre": np.r_[middle, (bottom + top) / 2], "wide": np.r_[vt[0], 0.], "normal": np.r_[vt[1], 0.],
            "width": float(np.ptp(along)), "thickness": float(np.ptp(across)), "bottom": bottom, "height": top - bottom}


def is_leaf(geometry):
    """Whether a door's jointed part is a leaf one could walk through when open (not a handle)."""
    return geometry["width"] >= LEAF_MIN_M[0] and geometry["height"] >= LEAF_MIN_M[1]


def opening_side(axis, pivot, centre, normal, lower, upper):
    """+1 or -1: the side of the closed leaf (along normal) into which an upright hinge swings it open."""
    sign = 1. if upper > abs(lower) else -1.  # the joint opens toward its larger limit
    arm = np.asarray(centre, float) - pivot
    arm[2] = 0.
    return 1 if float(np.cross(sign * np.asarray(axis, float), arm) @ normal) > 0 else -1


def side_points(geometry, side):
    """The 9 ray points on one side (+1 or -1 along the normal) of a closed leaf."""
    g = geometry
    return [g["centre"] + side * g["normal"] * (g["thickness"] / 2 + distance) + g["wide"] * fraction * g["width"]
            for distance in SIDE_DISTANCES_M for fraction in SIDE_FRACTIONS]


def classify(floor_points, low_points=None):
    """{+1: n, -1: n} points with floor (and points where the ray met something below the floor band) -> the side with no
    floor beyond a door with floor on its other side, or None. A lower floor (a step down: something below the band at
    FLOOR_MIN_POINTS points or more) keeps the side open; walls or furniture without floor under them, or a few low
    pieces, do not."""
    for side in (1, -1):
        void = floor_points[side] == 0 and (low_points is None or low_points[side] < FLOOR_MIN_POINTS)
        if void and floor_points[-side] >= FLOOR_MIN_POINTS:
            return side
    return None


def find(scene, robot):
    """Every door leaf of the loaded scene with its closed geometry, how it opens, the floor and rooms on each side, and
    whether it opens onto no floor (dicts with lists for JSON)."""
    from omnigibson.utils.constants import JointType
    from omnigibson.utils.sampling_utils import raytest_batch

    owners = {str(link.prim_path): obj for obj in scene.objects for link in obj.links.values()}
    robot_bodies = [str(link.prim_path) for link in robot.links.values()]

    def owner(path):
        while path and path not in owners:
            path = path.rsplit("/", 1)[0]
        return owners.get(path)

    doors = []
    for obj in scene.objects:
        if obj.category not in DOOR_CATEGORIES:
            continue
        bodies = [str(link.prim_path) for link in obj.links.values()]
        for name, joint in obj.joints.items():
            revolute = joint.joint_type == JointType.JOINT_REVOLUTE
            if not revolute and joint.joint_type != JointType.JOINT_PRISMATIC:
                continue
            try:
                doors.append(_door(obj, name, joint, revolute, bodies, robot_bodies, owner, raytest_batch))
            except Exception as error:  # a leaf the geometry cannot read is reported, not blocked
                doors.append({"object": obj.name, "category": obj.category, "joint": name,
                              "error": f"{type(error).__name__}: {error}"})
    return [door for door in doors if door is not None]


def _door(obj, name, joint, revolute, bodies, robot_bodies, owner, raytest_batch):
    """One door leaf's facts (see find), or None for a part without collision geometry or too small to be a leaf."""
    axis, pivot, points = leaf(obj, joint)
    if points is None:
        return None
    position = float(_values(joint.get_state()[0])[0])
    lower, upper = float(joint.lower_limit), float(joint.upper_limit)
    g = closed_geometry(points, axis, pivot, position, revolute)
    if not is_leaf(g):
        return None
    if revolute and abs(axis[2]) >= UPRIGHT:
        kind, swing = "hinged", opening_side(axis, pivot, g["centre"], g["normal"], lower, upper)
    elif not revolute and abs(axis[2]) <= 1 - UPRIGHT:
        kind, swing = "sliding", 0
    else:
        kind, swing = "lifting", 0
    sides = {}
    for side in (1, -1):
        ray_points = side_points(g, side)
        hits = raytest_batch([[p[0], p[1], g["bottom"] + RAY_TOP_M] for p in ray_points],
                             [[p[0], p[1], g["bottom"] + RAY_BOTTOM_M] for p in ray_points],
                             only_closest=False, ignore_bodies=robot_bodies + bodies)
        floors, empty, low, rooms = 0, 0, 0, {}
        for column in hits:
            heights = [float(_values(h["position"])[2]) - g["bottom"] for h in column if h["hit"]]
            found = [h for h in column if h["hit"] and
                     FLOOR_BAND_M[0] <= float(_values(h["position"])[2]) - g["bottom"] <= FLOOR_BAND_M[1]]
            floors += bool(found)
            empty += not heights
            low += any(z < FLOOR_BAND_M[0] for z in heights)
            for hit in found:
                for room in getattr(owner(hit["rigidBody"]), "in_rooms", None) or ():
                    rooms[room] = rooms.get(room, 0) + 1
        sides[side] = {"floor_points": floors, "empty_points": empty, "low_points": low, "points": len(ray_points),
                       "room": max(rooms, key=rooms.get) if rooms else None}
    edge = classify({side: sides[side]["floor_points"] for side in sides},
                    {side: sides[side]["low_points"] for side in sides})
    return {"object": obj.name, "category": obj.category, "joint": name, "kind": kind, "swing": swing,
            "position": position, "limits": [lower, upper], "axis": axis.tolist(), "pivot": pivot.tolist(),
            **{key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in g.items()},
            "sides": {str(side): value for side, value in sides.items()},
            "no_floor_side": edge, "prim_path": str(obj.prim_path)}


def block(doors):
    """Mark every door that opens onto no floor for a wall in its doorway (door["blocked"]: the wall's prim path) and
    return those paths. Nothing is made here: the Guard makes a wall when the robot comes near its doorway."""
    paths = []
    for door in doors:
        if door.get("no_floor_side") is not None:
            door["blocked"] = f"{BLOCKER_ROOT}/{door['object']}_{door['joint']}"
            paths.append(door["blocked"])
    return paths


def make_wall(stage, door):
    """A doorway's wall: a static, invisible collision box in the plane of the closed leaf (as wide and high, thinner),
    filtered against the door so that the leaf still turns or slides through it."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    wide, normal = np.asarray(door["wide"]), np.asarray(door["normal"])
    frame = np.column_stack([wide, normal, np.cross(wide, normal)])
    if frame[2, 2] < 0:
        frame[:, 0] *= -1
        frame[:, 2] *= -1
    w = math.sqrt(max(0., 1. + np.trace(frame))) / 2
    x, y, z = ((frame[2, 1] - frame[1, 2]) / (4 * w), (frame[0, 2] - frame[2, 0]) / (4 * w),
               (frame[1, 0] - frame[0, 1]) / (4 * w))
    cube = UsdGeom.Cube.Define(stage, door["blocked"])
    cube.CreateSizeAttr(1.)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, door["centre"])))
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    xform.AddScaleOp().Set(Gf.Vec3f(float(door["width"]), min(BLOCKER_THICKNESS_M, door["thickness"] / 2),
                                    float(door["height"])))
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.FilteredPairsAPI.Apply(cube.GetPrim()).CreateFilteredPairsRel().AddTarget(Sdf.Path(door["prim_path"]))
    return door["blocked"]


class Guard:
    """Makes a boundary doorway's wall when the robot's pelvis comes within GUARD_M of the doorway, and removes every wall
    when the next episode is reset, before its scene loads (reset): the scene holds no wall until the robot nears a
    boundary door, so episodes that never do run exactly as without walls (any collider, even one with collision off,
    changed PhysX's pair order)."""

    def __init__(self, doors, stage=None):
        self.doors = {door["blocked"]: door for door in doors or () if door.get("blocked")}
        self.walls = [(path, np.asarray(door["centre"][:2], float), np.asarray(door["wide"][:2], float),
                       door["width"] / 2) for path, door in self.doors.items()]
        self.stage, self.on = stage, set()

    def _edit(self):
        if self.stage is None:
            import omnigibson as og
            return og.sim.stage, og.sim.editing_usd()  # OmniGibson's guard: USD edits sync to Fabric in this context
        import contextlib
        return self.stage, contextlib.nullcontext()

    def _make(self, paths):
        stage, editing = self._edit()
        with editing:
            for path in paths:
                make_wall(stage, self.doors[path])

    def _remove(self, paths):
        stage, editing = self._edit()
        with editing:
            for path in paths:
                stage.RemovePrim(path)
            if not stage.GetPrimAtPath(BLOCKER_ROOT).GetChildren():
                stage.RemovePrim(BLOCKER_ROOT)
        if self.stage is None:
            # Removing a collider invalidates PhysX's simulation view: OmniGibson's physics views (the robot's joint
            # reads) must be rebuilt, as after removing an object.
            import omnigibson as og
            og.sim.update_handles()

    def reset(self):
        if self.on:
            self._remove(sorted(self.on))
            self.on = set()

    def update(self, xy):
        """xy: the robot's pelvis in the world (m). Returns the walls made now."""
        near = []
        for path, centre, wide, half in self.walls:
            if path in self.on:
                continue
            along = float(np.clip((np.asarray(xy, float) - centre) @ wide, -half, half))
            if np.linalg.norm(np.asarray(xy, float) - (centre + along * wide)) < GUARD_M:
                near.append(path)
        if near:
            self._make(near)
            self.on.update(near)
        return near


def _bearing(vector):
    """A horizontal direction in the start frame as words: "+x" etc. within 15 degrees of an axis, else its bearing."""
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    for label, at in (("+x", 0.), ("+y", 90.), ("-x", 180.), ("-y", -90.), ("-x", -180.)):
        if abs(angle - at) <= 15.:
            return label
    return f"bearing {angle:.0f} degrees"


def _xy(point):
    """(x, y) in metres to 0.1 m, never -0.0."""
    return f"({round(float(point[0]), 1) + 0.:.1f}, {round(float(point[1]), 1) + 0.:.1f})"


KINDS = {"door": "a door", "sliding_door": "a sliding door", "garage_door": "a garage door"}
HEADER = ("Room doors of this scene, each at its closed position in base/odom (the episode-start frame: x ahead, y to "
          "the left, in metres; a side of a door is named by the direction it faces, +x, -y or a bearing in degrees "
          "counterclockwise from +x)")
BLOCKED = "an invisible wall in its doorway keeps the robot from passing through, although the door itself still opens"


def describe(doors, origin):
    """The world text's paragraph on this scene's room doors (doors with floor on at least one side), positions in
    base/odom; origin: the episode-start base pose (position, rotation matrix) the odometry starts from."""
    start, rotation = (np.asarray(x, float) for x in origin)

    def local(point):
        return rotation.T @ (np.asarray(point, float) - start)

    shown = [d for d in doors if "error" not in d and max(s["floor_points"] for s in d["sides"].values()) > 0]
    if not shown:
        return HEADER + ": none."
    lines = []
    for door in sorted(shown, key=lambda d: tuple(np.round(local(d["centre"])[:2], 1))):
        centre, normal = local(door["centre"]), rotation.T @ np.asarray(door["normal"], float)
        start_side = 1 if float(-centre @ normal) > 0 else -1
        text = f"{KINDS[door['category']]} at {_xy(centre)}, {door['width']:.2f} m wide"
        rooms = {side: door["sides"][str(side)]["room"] for side in (1, -1)}
        if door["no_floor_side"] is None and rooms[1] and rooms[-1] and rooms[1] != rooms[-1]:
            text += f", between {rooms[1]} (its {_bearing(normal)} side) and {rooms[-1]} (its {_bearing(-normal)} side)"
        if door["kind"] == "hinged":
            into, hinge = door["swing"] * normal, local(door["pivot"])
            text += (f", hinged at {_xy(hinge)}: it swings open toward its {_bearing(into)} side, so "
                     f"it opens by a push from its {_bearing(-into)} side and by a pull from its {_bearing(into)} side; "
                     f"the start position is on its {'pull' if start_side == door['swing'] else 'push'} side")
        elif door["kind"] == "sliding":
            text += ": it slides sideways along its wall"
        else:
            text += ": it opens upward"
        if door["no_floor_side"] is not None:
            text += (f"; past its {_bearing(door['no_floor_side'] * normal)} side there is no floor (it leads out of "
                     f"the loaded rooms), and " + (BLOCKED if door.get("blocked") else
                                                    "walking through it drops the robot out of the world"))
        lines.append(text)
    return HEADER + ": " + "; ".join(f"({i}) {line}" for i, line in enumerate(lines, 1)) + "."


def write(doors, origin, folder=None):
    """Keep the doors and the paragraph in the run directory (scene-doors.json, scene-doors.txt): an operator record,
    and the text a task runner may add to the world device's description."""
    folder = folder or os.environ.get("ROSBRIDGE_RUN_DIR")
    if not folder:
        return None
    text = describe(doors, origin)
    Path(folder, "scene-doors.json").write_text(json.dumps({"doors": doors, "text": text}, indent=1))
    Path(folder, "scene-doors.txt").write_text(text + "\n" if text else "")
    return text
