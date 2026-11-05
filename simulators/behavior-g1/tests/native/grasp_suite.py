"""Native probe and regression suite for the G1's hands: assisted grasping (g1_grasp's rules) and physical grasping.

One command, in the behavior-g1-native environment (no pixi needed; Isaac hosts one app per process, so this runs the
world in process):
    ROSBRIDGE_PROJECT=<tree> PYTHONPATH=<tree>/src:<tree>/simulators/behavior:<tree>/.runtime/behavior-g1 \\
    ROSBRIDGE_GRASPING=assisted python <tree>/simulators/behavior-g1/tests/native/grasp_suite.py \\
        [--task can_meat] [--only a,b] [--stress] [--out results.jsonl]
Every scenario drives G1SharpaSim with the joint targets the devices send, places objects in the hand's frame (pinned
until the hand holds them from both sides, then free), reads the hand topics the worker reads (sim.status) and checks
the outcome against its expectation. The summary prints PASS/FAIL per scenario and exits non-zero on a failure. It
reads only OmniGibson's and the world's state, so the same file measures earlier versions of the world (their outcomes
then show where they differ). ROSBRIDGE_GRASPING=physical runs the physical-mode regression instead.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

SIDES = ("right", "left")
DT = .02  # one world step (50 Hz)


class Rig:
    """The world, the command vector the devices fill, and the hand and object helpers the scenarios share."""

    def __init__(self, task, sim=None):
        from rosbridge_behavior.backend import G1SharpaSim, pose_matrix
        if sim is None:  # a world of its own (sim: one already loaded, e.g. in a probe server)
            sim = G1SharpaSim(max_steps=10 ** 8)
            sim.reset(task, 0)
        self.sim, self.task = sim, task
        self.pose_matrix = pose_matrix
        self.robot = self.sim.episode.robot
        self.groups = {g.name: g for g in self.sim.layout.groups}
        self.names = list(self.sim.layout.joint_names)
        self.cmd = self.sim.state()[0][:len(self.names)].copy()
        self.cmd[list(self.groups["base"].indices)] = 0.
        self.home = self.cmd.copy()
        self.pins = {}

    def fresh(self):
        """The task's start state again (each scenario starts from it; a scenario may topple the robot)."""
        self.sim.reset(self.task, 0)
        self.robot = self.sim.episode.robot
        self.cmd, self.pins = self.home.copy(), {}
        for name in ("bratwurst_230", "bratwurst_231", "bratwurst_232", "bratwurst_233", "hinged_jar_235", "hinged_jar_236"):
            if self.obj(name) is not None:
                self.park(self.obj(name), 10)
        self.step(5)

    # world --------------------------------------------------------------------------------------------------------
    @property
    def scene(self):
        return self.robot.scene

    def obj(self, name):
        return self.scene.object_registry("name", name)

    def step(self, n=1, each=None):
        for k in range(n):
            self.sim.step(self.cmd)
            for obj, (p, q) in self.pins.items():
                obj.set_position_orientation(position=p, orientation=q)
                obj.keep_still()
            self.check_truth()
            if each is not None and each(k):
                return k
        return None

    truth_steps = truth_mismatch = 0

    def check_truth(self):
        """The hand topics agree with the simulator on every step (OmniGibson's grasp joint and its parameters)."""
        self.truth_steps += 1
        for side in SIDES:
            params = self.robot._ag_obj_constraint_params[side] if self.robot._ag_obj_in_hand[side] is not None else None
            expected = ("none", "none") if params is None else (
                "tethered" if params["joint_type"] == "SphericalJoint" else "held",
                f"{params['target_obj'].name}/{params['target_link_name']}")
            got = self.status(side)
            if got[0] != expected[0] or (got[1] is not None and got[1] != expected[1]):
                self.truth_mismatch += 1

    def status(self, side="right"):
        return self.sim.status.get(f"{side}_hand/attachment"), self.sim.status.get(f"{side}_hand/attached_object")

    def held(self, side="right"):
        return self.robot._ag_obj_in_hand[side] is not None and self.robot._ag_obj_constraint_params[side] is not None

    def link(self, name):
        return self.pose_matrix(*self.robot.links[name].get_position_orientation())

    # commands -----------------------------------------------------------------------------------------------------
    def joints(self, device):
        return [self.names[i] for i in self.groups[device].indices]

    def set_arm(self, side, **angles):
        g = self.groups[f"{side}_arm"]
        for key, value in angles.items():
            self.cmd[g.indices[self.joints(f"{side}_arm").index(f"{side}_{key}_joint")]] = value

    def arm_forward(self, side="right"):
        self.set_arm(side, shoulder_pitch=-1.2, shoulder_roll=0., shoulder_yaw=0., elbow=.3, wrist_roll=0.,
                     wrist_pitch=0., wrist_yaw=0.)

    def arm_home(self, side="right"):
        g = self.groups[f"{side}_arm"]
        self.cmd[list(g.indices)] = self.home[list(g.indices)]

    def set_hand(self, side, closure, per=None):
        """Every flexion joint at closure of its range (abduction at 0); per: {digit: closure} overrides."""
        g = self.groups[f"{side}_hand"]
        names, lo, hi = self.joints(f"{side}_hand"), np.array(g.lower), np.array(g.upper)
        c = np.full(len(names), float(closure))
        for key, value in (per or {}).items():
            c[[key in name for name in names]] = value
        self.cmd[list(g.indices)] = np.where(["_AA" in name for name in names], 0., lo + c * (hi - lo))

    def ready(self, side="right", closure=0., per=None, n=40):
        self.release_all()
        self.arm_forward(side)
        self.set_hand(side, closure, per)
        self.step(n)

    def release_all(self):
        for side in SIDES:
            if self.robot._ag_obj_constraints.get(side) is not None:
                self.robot.release_grasp_immediately(arm=side)
            self.robot._ag_obj_in_hand[side] = None
            self.robot._ag_release_counter[side] = None
            self.robot._ag_grasp_counter[side] = None
            grips = getattr(self.robot, "_g1_grips", None)
            if grips:
                grips[side].let_go()

    # objects ------------------------------------------------------------------------------------------------------
    def gravity(self, obj, on):
        for link in obj.links.values():
            (link.enable_gravity if on else link.disable_gravity)()

    def place(self, obj, position, rotation=None):
        import torch as th
        from scipy.spatial.transform import Rotation
        quat = obj.get_position_orientation()[1] if rotation is None else th.tensor(
            Rotation.from_matrix(rotation).as_quat(), dtype=th.float32)
        obj.set_position_orientation(position=th.tensor(np.asarray(position), dtype=th.float32), orientation=quat)
        obj.keep_still()

    def place_in_palm(self, obj, offset, side="right"):
        """The object's AABB centre at offset (m) in the palm frame (x out of the palm, z toward the fingers), its
        local z axis across the palm."""
        palm = self.link(f"{side}_hand_C_MC")
        rotation = np.c_[palm[:3, 2], palm[:3, 0], palm[:3, 1]]
        lo, hi = (v.numpy() for v in obj.aabb)
        p, q = obj.get_position_orientation()
        from scipy.spatial.transform import Rotation
        centre_local = Rotation.from_quat(q.numpy()).as_matrix().T @ ((lo + hi) / 2 - p.numpy())
        target = (palm @ np.r_[offset, 1.])[:3]
        self.place(obj, target - rotation @ centre_local, rotation)

    def pin(self, obj):
        self.pins[obj] = obj.get_position_orientation()

    def unpin(self, obj):
        self.pins.pop(obj, None)

    def park(self, obj, k=0):
        self.unpin(obj)
        self.gravity(obj, False)
        self.place(obj, [30. + k, 30., 5.])

    def hand_links(self, side, obj):
        """The hand links (names) touching obj: finger links and the palm."""
        from omnigibson.utils.usd_utils import RigidContactAPI
        hand = {link.prim_path for link in self.robot.finger_links[side]} | {
            self.robot.links[f"{side}_hand_C_MC"].prim_path}
        pairs = RigidContactAPI.get_contact_pairs(self.scene.idx, hand, None, False)
        return sorted({a.rsplit("/", 1)[-1] for a, b in pairs if b.startswith(obj.prim_path + "/")})

    def sides_touching(self, side, obj):
        out = set()
        for name in self.hand_links(side, obj):
            part = name.split("_", 1)[1]
            out.add("thumb" if part.startswith("thumb") or part in ("hand_C_MC", "pinky_MC") else "finger")
        return out

    def velocity_relative_to_hand(self, side, obj):
        eef = self.robot.eef_links[side]
        lv, av = eef.get_linear_velocity().numpy(), eef.get_angular_velocity().numpy()
        offset = obj.get_position_orientation()[0].numpy() - eef.get_position_orientation()[0].numpy()
        hand = lv + np.cross(av, offset)
        return float(np.linalg.norm(obj.get_linear_velocity().numpy() - hand)), float(np.linalg.norm(hand))

    def grasp(self, obj, side="right", close=.85, per=None, offset=(.0755, .01, .08), ramp=40, n=100, free=True):
        """Pin obj in the palm frame, ramp the hand closed; once thumb/palm and a finger both touch it, it is freed
        (weightless) so the squeeze centres it. Returns the world step of the attach or None; gravity is on after."""
        self.ready(side)
        self.gravity(obj, False)
        self.place_in_palm(obj, offset, side)
        self.step(1)
        self.pin(obj)
        first = None
        for k in range(n):
            if k < ramp:
                self.set_hand(side, close * (k + 1) / ramp, per)
            self.step(1)
            if free and obj in self.pins and self.sides_touching(side, obj) >= {"thumb", "finger"}:
                self.unpin(obj)
            if self.held(side):
                first = k
                break
        self.unpin(obj)
        self.gravity(obj, True)
        return first


# scenarios --------------------------------------------------------------------------------------------------------
SCENARIOS = {}


def scenario(name, modes=("assisted",), stress=False, furniture=False):
    """stress: long (--stress); furniture: teleports fixed furniture into the hand (--furniture; such placements have
    ended worlds in NaN, so they run apart)."""
    def register(fn):
        SCENARIOS[name] = (fn, modes, "furniture" if furniture else stress)
        return fn
    return register


def expect(row, **checks):
    """row["pass"] = every check true; row["checks"] = the failed ones."""
    failed = [key for key, ok in checks.items() if not ok]
    row.update({"pass": not failed, "failed": failed})
    return row


GLASS, SAUSAGE, SWITCH = "water_glass_bbpraa_0", "bratwurst_230", "electric_switch_gashan_9"
PLACES = ((.0755, .01, .08), (.0705, 0., .075), (.0805, .015, .085), (.0655, .005, .07))
GRAVITY = np.array([0., 0., -9.81])


def grasp_somewhere(rig, obj, side="right", **kw):
    """grasp() at the first of PLACES that attaches (a free object can be pushed out of the closing hand)."""
    for offset in PLACES:
        first = rig.grasp(obj, side, offset=offset, **kw)
        if first is not None:
            return first
        rig.set_hand(side, 0.)
        rig.step(10)
        rig.release_all()
        rig.park(obj)
    return None


def open_and_watch(rig, obj, side="right", steps=1, swing=False):
    """Open the hand over steps world steps. Returns (world steps until none, fling, shift): over 0.3 s after the
    release, or until the object touches something other than the hand, the largest speed of the object relative to
    the hand's velocity at the object less what free fall since the release gives (m/s; 0 when it leaves at the hand's
    velocity or rests in the opening hand), and how far it is sideways from where a throw at the velocity the hand had at
    the release would put it (m)."""
    from omnigibson.utils.usd_utils import RigidContactAPI
    hand = {link.prim_path for s in SIDES for link in rig.robot.finger_links[s]} | {
        rig.robot.links[f"{s}_hand_C_MC"].prim_path for s in SIDES}
    links = {link.prim_path for link in obj.links.values()}
    if swing:
        rig.set_arm(side, shoulder_pitch=-.8)
    released, fling, k, carried, shift = None, [], 0, None, None
    for k in range(60):
        if k < steps:
            rig.set_hand(side, .85 * (1 - (k + 1) / steps))
        rig.step()
        if released is not None:
            t = (k - released) * DT
            if t > .3 or any(b not in hand for _, b in RigidContactAPI.get_contact_pairs(rig.scene.idx, links, None, False)):
                break
            eef = rig.robot.eef_links[side]
            lv, av = eef.get_linear_velocity().numpy(), eef.get_angular_velocity().numpy()
            at = obj.get_position_orientation()[0].numpy()
            hand_v = lv + np.cross(av, at - eef.get_position_orientation()[0].numpy())
            if carried is None:  # the release point and the velocity the hand gave it
                carried = (at, hand_v)
            fling.append(float(np.linalg.norm(obj.get_linear_velocity().numpy() - hand_v)) - 9.81 * t)
            # how far sideways it is from a throw at the hand's velocity at the release
            shift = float(np.linalg.norm((at - carried[0] - carried[1] * t)[:2]))
        if released is None and rig.status(side)[0] == "none":
            released = k
    return released, (max(fling) if fling else None), shift


def touched_by_shape(rig, obj, closure, per, side="right", press=0.):
    """A shaped hand (no closing command) and obj pinned on the palm normal 3 cm out, moved 1 mm per step until the hand
    touches it, then press m further, then held 1 s: the step that attached, or None."""
    rig.park(obj)
    rig.ready(side, closure, per)
    gap, touched = .03, None
    for k in range(80):
        rig.place_in_palm(obj, (.021 + .0445 + gap, .03, .10), side)
        rig.pin(obj)
        rig.step()
        if rig.hand_links(side, obj):
            touched = gap
            break
        gap -= .001
    if touched is not None and press:
        rig.place_in_palm(obj, (.021 + .0445 + touched - press, .03, .10), side)
        rig.pin(obj)
    attached = rig.step(50, each=lambda k: rig.held(side))
    rig.unpin(obj)
    rig.release_all()
    rig.park(obj)
    return attached


@scenario("power_grasp_and_release")
def power_grasp_and_release(rig):
    """A power grasp of the glass sticks (held, named), holds 1 s under gravity, and lets go on opening without a
    fling: instant, gradual and swinging releases, both hands."""
    glass, rows = rig.obj(GLASS), []
    for side in SIDES:
        for how in ("instant", "gradual", "swing"):
            first = grasp_somewhere(rig, glass, side)
            row = {"side": side, "release": how, "attach_s": None if first is None else round((first + 1) * DT, 2)}
            if first is not None:
                z0 = glass.get_position_orientation()[0][2].item()
                rig.step(50)
                row.update(status=rig.status(side), dz=round(glass.get_position_orientation()[0][2].item() - z0, 3))
                row["released_after"], row["fling"], row["shift_m"] = open_and_watch(
                    rig, glass, side, 40 if how == "gradual" else 1, how == "swing")
            rows.append(row)
            rig.arm_forward(side)
            rig.release_all()
            rig.park(glass)
    ok = [r for r in rows if r["attach_s"] is not None]
    return expect({"trials": rows},
                  attached=len(ok) >= len(rows) - 1,
                  held_named=all(r["status"] == ("held", f"{GLASS}/base_link") and r["dz"] > -.03 for r in ok),
                  released=all(r["released_after"] is not None and r["released_after"] <= (45 if r["release"] == "gradual"
                                                                                           else 5) for r in ok),
                  no_fling=all(r["shift_m"] is None or r["shift_m"] <= .05 for r in ok))


@scenario("light_pinch")
def light_pinch(rig):
    """A thumb-index pinch with the other fingers straight sticks (under a mean-closure rule it never did: mean flexion 22%)."""
    glass = rig.obj(GLASS)
    first = rig.grasp(glass, close=.6, per={"middle": 0., "ring": 0., "pinky": 0.}, offset=(.0655, .01, .08), free=False)
    row = {"attach_s": None if first is None else round((first + 1) * DT, 2), "status": rig.status()}
    rig.set_hand("right", 0.)
    rig.step(10)
    rig.release_all()
    rig.park(glass)
    return expect(row, attached=first is not None)


@scenario("one_digit")
def one_digit(rig):
    """One fingertip (pad and bone) or the thumb alone never sticks: the sausage on the index pad, 8 placements."""
    sausage, attached = rig.obj(SAUSAGE), 0
    for k in range(8):
        rig.ready("right", .35, per={"thumb": 0., "middle": .9, "ring": .9, "pinky": .9})
        tip = (rig.link("right_index_DP") @ np.r_[.016, 0., .012 + .002 * (k % 4), 1.])[:3]
        rig.gravity(sausage, False)
        rig.place(sausage, tip + np.r_[0., 0., .0] * k)
        rig.pin(sausage)
        attached += rig.step(40, each=lambda i: rig.held("right")) is not None
        rig.unpin(sausage)
        rig.release_all()
    rig.park(sausage)
    return expect({"attached_of_8": attached}, never=attached == 0)


@scenario("shape_touching")
def shape_touching(rig):
    """A shaped hand brought to touch the glass, without a closing command: ring and pinky curled (adjacent fingers, no
    opposition) never stick, pressed or not; a fist that touches does not; a claw whose thumb and fingers both touch
    and are bent back by the pinned glass squeezes it, as the R1Pro's blocked jaws do (reported)."""
    glass = rig.obj(GLASS)
    shapes = {"ring+pinky": (0., {"ring": 1., "pinky": 1.}), "fist": (1., None), "claw": (.5, None)}
    rows = {name: {f"press_{mm}mm": touched_by_shape(rig, glass, c, per, press=mm / 1000) for mm in (0, 5, 10)}
            for name, (c, per) in shapes.items()}
    return expect(rows, touch_never=rows["ring+pinky"]["press_0mm"] is None and rows["fist"]["press_0mm"] is None,
                  adjacent_never=all(v is None for v in rows["ring+pinky"].values()))


@scenario("opening_hand")
def opening_hand(rig):
    """A quick close then an opening over 1 s never completes a grasp (the glass pinned in the aperture)."""
    glass = rig.obj(GLASS)
    rig.ready("right")
    rig.place_in_palm(glass, (.0755, .01, .08))
    rig.pin(glass)
    seen = []
    for k in range(60):
        rig.set_hand("right", .85 * (k + 1) / 10 if k < 10 else .85 * (1 - (k - 9) / 50))
        rig.step()
        seen.append(rig.status()[0])
    rig.unpin(glass)
    rig.release_all()
    rig.park(glass)
    return expect({"statuses": sorted(set(seen))}, never=set(seen) == {"none"})


@scenario("fixed_body_never_welds")
def fixed_body_never_welds(rig):
    """A light fixed-base switch closed in the hand (all digits) never attaches; walking back moves the hand."""
    switch = rig.obj(SWITCH)
    p0, q0 = switch.get_position_orientation()
    rig.ready("right")
    rig.place_in_palm(switch, (.0305, .01, .09))
    seen = []
    for k in range(70):
        rig.set_hand("right", .85 * min(1., (k + 1) / 30))
        rig.step()
        seen.append(rig.status()[0])
    rig.set_hand("right", 0.)
    rig.step(20)
    rig.release_all()
    switch.set_position_orientation(position=p0, orientation=q0)
    return expect({"statuses": sorted(set(seen))}, never=set(seen) == {"none"})


@scenario("fallen_body")
def fallen_body(rig):
    """No grasp is searched while the body has fallen (the embodiment's flag), with every digit on the glass."""
    from rosbridge_behavior.g1_sharpa import LOWER
    emb, glass = rig.sim.embodiment, rig.obj(GLASS)
    emb.frozen = rig.robot.get_joint_positions().numpy()[emb.body][:len(LOWER)].copy()
    emb.fallen = True
    try:
        rig.ready("right")
        rig.place_in_palm(glass, (.0755, .01, .08))
        rig.pin(glass)
        for k in range(60):
            rig.set_hand("right", .85 * min(1., (k + 1) / 30))
            rig.step()
        attached = rig.held("right")
    finally:
        emb.fallen, emb.frozen = False, None
        rig.unpin(glass)
        rig.release_all()
        rig.park(glass)
    return expect({"attached": attached}, never=not attached)


@scenario("heavy_part")
def heavy_part(rig):
    """A 12 kg glass is refused, a 9.9 kg one held (OmniGibson's 10 kg per link)."""
    glass = rig.obj(GLASS)
    mass, out = float(glass.links["base_link"].mass), {}
    for kg in (12., 9.9):
        glass.links["base_link"].mass = kg
        rig.ready("right")
        rig.place_in_palm(glass, (.0755, .01, .08))
        rig.pin(glass)
        out[kg] = rig.step(80, each=lambda k: (rig.set_hand("right", .85 * min(1., (k + 1) / 40)), rig.held("right"))[1])
        rig.unpin(glass)
        rig.set_hand("right", 0.)
        rig.step(10)
        rig.release_all()
        rig.park(glass)
    glass.links["base_link"].mass = mass
    return expect({"attach_step": out}, refused=out[12.] is None, held=out[9.9] is not None)


@scenario("distance")
def distance(rig):
    """The pinned glass on the palm normal at gaps of 0-3 cm sticks as the hand closes; at 5 and 10 cm
    (the fingertips may touch, nothing opposes) it does not."""
    glass, out = rig.obj(GLASS), {}
    for gap in (0., .01, .02, .03, .05, .10):
        rig.ready("right")
        rig.place_in_palm(glass, (.021 + .0445 + gap, .01, .08))
        rig.pin(glass)
        out[gap] = rig.step(80, each=lambda k: (rig.set_hand("right", .85 * min(1., (k + 1) / 40)), rig.held("right"))[1])
        rig.unpin(glass)
        rig.set_hand("right", 0.)
        rig.step(10)
        rig.release_all()
        rig.park(glass)
    near = [out[g] for g in (0., .01, .02, .03)]
    return expect({"attach_step_by_gap_m": out}, near=sum(v is not None for v in near) >= 3,
                  far=out[.05] is None and out[.10] is None)


@scenario("jam_squat")
def jam_squat(rig):
    """The held glass jammed (pinned in the world) while the body squats to 0.45 for 3 s: the body goes
    as low as a free squat (within 5 cm), or the attachment lets go first."""
    glass, pelvis = rig.obj(GLASS), rig.groups["pelvis"].indices[0]
    z_free = free_squat_height(rig)
    first = grasp_somewhere(rig, glass)
    if first is None:
        return expect({"attached": False}, attached=False)
    rig.step(25)
    rig.pin(glass)
    rig.cmd[pelvis] = .45
    rig.step(150)
    z = float(rig.link("pelvis")[2, 3] - rig.sim.embodiment.floor)
    row = {"z_free": round(z_free, 3), "z_jammed": round(z, 3), "status": rig.status(), "fallen": bool(rig.sim.embodiment.fallen)}
    rig.unpin(glass)
    rig.cmd[pelvis] = rig.home[pelvis]
    rig.set_hand("right", 0.)
    rig.step(100)
    rig.release_all()
    rig.park(glass)
    return expect(row, body_down=z <= z_free + .05 or row["status"][0] == "none")


@scenario("jam_walk")
def jam_walk(rig):
    """The held glass jammed (pinned in the world) while the robot walks back at 0.3 m/s for 3 s: the attachment
    lets go under the sustained load (or the load never passes the limit), and the robot does not fall."""
    from rosbridge_behavior import g1_sharpa
    glass, base = rig.obj(GLASS), list(rig.groups["base"].indices)
    first = grasp_somewhere(rig, glass)
    if first is None:
        return expect({"attached": False}, attached=False)
    rig.step(25)
    rig.pin(glass)
    p0 = rig.link("pelvis")[:3, 3].copy()
    rig.cmd[base] = (-.3, 0., 0.)
    ended = rig.step(150, each=lambda k: rig.status()[0] == "none")
    rig.cmd[base] = 0.
    grip = getattr(rig.robot, "_g1_grips", {}).get("right")
    row = {"ended_after_steps": ended, "why": getattr(grip, "why", None), "load_limit_n": getattr(
        g1_sharpa, "grasp_load_limit", lambda: None)(), "walked_m": round(float(np.linalg.norm(
            rig.link("pelvis")[:2, 3] - p0[:2])), 3), "fallen": bool(rig.sim.embodiment.fallen)}
    rig.unpin(glass)
    rig.set_hand("right", 0.)
    rig.step(50)
    rig.release_all()
    rig.park(glass)
    return expect(row, upright=not row["fallen"])


def light_objects(rig, n=6):
    """Up to n loose objects of distinct categories, 10 g to 3 kg, that fit a hand (the glass first)."""
    out, seen = [rig.obj(GLASS)], {rig.obj(GLASS).category}
    for obj in sorted(rig.scene.objects, key=lambda o: o.name):
        if len(out) >= n or obj.fixed_base or obj.category in seen or obj.prim_type != out[0].prim_type:
            continue
        try:
            mass = sum(float(link.mass) for link in obj.links.values())
            lo, hi = (v.numpy() for v in obj.aabb)
        except Exception:  # noqa: BLE001 - objects without links or extents
            continue
        size = sorted(hi - lo)
        if .01 <= mass <= 3. and .01 <= size[1] <= .15 and size[2] <= .4 and obj is not rig.robot:
            out.append(obj)
            seen.add(obj.category)
    return out


@scenario("cycles", stress=True)
def cycles(rig, n=200):
    """200 grasp/release cycles over six object categories and both hands (every 25th the left; every 10th release
    instant): no NaN, no crash, no attachment that does not release. A fall (the robot can be knocked over by a
    teleported object) is reset and counted."""
    objs = light_objects(rig)
    stats = {"objects": [f"{o.name} ({o.category})" for o in objs], "cycles": 0, "attached": 0, "released": 0,
             "stuck": 0, "nan": 0, "falls": 0, "by_object": {}}
    t0 = time.time()
    for k in range(n):
        if rig.sim.embodiment.fallen:
            stats["falls"] += 1
            rig.fresh()
        side, obj = SIDES[0 if k % 25 else 1], objs[k % len(objs)]
        first = rig.grasp(obj, side, offset=PLACES[k % len(PLACES)], n=60, ramp=10, free=False)  # a 0.2 s close
        rig.step(10)
        released, _, _ = open_and_watch(rig, obj, side, 1 if k % 10 == 0 else 15)
        stats["cycles"] += 1
        stats["attached"] += first is not None
        stats["released"] += first is not None and released is not None
        stats["stuck"] += first is not None and released is None
        stats["nan"] += not bool(np.isfinite(rig.robot.get_joint_positions().numpy()).all()
                                 and np.isfinite(obj.get_position_orientation()[0].numpy()).all())
        b = stats["by_object"].setdefault(obj.name, [0, 0])
        b[0] += 1
        b[1] += first is not None
        rig.release_all()
        rig.park(obj, k % 6)
        rig.arm_forward(side)
    stats["wall_s"] = round(time.time() - t0)
    return expect(stats, no_nan=stats["nan"] == 0, no_stuck=stats["stuck"] == 0,
                  released_all=stats["released"] == stats["attached"], six_categories=len(objs) >= 6)


@scenario("long_hold_moving", stress=True)
def long_hold_moving(rig, minutes=10.):
    """The glass held for minutes while the robot walks, turns and squats: held on every step, little drift."""
    glass, base, pelvis = rig.obj(GLASS), list(rig.groups["base"].indices), rig.groups["pelvis"].indices[0]
    first = grasp_somewhere(rig, glass)
    if first is None:
        return expect({"attached": False}, attached=False)
    rig.set_arm("right", shoulder_pitch=-.6, elbow=.9)  # a carrying pose, the hand in front of the chest
    rig.step(50)
    program = [((.3, 0., 0.), None, 3), ((-.3, 0., 0.), None, 3), ((0., 0., .5), None, 3), ((0., 0., -.5), None, 3),
               ((0., 0., 0.), .5, 5), ((0., 0., 0.), rig.home[pelvis], 5), ((0., .2, 0.), None, 2), ((0., -.2, 0.), None, 2),
               ((0., 0., 0.), None, 4)]
    eef = lambda: np.linalg.inv(rig.link("right_eef_link")) @ rig.pose_matrix(*glass.get_position_orientation())
    rel0, steps, held, worst, t0 = eef(), 0, 0, 0., time.time()
    while steps * DT < minutes * 60:
        for twist, height, seconds in program:
            rig.cmd[base] = twist
            if height is not None:
                rig.cmd[pelvis] = height
            for _ in range(round(seconds / DT)):
                rig.step()
                steps += 1
                held += rig.status()[0] == "held"
            rel = eef()
            worst = max(worst, float(np.linalg.norm(rel[:3, 3] - rel0[:3, 3])))
    rig.cmd[base] = 0.
    rig.cmd[pelvis] = rig.home[pelvis]
    fallen = bool(rig.sim.embodiment.fallen)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rel0[:3, :3].T @ eef()[:3, :3]) - 1) / 2, -1, 1))))
    rig.set_hand("right", 0.)
    rig.step(20)
    rig.release_all()
    rig.park(glass)
    row = {"steps": steps, "held_steps": held, "drift_m": round(worst, 4), "turned_deg": round(angle, 2), "fallen": fallen,
           "wall_s": round(time.time() - t0)}
    return expect(row, held_throughout=held == steps, drift=worst <= .01 and angle <= 5, upright=not fallen)


@scenario("physical_never_attaches", modes=("physical",))
def physical_never_attaches(rig):
    """Physical grasping: a power grasp and 20 cycles never attach; status none on every step; the glass is held by
    friction or dropped as physics decides (reported)."""
    glass, seen, kept = rig.obj(GLASS), set(), []
    for k in range(20):
        rig.ready("right")
        rig.gravity(glass, False)
        rig.place_in_palm(glass, PLACES[k % len(PLACES)])
        rig.pin(glass)
        for i in range(60):
            rig.set_hand("right", .85 * min(1., (i + 1) / 40))
            rig.step()
            if rig.sides_touching("right", glass) >= {"thumb", "finger"}:
                rig.unpin(glass)
            seen.add(rig.status())
        rig.unpin(glass)
        rig.gravity(glass, True)
        z0 = glass.get_position_orientation()[0][2].item()
        rig.step(50, each=lambda i: seen.add(rig.status()))
        kept.append(round(glass.get_position_orientation()[0][2].item() - z0, 3))
        rig.set_hand("right", 0.)
        rig.step(10)
        rig.park(glass)
    joints = [rig.robot._ag_obj_constraints[s] for s in SIDES]
    return expect({"statuses": sorted(map(str, seen)), "dz_after_1s": kept},
                  never=seen == {("none", "none")}, no_joint=all(j is None for j in joints))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default="can_meat")
    parser.add_argument("--only", default="")
    parser.add_argument("--stress", action="store_true", help="also the long stress scenarios")
    parser.add_argument("--furniture", action="store_true", help="also the scenarios that teleport fixed furniture")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    t0 = time.time()
    rig = Rig(args.task)
    print(f"grasp_suite: {args.task} ready in {time.time() - t0:.0f} s; tree {os.environ.get('ROSBRIDGE_PROJECT')}", flush=True)
    kinds = {kind for kind, on in (("stress", args.stress), ("furniture", args.furniture)) if on}
    failed = run(rig, [s for s in args.only.split(",") if s], kinds, args.out)
    sys.stdout.flush()
    os._exit(1 if failed else 0)  # OmniGibson's shutdown can hang; the results are written


def run(rig, only=(), stress=(), out=""):
    """Run the scenarios of the world's grasping mode (only: these; stress: also "stress" and/or "furniture" ones);
    returns the failed."""
    mode = rig.robot.grasping_mode
    results = {}
    for name, (fn, modes, is_stress) in SCENARIOS.items():
        wanted = {False: True, True: "stress" in stress, "furniture": "furniture" in stress}[is_stress]
        if mode not in modes or (only and name not in only) or (not wanted and name not in only):
            continue
        t1 = time.time()
        try:
            rig.fresh()
            row = fn(rig)
            row["fell"] = bool(rig.sim.embodiment.fallen)
        except Exception:  # noqa: BLE001 - a crash is a result
            row = {"pass": False, "failed": ["exception"], "error": traceback.format_exc()[-2000:]}
        row["wall_s"] = round(time.time() - t1, 1)
        results[name] = row
        print(f"{'PASS' if row['pass'] else 'FAIL'} {name} {json.dumps(row, default=str)[:600]}", flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps({"scenario": name, "mode": mode, "tree": os.environ.get("ROSBRIDGE_PROJECT"), **row},
                                   default=str) + "\n")
    results["status_truth"] = expect({"steps": rig.truth_steps, "mismatching": rig.truth_mismatch},
                                     consistent=rig.truth_mismatch == 0)
    print(f"{'PASS' if results['status_truth']['pass'] else 'FAIL'} status_truth {results['status_truth']}", flush=True)
    failed = [name for name, row in results.items() if not row["pass"]]
    print(f"grasp_suite: {len(results) - len(failed)} of {len(results)} passed" + (f"; failed: {failed}" if failed else ""))
    return failed



# furniture: a door's free edge put in the pinch -----------------------------------------------------------------------
def link_box(link):
    """A door link's panel in its own frame: (centre, extent) of the union of its collision pieces, with the centre
    along the thinnest axis taken from the large pieces only (the panel, not a handle standing out of it)."""
    from pxr import UsdGeom
    lows, highs = [], []
    for mesh in link.collision_meshes.values():
        prim = mesh.prim
        local = np.asarray(UsdGeom.XformCache().ComputeRelativeTransform(prim, link.prim)[0], dtype=float).T
        if prim.GetTypeName() == "Mesh":
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        elif prim.GetTypeName() == "Cube":
            s = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            pts = np.array([[a, b, c] for a in (-s, s) for b in (-s, s) for c in (-s, s)])
        else:
            continue
        pts = (local[:3, :3] @ pts.T).T + local[:3, 3]
        lows.append(pts.min(0))
        highs.append(pts.max(0))
    lows, highs = np.array(lows), np.array(highs)
    lo, hi = lows.min(0), highs.max(0)
    centre, extent = (lo + hi) / 2, hi - lo
    thin = int(np.argmin(extent))
    others = [i for i in range(3) if i != thin]
    area = (highs - lows)[:, others[0]] * (highs - lows)[:, others[1]]
    big = area >= .25 * area.max()
    centre[thin] = (lows[big, thin].min() + highs[big, thin].max()) / 2
    return centre, extent


def pinch_point(rig, side="right"):
    """Where the thumb and index pads meet in a pinch pre-shape, the pinch axis, and the direction out of the palm."""
    shape = lambda a: {"thumb": a, "index": a, "middle": .9, "ring": .9, "pinky": .9}
    rig.ready(side, 0., shape(.42), n=30)
    thumb = (rig.link(f"{side}_thumb_DP") @ np.r_[.02, 0., 0., 1.])[:3]
    index = (rig.link(f"{side}_index_DP") @ np.r_[.016, 0., .008, 1.])[:3]
    rig.set_hand(side, 0., shape(.2))
    rig.step(30)
    u = (index - thumb) / np.linalg.norm(index - thumb)
    w = rig.link(f"{side}_hand_C_MC")[:3, 0]
    w = w - u * np.dot(w, u)
    return (thumb + index) / 2, u, w / np.linalg.norm(w), shape


def put_edge_in_pinch(rig, obj, link, pinch, u, w, shift=0.):
    """Teleport a fixed object so that its link's free edge (farthest from the hinge, the link origin) lies at the
    pinch point, the panel's thickness along the pinch axis u and the panel extending away from the palm (w)."""
    from scipy.spatial.transform import Rotation
    centre, extent = link_box(link)
    thin = int(np.argmin(extent))
    far = max((i for i in range(3) if i != thin), key=lambda i: abs(centre[i]))
    edge = centre.copy()
    edge[far] += np.sign(centre[far]) * (extent[far] / 2 - .006)
    T = rig.pose_matrix(*link.get_position_orientation())
    e_w, u_w, n_w = (T @ np.r_[edge, 1.])[:3], T[:3, thin], T[:3, far] * np.sign(centre[far])
    A = np.c_[u_w, -n_w, np.cross(u_w, -n_w)]
    B = np.c_[u, w, np.cross(u, w)]
    turn = B @ A.T
    p, q = obj.get_position_orientation()
    rig.place(obj, pinch + w * shift + turn @ (p.numpy() - e_w), turn @ Rotation.from_quat(q.numpy()).as_matrix())


def furniture_doors(rig, n=3):
    """Revolute doors of fixed cabinets, two known cabinets' first."""
    from omnigibson.utils.constants import JointType
    out = []
    first = [rig.obj(name) for name in ("bottom_cabinet_no_top_gjeoer_0", "top_cabinet_lkxmne_0")]
    for obj in [o for o in first if o is not None] + [o for o in rig.scene.objects if o not in first]:
        if not (obj.fixed_base and "cabinet" in obj.category):
            continue
        for name, joint in obj.joints.items():
            if joint.joint_type == JointType.JOINT_REVOLUTE and len(out) < n:
                out.append((obj, joint, str(joint.body1).rsplit("/", 1)[-1]))
    return out


@scenario("handle_pinch_tether", furniture=True)
def handle_pinch_tether(rig):
    """A cabinet door's free edge pinched between thumb and index pads (others curled), 3 doors x 3 offsets: it
    tethers (named), the arm pulling home turns the door, and opening the hand releases it."""
    rows = []
    for obj, joint, link_name in furniture_doors(rig):
        p0, q0 = obj.get_position_orientation()
        for shift in (0., .004, -.004):
            rig.release_all()
            obj.set_position_orientation(position=p0, orientation=q0)
            ajar = .5 if abs(float(joint.upper_limit)) >= abs(float(joint.lower_limit)) else -.5
            joint.set_pos(ajar)
            rig.step(2)
            pinch, u, w, shape = pinch_point(rig)
            put_edge_in_pinch(rig, obj, obj.links[link_name], pinch, u, w, shift)
            rig.step(1)
            attached, seen = None, set()
            for k in range(60):
                if k < 30:
                    rig.set_hand("right", 0., shape(.2 + .65 * (k + 1) / 30))
                rig.step()
                seen |= set(rig.hand_links("right", obj))
                if attached is None and rig.held("right"):
                    attached = k
            status, q1 = rig.status(), float(joint.get_state()[0][0])
            g = rig.groups["right_arm"]
            start, home = rig.cmd[list(g.indices)].copy(), rig.home[list(g.indices)]
            for k in range(75):
                rig.cmd[list(g.indices)] = start + (home - start) * (k + 1) / 75
                rig.step()
            q2, after = float(joint.get_state()[0][0]), rig.status()
            rig.set_hand("right", 0.)
            released = rig.step(10, each=lambda k: rig.status()[0] == "none")
            two = sum(1 for part in ("thumb", "finger") if part in {
                "thumb" if n.split("_", 1)[1].startswith("thumb") or n.split("_", 1)[1] in ("hand_C_MC", "pinky_MC")
                else "finger" for n in seen})
            rows.append({"object": obj.name, "link": link_name, "shift": shift, "opposition_seen": two == 2,
                         "attach_step": attached, "status": status, "turned_rad": round(abs(q2 - q1), 3),
                         "after_pull": after, "released_after": released})
            rig.arm_forward("right")
            rig.release_all()
            rig.step(20)
        obj.set_position_orientation(position=p0, orientation=q0)
        joint.set_pos(0.)
        rig.step(5)
    att = [r for r in rows if r["attach_step"] is not None]
    return expect({"trials": rows, "tethered": f"{len(att)}/{len(rows)}"},
                  most=len(att) >= len(rows) * 2 // 3,
                  named=all(r["status"][0] == "tethered" and r["status"][1] == f"{r['object']}/{r['link']}" for r in att),
                  released=all(r["released_after"] is not None for r in att))


def free_squat_height(rig, seconds=3.):
    pelvis = rig.groups["pelvis"].indices[0]
    rig.ready("right")
    rig.cmd[pelvis] = .45
    rig.step(round(seconds / DT))
    z = rig.link("pelvis")[2, 3] - rig.sim.embodiment.floor
    rig.cmd[pelvis] = rig.home[pelvis]
    rig.step(100)
    return float(z)


@scenario("room_door_tether", furniture=True)
def room_door_tether(rig):
    """The 10.8 kg room door (door_tonlzz_0) pinched at its free edge: it tethers; walking back 3 s and bending the
    elbow keep the tether and turn the leaf; a 3 s squat on the tether lets the body down (as low as a free squat
    within 5 cm, or the attachment lets go first); opening releases."""
    door = rig.obj("door_tonlzz_0")
    if door is None:
        return expect({"door": None}, present=False)
    z_free = free_squat_height(rig)
    joint = next(j for j in door.joints.values() if str(j.body1).rsplit("/", 1)[-1] == "leaf")
    p0, q0 = door.get_position_orientation()
    base, pelvis = list(rig.groups["base"].indices), rig.groups["pelvis"].indices[0]
    pinch, u, w, shape = pinch_point(rig)
    put_edge_in_pinch(rig, door, door.links["leaf"], pinch, u, w)
    rig.step(1)
    for k in range(60):
        rig.set_hand("right", 0., shape(.2 + .65 * min(1., (k + 1) / 30)))
        rig.step()
    row = {"z_free": round(z_free, 3), "attached": rig.status()}
    if rig.status()[0] != "tethered":
        rig.set_hand("right", 0.)
        rig.step(20)
        rig.release_all()
        door.set_position_orientation(position=p0, orientation=q0)
        return expect(row, tethered=False)
    q_start = float(joint.get_state()[0][0])
    rig.cmd[base] = (-.3, 0., 0.)
    rig.step(150)
    rig.cmd[base] = 0.
    rig.set_arm("right", elbow=1.2)
    rig.step(50)
    row.update(turned_rad=round(abs(float(joint.get_state()[0][0]) - q_start), 3), after_pull=rig.status())
    rig.cmd[pelvis] = .45
    rig.step(150)
    row.update(z_squat=round(float(rig.link("pelvis")[2, 3] - rig.sim.embodiment.floor), 3), after_squat=rig.status(),
               fallen=bool(rig.sim.embodiment.fallen))
    rig.cmd[pelvis] = rig.home[pelvis]
    rig.step(100)
    rig.set_hand("right", 0.)
    row["released_after"] = rig.step(10, each=lambda k: rig.status()[0] == "none")
    rig.arm_forward("right")
    rig.release_all()
    joint.set_pos(0.)
    door.set_position_orientation(position=p0, orientation=q0)
    rig.step(50)
    return expect(row, kept_while_opening=row["after_pull"][0] == "tethered", turned=row["turned_rad"] >= .05,
                  body_not_hung=row["z_squat"] <= z_free + .05 or row["after_squat"][0] == "none",
                  released=row["released_after"] is not None)


def arm_fk(rig, side):
    """Forward kinematics of an arm from the composed URDF (g1_model.compose): q -> the palm frame in the world."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from rosbridge_behavior import g1_model
    joints = {j.find("child").get("link"): j for j in ET.fromstring(g1_model.compose(rig.sim.embodiment.source)).findall("joint")}
    chain, link = [], f"{side}_hand_C_MC"
    while link != "torso_link":
        chain.append(joints[link])
        link = joints[link].find("parent").get("link")
    chain.reverse()

    def palm(q):
        T = rig.link("torso_link").copy()
        for j in chain:
            o = j.find("origin")
            step = np.eye(4)
            if o is not None:
                step[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
                step[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            T = T @ step
            if j.get("type") == "revolute":
                R = np.eye(4)
                R[:3, :3] = Rotation.from_rotvec(np.array([float(v) for v in j.find("axis").get("xyz").split()])
                                                 * q[j.get("name")]).as_matrix()
                T = T @ R
        return T
    names = [j.get("name") for j in chain if j.get("type") == "revolute"]
    limits = [(float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))) for j in chain
              if j.get("type") == "revolute"]
    return palm, names, limits


@scenario("two_hands_hand_over")
def two_hands_hand_over(rig):
    """The glass held by the right hand; the left arm put (by IK) with its palm on the far side of the glass, then the
    left hand closes: the left takes it over in one step (the right's attachment ends as the left's begins, never
    both), nothing drops; the right then opens and the left keeps it; a hand that handed over does not take it back."""
    from scipy.optimize import minimize
    glass = rig.obj(GLASS)
    if grasp_somewhere(rig, glass, "right") is None:
        return expect({"right_attached": False}, right_attached=False)
    rig.step(20)
    palm, names, limits = arm_fk(rig, "left")
    idx = [rig.names.index(n) for n in names]
    target, facing = glass.get_position_orientation()[0].numpy(), -rig.link("right_hand_C_MC")[:3, 0]

    def cost_at(q, aim):
        T = palm(dict(zip(names, q)))
        return float(100 * np.sum(((T @ np.r_[.0755, -.01, .08, 1.])[:3] - aim) ** 2) + .05 * (1 - T[:3, 0] @ facing))
    seeds = (rig.cmd[idx].copy(), np.array([-1.2, .3, 0., .3, 1.57, 0., 0.]), np.array([-1.3, .2, .3, .2, -1.57, 0., 0.]))
    z0, both, changes, took, attempts = glass.get_position_orientation()[0][2].item(), 0, [], None, 0
    for shift in (0., .015, -.015, .03):  # the left palm's placement, moved along the right palm's normal if it missed
        attempts += 1
        aim = target + facing * shift
        q = min((minimize(lambda q: cost_at(q, aim), seed, bounds=limits, method="L-BFGS-B") for seed in seeds),
                key=lambda r: r.fun).x
        rig.set_hand("left", 0.)
        start = rig.cmd[idx].copy()
        for k in range(50):
            rig.cmd[idx] = start + (q - start) * min(1., (k + 1) / 35)
            rig.step()
        prev = None
        for k in range(60):
            rig.set_hand("left", .85 * min(1., (k + 1) / 40))
            rig.step()
            now = (rig.status("right")[0], rig.status("left")[0])
            both += now == ("held", "held")
            if now != prev:
                changes.append((attempts, k, *now))
                prev = now
        took = next((k for a, k, r, l in changes if l == "held"), None)
        if took is not None or rig.status("right")[0] != "held":
            break
    rig.set_hand("right", 0.)
    rig.step(40)
    row = {"changes": changes, "attempts": attempts, "both_held_steps": both,
           "after_right_opened": (rig.status("right"), rig.status("left")),
           "dz": round(glass.get_position_orientation()[0][2].item() - z0, 3)}
    rig.set_hand("left", 0.)
    rig.step(30)
    row["left_opened"] = rig.status("left")
    rig.release_all()
    rig.park(glass)
    return expect(row, never_both=both == 0, took_over=took is not None,
                  kept=row["after_right_opened"][1][0] == "held" and row["dz"] > -.03,
                  released=row["left_opened"][0] == "none")


# can_meat's movable furniture parts with a handle: object, link, handle point,
# the handle's outward normal and its long axis, in the world at the task's start.
HANDLES = {"cabinet door": ("bottom_cabinet_no_top_gjeoer_0", "doorleft2", (4.4625, .385, .7885), (1., 0, 0),
                            (0, 0, 1.)),
           "drawer": ("bottom_cabinet_rhdbzv_1", "left_drawer", (.8315, 1.133, .3345), (-1., 0, 0), (0, 0, 1.)),
           "microwave door": ("microwave_abzvij_0", "link_0", (4.4435, .915, 1.045), (1., 0, 0), (0, 1., 0))}


def hand_load(rig, side="right"):
    """The force (N) the arm passes to the hand: the palm link's incoming joint force (the world's load gauge)."""
    import torch as th
    from omnigibson.utils.usd_utils import ControllableObjectViewAPI, get_robot_kinematic_tree_pattern
    root = rig.robot.articulation_root_path
    view = ControllableObjectViewAPI._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(root)]
    row = view._idx[root]
    forces = th.as_tensor(view._view.get_link_incoming_joint_force()).reshape(view._view.count, -1, 6)
    return float(th.norm(forces[row, view._link_idx[row][f"{side}_hand_C_MC"], :3]))


def tether_handle(rig, obj, link_name, centre, normal, axis, back, side="right"):
    """The handle placement: the fixed object moved so that the handle lies between the thumb tip and the index and
    middle tips, `back` m towards the back of the hand, then the hand closed to 0.85. Returns the hand's status, the
    part's joint and the handle's opening direction (its motion when the joint opens, measured before the grasp)."""
    from scipy.spatial.transform import Rotation
    from omnigibson.utils.constants import JointType
    unit = lambda v: v / np.linalg.norm(v)
    rig.ready(side, .35, n=40)
    thumb = (rig.link(f"{side}_thumb_DP") @ np.r_[.018, 0., 0., 1.])[:3]
    tips = sum((rig.link(f"{side}_{f}_DP") @ np.r_[.016, 0., 0., 1.])[:3] for f in ("index", "middle")) / 2
    z = rig.link(f"{side}_hand_C_MC")[:3, 2]
    u = unit(tips - thumb)
    n = unit(-(z - (z @ u) * u))
    target = (thumb + tips) / 2 + n * back
    rig.set_hand(side, .1)
    rig.step(30)
    A, B = np.c_[normal, axis, np.cross(normal, axis)], np.c_[n, u, np.cross(n, u)]
    turn = B @ A.T
    p, q = obj.get_position_orientation()
    rig.place(obj, target + turn @ (p.numpy() - np.asarray(centre)), turn @ Rotation.from_quat(q.numpy()).as_matrix())
    rig.step(10)
    joint = next(j for j in obj.joints.values() if j.joint_type != JointType.JOINT_FIXED
                 and str(j.body1).endswith("/" + link_name))
    q0, link = float(joint.get_state()[0][0]), obj.links[link_name]
    step = .05 if float(joint.upper_limit) - q0 >= q0 - float(joint.lower_limit) else -.05
    before = rig.pose_matrix(*link.get_position_orientation())
    joint.set_pos(q0 + step)
    rig.step(1)
    after = rig.pose_matrix(*link.get_position_orientation())
    opening = unit((after @ np.linalg.inv(before) @ np.r_[target, 1.])[:3] - target)  # the handle as the part opens
    joint.set_pos(q0)
    rig.step(1)
    rig.set_hand(side, .85)
    rig.step(60)
    return rig.status(side), joint, opening


@scenario("door_cycles", furniture=True)
def door_cycles(rig, n=20, side="right"):
    """A cabinet door, a drawer and the microwave door tethered by a handle pinch (tether_handle's placement, up to six
    depths), then opened and closed n times by moving the hand along the handle's opening direction: the shoulder pitch
    (0.35 rad, 1 s each way, 0.5 s holds) when that direction is vertical, else walking at 0.2 m/s. Every cycle stays
    tethered and moves the part, the load limit never ends an opening (load p50/p99/max reported), opening the hand
    releases it."""
    rows, ok = {}, True
    g = rig.groups[f"{side}_arm"]
    pitch = g.indices[rig.joints(f"{side}_arm").index(f"{side}_shoulder_pitch_joint")]
    for label, (name, link_name, centre, normal, axis) in HANDLES.items():
        rig.fresh()  # the robot and the furniture at the task's start (the handle table's frame)
        rig.ready(side, .35, n=40)
        base, start = rig.cmd[pitch], rig.link(f"{side}_hand_C_MC")[:3, 3].copy()
        rig.cmd[pitch] = base + .1  # how the free hand moves when the shoulder pitch grows
        rig.step(25)
        raise_dir = rig.link(f"{side}_hand_C_MC")[:3, 3] - start
        rig.cmd[pitch] = base
        rig.step(25)
        obj = rig.obj(name)
        if obj is None:
            continue
        p0, q0 = obj.get_position_orientation()
        for back in (.012, .014, .010, .008, .016, .006):
            status, joint, opening = tether_handle(rig, obj, link_name, centre, normal, axis, back, side)
            if status[0] == "tethered":
                break
            rig.release_all()
            obj.set_position_orientation(position=p0, orientation=q0)
            rig.step(20)
        if status[0] != "tethered":
            rows[label] = {"tethered": False}
            ok = False
            continue
        vertical = abs(opening[2]) > .6
        sign = float(np.sign(raise_dir @ opening))
        fwd = rig.link("pelvis")[:3, 0] * [1, 1, 0]
        fwd /= np.linalg.norm(fwd)
        walk = np.array([opening @ fwd, opening @ np.cross([0., 0., 1.], fwd)])
        walk = .2 * walk / max(np.linalg.norm(walk), 1e-6)
        loads, travel, kept = [], [], 0
        for cycle in range(n):
            q = []
            for k in range(150):
                a = min(k, 50) / 50 if k < 75 else max(0., 1 - (k - 75) / 50)
                if vertical:
                    rig.cmd[pitch] = base + sign * .35 * a
                else:
                    s = 1 if k < 62 else 0 if k < 75 else -1 if k < 137 else 0
                    rig.cmd[list(rig.groups["base"].indices)[:2]] = s * walk
                rig.step()
                q.append(float(joint.get_state()[0][0]))
                loads.append(hand_load(rig, side))
            rig.cmd[list(rig.groups["base"].indices)] = 0.
            kept += rig.status(side)[0] == "tethered"
            travel.append(round(max(q) - min(q), 3))
            if rig.status(side)[0] != "tethered":
                break
        rig.cmd[pitch] = base
        rig.set_hand(side, 0.)
        released = rig.step(10, each=lambda k: rig.status(side)[0] == "none")
        rig.release_all()
        obj.set_position_orientation(position=p0, orientation=q0)
        rig.step(20)
        rows[label] = {"back_m": back, "by": "shoulder" if vertical else "walking", "tethered_cycles": kept,
                       "travel": travel, "load_p50_n": round(float(np.percentile(loads, 50)), 1),
                       "load_p99_n": round(float(np.percentile(loads, 99)), 1), "load_max_n": round(max(loads), 1),
                       "released_after": released}
        ok = ok and kept == n and min(travel) > .002 and released is not None
    return expect(rows, every_cycle_tethered_and_moving=ok)


if __name__ == "__main__":
    main()
