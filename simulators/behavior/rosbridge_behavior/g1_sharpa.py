"""A Unitree G1 with SharpaWave hands, walking on GR00T WholeBodyControl's decoupled lower body.

GR00T's balance and walking policies drive the legs and waist at their 50 Hz rate from a body
twist (vx, vy, wz in m/s and rad/s) and the torso orientation that the waist target describes,
as GR00T's decoupled whole-body policy derives it. Arms and hands track absolute joint targets
on position drives. The policies were trained for a G1 with Dex3 hands in MuJoCo; here they run
in PhysX, at their 200 Hz physics, with the heavier Sharpa hands.
"""
import json
import os
import sys

import numpy as np
import yaml

from rosbridge.core import ActionLayout, CommandGroup
from .backend import Overview
from . import scene_doors
from .environment import data_root, rotation, values
from .g1_model import ARMS, BODY, CAMERA, LEGS, MODEL, SIDES, WAIST, WBC, arm_links, standing_height
# The values a worker's outcome depends on live in g1_rules, which the device texts and the task note state.
from .g1_rules import (CLIP, FALL_HEIGHT, FALL_HOLD_S, FALL_TILT_DEG, FALL_UPRIGHT_DEG, GRIP_CLAMP_RAD, LOAD_LIMIT_N,
                       PELVIS_LIMITS, RELEASE_FOLLOW_S, RELEASE_MARGIN_MPS, SELF_COLLISIONS,
                       VELOCITY_LIMIT)
from .g1_grasp import Grip, digit, digits, leaving_velocity, phalanx  # noqa: F401 (digits: for callers)

NAME = "robot_g1"
LOWER = LEGS + WAIST
# GR00T's lower-body policy drives the waist itself from the torso orientation it is given, and realizes about half of
# the pitch and roll (its own waist targets match the measured joints, so it is the policy's choice, not drive
# tracking) while tracking yaw. Doubling pitch and roll before the policy makes the measured waist follow the device's
# targets (0.3 -> 0.34, 0.5 -> 0.51 pitch; standing and walking stay balanced); yaw is passed as is.
# Extra assisted-grasp ray start points around the middle of the thumb's distal link, 12 mm off its axis (the tip is at
# 0.026 m along x), so that a rim pinched between the thumb's pad and a fingertip pad, short of the thumb's tip, lies on
# a ray; rays from the back of the thumb pass through the thumb itself and count nothing.
THUMB_PADS = ((.016, 0., .012), (.016, 0., -.012), (.016, .012, 0.), (.016, -.012, 0.))
TORSO_GAIN = {"yaw": 1., "roll": 2., "pitch": 2.}
HEAD_RESOLUTION = 720
HEAD_HORIZONTAL_APERTURE = 40.  # the evaluator's R1Pro head optics
POLICIES = "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"
SIMULATOR = "behavior-g1"
# GR00T's default (low-elbow) upper body; other arm and hand joints start at zero.
ARM_POSE = {"left_shoulder_roll_joint": .2, "right_shoulder_roll_joint": -.2}
# Standing initialization as in RoboEnv's SIMPLE G1: GR00T holds the default posture until the
# floating base is still, outside the episode. GR00T's standing sway in PhysX stays near 0.02 m/s
# and rad/s, above SIMPLE's MuJoCo threshold of 1e-3.
STABILIZE_STEPS, SETTLE_STEPS, STILL = 300, 100, .05
# How objects are held: "physical" (contact and friction only, as the hands are) or "assisted" (OmniGibson's assisted
# grasping, as the BEHAVIOR challenge's R1Pro uses: an object held between the thumb or palm and a finger while the hand
# grips it is attached to the hand until the hand opens; g1_grasp). ROSBRIDGE_GRASPING selects it.
GRASPING_MODES = ("physical", "assisted")
# A fall (FALL_*): GR00T does not get up again, and its policies kept acting on the fallen body: walking slid and spun
# it ~0.5 m on its back, and the balance policy alone moved it 0.3-0.6 m in 3 s. From then
# on the legs and waist hold the joint positions they had when it fell, as a real G1's damping mode stops its legs.
FALL_TILT, FALL_UPRIGHT = np.deg2rad(FALL_TILT_DEG), np.deg2rad(FALL_UPRIGHT_DEG)
# The overview spectators stand near the G1's head height and aim at its hands.
OVERVIEW_DISTANCE, OVERVIEW_EYE, OVERVIEW_TARGET = 1.8, 1.4, .8  # metres; heights above the floor


class Body:
    """GR00T's robot model as its lower-body policy uses it: body joints in BODY order."""

    def get_joint_group_indices(self, group):
        return list(range({"body": len(BODY), "lower_body": len(LOWER)}[group]))


def grasping_mode():
    mode = os.environ.get("ROSBRIDGE_GRASPING", "physical")
    if mode not in GRASPING_MODES:
        raise ValueError(f"ROSBRIDGE_GRASPING must be one of {GRASPING_MODES}, not {mode!r}")
    return mode


def attachment(robot, arm):
    """What a hand holds: "none", "held" (an object fixed to the hand) or "tethered" (a part on a movable joint of its
    object, such as a door, drawer or lid, on a pivot). Physical grasping never attaches."""
    params = _attached(robot, arm)
    if params is None:
        return "none"
    return "tethered" if params["joint_type"] == "SphericalJoint" else "held"


def attached_object(robot, arm):
    """What a hand is attached to: "<object>/<link>" (the object's name and the link the joint holds), or "none"."""
    params = _attached(robot, arm)
    return "none" if params is None else f"{params['target_obj'].name}/{params['target_link_name']}"


def _attached(robot, arm):
    if robot.grasping_mode != "assisted" or robot._ag_obj_in_hand[arm] is None:
        return None
    return robot._ag_obj_constraint_params[arm]  # None once released, while the release window has not ended


def grasp_load_limit():
    """The sustained load (N) beyond which a held grasp lets go, or None (no limit): LOAD_LIMIT_N unless
    ROSBRIDGE_GRASP_LOAD_N sets another (0 or off: none)."""
    value = os.environ.get("ROSBRIDGE_GRASP_LOAD_N")
    if value is None:
        return LOAD_LIMIT_N
    return None if value.strip().lower() in ("", "0", "off", "none") else float(value)


def assisted_grasping(robot, hands, attached=lambda arm: None, fallen=lambda: False):
    """The G1's assisted grasp: OmniGibson's mechanism (its rays, joint types, joint creation, release window and saved
    state) with g1_grasp's rules for when a grasp forms and when it ends, once per physics step:
    - a target is what OmniGibson's search picks (touched, on a ray between the thumb or palm and the fingertips,
      closest to the eef link), now also through the palm's contacts and with rays from the thumb's pad; a fixed
      object's base, and parts fixed to it, never are (no weld to the world), its doors, drawers and lids are;
    - it is grasped once g1_grasp.Grip's window shows opposition and squeeze; there is no search while the commanded
      closure falls (an opening hand never completes a grasp), while the body has fallen, or while a hand pose is not
      finite;
    - an object one hand holds is taken over by the other hand when that hand's grasp forms: the first hand's
      attachment ends in the same step, after the new joint exists (a hand-over without a drop, and never a rigid
      two-hand chain); the first hand then presses lightly and grasps again only after it opens; and when the first
      hand opens while the other touches the object from both sides and grips it, the other takes it at once;
    - while it is held, the hand's flexion targets are clamped (g1_grasp.clamp), and the grasp lets go when the
      gripping digits stop gripping or are opened and no finger touches the object any more (LET_GO_S at most), or,
      for a held object (not a tether to a door, drawer or lid, which OmniGibson never breaks either), when the load the
      arm passes to the hand stays above grasp_load_limit();
    - a released free object leaves at the hand's velocity, and while it still touches the hand (RELEASE_FOLLOW_S at
      most) the fingers stay clamped and it moves no faster from the hand than falling gives (g1_grasp.leaving_velocity):
      no fling.
    attached(arm) is called when a hand attaches an object; fallen() says whether the body has fallen."""
    import types
    import omnigibson as og
    import torch as th
    from omnigibson.controllers import ControllerView
    import omnigibson.utils.transform_utils as T
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
    from omnigibson.utils.sampling_utils import raytest_batch
    from omnigibson.utils.usd_utils import ControllableObjectViewAPI, RigidContactAPI, get_robot_kinematic_tree_pattern

    dt, root = og.sim.get_physics_dt(), robot.articulation_root_path
    names = list(robot.dof_names_ordered)
    flexion = {arm: [i for i, joint in enumerate(hands[arm]) if "_AA" not in joint] for arm in hands}
    dofs = {arm: th.tensor([names.index(hands[arm][i]) for i in flexion[arm]]) for arm in hands}
    flexion_digit = {arm: [digit(hands[arm][i]) for i in flexion[arm]] for arm in hands}
    sensing = {arm: {link.prim_path for link in robot.finger_links[arm]} | {robot.links[f"{arm}_hand_C_MC"].prim_path}
               for arm in hands}  # the finger links and the palm
    ray_links = {arm: sorted({point.link_name for point in robot.assisted_grasp_start_points[arm]
                              + robot.assisted_grasp_end_points[arm]}) for arm in hands}
    # OmniGibson's rays run from the thumb's tip and the palm to the four fingertip pads (the robot definition); more
    # start around the thumb's distal link (THUMB_PADS), so that a pinch counts whose rim lies between the pads.
    point = lambda p: (p.link_name, th.as_tensor(p.position, dtype=th.float32))
    starts = {arm: [point(p) for p in robot.assisted_grasp_start_points[arm]]
              + [(f"{arm}_thumb_DP", th.tensor(pad)) for pad in THUMB_PADS] for arm in hands}
    ends = {arm: [point(p) for p in robot.assisted_grasp_end_points[arm]] for arm in hands}
    own = set(robot.link_prim_paths)
    limits = {arm: (robot.joint_lower_limits[dofs[arm]].tolist(), robot.joint_upper_limits[dofs[arm]].tolist())
              for arm in hands}
    grips = robot._g1_grips = {arm: Grip(dt, flexion_digit[arm], grasp_load_limit(), limits[arm]) for arm in hands}
    leaving = robot._g1_leaving = {arm: None for arm in hands}  # a released object near the hand: (obj, s, last touch s)
    offered = {arm: None for arm in hands}  # an object the other hand let go of this step: (obj, that hand)
    last = {arm: 0. for arm in hands}  # each hand's commanded closure at the previous physics step

    def flexion_state(arm):
        """The worker's flexion targets (the controller's control, as OmniGibson's own test reads it) and the measured
        flexion, or None before any control."""
        group_key, index = robot._controllers[f"gripper_{arm}"]
        control = ControllerView.get_control(group_key, index)
        if control is None:
            return None
        measured = th.as_tensor(ControllableObjectViewAPI.get_joint_positions(root))[dofs[arm]]
        return th.as_tensor(control)[flexion[arm]], measured

    def closure(arm, command):
        lower, upper = robot.joint_lower_limits[dofs[arm]], robot.joint_upper_limits[dofs[arm]]
        return float(((command - lower) / (upper - lower).clamp(min=1e-6)).clamp(0, 1).mean())

    def touching(arm):
        """Each body a hand link (finger link or palm) touched during the step's substeps -> the hand links."""
        out = {}
        for link, other in RigidContactAPI.get_contact_pairs(robot.scene.idx, sensing[arm], None, False):
            if other not in own:
                out.setdefault(other, set()).add(link)
        return out

    def on_rays(self, arm):
        """The bodies the first hit of any ray from a start point to an end point lands on (OmniGibson's
        _find_gripper_raycast_collisions with the points above); the robot's own links do not count."""
        points = []
        for link_name, offset in starts[arm] + ends[arm]:
            position, orientation = self.links[link_name].get_position_orientation()
            points.append(position + T.quat2mat(orientation) @ offset)
        n = len(starts[arm])
        origins = th.stack([points[i] for _ in ends[arm] for i in range(n)])
        targets = th.stack([points[n + j] for j in range(len(ends[arm])) for _ in range(n)]) + 1e-8
        return {hit["rigidBody"] for hit in raytest_batch(origins, targets, only_closest=True)
                if hit["hit"] and self.prim_path not in hit["rigidBody"]}

    def graspable(self, prim_path):
        """(object, link name) if a hand may attach the link: a dynamic link of a registered object, and of a fixed
        object only through a movable joint (OmniGibson's joint-type rule gives it a pivot); else None."""
        obj_path, link_name = prim_path.rsplit("/", 1)
        obj = self.scene.object_registry("prim_path", obj_path, None)
        if obj is None or link_name not in obj.links or not isinstance(obj.links[link_name], RigidDynamicPrim):
            return None
        if obj.fixed_base and self._get_assisted_grasp_joint_type(obj, link_name) != "SphericalJoint":
            return None  # a fixed object's base or a part fixed to it: the hand would be welded to the world
        return obj, link_name

    def candidate(self, arm, contacts, on_ray):
        """OmniGibson's choice (_calculate_in_hand_object): a touched, graspable link on a ray, closest to the eef."""
        eef = self.eef_links[arm].get_position_orientation()[0]
        found = []
        for path in set(contacts) & on_ray:
            target = graspable(self, path)
            if target is not None:
                found.append((float(th.norm(target[0].links[target[1]].get_position_orientation()[0] - eef)), path))
        return min(found)[1] if found else None

    def load(arm):
        """The force (N) the arm passes to the hand (the palm link's incoming joint): what the hand carries. The
        fingers' squeeze on a held object pushes back on the hand through the grasp joint, so it cancels within the
        hand and only the load from outside remains (the object's weight and motion, a pull, a press)."""
        view = ControllableObjectViewAPI._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(root)]
        row = view._idx[root]
        forces = th.as_tensor(view._view.get_link_incoming_joint_force()).reshape(view._view.count, -1, 6)
        return float(th.norm(forces[row, view._link_idx[row][f"{arm}_hand_C_MC"], :3]))

    def let_go_gently(self, arm, obj, seconds):
        """A released free object, still touching the hand: its velocity relative to the hand (the velocity the eef
        link has at the object's position) is limited by g1_grasp.leaving_velocity, so the contacts it was squeezed
        with separate it from the hand without flinging it; when let go it also turns with the hand."""
        if obj is None or obj.fixed_base:
            return
        eef = self.eef_links[arm]
        linear, angular = eef.get_linear_velocity(), eef.get_angular_velocity()
        hand = linear + th.linalg.cross(angular, obj.get_position_orientation()[0] - eef.get_position_orientation()[0])
        relative = obj.get_linear_velocity() - hand
        if not bool(th.isfinite(hand).all() and th.isfinite(relative).all()):
            return
        limited = th.tensor(leaving_velocity(relative.tolist(), seconds), dtype=hand.dtype)
        if not bool(th.allclose(limited, relative.to(hand.dtype))):
            obj.set_linear_velocity(hand + limited)
        if seconds == 0.:
            obj.set_angular_velocity(angular)

    def catch(self, arm, obj, giver, contacts, state):
        """A hand-over by letting go: the other hand opened its grip on obj this step, and this hand touches obj from
        both sides and grips it now (g1_grasp.Grip.grips), so it takes obj at once, as when its own grasp forms first,
        without waiting for its window (the giver held obj still, so the taker's contacts were not all on a ray)."""
        links = {link.prim_path for link in obj.links.values()}
        body = next((path for path in contacts if path in links), None)
        touch = [link for path in contacts if path in links for link in contacts[path]]
        if body is None or graspable(self, body) is None or not grips[arm].grips(touch, *(x.tolist() for x in state)):
            return False
        self._maybe_establish_grasp(target_obj=obj, target_link_name=body.rsplit("/", 1)[-1], arm=arm)
        if self._ag_obj_in_hand[arm] is None:
            return False
        grips[arm].hold(digits(touch))
        attached(arm)
        if leaving[giver] is not None and leaving[giver][0] is obj:
            leaving[giver] = None  # it is held again: the giver's release limit no longer applies
        return True

    def clamp_fingers(arm, command, measured):
        ControllableObjectViewAPI.set_joint_position_targets(root, th.minimum(command, measured + GRIP_CLAMP_RAD),
                                                             dofs[arm])

    def finite(self, arm):
        return all(bool(th.isfinite(self.links[name].get_position_orientation()[0]).all()) for name in ray_links[arm])

    def handle(self):
        # Replaces OmniGibson's _handle_assisted_grasping (its window counter is unused; Grip keeps the window).
        for arm in self.arm_names:
            grip, state = grips[arm], flexion_state(arm)
            caught, offered[arm] = offered[arm], None  # offered by the other hand on its last pass
            if leaving[arm] is not None:  # let go and still touching the hand (or on the step it parted): no fling
                obj, seconds, touched_at = leaving[arm]
                seconds += dt
                if any(path.startswith(obj.prim_path + "/") for path in touching(arm)):
                    touched_at = seconds
                parted = seconds - touched_at > dt + 1e-9
                leaving[arm] = None if parted or seconds > RELEASE_FOLLOW_S else (obj, seconds, touched_at)
                if leaving[arm] is not None:
                    let_go_gently(self, arm, obj, seconds)
                    if state is not None:  # nor does a squeeze still commanded push it out
                        clamp_fingers(arm, *state)
            if self._ag_obj_in_hand[arm] is not None and self._ag_release_counter[arm] is not None:
                self._handle_release_window(arm=arm)  # OmniGibson's release window (then the hand may grasp again)
                if self._ag_obj_in_hand[arm] is not None:
                    continue
            if self._ag_obj_in_hand[arm] is not None:
                obj = self._ag_obj_in_hand[arm]
                links = {link.prim_path for link in obj.links.values()}
                fingers_on = lambda: any(phalanx(finger) for body, fingers in touching(arm).items() if body in links
                                         for finger in fingers)  # an opened hand keeps it until its fingers left it
                tether = (_attached(self, arm) or {}).get("joint_type") == "SphericalJoint"
                load_n = None if tether else load(arm)  # a tether to a door, drawer or lid has no load limit
                if state is None or grip.keep(state[0].tolist(), state[1].tolist(), load_n, fingers_on):
                    self._release_grasp(arm=arm)
                    let_go_gently(self, arm, obj, 0.)
                    leaving[arm] = (obj, 0., 0.)
                    if grip.why != "load":  # opened: a hand that grips it now takes it (a hand-over)
                        offered.update({other: (obj, arm) for other in self.arm_names if other != arm})
                    grip.let_go()
                else:  # held: the fingers press it lightly (g1_grasp.clamp)
                    clamp_fingers(arm, *state)
                continue
            if state is None:
                grip.reset()
                continue
            now = closure(arm, state[0])
            opening, closing, last[arm] = now < last[arm] - 1e-6, now > last[arm] + 1e-6, now
            if grip.yielded:  # it handed its object over: it presses lightly and grasps again once it opens
                grip.yielded = not opening
                clamp_fingers(arm, *state)
            contacts = {} if opening or fallen() else touching(arm)
            if opening or fallen() or (contacts and not finite(self, arm)):
                grip.reset()
                continue
            if caught is not None and catch(self, arm, *caught, contacts, state):
                continue
            on_ray = on_rays(self, arm) if contacts else set()
            target = grip.observe(candidate(self, arm, contacts, on_ray), lambda link: contacts.get(link, ()),
                                  lambda link: link in on_ray, state[0].tolist(), state[1].tolist(), closing)
            if target is not None:
                obj, link_name = graspable(self, target)
                self._maybe_establish_grasp(target_obj=obj, target_link_name=link_name, arm=arm)
                if self._ag_obj_in_hand[arm] is not None:
                    grip.hold()
                    attached(arm)
                    for other in self.arm_names:  # a hand-over: the other hand's attachment ends in this same step,
                        if other != arm and (_attached(self, other) or {}).get("target_obj") is obj:  # before physics
                            self._release_grasp(arm=other)
                            grips[other].let_go()
                            grips[other].yielded = True

    robot._handle_assisted_grasping = types.MethodType(handle, robot)


def axis_rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def tilt(quaternion, angular_velocity):
    """The pelvis's pitch and roll from level (R = Rz(yaw) Ry(pitch) Rx(roll), the yaw-only base frame times the tilt)
    and their rates, from its world orientation and angular velocity."""
    matrix = rotation(quaternion)
    pitch, roll = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0])), np.arctan2(matrix[2, 1], matrix[2, 2])
    p, q, r = matrix.T @ values(angular_velocity)  # body rates
    return (pitch, roll), (q * np.cos(roll) - r * np.sin(roll), p + (q * np.sin(roll) + r * np.cos(roll)) * np.tan(pitch))


def torso_rpy(yaw, roll, pitch):
    """The torso orientation of waist joints (yaw about z, roll about x, pitch about y, in
    chain order) as roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    matrix = axis_rotation(2, yaw) @ axis_rotation(0, roll) @ axis_rotation(1, pitch)
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    return np.arctan2(matrix[2, 1], matrix[2, 2]), pitch, np.arctan2(matrix[1, 0], matrix[0, 0])


class G1Sharpa:
    pose_key = "r1pro"  # the presampled R1Pro start of each task instance
    rendering = "on_demand"  # DLAA captures (backend.HISTORY_S, CUT_DELIVERIES)
    # This episode: periods since it started, how long the body has looked fallen, and whether (and when) it fell.
    ticks, lying_s, fallen, fall_time_s, frozen = 0, 0., False, None, None
    doors = guard = None  # the scene's room doors and the boundary walls' guard (configure)
    frequencies = {"action_frequency": 50, "rendering_frequency": 50, "physics_frequency": 200}
    cameras = {"head_camera": f"{NAME}:{CAMERA}:Camera:0"}

    def __init__(self):
        self.source = data_root(SIMULATOR).parent
        model = data_root(SIMULATOR) / "omnigibson-robot-assets/models" / MODEL
        self.drives = json.loads((model / "drives.json").read_text())
        self.hands = yaml.safe_load((model / f"{MODEL}.yaml").read_text())["manipulation"]["finger_joint_names"]
        self.config = self.source / WBC / "sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml"
        self.wbc = yaml.safe_load(self.config.read_text())
        self.pose = {**dict(zip(LOWER, self.wbc["default_angles"], strict=True)), **ARM_POSE}
        self.height = standing_height(self.source, self.pose)
        pelvis = self.wbc["height_cmd"]  # GR00T holds the pelvis this high above the floor
        self.overview = Overview(OVERVIEW_DISTANCE, OVERVIEW_EYE - pelvis, OVERVIEW_TARGET - pelvis)
        self.arm_links = arm_links(self.source)

    def _controller(self, joints):
        return {"name": "JointController", "motor_type": "position", "use_delta_commands": False,
                "use_impedances": False, "command_input_limits": None, "command_output_limits": None,
                "isaac_kp": [self.drives[joint]["stiffness"] for joint in joints],
                "isaac_kd": [self.drives[joint]["damping"] for joint in joints]}

    def robot_config(self):
        parts = {"trunk": LOWER, **{f"arm_{side}": ARMS[side] for side in SIDES},
                 **{f"gripper_{side}": self.hands[side] for side in SIDES}}
        return {"model": MODEL, "name": NAME, "fixed_base": False, "self_collisions": SELF_COLLISIONS,
                "action_normalize": False, "action_type": "continuous", "grasping_mode": grasping_mode(),
                "obs_modalities": ["rgb", "depth_linear"],
                "sensor_config": {"VisionSensor": {"sensor_kwargs": {
                    "image_height": HEAD_RESOLUTION, "image_width": HEAD_RESOLUTION}}},
                "controller_config": {part: self._controller(joints) for part, joints in parts.items()}}

    def configure(self, robot):
        import omnigibson as og
        import torch

        robot.sensors[self.cameras["head_camera"]].horizontal_aperture = HEAD_HORIZONTAL_APERTURE
        self.weigh(robot)
        # Gravity is set through physics views, and a simulator restart drops it: a reset whose scene restore adds or
        # removes a particle system or an object (one a transition made, such as cooked popcorn or a diced onion)
        # stops and plays the simulator, after which the weightless legs could not stand and every reset failed.
        og.sim.add_callback_on_play(name=f"{NAME}_weight", callback=lambda: self.weigh(robot))
        # Stand in GR00T's default posture with open hands.
        robot.reset_joint_pos = torch.tensor([self.pose.get(name, 0.) for name in robot.dof_names_ordered])
        if robot.grasping_mode == "assisted":
            assisted_grasping(robot, self.hands, self._attached, lambda: self.fallen)
        self.attachments = 0
        # The scene's room doors, read from the loaded scene: their facts for the world text, and an invisible wall in
        # each doorway that opens onto no floor (a room the activity does not load), through which robots fell out of
        # the world (scene_doors). A scene the code cannot read keeps its doors as they are; the log says why.
        try:
            self.doors = scene_doors.find(robot.scene, robot)
            scene_doors.block(self.doors)
        except Exception:
            import traceback
            print(f"scene doors not read: {traceback.format_exc()}", file=sys.stderr, flush=True)
            self.doors = None
        self.guard = scene_doors.Guard(self.doors)  # each wall is on only while the robot is near its doorway

    def _attached(self, arm):
        self.attachments += 1

    def status(self, episode):
        """Whether each hand holds something and what (assisted grasping), for the hand devices' attachment and
        attached_object topics, and whether the body has fallen, for the base's."""
        return {**{f"{side}_hand/attachment": attachment(episode.robot, side) for side in SIDES},
                **{f"{side}_hand/attached_object": attached_object(episode.robot, side) for side in SIDES},
                "base/fall": "fallen" if self.fallen else "upright"}

    def counts(self):
        """The episode's attachments (assisted grasping: objects or furniture parts a hand attached) and fall."""
        return {"attachments": self.attachments, "falls": int(self.fallen), "fall_time_s": self.fall_time_s}

    def _start_episode(self):
        self.attachments, self.ticks, self.lying_s, self.fallen, self.fall_time_s, self.frozen = 0, 0, 0., False, None, None

    def weigh(self, robot):
        """OmniGibson makes every link below a moving joint weightless, as ideal gravity compensation. GR00T balances a
        body with weight: gravity acts on the pelvis, legs and torso, while the arms and hands stay compensated, as
        RoboEnv's SIMPLE G1 compensates its arms."""
        for name, link in robot.links.items():
            if name not in self.arm_links:
                link.enable_gravity()

    def start_pose(self, pose):
        """The pelvis, the robot's root, above the floor-level start pose at standing height."""
        import torch

        position = values(pose["position"]) + [0., 0., self.height]
        self.floor = float(values(pose["position"])[2])  # the pelvis device measures its height from here
        return torch.as_tensor(position, dtype=torch.float32), pose["orientation"]

    def layout(self, robot):
        """The base twist, the pelvis height, then the waist, arm and hand joints of each device."""
        names = list(robot.dof_names_ordered)
        lower, upper = values(robot.joint_lower_limits), values(robot.joint_upper_limits)
        self.devices = {"torso": WAIST, **{f"{side}_arm": ARMS[side] for side in SIDES},
                        **{f"{side}_hand": self.hands[side] for side in SIDES}}
        self.dofs = {device: np.array([names.index(joint) for joint in joints])
                     for device, joints in self.devices.items()}
        self.body = np.array([names.index(joint) for joint in BODY])
        channels = ["vx", "vy", "wz", "pelvis_height"]
        self.home = [0.] * 3 + [self.wbc["height_cmd"]]
        limit = (VELOCITY_LIMIT,) * 3
        groups = [CommandGroup("base", (0, 1, 2), tuple(-v for v in limit), limit, velocity=True),
                  CommandGroup("pelvis", (3,), PELVIS_LIMITS[:1], PELVIS_LIMITS[1:])]
        for device, joints in self.devices.items():
            dofs = self.dofs[device]
            groups.append(CommandGroup(device, tuple(range(len(channels), len(channels) + len(joints))),
                                       tuple(lower[dofs]), tuple(upper[dofs])))
            channels += joints
            self.home += [self.pose.get(joint, 0.) for joint in joints]
        order = list(robot.controller_order)
        if order != ["trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]:
            raise ValueError(f"unexpected G1 controllers: {order}")
        # GR00T pitches the pelvis forward as it lowers (about 6 degrees at 0.62 m, 27 below 0.40): the pelvis device
        # reports that tilt from the yaw-only base frame, so frames on the pelvis can follow it. Targets beyond a limit
        # are clipped to it: a worker publishes on a topic and learns of a refusal only from its next wait's reply.
        return ActionLayout(tuple(channels), tuple(groups), measured=(("pelvis", ("pelvis_pitch", "pelvis_roll")),),
                            clip=CLIP)

    def reset(self, episode):
        """A fresh lower-body policy that stands the G1 still before the episode starts."""
        root = str(self.source / WBC.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

        self.policy = G1GearWbcPolicy(Body(), str(self.config), POLICIES)
        self.policy.use_policy_action = True
        base, home = episode.robot.base_footprint_link, np.asarray(self.home)
        self._start_episode()
        for step in range(STABILIZE_STEPS):
            episode.step(self.action(episode, home))
            twist = np.r_[values(base.get_linear_velocity()), values(base.get_angular_velocity())]
            speed = np.abs(twist).max()
            if step >= SETTLE_STEPS and speed < STILL:
                self._start_episode()
                return
        raise RuntimeError("the G1 did not stand still within its standing initialization")

    def state(self, episode):
        """The base as odometry (x, y, yaw) and body twist (vx, vy, wz); the pelvis height above the floor
        (m, m/s); joints in radians; then the measured pelvis pitch and roll (rad, rad/s)."""
        robot = episode.robot
        q, dq = values(robot.get_joint_positions()), values(robot.get_joint_velocities())
        pose, twist = episode.odometry()
        position, quaternion = robot.get_position_orientation()
        height = values(position)[2] - self.floor
        angles, rates = tilt(quaternion, robot.get_angular_velocity())
        positions = [*pose, height, *(q[self.dofs[device]] for device in self.devices), *angles]
        velocities = [*twist, values(robot.get_linear_velocity())[2], *(dq[self.dofs[device]] for device in self.devices),
                      *rates]
        return np.hstack(positions), np.hstack(velocities)

    def action(self, episode, command):
        """GR00T's lower-body targets for this period, then the arm and hand targets, in the
        native controller order."""
        robot, start = episode.robot, 4
        targets = {}
        for device, joints in self.devices.items():
            targets[device], start = command[start:start + len(joints)], start + len(joints)
        q, dq = values(robot.get_joint_positions()), values(robot.get_joint_velocities())
        position, quaternion = robot.get_position_orientation()
        if self.guard is not None and self.guard.walls:
            self.guard.update(values(position)[:2])
        x, y, z, w = values(quaternion)
        frame = rotation(quaternion)
        # MuJoCo's free joint, which GR00T observes, reports angular velocity in the body frame.
        angular = frame.T @ values(robot.get_angular_velocity())
        period = 1. / self.frequencies["action_frequency"]
        self.ticks += 1
        if not self.fallen:
            low = values(position)[2] - self.floor < FALL_HEIGHT
            if frame[2, 2] < np.cos(FALL_TILT) or low:  # frame[2, 2]: the cosine of the pelvis's tilt
                self.lying_s += period
            elif frame[2, 2] > np.cos(FALL_UPRIGHT):
                self.lying_s = 0.
            if self.lying_s >= FALL_HOLD_S - 1e-9:
                self.fallen, self.fall_time_s = True, round(self.ticks * period, 6)
                self.frozen = q[self.body][:len(LOWER)].copy()
        if self.fallen:
            lower = self.frozen  # the legs and waist hold still after a fall
        else:
            yaw, roll, pitch = targets["torso"]
            self.policy.set_metric_command(command[:3], command[3], torso_rpy(
                TORSO_GAIN["yaw"] * yaw, TORSO_GAIN["roll"] * roll, TORSO_GAIN["pitch"] * pitch))
            self.policy.set_observation({
                "q": q[self.body], "dq": dq[self.body],
                "floating_base_pose": np.r_[values(position), w, x, y, z],
                "floating_base_vel": np.r_[frame.T @ values(robot.get_linear_velocity()), angular]})
            lower = self.policy.get_action()["body_action"][0][:len(LOWER)]
        native = [lower, *(targets[f"{side}_{part}"] for side in SIDES for part in ("arm", "hand"))]
        return np.hstack(native).astype(np.float32)
