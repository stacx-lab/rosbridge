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
from .environment import data_root, rotation, values
from .g1_model import ARMS, BODY, CAMERA, LEGS, MODEL, SIDES, WAIST, WBC, arm_links, standing_height
# The values a worker's outcome depends on live in g1_rules, which the device texts and the task note state.
from .g1_rules import (CLIP, FALL_HEIGHT, FALL_HOLD_S, FALL_TILT_DEG, FALL_UPRIGHT_DEG, GRASP_CLOSURE, PELVIS_LIMITS,
                       SELF_COLLISIONS, VELOCITY_LIMIT)

NAME = "robot_g1"
LOWER = LEGS + WAIST
# GR00T's lower-body policy drives the waist itself from the torso orientation it is given, and realizes about half of
# the pitch and roll (its own waist targets match the measured joints, so it is the policy's choice, not drive
# tracking) while tracking yaw. Doubling pitch and roll before the policy makes the measured waist follow the device's
# targets (0.3 -> 0.34, 0.5 -> 0.51 pitch; standing and walking stay balanced); yaw is passed as is.
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
# grasping, as the BEHAVIOR challenge's R1Pro uses: an object between the thumb and the other fingers, touched by two
# of them while the hand closes, is attached to the hand until it opens). ROSBRIDGE_GRASPING selects it.
GRASPING_MODES = ("physical", "assisted")
# A hand applies a grasp while its flexion joints are commanded past GRASP_CLOSURE of their closing range on average;
# the abduction joints (_AA) do not count. OmniGibson's own test (any joint off its open limit) is for parallel grippers.
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
    """What a hand holds: "none", "held" (an object fixed to the hand) or "tethered" (a door, drawer or lid of a piece
    of furniture, on a pivot). Physical grasping never attaches."""
    if robot.grasping_mode != "assisted" or robot._ag_obj_in_hand[arm] is None:
        return "none"
    params = robot._ag_obj_constraint_params[arm]
    if params is None:  # released; the release window has not ended
        return "none"
    return "tethered" if params["joint_type"] == "SphericalJoint" else "held"


def assisted_grasping(robot, hands, attached=lambda arm: None):
    """The G1's assisted grasp test, on finger flexion (its ray points are in the robot definition, g1_model).
    attached(arm) is called when a hand attaches an object."""
    import types
    import omnigibson as og
    import torch as th
    from omnigibson.controllers import ControllerView
    from omnigibson.robots.robot import m

    names = list(robot.dof_names_ordered)
    flexion = {side: th.tensor([i for i, joint in enumerate(hands[side]) if "_AA" not in joint]) for side in hands}

    def closure(arm):
        group_key, index = robot._controllers[f"gripper_{arm}"]
        control = ControllerView.get_control(group_key, index)
        if control is None:
            return 0.
        joints = th.tensor([names.index(joint) for joint in hands[arm]])[flexion[arm]]
        lower, upper = robot.joint_lower_limits[joints], robot.joint_upper_limits[joints]
        return float(((control[flexion[arm]] - lower) / (upper - lower).clamp(min=1e-6)).clamp(0, 1).mean())

    def handle(self):
        # OmniGibson's _handle_assisted_grasping with the closure test above.
        for arm in self.arm_names:
            applying = closure(arm) > GRASP_CLOSURE
            if self._ag_obj_in_hand[arm]:
                if self._ag_release_counter[arm] is not None:
                    self._handle_release_window(arm=arm)
                elif not applying:
                    self._release_grasp(arm=arm)
            elif applying:
                target = self._calculate_in_hand_object(arm=arm)
                if self._ag_grasp_counter[arm] is not None:
                    if target is None:
                        self._ag_grasp_counter[arm] = None
                    else:
                        self._ag_grasp_counter[arm] += 1
                        if self._ag_grasp_counter[arm] * og.sim.get_physics_dt() >= m.GRASP_WINDOW:
                            self._maybe_establish_grasp(target_obj=target[0], target_link_name=target[1], arm=arm)
                            self._ag_grasp_counter[arm] = None
                            if self._ag_obj_in_hand[arm] is not None:
                                attached(arm)
                elif target is not None:
                    self._ag_grasp_counter[arm] = 0
            else:
                self._ag_grasp_counter[arm] = None

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
            assisted_grasping(robot, self.hands, self._attached)
        self.attachments = 0

    def _attached(self, arm):
        self.attachments += 1

    def status(self, episode):
        """Whether each hand holds something (assisted grasping), for the hand devices' attachment topics, and
        whether the body has fallen, for the base's."""
        return {**{f"{side}_hand/attachment": attachment(episode.robot, side) for side in SIDES},
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
