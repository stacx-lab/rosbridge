"""A Unitree G1 with SharpaWave hands, walking on GR00T WholeBodyControl's decoupled lower body.

GR00T's balance and walking policies drive the legs and waist at their 50 Hz rate from a body
twist (vx, vy, wz in m/s and rad/s) and the torso orientation that the waist target describes,
as GR00T's decoupled whole-body policy derives it. Arms and hands track absolute joint targets
on position drives. The policies were trained for a G1 with Dex3 hands in MuJoCo; here they run
in PhysX, at their 200 Hz physics, with the heavier Sharpa hands.
"""
import json
import sys

import numpy as np
import yaml

from rosbridge.core import ActionLayout, CommandGroup
from .backend import Overview
from .environment import data_root, rotation, values
from .g1_model import ARMS, BODY, CAMERA, LEGS, MODEL, SIDES, WAIST, WBC, arm_links, standing_height

NAME = "robot_g1"
LOWER = LEGS + WAIST
# GR00T's keyboard navigation limits for vx, vy (m/s) and wz (rad/s).
VELOCITY_LIMIT = 0.5
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
# The overview spectator stands front-left or front-right of the G1, near its head height and
# aimed at its hands, and turns after it; it takes another bearing only where walls crowd those.
OVERVIEW_DISTANCE, OVERVIEW_EYE, OVERVIEW_TARGET = 1.8, 1.4, .8  # metres; heights above the floor
OVERVIEW_BEARINGS = tuple(np.deg2rad([45, -45, 90, -90, 0, 135, -135, 180]))


class Body:
    """GR00T's robot model as its lower-body policy uses it: body joints in BODY order."""

    def get_joint_group_indices(self, group):
        return list(range({"body": len(BODY), "lower_body": len(LOWER)}[group]))


def axis_rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def torso_rpy(yaw, roll, pitch):
    """The torso orientation of waist joints (yaw about z, roll about x, pitch about y, in
    chain order) as roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    matrix = axis_rotation(2, yaw) @ axis_rotation(0, roll) @ axis_rotation(1, pitch)
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    return np.arctan2(matrix[2, 1], matrix[2, 2]), pitch, np.arctan2(matrix[1, 0], matrix[0, 0])


class G1Sharpa:
    pose_key = "r1pro"  # the presampled R1Pro start of each task instance
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
        self.overview = Overview(OVERVIEW_DISTANCE, OVERVIEW_EYE - pelvis, OVERVIEW_TARGET - pelvis,
                                 OVERVIEW_BEARINGS, follow=True)
        self.arm_links = arm_links(self.source)

    def _controller(self, joints):
        return {"name": "JointController", "motor_type": "position", "use_delta_commands": False,
                "use_impedances": False, "command_input_limits": None, "command_output_limits": None,
                "isaac_kp": [self.drives[joint]["stiffness"] for joint in joints],
                "isaac_kd": [self.drives[joint]["damping"] for joint in joints]}

    def robot_config(self):
        parts = {"trunk": LOWER, **{f"arm_{side}": ARMS[side] for side in SIDES},
                 **{f"gripper_{side}": self.hands[side] for side in SIDES}}
        return {"model": MODEL, "name": NAME, "fixed_base": False, "self_collisions": False,
                "action_normalize": False, "action_type": "continuous", "grasping_mode": "physical",
                "obs_modalities": ["rgb", "depth_linear"],
                "sensor_config": {"VisionSensor": {"sensor_kwargs": {
                    "image_height": HEAD_RESOLUTION, "image_width": HEAD_RESOLUTION}}},
                "controller_config": {part: self._controller(joints) for part, joints in parts.items()}}

    def configure(self, robot):
        import torch

        robot.sensors[self.cameras["head_camera"]].horizontal_aperture = HEAD_HORIZONTAL_APERTURE
        # OmniGibson makes every link below a moving joint weightless, as ideal gravity
        # compensation. GR00T balances a body with weight: gravity acts on the pelvis, legs and
        # torso, while the arms and hands stay compensated, as RoboEnv's SIMPLE G1 compensates
        # its arms.
        for name, link in robot.links.items():
            if name not in self.arm_links:
                link.enable_gravity()
        # Stand in GR00T's default posture with open hands.
        robot.reset_joint_pos = torch.tensor([self.pose.get(name, 0.) for name in robot.dof_names_ordered])

    def start_pose(self, pose):
        """The pelvis, the robot's root, above the floor-level start pose at standing height."""
        import torch

        position = values(pose["position"]) + [0., 0., self.height]
        return torch.as_tensor(position, dtype=torch.float32), pose["orientation"]

    def layout(self, robot):
        """The base twist, then the waist, arm and hand joints of each device."""
        names = list(robot.dof_names_ordered)
        lower, upper = values(robot.joint_lower_limits), values(robot.joint_upper_limits)
        self.devices = {"torso": WAIST, **{f"{side}_arm": ARMS[side] for side in SIDES},
                        **{f"{side}_hand": self.hands[side] for side in SIDES}}
        self.dofs = {device: np.array([names.index(joint) for joint in joints])
                     for device, joints in self.devices.items()}
        self.body = np.array([names.index(joint) for joint in BODY])
        channels = ["vx", "vy", "wz"]
        self.home = [0.] * 3
        limit = (VELOCITY_LIMIT,) * 3
        groups = [CommandGroup("base", (0, 1, 2), tuple(-v for v in limit), limit, velocity=True)]
        for device, joints in self.devices.items():
            dofs = self.dofs[device]
            groups.append(CommandGroup(device, tuple(range(len(channels), len(channels) + len(joints))),
                                       tuple(lower[dofs]), tuple(upper[dofs])))
            channels += joints
            self.home += [self.pose.get(joint, 0.) for joint in joints]
        order = list(robot.controller_order)
        if order != ["trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]:
            raise ValueError(f"unexpected G1 controllers: {order}")
        return ActionLayout(tuple(channels), tuple(groups))

    def reset(self, episode):
        """A fresh lower-body policy that stands the G1 still before the episode starts."""
        root = str(self.source / WBC.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

        self.policy = G1GearWbcPolicy(Body(), str(self.config), POLICIES)
        self.policy.use_policy_action = True
        base, home = episode.robot.base_footprint_link, np.asarray(self.home)
        for step in range(STABILIZE_STEPS):
            episode.step(self.action(episode, home))
            twist = np.r_[values(base.get_linear_velocity()), values(base.get_angular_velocity())]
            speed = np.abs(twist).max()
            if step >= SETTLE_STEPS and speed < STILL:
                return
        raise RuntimeError("the G1 did not stand still within its standing initialization")

    def state(self, episode):
        """The base as odometry (x, y, yaw) and body twist (vx, vy, wz); joints in radians."""
        q, dq = values(episode.robot.get_joint_positions()), values(episode.robot.get_joint_velocities())
        pose, twist = episode.odometry()
        positions = [*pose, *(q[self.dofs[device]] for device in self.devices)]
        velocities = [*twist, *(dq[self.dofs[device]] for device in self.devices)]
        return np.hstack(positions), np.hstack(velocities)

    def action(self, episode, command):
        """GR00T's lower-body targets for this period, then the arm and hand targets, in the
        native controller order."""
        robot, start = episode.robot, 3
        targets = {}
        for device, joints in self.devices.items():
            targets[device], start = command[start:start + len(joints)], start + len(joints)
        q, dq = values(robot.get_joint_positions()), values(robot.get_joint_velocities())
        position, quaternion = robot.get_position_orientation()
        x, y, z, w = values(quaternion)
        frame = rotation(quaternion)
        # MuJoCo's free joint, which GR00T observes, reports angular velocity in the body frame.
        angular = frame.T @ values(robot.get_angular_velocity())
        self.policy.set_metric_command(command[:3], self.wbc["height_cmd"], torso_rpy(*targets["torso"]))
        self.policy.set_observation({
            "q": q[self.body], "dq": dq[self.body],
            "floating_base_pose": np.r_[values(position), w, x, y, z],
            "floating_base_vel": np.r_[frame.T @ values(robot.get_linear_velocity()), angular]})
        lower = self.policy.get_action()["body_action"][0][:len(LOWER)]
        native = [lower, *(targets[f"{side}_{part}"] for side in SIDES for part in ("arm", "hand"))]
        return np.hstack(native).astype(np.float32)
