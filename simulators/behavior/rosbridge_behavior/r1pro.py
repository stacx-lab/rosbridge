"""BEHAVIOR's R1Pro as the official evaluator configures it (OmniGibson eval/r1pro.yaml).

Commands are the evaluator's action vector: a normalized body twist for the holonomic base,
absolute torso and arm joint positions, and the smooth gripper value from -1 (closed) to 1
(open) with assisted grasping. Cameras are the evaluator's full-resolution RGB-D set.
"""
import importlib.util
from pathlib import Path

import numpy as np
import yaml

from rosbridge.core import ActionLayout, CommandGroup
from .backend import Overview
from .environment import values

# Device name: OmniGibson controller group, in the robot's controller order.
PARTS = {"base": "base", "torso": "trunk", "left_arm": "arm_left", "left_gripper": "gripper_left",
         "right_arm": "arm_right", "right_gripper": "gripper_right"}
ROLES = {"head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}
EVAL_BASE_LINK_MASS = 250.  # evaluator._apply_robot_eval_settings
EVAL_HEAD_HORIZONTAL_APERTURE = 40.


class R1Pro:
    pose_key = None
    frequencies = {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120}
    # RoboEnv's BEHAVIOR overview: 2.6 m out, the eye 2.4 m and the target 0.9 m above the floor.
    overview = Overview(2.6, 2.4, .9)

    def __init__(self):
        # Read without importing OmniGibson, which fixes its data path on import (environment.boot).
        package = Path(importlib.util.find_spec("omnigibson").origin).parent
        self.config = yaml.safe_load((package / "eval/r1pro.yaml").read_text())
        sensors = self.config.pop("eval")["camera_sensor_names"]
        self.cameras = {device: sensors[role] for role, device in ROLES.items()}

    def robot_config(self):
        """The evaluator's robot with its RGB-D full-resolution cameras (rgbd_full_res_wrapper)."""
        from omnigibson.eval.utils.eval_utils import HEAD_RESOLUTION, WRIST_RESOLUTION

        config = {**self.config, "obs_modalities": ["proprio", "rgb", "depth_linear"]}
        config["sensor_config"] = dict(config["sensor_config"])
        for device, sensor in self.cameras.items():
            height, width = HEAD_RESOLUTION if device == "head_camera" else WRIST_RESOLUTION
            config["sensor_config"][sensor.split(":", 1)[1]] = {
                "sensor_kwargs": {"image_height": height, "image_width": width}}
        return config

    def configure(self, robot):
        """A heavier base, so world contact does not shove it, and the evaluator's head optics."""
        import omnigibson as og

        og.sim.stop()  # base mass writes require a stopped simulation
        robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
        og.sim.play()
        robot.sensors[self.cameras["head_camera"]].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

    def start_pose(self, pose):
        return pose["position"], pose["orientation"]

    def layout(self, robot):
        """Channels in the native action order; grippers are scalars named after their device."""
        if list(robot.controller_order) != list(PARTS.values()):
            raise ValueError(f"unexpected R1Pro controllers: {robot.controller_order}")
        lower, upper = values(robot.joint_lower_limits), values(robot.joint_upper_limits)
        names, groups, self.dofs, self.grips = [], [], {}, {}
        for device, part in PARTS.items():
            dofs = values(robot.controller_joint_idx[part]).astype(int)
            action = values(robot.controller_action_idx[part]).astype(int)
            if list(action) != list(range(len(names), len(names) + len(action))):
                raise ValueError(f"{part} actions are not contiguous in controller order")
            self.dofs[device] = dofs
            channels = tuple(action)
            if part == "base":
                names += [robot.dof_names_ordered[i] for i in dofs]
                groups.append(CommandGroup(device, channels, (-1.,) * 3, (1.,) * 3, velocity=True))
            elif part.startswith("gripper"):
                names.append(device)
                groups.append(CommandGroup(device, channels, (-1.,), (1.,), scalar=True))
                # The smooth controller maps [-1, 1] onto the mean finger range.
                self.grips[device] = lower[dofs].mean(), upper[dofs].mean()
            else:
                names += [robot.dof_names_ordered[i] for i in dofs]
                groups.append(CommandGroup(device, channels, tuple(lower[dofs]), tuple(upper[dofs])))
        return ActionLayout(tuple(names), tuple(groups))

    def reset(self, episode):
        pass

    def state(self, episode):
        """Joints in radians; grippers as the smooth command their mean finger opening reaches;
        the base as odometry (x, y, yaw) and body twist (vx, vy, wz)."""
        q, dq = values(episode.robot.get_joint_positions()), values(episode.robot.get_joint_velocities())
        positions, velocities = [], []
        for device, part in PARTS.items():
            dofs = self.dofs[device]
            if part == "base":
                pose, twist = episode.odometry()
                positions += list(pose)
                velocities += list(twist)
            elif part.startswith("gripper"):
                low, high = self.grips[device]
                positions.append(2 * (q[dofs].mean() - low) / (high - low) - 1)
                velocities.append(2 * dq[dofs].mean() / (high - low))
            else:
                positions += list(q[dofs])
                velocities += list(dq[dofs])
        return np.asarray(positions), np.asarray(velocities)

    def action(self, episode, command):
        return np.asarray(command, dtype=np.float32)
