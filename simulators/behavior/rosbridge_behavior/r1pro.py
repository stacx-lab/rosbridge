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
# The holonomic base's driven virtual joints, by the kind of their USD drive, and its wheel links (limit_base_drive).
BASE_DRIVES = {"base_footprint_x_joint": "linear", "base_footprint_y_joint": "linear",
               "base_footprint_rz_joint": "angular"}
WHEELS = ("wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3")
WHEEL_MATERIAL = "Looks/wheel_physics_material"  # under the robot's prim


class R1Pro:
    parts = PARTS
    scalar_grippers = True  # the smooth gripper command; R1ProSharpa's hands are joint devices
    pose_key = None
    rendering = "evaluator"  # one render after each step that cameras read (backend.RERENDERS)
    frequencies = {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120}
    # RoboEnv's BEHAVIOR overview: 2.6 m out, the eye 2.4 m and the target 0.9 m above the floor.
    overview = Overview(2.6, 2.4, .9)
    # The base's odometry carries its full attitude (Episode.attitude): its z, roll and pitch joints are free, so it
    # tilts and can tip over when it pushes against something, and a planar pose showed a tipped robot upright.
    attitude = True
    # The base drive's effort limit (N along x and y, N m about z) on frictionless wheels (limit_base_drive), or None
    # for the evaluator's base, whose drive has no limit.
    drive_effort = None

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
        """A heavier base, so world contact does not shove it, and the evaluator's head optics; with drive_effort, a
        base drive of that effort on frictionless wheels (limit_base_drive)."""
        import omnigibson as og

        og.sim.stop()  # base mass writes require a stopped simulation, and so does binding the wheels' material
        robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
        if self.drive_effort is not None:
            limit_base_drive(robot, *self.drive_effort)
        og.sim.play()
        robot.sensors[self.cameras["head_camera"]].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

    def start_pose(self, pose):
        return pose["position"], pose["orientation"]

    def layout(self, robot):
        """Channels in the native action order; grippers are scalars named after their device."""
        if list(robot.controller_order) != list(self.parts.values()):
            raise ValueError(f"unexpected R1Pro controllers: {robot.controller_order}")
        lower, upper = values(robot.joint_lower_limits), values(robot.joint_upper_limits)
        names, groups, self.dofs, self.grips = [], [], {}, {}
        for device, part in self.parts.items():
            dofs = values(robot.controller_joint_idx[part]).astype(int)
            action = values(robot.controller_action_idx[part]).astype(int)
            if list(action) != list(range(len(names), len(names) + len(action))):
                raise ValueError(f"{part} actions are not contiguous in controller order")
            self.dofs[device] = dofs
            channels = tuple(action)
            if part == "base":
                names += [robot.dof_names_ordered[i] for i in dofs]
                groups.append(CommandGroup(device, channels, (-1.,) * 3, (1.,) * 3, velocity=True))
            elif part.startswith("gripper") and self.scalar_grippers:
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
        for device, part in self.parts.items():
            dofs = self.dofs[device]
            if part == "base":
                pose, twist = episode.odometry()
                positions += list(pose)
                velocities += list(twist)
            elif part.startswith("gripper") and self.scalar_grippers:
                low, high = self.grips[device]
                positions.append(2 * (q[dofs].mean() - low) / (high - low) - 1)
                velocities.append(2 * dq[dofs].mean() / (high - low))
            else:
                positions += list(q[dofs])
                velocities += list(dq[dofs])
        return np.asarray(positions), np.asarray(velocities)

    def action(self, episode, command):
        return np.asarray(command, dtype=np.float32)


def limit_base_drive(robot, force, torque):
    """Limit the base drive to force (N) along x and y and torque (N m) about z, on frictionless wheels. Call it while
    the simulator is stopped (R1Pro.configure).

    The base is three virtual drive joints (x and y along the robot's spawn pose, and yaw; z, roll and pitch are free)
    on three wheel spheres fixed to base_link. OmniGibson writes a 1000 N and N m limit to the drives through the
    articulation view at initialization, but the stop and play that sets the base mass (the evaluator's, and
    configure's) makes PhysX read the USD again, where the drives have none: PhysX holds FLT_MAX, and in contact the
    drive pushes with 100 kN and more, so the base climbs furniture and tips over (backing into a fridge: 89 degrees).
    The wheel spheres slide on the floor with PhysX's default friction, 0.5, which takes 1248 N at speed: with the
    1000 N limit back the base reached 0.45 of 0.75 m/s, and a limit that drives it (1600 N) still tilted it 23
    degrees sliding along a counter. On frictionless wheels (combined with any floor by the minimum) the drive alone
    moves the base, and its limit caps what the base pushes with. Measured on behavior-dex: at 500 N the base stays
    within 4 degrees against furniture, walls and doors and pushes a chair, a table and a 50 kg sofa as far as the
    unlimited drive, but a door that resists can turn it or push it aside; 700 N tilts it 9 to 21 degrees pushing a
    closed door or sliding along a counter, 300 N swings a room door open a quarter as far, and a wheel friction of 0.1
    eats 300 N of the limit. 150 N m turns the base pressed on a counter, where 100 N m turned it 18 of 120 degrees,
    and pushes with under 400 N at its corners.

    Both settings are USD, written here while the simulator is stopped: a material bound while it plays does not reach
    PhysX, and PhysX reads both at every later play, so they also hold after another stop and play (attaching an object
    does one: AttachedTo). The x and y limits act along the spawn pose's axes, so at headings between them the base
    pushes with up to sqrt(2) times force."""
    import omnigibson as og
    import omnigibson.lazy as lazy

    with og.sim.editing_usd():
        material = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
            prim_path=f"{robot.prim_path}/{WHEEL_MATERIAL}", name="wheel_physics_material",
            static_friction=0., dynamic_friction=0., restitution=0.)
        lazy.pxr.PhysxSchema.PhysxMaterialAPI.Apply(material.prim).CreateFrictionCombineModeAttr().Set("min")
        for name, kind in BASE_DRIVES.items():
            effort = force if kind == "linear" else torque
            lazy.pxr.UsdPhysics.DriveAPI.Get(robot.joints[name].prim, kind).GetMaxForceAttr().Set(float(effort))
    for link in WHEELS:
        for mesh in robot.links[link].collision_meshes.values():
            mesh.apply_physics_material(material)  # in its own editing_usd()
