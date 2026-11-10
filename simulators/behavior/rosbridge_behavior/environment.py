"""BEHAVIOR's official evaluation episode for any embodiment; runs in the native worker.

Follows OmniGibson's eval/evaluator.py and eval/eval.py for one environment: the activity's
scene with the evaluator's rooms and transition rules, a horizon of 1.5 average human
demonstrations, and public test instances restored and settled as the evaluator loads them.
"""
import json
import os
from pathlib import Path

import numpy as np
import yaml

from rosbridge.project import project_root
from .rendering import apply_render_settings, capture_launch_settings

LIGHT_TASKS = {"turning_out_all_lights_before_sleep"}  # evaluator.LIGHT_EVAL_TASKS
SETTLE_STEPS = 25  # evaluator._settle_and_finalize
DOORS = ("door", "sliding_door", "garage_door")
# A door's rooms are read on the scene's room map this far beyond its faces. A locked door's joints keep this range
# above their closed position: OmniGibson reads equal limits as no limits.
DOOR_SIDE_M, DOOR_LOCKED = .5, 1e-3


def values(value):
    """A torch tensor or sequence as a float NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def rotation(quaternion_xyzw):
    """The rotation matrix of an OmniGibson (x, y, z, w) quaternion."""
    x, y, z, w = values(quaternion_xyzw) / np.linalg.norm(values(quaternion_xyzw))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def turn(axis, angle):
    """The rotation matrix of angle (rad) about the x, y or z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return {"x": np.array([[1., 0., 0.], [0., c, -s], [0., s, c]]),
            "y": np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]),
            "z": np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])}[axis]


def doors_out(scene, rooms):
    """The doors with one side in a loaded room and the other not, on the scene's room map. A partial scene loads every
    door but only the loaded rooms' floors, so such a door opens onto no floor."""
    doors = []
    for door in scene.objects:
        if door.category not in DOORS or not door.joints:
            continue
        position, orientation = door.get_position_orientation()
        turn = rotation(orientation)
        size = values(door.native_bbox) * values(door.scale)  # the closed door, in its own frame
        centre = values(position) + turn @ values(door.scaled_bbox_center_in_base_frame)
        thin = int(np.argmin(size[:2]))
        across = turn[:, thin] * (size[thin] / 2 + DOOR_SIDE_M)
        if len({scene.seg_map.get_room_instance_by_point((centre + side * across)[:2]) in rooms
                for side in (1, -1)}) == 2:
            doors.append(door)
    return doors


def lock_doors_out(scene, rooms):
    """Close and lock the doors out of the loaded rooms (doors_out); returns them."""
    doors = doors_out(scene, rooms)
    for door in doors:
        for joint in door.joints.values():
            joint.set_pos(0.)  # a BEHAVIOR door's closed position
            joint.upper_limit = DOOR_LOCKED  # before the lower limit, which is at most 0, so the range never collapses
            joint.lower_limit = 0.
    return doors


def data_root(simulator):
    return project_root() / ".runtime" / simulator / "datasets"


def boot(simulator):
    """Configure OmniGibson before Kit starts: prepared data, headless Kit on one GPU."""
    from rosbridge.core.gpu import single_gpu

    os.environ.update(OMNIGIBSON_DATA_PATH=str(data_root(simulator)), OMNIGIBSON_HEADLESS="1",
                      OMNIGIBSON_GPU_ID="0", OMNIGIBSON_RENDER_GPU=str(single_gpu()))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from omnigibson.macros import gm
    from omnigibson.transition_rules import CookingSystemRule, MixingToolRule, ToggleableMachineRule

    gm.USE_GPU_DYNAMICS, gm.ENABLE_TRANSITION_RULES, gm.RENDER_VIEWER_CAMERA = False, True, False
    # The evaluator's DISABLED_TRANSITION_RULES (joylo gello.utils.og_teleop_cfg).
    for rule in (ToggleableMachineRule, MixingToolRule, CookingSystemRule):
        rule.ENABLED = False


class Episode:
    """One activity's environment; Isaac hosts one per process."""

    def __init__(self, task, embodiment, max_steps=None):
        import omnigibson as og
        from omnigibson.eval.utils.eval_utils import (
            EVAL_TIMEOUT_MULTIPLIER, TASK_NAMES_TO_ROOMS, generate_basic_environment_config)
        from omnigibson.eval.utils.score_utils import load_human_stats
        from omnigibson.macros import gm

        metadata = Path(gm.DATA_PATH) / "2026-challenge-task-instances/metadata"
        task_cfg = yaml.safe_load((metadata / "available_tasks.yaml").read_text())[task][0]
        cfg = generate_basic_environment_config(task_name=task, task_cfg=task_cfg)
        cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS[task]  # partial_scene_load
        # The evaluator's timeout counts its own steps; an embodiment at another rate gets the same time.
        rate = embodiment.frequencies["action_frequency"] / cfg["env"]["action_frequency"]
        cfg["env"].update(embodiment.frequencies)
        robot = embodiment.robot_config()
        position, orientation = embodiment.start_pose({"position": task_cfg["robot_start_position"],
                                                       "orientation": task_cfg["robot_start_orientation"]})
        robot["position"], robot["orientation"] = values(position).tolist(), values(orientation).tolist()
        cfg["robots"] = [robot]
        cfg["task"]["presampled_pose_key"] = embodiment.pose_key
        # load() places the robot at the instance's presampled pose, as the evaluator does, and the
        # scene's initial state keeps it; the task must not move a pelvis-rooted G1 to the floor.
        cfg["task"]["use_presampled_robot_pose"] = False
        self.max_steps = max_steps or int(load_human_stats(task)["length"] * EVAL_TIMEOUT_MULTIPLIER * rate)
        cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        self.task, self.embodiment = task, embodiment
        with capture_launch_settings():
            self.env = og.Environment(configs=cfg)
        if embodiment.rendering == "on_demand":
            apply_render_settings()  # DLAA; the R1Pro embodiments keep OmniGibson's anti-aliasing
        if self.env.num_envs != 1 or len(self.env.scene.robots) != 1:
            raise ValueError("BEHAVIOR episodes hold one scene with one robot")
        self.robot = self.env.scene.robots[0]
        embodiment.configure(self.robot)
        # Physics views must exist before the first load reads or restores robot joint state.
        og.sim.update_handles()
        # The robot cannot pass a door onto a room without its floor: such doors are closed and locked.
        self.locked_doors = [door.name for door in lock_doors_out(self.env.scene, cfg["scene"]["load_room_instances"])]
        print(f"doors locked out of the loaded rooms: {self.locked_doors}", flush=True)
        self.lights = None

    def load(self, seed):
        """Load public test instance seed % 20, as the evaluator's load_batch does."""
        import omnigibson as og
        from omnigibson.eval.utils.eval_utils import (
            NUM_PUBLIC_TEST_INSTANCES, TEST_INSTANCE_IDS, seed_everything)
        from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
        from omnigibson.tasks.behavior_task import get_presampled_robot_poses
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.bddl_utils import is_system_bddl_inst
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        seed_everything(seed)
        instance = TEST_INSTANCE_IDS[seed % NUM_PUBLIC_TEST_INSTANCES]
        env, robot, task, scene = self.env, self.robot, self.env.task, self.env.scene
        # The instance restores only the base pose; joints, controllers and grasps start clean.
        robot.reset()
        name = task.get_cached_activity_scene_filename(
            scene_model=task.scene_name, activity_name=task.activity_name,
            activity_definition_id=task.activity_definition_id, activity_instance_id=instance)
        path = get_task_instance_path(
            task.scene_name, f"{task.scene_name}_task_{task.activity_name}_instances/{name}-tro_state",
            mode="public_test")
        if path is None:
            raise FileNotFoundError(f"no public test instance {instance} for {self.task}")
        scope = task.object_scopes[0]
        for key, state in recursively_convert_to_torch(json.loads(Path(path).read_text())).items():
            if key == "robot_poses":
                pose = get_presampled_robot_poses(state, robot_model=robot.model,
                                                  pose_key=self.embodiment.pose_key)[0]
                robot.set_position_orientation(*self.embodiment.start_pose(pose), frame="scene")
                scene.write_task_metadata(key=key, data=state)
            else:
                scope[key].load_state(state, serialized=False)
        if self.task in LIGHT_TASKS:
            set_light_control_toggles(scope.values(), True)
        og.sim.update_handles()
        # Loading state can introduce jitter; hold task entities still while physics settles.
        for _ in range(SETTLE_STEPS):
            og.sim.step_physics()
            for entity_name, entity in scope.items():
                if entity is not None and not is_system_bddl_inst(entity_name):
                    entity.keep_still()
        scene.update_initial_file()
        env.reset()
        self.lights = LightToggleSynchronizer(scene) if self.task in LIGHT_TASKS else None
        if self.lights:
            self.lights.reset_from_current_state()
        self.instance = instance

    def base_pose(self):
        position, quaternion = self.robot.base_footprint_link.get_position_orientation()
        return values(position), rotation(quaternion)

    def _base_motion(self):
        """The base's offset and rotation relative to its episode-start pose, its rotation in the world, and its
        linear and angular velocity in its own frame."""
        (start, start_rotation), (position, rotation) = self.origin, self.base_pose()
        offset = start_rotation.T @ (position - start)
        relative = start_rotation.T @ rotation
        base = self.robot.base_footprint_link
        linear = rotation.T @ values(base.get_linear_velocity())
        angular = rotation.T @ values(base.get_angular_velocity())
        return offset, relative, rotation, linear, angular

    def odometry(self):
        """The base's planar pose (x, y, yaw) relative to its episode-start pose, and its twist
        (vx, vy, wz) in its own frame."""
        offset, relative, _, linear, angular = self._base_motion()
        return ((offset[0], offset[1], np.arctan2(relative[1, 0], relative[0, 0])),
                (linear[0], linear[1], angular[2]))

    def attitude(self):
        """The rest of the odometry of a base that can tilt (the R1Pro's z, roll and pitch joints are free), as an
        IMU and odometry read it: the height change z, roll and pitch relative to the episode-start pose (ZYX angles
        with the odometry's yaw, so yaw * pitch * roll is the base's rotation from its start), the vertical and roll
        and pitch rates (vz, wx, wy) in the base's own frame, and tilt, the angle of the base's z axis from vertical,
        in radians. The base starts upright, so roll and pitch are its tilt about its own axes.

        Roll is what the odometry's yaw and the pitch leave of the rotation, so the three rebuild it also at a pitch of
        exactly +-90 degrees, where yaw and roll are not separately defined and the odometry's yaw is whatever the
        rounding of the rotation gives."""
        offset, relative, rotation, linear, angular = self._base_motion()
        yaw = np.arctan2(relative[1, 0], relative[0, 0])  # the odometry's
        pitch = np.arctan2(-relative[2, 0], np.hypot(relative[2, 1], relative[2, 2]))
        roll_only = turn("y", -pitch) @ turn("z", -yaw) @ relative
        return {"z": float(offset[2]), "roll": float(np.arctan2(roll_only[2, 1], roll_only[1, 1])),
                "pitch": float(pitch), "vz": float(linear[2]), "wx": float(angular[0]), "wy": float(angular[1]),
                "tilt": float(np.arctan2(np.hypot(rotation[0, 2], rotation[1, 2]), rotation[2, 2]))}

    def step(self, action):
        """Advance one action period on the native action vector; returns native success.

        Physics steps exactly as the evaluator's but without rendering, so it does not depend on
        whether cameras are read; reads render afterwards (BehaviorSim._render).
        """
        import omnigibson as og
        import torch

        with og.sim.render_on_step(False):
            self.env.step(torch.as_tensor(action))
        if self.lights:
            # Toggles drive lights that are not serialized; re-derive their visibility.
            self.lights.sync_from_current_state()
        return bool(self.env.task.success[0])
