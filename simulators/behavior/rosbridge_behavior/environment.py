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
        apply_render_settings()
        if self.env.num_envs != 1 or len(self.env.scene.robots) != 1:
            raise ValueError("BEHAVIOR episodes hold one scene with one robot")
        self.robot = self.env.scene.robots[0]
        embodiment.configure(self.robot)
        # Physics views must exist before the first load reads or restores robot joint state.
        og.sim.update_handles()
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

    def odometry(self):
        """The base's planar pose (x, y, yaw) relative to its episode-start pose, and its twist
        (vx, vy, wz) in its own frame."""
        (start, start_rotation), (position, rotation) = self.origin, self.base_pose()
        offset = start_rotation.T @ (position - start)
        relative = start_rotation.T @ rotation
        base = self.robot.base_footprint_link
        linear = rotation.T @ values(base.get_linear_velocity())
        angular = rotation.T @ values(base.get_angular_velocity())
        return ((offset[0], offset[1], np.arctan2(relative[1, 0], relative[0, 0])),
                (linear[0], linear[1], angular[2]))

    def step(self, action):
        """Advance one action period on the native action vector; returns native success.

        Physics steps exactly as the evaluator's; cameras render on demand (render_for_capture),
        not every period.
        """
        import omnigibson as og
        import torch

        with og.sim.render_on_step(False):
            self.env.step(torch.as_tensor(action))
        if self.lights:
            # Toggles drive lights that are not serialized; re-derive their visibility.
            self.lights.sync_from_current_state()
        return bool(self.env.task.success[0])
