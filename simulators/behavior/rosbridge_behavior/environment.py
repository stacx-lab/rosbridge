"""BEHAVIOR's official evaluation episode on the R1Pro; runs in the native worker.

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

LIGHT_TASKS = {"turning_out_all_lights_before_sleep"}  # evaluator.LIGHT_EVAL_TASKS
SETTLE_STEPS = 25  # evaluator._settle_and_finalize
LIGHT_RERENDERS = 3  # the evaluator's renders after its light sync
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


def remove_particles_out(scene, rooms):
    """Remove the visual particles that the scene's room map places in a room other than the loaded ones; returns their
    names. BEHAVIOR's sampler places a floor's particles wherever the floor is traversable, and a floor can reach past
    its room's locked doors (lock_doors_out). Where the map names no room, a particle stays."""
    removed = []
    for system in scene.active_systems.values():
        if not scene.is_visual_particle_system(system.name) or not system.n_particles:
            continue
        names = list(system.particles)
        out = [index for index, position in enumerate(values(system.get_particles_position_orientation()[0]))
               if scene.seg_map.get_room_instance_by_point(position[:2]) not in (None, *rooms)]
        if out:
            system.remove_particles(idxs=out)
            removed += [names[index] for index in out]
    return removed


def data_root():
    return project_root() / ".runtime/behavior/datasets"


def boot():
    """Configure OmniGibson before Kit starts: prepared data, headless Kit on one GPU."""
    from rosbridge.core.gpu import single_gpu

    os.environ.update(OMNIGIBSON_DATA_PATH=str(data_root()), OMNIGIBSON_HEADLESS="1",
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

    def __init__(self, task, r1pro, max_steps=None):
        import omnigibson as og
        from omnigibson.eval.utils.eval_utils import (
            DEFAULT_EVAL_SEED, EVAL_TIMEOUT_MULTIPLIER, TASK_NAMES_TO_ROOMS, generate_basic_environment_config,
            seed_everything)
        from omnigibson.eval.utils.score_utils import load_human_stats
        from omnigibson.macros import gm

        metadata = Path(gm.DATA_PATH) / "2026-challenge-task-instances/metadata"
        task_cfg = yaml.safe_load((metadata / "available_tasks.yaml").read_text())[task][0]
        cfg = generate_basic_environment_config(task_name=task, task_cfg=task_cfg)
        cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS[task]  # partial_scene_load
        robot = r1pro.robot_config()
        robot["position"] = values(task_cfg["robot_start_position"]).tolist()
        robot["orientation"] = values(task_cfg["robot_start_orientation"]).tolist()
        cfg["robots"] = [robot]
        # load() places the robot at the instance's presampled pose, and env.reset places it there again once
        # the scene has settled, as the evaluator's does (the task's use_presampled_robot_pose).
        self.max_steps = max_steps or int(load_human_stats(task)["length"] * EVAL_TIMEOUT_MULTIPLIER)
        cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        self.task = task
        # The evaluator seeds once, before it builds the environment (omnigibson/eval/eval.py).
        seed_everything(DEFAULT_EVAL_SEED)
        self.env = og.Environment(configs=cfg)
        if self.env.num_envs != 1 or len(self.env.scene.robots) != 1:
            raise ValueError("BEHAVIOR episodes hold one scene with one robot")
        self.robot = self.env.scene.robots[0]
        r1pro.configure(self.robot)
        # Physics views must exist before the first load reads or restores robot joint state.
        og.sim.update_handles()
        # The robot cannot pass a door onto a room without its floor: such doors are closed and locked.
        self.rooms = cfg["scene"]["load_room_instances"]
        self.locked_doors = [door.name for door in lock_doors_out(self.env.scene, self.rooms)]
        print(f"doors locked out of the loaded rooms: {self.locked_doors}", flush=True)
        self.lights = None

    def load(self, seed):
        """Load public test instance seed % 20, as the evaluator's load_batch does."""
        import omnigibson as og
        from omnigibson.eval.utils.eval_utils import NUM_PUBLIC_TEST_INSTANCES, TEST_INSTANCE_IDS
        from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
        from omnigibson.tasks.behavior_task import get_presampled_robot_poses
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.bddl_utils import is_system_bddl_inst
        from omnigibson.utils.python_utils import recursively_convert_to_torch

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
                pose = get_presampled_robot_poses(state, robot_model=robot.model)[0]
                robot.set_position_orientation(pose["position"], pose["orientation"], frame="scene")
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
        # Particles out of the loaded rooms go before the initial state is saved, so env.reset restores the instance
        # without them.
        self.particles_removed = remove_particles_out(scene, self.rooms)
        scene.update_initial_file()
        env.reset()
        self.lights = LightToggleSynchronizer(scene) if self.task in LIGHT_TASKS else None
        if self.lights:
            self.lights.reset_from_current_state()
            self._show_lights()
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
        """Advance one action period on the native action vector, as the evaluator's env.step, which
        renders once; returns native success."""
        import torch

        self.env.step(torch.as_tensor(action))
        if self.lights:
            # Toggles drive lights that are not serialized; re-derive their visibility.
            self.lights.sync_from_current_state()
            self._show_lights()
        return bool(self.env.task.success[0])

    def _show_lights(self):
        """Render as the evaluator does after a light sync, so that frames show the synced lights."""
        import omnigibson as og

        for _ in range(LIGHT_RERENDERS):
            og.sim.render()
