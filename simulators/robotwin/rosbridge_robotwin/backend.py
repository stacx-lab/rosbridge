"""RoboTwin 2.0 tasks on the native SAPIEN ALOHA AgileX robot; no ROS dependencies.

Control follows RoboEnv's absolute-joint practice protocol: every 0.04 s frame
sets both arms' native joint drive targets, and the native normalized gripper
setter runs on each of the ten 250 Hz physics substeps. Upstream's evaluation
reset (setup, expert dry run, evaluation setup) and success checks are kept.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import os
from pathlib import Path
import random
import sys

import numpy as np
import yaml

from rosbridge.catalog import resolve
from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory, spectator_views
from rosbridge.project import project_root
from .assets import check_task

PHYSICS_DT = 1 / 250
SUBSTEPS = 10
CAMERAS = ("head_camera", "left_camera", "right_camera")
SIDES = ("left", "right")
# SAPIEN cameras look along +x with +y left and +z up; ROS optical frames look along +z.
OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
SPECTATORS = (("third_person", "head_camera"), ("left_wrist", "left_camera"),
              ("right_wrist", "right_camera"))


def runtime():
    return project_root() / ".runtime/robotwin"


@contextmanager
def in_runtime():
    """RoboTwin resolves ./assets and its configs relative to the working directory."""
    root = runtime()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    previous = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(previous)


def _yaml(path):
    return yaml.safe_load(Path(path).read_text())


def task_args(root, task_name, *, task_config="demo_clean", embodiment="aloha-agilex",
              planner="mplib_RRT"):
    """Upstream collect_data.py configuration assembly, with absolute embodiment paths."""
    configs = root / "env_cfg/task_config"
    args = _yaml(configs / f"{task_config}.yml")
    registry = _yaml(configs / "_embodiment_config.yml")
    robot_file = os.path.normpath(root / registry[embodiment]["file_path"])
    embodiment_config = _yaml(Path(robot_file) / "config.yml")
    # RoboEnv's device runtime plans the expert dry run with mplib, without cuRobo.
    embodiment_config["planner"] = planner
    args.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
                left_embodiment_config=embodiment_config,
                right_embodiment_config=deepcopy(embodiment_config),
                embodiment_name=embodiment, task_name=task_name, save_path=None, save_data=False,
                collect_data=False, eval_video_save_dir=None, render_freq=0, save_freq=25,
                need_plan=True)
    args["camera"]["collect_wrist_camera"] = True
    args["data_type"].update(rgb=True, depth=True, qpos=True, endpose=True)
    return args


def pose_matrix(pose):
    """SAPIEN Pose (p, q wxyz) -> 4x4."""
    w, x, y, z = pose.q
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    matrix[:3, 3] = pose.p
    return matrix


def wxyz(rotation):
    """Unit quaternion (w, x, y, z) with w >= 0 of a rotation matrix."""
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[0], q[1 + i] = (m[k, j] - m[j, k]) / s, s / 4
        q[1 + j], q[1 + k] = (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return q if q[0] >= 0 else -q


class RoboTwinSim:
    default_task = "handover_block"
    period = SUBSTEPS * PHYSICS_DT
    width, height = 320, 240

    def __init__(self, *, cameras=CAMERAS, max_steps=None, terminate_on_success=False):
        self.cameras = tuple(cameras)
        # env is the native upstream task instance; task is its catalog name.
        self.requested_max_steps = max_steps
        self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success
        self.env = None
        self.spectators, self._spectators, self._depth = (), {}, {}

    # Episode lifecycle ------------------------------------------------------------------

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views(SPECTATORS, width, height, views)

    @property
    def devices(self):
        cameras = {camera: camera for camera in self.cameras}
        return device_inventory(self.layout, cameras, depth=tuple(cameras))

    def reset(self, task, seed, randomize=True):
        """Episodes are fully determined by the seed, as in upstream evaluation."""
        spec = resolve("robotwin", task=task)[2]
        check_task(runtime(), spec["native_task"])
        candidate = RoboTwinSim(cameras=self.cameras, max_steps=self.requested_max_steps,
                                terminate_on_success=self.terminate_on_success)
        candidate.spectators = self.spectators
        # Native scene generation seeds process RNGs; a failed reset keeps the live
        # episode's future randomness.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            candidate._initialize(spec, seed)
            if hasattr(self, "layout") and candidate.layout != self.layout:
                raise ValueError("reset cannot change the action layout")
        except BaseException:
            candidate.close()
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            raise
        self.close()
        self.__dict__.update(candidate.__dict__)

    def _initialize(self, spec, seed):
        self.task, self.seed = spec["native_task"], seed
        with in_runtime() as root:
            # Upstream reads ./assets while importing its action utilities.
            importlib.import_module("envs.utils.action")
            task_class = getattr(importlib.import_module(f"envs.{self.task}"), self.task)
            args = task_args(root, self.task, task_config=spec.get("task_config", "demo_clean"),
                             embodiment=spec.get("embodiment", "aloha-agilex"))
            self.env = task_class()
            self.env.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **deepcopy(args))
            # Upstream's evaluation runs the expert once, then rebuilds the scene: the
            # dry run assigns attributes that some check_success methods read.
            try:
                self.env.play_once()
                self.expert_error = None
            except Exception as error:
                self.expert_error = f"{type(error).__name__}: {error}"
            self.expert_plan_success = bool(getattr(self.env, "plan_success", False))
            self.env.setup_demo(now_ep_num=seed, seed=seed, is_test=True,
                                 **{**deepcopy(args), "eval_mode": True})
        # Device control records nothing through upstream's data-collection hook.
        self.env._take_picture = lambda: None
        scene_dt = float(self.env.scene.get_timestep())
        if not np.isclose(scene_dt, PHYSICS_DT, rtol=0, atol=1e-9):
            raise ValueError(f"RoboTwin control requires 250 Hz physics, got dt={scene_dt}")
        for camera in self.cameras:
            self._camera(camera)
        self._index()
        self.steps, self.done = 0, False
        self.max_steps = self.requested_max_steps or int(self.env.step_lim)
        self.success = bool(self.env.check_success())
        self.reward = float(self.success)
        if self.spectators:
            for view in self.spectators:
                self._open_spectator(view)
            for view in self.spectators:
                self.render_spectator(view.name)

    def _index(self):
        robot = self.env.robot
        names, groups, self._arms, self._grippers = [], [], {}, {}
        for side in SIDES:
            entity = getattr(robot, f"{side}_entity")
            active = entity.get_active_joints()
            arm = getattr(robot, f"{side}_arm_joints")
            finger = getattr(robot, f"{side}_gripper")[0][0]
            lower, upper = getattr(robot, f"{side}_gripper_scale")
            self._arms[side] = (entity, [active.index(joint) for joint in arm])
            self._grippers[side] = (entity, active.index(finger), float(lower), float(upper))
            limits = np.array([joint.get_limits()[0] for joint in arm], dtype=float)
            offset = len(names)
            names += [joint.get_name() for joint in arm] + [f"{side}_gripper"]
            groups.append(CommandGroup(f"{side}_arm", tuple(range(offset, offset + len(arm))),
                                       tuple(limits[:, 0]), tuple(limits[:, 1])))
            groups.append(CommandGroup(f"{side}_gripper", (offset + len(arm),), (0.,), (1.,),
                                       scalar=True))
        self.layout = ActionLayout(tuple(names), tuple(groups))

    @property
    def time_s(self):
        return self.steps * self.period

    def state(self):
        """Arm joints in radians; grippers as the native normalized opening.

        The opening is the finger joint mapped through gripper_scale, so a
        measurement is also a valid native gripper command.
        """
        positions, velocities = [], []
        for side in SIDES:
            entity, indexes = self._arms[side]
            qpos, qvel = entity.get_qpos(), entity.get_qvel()
            _, finger, lower, upper = self._grippers[side]
            positions += [*qpos[indexes], (qpos[finger] - lower) / (upper - lower)]
            velocities += [*qvel[indexes], qvel[finger] / (upper - lower)]
        return np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is done")
        robot = self.env.robot
        action = np.asarray(action, dtype=float)
        grips = {}
        for side, offset in zip(SIDES, (0, 7)):
            robot.set_arm_joints(action[offset:offset + 6], np.zeros(6), side)
            grips[side] = action[offset + 6]
        for _ in range(SUBSTEPS):
            for side in SIDES:
                # Native saturation and per-call rate limit of the normalized setter.
                robot.set_gripper(grips[side], side)
            self.env.scene.step()
            self.success = bool(self.env.check_success()) or self.success
        self.steps += 1
        self.env.take_action_cnt = self.steps
        self.reward = float(self.success)
        self.done = bool((self.max_steps and self.steps >= self.max_steps)
                         or (self.terminate_on_success and self.success))

    # Cameras ----------------------------------------------------------------------------

    def _camera(self, name):
        cameras = self.env.cameras
        if name == "head_camera":
            return cameras.static_camera_list[cameras.static_camera_name.index(name)]
        if name in ("left_camera", "right_camera"):
            return getattr(cameras, name)
        raise KeyError(name)

    def camera_pose(self, camera):
        """Optical-frame pose (pos, quat_wxyz) in the world frame."""
        self.env._update_render()
        matrix = pose_matrix(self._camera(camera).get_global_pose())
        return matrix[:3, 3].copy(), wxyz(matrix[:3, :3] @ OPTICAL)

    def intrinsics(self, camera):
        """width, height, fx, fy, cx, cy with ROS pixel-center coordinates (SAPIEN uses pixel
        corners)."""
        k = self._camera(camera).get_intrinsic_matrix()
        return (self.width, self.height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]) - .5,
                float(k[1, 2]) - .5)

    def render(self, camera):
        if camera not in self.cameras:
            raise KeyError(camera)
        self.env._update_render()
        native = self._camera(camera)
        native.take_picture()
        color = np.asarray(native.get_picture("Color"))
        return np.ascontiguousarray((np.clip(color[..., :3], 0, 1) * 255).astype(np.uint8))

    def render_depth(self, camera):
        """Optical-axis depth at pixel centers from a raster twin of the camera.

        The ray-traced Position buffer holds the last sample, not the pixel center;
        a separate raster camera with the same projection leaves native RGB intact.
        """
        import sapien
        if camera not in self.cameras:
            raise KeyError(camera)
        native = self._camera(camera)
        if camera not in self._depth:
            depth = sapien.render.RenderCameraComponent(
                native.get_width(), native.get_height(),
                shader_dir=str(Path(sapien.__file__).parent / "vulkan_shader/default"))
            entity = sapien.Entity()
            entity.name = f"rosbridge_depth_{camera}"
            entity.add_component(depth)
            self.env.scene.add_entity(entity)
            self._depth[camera] = depth
        depth = self._depth[camera]
        k = native.get_intrinsic_matrix()
        depth.set_perspective_parameters(native.get_near(), native.get_far(), float(k[0, 0]),
                                         float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
                                         float(k[0, 1]))
        self.env._update_render()
        depth.entity.set_pose(native.get_global_pose())
        self.env.scene.update_render()
        depth.take_picture()
        position = np.asarray(depth.get_picture("Position"))
        z = -position[..., 2].copy()
        z[(position[..., 3] >= 1) | ~np.isfinite(z) | (z <= 0)] = 0
        return z.astype(np.float32)

    # Spectators: independent SAPIEN cameras that follow the observation mounts ----------

    def _open_spectator(self, view):
        native = self._camera(view.source)
        camera = self.env.scene.add_camera(f"rosbridge_spectator_{view.name}", view.width,
                                            view.height, 1., native.get_near(), native.get_far())
        k = native.get_intrinsic_matrix().copy()
        k[0] *= view.width / native.get_width()
        k[1] *= view.height / native.get_height()
        camera.set_perspective_parameters(native.get_near(), native.get_far(), k[0, 0], k[1, 1],
                                          k[0, 2], k[1, 2], k[0, 1])
        self._spectators[view.name] = (view, camera)

    def render_spectator(self, name):
        view, camera = self._spectators[name]
        if view.source == "head_camera":
            pose = self._camera(view.source).get_global_pose()
        else:
            # Read the live wrist mount without updating task cameras.
            pose = getattr(self.env.robot, view.source).get_pose()
        camera.entity.set_pose(pose)
        self.env.scene.update_render()
        camera.take_picture()
        pixels = np.asarray(camera.get_picture("Color"))
        return np.ascontiguousarray((np.clip(pixels[..., :3], 0, 1) * 255).astype(np.uint8))

    def spectator_intrinsics(self, name):
        k = self._spectators[name][1].get_intrinsic_matrix()
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]) - .5, float(k[1, 2]) - .5

    def close(self):
        if self.env is not None:
            for _, camera in self._spectators.values():
                try:
                    self.env.scene.remove_camera(camera)
                except Exception:
                    pass
            try:
                self.env.close_env(clear_cache=False)
            except Exception:
                pass
            self.env = None
        self._spectators, self._depth = {}, {}
