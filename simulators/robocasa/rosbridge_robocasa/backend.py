"""Official RoboCasa kitchens with the PandaOmron's native part controllers; no ROS dependencies."""
import os
import random

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from rosbridge.catalog import resolve
from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import device_inventory, spectator_views
from rosbridge.project import project_root
from .assets import check_task

OPTICAL = np.diag([1., -1., -1.])
ROBOT = 'PandaOmron'
CAMERAS = {'robot0_agentview_left': 'camera_left', 'robot0_agentview_right': 'camera_right',
           'robot0_eye_in_hand': 'camera_wrist'}
SPECTATORS = (('third_person', 'robot0_agentview_left'), ('wrist', 'robot0_eye_in_hand'))
# Upstream's object instance splits (obj_instance_split): "all" samples every instance, as its default does.
OBJECT_SPLITS = {'pretrain': 'pretrain', 'target': 'target', 'all': None}
KITCHEN_IDS = range(1, 61)  # upstream's layout and style ids (LayoutType, StyleType)


def scene_options(layout_id=1, style_id=1, object_split='all'):
    """The checked scene: the kitchen's upstream layout and style ids and its object instance split."""
    for name, value in (('layout_id', layout_id), ('style_id', style_id)):
        if type(value) is not int or value not in KITCHEN_IDS:
            raise ValueError(f'{name} must be an upstream id in 1-60, not {value!r}')
    if not isinstance(object_split, str) or object_split not in OBJECT_SPLITS:
        raise ValueError(f'object_split must be one of {", ".join(OBJECT_SPLITS)}, not {object_split!r}')
    return {'layout_id': layout_id, 'style_id': style_id, 'object_split': object_split}


def excluded(task, env, scene):
    """Why a task cannot build the scene's kitchen: the layout or style it excludes."""
    reasons = [f'{kind} {scene[f"{kind}_id"]} (it excludes {kind}s {sorted(set(ids))}; pass another --{kind}-id)'
               for kind, ids in (('layout', env.EXCLUDE_LAYOUTS), ('style', env.EXCLUDE_STYLES))
               if scene[f'{kind}_id'] in ids]
    return f'{task} excludes ' + ' and '.join(reasons)


def controller_config():
    """JOINT_POSITION arm and torso; native base velocity and gripper.

    Both joint parts start with native delta input, so the zero actions with which a
    kitchen settles during reset hold the initial posture; reset then makes them absolute.
    """
    from robosuite.controllers import load_composite_controller_config, load_part_controller_config
    config = load_composite_controller_config(robot=ROBOT)
    parts = config['body_parts']
    parts['right'] = {**load_part_controller_config(default_controller='JOINT_POSITION'),
                      'gripper': parts['right']['gripper']}
    return config


class RoboCasaSim:
    default_task = 'PickPlaceCounterToCabinet'
    period = 1 / 20

    def __init__(self, *, cameras=tuple(CAMERAS), width=256, height=256,
                 max_steps=None, early_stop=False):
        self.cameras, self.width, self.height = tuple(cameras), width, height
        self.requested_max_steps = self.max_steps = max_steps
        self.early_stop = early_stop
        self.env = self._depth = self._spectator = None
        self.spectators = ()

    @property
    def robot(self):
        return self.env.robots[0]

    @property
    def model(self):
        return self.env.sim.model._model

    @property
    def data(self):
        return self.env.sim.data._data

    @property
    def devices(self):
        cameras = {CAMERAS[camera]: camera for camera in self.cameras}
        return device_inventory(self.layout, cameras, depth=tuple(cameras))

    def configure_spectators(self, width=960, height=720, views=None):
        self.spectators = spectator_views(SPECTATORS, width, height, views)

    def reset(self, task, seed, randomize=True, *, layout_id=1, style_id=1, object_split='all'):
        """The scene options pin the kitchen and its object instances; with them, the seed determines the rest:
        fixtures and textures, objects, placements, the base pose and the instruction."""
        spec = resolve('robocasa', task=task)[2]
        scene = scene_options(layout_id, style_id, object_split)
        check_task(project_root() / '.runtime/robocasa', spec['native_task'])
        candidate = RoboCasaSim(cameras=self.cameras, width=self.width, height=self.height,
                                max_steps=self.requested_max_steps,
                                early_stop=self.early_stop)
        candidate.spectators = self.spectators
        # Kitchen generation also draws from process RNGs; a failed reset keeps the
        # live episode's future randomness.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            candidate._initialize(spec, seed, scene)
            if hasattr(self, 'layout') and candidate.layout != self.layout:
                raise ValueError('reset cannot change the action layout')
        except BaseException:
            candidate.close()
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            raise
        self.close()
        self.__dict__.update(candidate.__dict__)

    def _initialize(self, spec, seed, scene):
        import robocasa  # noqa: F401  (registers the kitchen environments)
        import robosuite
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS

        self.task, self.seed = spec['native_task'], seed
        random.seed(seed)
        np.random.seed(seed)
        self.env = robosuite.make(
            self.task, robots=ROBOT, controller_configs=controller_config(), has_renderer=False,
            has_offscreen_renderer=True, use_camera_obs=False, camera_names=list(self.cameras),
            camera_widths=self.width, camera_heights=self.height, camera_depths=False,
            ignore_done=True, translucent_robot=False, seed=seed, layout_ids=scene['layout_id'],
            style_ids=scene['style_id'], obj_instance_split=OBJECT_SPLITS[scene['object_split']])
        # Upstream drops the task's excluded layouts and styles from the kitchens it may build; no kitchen is built
        # until the first reset, so an excluded one fails here at no cost.
        if not self.env.layout_and_style_ids:
            raise ValueError(excluded(self.task, self.env, scene))
        self.env.reset()
        # Measurements and frames describe the current state, not the last substep's start.
        self.env.sim.forward()
        # The kitchen has settled under native zero-delta holds; goals are absolute from now on.
        for part in ('right', 'torso'):
            self.robot.part_controllers[part].input_type = 'absolute'
        for camera in (*self.cameras, *(v.source for v in self.spectators)):
            if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise ValueError(f'unknown camera: {camera}')
        self._index()
        # Seed the integrated gripper command from the fingers, so stop holds them.
        gripper = self.robot.part_controllers['right_gripper']
        fingers = self.data.qpos[gripper.qpos_index]
        self.robot.gripper['right'].current_action = np.clip(
            2 * (fingers - gripper.actuator_min) / (gripper.actuator_max - gripper.actuator_min) - 1,
            -1, 1)
        self._grip_rate = 0.
        self._origin = self.data.xpos[self._base].copy(), self.data.xmat[self._base].reshape(3, 3).copy()
        self.steps, self.done, self.success, self.reward = 0, False, False, 0.
        horizon = {**ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS}.get(self.task, {}).get('horizon')
        self.max_steps = self.requested_max_steps or horizon
        meta = self.env.get_ep_meta()
        self.scene = {'layout_id': meta['layout_id'], 'style_id': meta['style_id'],
                      'object_split': scene['object_split']}
        print(f'[robocasa] {self.task} seed {seed}, layout {meta["layout_id"]}, style {meta["style_id"]}, '
              f'objects {scene["object_split"]}: {meta.get("lang", "")}', flush=True)
        # Pixel-center depth from a separate single-sample context; RGB keeps native antialiasing.
        quality = self.model.vis.quality
        samples, quality.offsamples = quality.offsamples, 0
        try:
            self._depth = mj.Renderer(self.model, height=self.height, width=self.width)
        finally:
            quality.offsamples = samples
        if self.spectators:
            from rosbridge.core.mujoco_rendering import SpectatorRenderer
            context = self.env.sim._render_context_offscreen
            self._spectator = SpectatorRenderer(self.model, self.data, self.spectators,
                                               restore_context=context.gl_ctx, options=context.vopt)
            for view in self.spectators:
                self._spectator.capture(view.name)

    def _index(self):
        parts, model = self.robot.part_controllers, self.model
        joints = {name: [model.joint(j).name for j in parts[name].joint_index]
                  for name in ('right', 'torso', 'base')}
        self._qpos = [int(model.jnt_qposadr[j]) for name in ('right', 'torso')
                      for j in parts[name].joint_index]
        self._qvel = [int(model.jnt_dofadr[j]) for name in ('right', 'torso')
                      for j in parts[name].joint_index]
        self._base = model.body(self.robot.robot_model.base.root_body).id
        arm, torso = (model.jnt_range[parts[name].joint_index] for name in ('right', 'torso'))
        low, high = parts['base'].control_limits
        self.layout = ActionLayout((*joints['right'], 'gripper', *joints['torso'], *joints['base']), (
            CommandGroup('arm', tuple(range(7)), tuple(arm[:, 0]), tuple(arm[:, 1])),
            CommandGroup('gripper', (7,), (-1.,), (1.,), scalar=True),
            CommandGroup('torso', (8,), tuple(torso[:, 0]), tuple(torso[:, 1])),
            CommandGroup('base', (9, 10, 11), tuple(low), tuple(high), velocity=True)))

    @property
    def time_s(self):
        return self.steps * self.period

    def measures(self):
        """The episode's kitchen and object split, as its record states them."""
        return dict(self.scene)

    def _odometry(self):
        """Base pose relative to its episode-start pose, and its body-frame twist."""
        position, rotation = self._origin
        offset = rotation.T @ (self.data.xpos[self._base] - position)
        relative = rotation.T @ self.data.xmat[self._base].reshape(3, 3)
        twist = np.empty(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_XBODY, self._base, twist, 1)
        return np.r_[offset[:2], np.arctan2(relative[1, 0], relative[0, 0])], twist[[3, 4, 2]]

    def state(self):
        """Arm and torso joints, the integrated gripper command (-1 open, +1 closed), and
        the base odometry pose (x, y, yaw) and twist (vx, vy, wz)."""
        q, v = self.data.qpos[self._qpos], self.data.qvel[self._qvel]
        pose, twist = self._odometry()
        grip = self.robot.gripper['right'].current_action[1]
        return np.r_[q[:7], grip, q[7:], pose], np.r_[v[:7], self._grip_rate, v[7:], twist]

    def step(self, action):
        if self.done:
            raise RuntimeError('episode is done')
        action = np.asarray(action, dtype=float)
        gripper = self.robot.gripper['right']
        before = gripper.current_action[1]
        # The native gripper integrates a direction (+1 closes) by `speed` per step within
        # [-1, 1]; steer it toward the target while a native step gets strictly closer.
        error = action[7] - before
        after = np.clip(before + gripper.speed * np.sign(error), -1, 1)
        direction = float(np.sign(error)) if abs(action[7] - after) < abs(error) else 0.
        # In kitchens the native (legacy) base controller already takes body-frame velocity.
        _, reward, _, _ = self.env.step(self.robot.create_action_vector({
            'right': action[:7], 'right_gripper': [direction], 'torso': action[8:9],
            'base': action[9:12]}))
        self.env.sim.forward()
        self._grip_rate = (gripper.current_action[1] - before) / self.period
        self.steps += 1
        # Kitchen's reward is its success predicate; evaluating it again would advance
        # the history some tasks keep.
        self.reward, self.success = float(reward), bool(reward)
        self.done = bool((self.max_steps and self.steps >= self.max_steps)
                         or (self.early_stop and self.success))

    @staticmethod
    def _quat(matrix):
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, np.asarray(matrix).reshape(9))
        return quat

    def camera_pose(self, camera):
        cid = self.model.camera(camera).id
        return (self.data.cam_xpos[cid].copy(),
                self._quat(self.data.cam_xmat[cid].reshape(3, 3) @ OPTICAL))

    def intrinsics(self, camera):
        focal = .5 * self.height / np.tan(np.deg2rad(self.model.camera(camera).fovy[0]) / 2)
        w, h = self.width, self.height
        return w, h, float(focal), float(focal), (w - 1) / 2., (h - 1) / 2.

    def render(self, camera):
        if camera not in self.cameras:
            raise KeyError(camera)
        return np.ascontiguousarray(self.env.sim.render(camera_name=camera, width=self.width,
                                                        height=self.height)[::-1])

    def render_depth(self, camera):
        from rosbridge.core.mujoco_rendering import render_depth
        if camera not in self.cameras:
            raise KeyError(camera)
        options = self.env.sim._render_context_offscreen.vopt
        self._depth.update_scene(self.data, camera=camera, scene_option=options)
        return render_depth(self._depth, self.model)

    def render_spectator(self, name):
        return self._spectator.capture(name)

    def spectator_intrinsics(self, name):
        return self._spectator.intrinsics(name)

    def close(self):
        for renderer in (self._spectator, self._depth):
            if renderer is not None:
                renderer.close()
        self._spectator = self._depth = None
        if self.env is not None:
            self.env.close()
            self.env = None
