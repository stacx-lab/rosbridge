"""Official LIBERO physics with the native Panda joint controller; no ROS dependencies."""
import os
import random

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from rosbridge.core import ActionLayout, CommandGroup
from rosbridge.core.devices import SpectatorView, device_inventory
from .assets import check_task, configure, task_files, task_spec

OPTICAL = np.diag([1., -1., -1.])
INITIALIZATION_STEPS = 5
CAMERAS = {'agentview': 'camera_agentview', 'robot0_eye_in_hand': 'camera_wrist'}
FINGER_OPEN = .04
GRIP_TOLERANCE = .001  # metres of finger_joint1 within which the gripper holds


class LiberoSim:
    default_task = 'libero-spatial-01'
    period = 1 / 20

    def __init__(self, *, cameras=tuple(CAMERAS), width=256, height=256,
                 max_steps=None, terminate_on_success=False):
        self.cameras, self.width, self.height = tuple(cameras), width, height
        self.max_steps, self.terminate_on_success = max_steps, terminate_on_success
        self.wrapper = self._spectator = None
        self.spectators = ()

    @property
    def env(self):
        return self.wrapper.env

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
        return device_inventory(self.layout, {CAMERAS[camera]: camera for camera in self.cameras})

    def configure_spectators(self, width=960, height=720):
        self.spectators = tuple(SpectatorView(role, source, width, height) for role, source in
                               (('third_person', 'agentview'), ('wrist', 'robot0_eye_in_hand')))

    def reset(self, task, seed, randomize=True):
        runtime = configure()
        spec = task_spec(task)
        task = spec['native_task']
        check_task(runtime, task)
        candidate = LiberoSim(cameras=self.cameras, width=self.width, height=self.height,
                              max_steps=self.max_steps, terminate_on_success=self.terminate_on_success)
        candidate.spectators = self.spectators
        # Native construction uses process RNGs. A failed reset must also preserve
        # the live episode's future randomness.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            candidate._initialize(runtime, spec, seed, randomize)
            if hasattr(self, 'layout') and candidate.layout != self.layout:
                raise ValueError('reset cannot change the action layout')
        except BaseException:
            candidate.close()
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            raise
        self.close()
        self.__dict__.update(candidate.__dict__)

    def _initialize(self, runtime, spec, seed, randomize):
        import torch
        from libero.libero.envs.env_wrapper import ControlEnv

        self.task, self.seed = spec['native_task'], seed
        bddl, state_file = task_files(runtime, self.task)
        # Only trusted files from the pinned upstream are deserialized.
        states = torch.load(state_file, map_location='cpu', weights_only=False)
        index = int(np.random.default_rng(seed).integers(len(states))) if randomize else 0
        random.seed(seed)
        np.random.seed(seed)
        self.wrapper = ControlEnv.__new__(ControlEnv)
        self.wrapper.__init__(
            bddl_file_name=str(bddl), controller='JOINT_POSITION', use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=bool(self.cameras or self.spectators),
            camera_names=list(self.cameras),
            camera_widths=self.width, camera_heights=self.height, camera_depths=False,
            ignore_done=True, control_freq=20)
        self.wrapper.seed(seed)
        # The wrapper retries randomization forever. Let reset failures propagate.
        self.env.reset()
        self.wrapper.set_init_state(states[index])
        for camera in (*self.cameras, *(v.source for v in self.spectators)):
            if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise ValueError(f'unknown camera: {camera}')
        self._index()
        self.robot.controller.update(force=True)
        self.robot.controller.reset_goal()
        low, high = self.model.actuator_ctrlrange[self.robot._ref_joint_gripper_actuator_indexes].T
        fingers = self.data.qpos[self.robot._ref_gripper_joint_pos_indexes]
        self.robot.gripper.current_action = np.clip(2 * (fingers - low) / (high - low) - 1, -1, 1)
        initial = self.state()[0]
        for _ in range(INITIALIZATION_STEPS):
            self._advance(initial)
        self._time_origin = float(self.data.time)
        self.steps, self.done = 0, False
        self.success = bool(self.wrapper.check_success())
        self.reward = float(self.env.reward())
        if self.spectators:
            from rosbridge.core.mujoco_rendering import SpectatorRenderer
            context = self.env.sim._render_context_offscreen
            self._spectator = SpectatorRenderer(self.model, self.data, self.spectators,
                                               restore_context=context.gl_ctx, options=context.vopt)
            for view in self.spectators:
                self._spectator.capture(view.name)

    def _index(self):
        robot, model = self.robot, self.model
        self._qpos = list(robot._ref_joint_pos_indexes)
        self._qvel = list(robot._ref_joint_vel_indexes)
        bounds = model.jnt_range[robot._ref_joint_indexes]
        self.layout = ActionLayout(tuple(robot.robot_joints) + ('finger_joint1',), (
            CommandGroup('arm', tuple(range(7)), tuple(bounds[:, 0]), tuple(bounds[:, 1])),
            CommandGroup('gripper', (7,), (0.,), (FINGER_OPEN,), scalar=True)))

    @property
    def time_s(self):
        return float(self.data.time) - self._time_origin

    def state(self):
        # finger_joint1 is half the summed travel of both fingers.
        fingers = self.robot._ref_gripper_joint_pos_indexes
        velocities = self.robot._ref_gripper_joint_vel_indexes
        return (np.r_[self.data.qpos[self._qpos], self.data.qpos[fingers] @ [.5, -.5]],
                np.r_[self.data.qvel[self._qvel], self.data.qvel[velocities] @ [.5, -.5]])

    def _advance(self, action):
        # LIBERO integrates a gripper direction (-1 opens, +1 closes); steer it to the target.
        error = action[-1] - self.state()[0][-1]
        direction = 0. if abs(error) < GRIP_TOLERANCE else -1. if error > 0 else 1.
        controller = self.robot.controller
        controller.set_goal(np.zeros(7), set_qpos=action[:7])
        # Suppress only the policy goal update; preserve native torque,
        # gripper, reward, observable and task hooks on every substep.
        previous = controller.__dict__.get('set_goal')
        controller.set_goal = lambda _action: None
        try:
            result = self.env.step(np.r_[np.zeros(7), direction])
        finally:
            if previous is None:
                del controller.set_goal
            else:
                controller.set_goal = previous
        self.env.sim.forward()
        return result

    def step(self, action):
        if self.done:
            raise RuntimeError('episode is done')
        _, reward, _, _ = self._advance(action)
        self.steps += 1
        self.reward = float(reward)
        self.success = bool(self.wrapper.check_success())
        self.done = bool((self.max_steps and self.steps >= self.max_steps)
                         or (self.terminate_on_success and self.success))

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
        return float(focal), float(focal), (self.width - 1) / 2., (self.height - 1) / 2.

    def render(self, camera):
        if camera not in self.cameras:
            raise KeyError(camera)
        # robosuite 1.4 does not select its GL context inside MjSim.render.
        # Reset candidates and independent spectator capture can change it.
        self.env.sim._render_context_offscreen.gl_ctx.make_current()
        return np.ascontiguousarray(self.env.sim.render(camera_name=camera,
                                   width=self.width, height=self.height)[::-1])

    def render_spectator(self, name):
        return self._spectator.capture(name)

    def spectator_intrinsics(self, name):
        return self._spectator.intrinsics(name)

    def close(self):
        if self._spectator is not None:
            self._spectator.close()
            self._spectator = None
        if self.wrapper is not None:
            if hasattr(self.wrapper, 'env'):
                context = getattr(self.env.sim, '_render_context_offscreen', None)
                if context is not None:
                    context.gl_ctx.make_current()
                self.wrapper.close()
            self.wrapper = None
