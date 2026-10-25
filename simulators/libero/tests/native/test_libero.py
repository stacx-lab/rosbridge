"""Pinned LIBERO acceptance: every task, native rollout parity, gripper steering and render isolation."""
import random

import numpy as np
import pytest

from rosbridge.catalog import tasks
from rosbridge_libero.backend import LiberoSim
from rosbridge_libero.assets import configure, task_files


@pytest.mark.parametrize('task', [t['native_task'] for t in tasks(name='libero').values()])
def test_catalog_task_loads_and_steps(task):
    sim = LiberoSim(cameras=())
    try:
        sim.reset(task, seed=17, randomize=False)
        assert sim.steps == 0 and sim.time_s == 0 and sim.period == .05
        sim.step(sim.state()[0])
        assert sim.steps == 1 and sim.time_s == pytest.approx(.05)
        assert np.isfinite(sim.state()[0]).all()
        assert sim.reward == float(sim.env.reward())
        assert sim.success == bool(sim.wrapper.check_success())
    finally:
        sim.close()


def test_reset_determinism_failure_and_budget(monkeypatch):
    sim = LiberoSim(cameras=(), max_steps=2)
    try:
        sim.reset(sim.default_task, 19)
        initial = sim.data.qpos.copy()
        sim.step(sim.state()[0])
        sim.reset(sim.default_task, 19)
        np.testing.assert_array_equal(initial, sim.data.qpos)
        previous, timestamp = sim.env, sim.time_s
        with monkeypatch.context() as patch:
            patch.setattr(LiberoSim, '_index', lambda _: (_ for _ in ()).throw(ValueError('failure')))
            with pytest.raises(ValueError, match='failure'):
                sim.reset(sim.default_task, 20)
        assert sim.env is previous and sim.time_s == timestamp
        np.testing.assert_array_equal(initial, sim.data.qpos)
        for _ in range(2):
            sim.step(sim.state()[0])
        assert sim.done and sim.steps == 2 and sim.time_s == pytest.approx(.1)
        with pytest.raises(RuntimeError, match='done'):
            sim.step(sim.state()[0])
        sim.reset(sim.default_task, 18, False)
        assert not sim.done
        sim.terminate_on_success = True
        monkeypatch.setattr(sim.wrapper, 'check_success', lambda: True)
        sim.step(sim.state()[0])
        assert sim.done and sim.success
    finally:
        sim.close()


def test_rendering_calibration_and_failed_capture_preserve_world(monkeypatch):
    sim = LiberoSim()
    sim.configure_spectators()
    try:
        sim.reset(sim.default_task, 0, False)
        state = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.time
        images = {c: sim.render(c) for c in sim.cameras}
        vis = sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight
        for view in sim.spectators:
            rgb = sim.render_spectator(view.name)
            assert rgb.shape == (720, 960, 3) and rgb.std() > 1
            intrinsics = sim.spectator_intrinsics(view.name)
            expected_focal = 360 / np.tan(np.deg2rad(sim.model.camera(view.source).fovy[0]) / 2)
            assert intrinsics == pytest.approx((expected_focal, expected_focal, 479.5, 359.5))
            np.testing.assert_array_equal(state[0], sim.data.qpos)
            np.testing.assert_array_equal(state[1], sim.data.qvel)
            assert state[2] == sim.data.time
            assert vis == (sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight)
            for camera in sim.cameras:
                np.testing.assert_array_equal(images[camera], sim.render(camera))
                pos, quat = sim.camera_pose(camera)
                assert np.isfinite(pos).all() and np.linalg.norm(quat) == pytest.approx(1.)
        from rosbridge.core.mujoco_rendering import SpectatorRenderer
        old = sim.env
        with monkeypatch.context() as patch:
            def fail(*_):
                raise RuntimeError('capture failure')
            patch.setattr(SpectatorRenderer, 'capture', fail)
            with pytest.raises(RuntimeError, match='capture failure'):
                sim.reset(sim.default_task, 1)
        assert sim.env is old
        np.testing.assert_array_equal(images['agentview'], sim.render('agentview'))
        assert sim.render_spectator('wrist').shape == (720, 960, 3)
        sim.reset(sim.default_task, 0, False)
        for camera in sim.cameras:
            np.testing.assert_array_equal(images[camera], sim.render(camera))
    finally:
        sim.close()


def reference_world(seed):
    """Independent native construction, matching the documented initialization."""
    import torch
    from libero.libero.envs.env_wrapper import ControlEnv
    bddl, states = task_files(configure(), LiberoSim.default_task)
    random.seed(seed)
    np.random.seed(seed)
    native = ControlEnv(bddl_file_name=str(bddl), controller='JOINT_POSITION',
                        has_renderer=False, has_offscreen_renderer=False,
                        use_camera_obs=False, camera_names=[], ignore_done=True, control_freq=20)
    native.seed(seed)
    native.env.reset()
    native.set_init_state(torch.load(states, weights_only=False)[0])
    robot = native.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    low, high = native.sim.model.actuator_ctrlrange[robot._ref_joint_gripper_actuator_indexes].T
    positions = native.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
    robot.gripper.current_action = np.clip(2 * (positions - low) / (high - low) - 1, -1, 1)
    target = robot.controller.goal_qpos.copy()
    for _ in range(5):
        native_joint_step(native, target, 0.)
    return native


def native_joint_step(native, target, gripper):
    controller = native.env.robots[0].controller
    controller.set_goal(np.zeros(7), set_qpos=target)
    controller.set_goal = lambda _: None
    try:
        result = native.step(np.r_[np.zeros(7), gripper])
        native.sim.forward()
        return result
    finally:
        del controller.set_goal


def test_native_joint_rollout_parity_and_gripper_steering():
    sim, native = LiberoSim(cameras=()), None
    try:
        sim.reset(sim.default_task, 7, False)
        native = reference_world(7)
        def equal():
            np.testing.assert_allclose(sim.data.qpos, native.sim.data.qpos, atol=1e-12, rtol=0)
            np.testing.assert_allclose(sim.data.qvel, native.sim.data.qvel, atol=1e-12, rtol=0)
            assert sim.success == bool(native.check_success())
        equal()
        # Holding the measured finger position commands the native neutral direction.
        target = sim.state()[0]
        target[0] += .08
        for _ in range(5):
            sim.step(target)
            _, reward, _, _ = native_joint_step(native, target[:7], 0.)
            assert sim.reward == reward
            equal()
        for goal, reached in ((0., lambda q: q < .01), (.04, lambda q: q > .03)):
            target[-1] = goal
            for _ in range(30):
                sim.step(target)
            assert reached(sim.state()[0][-1]), sim.state()[0][-1]
    finally:
        sim.close()
        if native is not None:
            native.close()
