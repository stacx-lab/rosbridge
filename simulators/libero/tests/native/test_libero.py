"""Pinned LIBERO acceptance: every task, native rollout parity in both configurations, gripper steering and
render isolation."""
import random

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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
        sim.early_stop = True
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


def reference_world(seed, controller='JOINT_POSITION'):
    """Independent native construction, matching the documented initialization. OSC_POSE starts as upstream's
    evaluation does (libero/lifelong/metric.py): five idle actions after the initial state."""
    import torch
    from libero.libero.envs.env_wrapper import ControlEnv
    bddl, states = task_files(configure(), LiberoSim.default_task)
    random.seed(seed)
    np.random.seed(seed)
    native = ControlEnv(bddl_file_name=str(bddl), controller=controller,
                        has_renderer=False, has_offscreen_renderer=False,
                        use_camera_obs=False, camera_names=[], ignore_done=True, control_freq=20)
    native.seed(seed)
    native.env.reset()
    native.set_init_state(torch.load(states, weights_only=False)[0])
    if controller == 'OSC_POSE':
        for _ in range(5):
            native.step(np.zeros(7))
        native.sim.forward()
        return native
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


def world(sim, kind, name):
    """A body's or site's world pose, from MuJoCo, as a 4x4 transform."""
    element = getattr(sim.data, kind)(name)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = element.xmat.reshape(3, 3), element.xpos
    return result


def test_osc_initialization_and_rollout_match_upstream_osc_pose():
    """In osc a flange target that moves the grip site by a policy's delta steps exactly as upstream's OSC_POSE
    steps that delta, the clip included; the targets come from MuJoCo, not from the backend's frames."""
    sim, native = LiberoSim(cameras=(), configuration='osc'), None
    try:
        sim.reset(sim.default_task, 7, False)
        native = reference_world(7, 'OSC_POSE')

        def equal(atol):
            np.testing.assert_allclose(sim.data.qpos, native.sim.data.qpos, atol=atol, rtol=0)
            np.testing.assert_allclose(sim.data.qvel, native.sim.data.qvel, atol=atol, rtol=0)
        equal(1e-12)
        robot, rng = sim.robot, np.random.default_rng(0)
        # Every delta turns the hand (upstream keeps its orientation goal for an exactly zero one); the last
        # saturates the clip.
        deltas = [rng.uniform(-.6, .6, 6) for _ in range(9)] + [np.r_[1.5, -.2, .3, .4, 1.2, -.3]]
        for delta in deltas:
            base, hand = (world(sim, 'body', name) for name in (robot.robot_model.root_body, robot.robot_model.eef_name))
            site = world(sim, 'site', robot.controller.eef_name)
            scaled = np.clip(delta, -1, 1) * [.05, .05, .05, .5, .5, .5]
            goal = np.eye(4)
            goal[:3, :3], goal[:3, 3] = Rotation.from_rotvec(scaled[3:]).as_matrix() @ site[:3, :3], site[:3, 3] + scaled[:3]
            flange = np.linalg.inv(base) @ goal @ np.linalg.inv(site) @ hand
            sim.step(sim.state()[0], {'arm': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_quat()]})
            _, reward, _, _ = native.step(np.r_[delta, 0.])
            native.sim.forward()
            assert sim.reward == reward and sim.success == bool(native.check_success())
            equal(1e-9)
    finally:
        sim.close()
        if native is not None:
            native.close()


def test_osc_tracks_flange_poses_and_idles_as_upstream():
    sim = LiberoSim(cameras=(), configuration='osc')
    try:
        sim.reset(sim.default_task, 0, False)
        assert [group.tip for group in sim.layout.groups] == ['flange', None]
        assert [(device.name, device.kind) for device in sim.devices] == [('arm', 'joint'), ('gripper', 'gripper')]
        robot = sim.robot
        # Uncommanded, the arm idles as upstream's: the grip site stays and the hand turns onto the orientation
        # goal robosuite's controller started with, before the episode's state loaded (0.8 degrees measured).
        before = world(sim, 'site', robot.controller.eef_name)
        for _ in range(100):
            sim.step(sim.state()[0])
        after = world(sim, 'site', robot.controller.eef_name)
        assert np.linalg.norm(after[:3, 3] - before[:3, 3]) < .001
        assert Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).magnitude() < np.radians(1)
        # 5 cm up and 15 degrees about the flange's z axis, measured on robosuite's eef body.
        start = sim.tip_pose('arm')
        rotation = Rotation.from_quat(start[3:]) * Rotation.from_euler('z', 15, degrees=True)
        target = np.r_[start[:3] + [0, 0, .05], rotation.as_quat()]
        for _ in range(40):
            sim.step(sim.state()[0], {'arm': target})
        flange = np.linalg.inv(world(sim, 'body', robot.robot_model.root_body)) @ world(
            sim, 'body', robot.robot_model.eef_name)
        assert np.linalg.norm(flange[:3, 3] - target[:3]) < .002
        assert (Rotation.from_matrix(flange[:3, :3]).inv() * rotation).magnitude() < np.radians(.5)
        np.testing.assert_allclose(sim.tip_pose('arm')[:3], flange[:3, 3], atol=1e-12)
    finally:
        sim.close()
    with pytest.raises(ValueError, match="LIBERO has no configuration 'depth'"):
        LiberoSim(configuration='depth')
