"""Pinned RoboCasa acceptance on representative tasks: splits, determinism, native controls and rendering."""
import numpy as np
import pytest

from rosbridge_robocasa.backend import RoboCasaSim


@pytest.mark.parametrize('task,horizon,target', [
    ('PickPlaceCounterToCabinet', 750, True),   # atomic, evaluation split
    ('WashLettuce', 1650, True),                # composite, evaluation split
    ('PrepareVeggieDip', None, False),          # composite without a dataset horizon
])
def test_representative_tasks_load_step_and_keep_their_split(task, horizon, target):
    sim = RoboCasaSim(cameras=())
    try:
        sim.reset(task, 3)
        assert sim.steps == 0 and sim.time_s == 0 and sim.period == .05 and sim.max_steps == horizon
        meta = sim.env.get_ep_meta()
        if target:
            assert meta['layout_id'] == meta['style_id'] and 1 <= meta['layout_id'] <= 10
            assert sim.env.obj_instance_split == 'target'
        assert meta['lang']
        sim.step(sim.state()[0])
        assert sim.steps == 1 and sim.time_s == pytest.approx(.05)
        assert np.isfinite(np.concatenate(sim.state())).all()
        assert sim.reward == float(sim.success)
    finally:
        sim.close()


def test_seed_determines_the_kitchen_and_failed_resets_keep_the_episode(monkeypatch):
    sim = RoboCasaSim(cameras=(), max_steps=2)
    try:
        sim.reset(sim.default_task, 5)
        meta, qpos = sim.env.get_ep_meta(), sim.data.qpos.copy()
        sim.step(sim.state()[0])
        sim.reset(sim.default_task, 5)
        assert {k: sim.env.get_ep_meta()[k] for k in ('layout_id', 'style_id', 'lang')} == {
            k: meta[k] for k in ('layout_id', 'style_id', 'lang')}
        np.testing.assert_array_equal(qpos, sim.data.qpos)
        previous = sim.env
        with monkeypatch.context() as patch:
            patch.setattr(RoboCasaSim, '_index', lambda _: (_ for _ in ()).throw(ValueError('failure')))
            with pytest.raises(ValueError, match='failure'):
                sim.reset(sim.default_task, 6)
        assert sim.env is previous and sim.steps == 0
        np.testing.assert_array_equal(qpos, sim.data.qpos)
        for _ in range(2):
            sim.step(sim.state()[0])
        assert sim.done and sim.time_s == pytest.approx(.1)
        with pytest.raises(RuntimeError, match='done'):
            sim.step(sim.state()[0])
    finally:
        sim.close()


def test_native_controls_follow_the_device_semantics():
    sim = RoboCasaSim(cameras=())
    try:
        sim.reset(sim.default_task, 0)
        home = sim.state()[0]
        np.testing.assert_allclose(home[:7], sim.robot.init_qpos, atol=.1)  # native reset noise
        action = home.copy()
        action[0] += .1     # arm joint 1, radians
        action[8] = .1      # torso lift, metres
        action[7] = 1.      # gripper closes
        for _ in range(10):
            sim.step(action)
        q = sim.state()[0]
        assert q[7] == 1.  # the integrated command reaches the limit exactly
        for _ in range(30):
            sim.step(action)
        q = sim.state()[0]
        assert abs(q[0] - action[0]) < .02 and abs(q[8] - .1) < .01
        action[7] = .5  # an intermediate target holds within one native increment
        for _ in range(5):
            sim.step(action)
        assert abs(sim.state()[0][7] - .5) <= .1
        # Base: the native command is body-frame velocity, also after the body turned.
        action[9:] = (0., 0., 1.)
        for _ in range(100):
            if sim.state()[0][11] >= np.pi / 2:
                break
            sim.step(action)
        assert sim.state()[0][11] >= np.pi / 2
        action[9:] = 0.
        for _ in range(10):
            sim.step(action)
        start = sim.state()[0][9:]
        action[9:] = (-.5, 0., 0.)  # backward, away from the counter it was facing
        for _ in range(10):
            sim.step(action)
        q, v = sim.state()
        moved = q[9:11] - start[:2]
        heading = np.array([np.cos(q[11]), np.sin(q[11])])
        sideways = heading[0] * moved[1] - heading[1] * moved[0]
        assert moved @ heading < -.05 and abs(sideways) < .2 * np.linalg.norm(moved)
        assert v[9] < -.1 and abs(v[10]) < .3 * abs(v[9])
    finally:
        sim.close()


def test_rgbd_rendering_is_registered_and_never_advances_the_world():
    from robosuite.utils.camera_utils import get_real_depth_map
    sim = RoboCasaSim()
    sim.configure_spectators(320, 240)
    try:
        sim.reset(sim.default_task, 0)
        qpos, steps = sim.data.qpos.copy(), sim.steps
        for camera in sim.cameras:
            rgb, depth = sim.render(camera), sim.render_depth(camera)
            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
            assert depth.shape == (256, 256) and depth.dtype == np.float32 and (depth > 0).mean() > .9
            # Native multisampled depth agrees away from edges; ours samples pixel centres.
            _, buffer = sim.env.sim.render(camera_name=camera, width=256, height=256, depth=True)
            native = get_real_depth_map(sim.env.sim, buffer)[::-1]
            assert np.median(np.abs(native - depth)[depth > 0]) < .005
        assert sim.render_spectator('third_person').shape == (240, 320, 3)
        assert sim.steps == steps
        np.testing.assert_array_equal(qpos, sim.data.qpos)
        with pytest.raises(KeyError):
            sim.render('robot0_agentview_center')
    finally:
        sim.close()
