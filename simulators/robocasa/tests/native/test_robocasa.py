"""Pinned RoboCasa acceptance on representative tasks: upstream splits and seeds, kitchen ids, controls and rendering."""
import random

import numpy as np
import pytest

from rosbridge_robocasa.backend import RoboCasaSim


TARGET_KITCHENS = [(i, i) for i in range(1, 11)]


def kitchen(sim):
    meta = sim.env.get_ep_meta()
    return meta['layout_id'], meta['style_id']


@pytest.mark.parametrize('task,horizon', [
    ('PickPlaceCounterToCabinet', 750),   # atomic
    ('WashLettuce', 1650),                # composite
    ('PrepareVeggieDip', None),           # composite without a dataset horizon
])
def test_representative_tasks_load_and_step_without_a_split(task, horizon):
    sim = RoboCasaSim(cameras=())
    try:
        sim.reset(task, 3)
        assert sim.steps == 0 and sim.time_s == 0 and sim.period == .05 and sim.max_steps == horizon
        layout, style = kitchen(sim)  # create_env(split=None): every kitchen and object
        assert 1 <= layout <= 60 and 1 <= style <= 60 and sim.env.obj_instance_split is None
        assert sim.measures() == {'split': None, 'layout_id': layout, 'style_id': style}
        assert sim.env.get_ep_meta()['lang']
        sim.step(sim.state()[0])
        assert sim.steps == 1 and sim.time_s == pytest.approx(.05)
        assert np.isfinite(np.concatenate(sim.state())).all()
        assert sim.reward == float(sim.success)
    finally:
        sim.close()


@pytest.mark.parametrize('task,split,seed', [('CloseFridge', 'target', 3), ('PrepareVeggieDip', 'pretrain', 5)])
def test_split_and_seed_select_the_episode_upstream_create_env_does(task, split, seed):
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import create_env

    def episode(env):
        meta = env.get_ep_meta()
        return (meta['layout_id'], meta['style_id'], meta['lang'],
                [cfg['info']['mjcf_path'].split('/')[-2] for cfg in meta['object_cfgs']], env.obj_instance_split)
    random.seed(seed)  # rosbridge also seeds the process generators, which upstream leaves as they are
    np.random.seed(seed)
    native = create_env(task, split=split, seed=seed)
    try:
        native.reset()
        expected = episode(native)
    finally:
        native.close()
    sim = RoboCasaSim(cameras=())
    try:
        sim.reset(task, seed, split=split)
        assert episode(sim.env) == expected and expected[-1] == split
        if split == 'target':
            assert kitchen(sim) in TARGET_KITCHENS
        else:
            assert min(kitchen(sim)) >= 11
        assert sim.measures() == {'split': split, 'layout_id': expected[0], 'style_id': expected[1]}
    finally:
        sim.close()


def test_kitchen_ids_override_the_split_and_failed_resets_keep_the_episode(monkeypatch):
    sim = RoboCasaSim(cameras=(), max_steps=2)
    try:
        # The split's kitchens without the task's exclusions: DeliverStraw has no target kitchen 1.
        sim.reset('DeliverStraw', 0, split='target')
        assert kitchen(sim) in [(i, i) for i in (2, 4, 7, 8, 9, 10)]
        # An id not given ranges over the split's group, as upstream's collect_demos --layout does.
        sim.reset('CloseFridge', 5, split='target', layout_id=4)
        assert kitchen(sim)[0] == 4 and 1 <= kitchen(sim)[1] <= 10 and sim.env.obj_instance_split == 'target'
        scene = dict(split='target', layout_id=12, style_id=3)  # ids given override the split's kitchens
        sim.reset('PrepareVeggieDip', 5, **scene)
        meta, qpos = sim.env.get_ep_meta(), sim.data.qpos.copy()
        assert kitchen(sim) == (12, 3) and sim.env.obj_instance_split == 'target'
        assert sim.measures() == scene
        sim.step(sim.state()[0])
        sim.reset('PrepareVeggieDip', 5, **scene)
        assert {k: sim.env.get_ep_meta()[k] for k in ('layout_id', 'style_id', 'lang')} == {
            k: meta[k] for k in ('layout_id', 'style_id', 'lang')}
        np.testing.assert_array_equal(qpos, sim.data.qpos)
        previous, rngs = sim.env, (random.getstate(), np.random.get_state()[1].copy())

        def kept():
            assert sim.env is previous and sim.steps == 0 and sim.measures() == scene
            assert random.getstate() == rngs[0]
            np.testing.assert_array_equal(np.random.get_state()[1], rngs[1])
            np.testing.assert_array_equal(qpos, sim.data.qpos)
        with monkeypatch.context() as patch:
            patch.setattr(RoboCasaSim, '_index', lambda _: (_ for _ in ()).throw(ValueError('failure')))
            with pytest.raises(ValueError, match='failure'):
                sim.reset('PrepareVeggieDip', 6, **scene)
        kept()
        for options in ({'layout_id': 0}, {'layout_id': 61}, {'style_id': True}, {'layout_id': '3'}):
            with pytest.raises(ValueError, match='must be an upstream id in 1-60'):
                sim.reset('PrepareVeggieDip', 6, **options)
        with pytest.raises(ValueError, match='split must be one of target, pretrain, all'):
            sim.reset('PrepareVeggieDip', 6, split='test')
        # A kitchen the task excludes fails before it is built.
        with pytest.raises(ValueError, match=r'DeliverStraw excludes layout 1 \(it excludes layouts \[1, 3,'):
            sim.reset('DeliverStraw', 0, layout_id=1)
        with pytest.raises(ValueError, match=r'PrepareVeggieDip excludes style 13 \(it excludes styles \[13, 29'):
            sim.reset('PrepareVeggieDip', 0, style_id=13)
        kept()
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
