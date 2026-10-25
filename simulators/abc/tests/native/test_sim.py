"""Smoke test against the real simulator; needs assets for the task."""

import numpy as np
import pytest

pytest.importorskip("abc_sim")
from rosbridge_abc.backend import AbcSim  # noqa: E402


def test_step_moves_toward_command(monkeypatch):
    sim = AbcSim(cameras=("top",), width=64, height=48)
    try:
        sim.reset("put_plastic_bottles_in_bin", seed=0)
        # Adapter timing must come from the original environment, not a ROS rate.
        import abc_sim
        upstream = abc_sim.make_env(task=sim.task, render_cameras=False)
        try:
            upstream.reset(seed=0)
            native_period = float(upstream.model.opt.timestep) * upstream._control_decimation
            assert sim.env.model.opt.timestep == upstream.model.opt.timestep
            assert sim.env._control_decimation == upstream._control_decimation
            assert sim.period == native_period
            upstream_start = float(upstream.data.time)
            upstream.step(upstream.get_state(), render_obs=False)
            assert float(upstream.data.time) - upstream_start == pytest.approx(native_period)
        finally:
            upstream.close()
        pos, _ = sim.state()
        target = pos.copy()
        target[0] += 0.2
        for _ in range(30):
            before = sim.time_s
            sim.step(target)
            assert sim.time_s - before == pytest.approx(native_period, abs=1e-12)
        assert sim.time_s == pytest.approx(30 * sim.period, abs=1e-6)
        assert abs(sim.state()[0][0] - target[0]) < abs(pos[0] - target[0])
        assert sim.render("top").shape == (48, 64, 3)
        assert np.isfinite(sim.intrinsics("top")).all()
        original_env, original_time = sim.env, sim.time_s
        with monkeypatch.context() as patch:
            def fail_index(self, model):
                raise ValueError("synthetic indexing failure")
            patch.setattr(AbcSim, "_index", fail_index)
            with pytest.raises(ValueError, match="indexing failure"):
                sim.reset(sim.task, seed=9)
        assert sim.env is original_env and sim.time_s == original_time
        sim.step(target)
        assert sim.time_s - original_time == pytest.approx(native_period)
    finally:
        sim.close()


def test_spectator_isolation_and_reset_failure(monkeypatch):
    sim = AbcSim(cameras=('top', 'left', 'right'), width=64, height=48)
    try:
        sim.reset(sim.default_task, seed=0)
        images = {c: sim.render(c) for c in sim.cameras}
        state = sim.env.data.qpos.copy(), sim.env.data.qvel.copy(), float(sim.env.data.time)
        vis = (sim.env.model.vis.global_.offwidth, sim.env.model.vis.global_.offheight)
        from rosbridge.core.mujoco_rendering import SpectatorRenderer
        sim.configure_spectators(960, 720)
        renderer = SpectatorRenderer(sim.env.model, sim.env.data, sim.spectators)
        try:
            for view in sim.spectators:
                rgb = renderer.capture(view.name)
                assert rgb.shape == (720, 960, 3) and rgb.std() > 1
                assert np.isfinite(renderer.intrinsics(view.name)).all()
                np.testing.assert_array_equal(state[0], sim.env.data.qpos)
                np.testing.assert_array_equal(state[1], sim.env.data.qvel)
                assert float(sim.env.data.time) == state[2]
                assert vis == (sim.env.model.vis.global_.offwidth, sim.env.model.vis.global_.offheight)
                for camera in sim.cameras:
                    np.testing.assert_array_equal(images[camera], sim.render(camera))
        finally:
            renderer.close()
        sim.reset(sim.task, seed=0)
        previous = sim.env
        with monkeypatch.context() as patch:
            def fail(*args):
                raise RuntimeError('capture failed')
            patch.setattr(SpectatorRenderer, 'capture', fail)
            with pytest.raises(RuntimeError, match='capture failed'):
                sim.reset(sim.task, seed=1)
        assert sim.env is previous
        assert sim.render_spectator('third_person').shape == (720, 960, 3)
        sim.step(sim.state()[0])
        assert sim.time_s == pytest.approx(sim.period)
    finally:
        sim.close()


def test_wrapped_actions_match_native_rollout():
    import abc_sim
    sim = AbcSim(cameras=(), width=64, height=48)
    upstream = abc_sim.make_env(task=sim.default_task, render_cameras=False)
    try:
        sim.reset(sim.default_task, seed=19)
        upstream.reset(seed=19)
        np.testing.assert_allclose(sim.env.data.qpos, upstream.data.qpos, atol=1e-12)
        target = np.asarray(upstream.get_state(), dtype=np.float32)
        target[0] += .1
        target[7] += .1
        for _ in range(10):
            sim.step(target)
            upstream.step(target, render_obs=False)
            np.testing.assert_allclose(sim.env.data.qpos, upstream.data.qpos, atol=1e-12)
            np.testing.assert_allclose(sim.env.data.qvel, upstream.data.qvel, atol=1e-12)
    finally:
        sim.close()
        upstream.close()


from rosbridge.catalog import tasks  # noqa: E402


@pytest.mark.parametrize('task', [entry['native_task'] for entry in tasks(name='abc').values()])
def test_catalog_assets_load_and_step(task):
    """Check real XML/mesh references, beyond the preparation marker inventory."""
    sim = AbcSim(cameras=(), width=64, height=48, max_steps=2)
    try:
        sim.reset(task, seed=0)
        before = sim.time_s
        sim.step(sim.state()[0])
        assert sim.time_s - before == pytest.approx(sim.period)
        assert np.isfinite(sim.state()[0]).all()
    finally:
        sim.close()


def test_rgbd_cameras_and_supplemental_judging():
    sim = AbcSim()
    try:
        sim.reset("put_markers_in_top_drawer", 0, False)
        assert {d.name: d.kind for d in sim.devices if d.kind not in ("joint", "gripper")} == {
            "camera_top": "rgbd", "camera_left": "rgbd", "camera_right": "rgbd"}
        # Upstream ships no evaluator for this task; the judge-only rule scores it.
        assert sim._supplemental is not None and sim.success is False and sim.reward == 0.
        image = sim.render("top")
        depth = sim.render_depth("top")
        assert depth.shape == (sim.height, sim.width) and depth.dtype == np.float32
        assert (depth > 0).mean() > .9 and 0.2 < np.median(depth[depth > 0]) < 2.
        # Depth acquisition leaves the RGB renderer in color mode; repeated
        # MuJoCo color renders may already differ by one intensity level.
        assert np.abs(image.astype(int) - sim.render("top")).max() <= 1
        sim.step(sim.state()[0])
        assert sim._supplemental.last_time == pytest.approx(sim.env.data.time)
    finally:
        sim.close()
