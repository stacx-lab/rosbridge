"""Pinned RoboTwin acceptance: timing, native gripper mapping, resets, rendering and every task.

Each reset rebuilds the scene and runs upstream's expert dry run, so the every-task
sweep runs only with ROSBRIDGE_ALL_TASKS=1.
"""
import os

import numpy as np
import pytest

from rosbridge.catalog import tasks
from rosbridge_robotwin.backend import RoboTwinSim

SEEDS = {t["native_task"]: t.get("seed", 0) for t in tasks(name="robotwin").values()}
SELECTED = list(SEEDS) if os.environ.get("ROSBRIDGE_ALL_TASKS") else ["handover_block", "shake_bottle"]


def physics(sim):
    entities = [sim.env.robot.left_entity, sim.env.robot.right_entity]
    return np.concatenate([np.r_[e.get_qpos(), e.get_qvel()] for e in entities])


@pytest.fixture(scope="module")
def sim():
    sim = RoboTwinSim(max_steps=40)
    sim.configure_spectators()
    sim.reset("handover_block", 3)
    yield sim
    sim.close()


def test_native_period_layout_and_gripper_mapping(sim):
    assert sim.period == pytest.approx(.04) and sim.steps == 0 and sim.time_s == 0
    assert sim.layout.joint_names[6] == "left_gripper" and sim.layout.joint_names[13] == "right_gripper"
    assert [d.kind for d in sim.devices] == ["joint", "gripper", "joint", "gripper", "rgbd", "rgbd", "rgbd"]
    target = sim.state()[0]
    target[[6, 13]] = .5
    for _ in range(25):
        sim.step(target)
    assert sim.steps == 25 and sim.time_s == pytest.approx(1.)
    robot = sim.env.robot
    lower, upper = robot.left_gripper_scale
    # Upstream's normalized command: finger drive target = lower + v * (upper - lower).
    assert robot.left_gripper[0][0].get_drive_target()[0] == pytest.approx(lower + .5 * (upper - lower), abs=1e-6)
    assert sim.state()[0][6] == pytest.approx(.5, abs=.02)
    finger = robot.left_entity.get_qpos()[robot.left_entity.get_active_joints().index(robot.left_gripper[0][0])]
    assert sim.state()[0][6] == pytest.approx((finger - lower) / (upper - lower))


def test_rendering_and_spectators_leave_physics_unchanged(sim):
    before = physics(sim)
    images = {camera: sim.render(camera) for camera in sim.cameras}
    for camera, image in images.items():
        assert image.shape == (240, 320, 3) and image.std() > 1
        depth = sim.render_depth(camera)
        assert depth.dtype == np.float32 and (depth > 0).mean() > .9
    for view in sim.spectators:
        frame = sim.render_spectator(view.name)
        assert frame.shape == (720, 960, 3) and frame.std() > 1
        fx, fy, cx, cy = sim.spectator_intrinsics(view.name)
        assert fx == pytest.approx(fy) and (cx, cy) == pytest.approx((479.5, 359.5))
    np.testing.assert_array_equal(before, physics(sim))


def test_reset_is_seeded_bounded_and_failures_keep_the_episode(sim, monkeypatch):
    sim.reset("handover_block", 3)
    first = physics(sim)
    for _ in range(40):
        sim.step(sim.state()[0])
    assert sim.done and sim.steps == 40
    with pytest.raises(RuntimeError, match="done"):
        sim.step(sim.state()[0])
    sim.reset("handover_block", 3)
    np.testing.assert_allclose(first, physics(sim), atol=1e-6)
    previous = sim.env
    with monkeypatch.context() as patch:
        patch.setattr(RoboTwinSim, "_index", lambda _: (_ for _ in ()).throw(ValueError("failure")))
        with pytest.raises(ValueError, match="failure"):
            sim.reset("handover_block", 4)
    assert sim.env is previous and sim.steps == 0
    np.testing.assert_allclose(first, physics(sim), atol=1e-6)


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_and_steps(task):
    sim = RoboTwinSim(cameras=())
    try:
        sim.reset(task, SEEDS[task])
        assert sim.max_steps == int(sim.env.step_lim) and not sim.done
        sim.step(sim.state()[0])
        assert sim.steps == 1 and np.isfinite(sim.state()[0]).all()
        assert sim.reward == float(sim.success)
    finally:
        sim.close()
