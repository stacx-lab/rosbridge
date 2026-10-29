"""Pinned RoboDojo acceptance in the native environment.

Isaac hosts one app per process and its shutdown ends the process, so every check
drives RoboDojoSim through a worker subprocess. Each task needs its own Isaac boot
(minutes), so the every-task sweep runs only with ROSBRIDGE_ALL_TASKS=1.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.catalog import tasks
from rosbridge.core.worker import WorkerBackend
from tests.kinex_geometry import Rig, compose, requires_kinex, rt

ALL = [t["native_task"] for t in tasks(name="robodojo").values()]
SELECTED = ALL if os.environ.get("ROSBRIDGE_ALL_TASKS") else ["stack_bowls"]


class Native(WorkerBackend):
    simulator = "robodojo"
    close_timeout_s = 120.


def native(**options):
    return Native(python=sys.executable, env=dict(os.environ), **options)


@pytest.fixture(scope="module")
def sim():
    sim = native(max_steps=60)
    sim.configure_spectators()
    sim.reset("pack_objects_into_box", 0)
    yield sim
    sim.close()


def test_native_period_layout_and_gripper_mapping(sim):
    assert sim.period == pytest.approx(.04) and sim.steps == 0 and sim.time_s == 0
    names = sim.layout.joint_names
    assert names[:6] == names[7:13] == tuple(f"joint{i}" for i in range(1, 7))
    assert (names[6], names[13]) == ("left_gripper", "right_gripper")
    assert [d.kind for d in sim.devices] == ["joint", "gripper", "joint", "gripper", "rgb", "rgb", "rgb"]
    target = sim.state()[0]
    target[0] += .15
    target[[6, 13]] = .5
    for _ in range(25):
        sim.step(target)
    assert sim.steps == 25 and sim.time_s == pytest.approx(1.)
    measured = sim.state()[0]
    assert measured[0] == pytest.approx(target[0], abs=.02)
    # Upstream's normalized ee value: finger target = -0.01 + 0.054 v, measured on the primary finger.
    assert measured[[6, 13]] == pytest.approx([.5, .5], abs=.05)


def test_rendering_and_spectators_leave_physics_unchanged(sim):
    before = sim.state()
    for camera in sim.cameras:
        image = sim.render(camera)
        assert image.shape == (480, 640, 3) and image.std() > 1
        width, height, fx, fy, cx, cy = sim.intrinsics(camera)
        assert (width, height, cx, cy) == (640, 480, 319.5, 239.5) and fx == fy
    for view in sim.spectators:
        assert sim.render_spectator(view.name).shape == (720, 960, 3)
        assert sim.spectator_intrinsics(view.name)[2:] == (479.5, 359.5)
    after = sim.state()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_reset_is_bounded_reloads_the_layout_and_serves_one_task(sim):
    sim.reset("pack_objects_into_box", 0)
    initial = sim.state()[0]
    for _ in range(60):
        sim.step(initial + np.r_[.1, np.zeros(13)])
    assert sim.done and sim.steps == 60
    with pytest.raises(RuntimeError, match="done"):
        sim.step(initial)
    sim.reset("pack_objects_into_box", 0)
    assert not sim.done and sim.steps == 0
    np.testing.assert_allclose(sim.state()[0], initial, atol=1e-3)
    with pytest.raises(ValueError, match="one task"):
        sim.reset("stack_bowls", 0)
    assert not sim.done and sim.task == "pack_objects_into_box"


@requires_kinex
def test_rig_matches_native_arms_fingers_and_cameras(tmp_path, sim):
    rig = Rig("robodojo/robodojo-arx-x5", tmp_path)
    target = sim.state()[0]
    target[[0, 7]] += .2
    target[[1, 8]] += .3
    target[[6, 13]] = .6
    for _ in range(2):
        for _ in range(15):
            sim.step(target)
        q = sim.state()[0]
        for side, offset in (("left", 0), ("right", 7)):
            flange = rig.flange(f"{side}_arm", q[offset:offset + 6])
            np.testing.assert_allclose(flange[0], sim.native("link_pose", side, "link6")[0], atol=1e-4)
            np.testing.assert_allclose(flange[1], sim.native("link_pose", side, "link6")[1], atol=1e-4)
            grip = compose(flange, rig.mount(f"{side}_gripper"))
            for finger, link, atol in (("finger_a", "link7", 1e-4), ("finger_b", "link8", 2e-3)):
                expected = compose(grip, rig.link(f"{side}_gripper", [q[offset + 6]], finger))
                measured = sim.native("link_pose", side, link)
                np.testing.assert_allclose(expected[0], measured[0], atol=atol)
                np.testing.assert_allclose(expected[1], measured[1], atol=atol)
            camera = compose(flange, rig.mount(f"{side}_camera"))
            measured = rt(*sim.camera_pose(f"{side}_camera"))
            np.testing.assert_allclose(camera[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(camera[1], measured[1], atol=1e-4)
        head = rig.mount("head_camera"), rt(*sim.camera_pose("head_camera"))
        np.testing.assert_allclose(head[0][0], head[1][0], atol=1e-4)
        np.testing.assert_allclose(head[0][1], head[1][1], atol=1e-4)


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_and_steps(task):
    sim = native()
    try:
        sim.reset(task, 0)
        assert not sim.done and sim.max_steps > 0
        sim.step(sim.state()[0])
        assert sim.steps == 1 and np.isfinite(sim.state()[0]).all() and 0 <= sim.reward <= 1
    finally:
        sim.close()
