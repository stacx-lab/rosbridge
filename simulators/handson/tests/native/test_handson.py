"""Pinned HandsOn acceptance in the native environment.

Isaac hosts one episode per process and its shutdown ends the process, so every check
drives HandsOnSim through a worker subprocess. Each task needs its own Isaac boot
(minutes), so the every-task sweep runs only with ROSBRIDGE_ALL_TASKS=1.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.catalog import tasks
from rosbridge.core.worker import WorkerBackend
from tests.kinex_geometry import Rig, compose, requires_kinex, rt

# The default task: both arms move freely.
TASK, STEPS = "t24_place_ball_in_bowl", 150
ALL = [t["native_task"] for t in tasks(name="handson").values()]
SELECTED = ALL if os.environ.get("ROSBRIDGE_ALL_TASKS") else ["t12_load_and_start_microwave"]
SIDES = (("right", 0), ("left", 27))
CURL = [4, 6, 7, 8, 10, 11]  # index and middle finger flexion, in the hand's finger-major order
# joint1 swings a hand vertically and the arms mirror each other: these lift both hands clear of the table.
LIFT = {"right": .1, "left": -.1}


class Native(WorkerBackend):
    simulator = "handson"
    close_timeout_s = 120.


def native(**options):
    return Native(python=sys.executable, env=dict(os.environ), **options)


@pytest.fixture(scope="module")
def sim():
    sim = native(max_steps=STEPS)
    sim.configure_spectators()
    sim.reset(TASK, 0)
    yield sim
    sim.close()


def test_native_period_layout_and_devices(sim):
    assert sim.period == pytest.approx(1 / 30) and sim.steps == 0 and sim.max_steps == STEPS
    names = sim.layout.joint_names
    for side, offset in SIDES:
        assert names[offset:offset + 7] == tuple(f"joint{i}_{side}" for i in range(1, 8))
        assert names[offset + 7:offset + 27] == tuple(f"{side}_finger{f}_joint{j}"
                                                      for f in range(1, 6) for j in range(1, 5))
    assert [(d.name, d.kind) for d in sim.devices] == [
        ("right_arm", "joint"), ("right_hand", "joint"), ("left_arm", "joint"), ("left_hand", "joint"),
        ("head_camera", "rgb"), ("left_camera", "rgb"), ("right_camera", "rgb")]


def test_arms_and_hands_track_absolute_targets(sim):
    target = sim.state()[0]
    for side, offset in SIDES:
        target[offset] += LIFT[side]
        target[[offset + 7 + i for i in CURL]] += .5
    for _ in range(45):
        sim.step(target)
    assert sim.steps == 45 and sim.time_s == pytest.approx(1.5)
    measured = sim.state()[0]
    for _, offset in SIDES:
        assert measured[offset] == pytest.approx(target[offset], abs=.02)
        hand = [offset + 7 + i for i in CURL]
        np.testing.assert_allclose(measured[hand], target[hand], atol=.1)


def test_cameras_render_the_contract_and_leave_physics_unchanged(sim):
    before = sim.state()
    for camera, size in (("head_camera", (854, 480)), ("left_camera", (848, 480)), ("right_camera", (848, 480))):
        image = sim.render(camera)
        assert image.shape == (size[1], size[0], 3) and image.dtype == np.uint8 and image.std() > 1
        assert sim.intrinsics(camera)[:2] == size
    # The head camera keeps its calibrated, off-center principal point.
    assert sim.intrinsics("head_camera")[4:] == pytest.approx((431.767, 239.919), abs=1e-3)
    for view in sim.spectators:
        assert sim.render_spectator(view.name).shape == (720, 960, 3)
        assert sim.spectator_intrinsics(view.name)[2:] == (479.5, 359.5)
    after = sim.state()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


@requires_kinex
def test_rig_matches_native_palms_fingers_and_cameras(tmp_path, sim):
    rig = Rig("handson/handson-rokae-wuji", tmp_path)
    target = sim.state()[0]
    for _, offset in SIDES:
        target[[offset + 1, offset + 3]] += .2
        target[[offset + 7 + i for i in CURL]] += .3
    for _ in range(2):
        for _ in range(15):
            sim.step(target)
        q = sim.state()[0]
        palms = {}
        for side, offset in SIDES:
            palms[side] = palm = rig.flange(f"{side}_arm", q[offset:offset + 7])
            measured = sim.native("link_pose", side, f"{side}_palm_link")
            np.testing.assert_allclose(palm[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(palm[1], measured[1], atol=1e-4)
            hand = compose(palm, rig.mount(f"{side}_hand"))
            for finger in range(1, 6):
                link = f"{side}_finger{finger}_link4"
                expected = compose(hand, rig.link(f"{side}_hand", q[offset + 7:offset + 27], link))
                measured = sim.native("link_pose", side, link)
                np.testing.assert_allclose(expected[0], measured[0], atol=1e-4)
                np.testing.assert_allclose(expected[1], measured[1], atol=1e-4)
        sim.render("head_camera")  # a capture places every camera at its nominal mount
        for camera, expected in (("head_camera", rig.mount("head_camera")),
                                 ("left_camera", compose(palms["left"], rig.mount("left_camera"))),
                                 ("right_camera", compose(palms["right"], rig.mount("right_camera")))):
            measured = rt(*sim.camera_pose(camera))
            np.testing.assert_allclose(expected[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(expected[1], measured[1], atol=1e-4)


def test_episode_ends_at_its_step_limit_and_serves_one_episode(sim):
    hold = sim.state()[0]
    while not sim.done:
        sim.step(hold)
    assert sim.steps == STEPS and sim.time_s == pytest.approx(STEPS / 30)
    with pytest.raises(RuntimeError, match="done"):
        sim.step(hold)
    with pytest.raises(ValueError, match="one episode"):
        sim.reset(TASK, 0)
    assert set(sim.native("measures")) == {"failure"}


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_and_steps(task):
    sim = native()
    try:
        sim.reset(task, 0)
        assert not sim.done and sim.max_steps > 0
        sim.step(sim.state()[0])
        assert sim.steps == 1 and np.isfinite(sim.state()[0]).all() and sim.reward in (0., 1.)
    finally:
        sim.close()
