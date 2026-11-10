"""Pinned BEHAVIOR acceptance with the R1Pro in the native environment.

Isaac hosts one app per process and OmniGibson's shutdown ends it, so every check drives
R1ProSim through a worker subprocess. Each activity loads its own house (minutes), so the
every-task sweep runs only with ROSBRIDGE_ALL_TASKS=1.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.catalog import tasks
from rosbridge.core.worker import WorkerBackend
from tests.kinex_geometry import Rig, compose, requires_kinex

ALL = [t["native_task"] for t in tasks(name="behavior").values()]
SELECTED = ALL if os.environ.get("ROSBRIDGE_ALL_TASKS") else ["picking_up_trash"]


class Native(WorkerBackend):
    simulator = "behavior"
    close_timeout_s = 120.


def native(**options):
    return Native(python=sys.executable, env=dict(os.environ), **options)


@pytest.fixture(scope="module")
def sim():
    sim = native(max_steps=90)
    sim.reset("turning_on_radio", 0)
    yield sim
    sim.close()


def channels(sim, device):
    return list(next(g for g in sim.layout.groups if g.name == device).indices)


def test_evaluator_period_layout_and_full_resolution_rgbd_cameras(sim):
    assert sim.period == pytest.approx(1 / 30) and sim.max_steps == 90 and sim.steps == 0
    assert [d.name for d in sim.devices] == ["base", "torso", "left_arm", "left_gripper", "right_arm",
                                             "right_gripper", "head_camera", "left_camera", "right_camera"]
    assert sim.layout.joint_names[14] == "left_gripper" and len(sim.layout.joint_names) == 23
    assert sim.intrinsics("head_camera") == pytest.approx((720, 720, 306., 306., 359.5, 359.5))
    for camera in ("left_camera", "right_camera"):
        width, height, fx, fy, cx, cy = sim.intrinsics(camera)
        assert (width, height, cx, cy) == (480, 480, 239.5, 239.5) and fx == fy
    before = sim.state()
    for camera in sim.cameras:
        width, height = sim.intrinsics(camera)[:2]
        image, depth = sim.render(camera), sim.render_depth(camera)
        assert image.shape == (height, width, 3) and image.dtype == np.uint8 and image.std() > 1
        assert depth.shape == (height, width) and depth.dtype == np.float32 and (depth > 0).mean() > .9
    after = sim.state()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_every_step_renders_and_frames_follow_the_robot(sim):
    sim.reset("turning_on_radio", 0)
    target = sim.state()[0]
    target[channels(sim, "base")] = 0., 0., 1.
    depths = []
    for _ in range(3):
        sim.step(target)  # unread, it renders too
        sim.step(target)
        depths.append(sim.render_depth("head_camera"))
    # Turning, each frame read differs from the last: the steps rendered, and reads do not render
    assert all(np.median(np.abs(new - old)) > 1e-3 for old, new in zip(depths, depths[1:]))
    sim.reset("turning_on_radio", 0)  # the next checks start still, facing ahead


def test_normalized_base_smooth_grippers_and_joint_targets(sim):
    target = sim.state()[0]
    base, grip, arm = channels(sim, "base"), channels(sim, "right_gripper"), channels(sim, "left_arm")
    target[base] = .5, 0., 0.
    target[grip] = -1.
    target[arm[3]] -= .2
    for _ in range(30):
        sim.step(target)
    position, velocity = sim.state()
    # The evaluator scales a normalized 0.5 to 0.375 m/s along the base's x axis.
    assert velocity[base[0]] == pytest.approx(.375, abs=.05) and position[base[0]] > .25
    assert abs(position[base[1]]) < .05 and position[grip[0]] < -.95
    assert position[arm[3]] == pytest.approx(target[arm[3]], abs=.02)


def test_step_limit_ends_the_episode_and_reset_restores_the_instance(sim):
    initial = sim.state()[0]
    while not sim.done:
        sim.step(initial)
    assert sim.steps == 90 and not sim.success
    assert sim.native("measures") == {"q_score": 0.}  # the radio, the goal's one literal, is still off
    with pytest.raises(RuntimeError, match="done"):
        sim.step(initial)
    sim.reset("turning_on_radio", 0)
    assert not sim.done and sim.steps == 0 and sim.time_s == 0
    np.testing.assert_allclose(sim.state()[0][:3], 0, atol=1e-6)
    with pytest.raises(ValueError, match="one task"):
        sim.reset("picking_up_trash", 0)
    assert not sim.done and sim.task == "turning_on_radio"


def rt(matrix):
    return matrix[:3, :3].tolist(), matrix[:3, 3].tolist()


def assert_pose(expected, measured, atol=1e-4):
    np.testing.assert_allclose(expected[0], measured[0], atol=atol)
    np.testing.assert_allclose(expected[1], measured[1], atol=atol)


@requires_kinex
def test_rig_matches_native_chains_grippers_and_cameras(tmp_path, sim):
    rig = Rig("behavior/behavior-r1pro", tmp_path)
    target = sim.state()[0]
    torso, grips = channels(sim, "torso"), [channels(sim, f"{side}_gripper")[0] for side in ("left", "right")]
    target[torso[1]] += .1
    target[grips] = 0.
    for _ in range(30):
        sim.step(target)
    q = sim.state()[0]
    base = np.linalg.inv(sim.native("link_pose", "base_link"))

    def measured(link):
        return rt(base @ sim.native("link_pose", link))
    top = compose(rig.mount("torso"), rig.link("torso", q[torso], "top"))
    assert_pose(top, measured("torso_link4"))
    assert_pose(compose(top, rig.mount("head_camera")), rt(base @ sim.camera_pose("head_camera")))
    for side in ("left", "right"):
        flange = compose(top, compose(rig.mount(f"{side}_arm"), rig.link(f"{side}_arm", q[channels(sim, f"{side}_arm")])))
        assert_pose(flange, measured(f"{side}_arm_link7"))
        assert_pose(compose(flange, rig.mount(f"{side}_camera")), rt(base @ sim.camera_pose(f"{side}_camera")))
        hand = compose(flange, rig.mount(f"{side}_gripper"))
        grip = q[channels(sim, f"{side}_gripper")]
        for finger, link in (("finger_a", "1"), ("finger_b", "2")):
            assert_pose(compose(hand, rig.link(f"{side}_gripper", grip, finger)),
                        measured(f"{side}_gripper_finger_link{link}"), atol=2e-3)


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_steps_and_records(task):
    sim = native()
    try:
        sim.configure_spectators()
        sim.reset(task, 0)
        assert not sim.done and sim.max_steps > 0
        sim.step(sim.state()[0])
        assert sim.steps == 1 and np.isfinite(sim.state()[0]).all() and sim.reward in (0., 1.)
        assert 0. <= sim.native("measures")["q_score"] <= 1.
        for view in sim.spectators:
            assert sim.render_spectator(view.name).shape == (720, 960, 3)
            assert sim.spectator_intrinsics(view.name)[2:] == (479.5, 359.5)
    finally:
        sim.close()
