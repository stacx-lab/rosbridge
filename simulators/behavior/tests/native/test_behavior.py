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


def test_reads_render_once_after_a_step_with_omnigibsons_antialiasing_and_current_frames(sim):
    settings = sim.native("render_settings")
    assert settings["/rtx/post/aa/op"] != 4 and settings["/rtx/post/dlss/execMode"] == 0  # DLSS Performance
    sim.reset("turning_on_radio", 0)
    start = sim.native("render_count")
    sim.render("head_camera")
    assert sim.native("render_count") == start  # env.reset rendered the reset state
    target = sim.state()[0]
    target[channels(sim, "base")] = 0., 0., 1.
    for _ in range(3):
        sim.step(target)  # unread steps do not render
    assert sim.native("render_count") == start
    depths = []
    for step in range(4):
        sim.step(target)
        before = sim.state()
        for camera in sim.cameras:
            sim.render(camera), sim.render_depth(camera)
        np.testing.assert_array_equal(before[0], sim.state()[0])  # renders do not step physics
        depths.append(sim.render_depth("head_camera"))
        # The first read after unread steps has no recent history; later reads render once.
        assert sim.native("render_count") == start + 3 + step
    # Turning, consecutive frames differ; more renders of the same state change nothing, so
    # each read's frame showed the state it was read in, not the previous step's.
    assert np.median(np.abs(depths[-1] - depths[-2])) > 1e-3
    sim.native("_render")
    sim.native("__setattr__", "_frames", {})
    assert np.median(np.abs(sim.render_depth("head_camera") - depths[-1])) < 1e-3
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


def test_goal_particles_beyond_a_locked_door_are_removed_at_every_load_and_recorded():
    """sweeping_garage's instance 301 has a sand and a dust particle on the floor slab outside its garage door, which
    is locked: the room map places them in the garden, which the activity does not load."""
    sim = native()
    try:
        for _ in range(2):  # a reset reloads the instance, and the same two go
            sim.reset("sweeping_garage", 0)
            assert sim.native("measures") == {
                "q_score": 0., "particles_removed_out_of_loaded_rooms": ["sandParticle11", "dustParticle6"]}
    finally:
        sim.close()


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_steps_and_records(task):
    sim = native()
    try:
        sim.configure_spectators()
        sim.reset(task, 0)
        assert not sim.done and sim.max_steps > 0
        start = sim.native("render_count")
        sim.step(sim.state()[0])
        assert sim.steps == 1 and np.isfinite(sim.state()[0]).all() and sim.reward in (0., 1.)
        assert 0. <= sim.native("measures")["q_score"] <= 1.
        sim.step(sim.state()[0])
        # Recording renders every step; the reset settled the spectators at their new pose.
        assert sim.native("render_count") == start + 2
        for view in sim.spectators:
            assert sim.render_spectator(view.name).shape == (720, 960, 3)
            assert sim.spectator_intrinsics(view.name)[2:] == (479.5, 359.5)
        sim.render("head_camera")
        assert sim.native("render_count") == start + 3  # one render serves every view
    finally:
        sim.close()
