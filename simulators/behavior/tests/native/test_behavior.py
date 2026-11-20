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


def test_the_episode_starts_as_the_evaluator_starts_it(tmp_path):
    """omnigibson/eval: one seed_everything(0) before the environment is built, the robot at the instance's
    presampled pose 0 once env.reset has placed it again, and the episode truncated on the step after the Timeout's
    limit, where upstream's Timeout ends it (it checks the step count before counting the step)."""
    import json
    import subprocess
    path = tmp_path / "start.json"
    run = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "evaluator_start.py"), str(path)],
                         capture_output=True, text=True, timeout=2400)
    assert path.is_file(), run.stderr[-4000:]
    start = json.loads(path.read_text())
    assert start["events"] == [["seed", 0], ["environment"]]
    (position, orientation, frame), = start["teleports"]  # env.reset places the robot once, at the presampled pose
    assert frame == "scene"
    np.testing.assert_allclose(position + orientation, start["sampled"], atol=1e-6)
    # env.reset then runs one physics step, which lifts the base off the floor by a few millimetres.
    np.testing.assert_allclose(start["robot"][:2], start["sampled"][:2], atol=1e-4)
    assert abs(start["robot"][2] - start["sampled"][2]) < .01
    assert abs(np.dot(start["robot"][3:], start["sampled"][3:])) > 1 - 1e-6  # the same rotation, up to sign
    assert start["max_steps"] == start["timeout"] + 1


def test_particles_out_of_the_loaded_rooms_are_removed_at_every_load_and_recorded():
    """sweeping_garage's instance 301 has a sand and a dust particle of its goal on the garage floor just outside the
    locked garage door, which the room map places in the garden, a room the activity does not load."""
    sim = native()
    try:
        for _ in range(2):  # a reset loads the instance again, and the same two go
            sim.reset("sweeping_garage", 0)
            measures = sim.native("measures")
            assert sorted(measures.pop("particles_removed_out_of_loaded_rooms")) == ["dustParticle6", "sandParticle11"]
            assert measures == {"q_score": 0.}
    finally:
        sim.close()


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
