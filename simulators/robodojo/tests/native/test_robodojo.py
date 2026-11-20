"""Pinned RoboDojo acceptance in the native environment.

Isaac hosts one app per process and its shutdown ends the process, so every check
drives RoboDojoSim through a worker subprocess. Each task needs its own Isaac boot
(minutes), so the every-task sweep runs only with ROSBRIDGE_ALL_TASKS=1.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.catalog import simulator, tasks
from rosbridge.catalog.robots import catalog_profiles
from rosbridge.core.worker import WorkerBackend

ROOT, _, MANIFEST = simulator("robodojo")
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
    assert {d.name: d.kind for d in sim.devices} == MANIFEST["devices"]
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


def test_a_gripper_stop_keeps_a_held_object_gripped(tmp_path):
    """Kinex's grip contract through the world node's command buffer: a gripper stop keeps the closed target, so a
    toothbrush held between the pads stays there. Holding the measured opening instead dropped it 14 cm in 0.4 s."""
    import json
    import subprocess
    path = tmp_path / "grip.json"
    run = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "grip_stop.py"), str(path)],
                         capture_output=True, text=True, timeout=1800)
    assert path.is_file(), run.stderr[-4000:]
    grip = json.loads(path.read_text())
    assert .2 < grip["held"] < .9  # stalled on the toothbrush, short of closed (about 0.185) and open (1)
    assert grip["target"] == 0.  # the closed command, not the measured opening
    assert max(grip["offset"]) < .01 and abs(grip["drop"]) < .005


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


def test_the_depth_configuration_renders_z_depth_registered_to_each_image():
    """Upstream's distance_to_image_plane: metres along the optical axis, within the profile's range, so the
    head camera sees the table flat at the arms' mount height."""
    sim = native(configuration="depth")
    try:
        sim.reset("pack_objects_into_box", 0)
        assert {d.name: d.kind for d in sim.devices} == MANIFEST["devices"] | MANIFEST["configurations"]["depth"]["devices"]
        profiles = catalog_profiles(ROOT, "robodojo/robodojo-arx-x5", "depth")
        for camera in sim.cameras:
            image, depth = sim.render(camera), sim.render_depth(camera)
            assert image.shape == (480, 640, 3) and depth.shape == (480, 640) and depth.dtype == np.float32
            span = profiles[camera]["depth"]["valid_range_m"]
            assert ((depth >= span["min"]) & (depth <= span["max"])).mean() > .99
        width, height, fx, fy, cx, cy = sim.intrinsics("head_camera")
        position, (w, x, y, z) = sim.native("camera_pose", "head_camera")
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        v, u = np.mgrid[380:420, 128:512]  # the bare table between the arms, up to 33 degrees off axis
        depth = sim.render_depth("head_camera")[380:420, 128:512]
        points = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1) @ rotation.T + position
        table = profiles["left_arm"]["frames"]["base"]["translation_m"][2]
        assert np.median(np.abs(points[..., 2] - table), axis=0).max() < .003
    finally:
        sim.close()
