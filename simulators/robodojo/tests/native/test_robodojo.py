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

    def joint_step(self, group, pose):  # the backend's own IK for a step of a move goal
        return self._call("joint_step", group, pose)


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
        assert sim.render_spectators()[view.name].shape == (720, 960, 3)
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


def flange(urdf, q):
    """The catalog arm URDF's tip transform in its base at joints q."""
    import xml.etree.ElementTree as ET
    import transforms3d as t3d
    pose = np.eye(4)
    for joint, angle in zip(ET.parse(urdf).getroot().iter("joint"), q):
        origin, axis = joint.find("origin"), joint.find("axis")
        step = np.eye(4)
        step[:3, :3] = t3d.euler.euler2mat(*map(float, origin.get("rpy").split()), "sxyz")
        step[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        turn = np.eye(4)
        turn[:3, :3] = t3d.axangles.axangle2mat([float(v) for v in axis.get("xyz").split()], angle)
        pose = pose @ step @ turn
    return pose


def test_a_goal_step_becomes_joints_through_curobo_ik_or_has_none():
    """A step of an arm's move goal through cuRobo's IK: the measured pose keeps the arm still, a straight lift
    arrives step by step, and a pose beyond reach, or one that needs a change of posture, has no joints."""
    import transforms3d as t3d
    sim = native(max_steps=200)
    try:
        sim.reset("stack_bowls", 0)
        arm = next(group for group in sim.layout.groups if group.name == "left_arm")
        assert (arm.tip, arm.joint_targets) == ("flange", True)
        assert sim.native("__getattribute__", "_limits")["left"][2] == pytest.approx(5. * .04)  # the URDF's velocity
        joints = list(arm.indices)
        held = sim.state()[0]
        tip = np.asarray(sim.tip_pose("left_arm"))
        # pose_state is the catalog URDF's flange at the measured joints.
        expected = flange(ROOT / "catalog/robots/robodojo/robodojo-arx-x5/devices/left_arm/arm.urdf", held[joints])
        x, y, z, w = tip[3:]
        assert tip[:3] == pytest.approx(expected[:3, 3], abs=1e-4)
        assert t3d.quaternions.quat2mat([w, x, y, z]) == pytest.approx(expected[:3, :3], abs=1e-4)

        def follow(poses):
            """Step through poses, each solved from the measured joints; the largest joint travel asked of a step."""
            asked, target = 0., sim.state()[0]
            for pose in poses:
                row = np.asarray(sim.joint_step("left_arm", pose))
                asked = max(asked, np.abs(row - sim.state()[0][joints]).max())
                target[joints] = row
                sim.step(target)
            return asked

        def error(pose):
            now = np.asarray(sim.tip_pose("left_arm"))
            return np.linalg.norm(now[:3] - pose[:3]), 2 * np.arccos(min(1., abs(float(np.dot(now[3:], pose[3:])))))

        assert follow([tip] * 10) < .05  # the measured pose holds the arm within upstream's IK tolerances
        moved, turned = error(tip)
        assert moved < 1.5e-3 and turned < .03
        goal = tip.copy()
        goal[2] += .05
        lift = [np.r_[tip[:2], tip[2] + .05 * s, tip[3:]] for s in np.linspace(0, 1, 26)[1:]]
        assert follow(lift + [goal] * 10) < .2  # a straight 5 cm lift over one second, then held
        moved, turned = error(goal)
        assert moved < 2e-3 and turned < .03
        far = goal.copy()
        far[0] += 2.
        with pytest.raises(ValueError, match="cuRobo's IK has no solution"):
            sim.joint_step("left_arm", far)
        # A quarter turn about the flange's approach axis at once: joint6 would turn 1.57 rad in one step.
        quarter = t3d.quaternions.qmult([goal[6], *goal[3:6]], [np.sqrt(.5), np.sqrt(.5), 0., 0.])
        with pytest.raises(ValueError, match="the nearest IK solution moves .* in one step, a change of posture"):
            sim.joint_step("left_arm", np.r_[goal[:3], quarter[1:], quarter[0]])
    finally:
        sim.close()


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_describes_itself_and_steps(task):
    sim = native()
    try:
        sim.reset(task, 0)
        assert not sim.done and sim.max_steps > 0
        # Upstream's instruction pair for the layout, as its policy receives it.
        assert sim.description["instruction"] and "## Scoring" in sim.description["detailed_instruction"]
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
