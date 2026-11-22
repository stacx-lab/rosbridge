"""Pinned HandsOn acceptance in the native environment.

Isaac hosts one episode per process and its shutdown ends the process, so every check
drives HandsOnSim through a worker subprocess. Each task needs its own Isaac boot
(minutes), so the every-task sweep runs only with ROSBRIDGE_ALL_TASKS=1.
"""
import os
import sys

import numpy as np
import pytest
import yaml

from rosbridge.catalog import tasks
from rosbridge.catalog.robots import setup
from rosbridge.core.worker import WorkerBackend

# The default task: both arms move freely.
TASK, STEPS = "t24_place_ball_in_bowl", 150
ALL = {t["native_task"]: t["id"] for t in tasks(name="handson").values()}
# A bimanual task and one whose left arm and hand upstream holds.
SELECTED = list(ALL) if os.environ.get("ROSBRIDGE_ALL_TASKS") else ["t12_load_and_start_microwave",
                                                                    "eval_gen01_cover_lunchbox"]
SIDES = (("right", 0), ("left", 27))
CURL = [4, 6, 7, 8, 10, 11]  # index and middle finger flexion, in the hand's finger-major order
IDLE = {"right_arm": None, "left_arm": None}  # arms not yet commanded


class Native(WorkerBackend):
    simulator = "handson"
    close_timeout_s = 120.


def native(**options):
    return Native(python=sys.executable, env=dict(os.environ), **options)


def lifted(sim, side, height):
    """The measured palm pose in its arm base, height metres higher in the world."""
    palm = np.asarray(sim.native("tip_pose", f"{side}_arm"), dtype=float)
    rotation, _ = sim.native("link_pose", side, f"link0_{side}")
    palm[:3] += np.asarray(rotation).T @ [0., 0., height]
    return palm


def angle(a, b):
    """The rotation between two unit quaternions, in radians."""
    return 2 * np.arccos(min(1., abs(float(np.dot(a, b)))))


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
    assert [group.tip for group in sim.layout.groups] == ["palm", None, "palm", None]
    assert [(d.name, d.kind) for d in sim.devices] == [
        ("right_arm", "joint"), ("right_hand", "joint"), ("left_arm", "joint"), ("left_hand", "joint"),
        ("head_camera", "rgb"), ("left_camera", "rgb"), ("right_camera", "rgb")]


def test_palms_track_pose_targets_and_hands_joint_targets(sim):
    start = {side: np.asarray(sim.native("tip_pose", f"{side}_arm")) for side, _ in SIDES}
    action = sim.state()[0]
    for _ in range(15):
        sim.step(action, IDLE)  # upstream's idle: the arms hold
    for side, _ in SIDES:
        np.testing.assert_allclose(sim.native("tip_pose", f"{side}_arm")[:3], start[side][:3], atol=2e-3)
    # Upstream's IK lifts both palms clear of the table; the hands curl two fingers.
    poses = {f"{side}_arm": lifted(sim, side, .05) for side, _ in SIDES}
    for _, offset in SIDES:
        action[[offset + 7 + i for i in CURL]] += .5
    for _ in range(45):
        sim.step(action, poses)
    assert sim.steps == 60 and sim.time_s == pytest.approx(2.)
    measured = sim.state()[0]
    for side, offset in SIDES:
        palm, target = np.asarray(sim.native("tip_pose", f"{side}_arm")), poses[f"{side}_arm"]
        np.testing.assert_allclose(palm[:3], target[:3], atol=5e-3)
        assert angle(palm[3:], target[3:]) < np.radians(2)
        hand = [offset + 7 + i for i in CURL]
        np.testing.assert_allclose(measured[hand], action[hand], atol=.1)


def test_cameras_render_the_contract_and_leave_physics_unchanged(sim):
    before = sim.state()
    for camera, size in (("head_camera", (854, 480)), ("left_camera", (848, 480)), ("right_camera", (848, 480))):
        image = sim.render(camera)
        assert image.shape == (size[1], size[0], 3) and image.dtype == np.uint8 and image.std() > 1
        assert sim.intrinsics(camera)[:2] == size
    # The head camera keeps its calibrated, off-center principal point, in ROS pixel centers.
    assert sim.intrinsics("head_camera")[4:] == pytest.approx((431.267, 239.419), abs=1e-3)
    assert set(sim.distortion) == {"head_camera"} and len(sim.distortion["head_camera"]) == 5
    for view in sim.spectators:
        assert sim.render_spectators()[view.name].shape == (720, 960, 3)
        assert sim.spectator_intrinsics(view.name)[2:] == (479.5, 359.5)
    after = sim.state()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def lined_up(image, view, spectator_k, rays, size=96):
    """Shifts (x, y, dx, dy) in pixels of a camera image's textured patches against its spectator view, an
    undistorted pinhole at the same pose, resampled where rays (H x W x 3, the profile's) meet it."""
    import cv2
    fs, _, cx, cy = spectator_k
    gray = lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)  # noqa: E731
    expected = cv2.remap(gray(view), (fs * rays[..., 0] + cx).astype(np.float32),
                         (fs * rays[..., 1] + cy).astype(np.float32), cv2.INTER_LINEAR)
    measured, window, shifts = gray(image), cv2.createHanningWindow((size, size), cv2.CV_32F), []
    for y in range(0, measured.shape[0] - size + 1, size // 2):
        for x in range(0, measured.shape[1] - size + 1, size // 2):
            patch = measured[y:y + size, x:x + size].copy()
            if patch.std() > 4:
                (dx, dy), response = cv2.phaseCorrelate(expected[y:y + size, x:x + size].copy(), patch, window)
                if response > .2:
                    shifts.append((x + size / 2, y + size / 2, dx, dy))
    return np.array(shifts)


def test_cameras_render_the_geometry_their_profiles_state(tmp_path, sim):
    """Each image lines up with its undistorted spectator through the rays of the camera's profile: the pinhole's
    for the wrists, and for the head camera those the world node serves through its lens (rosbridge.core.lens).
    Without the head camera's lens, its edges miss by pixels."""
    from rosbridge.catalog.cameras import BROWN_CONRADY
    from rosbridge.core.lens import LensGeometry
    setup(TASK, tmp_path, seed=0)  # the workspace of the episode the fixture loaded
    for camera, view in (("head_camera", "third_person"), ("left_camera", "left_wrist"),
                         ("right_camera", "right_wrist")):
        color = yaml.safe_load((tmp_path / ".kinex/devices" / camera / "profile.yaml").read_text())["color"]
        width, height = color["width"], color["height"]
        fx, fy, cx, cy = (color["projection"][key] for key in ("fx", "fy", "cx", "cy"))
        u, v = (grid.ravel() for grid in np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float)))
        image, spectator = sim.render(camera), sim.render_spectators()[view]
        pinhole = np.column_stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)])
        if color["distortion"]["model"] == "none":
            through = pinhole
        else:
            lens = LensGeometry(f"{camera}/base", width, height, fx, fy, cx, cy,
                                [color["distortion"][key] for key in BROWN_CONRADY])
            through = lens.rays(np.column_stack([u, v]))
        shifts = lined_up(image, spectator, sim.spectator_intrinsics(view), through.reshape(height, width, 3))
        size = np.hypot(shifts[:, 2], shifts[:, 3])
        assert len(shifts) > 50 and np.median(size) < .3, camera
        assert abs(shifts[:, 2].mean()) < .15 and abs(shifts[:, 3].mean()) < .15, camera  # principal point
        if camera == "head_camera":
            edges = np.abs(shifts[:, 0] - color["projection"]["cx"]) > 300
            assert np.median(size[edges]) < .35
            without = lined_up(image, spectator, sim.spectator_intrinsics(view), pinhole.reshape(height, width, 3))
            assert np.median(np.hypot(*without[edges][:, 2:].T)) > 1.


def test_episode_ends_at_its_step_limit_and_serves_one_episode(sim):
    hold = sim.state()[0]
    while not sim.done:
        sim.step(hold, IDLE)
    assert sim.steps == STEPS and sim.time_s == pytest.approx(STEPS / 30)
    with pytest.raises(RuntimeError, match="done"):
        sim.step(hold, IDLE)
    with pytest.raises(ValueError, match="one episode"):
        sim.reset(TASK, 0)
    assert sim.native("measures") == {"native_seed": 0}


@pytest.mark.parametrize("task", SELECTED)
def test_catalog_task_loads_steps_and_commands_the_arms_upstream_drives(task, tmp_path):
    sim = native()
    try:
        sim.reset(task, 0)
        assert sim.description["instruction"]
        held = sim.native("held_arms")
        setup(ALL[task], tmp_path, seed=0)
        operations = {name: yaml.safe_load((tmp_path / ".kinex/devices" / name / "profile.yaml").read_text())
                      ["operations"] for name in ("right_arm", "right_hand", "left_arm", "left_hand")}
        for side, _ in SIDES:
            assert ("pose_target" in operations[f"{side}_arm"]) == (side not in held), side
            assert ("joint_target" in operations[f"{side}_hand"]) == (side not in held), side
        assert not sim.done and sim.max_steps > 0
        sim.step(sim.state()[0], IDLE)
        assert sim.steps == 1 and np.isfinite(sim.state()[0]).all() and sim.reward in (0., 1.)
    finally:
        sim.close()
