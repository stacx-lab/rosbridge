"""BEHAVIOR's R1Pro with Sharpa hands in the native environment.

Isaac hosts one app per process and OmniGibson's shutdown ends it, so every check drives
R1ProSharpaSim through a worker subprocess. The body, base and cameras must be the R1Pro's; the
hands replace the grippers on the same mount and track joint targets.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.core.worker import WorkerBackend
from rosbridge_behavior import dex_model
from tests.kinex_geometry import Rig, compose, requires_kinex

SIDES = ("left", "right")
# Flexion of the index, middle and ring fingers (MCP, PIP, DIP) in the hand's joint order.
CURL = [i for finger in (5, 9, 13) for i in (finger, finger + 2, finger + 3)]
# Hands held up in front of the body to work, solved with Kinex IK on the rig at the evaluator's
# torso posture: wrists 0.40 m ahead of the base, 0.18 m to each side and 0.97 m high, fingers
# forward pitched 35 degrees down and palms facing each other (`table`); and wrists at chest
# height, 0.35 m ahead, 0.20 m out and 1.03 m high, fingers pitched 15 degrees down (`chest`).
WORKING = {
    "table": {"left": [.0497, -.1089, -.1738, -1.7597, -.0102, .7154, .1979],
              "right": [.0523, .125, .1572, -1.7597, .0402, .7149, -.1719]},
    "chest": {"left": [.1501, -.0384, -.1994, -2.0701, -.029, .5722, .1858],
              "right": [.1533, .0553, .1854, -2.0701, .0505, .5714, -.1614]},
}


class Native(WorkerBackend):
    simulator = "behavior-dex"
    close_timeout_s = 120.


@pytest.fixture(scope="module")
def sim():
    sim = Native(python=sys.executable, env=dict(os.environ), max_steps=600)
    sim.reset("turning_on_radio", 0)
    yield sim
    sim.close()


def channels(sim, device):
    return list(next(g for g in sim.layout.groups if g.name == device).indices)


def in_link7(sim, side, matrix):
    return np.linalg.inv(sim.native("link_pose", f"{side}_arm_link7")) @ matrix


def test_r1pro_body_and_cameras_with_hands_in_the_evaluator_posture(sim):
    assert sim.period == pytest.approx(1 / 30)
    assert [d.name for d in sim.devices] == ["base", "torso", "left_arm", "left_hand", "right_arm",
                                             "right_hand", "head_camera", "left_camera", "right_camera"]
    assert [len(channels(sim, d)) for d in ("base", "torso", "left_arm", "left_hand")] == [3, 4, 7, 22]
    assert sim.intrinsics("head_camera") == pytest.approx((720, 720, 306., 306., 359.5, 359.5))
    for camera in ("left_camera", "right_camera"):
        width, height, fx, fy, cx, cy = sim.intrinsics(camera)
        assert (width, height, cx, cy) == (480, 480, 239.5, 239.5) and fx == fy
    for camera in sim.cameras:
        image, depth = sim.render(camera), sim.render_depth(camera)
        assert image.std() > 1 and (depth > 0).mean() > .9
    settings = sim.native("render_settings")
    assert settings["/rtx/post/aa/op"] != 4  # the R1Pro's evaluator rendering
    q = sim.state()[0]
    np.testing.assert_allclose(q[channels(sim, "torso")], [1.025, -1.45, -.47, 0.], atol=.02)
    for side in SIDES:
        # Open and flat, within the friction deadband of Sharpa's drives (up to 0.022 rad).
        np.testing.assert_allclose(q[channels(sim, f"{side}_hand")], 0., atol=.03)


def test_wrist_cameras_see_past_the_hands_from_their_backs(sim):
    sim.reset("turning_on_radio", 0)
    for side in SIDES:
        flange = in_link7(sim, side, sim.native("link_pose", f"{side}_hand_flange"))
        measured = in_link7(sim, side, sim.camera_pose(f"{side}_camera"))
        optical = np.diag([1., -1., -1., 1.])  # the ROS optical frame of the USD camera
        np.testing.assert_allclose(measured, flange @ dex_model.camera_pose() @ optical, atol=1e-4)
        # The open hand, within 0.3 m, fills only the bottom of the view; the rest is the scene.
        depth = sim.render_depth(f"{side}_camera")
        valid = depth[np.isfinite(depth) & (depth > 0)]
        assert np.median(valid) > .3 and (valid < .3).mean() < .25


def test_palms_face_the_midline_and_thumbs_lead_forward(sim):
    for side, inward in (("left", -1), ("right", 1)):
        palm = in_link7(sim, side, sim.native("link_pose", f"{side}_hand_C_MC"))
        thumb = in_link7(sim, side, sim.native("link_pose", f"{side}_thumb_DP"))
        assert palm[:3, 0] @ [0, inward, 0] > .99  # the palm normal, in arm_link7's frame
        assert palm[:3, 2] @ [0, 0, -1] > .99  # fingers continue the gripper's approach
        assert thumb[0, 3] > palm[0, 3] + .02  # the thumb leads, toward arm_link7's +x


def test_hands_and_arms_track_targets_while_the_base_stays(sim):
    sim.reset("turning_on_radio", 0)
    target = sim.state()[0]
    target[channels(sim, "base")] = 0.
    left_arm = channels(sim, "left_arm")
    target[left_arm[3]] -= .2
    for side in SIDES:
        hand = channels(sim, f"{side}_hand")
        target[[hand[i] for i in CURL]] += .6
    start = sim.native("link_pose", "base_link")
    for _ in range(60):
        sim.step(target)
    q = sim.state()[0]
    assert q[left_arm[3]] == pytest.approx(target[left_arm[3]], abs=.02)
    for side in SIDES:
        hand = channels(sim, f"{side}_hand")
        np.testing.assert_allclose(q[hand], target[hand], atol=.1)
    moved = sim.native("link_pose", "base_link")[:3, 3] - start[:3, 3]
    assert np.linalg.norm(moved) < .02 and np.isfinite(q).all()


@pytest.mark.parametrize("posture", list(WORKING))
def test_head_camera_sees_both_hands_held_up_to_work(sim, posture):
    sim.reset("turning_on_radio", 0)
    target = sim.state()[0]
    target[channels(sim, "base")] = 0.
    for side in SIDES:
        target[channels(sim, f"{side}_arm")] = WORKING[posture][side]
    for _ in range(120):
        sim.step(target)
    width, height, fx, fy, cx, cy = sim.intrinsics("head_camera")
    camera = np.linalg.inv(sim.camera_pose("head_camera"))
    depth = sim.render_depth("head_camera")
    for side in SIDES:
        np.testing.assert_allclose(sim.state()[0][channels(sim, f"{side}_arm")], WORKING[posture][side], atol=.02)
        for link in ("hand_C_MC", "index_DP", "thumb_DP", "pinky_DP"):
            x, y, z, _ = camera @ sim.native("link_pose", f"{side}_{link}")[:, 3]
            u, v = fx * x / z + cx, fy * y / z + cy
            assert z > 0 and 0 <= u < width and 0 <= v < height, f"{side}_{link} is outside the head view"
            # What the head camera sees there is the hand itself, not something in front of it.
            assert abs(depth[int(v), int(u)] - z) < .06, f"{side}_{link} is hidden in the head view"


def rt(matrix):
    return matrix[:3, :3].tolist(), matrix[:3, 3].tolist()


@requires_kinex
def test_rig_matches_native_chains_hands_and_cameras(tmp_path, sim):
    rig = Rig("behavior-dex/behavior-r1pro-sharpa", tmp_path)
    q = sim.state()[0]
    base = np.linalg.inv(sim.native("link_pose", "base_link"))

    def measured(link):
        return rt(base @ sim.native("link_pose", link))

    def close(expected, actual, atol=1e-4):
        np.testing.assert_allclose(expected[0], actual[0], atol=atol)
        np.testing.assert_allclose(expected[1], actual[1], atol=atol)
    top = compose(rig.mount("torso"), rig.link("torso", q[channels(sim, "torso")], "top"))
    close(top, measured("torso_link4"))
    close(compose(top, rig.mount("head_camera")), rt(base @ sim.camera_pose("head_camera")))
    for side in SIDES:
        flange = compose(top, compose(rig.mount(f"{side}_arm"), rig.link(f"{side}_arm", q[channels(sim, f"{side}_arm")])))
        close(flange, measured(f"{side}_arm_link7"))
        close(compose(flange, rig.mount(f"{side}_camera")), rt(base @ sim.camera_pose(f"{side}_camera")))
        hand = compose(flange, rig.mount(f"{side}_hand"))
        fingers = q[channels(sim, f"{side}_hand")]
        for link in (f"{side}_hand_C_MC", f"{side}_index_DP", f"{side}_thumb_DP"):
            close(compose(hand, rig.link(f"{side}_hand", fingers, link)), measured(link), atol=2e-3)


def test_doors_start_at_a_small_effort_and_locked_doors_stay_locked(sim):
    """door_friction: in turning_on_radio's house, a room door between two loaded rooms (9.9 kg) needs 14 N m under
    OmniGibson's friction; here it opens at 3 N m but holds against 1 N m. The corridor's door to the unloaded garage
    keeps its lock."""
    sim.reset("turning_on_radio", 0)
    hold = sim.state()[0]
    hold[channels(sim, "base")] = 0.

    def run(seconds):
        for _ in range(round(seconds / sim.period)):
            sim.step(hold)

    sim.native("drive_object_joint", "door_vudhlc_3", "j_leaf", 0., effort=1.)
    run(1.)
    assert sim.native("object_joint", "door_vudhlc_3", "j_leaf") < .01
    sim.native("drive_object_joint", "door_vudhlc_3", "j_leaf", 0., effort=3.)
    run(1.)
    assert sim.native("object_joint", "door_vudhlc_3", "j_leaf") > .1
    sim.native("drive_object_joint", "door_vudhlc_3", "j_leaf", 0.)
    sim.native("drive_object_joint", "door_bexenl_0", "j_leaf", effort=200.)
    run(.5)
    sim.native("drive_object_joint", "door_bexenl_0", "j_leaf")
    assert sim.native("object_joint", "door_bexenl_0", "j_leaf") <= .0011


def test_reads_render_once_after_a_step(sim):
    start = sim.native("render_count")
    hold = sim.state()[0]
    hold[channels(sim, "base")] = 0.
    sim.step(hold)
    sim.render("left_camera")
    first = sim.native("render_count")
    sim.step(hold)
    sim.render("left_camera"), sim.render_depth("right_camera"), sim.render("head_camera")
    assert sim.native("render_count") == first + 1 and first > start
