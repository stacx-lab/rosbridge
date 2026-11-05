"""BEHAVIOR with the G1 and Sharpa hands in the native environment.

Isaac hosts one app per process and OmniGibson's shutdown ends it, so every check drives
G1SharpaSim through a worker subprocess. GR00T's lower-body policy must keep the G1 upright
while it stands, walks, turns and moves its arms and hands; a fall fails the checks.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.core.worker import WorkerBackend
from tests.kinex_geometry import Rig, compose, requires_kinex

FALL_M = .2  # a drop of the pelvis, the robot's root, below its standing height


class Native(WorkerBackend):
    simulator = "behavior-g1"
    close_timeout_s = 120.


@pytest.fixture(scope="module")
def sim():
    sim = Native(python=sys.executable, env=dict(os.environ))
    sim.configure_spectators()
    sim.reset("turning_on_radio", 0)
    yield sim
    sim.close()


def channels(sim, device):
    return list(next(g for g in sim.layout.groups if g.name == device).indices)


def run(sim, command, seconds):
    """Step for seconds; returns the largest drop of the pelvis below its standing height."""
    standing, low = sim.native("link_pose", "pelvis")[2, 3], np.inf
    for _ in range(round(seconds / sim.period)):
        sim.step(command)
        low = min(low, sim.native("link_pose", "pelvis")[2, 3])
    return standing - low


def test_layout_period_and_head_camera(sim):
    assert sim.period == pytest.approx(.02)
    assert sim.max_steps == 5374  # the evaluator's 3224 steps at 30 Hz, in time
    assert [d.name for d in sim.devices] == ["base", "pelvis", "torso", "left_arm", "right_arm", "left_hand",
                                             "right_hand", "head_camera"]
    assert [len(channels(sim, d)) for d in ("base", "pelvis", "torso", "left_arm", "left_hand")] == [3, 1, 3, 7, 22]
    width, height, fx, fy, cx, cy = sim.intrinsics("head_camera")
    assert (width, height, fx, cx) == (720, 720, pytest.approx(306.), 359.5)
    image, depth = sim.render("head_camera"), sim.render_depth("head_camera")
    assert image.shape == (720, 720, 3) and image.std() > 1 and (depth > 0).mean() > .9
    assert sim.native("render_settings")["/rtx/post/aa/op"] == 4  # on-demand captures keep DLAA
    for view in sim.spectators:
        assert sim.render_spectator(view.name).shape == (720, 960, 3)


def test_stands_walks_and_turns_on_gr00t_without_falling(sim):
    hold = sim.state()[0]
    hold[channels(sim, "base")] = 0.
    assert run(sim, hold, 4.) < FALL_M
    walk = hold.copy()
    walk[channels(sim, "base")[0]] = .3
    assert run(sim, walk, 5.) < FALL_M
    position, velocity = sim.state()
    assert position[0] > .6 and velocity[0] == pytest.approx(.3, abs=.15)
    turn = hold.copy()
    turn[channels(sim, "base")[2]] = .4
    yaw = position[2]
    assert run(sim, turn, 3.) < FALL_M
    assert sim.state()[0][2] - yaw > .5
    assert run(sim, hold, 2.) < FALL_M
    assert np.abs(sim.state()[1][:3]).max() < .1


def test_squats_on_the_pelvis_height_command_and_stands_back_up(sim):
    """GR00T takes the pelvis height as a command: it squats to 0.20 without falling (the pelvis pitches forward while
    the torso stays level) and stands back up in one command; the pelvis device measures the height."""
    hold = sim.state()[0]
    hold[channels(sim, "base")] = 0.
    pelvis = channels(sim, "pelvis")[0]
    hold[pelvis] = .74
    assert run(sim, hold, 2.) < FALL_M
    assert sim.state()[0][pelvis] == pytest.approx(.75, abs=.04)
    squat = hold.copy()
    squat[pelvis] = .20
    run(sim, squat, 6.)
    assert sim.state()[0][pelvis] < .35
    run(sim, hold, 6.)
    assert sim.state()[0][pelvis] > .66
    assert sim.native("link_pose", "pelvis")[2, 2] > .9  # upright again


def test_waist_follows_pitch_and_roll_targets_while_balancing(sim):
    """GR00T realizes about half the torso pitch and roll it is given; the embodiment doubles them so the measured
    waist follows the device's targets."""
    target = sim.state()[0]
    target[channels(sim, "base")] = 0.
    torso = channels(sim, "torso")
    for index, angle in ((2, .3), (1, .3)):
        target[torso] = 0.
        target[torso[index]] = angle
        assert run(sim, target, 3.) < FALL_M
        assert sim.state()[0][torso[index]] == pytest.approx(angle, abs=.07)
    target[torso] = 0.
    assert run(sim, target, 2.) < FALL_M


def test_arms_and_hands_track_joint_targets_while_balancing(sim):
    target = sim.state()[0]
    target[channels(sim, "base")] = 0.
    arm, hand = channels(sim, "left_arm"), channels(sim, "right_hand")
    target[arm[3]] += .4
    lower = np.array(next(g for g in sim.layout.groups if g.name == "right_hand").lower)
    upper = np.array(next(g for g in sim.layout.groups if g.name == "right_hand").upper)
    target[hand] = lower + .6 * (upper - lower)
    assert run(sim, target, 3.) < FALL_M
    measured = sim.state()[0]
    assert measured[arm[3]] == pytest.approx(target[arm[3]], abs=.05)
    np.testing.assert_allclose(measured[hand], target[hand], atol=.1)


def rt(matrix):
    return matrix[:3, :3].tolist(), matrix[:3, 3].tolist()


@requires_kinex
def test_rig_matches_native_chains_hands_and_head_camera(tmp_path, sim):
    rig = Rig("behavior-g1/behavior-g1-sharpa", tmp_path)
    q = sim.state()[0]
    base = np.linalg.inv(sim.native("link_pose", "pelvis"))

    def measured(link):
        return rt(base @ sim.native("link_pose", link))

    def close(expected, actual, atol=1e-3):
        np.testing.assert_allclose(expected[0], actual[0], atol=atol)
        np.testing.assert_allclose(expected[1], actual[1], atol=atol)
    top = compose(rig.mount("torso"), rig.link("torso", q[channels(sim, "torso")], "top"))
    close(top, measured("torso_link"))
    close(compose(top, rig.mount("head_camera")), rt(base @ sim.camera_pose("head_camera")))
    for side in ("left", "right"):
        flange = compose(top, compose(rig.mount(f"{side}_arm"), rig.link(f"{side}_arm", q[channels(sim, f"{side}_arm")])))
        close(flange, measured(f"{side}_wrist_yaw_link"))
        hand = compose(flange, rig.mount(f"{side}_hand"))
        for link in (f"{side}_index_DP", f"{side}_thumb_DP"):
            fingers = q[channels(sim, f"{side}_hand")]
            close(compose(hand, rig.link(f"{side}_hand", fingers, link)), measured(link), atol=2e-3)


def test_status_and_episode_counts_reach_the_world_through_the_worker(sim):
    """The native worker's snapshot carries the status texts the world publishes, and the record's counts."""
    assert sim.status == {"left_hand/attachment": "none", "right_hand/attachment": "none", "base/fall": "upright"}
    extra = sim.outcome_extra()
    assert (extra["attachments"], extra["falls"], extra["fall_time_s"]) == (0, 0, None)
