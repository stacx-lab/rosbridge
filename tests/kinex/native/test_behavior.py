"""The behavior/behavior-r1pro rig agrees with OmniGibson through Kinex's URDF kinematics, in behavior's native environment.

The simulator runs in a worker subprocess, as in its own native suite.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.core.worker import WorkerBackend
from tests.kinex.geometry import Rig, compose, requires_kinex

pytestmark = requires_kinex


class Native(WorkerBackend):
    simulator = "behavior"
    close_timeout_s = 120.


@pytest.fixture(scope="module")
def sim():
    sim = Native(python=sys.executable, env=dict(os.environ), max_steps=90)
    sim.reset("turning_on_radio", 0)
    yield sim
    sim.close()


def channels(sim, device):
    return list(next(g for g in sim.layout.groups if g.name == device).indices)


def rt(matrix):
    return matrix[:3, :3].tolist(), matrix[:3, 3].tolist()


def assert_pose(expected, measured, atol=1e-4):
    np.testing.assert_allclose(expected[0], measured[0], atol=atol)
    np.testing.assert_allclose(expected[1], measured[1], atol=atol)


def test_rig_matches_native_chains_grippers_and_cameras(tmp_path, sim):
    rig = Rig("behavior", tmp_path)
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
