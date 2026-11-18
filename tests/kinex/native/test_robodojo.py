"""The robodojo/robodojo-arx-x5 rig agrees with Isaac Sim through Kinex's URDF kinematics, in robodojo's native environment.

The simulator runs in a worker subprocess, as in its own native suite.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.core.worker import WorkerBackend
from tests.kinex.geometry import Rig, compose, requires_kinex, rt

pytestmark = requires_kinex


class Native(WorkerBackend):
    simulator = "robodojo"
    close_timeout_s = 120.


@pytest.fixture(scope="module")
def sim():
    sim = Native(python=sys.executable, env=dict(os.environ), max_steps=60)
    sim.configure_spectators()
    sim.reset("pack_objects_into_box", 0)
    yield sim
    sim.close()


def test_rig_matches_native_arms_fingers_and_cameras(tmp_path, sim):
    rig = Rig("robodojo", tmp_path)
    target = sim.state()[0]
    target[[0, 7]] += .2
    target[[1, 8]] += .3
    target[[6, 13]] = .6
    for _ in range(2):
        for _ in range(15):
            sim.step(target)
        q = sim.state()[0]
        for side, offset in (("left", 0), ("right", 7)):
            flange = rig.flange(f"{side}_arm", q[offset:offset + 6])
            np.testing.assert_allclose(flange[0], sim.native("link_pose", side, "link6")[0], atol=1e-4)
            np.testing.assert_allclose(flange[1], sim.native("link_pose", side, "link6")[1], atol=1e-4)
            grip = compose(flange, rig.mount(f"{side}_gripper"))
            for finger, link, atol in (("finger_a", "link7", 1e-4), ("finger_b", "link8", 2e-3)):
                expected = compose(grip, rig.link(f"{side}_gripper", [q[offset + 6]], finger))
                measured = sim.native("link_pose", side, link)
                np.testing.assert_allclose(expected[0], measured[0], atol=atol)
                np.testing.assert_allclose(expected[1], measured[1], atol=atol)
            camera = compose(flange, rig.mount(f"{side}_camera"))
            measured = rt(*sim.camera_pose(f"{side}_camera"))
            np.testing.assert_allclose(camera[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(camera[1], measured[1], atol=1e-4)
        head = rig.mount("head_camera"), rt(*sim.camera_pose("head_camera"))
        np.testing.assert_allclose(head[0][0], head[1][0], atol=1e-4)
        np.testing.assert_allclose(head[0][1], head[1][1], atol=1e-4)
