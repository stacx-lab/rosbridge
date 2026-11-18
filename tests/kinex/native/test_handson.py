"""The handson/handson-rokae-wuji rig agrees with Isaac Lab through Kinex's URDF kinematics, in handson's native environment.

The simulator runs in a worker subprocess, as in its own native suite.
"""
import os
import sys

import numpy as np
import pytest

from rosbridge.core.worker import WorkerBackend
from tests.kinex.geometry import Rig, compose, requires_kinex, rt, xyzq

pytestmark = requires_kinex


class Native(WorkerBackend):
    simulator = "handson"
    close_timeout_s = 120.


# The default task: both arms move freely.
TASK = "t24_place_ball_in_bowl"
SIDES = (("right", 0), ("left", 27))
CURL = [4, 6, 7, 8, 10, 11]  # index and middle finger flexion, in the hand's finger-major order


@pytest.fixture(scope="module")
def sim():
    sim = Native(python=sys.executable, env=dict(os.environ), max_steps=150)
    sim.configure_spectators()
    sim.reset(TASK, 0)
    yield sim
    sim.close()


def lifted(sim, side, height):
    """The measured palm pose in its arm base, height metres higher in the world."""
    palm = np.asarray(sim.native("tip_pose", f"{side}_arm"), dtype=float)
    rotation, _ = sim.native("link_pose", side, f"link0_{side}")
    palm[:3] += np.asarray(rotation).T @ [0., 0., height]
    return palm


def test_rig_matches_native_palms_fingers_and_cameras(tmp_path, sim):
    rig = Rig(TASK, tmp_path, seed=0)  # the workspace of the episode the fixture loaded
    action = sim.state()[0]
    for _, offset in SIDES:
        action[[offset + 7 + i for i in CURL]] += .3
    for _ in range(2):
        poses = {f"{side}_arm": lifted(sim, side, .03) for side, _ in SIDES}
        for _ in range(15):
            sim.step(action, poses)
        q = sim.state()[0]
        palms = {}
        for side, offset in SIDES:
            palms[side] = palm = rig.flange(f"{side}_arm", q[offset:offset + 7])
            measured = sim.native("link_pose", side, f"{side}_palm_link")
            np.testing.assert_allclose(palm[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(palm[1], measured[1], atol=1e-4)
            # A pose target is the palm in the arm base, which Isaac places at the arm's profile mount.
            in_world = compose(rig.mount(f"{side}_arm"), xyzq(list(sim.native("tip_pose", f"{side}_arm"))))
            np.testing.assert_allclose(in_world[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(in_world[1], measured[1], atol=1e-4)
            hand = compose(palm, rig.mount(f"{side}_hand"))
            for finger in range(1, 6):
                link = f"{side}_finger{finger}_link4"
                expected = compose(hand, rig.link(f"{side}_hand", q[offset + 7:offset + 27], link))
                measured = sim.native("link_pose", side, link)
                np.testing.assert_allclose(expected[0], measured[0], atol=1e-4)
                np.testing.assert_allclose(expected[1], measured[1], atol=1e-4)
        sim.render("head_camera")  # a capture places every camera at its drawn mount
        for camera, expected in (("head_camera", rig.mount("head_camera")),
                                 ("left_camera", compose(palms["left"], rig.mount("left_camera"))),
                                 ("right_camera", compose(palms["right"], rig.mount("right_camera")))):
            measured = rt(*sim.camera_pose(camera))
            np.testing.assert_allclose(expected[0], measured[0], atol=1e-4)
            np.testing.assert_allclose(expected[1], measured[1], atol=1e-4)
