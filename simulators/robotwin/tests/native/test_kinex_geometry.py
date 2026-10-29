"""The robotwin/robotwin-aloha-agilex rig agrees with SAPIEN link, TCP and camera poses."""
import numpy as np
import pytest

from rosbridge_robotwin.backend import RoboTwinSim, pose_matrix
from tests.kinex_geometry import Rig, compose, requires_kinex, rt

pytestmark = requires_kinex
PREFIX = {"left": "fl", "right": "fr"}


def link(sim, side, name):
    matrix = pose_matrix(getattr(sim.env.robot, f"{side}_entity").find_link_by_name(name).get_pose())
    return matrix[:3, :3], matrix[:3, 3]


def assert_pose(expected, measured, atol=2e-5):
    # SAPIEN poses are float32.
    np.testing.assert_allclose(expected[0], measured[0], atol=atol)
    np.testing.assert_allclose(expected[1], measured[1], atol=atol)


@pytest.fixture(scope="module")
def sim():
    sim = RoboTwinSim()
    sim.reset("handover_block", 0)
    yield sim
    sim.close()


def test_rig_matches_native_arms_grippers_tcp_and_cameras(tmp_path, sim):
    rig = Rig("robotwin/robotwin-aloha-agilex", tmp_path)
    target = sim.state()[0]
    target[[0, 7]] += .2
    target[[1, 8]] += .3
    target[[6, 13]] = .5
    for _ in range(3):
        for _ in range(15):
            sim.step(target)
        q = sim.state()[0]
        for side, offset in (("left", 0), ("right", 7)):
            p = PREFIX[side]
            flange = rig.flange(f"{side}_arm", q[offset:offset + 6])
            assert_pose(flange, link(sim, side, f"{p}_link6"))
            grip = compose(flange, rig.mount(f"{side}_gripper"))
            opening = q[offset + 6]
            assert .2 < opening < 1
            # finger_a carries the measured joint; finger_b has its own drive.
            for finger, native, atol in (("finger_a", f"{p}_link7", 2e-5), ("finger_b", f"{p}_link8", 2e-3)):
                assert_pose(compose(grip, rig.link(f"{side}_gripper", [opening], finger)),
                            link(sim, side, native), atol)
            tcp = getattr(sim.env.robot, f"get_{side}_tcp_pose")()
            assert_pose(compose(grip, rig.mount(f"{side}_gripper", "tcp")), rt(tcp[:3], tcp[3:]))
            assert_pose(compose(flange, rig.mount(f"{side}_camera")), rt(*sim.camera_pose(f"{side}_camera")))
        assert_pose(rig.mount("head_camera"), rt(*sim.camera_pose("head_camera")))


def test_rig_intrinsics_match_rendering(tmp_path, sim):
    rig = Rig("robotwin/robotwin-aloha-agilex", tmp_path)
    for camera in sim.cameras:
        profile = rig.profiles[camera]
        color = profile["color"]
        pinhole = [color["projection"][k] for k in ("fx", "fy", "cx", "cy")]
        assert [color["width"], color["height"], *pinhole] == pytest.approx(sim.intrinsics(camera))
        depth = sim.render_depth(camera)
        span = profile["depth"]["valid_range_m"]
        assert (depth > 0).mean() > .9 and span["min"] <= depth[depth > 0].min()
