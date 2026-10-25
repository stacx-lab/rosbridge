"""The abc/abc-yam rig agrees with MuJoCo across patched and unpatched scenes."""
import numpy as np
import pytest

from rosbridge_abc.backend import AbcSim
from tests.kinex_geometry import Rig, compose, requires_kinex, rt

pytestmark = requires_kinex


def body(sim, name):
    data, index = sim.env.data, sim.env.model.body(name).id
    return data.xmat[index].reshape(3, 3), data.xpos[index]


def assert_pose(expected, measured, atol=1e-6):
    np.testing.assert_allclose(expected[0], measured[0], atol=atol)
    np.testing.assert_allclose(expected[1], measured[1], atol=atol)


@pytest.mark.parametrize("task", ["put_plastic_bottles_in_bin", "count_into_opaque_box",
                                  "pouring", "put_markers_in_top_drawer"])
def test_rig_matches_native_arms_grippers_and_cameras(tmp_path, task):
    rig = Rig("abc/abc-yam", tmp_path)
    sim = AbcSim()
    try:
        sim.reset(task, 0, False)
        target = sim.state()[0]
        target[[0, 7]] += .1
        target[[6, 13]] = .5
        # Reset places only the actuated finger; the soft finger equality settles in a few steps.
        for _ in range(10):
            sim.step(target)
        for _ in range(4):
            q = sim.state()[0]
            for side, offset in (("left", 0), ("right", 7)):
                flange = rig.flange(f"{side}_arm", q[offset:offset + 6])
                assert_pose(flange, body(sim, f"{side}_link_6"))
                grip = compose(flange, rig.mount(f"{side}_gripper"))
                opening = q[offset + 6]
                assert 0 < opening < 1
                for finger, atol in (("left", 1e-6), ("right", 1e-4)):
                    link = rig.link(f"{side}_gripper", [opening], f"{finger}_finger")
                    assert_pose(compose(grip, link), body(sim, f"{side}_link_{finger}_finger"), atol)
                site = sim.env.model.site(f"{side}_grasp_site").id
                assert_pose(compose(grip, rig.mount(f"{side}_gripper", "tcp")),
                            (sim.env.data.site_xmat[site].reshape(3, 3), sim.env.data.site_xpos[site]))
                assert_pose(compose(flange, rig.mount(f"camera_{side}")), rt(*sim.camera_pose(side)))
            assert_pose(rig.mount("camera_top"), rt(*sim.camera_pose("top")))
            sim.step(target)
    finally:
        sim.close()


def test_rig_intrinsics_and_depth_range_match_rendering(tmp_path):
    rig = Rig("abc/abc-yam", tmp_path)
    sim = AbcSim()
    try:
        sim.reset(sim.default_task, 0, False)
        for camera in sim.cameras:
            profile = rig.profiles[f"camera_{camera}"]
            projection = profile["color"]["projection"]
            assert (profile["color"]["width"], profile["color"]["height"]) == (sim.width, sim.height)
            assert [projection[k] for k in ("fx", "fy", "cx", "cy")] == pytest.approx(sim.intrinsics(camera))
            depth = sim.render_depth(camera)
            valid = depth[depth > 0]
            span = profile["depth"]["valid_range_m"]
            assert valid.size and span["min"] <= valid.min() and valid.max() <= span["max"]
    finally:
        sim.close()
