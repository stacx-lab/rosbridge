"""Catalog kinematics and calibration agree with independent native measurements."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from rosbridge.catalog.robots import setup
from rosbridge_libero.backend import LiberoSim
from tests.kinex_geometry import requires_kinex

pytestmark = requires_kinex


@pytest.mark.parametrize("variant,task", [
    ("tabletop", "libero-spatial-01"), ("floor", "libero-object-01"),
    ("kitchen", "libero-10-03"), ("living-room", "libero-10-01"),
    ("study", "libero-10-06"),
])
def test_catalog_arm_and_camera_geometry_matches_measurements(tmp_path, variant, task):
    from kinex.lib.geometry import compose_rt, xyzq_to_rt
    from kinex.lib.joints import JointModel
    project = Path(__file__).resolve().parents[4]
    setup(f"libero/libero-panda-{variant}", tmp_path, project=project)
    directory = tmp_path / ".kinex/devices"
    profiles = {name: yaml.safe_load((directory / name / "profile.yaml").read_text())
                for name in ("arm", "camera_agentview", "camera_wrist")}
    model = JointModel(profiles["arm"], directory / "arm")

    def mount(name):
        value = profiles[name]["frames"]["base"]
        return xyzq_to_rt(value["translation_m"] + value["rotation_xyzw"])

    sim = LiberoSim(cameras=())
    try:
        sim.reset(task, 0, False)
        target = sim.state()[0]
        target[0] += .1
        for _ in range(4):
            joints = sim.state()[0][:7].tolist()
            flange = compose_rt(mount("arm"), xyzq_to_rt(model.fk(joints)))
            body = sim.model.body("robot0_right_hand").id
            np.testing.assert_allclose(flange[0], sim.data.xmat[body].reshape(3, 3), atol=1e-6)
            np.testing.assert_allclose(flange[1], sim.data.xpos[body], atol=1e-6)
            for name, source in (("camera_agentview", "agentview"), ("camera_wrist", "robot0_eye_in_hand")):
                expected = mount(name) if name == "camera_agentview" else compose_rt(flange, mount(name))
                position, quaternion = sim.camera_pose(source)
                measured = xyzq_to_rt([*position, *quaternion[1:], quaternion[0]])
                np.testing.assert_allclose(expected[0], measured[0], atol=1e-6)
                np.testing.assert_allclose(expected[1], measured[1], atol=1e-6)
            sim.step(target)
    finally:
        sim.close()
