"""The exported mobile rig's chain base -> torso:top -> arm:flange agrees with MuJoCo."""
import numpy as np

from rosbridge_robocasa.backend import RoboCasaSim
from tests.kinex_geometry import Rig, compose, requires_kinex, rt

pytestmark = requires_kinex


def assert_pose(expected, measured):
    np.testing.assert_allclose(expected[0], measured[0], atol=1e-6)
    np.testing.assert_allclose(expected[1], measured[1], atol=1e-6)


def test_catalog_chain_cameras_and_fingers_match_measurements(tmp_path):
    rig = Rig("robocasa/robocasa-panda-omron", tmp_path)
    sim = RoboCasaSim()
    try:
        sim.reset(sim.default_task, 0)
        model, data = sim.model, sim.data

        def body(name):
            return rt(data.xpos[model.body(name).id], data.xquat[model.body(name).id])

        # The fingers follow the integrated command only at rest, so check them at reset.
        grip = [sim.state()[0][7]]
        for link, finger in (("leftfinger", "gripper0_right_leftfinger"),
                             ("rightfinger", "gripper0_right_rightfinger")):
            expected = compose(body("gripper0_right_right_gripper"), rig.link("gripper", grip, link))
            np.testing.assert_allclose(expected[1], body(finger)[1], atol=2e-4)
        target = sim.state()[0]
        rng = np.random.default_rng(0)
        for _ in range(4):
            q = sim.state()[0]
            top = compose(compose(body("mobilebase0_base"), rig.mount("torso")),
                          rig.link("torso", q[8:9], "top"))
            flange = compose(compose(top, rig.mount("arm")), rig.link("arm", q[:7]))
            assert_pose(flange, body("robot0_right_hand"))
            gripper = compose(flange, rig.mount("gripper"))
            assert_pose(gripper, body("gripper0_right_right_gripper"))
            assert_pose(compose(gripper, rig.mount("gripper", "tcp")), body("gripper0_right_eef"))
            for name, camera, parent in (("camera_left", "robot0_agentview_left", top),
                                         ("camera_right", "robot0_agentview_right", top),
                                         ("camera_wrist", "robot0_eye_in_hand", flange)):
                assert_pose(compose(parent, rig.mount(name)), rt(*sim.camera_pose(camera)))
            target[:7] = q[:7] + rng.uniform(-.3, .3, 7)
            target[8] = rng.uniform(.05, .3)
            for _ in range(5):
                sim.step(target)
    finally:
        sim.close()
