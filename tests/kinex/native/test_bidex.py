"""The exported rigs' arms, hands and cameras agree with MuJoCo through Kinex's URDF kinematics."""
import numpy as np
import pytest

from rosbridge_bidex.backend import BiDexSim
from tests.kinex.geometry import Rig, compose, requires_kinex, rt

pytestmark = requires_kinex


# Out-of-domain seeds move the head camera, the wrist cameras or both, as upstream's evaluator draws them.
@pytest.mark.parametrize("task,seed,flange,tolerance", [
    ("put-apple-in-white-bowl", 0, "{}_fr3v2_link8", 1e-6),
    ("put-apple-in-white-bowl-ood", 0, "{}_fr3v2_link8", 1e-6),  # head only
    ("put-apple-in-white-bowl-ood", 4, "{}_fr3v2_link8", 1e-6),  # wrists only
    ("take-milk-and-lemon-from-shelf-into-tray-ood", 3, "{}_fr3v2_link8", 1e-6),  # head and wrists
    # The Astribot torso and head are held by servos, not welded; they settle within a millimetre.
    ("pick-up-peach", 0, "astribot_arm_{}_tool_link", 1e-3),
    ("pick-up-peach-ood", 1, "astribot_arm_{}_tool_link", 1e-3),  # head and wrists
])
def test_the_workspace_of_a_task_and_seed_matches_its_episode(tmp_path, task, seed, flange, tolerance):
    rig = Rig(f"bidex/{task}", tmp_path, seed=seed)  # the workspace rosbridge setup exports for it
    sim = BiDexSim()
    try:
        sim.reset(task, seed)
        model, data = sim.env.model, sim.env.data
        groups = {group.name: list(group.indices) for group in sim.layout.groups}
        head = rig.mount("right_arm") if sim.env.robot_type == "dual_fr3v2_sharpa" else None

        def body(name):
            return rt(data.xpos[model.body(name).id], data.xquat[model.body(name).id])

        def same(expected, measured):
            for a, b in zip(expected, measured):
                np.testing.assert_allclose(a, b, atol=tolerance)

        rng, target = np.random.default_rng(0), sim.state()[0]
        for _ in range(3):
            q = sim.state()[0]
            for side in ("right", "left"):
                tip = rig.flange(f"{side}_arm", q[groups[f"{side}_arm"]])
                same(tip, body(flange.format(side)))
                hand = compose(tip, rig.mount(f"{side}_hand"))
                same(hand, body(f"{side}_hand_C_MC"))
                for link in (f"{side}_thumb_DP", f"{side}_index_DP", f"{side}_pinky_DP"):
                    same(compose(hand, rig.link(f"{side}_hand", q[groups[f"{side}_hand"]], link)), body(link))
                same(compose(tip, rig.mount(f"{side}_camera")), rt(*sim.camera_pose(f"{side}_wrist")))
            camera = rig.mount("head_camera")
            same(camera if head is None else compose(head, camera), rt(*sim.camera_pose("head")))
            target[:14] = q[:14] + rng.uniform(-.2, .2, 14)
            target[14:] = rng.uniform(0, .5, 44)
            for _ in range(5):
                sim.step(target)
    finally:
        sim.close()
