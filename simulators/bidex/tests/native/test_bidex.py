"""Pinned BiDex-100 acceptance on representative tasks: catalog rigs, evaluator episodes, scoring, rendering."""
import numpy as np
import pytest

from rosbridge.catalog import tasks
from rosbridge_bidex.backend import BiDexSim, episode

RIGS = {("dual_fr3v2_sharpa", (0.5, -0.45, 0.6)): "bidex/bidex-fr3v2-sharpa-tabletop",
        ("dual_fr3v2_sharpa", (0.8, -0.45, 0.7)): "bidex/bidex-fr3v2-sharpa-dexjoco",
        ("astribot_s1_fixed_sharpa", (0.8, 0.0, 0.0)): "bidex/bidex-astribot-s1-sharpa"}


def test_every_task_runs_on_the_rig_its_yaml_names():
    from bidex import load_task
    for entry in tasks(name="bidex").values():
        task = load_task(*entry["native_task"].split("/"))
        assert RIGS[task.robot_type, tuple(task.robot_pose["pos"])] == entry["robot"], entry["id"]
    # Seeds 0-49 are the evaluator's ID eval seeds 0 and 1, and OOD eval seeds 2 and 3.
    assert [episode(s, ood) for ood in (False, True) for s in (0, 24, 25, 49)] == [
        (0, 0), (0, 24), (1, 0), (1, 24), (2, 0), (2, 24), (3, 0), (3, 24)]


@pytest.mark.parametrize("task,seed,joint", [
    ("put-apple-in-white-bowl", 0, "right_fr3v2_joint1"),
    ("put-apple-in-white-bowl-ood", 7, "right_fr3v2_joint1"),
    ("take-milk-and-lemon-from-shelf-into-tray", 3, "right_fr3v2_joint1"),
    ("pick-up-peach-ood", 30, "astribot_arm_right_joint_1"),
])
def test_representative_tasks_reset_hold_step_and_render(task, seed, joint):
    from bidex.eval.run_leaderboard_eval import max_steps_for_task
    sim = BiDexSim()
    try:
        sim.reset(task, seed)
        suite, name = sim.task.removesuffix("__ood").split("/")
        assert (sim.steps, sim.time_s, sim.period, sim.max_steps) == (0, 0, .04, max_steps_for_task(suite, name))
        assert sim.layout.joint_names[0] == joint and len(sim.layout.joint_names) == 58
        assert [g.name for g in sim.layout.groups] == ["right_arm", "left_arm", "right_hand", "left_hand"]
        hold = sim.state()[0]
        for _ in range(25):
            sim.step(hold)
        # Native position servos hold the pose; the Astribot's soft arms sag ~0.12 rad under gravity.
        assert sim.time_s == pytest.approx(1.) and np.abs(sim.state()[0] - hold).max() < .15
        qpos = sim.env.data.qpos.copy()
        for camera in sim.cameras:
            rgb, depth = sim.render(camera), sim.render_depth(camera)
            assert rgb.shape == (360, 480, 3) and rgb.dtype == np.uint8 and rgb.std() > 5
            assert depth.shape == (360, 480) and depth.dtype == np.float32 and (depth > 0).mean() > .9
        np.testing.assert_array_equal(qpos, sim.env.data.qpos)  # rendering never advances the world
        assert sim.measures() == {"score": 0} and not sim.success
    finally:
        sim.close()


def test_seed_replays_one_episode_and_ood_cameras_stay_nominal():
    from bidex.eval.ood import camera_occurrence_mode
    # OOD seed 0 is upstream's eval seed 2, episode 0, whose plan would move the head camera.
    assert camera_occurrence_mode("Generalization", "put_apple_in_white_bowl", 2, 0) == "head_only"
    sims = [BiDexSim() for _ in range(3)]
    try:
        for sim, (task, seed) in zip(sims, (("put-apple-in-white-bowl-ood", 0),) * 2 + (
                ("put-apple-in-white-bowl", 0),)):
            sim.reset(task, seed)
        ood, again, nominal = sims
        np.testing.assert_array_equal(ood.env.data.qpos, again.env.data.qpos)
        # The same scene; separate GL contexts may round a multisampled pixel differently.
        np.testing.assert_allclose(ood.render("head"), again.render("head"), atol=1)
        np.testing.assert_array_equal(ood.env.model.cam_pos, nominal.env.model.cam_pos)
        np.testing.assert_array_equal(ood.env.model.cam_quat, nominal.env.model.cam_quat)
        # A step is the evaluator's env.step, whose rendering never changes the physics.
        action = ood.state()[0] + np.r_[np.full(14, .05), np.full(44, .2)]
        for _ in range(10):
            ood.step(action)
            again.env.step({"arm_joint_position": action[:14], "dex_joint_position": action[14:]})
        np.testing.assert_array_equal(ood.env.data.qpos, again.env.data.qpos)
    finally:
        for sim in sims:
            sim.close()


def test_success_and_score_follow_the_latched_subtasks():
    sim = BiDexSim(terminate_on_success=True)
    try:
        sim.reset(sim.default_task, 0)
        env, hold = sim.env, sim.state()[0]
        role = {o.name: o.instance_id for o in env.task.objects}
        apple = env.model.joint(f"obj{next(i for n, i in role.items() if 'apple' in n)}_freejoint")
        bowl = env.data.body(f"obj{next(i for n, i in role.items() if 'bowl' in n)}_root").xpos
        address = apple.qposadr[0]
        env.data.qpos[address + 2] += .2  # lifted: the first subtask latches, worth 50
        sim.step(hold)
        assert sim.measures() == {"score": 50} and not sim.done
        env.data.qpos[address:address + 3] = bowl + (0, 0, .05)  # dropped into the bowl
        env.data.qvel[apple.dofadr[0]:apple.dofadr[0] + 6] = 0
        while not sim.done:
            sim.step(hold)
        assert sim.success and sim.reward == 1. and sim.measures() == {"score": 100}
        assert sim.steps < 50
    finally:
        sim.close()


def test_failed_reset_keeps_the_episode(monkeypatch):
    sim = BiDexSim(max_steps=2)
    try:
        sim.reset(sim.default_task, 1)
        previous, qpos = sim.env, sim.env.data.qpos.copy()
        with monkeypatch.context() as patch:
            patch.setattr(BiDexSim, "_index", lambda _: (_ for _ in ()).throw(ValueError("failure")))
            with pytest.raises(ValueError, match="failure"):
                sim.reset(sim.default_task, 2)
        assert sim.env is previous
        np.testing.assert_array_equal(qpos, sim.env.data.qpos)
        for _ in range(2):
            sim.step(sim.state()[0])
        assert sim.done
        with pytest.raises(RuntimeError, match="done"):
            sim.step(sim.state()[0])
    finally:
        sim.close()
