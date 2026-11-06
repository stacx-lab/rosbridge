"""PSI additions on a real kitchen: scene choice, evidence logging, no reset, offline re-rendering, time limit."""
import json

import numpy as np
import pytest

from rosbridge_psi_kitchen import render
from rosbridge_psi_kitchen.backend import PsiKitchenSim

CAMERA = 'robot0_agentview_left'


@pytest.fixture
def make_sim():
    """PsiKitchenSim(**kwargs), each closed after the test."""
    sims = []
    yield lambda **kwargs: sims.append(PsiKitchenSim(**kwargs)) or sims[-1]
    for sim in sims:
        sim.close()


def test_scene_evidence_no_reset_and_rerender(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('PSI_SCENE', '3')
    monkeypatch.delenv('PSI_TIME_LIMIT_S', raising=False)
    assert PsiKitchenSim.resettable is False
    sim = PsiKitchenSim(cameras=(CAMERA,))
    try:
        sim.reset('OpenDrawer', 0)
        assert sim.max_steps == 750
        for _ in range(5):
            sim.step(sim.state()[0])
        with pytest.raises(RuntimeError, match='does not reset'):
            sim.reset('OpenDrawer', 0)
        assert sim.steps == 5
        live = sim.render(CAMERA)
    finally:
        sim.close()

    folder = tmp_path / 'episode-0'
    meta = json.loads((folder / 'meta.json').read_text())
    assert meta['scene'] == 3 and meta['layout_style'] == [3, 3] and meta['time_limit_s'] == pytest.approx(37.5)
    rows = np.fromfile(folder / 'steps.f64').reshape(-1, len(meta['row']))
    assert meta['instruction'] and len(rows) == 6 and rows[-1, 0] == pytest.approx(.25)
    frames = list(render.frames(folder, camera=CAMERA, width=256, height=256))
    assert len(frames) == 6
    assert np.abs(frames[-1].astype(int) - live.astype(int)).mean() < 10  # matches the live view
    assert render.render(folder).stat().st_size > 0


def test_time_limit_and_unknown_scene(monkeypatch, make_sim):
    monkeypatch.setenv('PSI_TIME_LIMIT_S', '0.15')
    monkeypatch.delenv('PSI_SCENE', raising=False)
    sim = make_sim(cameras=())
    sim.reset('OpenDrawer', 0)
    for _ in range(3):
        sim.step(sim.state()[0])
    assert sim.max_steps == 3 and sim.done
    monkeypatch.setenv('PSI_SCENE', '11')
    with pytest.raises(ValueError, match='unknown scene'):
        make_sim(cameras=()).reset('OpenDrawer', 0)


def test_dataset_labels():
    from rosbridge_psi_kitchen import dataset
    positions = np.arange(10, dtype=float)[:, None] * np.ones(12)
    arm, torso = np.full((10, 7), 5.), np.full(10, .2)
    grip, base = np.array([1., 1., 0., -1., 0, 0, 0, 0, 0, 0]), np.arange(30.).reshape(10, 3)
    action = dataset.labels(positions, arm, torso, grip, base)
    assert (action[:, :7] == 5).all() and (action[:, 8] == .2).all()
    assert list(action[:4, 7]) == [1, 1, 2, -1]  # recorded direction as a target; 0 holds the command
    assert (action[:, 9:] == base).all()
    assert list(dataset.grip_commands(-1., [1, 1, 1, -1, 0], .4)) == pytest.approx([-1, -.6, -.2, .2, -.2])


def test_dataset_aims_at_the_commanded_hand_pose(monkeypatch, make_sim):
    """A recorded Cartesian command becomes the joints for the hand pose it aimed at (5 cm along the arm base's x)."""
    import mujoco as mj
    from rosbridge_psi_kitchen import dataset
    monkeypatch.setenv('PSI_SCENE', '3')
    sim = make_sim(cameras=())
    sim.reset('OpenDrawer', 0)
    arm, model = sim.robot.part_controllers['right'], sim.model
    hand, base = model.site(arm.ref_name).id, model.site(f'{arm.naming_prefix}{arm.part_name}_center').id
    start, frame = sim.data.site_xpos[hand].copy(), sim.data.site_xmat[base].reshape(3, 3).copy()
    scratch = mj.MjData(model)
    target, _ = dataset.aim(sim, scratch, np.array([1., 0, 0, 0, 0, 0]), -1., None, dataset.arm_scale())
    scratch.qpos[:] = sim.data.qpos
    scratch.qpos[model.jnt_qposadr[arm.joint_index]] = target
    mj.mj_kinematics(model, scratch)
    assert scratch.site_xpos[hand] - start == pytest.approx(.05 * frame[:, 0], abs=1e-4)


def test_dataset_conversion_matches_the_live_robot(monkeypatch, make_sim):
    """A recorded episode restored into another robot reads and looks as the live one did."""
    from rosbridge_psi_kitchen import dataset
    monkeypatch.setenv('PSI_SCENE', '3')
    monkeypatch.delenv('ROSBRIDGE_RUN_DIR', raising=False)
    live = make_sim()
    live.reset('OpenDrawer', 0)
    states, positions = [live.env.sim.get_state().flatten()], [live.state()[0]]
    command = live.state()[0]
    for _ in range(8):
        command[0] += .02
        command[7] = 1.
        live.step(command)
        states.append(live.env.sim.get_state().flatten())
        positions.append(live.state()[0])
    frame = live.render(CAMERA)
    xml, ep_meta = live.env.sim.model.get_xml(), live.env.get_ep_meta()
    robot = make_sim()
    robot.reset('OpenDrawer', 1)  # another kitchen state, replaced by the recording
    commands = {'hand': np.zeros((9, 6)), 'mode': -np.ones(9), 'grip': np.r_[np.ones(8), 0.],
                'base': np.zeros((9, 3)), 'torso': np.zeros(9)}
    state, images, action, _ = dataset.convert(robot, ep_meta, xml, np.array(states), commands, 256)
    assert state == pytest.approx(np.array(positions), abs=1e-6)
    assert np.abs(images['camera_left'][-1].astype(int) - frame.astype(int)).mean() < 1
    assert action.shape == (9, 12)
    assert action[:, :7] == pytest.approx(state[:, :7], abs=1e-6)  # no command: hold the arm where it is
    assert action[:, 8] == pytest.approx(state[:, 8])
