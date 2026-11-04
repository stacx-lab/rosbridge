"""PSI additions on a real kitchen: scene choice, evidence logging, no reset, offline re-rendering, time limit."""
import json

import numpy as np
import pytest

from rosbridge_psi_kitchen import render
from rosbridge_psi_kitchen.backend import PsiKitchenSim

CAMERA = 'robot0_agentview_left'


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


def test_time_limit_and_unknown_scene(monkeypatch):
    monkeypatch.setenv('PSI_TIME_LIMIT_S', '0.15')
    monkeypatch.delenv('PSI_SCENE', raising=False)
    sim = PsiKitchenSim(cameras=())
    try:
        sim.reset('OpenDrawer', 0)
        for _ in range(3):
            sim.step(sim.state()[0])
        assert sim.max_steps == 3 and sim.done
    finally:
        sim.close()
    monkeypatch.setenv('PSI_SCENE', '11')
    with pytest.raises(ValueError, match='unknown scene'):
        PsiKitchenSim(cameras=()).reset('OpenDrawer', 0)
