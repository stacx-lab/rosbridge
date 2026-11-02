"""PSI additions on a real kitchen: evidence logging, the reset guard and offline re-rendering."""
import json

import numpy as np
import pytest

from rosbridge_psi_kitchen import render
from rosbridge_psi_kitchen.backend import PsiKitchenSim

CAMERA = 'robot0_agentview_left'


def test_evidence_reset_guard_and_rerender(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSBRIDGE_RUN_DIR', str(tmp_path))
    monkeypatch.delenv('PSI_ALLOW_RESET', raising=False)
    sim = PsiKitchenSim(cameras=(CAMERA,))
    try:
        sim.reset('OpenDrawer', 0)
        assert sim.max_steps == 750
        for _ in range(5):
            sim.step(sim.state()[0])
        with pytest.raises(RuntimeError, match='disabled'):
            sim.reset('OpenDrawer', 0)
        assert sim.steps == 5
        live = sim.render(CAMERA)
    finally:
        sim.close()

    folder = tmp_path / 'episode-0'
    meta = json.loads((folder / 'meta.json').read_text())
    rows = np.fromfile(folder / 'steps.f64').reshape(-1, len(meta['row']))
    assert meta['instruction'] and len(rows) == 6 and rows[-1, 0] == pytest.approx(.25)
    frames = list(render.frames(folder, camera=CAMERA, width=256, height=256))
    assert len(frames) == 6
    assert np.abs(frames[-1].astype(int) - live.astype(int)).mean() < 10  # matches the live view
    assert render.render(folder).stat().st_size > 0


def test_step_limit_override(monkeypatch):
    monkeypatch.setenv('PSI_MAX_STEPS', '3')
    sim = PsiKitchenSim(cameras=())
    try:
        sim.reset('OpenDrawer', 0)
        for _ in range(3):
            sim.step(sim.state()[0])
        assert sim.max_steps == 3 and sim.done
    finally:
        sim.close()
