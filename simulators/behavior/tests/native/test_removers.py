"""A particle remover follows its object on steps that do not render (patches/behavior/0005). A remover reads its
volume from Fabric, which Kit updates from PhysX when it renders, and rosbridge steps an episode without rendering:
unpatched, a broom moved by physics removes nothing until a camera is read. Isaac hosts one app per process and
OmniGibson's shutdown ends it, so remover_pass.py runs the pass in a process of its own."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_a_remover_moved_by_physics_removes_the_particles_it_passes_without_a_render(tmp_path):
    path = tmp_path / "report.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("remover_pass.py")), str(path)],
                         capture_output=True, text=True, timeout=1200)
    errors = [line for line in (run.stdout + run.stderr).splitlines() if "[py stderr]" in line]  # as Kit logs them
    assert path.is_file(), "\n".join(errors[-40:]) or run.stderr[-4000:]
    report = json.loads(path.read_text())
    # The broom crossed the dust in a straight line at the speed it was given: its box started short of the dust
    assert report["particles"] == 20 and report["drift"] < 1e-3
    assert report["travel"] == pytest.approx(.3 / 30 * report["steps"], rel=.01) and report["travel"] > .47
    # Every particle went at the step that brought it inside the box, which the remover read where the broom was
    assert (report["left"], report["late"]) == (0, 0) and report["lag"] < 1e-3
