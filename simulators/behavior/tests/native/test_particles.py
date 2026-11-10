"""OmniGibson's macro particle systems as patches/behavior/0004 makes them: a call that adds or removes physical
particles, or removes visual ones, updates the physics handles (og.sim.update_handles, which rebuilds every physics
view) once, and afterwards a jointed object's joints can be set and read. Removing any prim, a visual particle's too,
drops the handle of every articulation view until the handles are updated. Isaac hosts one app per process and
OmniGibson's shutdown ends it, so particle_batches.py runs the calls in a process of its own."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_a_particle_batch_updates_the_physics_handles_once_and_leaves_joints_readable(tmp_path):
    path = tmp_path / "report.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("particle_batches.py")), str(path)],
                         capture_output=True, text=True, timeout=1200)
    errors = [line for line in (run.stdout + run.stderr).splitlines() if "[py stderr]" in line]  # as Kit logs them
    assert path.is_file(), "\n".join(errors[-40:]) or run.stderr[-4000:]
    report = json.loads(path.read_text())
    physical = {label: report[label] for label in ("generate", "remove", "sync up", "sync down", "remove all")}
    assert {label: call["updates"] for label, call in physical.items()} == dict.fromkeys(physical, 1)
    assert report["generate"]["left"] == 20 and report["generate"]["error"] <= report["generate"]["offset"] + 1e-4
    assert [physical[label]["left"] for label in ("remove", "sync up", "sync down", "remove all")] == [10, 15, 5, 0]
    # Adding visual particles deletes no prim; each call that removes some updates the handles once, a restore too
    visual = {label: report[label] for label in ("visual generate", "visual remove", "visual restore",
                                                 "visual remove group")}
    assert {label: call["updates"] for label, call in visual.items()} == {
        "visual generate": 0, "visual remove": 1, "visual restore": 1, "visual remove group": 1}
    assert [call["left"] for call in visual.values()] == [20, 10, 20, 0]
    # After every call the cabinet's joints take and return a new position, and the box still rests on the ground
    assert {label: call["joints"] for label, call in {**physical, **visual}.items()} == {
        label: pytest.approx(0., abs=1e-4) for label in (*physical, *visual)}
    assert abs(report["box z"] - .2) < 1e-2
