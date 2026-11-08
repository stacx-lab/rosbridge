"""OmniGibson's macro particle systems as patches/behavior/0004 makes them: a call that adds or removes physical
particles updates the physics handles (og.sim.update_handles, which rebuilds every physics view) once, and removing
visual particles, which are in no physics view, updates none. Isaac hosts one app per process and OmniGibson's
shutdown ends it, so particle_batches.py runs the calls in a process of its own."""
import json
import subprocess
import sys
from pathlib import Path


def test_a_particle_batch_updates_the_physics_handles_once_and_a_visual_removal_never(tmp_path):
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
    visual = {label: report[label] for label in ("visual generate", "visual remove", "visual remove group")}
    assert {label: call["updates"] for label, call in visual.items()} == dict.fromkeys(visual, 0)
    assert [visual[label]["left"] for label in ("visual remove", "visual remove group")] == [10, 0]
    # The physics views stayed valid through the visual removals, and the box still rests on the ground
    assert report["view valid"] and abs(report["box z"] - .2) < 1e-2
